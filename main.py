import argparse
import os
import os.path as osp
from shutil import copy2

import torch
from torch import optim
import numpy as np

import scipy.io
import mesh2
import net
import train_heart_torso
import utils
from torch_geometric.data import DataLoader
from torch_geometric.data import HeartGraphDataset
from torch_geometric.data import MySimpleHeartDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    """
    Args:
        config: json file with hyperparams and exp settings
        seed: random seed value
        stage: 1 for traing STGCNN, 2 for evaluation,  and 3 for clinical dataset
        logging: 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='params', help='config filename')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--logging', type=bool, default=True, help='logging')
    parser.add_argument('--stage', type=int, default=1, help='1.Train, 2.Eval Simulation, 3.Eval Clincal')

    args = parser.parse_args()
    return args


def learn_vae_heart_torso(hparams, training=True, fine_tune=False):
    """Generative modeling of the HD tissue properties
    """
    model_type = hparams.model_type
    batch_size = hparams.batch_size
    num_epochs = hparams.num_epochs
    seq_len = hparams.seq_len
    heart_torso = hparams.heart_torso
    anneal = hparams.anneal

    # directory path for training and testing datasets
    data_dir = osp.join(osp.dirname(osp.realpath('__file__')),
                        'data', 'training')

    # directory path to save the model/results
    model_dir = osp.join(osp.dirname(osp.realpath('__file__')),
                         'experiments', model_type, hparams.model_name)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    
    if training:
        copy2(net_path, model_dir)
    copy2(json_path, model_dir)

    # logging the training procedure
    # if args.logging:
    #    sys.stdout = open(model_dir+'/log.txt','wt')

    corMfrees = dict()
    train_loaders = dict()
    test_loaders = dict()
    heart_names = hparams.heart_name
    graph_names = hparams.graph_name
    num_meshfrees = hparams.num_meshfree
    structures = hparams.structures
    sample = hparams.sample if training else 1
    subset = hparams.subset if training else 1

    # initialize the model
    model = net.GraphTorsoHeart(hparams)
    
    for graph_name, heart_name, num_meshfree, structure in zip(graph_names, heart_names, num_meshfrees, structures):
        root_dir = osp.join(data_dir, heart_name)
        graph_dir = osp.join(root_dir, 'raw', graph_name)
        # Create graph and load graph information
        if training and hparams.makegraph:
            g = mesh2.GraphPyramid(heart_name, structure, num_meshfree, seq_len)
            g.make_graph()
        graphparams = net.get_graphparams(graph_dir, device, batch_size, heart_torso)

        # initialize datasets and dataloader
        train_dataset = HeartGraphDataset(root=root_dir, num_meshfree=num_meshfree, seq_len=seq_len,
                                        mesh_graph=graphparams["g"], mesh_graph_torso=graphparams["t_g"],
                                        heart_torso=heart_torso, train=True, subset=subset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=training, drop_last=True)
        
        corMfrees[heart_name] = train_dataset.getCorMfree()
        train_loaders[heart_name] = train_loader

        model.set_graphs(graphparams, heart_name)

    if training:
        val_heart = hparams.val_heart
        val_graph = hparams.val_graph
        val_meshfree = hparams.val_meshfree
        val_structures = hparams.val_structures
        root_dir = osp.join(data_dir, val_heart[0])

        graph_dir = osp.join(root_dir, 'raw', val_graph[0])
        if hparams.makegraph:
            vg = mesh2.GraphPyramid(val_heart[0], val_structures[0], val_meshfree[0], seq_len)
            vg.make_graph()
        graphparams = net.get_graphparams(graph_dir, device, batch_size, heart_torso)
        # state = not fine_tune
        state = False
        test_dataset = HeartGraphDataset(root=root_dir, num_meshfree=val_meshfree[0], seq_len=seq_len,
                                        mesh_graph=graphparams["g"], mesh_graph_torso=graphparams["t_g"],
                                        heart_torso=heart_torso, train=state)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loaders[val_heart[0]] = test_loader
        model.set_graphs(graphparams, val_heart[0])
        if fine_tune:
            pre_model_dir = osp.join(osp.dirname(osp.realpath('__file__')), 'experiments', model_type, hparams.pre_model_name)
            model.load_state_dict(torch.load(pre_model_dir + '/' + hparams.vae_latest, map_location='cuda:0'))
        
        model.to(device)
        loss_function = net.loss_function
        
        optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
        train_heart_torso.train_vae(model, optimizer, train_loaders, test_loaders, loss_function,
                        model_dir, num_epochs, batch_size, seq_len, corMfrees, anneal, sample)
    else:
        pre_model_dir = osp.join(osp.dirname(osp.realpath('__file__')), 'experiments', model_type, hparams.pre_model_name)
        model.load_state_dict(torch.load(pre_model_dir + '/' + hparams.vae_latest, map_location='cuda:0'))
        model = model.eval().to(device)
        train_heart_torso.eval_vae(model, train_loaders, model_dir, batch_size, seq_len, corMfrees)
        # train_heart_torso.eval_real_new(model, train_loaders, exp_dir, corMfrees)


def real_data(hparams):
    # bsp = scipy.io.loadmat('/home/xj7056/projects/Improving-Generalization/data/BSP7620a566.mat')
    bsp = scipy.io.loadmat('/home/xj7056/projects/Improving-Generalization/data/BSP5b143.mat')
    
    outY = (bsp['PHIB120_AVG'])['PHI_avg'][0][0]
    # outY = (1 / (1500)) * outY[0:1300, :]
    outY = (2 / (3 * 1e3)) * outY[0:950, :]

    model_type = hparams.model_type
    batch_size = hparams.batch_size
    num_epochs = hparams.num_epochs
    seq_len = hparams.seq_len
    heart_torso = hparams.heart_torso
    anneal = hparams.anneal

    a, b = outY.shape
    outY = outY[(np.linspace(0, a - 1, num=seq_len)).astype(int), :]

    model = net.GraphTorsoHeart(hparams)

    data_dir = osp.join(osp.dirname(osp.realpath('__file__')), 'data', 'training')
    model_dir = osp.join(osp.dirname(osp.realpath('__file__')), 'experiments', model_type, hparams.model_name)

    heart_names = hparams.heart_name
    graph_names = hparams.graph_name
    heart_name = heart_names[0]
    root_dir = osp.join(data_dir, heart_name)
    graph_dir = osp.join(root_dir, 'raw', graph_names[0])
    graphparams = net.get_graphparams(graph_dir, device, batch_size, heart_torso)
    model.set_graphs(graphparams, heart_name)
    corMfree = graphparams['g'].pos

    model.load_state_dict(torch.load(model_dir + '/' + hparams.vae_latest, map_location='cuda:0'))
    model = model.eval().to(device)
    train_heart_torso.eval_real(model, outY, heart_name, model_dir, corMfree)


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # filename of the params
    fname_config = args.config + '.json'
    # read the params file
    json_path = osp.join(osp.dirname(osp.realpath('__file__')), "config", fname_config)
    net_path = osp.join(osp.dirname(osp.realpath('__file__')), 'net.py')
    hparams = utils.Params(json_path)

    if args.stage == 1:  # generative modeling
        print('Stage 1: begin training STGCNN for heart & torso ...')
        learn_vae_heart_torso(hparams)
        print('Training STGCNN completed!')
        print('--------------------------------------')
    elif args.stage == 2:
        print('Stage 2: begin evaluating STGCNN for heart & torso ...')
        learn_vae_heart_torso(hparams, training=False)
        print('Evaluating STGCNN completed!')
        print('--------------------------------------')
    elif args.stage == 3:
        print('Stage 3: begin evaluating STGCNN for real data ...')
        real_data(hparams)
        print('Evaluating STGCNN completed!')
        print('--------------------------------------')
    else:
        print('invalid stage option; valid 1, 2, 3')
