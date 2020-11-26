import os.path as osp
import numpy as np

import scipy.io
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data


class HeartGraphDataset(Dataset):
    """
    A dataset of Data objects (in pytorch geometric) with graph attributes
    from a pre-defined graph hierarchy. 
    """

    def __init__(self,
                 root,
                 num_meshfree=None,
                 seq_len=None,
                 mesh_graph=None,
                 mesh_graph_torso=None,
                 heart_torso=None,
                 train=True,
                 subset=1,
                 label_type='size'):
        self.root = osp.expanduser(osp.normpath(root))
        self.raw_dir = osp.join(self.root, 'raw')
        self.heart_torso = heart_torso
        filename = '_ir_ic_' + str(num_meshfree) + '.mat'
        # print(label_type)
        if train:
            filename = 'training' + filename
            self.data_path = osp.join(self.raw_dir, filename)
            matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
            corMfree = matFiles['cor']
            dataset = matFiles['params_t']
            label_aha = matFiles['label_t']
            
        else:
            filename = 'testing' + filename
            self.data_path = osp.join(self.raw_dir, filename)
            matFiles = scipy.io.loadmat(self.data_path, squeeze_me=True, struct_as_record=False)
            corMfree = matFiles['cor']
            dataset = matFiles['params_e']
            label_aha = matFiles['label_e']

        #N = int(N / 10)

        # if label_type == 'size':
        #     self.label = torch.from_numpy(label_size[0:N]).float()
        # else:
        #     self.label = torch.from_numpy(label_aha[0:N]).float()
        dataset = dataset.reshape(num_meshfree, -1, seq_len)

        N = dataset.shape[1]
        if subset == 1:
            index = np.arange(N)
        elif subset == 0:
            raise RuntimeError('No data')
        else:
            indices = list(range(N))
            np.random.shuffle(indices)
            split = int(np.floor(subset * N))
            sub_index = indices[:split]
            dataset = dataset[:, sub_index, :]
            index = np.arange(dataset.shape[1])
        
        label_aha = label_aha.astype(int)
        # self.label = torch.from_numpy(label_aha[0:N]).float()
        if self.heart_torso == 0:
            self.label = torch.from_numpy(label_aha[index])
            self.graph = mesh_graph
            self.datax = torch.from_numpy(dataset[:, index]).float()
            self.corMfree = corMfree
            print('final data size: {}'.format(self.datax.shape[1]))
        elif self.heart_torso == 1:
            self.label = torch.from_numpy(label_aha[index])
            self.heart = mesh_graph
            self.torso = mesh_graph_torso
            self.data_heart = torch.from_numpy(dataset[0:-120, index]).float()
            self.data_torso = torch.from_numpy(dataset[-120:, index]).float()
            self.heart_cor = corMfree[0:-120, 0:3]
            self.torso_cor = corMfree[-120:, 0:3]
            print('heart data size: {}'.format(self.data_heart.shape[1]))
            print('torso data size: {}'.format(self.data_torso.shape[1]))
        elif self.heart_torso == 2:
            self.label = torch.from_numpy(label_aha[index])
            self.heart = mesh_graph
            self.torso = mesh_graph_torso
            self.data_heart = torch.from_numpy(dataset[0:-771, index]).float()
            self.data_torso = torch.from_numpy(dataset[-771:, index]).float()
            self.heart_cor = corMfree[0:-771, 0:3]
            self.torso_cor = corMfree[-771:, 0:3]
            print('heart data size: {}'.format(self.data_heart.shape[1]))
            print('torso data size: {}'.format(self.data_torso.shape[1]))

    def getCorMfree(self):
        if self.heart_torso == 0:
            return self.corMfree
        else:
            return (self.heart_cor, self.torso_cor)

    def __len__(self):
        if self.heart_torso == 0:
            return (self.datax.shape[1])
        else:
            return (self.data_heart.shape[1])

    def __getitem__(self, idx):
        if self.heart_torso == 0:
            x = self.datax[:, [idx]]  # torch.tensor(dataset[:,[i]],dtype=torch.float)
            y = self.label[[idx]]  # torch.tensor(label_aha[[i]],dtype=torch.float)

            sample = Data(
                x=x,
                y=y,
                edge_index=self.graph.edge_index,
                edge_attr=self.graph.edge_attr,
                pos=self.graph.pos
            )
            return sample
        elif self.heart_torso == 1 or self.heart_torso == 2:
            sample = Data(
                x=self.data_torso[:, [idx]],
                y=self.data_heart[:, [idx]],
                pos=self.label[[idx]]
            )
            return sample
