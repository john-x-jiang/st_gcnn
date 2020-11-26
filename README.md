
## Learning Geometry-Dependent and Physics-Based Inverse Image Reconstruction
This repository provides the code and data described in the paper:

**[Learning Geometry-Dependent and Physics-Based Inverse Image Reconstruction](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_47)**
 Xiajun Jiang, Sandesh GhimireJwala DhamalaZhiyuan LiPrashnna Kumar Gyawali,  <a href="[https://pht180.rit.edu/cblwang/](https://pht180.rit.edu/cblwang/)" target="_blank">Linwei Wang</a>

 Published on [MICCAI 2020](https://www.miccai2020.org/en/).

Please cite the following if you use the data or the model in your work:
```
@inproceedings{jiang2020learning,
  title={Learning Geometry-Dependent and Physics-Based Inverse Image Reconstruction},
  author={Jiang, Xiajun and Ghimire, Sandesh and Dhamala, Jwala and Li, Zhiyuan and Gyawali, Prashnna Kumar and Wang, Linwei},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={487--496},
  year={2020},
  organization={Springer}
}
```

# Code

### Requirements

The key requirements are listed under `requirements.txt`
In addition, the code  uses  modified versions of and [torch_geometric](https://github.com/rusty1s/pytorch_geometric). The modified versions of bayesopt and torch_geometric are included with this repo. Go to each folder and install them in the `develop` mode.  

    python setup.py develop

### Running

ST_GCNN is composed of two stages:

- Stage 1: Training a STGCNN model
- Stage 2: Evaluating a STGCNN model on simulation datasets.
- Stage 3: Evaluating a STGCNN model on clinical datasets.

`main.py` provides functionality to either of these stages in isolation or in succession. 

**Configurations:** The configurations are provided in the form of a `.json` file. Two examples are included in the folder `config`. 
**Results**: For each run, a folder named `model_name (from .json file)` is created in the experiment directory inside which a copy of the `.json` file, trained model, training logs, diagrams and other results are saved.

#### Example scripts:
To run stage 1 with settings listed in `params_gvae.json`:

    python main.py --config params_gvae --stage 1

To run stage 2 (assuming stage 1 is complete) with settings listed in `params_gvae.json`:

    python main.py --config params_gvae --stage 2


# Data
The signal and geometry of both the heart and torso should be provided. Please contact us for more details.

# Contact
Please don't hesitate to contact me for any questions or comments. My email: [xj7056@rit.edu](xj7056@rit.edu). 
