#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a testing on any dataset
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import os
import numpy as np
import argparse

# Dataset
from datasets.H3D import *
from datasets.Paris import *
from datasets.Semantic3D import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test a model')
    # Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #'dataset_name/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model
    parser.add_argument('-s', '--snap', type=str, required=True, help='snapshot path')
    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    parser.add_argument('-i', '--idx', type=int, default=None, help='checkpoint index')
    parser.add_argument('-m', '--mode', type=str, default='val', help='validation or test')
    args = parser.parse_args()

    ##################
    # Choose the model
    ##################
    chosen_log = 'results/' +  args.snap


    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if args.idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[args.idx]
    print(chosen_chkp)
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################
    config.test_mode = True


    ##############
    # Prepare Data
    ##############
    print()
    print('Data Preparation')
    print('****************')

    if args.mode == 'val':
        data_set = 'validation'
    elif args.mode == 'test':
        data_set = 'test'
    else:
        raise ValueError('Unsupported mode : ' + args.mode)

    # Initiate dataset
    if config.dataset == 'H3D':
        test_dataset = H3DDataset(config, set=data_set)
        test_sampler = H3DSampler(test_dataset)
        collate_fn = H3DCollate
    elif config.dataset == 'Paris':
        test_dataset = ParisDataset(config, set=data_set)
        test_sampler = ParisSampler(test_dataset)
        collate_fn = ParisCollate
    elif config.dataset == 'Semantic3D':
        test_dataset = Semantic3DDataset(config, set=data_set)
        test_sampler = Semantic3DSampler(test_dataset)
        collate_fn = Semantic3DCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_num,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    if config.dataset_task == 'classification':
        net = KPCNN(config)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    if config.dataset_task == 'classification':
        tester.classification_test(net, test_loader, config)
    elif config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config)
    elif config.dataset_task == 'slam_segmentation':
        tester.slam_segmentation_test(net, test_loader, config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)