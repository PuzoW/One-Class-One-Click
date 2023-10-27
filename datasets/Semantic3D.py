#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling Semantic3D dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import numpy as np
from pandas import read_csv
import pickle
import torch
from multiprocessing import Lock
import os


# OS functions
from os import listdir
from os.path import exists, join

# Dataset parent class
from datasets.common import *
from torch.utils.data import Sampler, get_worker_info

from utils.config import bcolors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class Semantic3DDataset(PointCloudDataset):
    """Class to handle Semantic3D dataset."""

    def __init__(self, config, set='training', use_potentials=True):

        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self, 'Semantic3D')

        ############
        # Parameters
        ############

        # Dict from labels to names
        self.label_to_names = {
            0: 'man-made terrain',
            1: 'natural terrain',
            2: 'high vegetation',
            3: 'low vegetation',
            4: 'buildings',
            5: 'hard scape',
            6: 'scanning artefacts',
            7: 'cars'
                               }

        # Ascii files dict for testing
        self.ascii_files = {
            'MarketplaceFeldkirch_Station4': 'marketsquarefeldkirch4-reduced.labels',
            'sg27_station10': 'sg27_10-reduced.labels',
            'sg28_Station2': 'sg28_2-reduced.labels',
            'StGallenCathedral_station6': 'stgallencathedral6-reduced.labels',
        }

        # Initialize a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training (can be empty)
        self.ignored_labels = np.array([])

        # Dataset folder
        self.path = 'Data/Semantic3D_reduced-8'
        self.train_path = join(self.path, 'train', 'ply_subsampled')
        self.test_path = join(self.path, 'test')

        # Type of task conducted on this dataset
        self.dataset_task = 'cloud_segmentation'

        self.fea_names = ['red', 'green', 'blue']

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.set = set

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        ################
        # Load ply files
        ################

        ## preparation at first time
        self.prepare_data()

        if self.set == 'training':
            self.cloud_names = ['bildstein_station1_xyz',
                                'bildstein_station3_xyz',
                                'domfountain_station1_xyz',
                                'domfountain_station2_xyz',
                                'neugasse_station1_xyz',
                                'sg27_station1',
                                'sg27_station2',
                                'sg27_station4',
                                'sg27_station5',
                                'sg28_station4',
                                'untermaederbrunnen_station1_xyz',]
            self.files = [os.path.join(self.train_path, n + '.ply') for n in self.cloud_names]

        elif self.set == 'validation':
            self.cloud_names = ['bildstein_station5_xyz',
                                'domfountain_station3_xyz',
                                'sg27_station9',
                                'untermaederbrunnen_station3_xyz']
            self.files = [os.path.join(self.train_path, n + '.ply') for n in self.cloud_names]

        elif self.set == 'test':
            self.cloud_names = ['MarketplaceFeldkirch_Station4',
                                'sg27_station10',
                                'sg28_Station2',
                                'StGallenCathedral_station6']
            self.files = [os.path.join(self.test_path, n + '.ply') for n in self.cloud_names]


        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers

        # point-level
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.input_masks = []
        self.input_tod = []
        self.input_probs = []
        # self.input_pmasks = []

        # coarse_point-level
        self.pot_trees = []
        self.pot_nei = []
        self.pot_local_nei = []
        self.pot_local_w = []
        self.pot_ind = []
        self.pot_tod = []
        self.pot_base_prob = []
        self.pot_prob = []
        self.pot_pred = []
        
        # subcloud-level
        self.seed_pts = []
        self.seed_mask = []
        self.seed_ind = []
        self.seed_pmasks = []
        self.seed_probs =[]
        self.seed_probs_ori = []

        # for test
        self.test_proj = []
        self.validation_labels = []

        # Start loading
        self.load_subsampled_clouds()

        # Number of models used per epoch
        if self.config.test_mode:
            self.epoch_n = len(self.val_cind)
        elif self.set == 'training':
            self.epoch_n = config.epoch_steps
        elif self.set in ['validation', 'test']:
            self.epoch_n = config.validation_size
        else:
            raise ValueError('Unknown set for Semantic3D data: ', self.set)

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = int(config.batch_limit)

        # Initialize potentials
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for i, tree in enumerate(self.pot_trees):
                self.potentials += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)]
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch.from_numpy(np.array(self.argmin_potentials, dtype=np.int64))
            self.min_potentials = torch.from_numpy(np.array(self.min_potentials, dtype=np.float64))
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()

            self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            N = config.epoch_steps * config.batch_num
            self.epoch_inds = torch.from_numpy(np.zeros((2, N), dtype=np.int64))
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()


        self.worker_lock = Lock()

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.cloud_names)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """
        if self.config.test_mode:
            return self.continuous_block_item(batch_i)   #test
        elif self.config.weak_level is None:
            return self.potential_item(batch_i)    #full supervision
        else:
            return self.batch_item(batch_i)     #weak supervision

    def prepare_data(self):
        """
        Download and precompute Seamntic3D point clouds

        """

        if not exists(self.train_path):
            makedirs(self.train_path)
        if not exists(self.test_path):
            makedirs(self.test_path)

        # Folder names
        train_folder = join(self.path, 'train')
        test_folder = join(self.path, 'test')

        print('Preparing training data')
        # Text files containing points
        cloud_names = [file_name[:-4] for file_name in listdir(train_folder) if file_name[-4:] == '.txt']

        for cloud_name in cloud_names:

            # Name of the files
            txt_file = join(train_folder, cloud_name + '.txt')
            label_file = join(train_folder, cloud_name + '.labels')

            sub_cloud_name = cloud_name[:-14]
            ply_file_full = join(self.train_path, sub_cloud_name + '.ply')

            # Pass if already done
            if exists(ply_file_full):
                print('{:s} already done\n'.format(sub_cloud_name))
                continue

            print('Preparation of {:s}'.format(sub_cloud_name))

            # data = []
            # with open(txt_file) as txt_data:
            #     for line in txt_data:
            #         pt = np.fromstring(line, dtype=np.float32, sep=' ')
            #         data.append(pt)
            # data = np.array(data)
            data = read_csv(txt_file, delimiter=' ', header=None,
                            dtype={0: np.float32, 1: np.float32, 2: np.float32,
                                   3: np.int32, 4: np.uint8, 5: np.uint8, 6: np.uint8})
            points = data.iloc[:, [0, 1, 2]].values
            colors = data.iloc[:, [4, 5, 6]].values
            print(points.shape)

            # Load labels
            # labels = np.loadtxt(label_file, dtype=np.int32)
            labels = read_csv(label_file, delimiter=' ').iloc[:, 0].values.astype(np.int32)
            valid_ind = np.where(labels > 0)[0]


            # Subsample to save space
            sub_points, sub_colors, sub_labels = grid_subsampling(points[valid_ind],
                                                                  features=colors[valid_ind],
                                                                  labels=labels[valid_ind]-1,
                                                                  sampleDl=0.01)

            # Write the subsampled ply file
            write_ply(ply_file_full,
                      [sub_points, sub_colors.astype(np.uint8), sub_labels],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])


        print('Preparing test data')
        # Text files containing points
        cloud_names = [file_name[:-4] for file_name in listdir(test_folder) if file_name[-4:] == '.txt']

        for cloud_name in cloud_names:

            # Name of the files
            txt_file = join(test_folder, cloud_name + '.txt')

            sub_cloud_name = cloud_name[:-22]
            ply_file_full = join(self.test_path, sub_cloud_name + '.ply')

            # Pass if already done
            if exists(ply_file_full):
                print('{:s} already done\n'.format(sub_cloud_name))
                continue

            print('Preparation of {:s}'.format(sub_cloud_name))
            data = read_csv(txt_file, delimiter=' ', header=None,
                            dtype={0: np.float32, 1: np.float32, 2: np.float32,
                                   3: np.int32, 4: np.uint8, 5: np.uint8, 6: np.uint8})
            points = data.iloc[:, [0, 1, 2]].values
            colors = data.iloc[:, [4, 5, 6]].values
            print(points.shape)
            write_ply(ply_file_full,
                      [points, colors],
                      ['x', 'y', 'z', 'red', 'green', 'blue'])

    def batch_item(self, batch_i):
        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        gl_list = []
        pl_list = []
        m_list = []
        pm_list = []
        pi_list = []
        ci_list = []
        bi_list = []
        s_list = []
        R_list = []

        for i in range(len(self.batches[batch_i])):

            t += [time.time()]

            cloud_ind = self.batches_cloud[batch_i][i]
            point_ind = self.batches[batch_i][i]

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)
            # Indices of points in input region
            center_point = self.seed_pts[cloud_ind][point_ind]
            input_inds = self.seed_ind[cloud_ind][point_ind]

            t += [time.time()]

            # Collect labels and colors
            input_points = points[input_inds].astype(np.float32)
            input_points[:, :] -= center_point
            input_colors = self.input_colors[cloud_ind][input_inds]
            input_masks = self.input_masks[cloud_ind][input_inds]
            if self.config.weak_level is not None:
                wl_masks = np.zeros(input_points.shape[0], dtype=np.float32)
                wl_masks[self.seed_mask[cloud_ind][point_ind]] = 1.0
                input_masks[:] *= wl_masks
            if self.set in ['test']:
                input_labels = np.zeros(input_points.shape[0], dtype=np.int32)
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])
            input_glabels = np.zeros(self.num_classes, dtype=np.float32)
            input_glabels[np.unique(input_labels[np.where(input_masks > 1e-6)])] = 1.0

            input_plabels = np.argmax(self.seed_probs[cloud_ind][point_ind], axis=1)
            input_pmasks = self.seed_pmasks[cloud_ind][point_ind]
            input_pmasks[self.seed_mask[cloud_ind][point_ind]] = 0.0

            t += [time.time()]

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack((input_colors, input_points[:, 2][:, None])).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            gl_list += [input_glabels[None, :]]
            pl_list += [input_plabels]
            m_list += [input_masks]
            pm_list += [input_pmasks]
            pi_list += [input_inds]
            ci_list += [cloud_ind]
            bi_list += [point_ind]
            s_list += [scale]
            R_list += [R]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        glabels = np.concatenate(gl_list, axis=0)
        plabels = np.concatenate(pl_list, axis=0)
        masks = np.concatenate(m_list, axis=0)
        pmasks = np.concatenate(pm_list, axis=0)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        batch_inds = np.array(bi_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 3:
            stacked_features = features[:, :-1]
        elif self.config.in_features_dim == 4:
            stacked_features = features

        else:
            raise ValueError('Only accepted input dimensions are 1, 3, and 4 (without and with rgb)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points, stack_lengths)

        t += [time.time()]

        input_list += [stacked_features, labels, glabels, plabels, masks, pmasks]
        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, batch_inds, input_inds]

        return input_list

    def continuous_block_item(self, batch_i):

        t = [time.time()]

        p_list = []
        f_list = []
        l_list = []
        gl_list = []
        pl_list = []
        m_list = []
        pm_list = []
        pi_list = []
        ci_list = []
        bi_list = []
        s_list = []
        R_list = []
        batch_n = 0

        t += [time.time()]

        if True:
            cloud_ind = self.val_cind[batch_i]
            point_ind = 0  # meaningless

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Indices of points in input region
            center_point = self.val_center_pts[batch_i]
            input_inds = self.val_ind[batch_i]

            t += [time.time()]

            # Collect labels and colors
            input_points = points[input_inds].astype(np.float32)
            input_points[:, :] -= center_point
            input_colors = self.input_colors[cloud_ind][input_inds]
            input_masks = self.input_masks[cloud_ind][input_inds]
            if self.set in ['test']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])
            input_glabels = np.zeros(self.num_classes, dtype=np.float32)

            input_plabels = np.zeros_like(input_labels)
            input_pmasks = np.zeros_like(input_masks)

            t += [time.time()]

            # No data augmentation
            # input_points, scale, R = self.augmentation_transform(input_points)
            scale = np.random.rand(input_points.shape[1]).astype(np.float32)
            R = np.eye(input_points.shape[1]).astype(np.float32)

            # Get original height as additional feature
            input_features = np.hstack((input_colors, input_points[:, 2][:, None])).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            gl_list += [input_glabels[None, :]]
            pl_list += [input_plabels]
            m_list += [input_masks]
            pm_list += [input_pmasks]
            pi_list += [input_inds]
            bi_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        glabels = np.concatenate(gl_list, axis=0)
        plabels = np.concatenate(pl_list, axis=0)
        masks = np.concatenate(m_list, axis=0)
        pmasks = np.concatenate(pm_list, axis=0)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        batch_inds = np.array(bi_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stacked_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 3:
            stacked_features = features[:, :3]
        elif self.config.in_features_dim == 4:
            stacked_features = features
        else:
            raise ValueError('Only accepted input dimensions are 1, 3 and 4 (without and with RGB)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points, stacked_lengths)

        t += [time.time()]

        input_list += [stacked_features, labels, glabels, plabels, masks, pmasks]
        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, batch_inds, input_inds]

        return input_list


    def potential_item(self, batch_i):

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        gl_list = []
        m_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0

        while True:

            t += [time.time()]

            with self.worker_lock:
                # Get potential minimum
                cloud_ind = int(torch.argmin(self.min_potentials))
                point_ind = int(self.argmin_potentials[cloud_ind])

                # Get potential points from tree structure
                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)

                # Center point of input region
                # circle
                center_point = pot_points[point_ind, :].reshape(1, -1)

                # Add a small noise to center point
                center_point += np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)

                # Indices of points in input region
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(center_point,
                                                                         r=self.config.in_radius,
                                                                         return_distance=True)

                d2s = np.square(dists[0])
                pot_inds = pot_inds[0]

                # Update potentials (Tukey weights)
                if len(pot_inds) > 1:
                    tukeys = np.square(1 - d2s / np.max(d2s))
                    tukeys[tukeys < 0] = 0
                    self.potentials[cloud_ind][pot_inds] += torch.from_numpy(tukeys)
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind
                else:
                    self.potentials[cloud_ind][pot_inds] += torch.from_numpy(np.array(1e6))
                    min_ind = torch.argmin(self.potentials[cloud_ind])
                    self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                    self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)


            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            t += [time.time()]

            # Number collected
            n = input_inds.shape[0]
            if n==1:
                continue

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

            # Collect labels and colors
            input_points = points[input_inds].astype(np.float32)
            input_points[:, :] -= center_point[0, :]
            input_colors = self.input_colors[cloud_ind][input_inds]
            input_masks = self.input_masks[cloud_ind][input_inds]
            if self.set in ['test']:
                input_labels = np.zeros(input_points.shape[0], dtype=np.int32)
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])
            input_glabels = np.zeros(self.num_classes, dtype=np.float32)
            input_glabels[np.unique(input_labels[np.where(input_masks>0.1)])] = 1.0

            t += [time.time()]

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature
            input_features = np.hstack((input_colors, input_points[:, 2][:, None])).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
            l_list += [input_labels]
            gl_list += [input_glabels[None, :]]
            m_list += [input_masks]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

        ###################
        # Concatenate batch
        ###################

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        glabels = np.concatenate(gl_list, axis=0)
        masks = np.concatenate(m_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 3:
            stacked_features = features[:, :-1]
        elif self.config.in_features_dim == 4:
            stacked_features = features
        else:
            raise ValueError('Only accepted input dimensions are 1, 3, and 4 (without and with RGB)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points, stack_lengths)

        t += [time.time()]

        input_list += [stacked_features, labels, glabels, labels.copy(), masks, masks.copy()]
        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds, input_inds]

        return input_list


    def load_subsampled_clouds(self):

        # Parameter
        dl = self.config.first_subsampling_dl

        if self.set == 'training':
            subclouds_perdata = int(self.config.al_initnum/len(self.files))

        # Create path for files
        self.tree_path = join(self.train_path, 'input_{:.3f}'.format(dl))
        if not exists(self.tree_path):
            makedirs(self.tree_path)

        self.val_center_pts = []
        self.val_ind = []
        self.val_cind = []

        total_points = 0
        wl_points = 0

        ##############
        # Load KDTrees
        ##############

        for i, file_path in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[i]

            # Name of the input files
            KDTree_file = join(self.tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(self.tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_labels = data['class']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                if self.set == 'test':
                    labels = np.zeros(points.shape[0], dtype=np.int32)
                else:
                    labels = data['class']


                # Subsample cloud
                sub_points, sub_colors, sub_labels = grid_subsampling(points,
                                                                      features=colors,
                                                                      labels=labels,
                                                                      sampleDl=dl)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                sub_labels = np.squeeze(sub_labels)


                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file,
                          [sub_points, sub_colors, sub_labels],
                          ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            sub_mask = np.ones(len(sub_labels), dtype=np.float32)

            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_labels += [sub_labels]
            self.input_masks += [sub_mask]
            self.input_tod += [np.zeros(len(sub_labels), dtype=np.float32)]
            
            subclouds_file = join(self.tree_path, '{:s}_r{:.1f}.pkl'.format(cloud_name, self.config.in_radius))

            # Check if inputs have already been computed
            if exists(subclouds_file):
                print('\nFound subclouds for cloud {:s}, radius {:.1f}'.format(cloud_name, self.config.in_radius))
                # Read pkl with search tree
                with open(subclouds_file, 'rb') as f:
                    center_pts, batch_inds = pickle.load(f)

            else:
                print('\nPreparing subclouds for cloud {:s}, radius {:.1f}'.format(cloud_name, self.config.in_radius))

                center_pts, batch_inds = self.subcloud_initilization(search_tree)

                with open(subclouds_file, 'wb') as f:
                    pickle.dump([center_pts, batch_inds], f)
            
            # times = np.zeros(len(search_tree.data), dtype=np.int32)
            # for b_inds in batch_inds:
            #     times[b_inds] += 1
            # pot_name = join(self.tree_path, '{:s}_pot.ply'.format(cloud_name))
            # write_ply(pot_name,
            #           [np.array(search_tree.data), times],
            #           ['x', 'y', 'z', 'pots'])


            self.val_center_pts.extend(center_pts)
            self.val_ind.extend(batch_inds)
            self.val_cind.extend(i * np.ones(len(center_pts), dtype=np.int32))

            if self.set == 'training':

                # continue training
                if self.config.previous_training_path:
                    subcloud_folder = join('results', self.config.dataset, self.config.previous_training_path, 'subclouds')
                    n_epoch =  np.sort(os.listdir(subcloud_folder))[self.config.chkp_idx]
                    prob_file = join(subcloud_folder, n_epoch, '{:s}_prob.npy'.format(cloud_name))
                    with open(prob_file, 'rb') as f:
                        self.input_probs += [np.load(f).astype(np.float32)]
                    sc_file = join(subcloud_folder, n_epoch, '{:s}_batch.pkl'.format(cloud_name))
                    with open(sc_file, 'rb') as f:
                        init_center_pts, init_weak_mask, init_batch_inds = pickle.load(f)
                    init_batch_probs = [self.input_probs[i][inds] for inds in init_batch_inds]
                    init_batch_pmasks = [1 + np.sum(probs * (np.log(probs + 1e-10)), axis=1) / np.log(self.num_classes) 
                                        for probs in init_batch_probs]
                
                else:
                    # load weak labels
                    weaklabel_file = join(self.tree_path, '{:s}_wl{:d}.pkl'.format(cloud_name, self.config.weak_level))
                    if exists(weaklabel_file):
                        print('\nFound weak labels for cloud {:s}, weak level {:d}'.format(cloud_name, self.config.weak_level))
                        with open(weaklabel_file, 'rb') as f:
                            weak_mask = pickle.load(f)
                    else:
                        print('\nPreparing weak labels for cloud {:s}, weak level {:d}'.format(cloud_name, self.config.weak_level))
                        weak_mask = self.weaklabel_initilization(batch_inds, sub_labels)

                        with open(weaklabel_file, 'wb') as f:
                            pickle.dump(weak_mask, f)

                    # load initial sub-clouds
                    if self.config.al_initnum==0:
                        random_sel = np.arange(len(center_pts))
                    else:
                        batch_file = join(self.tree_path,
                                        '{:s}_r{:.1f}_sc{:d}_s{:s}.txt'.format(cloud_name, self.config.in_radius,
                                                                            self.config.al_initnum, self.config.serial))
                        if exists(batch_file):
                            random_sel = np.loadtxt(batch_file).astype(np.int32)
                        else:
                            random_sel = np.random.choice(range(len(center_pts)), size=subclouds_perdata,
                                                        replace=False)
                            np.savetxt(batch_file, random_sel, fmt='%d')

                    init_center_pts = center_pts[random_sel]
                    init_weak_mask = [weak_mask[i] for i in random_sel]
                    init_batch_inds = [batch_inds[i] for i in random_sel]
                    init_batch_pmasks = [np.zeros(len(batch_inds[i]), dtype=np.float32) for i in random_sel]
                    init_batch_probs = [np.zeros((len(batch_inds[i]), self.num_classes), dtype=np.float32) for i in random_sel]
                    self.input_probs += [np.zeros((len(sub_labels), self.num_classes), dtype=np.float32)]

                self.seed_pts += [init_center_pts]
                self.seed_mask += [init_weak_mask]
                self.seed_ind += [init_batch_inds]
                self.seed_pmasks += [init_batch_pmasks]
                self.seed_probs += [init_batch_probs]
                self.seed_probs_ori += [init_batch_probs.copy()]

                # count label distribution
                if self.config.weak_level is None:
                    self.num_per_class += np.bincount(sub_labels)
                else:
                    wl_ind = []
                    for b, w in enumerate(init_weak_mask):
                        wl_ind.extend(init_batch_inds[b][w])
                    wl_ind = np.array(wl_ind)
                    wl = sub_labels[wl_ind]
                    count_wl = np.bincount(wl)
                    for c, n in enumerate(count_wl):
                        self.num_per_class[c] += n

                    total_points += len(sub_labels)
                    wl_points += len(wl)

        if self.set == 'training' and self.config.segloss_balance == 'class':
            assert np.min(self.num_per_class) > 0, "Not all categories are considered in initialized labels"
            self.class_weight()

        if self.set == 'training' and self.config.weak_level != None:
            print('The number of weak label: {:d}, accounting for {:.2f}‱ of total points ({:d})'
                  .format(wl_points, 10000.0*wl_points/total_points, total_points))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials and self.set == 'training':
            print('\nPreparing potentials')

            # Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 5
            cloud_ind = 0

            for i, file_path in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[i]

                # Name of the input files
                coarse_KDTree_file = join(self.tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))

                # Check if inputs have already been computed
                if exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, 'rb') as f:
                        coarse_search_tree = pickle.load(f)

                    # np.savetxt(join(self.tree_path, '{:s}_coarse_KDTree.txt'.format(cloud_name)),
                    #            coarse_search_tree.data,
                    #            fmt='%.2f')

                else:
                    # Subsample cloud
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)

                    # Get chosen neighborhoods
                    coarse_search_tree = KDTree(coarse_points[:], leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(coarse_search_tree, f)

                # Fill data containers
                self.pot_trees += [coarse_search_tree]
                self.pot_tod += [np.zeros(len(coarse_search_tree.data), dtype=np.float32)]
                self.pot_pred += [np.zeros(len(coarse_search_tree.data), dtype=np.int32)]
                self.pot_prob += [np.zeros((len(coarse_search_tree.data), self.num_classes), dtype=np.float32)]
                self.pot_base_prob += [
                    np.zeros((len(coarse_search_tree.data), self.num_classes), dtype=np.float32)]

                # store pot neighbors to speed up subclouds generation
                # continue # for full supervision

                coarse_neighbors_file = join(self.tree_path, '{:s}_coarse_neighbors_{:.2f}.pkl'.format(cloud_name,
                                                                                                        self.config.in_radius))

                if exists(coarse_neighbors_file):
                    with open(coarse_neighbors_file, 'rb') as f:
                        pot_ind, pot_nei, pot_local_nei, pot_local_dis = pickle.load(f)

                else:
                    _, pot_ind = self.input_trees[cloud_ind].query(coarse_search_tree.data, k=1)
                    pot_ind = np.squeeze(pot_ind)
                    pot_nei = coarse_search_tree.query_radius(coarse_search_tree.data, self.config.in_radius*2)
                    pot_local_dis, pot_local_nei = coarse_search_tree.query(coarse_search_tree.data, k=15)

                    with open(coarse_neighbors_file, 'wb') as f:
                        pickle.dump([pot_ind, pot_nei, pot_local_nei, pot_local_dis], f)

                self.pot_ind += [pot_ind]
                self.pot_nei += [pot_nei]
                self.pot_local_nei += [pot_local_nei]

                d2s = np.square(pot_local_dis)
                dis_w = np.square(1 - d2s / np.max(d2s, axis=1)[:, np.newaxis])
                dis_w[dis_w < 0] = 0
                self.pot_local_w += [dis_w]


                cloud_ind += 1

            print('Done in {:.1f}s'.format(time.time() - t0))


        ######################
        # Reprojection indices
        ######################

        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        # Only necessary for validation and test sets
        if self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[i]

                # File name for saving
                proj_file = join(self.tree_path, '{:s}_proj.pkl'.format(cloud_name))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    if self.set == 'test':
                        labels = np.zeros(points.shape[0], dtype=np.int32)
                    else:
                        labels = data['class'].astype(np.int32)

                    # Compute projection inds
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()
        return

    
    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class Semantic3DSampler(Sampler):
    """Sampler for Semantic3D"""

    def __init__(self, dataset: Semantic3DDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset
        self.batch_limit = dataset.batch_limit

        # Number of step per epoch
        self.N = dataset.epoch_n

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N
    
    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
        """
        Method performing batch and neighbors calibration.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.tree_path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            #####################
            # Perform calibration
            #####################

            for _, batch in enumerate(dataloader):
                # Update neighborhood histogram
                counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                neighb_hists += np.vstack(hists)

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)

        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class Semantic3DCustomBatch:
    """Custom batch definition with memory pinning for Semantic3D"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        # input_list = input_list[0]

        # Number of layers
        L = (len(input_list) - 11) // 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind + L]]
        ind += L
        self.features = torch.from_numpy(input_list[ind])
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.glabels = torch.from_numpy(input_list[ind])
        ind += 1
        self.plabels = torch.from_numpy(input_list[ind])
        ind += 1
        self.masks = torch.from_numpy(input_list[ind])
        ind += 1
        self.pmasks = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.batch_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()
        self.glabels = self.glabels.pin_memory()
        self.plabels = self.plabels.pin_memory()
        self.masks = self.masks.pin_memory()
        self.pmasks = self.pmasks.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.batch_inds = self.batch_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.glabels = self.glabels.to(device)
        self.plabels = self.plabels.to(device)
        self.masks = self.masks.to(device)
        self.pmasks = self.pmasks.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def Semantic3DCollate(batch_data):
    return Semantic3DCustomBatch(batch_data[0])