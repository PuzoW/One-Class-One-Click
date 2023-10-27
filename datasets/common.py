#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling datasets
#
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import os
from os import makedirs, remove, rename, listdir
from os.path import exists, join
import numpy as np
from numpy.linalg import norm
import random
import pickle
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KDTree
from utils.config import Config
from kernels.kernel_points import create_3D_rotations

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors

# PLY reader
from utils.ply import write_ply, read_ply
from utils.las import read_las

# Configuration class
from utils.config import Config

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Sunsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class definition
#       \**********************/


class PointCloudDataset(Dataset):
    """Parent class for Point Cloud Datasets."""

    def __init__(self, name):
        """
        Initialize parameters of the dataset here.
        """

        self.name = name
        self.path = ''
        self.label_to_names = {}
        self.num_classes = 0
        self.label_values = np.zeros((0,), dtype=np.int32)
        self.label_names = []
        self.label_to_idx = {}
        self.name_to_label = {}
        self.config = Config()
        self.neighborhood_limits = []

        return

    def __len__(self):
        """
        Return the length of data here
        """
        return 0

    def __getitem__(self, idx):
        """
        Return the item at the given index
        """

        return 0

    def init_labels(self):

        # Initialize all label parameters given the label_to_names dict
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.num_per_class = np.zeros(self.num_classes, dtype=np.float32)

    def cal_label_weight(self):
        # print the number of labeled points in each class
        print('number of labeled points in each class: ', self.num_per_class)
        weight = 1 / (np.sqrt(self.num_per_class))
        self.label_weight = weight / np.sum(weight)
        self.label_weight /= np.min(self.label_weight)

    def class_weight(self):
        self.cal_label_weight()
        for n, m in enumerate(self.input_masks):
            self.input_masks[n] = self.label_weight[self.input_labels[n]]

    def augmentation_transform2(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Create random rotations
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Choose two random angles for the first vector in polar coordinates
                theta = np.random.rand() * 2 * np.pi
                phi = (np.random.rand() - 0.5) * np.pi

                # Create the first vector in carthesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle
                alpha = np.random.rand() * 2 * np.pi

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise * 2.0).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        # augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise

        pt_idxs = np.arange(0, points.shape[0])
        np.random.shuffle(pt_idxs)
        num = int(points.shape[0]*0.9)
        augmented_points = augmented_points[pt_idxs[0:num], :]

        if normals is None:
            return augmented_points, scale, R, pt_idxs[0:num]
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2] * 0, augmented_points[:, 2] * 0 + 1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R, pt_idxs[0:num]

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.config.augment_rotation == 'vertical':

                # Create random rotations
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.config.augment_rotation == 'all':

                # Choose two random angles for the first vector in polar coordinates
                theta = np.random.rand() * 2 * np.pi
                phi = (np.random.rand() - 0.5) * np.pi

                # Create the first vector in carthesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle
                alpha = np.random.rand() * 2 * np.pi

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.config.augment_scale_min
        max_s = self.config.augment_scale_max
        if self.config.augment_scale_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) - min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.config.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors

    def classification_inputs(self,
                              stacked_points,
                              stacked_features,
                              labels,
                              stack_lengths):

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 1), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # Save deform layers

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_stack_lengths
        li += [stacked_features, labels]

        return li


    def segmentation_inputs(self, stacked_points, stack_lengths):

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                if self.set == 'training':
                    pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)
                else:
                    pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl, random_grid_orient=False)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors(stacked_points, pool_p, stack_lengths, pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 3), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
                up_i = np.zeros((0, 1), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            if up_i.shape[0] > 0:
                up_i = self.big_neighborhood_filter(up_i, len(input_points)+1)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_upsamples += [up_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths

        return li

    # generate mini-batch
    def epoch_batch(self):
        self.batches = []
        self.batches_cloud = []

        # gather subclouds from all files
        seeds = []
        seeds_cloud = []
        for i, s in enumerate(self.seed_ind):
            # record seed indices and cloud indices
            seeds.extend(np.arange(len(s)).astype(np.int32))
            seeds_cloud.extend(np.ones(len(s), dtype=np.int32) * i)

        # ensure sufficient batches in early training cycles
        while len(self.batches)<20: 
            # shuffle seeds
            ind = list(np.arange(len(seeds)))
            random.shuffle(ind)

            while ind:
                batch = []
                batch_c = []
                batch_num = 0

                while ind:
                    n = ind.pop()
                    batch_num += len(self.seed_ind[seeds_cloud[n]][seeds[n]])
                    if batch_num > int(self.batch_limit) and len(batch) > 0:
                        ind.append(n)
                        break

                    batch.append(seeds[n])
                    batch_c.append(seeds_cloud[n])

                self.batches.extend([batch])
                self.batches_cloud.extend([batch_c])

    def subcloud_initilization(self, search_tree):

        sub_points = np.array(search_tree.data)
        center_pts = []
        batch_inds = []

        # potential search
        pot = np.random.rand(sub_points.shape[0]) * 1e-3

        while np.min(pot) < 1.0:
            min_ind = np.argmin(pot)
            center_point = sub_points[min_ind].reshape(1, -1)
            pot_inds, dists = search_tree.query_radius(center_point, r=self.config.in_radius, return_distance=True)

            if len(pot_inds[0]) < 1000:
                _, pot_inds = search_tree.query(center_point, k=1000)
                tukeys = 1
            else:
                # potential function
                tukeys = np.sqrt(1 - dists[0] / self.config.in_radius)
                tukeys[tukeys < 0] = 0
            pot_inds = pot_inds[0]
            pot[pot_inds]+=tukeys

            center_pts.extend(center_point)
            batch_inds.append(pot_inds)

        center_pts = np.array(center_pts, dtype=np.float32)

        return center_pts, batch_inds


    def weaklabel_initilization(self, batch_inds, sub_labels):
        weak_mask = []

        for _, inds in enumerate(batch_inds):
            # collect labels of each subcloud
            labels = sub_labels[inds]
            # get contained classes
            glabels = np.unique(labels)
            # get weak labels
            i_ind = []
            for l in glabels:
                # get point index of each class
                l_ind = np.where(labels == l)[0]
                weak_level = min(self.config.weak_level, len(l_ind))
                # randomly select weak labels based on weak level
                i_ind.append(np.random.choice(l_ind, weak_level, replace=False))
            i_ind = np.array(i_ind).squeeze(-1)

            weak_mask.append(i_ind)

        return weak_mask


    # update pseudo labels
    def update_subcloud_pl(self, probs, batch, p_c, po_c, smooth=0.9):
        stacked_probs = probs.cpu().detach().numpy()
        stacked_gl = batch.glabels.cpu().detach().numpy()
        lengths = batch.lengths[0].cpu().numpy()
        cloud_inds = batch.cloud_inds.cpu().numpy()
        batch_inds = batch.batch_inds.cpu().numpy()
        stacked_labels = batch.labels.cpu().detach().numpy()

        # weighted importance based on Shannon entropy
        stacked_pl_w = 1 + np.sum(stacked_probs * (np.log(stacked_probs + 1e-10)), axis=1) / np.log(self.num_classes)

        i0 = 0
        for i, length in enumerate(lengths):
            # Get prediction
            c_i = cloud_inds[i]
            b_i = batch_inds[i]

            probs_i = stacked_probs[i0:i0 + length]
            # exponential moving average
            self.seed_probs_ori[c_i][b_i] = smooth * self.seed_probs_ori[c_i][b_i] + (1 - smooth) * probs_i
            preds_ori_i = np.argmax(self.seed_probs_ori[c_i][b_i], axis=1).astype(np.int32)
            # scene-level context constraint
            gl_i = stacked_gl[i]
            probs_i *= gl_i
            self.seed_probs[c_i][b_i] = smooth * self.seed_probs[c_i][b_i] + (1 - smooth) * probs_i
            preds_i = np.argmax(self.seed_probs[c_i][b_i], axis=1).astype(np.int32)

            w_i = stacked_pl_w[i0:i0 + length]
            self.seed_pmasks[c_i][b_i] = w_i

            # accuracy statistics
            l_i = stacked_labels[i0:i0 + length]
            correct_ori = (preds_ori_i == l_i).sum()
            correct = (preds_i == l_i).sum()
            p_c += np.array((correct, len(l_i)))
            po_c += np.array((correct_ori, len(l_i)))

            i0 += length

    # Save subclouds for next training cycle
    def save_subclouds(self, folder):

        for i in range(len(self.seed_pts)):
            weak_mask = self.seed_mask[i]
            batch_inds = self.seed_ind[i]
            batch_center = self.seed_pts[i]
            search_tree = self.input_trees[i]
            sub_colors = self.input_colors[i]
            sub_labels = self.input_labels[i]
            cloud_name = self.cloud_names[i]

            batch_prob = self.seed_probs
            batch_prob_ori = self.seed_probs_ori
            probs = np.zeros((len(sub_labels), self.num_classes), dtype=np.float32)
            probs_ori = np.zeros((len(sub_labels), self.num_classes), dtype=np.float32)

            batch_file = join(folder, '{:s}_batch.pkl'.format(cloud_name))
            with open(batch_file, 'wb') as f:
                pickle.dump([batch_center, weak_mask, batch_inds], f)

            wl_ind = []
            pts_ind = []
            for b, w in enumerate(weak_mask):
                wl_ind.extend(batch_inds[b][w])
                pts_ind.extend(batch_inds[b])

                probs[batch_inds[b]] += batch_prob[i][b]
                probs_ori[batch_inds[b]] += batch_prob_ori[i][b]

            sub_points = np.array(search_tree.data)
            wl_ind = np.array(wl_ind)
            pts_ind = np.array(pts_ind)
            pts_ind, pts_counts = np.unique(pts_ind, return_counts=True)

            preds = np.argmax(probs[pts_ind], axis=1).astype(np.int32)
            preds_ori = np.argmax(probs_ori[pts_ind], axis=1).astype(np.int32)

            sc_file = join(folder, '{:s}_sc{:.1f}'.format(cloud_name, self.config.in_radius))
            sc_cols = ['x', 'y', 'z', 'pred_ori', 'pred', 'label']
            sc_cols[3:3] = self.fea_names
            write_ply(sc_file,
                      [sub_points[pts_ind], sub_colors[pts_ind], preds_ori, preds, sub_labels[pts_ind]],
                      sc_cols)
            wl_file = join(folder, '{:s}_wl{:.1f}'.format(cloud_name, self.config.in_radius))
            wl_cols = ['x', 'y', 'z', 'label']
            wl_cols[3:3] = self.fea_names
            write_ply(wl_file,
                      [sub_points[wl_ind], sub_colors[wl_ind], sub_labels[wl_ind]],
                      wl_cols)

    # save predicted info
    def save_info(self, folder):

        for i, tree in enumerate(self.pot_trees):
            points = np.array(tree.data, dtype=np.float32)
            tod = self.pot_tod[i]
            pred = self.pot_pred[i]
            label = self.input_labels[i][self.pot_ind[i]]
            cloud_name = self.cloud_names[i]

            write_ply(join(folder, '{:s}_info'.format(cloud_name)),
                      [points, tod, pred, label],
                      ['x', 'y', 'z', 'tod', 'pred', 'label'])

            with open(join(folder, '{:s}_prob.npy'.format(cloud_name)), 'wb') as f:
                np.save(f, self.input_probs[i].astype(np.float16))

    # Choose subclouds and weak labels based on TOD
    def add_labels(self):

        argmax_pot = []
        max_pot = []
        added_wl = []
        added_l = []
        sel_info = []
        # sel_sc_infos = []
        sel_inds = []
        sel_trees = []

        m_info = self.pot_tod
        p_info = self.input_tod

        for i in range(len(self.input_trees)):

            # Get local maximum seeds
            sel_ind = []
            for p, nei in enumerate(self.pot_nei[i]):
                if len(nei) > 1 and m_info[i][p] >= np.max(m_info[i][nei]):
                    sel_ind.append(p)
            sel_ind = np.array(sel_ind)
            sel_inds += [sel_ind]

            sel_info += [m_info[i][sel_ind]]
            max_ind = np.argmax(sel_info[i])
            argmax_pot += [max_ind]
            max_pot += [sel_info[i][max_ind]]

            sel_tree = KDTree(np.array(self.pot_trees[i].data)[sel_ind], leaf_size=10)
            sel_trees += [sel_tree]

            # sel_sc_info = np.zeros((len(sel_ind), self.num_classes), dtype=np.float32)
            # for n, ind in enumerate(sel_ind):
            #     sc_pred = self.pot_prob[i]
            #     sel_sc_info[n] = np.mean(sc_pred[self.pot_nei[i][ind]], axis=0)
            # sel_sc_infos += [sel_sc_info]

        # Top-K subclouds with highest TOD
        for i in range(self.config.al_num):
            cloud_ind = np.argmax(max_pot)
            point_ind = argmax_pot[cloud_ind]

            pot_points = np.array(sel_trees[cloud_ind].data, copy=False)
            center_point = pot_points[point_ind]
            n_inds = sel_trees[cloud_ind].query_radius(center_point.reshape(1, -1), r=self.config.in_radius)[0]
            sel_info[cloud_ind][n_inds] = 0.0

            max_ind = np.argmax(sel_info[cloud_ind])
            argmax_pot[cloud_ind] = max_ind
            max_pot[cloud_ind] = sel_info[cloud_ind][max_ind]

            inds = self.input_trees[cloud_ind].query_radius(center_point.reshape(1, -1), r=self.config.in_radius)[0]
            if len(inds) < 1000:
                _, inds = self.input_trees[cloud_ind].query(center_point.reshape(1, -1), k=1000)
                inds = inds[0]

            labels = self.input_labels[cloud_ind]
            sub_labels = labels[inds]

            sub_info = p_info[cloud_ind][inds]
            glabels = np.unique(sub_labels)
            i_ind = []
            added_wl.extend(glabels)
            added_l.extend(sub_labels)


            # ococ labeling simulation

            # focus on salient regions
            points = np.array(self.input_trees[cloud_ind].data)
            sub_points = points[inds]
            patch_tree = KDTree(sub_points, leaf_size=10)
            _, neighbors = patch_tree.query(sub_points, k=30)
            nei_label = sub_labels[neighbors]
            homo_label = nei_label - nei_label[:, 0][:, None]
            nei_info = (homo_label == 0).astype(int)
            indicator = np.sum(nei_info * sub_info[neighbors], axis=1)

            # avoid edge area
            sub_points2d = sub_points[:, :2] - center_point[:2]
            dist = np.sqrt(np.sum(np.square(sub_points2d), axis=1))
            dis_decay = 1 - np.exp(-self.config.in_radius * (self.config.in_radius - dist))
            dis_decay[dis_decay < 0] = 0
            indicator *= dis_decay

            for l in glabels:
                l_ind = np.where(sub_labels == l)[0]
                l_in = indicator[l_ind]
                i_ind.append([l_ind[np.argmax(l_in)]])
                # i_ind.append(np.random.choice(l_ind, 1))
            i_ind = np.array(i_ind).squeeze(-1)

            self.seed_pts[cloud_ind] = np.append(self.seed_pts[cloud_ind], [center_point], axis=0)
            self.seed_mask[cloud_ind].append(i_ind)
            self.seed_ind[cloud_ind].append(inds)
            self.seed_probs[cloud_ind].append(np.zeros((len(inds), self.num_classes), dtype=np.float32))
            self.seed_pmasks[cloud_ind].append(np.zeros(len(inds), dtype=np.float32))
            self.seed_probs_ori[cloud_ind].append(np.zeros((len(inds), self.num_classes), dtype=np.float32))

            # save individual subclouds
            # if i < 10:
            #     sub_color = self.input_colors[cloud_ind][inds]
            #     name_cols = ['x', 'y', 'z', 'info', 'indicator', 'label']
            #     name_cols[3:3]=self.fea_names
            #     write_ply(join(save_folder, str(i)),
            #               [sub_points, sub_color, sub_info, indicator, sub_labels],
            #               name_cols)
            #     np.savetxt(join(save_folder, str(i) + '_wl.txt'),
            #                np.hstack((sub_points[i_ind], sub_labels[i_ind, None])),
            #                fmt=['%.2f', '%.2f', '%.2f', '%d'])

        # reset label weight
        count_wl = np.bincount(added_wl)
        for n in range(len(count_wl)):
            self.num_per_class[n] += count_wl[n]
        if self.config.segloss_balance == 'class':
            self.class_weight()

    def random_add_labels(self):

        # randomly choose center points from potential points
        sel_inds = []
        cloud_inds = []
        for n, pot in enumerate(self.pot_trees):
            sel_inds.extend(np.arange(len(pot.data)))
            cloud_inds.extend(np.ones(len(pot.data), dtype=np.int32) * n)
        sel_inds = np.array(sel_inds)
        cloud_inds = np.array(cloud_inds)
        inds = np.random.choice(np.arange(len(sel_inds)), size=self.config.al_num, replace=False)

        added_wl = []
        added_l = []
        for ind in inds:
            cloud_ind = cloud_inds[ind]
            point_ind = sel_inds[ind]

            pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)
            center_point = pot_points[point_ind]
            inds = self.input_trees[cloud_ind].query_radius(center_point.reshape(1, -1), r=self.config.in_radius)[0]
            if len(inds) < 1000:
                _, inds = self.input_trees[cloud_ind].query(center_point.reshape(1, -1), k=1000)
                inds = inds[0]

            # randomly collect labels of each subcloud
            labels = self.input_labels[cloud_ind]
            sub_labels = labels[inds]
            glabels = np.unique(sub_labels)
            i_ind = []
            added_wl.extend(glabels)
            added_l.extend(sub_labels)

            for l in glabels:
                l_ind = np.where(sub_labels == l)[0]
                i_ind.append(np.random.choice(l_ind, 1))
            i_ind = np.array(i_ind).squeeze(-1)

            self.seed_pts[cloud_ind] = np.append(self.seed_pts[cloud_ind], [center_point], axis=0)
            self.seed_mask[cloud_ind].append(i_ind)
            self.seed_ind[cloud_ind].append(inds)
            self.seed_probs[cloud_ind].append(np.zeros((len(inds), self.num_classes), dtype=np.float32))
            self.seed_pmasks[cloud_ind].append(np.zeros(len(inds), dtype=np.float32))
            self.seed_probs_ori[cloud_ind].append(np.zeros((len(inds), self.num_classes), dtype=np.float32))

        # reset label weight
        count_wl = np.bincount(added_wl)
        for n in range(len(count_wl)):
            self.num_per_class[n] += count_wl[n]
        if self.config.segloss_balance == 'class':
            self.class_weight()