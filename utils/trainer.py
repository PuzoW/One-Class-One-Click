#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import numpy as np
from numpy.linalg import norm
from os import makedirs, remove
from os.path import exists, join
import time
from datetime import datetime

# PLY reader
from utils.ply import write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import Config


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelTrainer:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############

        # Epoch index
        self.epoch = 0
        self.step = 0

        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k]
        # other_params2 = [v for k, v in net_ema.named_parameters() if 'offset' not in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        # self.optimizer = torch.optim.SGD([{'params': other_params},
        #                                   {'params': other_params2},
        #                                   {'params': deform_params, 'lr': deform_lr}],
        #                                  lr=config.learning_rate,
        #                                  momentum=config.momentum,
        #                                  weight_decay=config.weight_decay)
        self.optimizer = torch.optim.Adam([{'params': other_params}],
                                         lr=config.learning_rate,
                                         weight_decay=config.weight_decay)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)


        ##########################
        # Load previous checkpoint
        ##########################
        if (chkp_path is not None):
            print("previous model: {}".format(chkp_path))
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'], strict=False)
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = config.learning_rate
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                time_string = datetime.now().strftime('/Log_%Y-%m-%d-%H-%M-%S')
                config.saving_path = 'results/' + config.dataset + time_string
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, training_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """

        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps l_loss train_accuracy time\n')

            with open(join(config.saving_path, 'num_subclouds.txt'), "w") as file:
                file.write('epochs subclouds weaklabels\n')

            with open(join(config.saving_path, 'loss_epoch.txt'), "w") as file:
                file.write('epochs c_epochs steps mean_segloss mean_glloss mean_maxloss mean_plloss mean_acc ema_acc pl_acc_ori pl_acc lr\n')

            
            if config.previous_training_path =='':
                s_num = 0
                for s in training_loader.dataset.seed_pts:
                    s_num += len(s)
                with open(join(config.saving_path, 'num_subclouds.txt'), "a") as file:
                    message = '{:d} {:d} {:d}\n'
                    file.write(message.format(self.epoch, s_num, int(np.sum(training_loader.dataset.num_per_class))))

            # Killing file (simply delete this file when you want to stop the training)
            PID_file = join(config.saving_path, 'running_PID.txt')
            if not exists(PID_file):
                with open(PID_file, "w") as file:
                    file.write('Launched with PyCharm')
            
            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
            
            subcloud_directory = join(config.saving_path, 'subclouds')
            if not exists(subcloud_directory):
                makedirs(subcloud_directory)

            epoch_subcloud = join(subcloud_directory, str(self.epoch))
            if not exists(epoch_subcloud):
                makedirs(epoch_subcloud)

        else:
            checkpoint_directory = None
            PID_file = None
            subcloud_directory = None 
            epoch_subcloud = None

        
        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        global_step = self.epoch*config.epoch_steps
        softmax = torch.nn.Softmax(1)
        sigmoid = torch.nn.Sigmoid()

        min_loss = 1e4
        ema_acc_epoch = 0.0
        n_epoch = 0
        itr_epoch = 0
        current_lr=config.learning_rate

        # initialize best model
        # best_params = []
        # for p_ind, param in enumerate(net.parameters()):
        #     best_params += [param.data.clone()]

        #  initialize probs of the training set for TOD calculation
        net.eval()
        self.update_pred_info(net, training_loader, config, epoch_subcloud, init=True)
        if config.previous_training_path:
            training_loader.dataset.add_labels()

            s_num = 0
            for s in training_loader.dataset.seed_pts:
                s_num += len(s)
            with open(join(config.saving_path, 'num_subclouds.txt'), "a") as file:
                message = '{:d} {:d} {:d}\n'
                file.write(message.format(self.epoch, s_num, int(np.sum(training_loader.dataset.num_per_class))))
                        
        net.train()

        # Start training loop
        for itr in range(config.al_itr):
            training_loader.dataset.save_subclouds(epoch_subcloud)
            while True:
            # for epoch in range(config.max_epoch):

                ## initialize batch in active learning
                training_loader.dataset.epoch_batch()
                training_loader.sampler.N = len(training_loader.dataset.batches)

                pred_count = np.zeros(2, np.float32)
                pred_ori_count = np.zeros(2, np.float32)

                n_step = 0
                loss_epoch = 0.0
                gl_epoch = 0.0
                max_epoch = 0.0
                pl_epoch = 0.0
                acc_epoch = 0.0

                for batch in training_loader:

                    # Check kill signal (running_PID.txt deleted)
                    if config.saving and not exists(PID_file):
                        continue

                    ##################
                    # Processing batch
                    ##################

                    # New time
                    t = t[-1:]
                    t += [time.time()]

                    ## save batch information
                    # lengths = batch.lengths[0].numpy()
                    # points = batch.points[0].numpy()
                    # labels = batch.labels.numpy()
                    # in1 = batch.inter_inds1.numpy()
                    # in2 = batch.inter_inds2.numpy()
                    # i0 = 0
                    # for b_i, length in enumerate(lengths):
                    #     # Get prediction
                    #     b_points = points[i0:i0 + length]
                    #     b_labels = labels[i0:i0 + length]
                    #
                    #     np.savetxt(join(config.saving_path, 'batch'+str(batch_i)+'.txt'),
                    #                np.hstack((b_points, b_labels[:, None])),
                    #                fmt=['%.2f', '%.2f', '%.2f', '%d'])
                    #
                    #     batch_i+=1
                    #     i0 += length

                    if 'cuda' in self.device.type:
                        batch.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # # Forward pass
                    global_output, outputs = net(batch, config)
                    probs = softmax(outputs)
                    global_preds = sigmoid(global_output)
                    max_probs = net.globalmaxpooling(probs, batch, 0)

                    l_loss = net.masked_loss(outputs, batch.labels, batch.masks, self.device)
                    # l_loss = net.loss(outputs, batch.labels, self.device)
                    loss_epoch += l_loss.detach().cpu().numpy()

                    g_loss = net.multilabel_loss(global_preds, batch.glabels)
                    gl_epoch += g_loss.detach().cpu().numpy()
                    # g_loss = torch.tensor(0.0, dtype=torch.float32)

                    m_loss = net.multilabel_loss(max_probs, batch.glabels)
                    max_epoch += m_loss.detach().cpu().numpy()
                    # m_loss = torch.tensor(0.0, dtype=torch.float32)

                    if itr>0 or config.previous_training_path:
                        pl_loss = net.masked_loss(outputs, batch.plabels, batch.pmasks, self.device)
                        pl_epoch += pl_loss.detach().cpu().numpy()
                    else:
                        pl_loss = torch.tensor(0.0, dtype=torch.float32)
                    # update ema prediction for pseudo-label generation
                    training_loader.dataset.update_subcloud_pl(probs, batch, pred_count, pred_ori_count)

                    loss = l_loss+g_loss+m_loss+pl_loss
                    acc = net.accuracy_mask(probs, batch.labels, batch.masks)
                    acc_epoch += acc

                    

                    t += [time.time()]

                    # Backward + optimize
                    loss.backward()

                    if config.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                    self.optimizer.step()
                    torch.cuda.synchronize(self.device)

                    t += [time.time()]

                    # Average timing
                    if self.step < 2:
                        mean_dt = np.array(t[1:]) - np.array(t[:-1])
                    else:
                        mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Console display (only one per second)
                    if (t[-1] - last_display) > 1.0:
                        last_display = t[-1]
                        message = 'e{:03d}-i{:04d} => L_seg={:.3f} L_gl={:.3f} L_max={:.3f} L_pl={:.3f} ' \
                                  'acc={:3.0f}% / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                        print(message.format(self.epoch, self.step,
                                             l_loss.item(),
                                             g_loss.item(),
                                             m_loss.item(),
                                             pl_loss.item(),
                                             100 * acc,
                                             1000 * mean_dt[0],
                                             1000 * mean_dt[1],
                                             1000 * mean_dt[2]))

                    # Log file
                    if config.saving:
                        with open(join(config.saving_path, 'training.txt'), "a") as file:
                            message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                            file.write(message.format(self.epoch,
                                                      self.step,
                                                      l_loss,
                                                      g_loss,
                                                      m_loss,
                                                      pl_loss,
                                                      acc,
                                                      t[-1] - t0))

                    self.step += 1
                    n_step += 1
                    global_step += 1


                ##############
                # End of epoch
                ##############

                # Check kill signal (running_PID.txt deleted)
                if config.saving and not exists(PID_file):
                    break

                # Update epoch
                self.epoch += 1
                itr_epoch += 1

                pl_ori_acc = pred_ori_count[0]/pred_ori_count[1]*100
                pl_acc = pred_count[0]/pred_count[1]*100
                print('pl_oa: {:.2f}% {:.2f}%'.format(pl_ori_acc, pl_acc))
                # pl_ori_acc = 0.0
                # pl_acc = 0.0

                # Update learning rate
                if itr_epoch in config.lr_decays:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= config.lr_decays[itr_epoch]

                # Saving
                if config.saving:
                    # Get current state dict
                    save_dict = {'epoch': self.epoch,
                                 'model_state_dict': net.state_dict(),
                                 'optimizer_state_dict': self.optimizer.state_dict(),
                                 'saving_path': config.saving_path}

                    # Save current state of the network (for restoring purposes)
                    checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                    torch.save(save_dict, checkpoint_path)

                    # Save checkpoints occasionally
                    loss_epoch /= n_step
                    gl_epoch /= n_step
                    max_epoch /= n_step
                    pl_epoch /= n_step
                    acc_epoch /= n_step

                    #ema training accuracy
                    if itr_epoch<=10:
                        ema_acc_epoch = acc_epoch
                    else:
                        ema_acc_epoch = 0.9*ema_acc_epoch + 0.1*acc_epoch

                    # accumulative epochs of convergence
                    if loss_epoch < min_loss:
                        # update best model
                        # for p_ind, param in enumerate(net.parameters()):
                        #     best_params[p_ind] = param.data.clone()

                        min_loss = loss_epoch
                        n_epoch = 1
                    else:
                        n_epoch += 1

                    with open(join(config.saving_path, 'loss_epoch.txt'), "a") as file:
                        message = '{:d} {:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f} {:.2f} {:.2f} {:.2f} {:.2f} {:.5f}\n'
                        file.write(message.format(self.epoch, n_epoch, n_step, loss_epoch, gl_epoch, max_epoch, pl_epoch, acc_epoch*100, ema_acc_epoch*100, pl_ori_acc, pl_acc, current_lr/config.learning_rate))
                    if itr_epoch in config.lr_decays:
                        current_lr *= config.lr_decays[itr_epoch]

                    if ema_acc_epoch > config.acc_thr:
                    # if self.epoch % config.checkpoint_gap == 0:

                        # Load best model
                        # for p_ind, param in enumerate(net.parameters()):
                        #     param.data = best_params[p_ind].clone()

                        itr_epoch = 0
                        n_epoch = 0
                        min_loss = 1e4
                        ema_acc_epoch = 0.0

                        checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(save_dict['epoch']))
                        torch.save(save_dict, checkpoint_path)

                        if itr + 1 < config.al_itr:
                        # if True:
                        
                            if exists(subcloud_directory):
                                epoch_subcloud = join(subcloud_directory, str(self.epoch))
                                if not exists(epoch_subcloud):
                                    makedirs(epoch_subcloud)
                        
                            # iteratively add sub-clouds
                            net.eval()
                            self.update_pred_info(net, training_loader, config, epoch_subcloud)
                            training_loader.dataset.add_labels()
                            # training_loader.dataset.random_add_labels()
                        
                            s_num = 0
                            for s in training_loader.dataset.seed_pts:
                                s_num += len(s)
                            with open(join(config.saving_path, 'num_subclouds.txt'), "a") as file:
                                message = '{:d} {:d} {:d}\n'
                                file.write(message.format(self.epoch, s_num, int(np.sum(training_loader.dataset.num_per_class))))
                        
                            # reset lr
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = config.learning_rate
                            current_lr=config.learning_rate
                        
                            torch.cuda.empty_cache()
                            net.train()

                        break

                # if self.epoch % 10 == 0:
                # net.eval()
                # self.validation(net, val_loader, config)
                torch.cuda.empty_cache()
                net.train()


        if exists(PID_file):
            remove(PID_file)

        print('Finished Training')
        return

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_pred_info(self, net, data_loader, config, epoch_subcloud, init=False):

        print('Infer predictions')

        d = data_loader.dataset
        softmax = torch.nn.Softmax(1)
        d.config.test_mode = True

        Temp_N = data_loader.sampler.N
        data_loader.sampler.N = len(d.val_center_pts)

        pts_prob = [np.zeros((l.shape[0], d.num_classes),dtype=np.float32) for l in d.input_labels]
        pts_times = [np.zeros(l.shape[0], dtype=np.float32) for l in d.input_labels]

        last_display = time.time()
        t= [last_display]

        for i, batch in enumerate(data_loader):

            t = t[-1:]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            _, outputs = net(batch, config)
            probs = softmax(outputs)
            lengths = batch.lengths[0].cpu().numpy()
            cloud_inds = batch.cloud_inds.numpy()
            in_inds = batch.input_inds.cpu().numpy()

            i0 = 0
            for b_i, length in enumerate(lengths):
                # Get prediction
                b_prob = probs[i0:i0 + length].cpu().detach().numpy()
                inds = in_inds[i0:i0 + length]
                c_i = cloud_inds[b_i]

                pts_prob[c_i][inds] += b_prob
                pts_times[c_i][inds] += 1
                i0 += length

            t += [time.time()]
            if (t[-1] - last_display) > 10.0:
                last_display = t[-1]
                message = 'i{:04d} => {:.1f}%'
                print(message.format(i, 100 * i / len(d.val_center_pts)))

        for i, pot in enumerate(d.potentials):
            pts_prob[i] /= pts_times[i][:, None]
            pts_prob[i].astype(np.float32)
            
            d.input_tod[i] = np.sum(np.square(pts_prob[i] - d.input_probs[i]), axis=1)
            pot_prob = pts_prob[i][d.pot_ind[i]]
            d.pot_tod[i] = d.input_tod[i][d.pot_ind[i]]
            
            d.input_probs[i] = pts_prob[i]
            d.pot_prob[i] = pot_prob
            d.pot_pred[i] = np.argmax(pot_prob, axis=1).astype(np.int32)

            # local smooth
            pred_w = np.sum(pot_prob[d.pot_local_nei[i]] * pot_prob[d.pot_local_nei[i][:, :1]], axis=-1)
            pred_norm = norm(pot_prob[d.pot_local_nei[i]], axis=-1) * norm(pot_prob[d.pot_local_nei[i][:, :1]], axis=-1)
            pred_w /= pred_norm
            w = d.pot_local_w[i] * pred_w
            w = w / np.sum(w, axis=1, keepdims=True)
            d.pot_tod[i] = np.sum(d.pot_tod[i][d.pot_local_nei[i]] * w, axis=1)

        data_loader.dataset.config.test_mode = False
        data_loader.sampler.N = Temp_N

        data_loader.dataset.save_info(epoch_subcloud)

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------
    def validation(self, net, val_loader, config: Config):

        if config.dataset_task == 'classification':
            self.object_classification_validation(net, val_loader, config)
        elif config.dataset_task == 'segmentation':
            self.object_segmentation_validation(net, val_loader, config)
        elif config.dataset_task == 'cloud_segmentation':
            self.cloud_segmentation_validation(net, val_loader, config)
        elif config.dataset_task == 'cloud_segmentation_noproj':
            self.cloud_segmentation_validation_noproj(net, val_loader, config)
        elif config.dataset_task == 'slam_segmentation':
            self.slam_segmentation_validation(net, val_loader, config)
        else:
            raise ValueError('No validation method implemented for this network type')

    def object_classification_validation(self, net, val_loader, config):
        """
        Perform a round of validation and show/save results
        :param net: network object
        :param val_loader: data loader for validation set
        :param config: configuration object
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Number of classes predicted by the model
        nc_model = config.num_classes
        softmax = torch.nn.Softmax(1)

        # Initialize global prediction over all models
        if not hasattr(self, 'val_probs'):
            self.val_probs = np.zeros((val_loader.dataset.num_models, nc_model))

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        obj_inds = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start validation loop
        for batch in val_loader:

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            probs += [softmax(outputs).cpu().detach().numpy()]
            targets += [batch.labels.cpu().numpy()]
            obj_inds += [batch.model_inds.cpu().numpy()]
            torch.cuda.synchronize(self.device)

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * len(obj_inds) / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Stack all validation predictions
        probs = np.vstack(probs)
        targets = np.hstack(targets)
        obj_inds = np.hstack(obj_inds)

        ###################
        # Voting validation
        ###################

        self.val_probs[obj_inds] = val_smooth * self.val_probs[obj_inds] + (1-val_smooth) * probs

        ############
        # Confusions
        ############

        validation_labels = np.array(val_loader.dataset.label_values)

        # Compute classification results
        C1 = fast_confusion(targets,
                            np.argmax(probs, axis=1),
                            validation_labels)

        # Compute votes confusion
        C2 = fast_confusion(val_loader.dataset.input_labels,
                            np.argmax(self.val_probs, axis=1),
                            validation_labels)


        # Saving (optionnal)
        if config.saving:
            print("Save confusions")
            conf_list = [C1, C2]
            file_list = ['val_confs.txt', 'vote_confs.txt']
            for conf, conf_file in zip(conf_list, file_list):
                test_file = join(config.saving_path, conf_file)
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')
                else:
                    with open(test_file, "w") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')

        val_ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
        vote_ACC = 100 * np.sum(np.diag(C2)) / (np.sum(C2) + 1e-6)
        print('Accuracies : val = {:.1f}% / vote = {:.1f}%'.format(val_ACC, vote_ACC))

        return C1

    def cloud_segmentation_validation(self, net, val_loader, config, debug=False):
        """
        Validation method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Do not validate if dataset has no validation cloud
        # if val_loader.dataset.validation_split not in val_loader.dataset.all_splits:
        #     return

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        #print(nc_tot)
        #print(nc_model)

        # Initiate global prediction over validation clouds
        if not hasattr(self, 'validation_probs'):
            self.validation_probs = [np.zeros((l.shape[0], nc_model))
                                     for l in val_loader.dataset.input_labels]
            self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in val_loader.dataset.label_values:
                if label_value not in val_loader.dataset.ignored_labels:
                    self.val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                      for labels in val_loader.dataset.validation_labels])
                    i += 1

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            _, outputs = net(batch, config)

            # Get probs and labels
            stacked_probs = softmax(outputs).cpu().detach().numpy()
            labels = batch.labels.cpu().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            in_inds = batch.input_inds.cpu().numpy()
            cloud_inds = batch.cloud_inds.cpu().numpy()
            torch.cuda.synchronize(self.device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                target = labels[i0:i0 + length]
                probs = stacked_probs[i0:i0 + length]
                inds = in_inds[i0:i0 + length]
                c_i = cloud_inds[b_i]

                # Update current probs in whole cloud
                self.validation_probs[c_i][inds] = val_smooth * self.validation_probs[c_i][inds] \
                                                   + (1 - val_smooth) * probs

                # Stack all prediction for this epoch
                predictions.append(probs)
                targets.append(target)
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)

        t3 = time.time()

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)


        t4 = time.time()

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            # Name of saving file
            test_file = join(config.saving_path, 'val_IoUs.txt')

            # Line to write:
            line = ''
            for IoU in IoUs:
                line += '{:.3f} '.format(IoU)
            line += '{:.3f} '.format(np.mean(IoUs))
            line = line + '\n'

            # Write in file
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)

            # Save potentials
            pot_path = join(config.saving_path, 'potentials')
            if not exists(pot_path):
                makedirs(pot_path)
            files = val_loader.dataset.files
            for i, file_path in enumerate(files):
                pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
                cloud_name = file_path.split('/')[-1]
                pot_name = join(pot_path, cloud_name)
                pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
                write_ply(pot_name,
                          [pot_points.astype(np.float32), pots],
                          ['x', 'y', 'z', 'pots'])

        t6 = time.time()

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} mean IoU = {:.1f}%'.format(config.dataset, mIoU))

        # Save predicted cloud occasionally
        # if config.saving and (self.epoch + 1) % config.checkpoint_gap == 0:
        #     val_path = join(config.saving_path, 'val_preds_{:d}'.format(self.epoch + 1))
        #     if not exists(val_path):
        #         makedirs(val_path)
        #     files = val_loader.dataset.files
        #     for i, file_path in enumerate(files):
        #
        #         # Get points
        #         points = val_loader.dataset.load_evaluation_points(file_path)
        #
        #         # Get probs on our own ply points
        #         sub_probs = self.validation_probs[i]
        #
        #         # Insert false columns for ignored labels
        #         for l_ind, label_value in enumerate(val_loader.dataset.label_values):
        #             if label_value in val_loader.dataset.ignored_labels:
        #                 sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)
        #
        #         # Get the predicted labels
        #         sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]
        #
        #         # Reproject preds on the evaluations points
        #         preds = (sub_preds[val_loader.dataset.test_proj[i]]).astype(np.int32)
        #
        #         # Path of saved validation file
        #         cloud_name = file_path.split('/')[-1][:-4]
        #         val_name = join(val_path, cloud_name)
        #
        #         # Save file
        #         labels = val_loader.dataset.validation_labels[i].astype(np.int32)
        #         write_ply(val_name,
        #                   [points, preds, labels],
        #                   ['x', 'y', 'z', 'preds', 'class'])

        # Display timings
        t7 = time.time()
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('Confs bis . {:.1f}s'.format(t4 - t3))
            print('IoU ....... {:.1f}s'.format(t5 - t4))
            print('Save1 ..... {:.1f}s'.format(t6 - t5))
            print('Save2 ..... {:.1f}s'.format(t7 - t6))
            print('\n************************\n')

        return

    def slam_segmentation_validation(self, net, val_loader, config, debug=True):
        """
        Validation method for slam segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Do not validate if dataset has no validation cloud
        if val_loader is None:
            return

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Create folder for validation predictions
        if not exists (join(config.saving_path, 'val_preds')):
            makedirs(join(config.saving_path, 'val_preds'))

        # initiate the dataset validation containers
        val_loader.dataset.val_points = []
        val_loader.dataset.val_labels = []

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        inds = []
        val_i = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            stk_probs = softmax(outputs).cpu().detach().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            f_inds = batch.frame_inds.cpu().numpy()
            r_inds_list = batch.reproj_inds
            r_mask_list = batch.reproj_masks
            labels_list = batch.val_labels
            torch.cuda.synchronize(self.device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                probs = stk_probs[i0:i0 + length]
                proj_inds = r_inds_list[b_i]
                proj_mask = r_mask_list[b_i]
                frame_labels = labels_list[b_i]
                s_ind = f_inds[b_i, 0]
                f_ind = f_inds[b_i, 1]

                # Project predictions on the frame points
                proj_probs = probs[proj_inds]

                # Safe check if only one point:
                if proj_probs.ndim < 2:
                    proj_probs = np.expand_dims(proj_probs, 0)

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        proj_probs = np.insert(proj_probs, l_ind, 0, axis=1)

                # Predicted labels
                preds = val_loader.dataset.label_values[np.argmax(proj_probs, axis=1)]

                # Save predictions in a binary file
                filename = '{:s}_{:07d}.npy'.format(val_loader.dataset.sequences[s_ind], f_ind)
                filepath = join(config.saving_path, 'val_preds', filename)
                if exists(filepath):
                    frame_preds = np.load(filepath)
                else:
                    frame_preds = np.zeros(frame_labels.shape, dtype=np.uint8)
                frame_preds[proj_mask] = preds.astype(np.uint8)
                np.save(filepath, frame_preds)

                # Save some of the frame pots
                if f_ind % 20 == 0:
                    seq_path = join(val_loader.dataset.path, 'sequences', val_loader.dataset.sequences[s_ind])
                    velo_file = join(seq_path, 'velodyne', val_loader.dataset.frames[s_ind][f_ind] + '.bin')
                    frame_points = np.fromfile(velo_file, dtype=np.float32)
                    frame_points = frame_points.reshape((-1, 4))
                    write_ply(filepath[:-4] + '_pots.ply',
                              [frame_points[:, :3], frame_labels, frame_preds],
                              ['x', 'y', 'z', 'gt', 'pre'])

                # Update validation confusions
                frame_C = fast_confusion(frame_labels,
                                         frame_preds.astype(np.int32),
                                         val_loader.dataset.label_values)
                val_loader.dataset.val_confs[s_ind][f_ind, :, :] = frame_C

                # Stack all prediction for this epoch
                predictions += [preds]
                targets += [frame_labels[proj_mask]]
                inds += [f_inds[b_i, :]]
                val_i += 1
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (preds, truth) in enumerate(zip(predictions, targets)):

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)

        t3 = time.time()

        #######################################
        # Results on this subpart of validation
        #######################################

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Balance with real validation proportions
        C *= np.expand_dims(val_loader.dataset.class_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        #####################################
        # Results on the whole validation set
        #####################################

        t4 = time.time()

        # Sum all validation confusions
        C_tot = [np.sum(seq_C, axis=0) for seq_C in val_loader.dataset.val_confs if len(seq_C) > 0]
        C_tot = np.sum(np.stack(C_tot, axis=0), axis=0)

        if debug:
            s = '\n'
            for cc in C_tot:
                for c in cc:
                    s += '{:8.1f} '.format(c)
                s += '\n'
            print(s)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C_tot = np.delete(C_tot, l_ind, axis=0)
                C_tot = np.delete(C_tot, l_ind, axis=1)

        # Objects IoU
        val_IoUs = IoU_from_confusions(C_tot)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            IoU_list = [IoUs, val_IoUs]
            file_list = ['subpart_IoUs.txt', 'val_IoUs.txt']
            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = join(config.saving_path, IoU_file)

                # Line to write:
                line = ''
                for IoU in IoUs_to_save:
                    line += '{:.3f} '.format(IoU)
                line = line + '\n'

                # Write in file
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        text_file.write(line)
                else:
                    with open(test_file, "w") as text_file:
                        text_file.write(line)

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} : subpart mIoU = {:.1f} %'.format(config.dataset, mIoU))
        mIoU = 100 * np.mean(val_IoUs)
        print('{:s} :     val mIoU = {:.1f} %'.format(config.dataset, mIoU))

        t6 = time.time()

        # Display timings
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('IoU1 ...... {:.1f}s'.format(t4 - t3))
            print('IoU2 ...... {:.1f}s'.format(t5 - t4))
            print('Save ...... {:.1f}s'.format(t6 - t5))
            print('\n************************\n')

        return



































