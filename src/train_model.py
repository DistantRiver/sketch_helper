from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
import time
import copy
import torch.nn.functional as F
import numpy as np
import os
from tqdm import *
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

def calc_loss(img_feas, str_out, cls_out, labels, hyper_parameters):
    
    alpha = hyper_parameters['alpha']
    
    img_num = len(img_feas)
    batch_size = img_num // 4
    img_feas_gt = img_feas.clone().detach()
    for i in range(batch_size):
        img_feas_gt[(i*4):(i*4+3)] = img_feas[(i*4+1):(i*4+4)]
    term1 = (img_feas_gt - str_out).pow(2).mean(-1).mean()
    
    l_mean = nn.CrossEntropyLoss(reduction='mean')
    labels = labels.view(-1)
    term2 = l_mean(cls_out, labels)
    
    im_loss = alpha * term1 + term2

    return im_loss


def train_model(model, data_loaders, hyper_parameters, optimizer, scheduler=None, device="cpu", num_epochs=500, save_name='best'):
    since = time.time()
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = -1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for imgs, labels in tqdm(data_loaders[phase]):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    imgs = imgs.to(device)
                    labels = labels.to(device)


                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    img_feas, str_out, cls_out = model(imgs)

                    loss = calc_loss(img_feas, str_out, cls_out, labels, hyper_parameters)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                continue
                
            if phase == 'valid' and (best_loss < 0 or epoch_loss < best_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))
    
    save_folder = '../weights'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(best_model_wts, os.path.join(save_folder, save_name))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss_history

def train_model_select(model, model_select, select_num, data_loaders, hyper_parameters, optimizer, scheduler=None, device="cpu", num_epochs=500, save_name='best'):
    since = time.time()
    valid_sketch_acc_history = []
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for sketches, sketches_adj, videos, videos_adj, labels, sketches_scene, videos_scene in data_loaders[phase]:
                if torch.sum(sketches != sketches)>1 or torch.sum(videos != videos)>1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    sketches = sketches.to(device)
                    videos = videos.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)
                    
                    sketches_scene = sketches_scene.to(device)
                    videos_scene = videos_scene.to(device)


                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    videos, videos_adj = select_frame(model_select, select_num, sketches_scene, videos_scene, videos, videos_adj)

                    # Forward
                    view1_feature, view2_feature = model(sketches[:, 0, 0, :], videos[:, :, 0, :])

                    loss = calc_loss(view1_feature, view2_feature, hyper_parameters)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            if phase == 'train':
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                continue

            t_sketches, t_videos, t_labels = [], [], []
            with torch.no_grad():
                for sketches, sketches_adj, videos, videos_adj, labels, sketches_scene, videos_scene in data_loaders[phase]:

                    sketches = sketches.to(device)
                    videos = videos.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)
                    
                    sketches_scene = sketches_scene.to(device)
                    videos_scene = videos_scene.to(device)
                    
                    videos, videos_adj = select_frame(model_select, select_num, sketches_scene, videos_scene, videos, videos_adj)
                            
                    t_view1_feature, t_view2_feature = model(sketches[:, 0, 0, :], videos[:, :, 0, :])
                    t_sketches.append(t_view1_feature.cpu().numpy())
                    t_videos.append(t_view2_feature.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_sketches = np.concatenate(t_sketches)
            t_videos = np.concatenate(t_videos)
            t_labels = np.concatenate(t_labels).argmax(1)
            Sketch2Video_map = fx_calc_map_label(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_map_label(t_videos, t_sketches, t_labels)
            Sketch2Video = fx_calc_recall(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_recall(t_videos, t_sketches, t_labels)

            #print('{} Loss: {:.4f} Sketch2Video: {:.4f}  Video2Sketch: {:.4f}'.format(phase, epoch_loss, Sketch2Video, Video2Sketch))
            print('{} Loss: {:.4f} Sketch2Video: mAP = {:.4f} R1 = {:.4f} R5 = {:.4f} R10 = {:.4f}'.format(phase, epoch_loss, Sketch2Video_map, Sketch2Video[0], Sketch2Video[1], Sketch2Video[2]))

            Sketch2Video_map_10 = fx_calc_map_label(t_sketches, t_videos, t_labels, 10)
            # deep copy the model
            #Sketch2Video_mean = np.mean(Sketch2Video)
            if phase == 'valid' and Sketch2Video_map_10 > best_acc:
                best_acc = Sketch2Video_map_10
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                valid_sketch_acc_history.append(Sketch2Video_map)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best mAP_10: {:4f}'.format(best_acc))
    
    save_folder = '../weights'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(best_model_wts, os.path.join(save_folder, save_name))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, valid_sketch_acc_history, epoch_loss_history

def train_model_graph(model, data_loaders, hyper_parameters, optimizer, scheduler=None, device="cpu", num_epochs=500, save_name='best'):
    since = time.time()
    valid_sketch_acc_history = []
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for sketches, sketches_adj, videos, videos_adj, labels in data_loaders[phase]:
                if torch.sum(sketches != sketches)>1 or torch.sum(videos != videos)>1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    sketches = sketches.to(device)
                    sketches_adj = sketches_adj.to(device)
                    videos = videos.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    view1_feature, view2_feature = model(sketches, sketches_adj, videos, videos_adj)

                    loss = calc_loss(view1_feature, view2_feature, hyper_parameters)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            if phase == 'train':
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                continue

            t_sketches, t_videos, t_labels = [], [], []
            with torch.no_grad():
                for sketches, sketches_adj, videos, videos_adj, labels in data_loaders[phase]:

                    sketches = sketches.to(device)
                    sketches_adj = sketches_adj.to(device)
                    videos = videos.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)
                            
                    t_view1_feature, t_view2_feature = model(sketches, sketches_adj, videos, videos_adj)
                    t_sketches.append(t_view1_feature.cpu().numpy())
                    t_videos.append(t_view2_feature.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_sketches = np.concatenate(t_sketches)
            t_videos = np.concatenate(t_videos)
            t_labels = np.concatenate(t_labels).argmax(1)
            Sketch2Video_map = fx_calc_map_label(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_map_label(t_videos, t_sketches, t_labels)
            Sketch2Video = fx_calc_recall(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_recall(t_videos, t_sketches, t_labels)

            #print('{} Loss: {:.4f} Sketch2Video: {:.4f}  Video2Sketch: {:.4f}'.format(phase, epoch_loss, Sketch2Video, Video2Sketch))
            print('{} Loss: {:.4f} Sketch2Video: mAP = {:.4f} R1 = {:.4f} R5 = {:.4f} R10 = {:.4f}'.format(phase, epoch_loss, Sketch2Video_map, Sketch2Video[0], Sketch2Video[1], Sketch2Video[2]))

            Sketch2Video_map_10 = fx_calc_map_label(t_sketches, t_videos, t_labels, 10)
            # deep copy the model
            #Sketch2Video_mean = np.mean(Sketch2Video)
            if phase == 'valid' and Sketch2Video_map_10 > best_acc:
                best_acc = Sketch2Video_map_10
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                valid_sketch_acc_history.append(Sketch2Video_map)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best mAP_10: {:4f}'.format(best_acc))
    
    save_folder = '../weights'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(best_model_wts, os.path.join(save_folder, save_name))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, valid_sketch_acc_history, epoch_loss_history


def CrossModel_triplet_loss_hard_fuse(view1_features_rgb, view1_features_cls, view2_features_rgb, view2_features_cls, margin):
    loss = torch.tensor(0.0).cuda()
    loss.requires_grad = True
    
    num_cls = len(view1_features_cls)
    
    view1_tri_loss = []
    for index, view1_feature_cls in enumerate(view1_features_cls):
        d_all = (view1_feature_cls - view2_features_cls).pow(2).sum(-1)
        d_p = d_all[index]
        n_id = 0
        if index == 0:
            d_n = d_all[1:].min()
            n_id = d_all[1:].argmin().item()
        elif index == num_cls - 1:
            d_n = d_all[:index].min()
            n_id = d_all[:index].argmin().item()
        else:
            d_n1 = d_all[:index].min()
            d_n2 = d_all[(index + 1):].min()
            if d_n1 > d_n2:
                d_n = d_n2
                n_id = d_all[(index + 1):].argmin().item()
            else:
                d_n = d_n1
                n_id = d_all[:index].argmin().item()
        
        d_p_rgb = (view1_features_rgb[index] - view2_features_rgb[index]).pow(2).sum(-1)
        d_n_cls = (view1_features_rgb[n_id] - view2_features_rgb[n_id]).pow(2).sum(-1)
        
        view1_tri_loss.append(F.relu(margin + d_p - d_n + d_p_rgb - d_n_cls).unsqueeze(0))
    view1_tri_loss = torch.cat(view1_tri_loss)
    
    loss = view1_tri_loss.mean()
    
    return loss


def calc_loss_fuse(view1_feature_rgb, view1_feature_cls, view2_feature_rgb, view2_feature_cls, hyper_parameters):
    
    cm_tri = hyper_parameters['cm_tri']
    margin = hyper_parameters['margin']

    term1 = CrossModel_triplet_loss_hard_fuse(view1_feature_rgb, view1_feature_cls, view2_feature_rgb, view2_feature_cls, margin)
    
    im_loss = cm_tri * term1

    return im_loss


def train_model_graph_fuse(model, data_loaders, hyper_parameters, optimizer, scheduler=None, device="cpu", num_epochs=500, save_name='best'):
    since = time.time()
    valid_sketch_acc_history = []
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for sketches_rgb, sketches_cls, sketches_adj, videos_rgb, videos_cls, videos_adj, labels in data_loaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    sketches_rgb = sketches_rgb.to(device)
                    sketches_cls = sketches_cls.to(device)
                    sketches_adj = sketches_adj.to(device)
                    videos_rgb = videos_rgb.to(device)
                    videos_cls = videos_cls.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    view1_feature, view2_feature = model(sketches_rgb, sketches_cls, sketches_adj, videos_rgb, videos_cls, videos_adj)

                    loss = calc_loss(view1_feature, view2_feature, hyper_parameters)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            if phase == 'train':
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                continue

            t_sketches, t_videos, t_labels = [], [], []
            with torch.no_grad():
                for sketches_rgb, sketches_cls, sketches_adj, videos_rgb, videos_cls, videos_adj, labels in data_loaders[phase]:

                    sketches_rgb = sketches_rgb.to(device)
                    sketches_cls = sketches_cls.to(device)
                    sketches_adj = sketches_adj.to(device)
                    videos_rgb = videos_rgb.to(device)
                    videos_cls = videos_cls.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)
                            
                    t_view1_feature, t_view2_feature = model(sketches_rgb, sketches_cls, sketches_adj, videos_rgb, videos_cls, videos_adj)
                    
                    t_sketches.append(t_view1_feature.cpu().numpy())
                    t_videos.append(t_view2_feature.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_sketches = np.concatenate(t_sketches)
            t_videos = np.concatenate(t_videos)
            t_labels = np.concatenate(t_labels).argmax(1)
            Sketch2Video_map = fx_calc_map_label(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_map_label(t_videos, t_sketches, t_labels)
            Sketch2Video = fx_calc_recall(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_recall(t_videos, t_sketches, t_labels)

            #print('{} Loss: {:.4f} Sketch2Video: {:.4f}  Video2Sketch: {:.4f}'.format(phase, epoch_loss, Sketch2Video, Video2Sketch))
            print('{} Loss: {:.4f} Sketch2Video: mAP = {:.4f} R1 = {:.4f} R5 = {:.4f} R10 = {:.4f}'.format(phase, epoch_loss, Sketch2Video_map, Sketch2Video[0], Sketch2Video[1], Sketch2Video[2]))

            Sketch2Video_map_10 = fx_calc_map_label(t_sketches, t_videos, t_labels, 10)
            # deep copy the model
            #Sketch2Video_mean = np.mean(Sketch2Video)
            if phase == 'valid' and Sketch2Video_map_10 > best_acc:
                best_acc = Sketch2Video_map_10
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                valid_sketch_acc_history.append(Sketch2Video_map)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best mAP_10: {:4f}'.format(best_acc))
    
    save_folder = '../weights'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(best_model_wts, os.path.join(save_folder, save_name))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, valid_sketch_acc_history, epoch_loss_history

def select_frame_fuse(model_select, select_num, sketches_scene, videos_scene, videos_rgb, videos_cls, videos_adj):
    embed1, embed2 = model_select(sketches_scene, videos_scene)
                    
    embed1_r = embed1.repeat(1, embed2.shape[-2], 1)
    embed2_r = embed2.repeat(1, embed1.shape[-2], 1)
    dis = -(embed1_r - embed2_r).pow(2).sum(-1)
    _, idx = dis.topk(select_num, -1)
    idx, _ = idx.sort(-1)
                    
    new_videos_rgb = []
    new_videos_cls = []
    new_videos_adj = []
    for i in range(len(videos_rgb)):
        new_videos_rgb.append(videos_rgb[i][idx[i]].unsqueeze(0))
        new_videos_cls.append(videos_cls[i][idx[i]].unsqueeze(0))
        new_videos_adj.append(videos_adj[i][idx[i]].unsqueeze(0))
    videos_rgb = torch.cat(new_videos_rgb, 0)
    videos_cls = torch.cat(new_videos_cls, 0)
    videos_adj = torch.cat(new_videos_adj, 0)
    
    return videos_rgb, videos_cls, videos_adj

def train_model_graph_fuse_select(model, model_select, data_loaders, hyper_parameters, optimizer, scheduler=None, device="cpu", num_epochs=500, save_name='best'):
    since = time.time()
    valid_sketch_acc_history = []
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for sketches_rgb, sketches_cls, sketches_adj, videos_rgb, videos_cls, videos_adj, labels in data_loaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    sketches_rgb = sketches_rgb.to(device)
                    sketches_cls = sketches_cls.to(device)
                    sketches_adj = sketches_adj.to(device)
                    videos_rgb = videos_rgb.to(device)
                    videos_cls = videos_cls.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    videos_rgb, videos_cls, videos_adj = select_frame_fuse(model_select, select_num, sketches_rgb[:, :, 0, :], videos_rgb[:, :, 0, :], videos_rgb, videos_cls, videos_adj)
                    
                    # Forward
                    view1_feature, view2_feature = model(sketches_rgb, sketches_cls, sketches_adj, videos_rgb, videos_cls, videos_adj)

                    loss = calc_loss(view1_feature, view2_feature, hyper_parameters)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            if phase == 'train':
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                continue

            t_sketches, t_videos, t_labels = [], [], []
            with torch.no_grad():
                for sketches_rgb, sketches_cls, sketches_adj, videos_rgb, videos_cls, videos_adj, labels in data_loaders[phase]:

                    sketches_rgb = sketches_rgb.to(device)
                    sketches_cls = sketches_cls.to(device)
                    sketches_adj = sketches_adj.to(device)
                    videos_rgb = videos_rgb.to(device)
                    videos_cls = videos_cls.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)
                    
                    videos_rgb, videos_cls, videos_adj = select_frame_fuse(model_select, select_num, sketches_rgb[:, :, 0, :], videos_rgb[:, :, 0, :], videos_rgb, videos_cls, videos_adj)
                            
                    t_view1_feature, t_view2_feature = model(sketches_rgb, sketches_cls, sketches_adj, videos_rgb, videos_cls, videos_adj)
                    
                    t_sketches.append(t_view1_feature.cpu().numpy())
                    t_videos.append(t_view2_feature.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_sketches = np.concatenate(t_sketches)
            t_videos = np.concatenate(t_videos)
            t_labels = np.concatenate(t_labels).argmax(1)
            Sketch2Video_map = fx_calc_map_label(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_map_label(t_videos, t_sketches, t_labels)
            Sketch2Video = fx_calc_recall(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_recall(t_videos, t_sketches, t_labels)

            #print('{} Loss: {:.4f} Sketch2Video: {:.4f}  Video2Sketch: {:.4f}'.format(phase, epoch_loss, Sketch2Video, Video2Sketch))
            print('{} Loss: {:.4f} Sketch2Video: mAP = {:.4f} R1 = {:.4f} R5 = {:.4f} R10 = {:.4f}'.format(phase, epoch_loss, Sketch2Video_map, Sketch2Video[0], Sketch2Video[1], Sketch2Video[2]))

            Sketch2Video_map_10 = fx_calc_map_label(t_sketches, t_videos, t_labels, 10)
            # deep copy the model
            #Sketch2Video_mean = np.mean(Sketch2Video)
            if phase == 'valid' and Sketch2Video_map_10 > best_acc:
                best_acc = Sketch2Video_map_10
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                valid_sketch_acc_history.append(Sketch2Video_map)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best mAP_10: {:4f}'.format(best_acc))
    
    save_folder = '../weights'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(best_model_wts, os.path.join(save_folder, save_name))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, valid_sketch_acc_history, epoch_loss_history


def CrossModel_triplet_loss_hard_sp(view1_features, view2_features, margin):
    loss = torch.tensor(0.0).cuda()
    loss.requires_grad = True
    
    num_cls = len(view1_features)
    
    view1_tri_loss = []
    for index, view1_feature_sp in enumerate(view1_features):
        for view1_feature in view1_feature_sp:
            d_all = (view1_feature - view2_features).pow(2).sum(-1)
            d_p = d_all[index]
            if index == 0:
                d_n = d_all[1:].min()
            elif index == num_cls - 1:
                d_n = d_all[:index].min()
            else:
                d_n1 = d_all[:index].min()
                d_n2 = d_all[(index + 1):].min()
                if d_n1 > d_n2:
                    d_n = d_n2
                else:
                    d_n = d_n1
            view1_tri_loss.append(F.relu(margin + d_p - d_n).unsqueeze(0))
    view1_tri_loss = torch.cat(view1_tri_loss)
    
    loss = view1_tri_loss.mean()
    
    return loss


def calc_loss_sp(view1_feature, view2_feature, hyper_parameters):
    
    cm_tri = hyper_parameters['cm_tri']
    margin = hyper_parameters['margin']

    term1 = CrossModel_triplet_loss_hard_sp(view1_feature, view2_feature, margin)
    
    im_loss = cm_tri * term1

    return im_loss

def train_model_graph_sp(model, data_loaders, hyper_parameters, optimizer, scheduler=None, device="cpu", num_epochs=500, save_name='best'):
    since = time.time()
    valid_sketch_acc_history = []
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for sketches, sketches_adj, videos, videos_adj, labels in data_loaders[phase]:
                if torch.sum(sketches != sketches)>1 or torch.sum(videos != videos)>1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    sketches = sketches.to(device)
                    sketches_adj = sketches_adj.to(device)
                    videos = videos.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    view1_feature, view2_feature = model(sketches, sketches_adj, videos, videos_adj)

                    loss = calc_loss_sp(view1_feature, view2_feature, hyper_parameters)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            if phase == 'train':
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                continue

            t_sketches, t_videos, t_labels = [], [], []
            with torch.no_grad():
                for sketches, sketches_adj, videos, videos_adj, labels in data_loaders[phase]:

                    sketches = sketches.to(device)
                    sketches_adj = sketches_adj.to(device)
                    videos = videos.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)
                            
                    t_view1_feature, t_view2_feature = model(sketches, sketches_adj, videos, videos_adj)
                    t_sketches.append(t_view1_feature.cpu().numpy())
                    t_videos.append(t_view2_feature.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_sketches = np.concatenate(t_sketches)
            t_videos = np.concatenate(t_videos)
            t_labels = np.concatenate(t_labels).argmax(1)
            Sketch2Video_map = fx_calc_map_label_sp(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_map_label(t_videos, t_sketches, t_labels)
            Sketch2Video = fx_calc_recall_sp(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_recall(t_videos, t_sketches, t_labels)

            #print('{} Loss: {:.4f} Sketch2Video: {:.4f}  Video2Sketch: {:.4f}'.format(phase, epoch_loss, Sketch2Video, Video2Sketch))
            print('{} Loss: {:.4f} Sketch2Video: mAP = {:.4f} R1 = {:.4f} R5 = {:.4f} R10 = {:.4f}'.format(phase, epoch_loss, Sketch2Video_map, Sketch2Video[0], Sketch2Video[1], Sketch2Video[2]))

            Sketch2Video_map_10 = fx_calc_map_label_sp(t_sketches, t_videos, t_labels, 10)
            # deep copy the model
            #Sketch2Video_mean = np.mean(Sketch2Video)
            if phase == 'valid' and Sketch2Video_map_10 > best_acc:
                best_acc = Sketch2Video_map_10
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                valid_sketch_acc_history.append(Sketch2Video_map)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best mAP_10: {:4f}'.format(best_acc))
    
    save_folder = '../weights'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(best_model_wts, os.path.join(save_folder, save_name))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, valid_sketch_acc_history, epoch_loss_history


def select_frame(model_select, select_num, sketches_scene, videos_scene, videos, videos_adj):
    embed1, embed2 = model_select(sketches_scene, videos_scene)
                    
    embed1_r = embed1.repeat(1, embed2.shape[-2], 1)
    embed2_r = embed2.repeat(1, embed1.shape[-2], 1)
    dis = -(embed1_r - embed2_r).pow(2).sum(-1)
    _, idx = dis.topk(select_num, -1)
    idx, _ = idx.sort(-1)
                    
    new_videos = []
    new_videos_adj = []
    for i in range(len(videos)):
        new_videos.append(videos[i][idx[i]].unsqueeze(0))
        new_videos_adj.append(videos_adj[i][idx[i]].unsqueeze(0))
    videos = torch.cat(new_videos, 0)
    videos_adj = torch.cat(new_videos_adj, 0)
    
    return videos, videos_adj

def train_model_graph_select(model, model_select, select_num, data_loaders, hyper_parameters, optimizer, scheduler=None, device="cpu", num_epochs=500, save_name='best'):
    since = time.time()
    valid_sketch_acc_history = []
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            # Iterate over data.
            for sketches, sketches_adj, videos, videos_adj, labels, sketches_scene, videos_scene in data_loaders[phase]:
                if torch.sum(sketches != sketches)>1 or torch.sum(videos != videos)>1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    sketches = sketches.to(device)
                    sketches_adj = sketches_adj.to(device)
                    videos = videos.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)
                    sketches_scene = sketches_scene.to(device)
                    videos_scene = videos_scene.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    videos, videos_adj = select_frame(model_select, select_num, sketches_scene, videos_scene, videos, videos_adj)

                    # Forward
                    view1_feature, view2_feature = model(sketches, sketches_adj, videos, videos_adj)

                    loss = calc_loss(view1_feature, view2_feature, hyper_parameters)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler:
                            scheduler.step()

                # statistics
                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            if phase == 'train':
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                continue

            t_sketches, t_videos, t_labels = [], [], []
            with torch.no_grad():
                for sketches, sketches_adj, videos, videos_adj, labels, sketches_scene, videos_scene in data_loaders[phase]:

                    sketches = sketches.to(device)
                    sketches_adj = sketches_adj.to(device)
                    videos = videos.to(device)
                    videos_adj = videos_adj.to(device)
                    labels = labels.to(device)
                    sketches_scene = sketches_scene.to(device)
                    videos_scene = videos_scene.to(device)
                    
                    videos, videos_adj = select_frame(model_select, select_num, sketches_scene, videos_scene, videos, videos_adj)
                    
                    t_view1_feature, t_view2_feature = model(sketches, sketches_adj, videos, videos_adj)
                    t_sketches.append(t_view1_feature.cpu().numpy())
                    t_videos.append(t_view2_feature.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_sketches = np.concatenate(t_sketches)
            t_videos = np.concatenate(t_videos)
            t_labels = np.concatenate(t_labels).argmax(1)
            Sketch2Video_map = fx_calc_map_label(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_map_label(t_videos, t_sketches, t_labels)
            Sketch2Video = fx_calc_recall(t_sketches, t_videos, t_labels)
            #Video2Sketch = fx_calc_recall(t_videos, t_sketches, t_labels)

            #print('{} Loss: {:.4f} Sketch2Video: {:.4f}  Video2Sketch: {:.4f}'.format(phase, epoch_loss, Sketch2Video, Video2Sketch))
            print('{} Loss: {:.4f} Sketch2Video: mAP = {:.4f} R1 = {:.4f} R5 = {:.4f} R10 = {:.4f}'.format(phase, epoch_loss, Sketch2Video_map, Sketch2Video[0], Sketch2Video[1], Sketch2Video[2]))

            Sketch2Video_map_10 = fx_calc_map_label(t_sketches, t_videos, t_labels, 10)
            # deep copy the model
            #Sketch2Video_mean = np.mean(Sketch2Video)
            if phase == 'valid' and Sketch2Video_map_10 > best_acc:
                best_acc = Sketch2Video_map_10
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                valid_sketch_acc_history.append(Sketch2Video_map)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best mAP_10: {:4f}'.format(best_acc))
    
    save_folder = '../weights'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(best_model_wts, os.path.join(save_folder, save_name))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, valid_sketch_acc_history, epoch_loss_history