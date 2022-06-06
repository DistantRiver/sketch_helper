import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

class SHNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(SHNet, self).__init__()
        self.img_net = models.resnet34(pretrained=True)
        self.img_cnn = nn.Sequential(*list(self.img_net.children())[:-1])
        self.img_fea_len = 512
        self.cls_num = 345
        self.dropout = nn.Dropout(p=0.5)
        self.linearLayer1 = nn.Linear(self.img_fea_len, self.img_fea_len)
        self.linearLayer2 = nn.Linear(self.img_fea_len, self.cls_num)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        x = x.view(-1, 3, 100, 100)
        img_feas = self.img_cnn(x)
        new_x = img_feas.view(-1, self.img_fea_len)
        out = self.dropout(new_x)
        str_out = self.linearLayer1(out)
        cls_out = self.linearLayer2(str_out)
        return img_feas.view(-1, self.img_fea_len), str_out.view(-1, self.img_fea_len), cls_out.view(-1, self.cls_num)


class SHNet_lstm(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(SHNet_lstm, self).__init__()
        self.img_net = models.resnet34(pretrained=True)
        self.img_cnn = nn.Sequential(*list(self.img_net.children())[:-1])
        self.img_fea_len = 512
        self.cls_num = 345
        self.lstm = nn.LSTM(input_size=self.img_fea_len, hidden_size=345, num_layers=1, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(p=0.5)
        self.linearLayer1 = nn.Linear(self.cls_num, self.img_fea_len)
        self.linearLayer2 = nn.Linear(self.img_fea_len, self.cls_num)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        x = x.view(-1, 3, 100, 100)
        img_feas = self.img_cnn(x)
        new_x = img_feas.view(-1, 4, self.img_fea_len)
        out, (h,c) = self.lstm(new_x)
        out = self.dropout(out)
        str_out = self.linearLayer1(out)
        cls_out = self.linearLayer2(str_out)
        return img_feas.view(-1, self.img_fea_len), str_out.view(-1, self.img_fea_len), cls_out.view(-1, self.cls_num)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, adj):
        out = F.relu(self.gc1(x, adj))
        out = self.dropout(out)
        out = self.gc2(out, adj)
        return out

    
class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=256,kernel_size=13,stride=1)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        return out


class VideoNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=4096, output_dim=1024):
        super(VideoNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out

class Keyframe(nn.Module):
    """Network to learn text representations"""
    def __init__(self, sketch_input_dim=4096, sketch_output_dim=1024,
                 video_input_dim=1024, video_output_dim=1024, video_frame_num=5, minus_one_dim=1024):
        super(Keyframe, self).__init__()
        self.sketch_net = ImgNN(sketch_input_dim, sketch_output_dim)
        self.video_net = VideoNN(video_input_dim, video_output_dim)
        self.linearLayer1 = nn.Linear(sketch_output_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(video_output_dim, minus_one_dim)

    def forward(self, sketch, video):
        view1_feature = self.sketch_net(sketch)
        view2_feature = self.video_net(video)
        view1_feature = self.linearLayer1(view1_feature)
        view2_feature = self.linearLayer2(view2_feature)

        return view1_feature, view2_feature

class Baseline(nn.Module):
    """Network to learn text representations"""
    def __init__(self, sketch_input_dim=4096, sketch_output_dim=1024,
                 video_input_dim=1024, video_output_dim=1024, video_frame_num=5, minus_one_dim=1024):
        super(Baseline, self).__init__()
        self.sketch_net = ImgNN(sketch_input_dim, sketch_output_dim)
        self.video_net = VideoNN(video_input_dim, video_output_dim)
        self.linearLayer1 = nn.Linear(sketch_output_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(video_output_dim, minus_one_dim)
        
        self.video_attn1 = nn.Linear(video_frame_num, video_frame_num)
        self.video_attn2 = nn.Linear(video_frame_num, video_frame_num)

    def forward(self, sketch, video):
        view1_feature = self.sketch_net(sketch)
        view2_feature = self.video_net(video)
        view1_feature = self.linearLayer1(view1_feature)
        view2_feature = self.linearLayer2(view2_feature)
        
        view2_attn = torch.sigmoid(self.video_attn2(F.relu(self.video_attn1(view2_feature.mean(-1)))))
        view2_attn = view2_attn.unsqueeze(-1).expand_as(view2_feature)
        view2_feature = (view2_feature * view2_attn).mean(-2)
        #view2_feature = view2_feature.mean(-2)

        return view1_feature, view2_feature
