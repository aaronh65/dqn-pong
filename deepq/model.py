# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# class DQNbn(nn.Module):
#     def __init__(self, in_channels=4, n_actions=14):
#         """
#         Initialize Deep Q Network

#         Args:
#                 in_channels (int): number of input channels
#                 n_actions (int): number of outputs
#         """
#         super(DQNbn, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.fc4 = nn.Linear(7 * 7 * 64, 512)
#         self.head = nn.Linear(512, n_actions)

#     def forward(self, x):
#         x = x.float() / 255
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.fc4(x.view(x.size(0), -1)))
        # return self.head(x)

class DQNBase(nn.Module):
   def __init__(self, in_channels=4, n_actions=14):
       """
       Initialize Deep Q Network

       Args:
               in_channels (int): number of input channels
               n_actions (int): number of outputs
       """
       super(DQNBase, self).__init__()

       self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
       # self.bn1 = nn.BatchNorm2d(32)
       self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
       # self.bn2 = nn.BatchNorm2d(64)
       self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
       # self.bn3 = nn.BatchNorm2d(64)
       self.fc4 = nn.Linear(7 * 7 * 64, 512)
       self.head = nn.Linear(512, n_actions)

   def forward(self, x):
       x = x.float() / 255
       x = F.relu(self.conv1(x))
       x = F.relu(self.conv2(x))
       x = F.relu(self.conv3(x))
       x = F.relu(self.fc4(x.view(x.size(0), -1)))
       return self.head(x)

class DQNEncodedFeatures(nn.Module):
   def __init__(self, in_channels=4, n_actions=14):
       """
       Initialize Deep Q Network

       Args:
               in_channels (int): number of input channels
               n_actions (int): number of outputs
       """
       super(DQNEncodedFeatures, self).__init__()
       self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
       self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
       self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
       self.fc4 = nn.Linear(7 * 7 * 64, 512)
       self.head = nn.Linear(512, n_actions)

   def forward(self, x):
        print(x.shape)
    #    x = x.float() / 255
    #    x = F.relu(self.conv1(x))
    #    x = F.relu(self.conv2(x))
    #    x = F.relu(self.conv3(x))
    #    x = F.relu(self.fc4(x.view(x.size(0), -1)))
    #    return self.head(x)
        return x

# class DQNBase(nn.Module):
#     def __init__(self, n_actions, history_size=4):
#         super().__init__()
#         self.network = torchvision.models.resnet18(pretrained=False, num_classes=n_actions)
#         #self.network = nn.Sequential(*list(resnet18.children())[:]) 
#         old = self.network.conv1
#         self.network.conv1 = nn.Conv2d(
#             history_size, old.out_channels,
#             kernel_size=old.kernel_size, stride=old.stride,
#             padding=old.padding, bias=old.padding)

#     def forward(self, x):
#         x = self.network(x) # N,512,8,5
#         return x


