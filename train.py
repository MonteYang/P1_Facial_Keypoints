# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import Rescale, RandomCrop, Normalize, ToTensor
from data_load import FacialKeypointsDataset
import torch.optim as optim







def train_net(n_epochs, batch_size=128, lr=0.0001, model_dir="saved_models/"):
    #  prepare data
    data_transform = transforms.Compose([Rescale(250),
                                         RandomCrop(224),
                                         Normalize(),
                                         ToTensor()])
    transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                 root_dir='data/training/',
                                                 transform=data_transform)

    # testing that you've defined a transform
    assert (data_transform is not None), 'Define a data_transform'
    train_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    # prepare the net for training
    net = Net()
    net.train()
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_i = 0

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        epoch_i += 1
        running_loss = 0.0
        # 每100个epoch保存一次模型
        if epoch_i % 100 == 0:
            # after training, save your model parameters in the dir 'saved_models'
            torch.save(net.state_dict(), model_dir + 'keypoints_model_{}.pt'.format(epoch_i))

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 1000))
                running_loss = 0.0
        # test
        # if epoch_i == 1:
        #     break

    print('Finished Training')


if __name__ == '__main__':

    train_net(10000)
