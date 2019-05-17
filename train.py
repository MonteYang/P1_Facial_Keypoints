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
import os
import re

data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])


def train_net(n_epochs,load_model_dir, batch_size=128, lr=0.0001, save_model_dir="saved_models/"):
    #  prepare data
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

    # loading previous weights
    if os.path.exists(load_model_dir):
        print("loading previous weights...")
        net.load_state_dict(torch.load(load_model_dir))
        epoch_p = int(re.findall("\d+", load_model_dir)[-1])
    else:
        print("training from 0...")
        epoch_p = 0

    # setting optimizer , criterion
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_i =0
    for epoch in range(epoch_p, n_epochs):  # loop over the dataset multiple times
        epoch_i += 1
        running_loss = 0.0
        # 每100个epoch保存一次模型
        if epoch_i % 100 == 0:
            # after training, save your model parameters in the dir 'saved_models'
            torch.save(net.state_dict(), save_model_dir + 'keypoints_model_{}.pt'.format(epoch_i))

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


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    for i in range(batch_size):
        plt.figure(figsize=(40, 10))
        ax = plt.subplot(1, batch_size, i + 1)

        # un-transform the image data
        image = test_images[i].data  # get the image from it's wrapper
        image = image.cpu()
        image = image.numpy()  # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))  # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.cpu()
        predicted_key_pts = predicted_key_pts.numpy()

        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts * 50.0 + 100

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100

        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)

        plt.axis('off')

    plt.show()


def net_sample_output(net):
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.cuda.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


if __name__ == '__main__':

    # Train = 1为训练, =0 为测试
    Train = 1
    load_model_dir = "saved_models/keypoints_model_200.pt"
    n_epochs = 10000

    if Train:
        train_net(n_epochs, load_model_dir=load_model_dir)
    else:
        test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                              root_dir='data/test/',
                                              transform=data_transform)
        batch_size = 10

        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4)
        net = Net()
        net.cuda()
        net.load_state_dict(torch.load(load_model_dir))
        net.eval()
        test_images, test_outputs, gt_pts = net_sample_output(net)
        visualize_output(test_images, test_outputs, gt_pts)
