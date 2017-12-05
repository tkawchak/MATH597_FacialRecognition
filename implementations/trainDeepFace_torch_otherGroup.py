import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import random
import torch.nn.functional as F
import os
import random
import io
from PIL import Image
import torchvision.transforms as transforms

#https://stackoverflow.com/questions/33330779/whats-the-triplet-loss-back-propagation-gradient-formula

# class TripletLoss(torch.nn.Module):
#
#     def __init__(self, margin=2.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, output_anchor, output_negative, output_positive, label):
#         alpha = 1
#         euclidean_distance_positive = F.pairwise_distance(output_anchor, output_positive)
#         euclidean_distance_negative = F.pairwise_distance(output_anchor, output_negative)
#         triplet_loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))




# Define DeepFace Class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv8 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv9 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv11 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Conv2d(512, 4096, 3, padding = 0)
        self.fc2 = nn.Conv2d(4096, 4096, 3, padding = 0)
        self.fc3 = nn.Conv2d(4096, 2622, 3, padding = 0)


    def forward(self, inputdata):

        hidden = F.relu(self.conv1(inputdata))
        hidden = self.pool(F.relu(self.conv2(hidden)))

        hidden = F.relu(self.conv3(hidden))
        hidden = self.pool(F.relu(self.conv4(hidden)))

        hidden = F.relu(self.conv5(hidden))
        hidden = F.relu(self.conv6(hidden))
        hidden = self.pool(F.relu(self.conv7(hidden)))

        hidden = F.relu(self.conv8(hidden))
        hidden = F.relu(self.conv9(hidden))
        hidden = self.pool(F.relu(self.conv10(hidden)))

        hidden = F.relu(self.conv11(hidden))
        hidden = F.relu(self.conv12(hidden))
        hidden = self.pool(F.relu(self.conv13(hidden)))

        hidden = F.relu(self.fc1(hidden))
        hidden = F.relu(self.fc2(hidden))

        out = F.softmax(self.fc3(hidden))

        return out




def load_dataToDict(data_path):
    mydict = {}
    mydict_multi = {}
    for i in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, i)):
            mydict[i] = []
            for j in os.listdir(os.path.join(data_path, i)):
                mydict[i].append(j)
            if len(mydict[i])>1:
                mydict_multi[i] = mydict[i]

    return mydict, mydict_multi

# Generates triplets such that first two elements are same and third is different (anchor, positive, negative)
def generateRandomTriplets(data_path, data_dict, data_multiInstances, size):

    dataset = []
    keys = data_multiInstances.keys()
    out = open('Dataset.txt', 'w')
    for i in range(0, size):
        num = random.randint(0, len(keys)-1 )
        im1 = random.randint(0, len(data_multiInstances[keys[num]])-1 )
        im2 = random.randint(0, len(data_multiInstances[keys[num]])-2 )
        if im2 >= im1:
            im2 = im2 + 1

        a = os.path.join(data_path, keys[num], data_multiInstances[keys[num]][im1])
        p = os.path.join(data_path, keys[num], data_multiInstances[keys[num]][im2])

        keys_all = data_dict.keys()
        num_negative = random.randint(0, len(keys_all)-2 )
        if num_negative >= num:
            num_negative = num_negative + 1

        im_neg = random.randint(0, len(data_dict[keys_all[num_negative]])-1 )
        n = os.path.join(data_path, keys_all[num_negative], data_dict[keys_all[num_negative]][im_neg])

        out.write(data_multiInstances[keys[num]][im1] + ', ' + data_multiInstances[keys[num]][im2] + ', ' + data_dict[keys_all[num_negative]][im_neg] + '\n')

        dataset.append([a, p, n])
    return dataset


loader = transforms.Compose([
    #transforms.Scale(448),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


learning_rate = 0.01
def train(dataset):

    neural_net = Net()
    triplet_loss = nn.TripletMarginLoss(margin = 1.0)

    for data in dataset:
        print data[0], data[1], data[2]
        a = Variable(loader(Image.open(data[0]))).unsqueeze(0)
        p = Variable(loader(Image.open(data[1]))).unsqueeze(0)
        n = Variable(loader(Image.open(data[2]))).unsqueeze(0)

        anc = neural_net(a)
        pos = neural_net(p)
        neg = neural_net(n)


        anc = anc.squeeze(2)
        anc = anc.squeeze(2)
        pos = pos.squeeze(2)
        pos = pos.squeeze(2)
        neg = neg.squeeze(2)
        neg = neg.squeeze(2)

        loss = triplet_loss(anc, pos, neg)
        loss.backward()

        for p in neural_net.parameters():
             p.data.add_(-learning_rate, p.grad.data)

data, data_simiar = load_dataToDict('lfw')
dataset = generateRandomTriplets('lfw', data, data_simiar, 10)
train(dataset)


#print data.keys()
