
# coding: utf-8

# In[1]:


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset


# In[2]:


import torchvision.models as models
from PIL import Image


# In[3]:


import numpy as np
#from skimage import io, transform
import random


# In[4]:


# define some constants to use
BATCH_SIZE = 4
TEST_BATCH_SIZE = 256
LOG_INTERVAL = 1
LEARNING_RATE = 0.0001
DROPOUT = 0.2
EPOCHS = 1
DATASET = "Sample_Megaface_Images"


# In[5]:


# arguments for training the model
#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
kwargs = {}


# ## Model layer sizes

# In[6]:


# convolution blocks
INPUT_SIZE = [96, 96]
INPUT_DEPTH = 3
BLOCK1_SIZE = (np.array(INPUT_SIZE) / 2).astype(int).tolist()
BLOCK1_DEPTH = 64
BLOCK2_SIZE = (np.array(BLOCK1_SIZE) / 2).astype(int).tolist()
BLOCK2_DEPTH = BLOCK1_DEPTH
BLOCK3_SIZE = (np.array(BLOCK2_SIZE) / 2).astype(int).tolist()
BLOCK3_DEPTH = BLOCK2_DEPTH * 2
BLOCK4_SIZE = (np.array(BLOCK3_SIZE) / 2).astype(int).tolist()
BLOCK4_DEPTH = BLOCK3_DEPTH * 2
BLOCK5_SIZE = (np.array(BLOCK4_SIZE) / 2).astype(int).tolist()
BLOCK5_DEPTH = BLOCK4_DEPTH * 2

# fully connected sizes
FC1_SIZE = BLOCK5_SIZE[0]*BLOCK5_SIZE[1]*BLOCK5_DEPTH
OUTPUT_SIZE = 672075

# check the sizes
print("convolutional layers")
print("input: ({0}, {1}, {2})".format(INPUT_DEPTH, INPUT_SIZE[0], INPUT_SIZE[1]))
print("block1: ({0}, {1}, {2})".format(BLOCK1_DEPTH, BLOCK1_SIZE[0], BLOCK1_SIZE[1]))
print("block2: ({0}, {1}, {2})".format(BLOCK2_DEPTH, BLOCK2_SIZE[0], BLOCK2_SIZE[1]))
print("block3: ({0}, {1}, {2})".format(BLOCK3_DEPTH, BLOCK3_SIZE[0], BLOCK3_SIZE[1]))
print("block4: ({0}, {1}, {2})".format(BLOCK4_DEPTH, BLOCK4_SIZE[0], BLOCK4_SIZE[1]))
print("block5: ({0}, {1}, {2})".format(BLOCK5_DEPTH, BLOCK5_SIZE[0], BLOCK5_SIZE[1]))
print("fully connected layers")
print("fc1: ({0}, {1})".format(FC1_SIZE, OUTPUT_SIZE))
print("output: {0}".format(OUTPUT_SIZE))


# ## Define the Model
# 18 Layer residual net model inspired by resnet-18
# 
# TODO:
# * Make sure the reshapes (.view()) are correctly applied
#     * Correct dimensions as each argument (depth, width, height) right now

# In[7]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # block1
        """
        Parameters:	
        num_features – num_features from an expected input of size batch_size x num_features x height x width
        eps – a value added to the denominator for numerical stability. Default: 1e-5
        momentum – the value used for the running_mean and running_var computation. Default: 0.1
        affine – a boolean value that when set to True, gives the layer learnable affine parameters. Default: True

        """
        self.drop1 = nn.Dropout(p=DROPOUT)
        self.bn1 = nn.BatchNorm2d(INPUT_DEPTH)
        self.layer1 = nn.Conv2d(INPUT_DEPTH, BLOCK1_DEPTH, kernel_size=7, stride=2, padding=3)
        
        # pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # block2
        self.drop3 = nn.Dropout(p=DROPOUT)
        self.bn3 = nn.BatchNorm2d(BLOCK1_DEPTH)
        self.layer3 = nn.Conv2d(BLOCK1_DEPTH, BLOCK2_DEPTH, kernel_size=3, padding=1)
        self.drop4 = nn.Dropout(p=DROPOUT)
        self.bn4 = nn.BatchNorm2d(BLOCK2_DEPTH)
        self.layer4 = nn.Conv2d(BLOCK2_DEPTH, BLOCK2_DEPTH, kernel_size=3, padding=1)
        self.drop5 = nn.Dropout(p=DROPOUT)
        self.bn5 = nn.BatchNorm2d(BLOCK2_DEPTH)
        self.layer5 = nn.Conv2d(BLOCK2_DEPTH, BLOCK2_DEPTH, kernel_size=3, padding=1)
        self.drop6 = nn.Dropout(p=DROPOUT)
        self.bn6 = nn.BatchNorm2d(BLOCK2_DEPTH)
        self.layer6 = nn.Conv2d(BLOCK2_DEPTH, BLOCK2_DEPTH, kernel_size=3, padding=1)
        
        # block3
        self.drop7 = nn.Dropout(p=DROPOUT)
        self.bn7 = nn.BatchNorm2d(BLOCK2_DEPTH)
        self.layer7 = nn.Conv2d(BLOCK2_DEPTH, BLOCK3_DEPTH, kernel_size=3, padding=1, stride=2)
        self.drop8 = nn.Dropout(p=DROPOUT)
        self.bn8 = nn.BatchNorm2d(BLOCK3_DEPTH)
        self.layer8 = nn.Conv2d(BLOCK3_DEPTH, BLOCK3_DEPTH, kernel_size=3, padding=1)
        self.layer8_res = nn.Linear(BLOCK2_DEPTH*BLOCK2_SIZE[0]*BLOCK2_SIZE[1], 
                                    BLOCK3_DEPTH*BLOCK3_SIZE[0]*BLOCK3_SIZE[1])
        self.drop9 = nn.Dropout(p=DROPOUT)
        self.bn9 = nn.BatchNorm2d(BLOCK3_DEPTH)
        self.layer9 = nn.Conv2d(BLOCK3_DEPTH, BLOCK3_DEPTH, kernel_size=3, padding=1)
        self.drop10 = nn.Dropout(p=DROPOUT)
        self.bn10 = nn.BatchNorm2d(BLOCK3_DEPTH)
        self.layer10 = nn.Conv2d(BLOCK3_DEPTH, BLOCK3_DEPTH, kernel_size=3, padding=1)
        
        # block4
        self.drop11 = nn.Dropout(p=DROPOUT)
        self.bn11 = nn.BatchNorm2d(BLOCK3_DEPTH)
        self.layer11 = nn.Conv2d(BLOCK3_DEPTH, BLOCK4_DEPTH, kernel_size=3, padding=1, stride=2)
        self.drop12 = nn.Dropout(p=DROPOUT)
        self.bn12 = nn.BatchNorm2d(BLOCK4_DEPTH)
        self.layer12 = nn.Conv2d(BLOCK4_DEPTH, BLOCK4_DEPTH, kernel_size=3, padding=1)
        self.layer12_res = nn.Linear(BLOCK3_DEPTH*BLOCK3_SIZE[0]*BLOCK3_SIZE[1], 
                                    BLOCK4_DEPTH*BLOCK4_SIZE[0]*BLOCK4_SIZE[1])
        self.drop13 = nn.Dropout(p=DROPOUT)
        self.bn13 = nn.BatchNorm2d(BLOCK4_DEPTH)
        self.layer13 = nn.Conv2d(BLOCK4_DEPTH, BLOCK4_DEPTH, kernel_size=3, padding=1)
        self.drop14 = nn.Dropout(p=DROPOUT)
        self.bn14 = nn.BatchNorm2d(BLOCK4_DEPTH)
        self.layer14 = nn.Conv2d(BLOCK4_DEPTH, BLOCK4_DEPTH, kernel_size=3, padding=1)
        
        # block5
        self.drop15 = nn.Dropout(p=DROPOUT)
        self.bn15 = nn.BatchNorm2d(BLOCK4_DEPTH)
        self.layer15 = nn.Conv2d(BLOCK4_DEPTH, BLOCK5_DEPTH, kernel_size=3, padding=1, stride=2)
        self.drop16 = nn.Dropout(p=DROPOUT)
        self.bn16 = nn.BatchNorm2d(BLOCK5_DEPTH)
        self.layer16 = nn.Conv2d(BLOCK5_DEPTH, BLOCK5_DEPTH, kernel_size=3, padding=1)
        self.layer16_res = nn.Linear(BLOCK4_DEPTH*BLOCK4_SIZE[0]*BLOCK4_SIZE[1], 
                                    BLOCK5_DEPTH*BLOCK5_SIZE[0]*BLOCK5_SIZE[1])
        self.drop17 = nn.Dropout(p=DROPOUT)
        self.bn17 = nn.BatchNorm2d(BLOCK5_DEPTH)
        self.layer17 = nn.Conv2d(BLOCK5_DEPTH, BLOCK5_DEPTH, kernel_size=3, padding=1)
        self.drop18 = nn.Dropout(p=DROPOUT)
        self.bn18 = nn.BatchNorm2d(BLOCK5_DEPTH)
        self.layer18 = nn.Conv2d(BLOCK5_DEPTH, BLOCK5_DEPTH, kernel_size=3, padding=1)
        
        # pooling layer
        self.pool19 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # fully connected
        #self.drop1 = nn.Dropout(p=DROPOUT)
        self.bn20 = nn.BatchNorm1d(FC1_SIZE)
        self.layer20 = nn.Linear(FC1_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        
        # input
        f0 = x
        
        # block 1
        f1 = self.layer1(F.relu(self.bn1(self.drop1(f0))))
        
        # pool
        f2 = self.pool2(f1)
        
        # block 2
        f3 = self.layer3(F.relu(self.bn3(self.drop3(f2))))
        f4 = self.layer4(F.relu(self.bn4(self.drop4(f3)))) + f2
        f5 = self.layer5(F.relu(self.bn5(self.drop5(f4))))
        f6 = self.layer6(F.relu(self.bn6(self.drop6(f5)))) + f4
        
        # block 3
        f7 = self.layer7(F.relu(self.bn7(self.drop7(f6))))
        f8 = self.layer8(F.relu(self.bn8(self.drop8(f7)))) +             self.layer8_res(f6.view(-1, BLOCK2_DEPTH*BLOCK2_SIZE[0]*BLOCK2_SIZE[1]))            .view(-1, BLOCK3_DEPTH, BLOCK3_SIZE[0], BLOCK3_SIZE[1])
        f9 = self.layer9(F.relu(self.bn9(self.drop9(f8))))
        f10 = self.layer10(F.relu(self.bn10(self.drop10(f9)))) + f8
        
        # block 4
        f11 = self.layer11(F.relu(self.bn11(self.drop11(f10))))
        f12 = self.layer12(F.relu(self.bn12(self.drop12(f11)))) +             self.layer12_res(f10.view(-1, BLOCK3_DEPTH*BLOCK3_SIZE[0]*BLOCK3_SIZE[1]))            .view(-1, BLOCK4_DEPTH, BLOCK4_SIZE[0], BLOCK4_SIZE[1])
        f13 = self.layer13(F.relu(self.bn13(self.drop13(f12))))
        f14 = self.layer14(F.relu(self.bn14(self.drop14(f13)))) + f12
        
        # block 5
        f15 = self.layer15(F.relu(self.bn15(self.drop15(f14))))
        f16 = self.layer16(F.relu(self.bn16(self.drop16(f15)))) +             self.layer16_res(f14.view(-1, BLOCK4_DEPTH*BLOCK4_SIZE[0]*BLOCK4_SIZE[1]))            .view(-1, BLOCK5_DEPTH, BLOCK5_SIZE[0], BLOCK5_SIZE[1])
        f17 = self.layer17(F.relu(self.bn17(self.drop17(f16))))
        f18 = self.layer18(F.relu(self.bn18(self.drop18(f17))))
        
        # pool 
        f19 = self.pool19(f18)
        
        # fc
        f20 = self.layer20(F.relu(self.bn20(f19.view(-1, FC1_SIZE))))
        
        # return the softmax of the probability
        return F.log_softmax(x)


# ## Instantiate the Model

# In[8]:


model = Net()

# if we want to use gpu: 
#model.cuda()


# ## Create the data loaders
# Load in the training data and test data from batches
# 
# TODO:
# * configure the correct batch sizes

# In[10]:


transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.STL10(root='./data', split='train',
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.STL10(root='./data', split='test',
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[ ]:


class MegaFaceDataset(Dataset):

    """
    All datasets are subclasses of torch.utils.data.Dataset i.e, they have
    __getitem__ and __len__ methods implemented.
    Hence, they can all be passed to a torch.utils.data.DataLoader which can
    load multiple samples parallelly using torch.multiprocessing workers

    Source: http://pytorch.org/docs/master/torchvision/datasets.html

    The data in the root folder must be arranged in this way:

    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/xxz.png

    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/asd932_.png
    """

    def __init__(self, imageFolderDataset, transform = None, should_invert = True):
        self.imageFolderDataset = imageFolderDataset #path to either training or testing dir
        self.transform = transform #tranform the input (image augmentation)
        self.should_invert = should_invert

    def __getitem__(self, index): #Check what index does?
        image_tuple = random.choice(self.imageFolderDataset.imgs)
        image = Image.open(image_tuple[0])

        image = image.convert("L") #Converts an image to grayscale

        if self.should_invert:
            image = PIL.ImageOps.invert(image)

        if self.transform:
            image = self.transform(image)

        return image, 1

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
    
megaface_dataset = datasets.ImageFolder(root=DATASET)

#Loading the data
megaface_dataset = MegaFaceDataset(imageFolderDataset=megaface_dataset,
                                        transform=transforms.Compose([transforms.Scale(tuple(INPUT_SIZE)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)
train_loader = DataLoader(megaface_dataset, batch_size=BATCH_SIZE)
print(len(megaface_dataset))
print(len(train_loader)) #Print the number of images loaded


# In[ ]:


# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor()
#                    ])),
#     batch_size=BATCH_SIZE, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor()
#                    ])),
# batch_size=BATCH_SIZE, shuffle=True, **kwargs)


# print out the data to check
#for batch_idx, (data, target) in enumerate(train_loader):
#    print(data)


# ## Optimizer

# In[11]:


# define the optimizer
"""
params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
lr (float, optional) – learning rate (default: 1e-3)
betas (Tuple[float, float], optional) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-8)
weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
"""
optimizer = optim.Adam(model.parameters())


# ## Specifiy what training will take place

# In[12]:


# define training function
def train(epoch, model):
    """
        Train the model
        Inputs:
            epoch - number of the current epoch
            
        Outputs:
            
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


# ## How will we test the model

# In[13]:


def test(model):
    """
        Test the model's accuracy
        Inputs:
            None
        Outputs: 
            Prints the test output results
    """
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# ## Run the Training & Testing

# In[14]:


for epoch in range(1, EPOCHS+1):
    train(epoch, model)
    test(model)


# # Some extra cells to print testing stuff

# In[ ]:


print(model.parameters())


# In[ ]:




