import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim


## load mnist dataset
root = './data'
download = True

# specify transforms to apply to input data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False, **kwargs)

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, y, target):
        """
        custom loss function - implementation in numpy

        Inputs:
            y - predicted output
            target - actual class labels
        Outputs: 
            loss - the loss between y and target
        """
        # indices = target:view(-1,1)
        one_hot = torch.LongTensor(torch.zeros(128, 10))
        one_hot:scatter(2, target, 1)
        print(one_hot.size())
        print(one_hot[0:2])

        print(y[:, 0].size())
        print(target.size())
        loss = F.pairwise_distance(y, one_hot)
        return 1


class CustomNetwork(nn.Module):

    def __init__(self):
        """
        initialize model and weights and such 
        """
        super(CustomNetwork, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        self.loss = Loss()

    def forward(self, x, target):
        """
        implement the forward pass of network
        backward pass handled by pytorch
        """
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = F.relu(self.fc3(x))
        loss = self.loss(y, target)
        #print("sizes - ", "pred: ", y.size(), ", target: ", target.size())
        return y, loss

    def name(self):
        return 'myNetwork'

## training
model = CustomNetwork().cuda()

# define the optimizer to use, pass in model
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for epoch in range(10):

    # trainning
    for batch_idx, (x, target) in enumerate(train_loader):
        # zero out the gradients
        optimizer.zero_grad()
        # load data
        x, target = Variable(x.cuda()), Variable(target.cuda())
        # perform prediction
        _, loss = model(x, target)
        # let pytorch train model
        loss.backward()
        optimizer.step()
        # print out updates
        if batch_idx % 100 == 0:
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, loss.data[0]))


    # testing
    correct_cnt, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
        score, loss = model(x, target)
        _, pred_label = torch.max(score.data, 1)
        correct_cnt += (pred_label == target.data).sum()
        ave_loss += loss.data[0]
    accuracy = correct_cnt*1.0/len(test_loader)/batch_size
    ave_loss /= len(test_loader)
    print('==>>> epoch: {}, test loss: {:.6f}, accuracy: {:.4f}'.format(epoch, ave_loss, accuracy))

# save the model so we can use it later if we want to 
torch.save(model.state_dict(), model.name())
