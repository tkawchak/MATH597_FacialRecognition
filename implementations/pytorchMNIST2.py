import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# define some constants for the loss function
MARGIN = 20000
K = 2
ALPHA = 0.5
BETA = 0.5


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

    def __init__(self, margin, k, alpha, beta):
        super(Loss, self).__init__()
        self.margin = margin
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def forward(self, y, target):
        """
        custom loss function - implementation in numpy

        Inputs:
            y - predicted output
            target - actual class labels
        Outputs: 
            loss - the loss between y and target
        """
        # store the predicted values so we can use the size for the gradient computation
        self.pred = y
        self.in_data = y
        self.target = target

        # taken from function on rangeloss implementation

        # convert the data into numpy arrays so they can be manipulated accordingly
        features = y.data.numpy()
        labels = target.data.numpy()

        # count the number of class labels and the number of data per class label
        unique_labels, counts = np.unique(labels, return_counts=True)

        # store the centers for each class (mean of data)
        centers = np.zeros((unique_labels.shape[0], features.shape[1]))
        # find the top k Euclidean distances for each class
        d = np.zeros((unique_labels.shape[0], self.k))

        # initialize the harmonic mean (intra-class loss) for each class
        l_r = np.zeros((unique_labels.shape[0]))

        # compute the per-class items like average and whatnot
        for idx, l in enumerate(unique_labels):
            indices = np.where(labels == l)[0]
            # extract the features of the one group
            features_l = features[indices, :]
            # compute the centers (means)
            centers[idx, :] = np.mean(features_l, axis=0)
            # compute the top k Euclidean distances
            d[idx, :] = self.compute_top_k(features_l)
            # compute the harmonic mean, intra-class loss
            l_r[idx] = self.k / np.sum(d[idx, :])

        # compute total intra-class loss
        l_intra = np.sum(l_r)
        self.l_intra = l_intra
        # compute the shortest distances among all feature centers
        d_center = self.compute_min_dist(centers)
        # compute total inter-class loss
        l_inter = max(self.margin - d_center, 0)
        self.l_inter = l_inter
        # combine these together for the total loss
        loss = l_intra * self.alpha + l_inter * self.beta
        #self.assign(out_data[0], req[0], mx.nd.array(loss))

        # compute the loss
        loss_autograd = torch.autograd.Variable(torch.from_numpy(np.array([loss])), requires_grad=True)
        return loss_autograd

    def compute_top_k(self, features):

        # this can probably be done faster in a list comprehension

        # initialize an array of zeros and fill in pairwise distances
        num = features.shape[0]
        dists = np.zeros((num, num))
        for id_1 in range(num):
            for id_2 in range(id_1+1, num):
                dist = np.linalg.norm(features[id_1] - features[id_2])
                dists[id_1, id_2] = dist

        # reshape the array to be 1D
        dist_array = dists.reshape((-1, ))
        dist_array.sort()
        return dist_array[-self.k:]

    def compute_min_dist(self, centers):

        # this can probably be done faster in a list compreshension

        # compute the min distances between all class feature centers
        # pretty much the same and the top_k features
        num = centers.shape[0]
        dists = np.ones((num, num)) * -1
        for id_1 in range(num):
            for id_2 in range(id_1+1, num):
                dist = np.linalg.norm(centers[id_1] - centers[id_2])
                dists[id_1, id_2] = dist

                # reshape the dists array to be 1D and then find where distances > 0
                # why do we do where dist_array > 0???
                dist_array = dists.reshape((1, -1))
                dist_array = dist_array[np.where(dist_array > 0)]

        # return the minimum distance
        dist_array.sort()
        return dist_array[0]

    def backward(self, grad_out):
        """
        Compute the gradient with respect to input to layer
        """
        grad_in = torch.FloatTensor(self.pred.size())


        features = in_data[0].asnumpy()
        labels = in_data[1].asnumpy().ravel().astype(np.int)

        unique_labels, counts = np.unique(labels, return_count=True)

        centers = np.zeros((unique_labels.shape[0], self.k))
        d = np.zeros((unique_labels.shape[0], self.k))

        l_r = np.zeros



        return grad_in

    def name(self):
        return "rangeloss"



class CustomNetwork(nn.Module):

    def __init__(self):
        """
        initialize model and weights and such 
        """
        super(CustomNetwork, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        self.loss = Loss(MARGIN, K, ALPHA, BETA)

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
model = CustomNetwork()

# define the optimizer to use, pass in model
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for epoch in range(10):

    # trainning
    for batch_idx, (x, target) in enumerate(train_loader):
        # zero out the gradients
        optimizer.zero_grad()
        # load data
        x, target = Variable(x, requires_grad=True), Variable(target, requires_grad=True)
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
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        score, loss = model(x, target)
        _, pred_label = torch.max(score.data, 1)
        correct_cnt += (pred_label == target.data).sum()
        ave_loss += loss.data[0]
    accuracy = correct_cnt*1.0/len(test_loader)/batch_size
    ave_loss /= len(test_loader)
    print('==>>> epoch: {}, test loss: {:.6f}, accuracy: {:.4f}'.format(epoch, ave_loss, accuracy))

# save the model so we can use it later if we want to 
torch.save(model.state_dict(), model.name())
