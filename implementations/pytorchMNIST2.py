import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# define some constants for the loss function
MARGIN = torch.autograd.Variable(torch.LongTensor([20000]))
K = torch.autograd.Variable(torch.LongTensor([2]))
ALPHA = torch.autograd.Variable(torch.FloatTensor([0.5]))


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


class Loss(torch.autograd.Function):

   # def __init__(self, margin, k, alpha, beta):
   #     super(Loss, self).__init__()
   #     self.save_margin = margin
   #     self.k = k
   #     self.alpha = alpha
   #     self.beta = beta

    def forward(self, y, target, k, a, m):
        """
        custom loss function - implementation in numpy

        Inputs:
            y - predicted output
            target - actual class labels
        Outputs: 
            loss - the loss between y and target
        """
        self.save_for_backward(k, a, m, y, target)
        b = 1.0 - a
        
        # convert the data into numpy arrays so they can be manipulated accordingly
        features = y.numpy()
        labels = target.numpy()

        # fix this!!
        loss, counts, centers, l_intra, inter_indices, l_inter, d = self.compute_loss(features, labels, k.numpy(), a.numpy(), b.numpy(), m.numpy())

        # compute the loss
        loss_autograd = torch.autograd.Variable(torch.from_numpy(np.array([loss])), requires_grad=True)
        # do i need to clone the output???
        return loss_autograd.data

    def compute_loss(self, features, labels, k, a, b, m):

        # count the number of class labels and the number of data per class label
        unique_labels, counts = np.unique(labels, return_counts=True)

        # store the centers for each class (mean of data)
        centers = np.zeros((unique_labels.shape[0], features.shape[1]))
        # find the top k Euclidean distances for each class
        d = np.zeros((unique_labels.shape[0], k[0]))

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
            d[idx, :] = self.compute_top_k(features_l, k[0])
            # compute the harmonic mean, intra-class loss
            l_r[idx] = k[0] / np.sum(d[idx, :])

        # compute total intra-class loss
        l_intra = np.sum(l_r)
        # compute the shortest distances among all feature centers
        d_center, inter_indices = self.compute_min_dist(centers)
        # compute total inter-class loss
        l_inter = max(m[0] - d_center, 0)
        # combine these together for the total loss
        loss = l_intra * a[0] + l_inter * b[0]
        #self.assign(out_data[0], req[0], mx.nd.array(loss))
        
        return loss, counts, centers, l_intra, inter_indices, l_inter, d


    def compute_top_k(self, features, k):

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
        return dist_array[-k:]

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
        min_value = dist_array[0]
        min_idx = [(col, row) for row in range(num) if min_value in dists[row] for col in range(num) if min_value == dists[row][col]][0]
        return min_value, min_idx

    def backward(self, grad_out):
        """
        Compute the gradient with respect to input to layer
        """

        # *********************************************
        # check this with torch.autograd.gradcheck !!!!
        # *********************************************

        k, a, m, y, targets = self.saved_tensors
        b = 1.0 - a

        features = y.numpy()
        labels = targets.numpy()

        loss, counts, centers, l_intra, inter_indices, l_inter, d = self.compute_loss(features, labels, k.numpy(), a.numpy(), b.numpy(), m.numpy())

        grad_inter = torch.FloatTensor(y.size())
        grad_intra = torch.FloatTensor(y.size())

        idx1 = inter_indices[0]
        idx2 = inter_indices[1]
        grad_inter[idx1] = torch.from_numpy(0.5 / (counts[idx1]) * np.abs(centers[idx1] - centers[idx2]))
        grad_inter[idx2] = torch.from_numpy(0.5 / (counts[idx2]) * np.abs(centers[idx2] - centers[idx1]))

        # compute intra class gradients with respect to xi, xj
        # only nonzero for these two values

        # *********************************************************
        # HOW TO COMPUTE GRADIENTS WITH RESPECT TO MULTIPLE SAMPLES
        # WHEN LOSS IS JUST COMPUTED OVERALL????
        # *********************************************************

        for idx in range(y.size()[1]):
            denom = np.array([np.power(d[idx,0]*np.sum(d[idx,:]),2)])
            grad = 2*k.double() / torch.from_numpy(denom)
            for entry in range(y.size()[0]):
                grad_intra[entry, idx] = grad[0]

        # compute inter class gradients with respect to xq, xr
        # only nonzero for these two values

        # ****************************************
        # SOMEHOW THE GRADIENT IS WAY TOO BIG ****
        # ****************************************
        grad_in = a*grad_intra + b*grad_inter
        print(grad_in)
        return grad_in, torch.DoubleTensor([0]), torch.DoubleTensor([0]), torch.DoubleTensor([0]), torch.DoubleTensor([0])

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
        self.loss = Loss()

    def forward(self, x, target, k, a, m):
        """
        implement the forward pass of network
           backward pass handled by pytorch
        """
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = F.relu(self.fc3(x))
        loss = self.loss(y, target, k, a, m)
        return y, loss

    def name(self):
        return 'myNetwork'

## training
model = CustomNetwork()

# define the optimizer to use, pass in model
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(10):

    # trainning
    for batch_idx, (x, target) in enumerate(train_loader):
        # zero out the gradients
        optimizer.zero_grad()
        # load data
        x, target = Variable(x, requires_grad=True), Variable(target, requires_grad=True)
        # perform prediction
        _, loss = model(x, target, K, ALPHA, MARGIN)
        # let pytorch train model
        loss.backward(retain_graph=True)
        optimizer.step()
        # print out updates
        if batch_idx % 100 == 0:
            print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, loss.data[0]))


    # testing
    correct_cnt, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        score, loss = model(x, target, K, ALPHA, MARGIN)
        _, pred_label = torch.max(score.data, 1)
        correct_cnt += (pred_label == target.data).sum()
        ave_loss += loss.data[0]
    accuracy = correct_cnt*1.0/len(test_loader)/batch_size
    ave_loss /= len(test_loader)
    print('==>>> epoch: {}, test loss: {:.6f}, accuracy: {:.4f}'.format(epoch, ave_loss, accuracy))

# save the model so we can use it later if we want to 
torch.save(model.state_dict(), model.name())
