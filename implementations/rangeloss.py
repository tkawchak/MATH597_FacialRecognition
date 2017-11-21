
# coding: utf-8

# In[5]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict


# In[9]:


X_test = np.array([
    [1, 1, 1], 
    [1, 1, 2], 
    [2, 2, 3], 
    [2, 2, 2], 
    [3, 4, 2], 
    [4, 4, 5]
])
labels_test= np.array([
    0,
    0,
    1,
    1,
    2,
    2
])
m_test = 0.2
w_test = 0.5


# In[26]:


def rangeloss(x, labels, m, w):
    """
    Inputs: 
        x - feature set from last fully connected layer
        labels - the class labels (identities for each feature vector)
        m - margin
        w - lambda, or the balance between intra / inter class loss
        
    Output:
        Loss, gradients (with respect to x)
    """
    k = 2
    m = 0.4
    alpha = 0.5
    beta = 1.0 - alpha
    
    inter_class_grad = np.zeros(x.shape[0])
    intra_class_grad = np.zeros(x.shape[0])
    
    data_by_class = defaultdict(lambda: [])
    average_by_class = defaultdict(lambda: 0)
    k_largest_dist = defaultdict(lambda: [])
    harmonic_mean_by_class = defaultdict(lambda: 0)
    
    # group data by class
    for feature, label in zip(x, labels):
        data_by_class[label].append(feature)
        
    #compute mean
    keys = data_by_class.keys()
    for key in keys:
        average_by_class[key] = np.sum(data_by_class[key], axis=0)/len(data_by_class[key])
    
    # compute k largest Distances (euclidean) among
    # features xi of class i
    for key in keys:
        distances = np.array([np.linalg.norm(np.array(a)-np.array(b)) for a in data_by_class[key] for b in data_by_class[key]])
        distances.sort()
        distances = distances[::-1]
        for i in range(k):
            k_largest_dist[key].append(distances[i])
            
    # compute the harmonic range
    for key in keys:
        harmonic_mean_by_class[key] = np.sum([1.0 / k_largest_dist[key][i] for i in range(k)])
        
    # compute total intra class loss
    intra_class_loss = k * np.sum([1.0 / harmonic_mean_by_class[key] for key in keys])
    
    # compute intra class gradient???
    
    
    # compute shortest distances between all feature centers
    inter_class_dist = np.array([np.linalg.norm(np.array(average_by_class[key1])-np.array(average_by_class[key2])) for key1 in keys for key2 in keys if key1 != key2])
    inter_class_dist.sort()
    
    #compute inter-class loss
    inter_class_loss = m - inter_class_dist[0]
    if inter_class_loss < 0:
        inter_class_loss = 0
    
    # compute the inter-class gradient???
    if inter_class_loss > 0:
        inter_class_grad = np.zeros(x.shape[0])
    else:
        inter_class_grad = np.zeros(x.shape[0])
    
    # combine the loss and gradients
    loss = alpha * inter_class_loss + beta * intra_class_loss
    gradients = alpha * inter_class_grad + beta * intra_class_grad
    
    #print("inter-class loss: ", inter_class_loss)
    #print("intra-class loss: ", intra_class_loss)
    #print("inter-class gradients: ", inter_class_grad)
    #print("intra-class gradients: ", intra_class_grad)
    
    #print("data by class: ", data_by_class)
    #print("averages by class: ", average_by_class)
    #print("k largest distances per class: ", k_largest_dist)
    #print("harmonic means by class: ", harmonic_mean_by_class)
    #print("inter-class distances: ", inter_class_dist)
    
    return loss
    


# In[27]:


result_test = rangeloss(X_test, labels_test, m_test, w_test)
print(result_test)


# In[ ]:




