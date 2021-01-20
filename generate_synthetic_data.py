#!/usr/bin/env python
# coding: utf-8

''' 
This file was token from https://github.com/mbilalzafar/fair-classification/tree/master/disparate_impact/synthetic_data_demo
Credit to Muhammad Bilal Zafar
'''

import math
import numpy as np
import matplotlib.pyplot as plt 
from random import seed, shuffle
from scipy.stats import multivariate_normal # generating synthetic data


def generate_synthetic_data(SEED, param, plot_data=False):

    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).
    """
    
    seed(SEED) # set the random seed so that the random permutations can be reproduced again
    np.random.seed(SEED)

    n_samples = 1000 # generate these many data points per class
    disc_factor = param # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination

    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv,X,y

    """ Generate the non-sensitive features randomly """
    # We will generate one gaussian cluster for each class
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2,-2], [[10, 1], [1, 3]]
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1) # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1) # negative class

    # join the posisitve and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = list(range(0, n_samples*2))
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    
    rotation_mult = np.array([[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
    X_aux = np.dot(X, rotation_mult)

    """ Generate the sensitive feature here """
    x_control = [] # this array holds the sensitive feature value
    for i in range (0, len(X)):
        x = X_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)
        
        # normalize the probabilities from 0 to 1
        s = p1+p2
        p1 = p1/s
        p2 = p2/s
        
        r = np.random.uniform() # generate a random number from 0 to 1

        if r < p1: # the first cluster is the positive class
            x_control.append(1.0) # 1.0 means its male
        else:
            x_control.append(0.0) # 0.0 -> female

    x_control = np.array(x_control)

#     """ Show the data """
#     if plot_data:
#         num_to_draw = 200 # we will only draw a small number of points to avoid clutter
#         x_draw = X[:num_to_draw]
#         y_draw = y[:num_to_draw]
#         x_control_draw = x_control[:num_to_draw]

#         X_s_0 = x_draw[x_control_draw == 0.0]
#         X_s_1 = x_draw[x_control_draw == 1.0]
#         y_s_0 = y_draw[x_control_draw == 0.0]
#         y_s_1 = y_draw[x_control_draw == 1.0]
#         plt.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=30, linewidth=1.5, label= "Prot. +ve")
#         plt.scatter(X_s_0[y_s_0==-1.0][:, 0], X_s_0[y_s_0==-1.0][:, 1], color='red', marker='x', s=30, linewidth=1.5, label = "Prot. -ve")
#         plt.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='green', marker='o', facecolors='none', s=30, label = "Non-prot. +ve")
#         plt.scatter(X_s_1[y_s_1==-1.0][:, 0], X_s_1[y_s_1==-1.0][:, 1], color='red', marker='o', facecolors='none', s=30, label = "Non-prot. -ve")

        
#         plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
#         plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
#         plt.legend(loc=2, fontsize=15)
#         plt.xlim((-15,10))
#         plt.ylim((-10,15))
#         plt.savefig("testData_img/test_data.png")
#         plt.show()

    x_control = {"s1": x_control} # all the sensitive features are stored in a dictionary
    
    ## Save data to txt
    full_data = np.zeros([n_samples*2, 4])
    full_data[:, 0] = y
    full_data[:, 1:3] = X
    full_data[:, 3] = list(x_control.values())[0]
    np.savetxt("data/testData_seed%s_param%s.txt"%(SEED, param), full_data, fmt='%.6f')
    
    return X, y, x_control

