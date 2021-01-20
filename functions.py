#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# Biobjective formulation of Synthetic data: consider single binary sensitive attribute, minimize prediction loss and disparate impact, including info of functions and gradients

class Fairness_LogRe:
    setting = "finite_sum"
    projection = 0
    num_group = 2
    name = "Fairness_LogRe"
    lb = -2
    ub = 2
    m = 2
    
    
    def __init__(self, file_path, dataset, split):
        self.data = np.loadtxt(file_path)
        
        ## add intercept
        self.num_data, self.dim_prob = self.data.shape
        self.n = self.dim_prob - 1
        extened_data = np.zeros([self.num_data, self.dim_prob + 1])
        extened_data[:, :self.dim_prob] = self.data
        extened_data[:, self.dim_prob] = np.ones(self.num_data)
        self.data = extened_data
        self.dim_prob = self.dim_prob + 1
        
        self.data_name = dataset
        self.lambda_ = 1.0/1000
        self.split = split
        self.idx = np.delete(np.arange(self.dim_prob), [0, split])
        print ('Sensitive feature index: ', self.split)
        print ("#Training data size: ", self.num_data)
        
        # compute z bar
        self.zbar = np.sum(self.data[:, split])*1.0/self.num_data
        print ('Number of positive ones: ', np.sum(self.data[:, split]))
        
        # data size for function value evaluation
        self.eval_size = self.num_data
        
    ## logistic loss only
    def loss(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+np.exp(- data1[:,0].reshape(1, sizek)*np.matmul(x, A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek
        return f1   
    
    ## first objective is to minimize logistic regression loss
    def f1(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+np.exp(-data1[:,0].reshape(1,sizek)*np.matmul(x,A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek + self.lambda_/2*np.linalg.norm(x,axis = 1)**2
        return f1
    
    ## second objective is to minimize the fairness
    def f2(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_each = 1.0/sizek*np.matmul(x,A.T)*(data1[:,self.split] - self.zbar).reshape(1, sizek)
        f2 = np.sum(f2_each,axis = 1)
        f2 = f2**2
        return f2
    
    ## stochastic gradients for f1: k is the random index set
    def g1(self, x, k=[1]):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        part1 = - np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T))/(1 + np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T)))
        part2 = data1[:, 0].reshape(1,sizek)*A.T
        g1_each = part1.reshape(1,1,sizek)*part2.reshape(1,self.n,sizek)
        g1 = np.sum(g1_each,axis = 2)
        g1 = g1/sizek + self.lambda_*x
        return g1[0]
    
    ## stochastic gradients for f2: use a mini-batch
    def g2(self, x, k=range(10)):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        g2_each = (2.0/sizek)*(data1[:,self.split]- self.zbar).reshape(1,sizek)*np.matmul(x, A.T)
        g3_each = (1.0/sizek)*(data1[:, self.split] - self.zbar).reshape(sizek,1)*A
        g23 = (np.sum(g2_each,axis = 1).reshape(1,1)) * (np.sum(g3_each,axis = 0).reshape(1,self.n))
        return g23[0]
    
    ## compute prediction accuracy for a single solution
    def predict_accuracy(self, x):        
        num_group = self.num_group
        count_group = np.zeros(num_group).astype(float)
        sum_correct = np.zeros(num_group).astype(float)
        sum_one = np.zeros(num_group).astype(float)
        sum_zero = np.zeros(num_group).astype(float)
        sum_FPR = np.zeros(num_group).astype(float)
        sum_FNR = np.zeros(num_group).astype(float)
        Accuracy = np.zeros(num_group).astype(float)
                
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i] # split to group
            count_group[i] = len(A)
            target = A[:, 0]
            APos = A[A[:, 0] == 1]
            ANeg = A[A[:, 0] == -1]
            A = A[:, self.idx]
            
            sum_correct[i] =  np.sum(target*np.matmul(x, A.T) >= 0)
            sum_one[i] = np.sum(np.matmul(x, A.T) >= 0)
            sum_zero[i] = count_group[i] - sum_one[i]            
            
            sum_FPR[i] = np.sum(np.matmul(x, APos[:, self.idx].T) < 0)
            sum_FNR[i] = np.sum(np.matmul(x, ANeg[:, self.idx].T) > 0)
            
        Accuracy = sum_correct/count_group
        FPR = sum_FPR/sum_one
        FNR = sum_FNR/sum_zero
        total_accuracy = np.sum(sum_correct)*1.0/self.num_data
        
        return total_accuracy, Accuracy.reshape(1, num_group), FPR.reshape(1, num_group), FNR.reshape(1, num_group)
    
    ## compute CV score for a single solution
    def disparate_impact(self, x):
        num_group = self.num_group
        sum_positive = np.zeros(num_group).astype(float)
        count_group = np.zeros(num_group).astype(float)
        
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i]
            A = A[:, self.idx]
            count_group[i] = len(A)
            sum_positive[i] = np.sum(np.matmul(x, A.T) >= 0)
                           
        ratio = sum_positive/count_group
        CV = np.max(ratio) - np.min(ratio)
        pvalue = np.min(ratio)/np.max(ratio)
        return ratio, CV, pvalue

    def compute_accuracy(self, list_f1, list_f2, list_pts, num_pts):  
        '''
        This function is to compute training/testing loss/accuracy, disparate impact for a number of nondominated solutions.
        Input
        list_f1: The list of first objective function values
        list_f2: The list of second objective function values
        list_pts: The list of nondominated solutions
        num_pts: Number of solutions to be measure
        Output
        disparate_impact: CV scores
        percentage: Demographic decomposition in positive prediction class
        pvalue: percent of minorty group of positive prediction/percent of majority group of positive prediction. when pvalue is greater 0.8, we say it fair. 
        total_accuracy: Training accuracy of the entire dataset
        training_accuracy: Training accuracy of each demograpic group
        training_FPR: FPR of each demograpic group
        training_FNR: FNR of each demograpic group
        training_loss: Training loss of the entire dataset
        training_obj1: Regularized training loss of the entire dataset
        training_obj2: Second objective value, i.e., square of covariance
        '''
        
        temp = np.copy(list_f1)
        sort_acc_index = np.argsort(temp)
        idx = range(0, len(list_f1), int(math.floor(len(list_f1)/num_pts)))
        sort_index = sort_acc_index[idx]
        num_pts = len(sort_index)
        
        num_group = self.num_group
        total_accuracy = np.zeros(num_pts)
        training_accuracy = np.zeros([num_pts, num_group])
        training_FPR = np.zeros([num_pts, num_group])
        training_FNR = np.zeros([num_pts, num_group])
        disparate_impact = np.zeros(num_pts)
        percentage = np.zeros([num_pts, num_group])
        pvalue = np.zeros(num_pts)
        
        training_loss = self.loss(list_pts[sort_index])
        training_obj1 = self.f1(list_pts[sort_index])
        training_obj2 = self.f2(list_pts[sort_index])
        
        for i in range(num_pts):   
            total_accuracy[i], training_accuracy[i, :], training_FPR[i, :], training_FNR[i, :] = self.predict_accuracy(list_pts[sort_index[i]])
            percentage[i, :], disparate_impact[i], pvalue[i] = self.disparate_impact(list_pts[sort_index[i]])

        return disparate_impact, percentage, pvalue, total_accuracy, training_accuracy, training_FPR, training_FNR,                training_loss, training_obj1, training_obj2


# Biobjective formulation: works for any dataset with single binary sensitive attribute, minimize prediction loss and disparate impact, including info of functions and gradients

class Fairness_LogRe_DI_binary:
    setting = "finite_sum"
    projection = 0
    name = "Fairness_LogRe"
    lb = - 5
    ub = 5
    m = 2
    num_group = 2
    
    
    def __init__(self, file_path, dataset, split, SEED, NumData, train_or_test):
        self.Alldata = np.loadtxt(file_path)
        NUM_Alldata = len(self.Alldata)
        np.random.seed(SEED)
        
        if train_or_test == "train":
            w = np.random.choice(NUM_Alldata, NumData, replace=False)
            self.data = self.Alldata[w, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Training data size: ", self.num_data)
        elif train_or_test == "train_ds": # downsample the training data to make two sensitive groups have equal size
            
            g1_idx = np.where(self.Alldata[:, split] == 0)[0]
            g2_idx = np.where(self.Alldata[:, split] == 1)[0]
            w_g1 = np.random.choice(g1_idx, int(NumData*0.5), replace=False)
            w_g2 = np.random.choice(g2_idx, int(NumData*0.5), replace=False)
            w = np.concatenate((w_g1, w_g2))
            
            self.data = self.Alldata[w, :]
            print ("#high-income Female", len(np.where((self.data[:, split] == 0) & (self.data[:, 0] == 1))[0]))
            print ("#high-income Male", len(np.where((self.data[:, split] == 1) & (self.data[:, 0] == 1))[0]))
            self.num_data, self.dim_prob = self.data.shape
            print ("#Training data size: ", self.num_data)
        elif train_or_test == "test_ds": # downsample the testing data to make two sensitive groups have equal size
            g1_idx = np.where(self.Alldata[:, split] == 0)[0]
            g2_idx = np.where(self.Alldata[:, split] == 1)[0]
            w_g1 = np.random.choice(g1_idx, int(NumData*0.5), replace=False)
            w_g2 = np.random.choice(g2_idx, int(NumData*0.5), replace=False)
            w = np.concatenate((w_g1, w_g2))
            
            test_data_idx = np.delete(np.arange(NUM_Alldata), w)
            self.data = self.Alldata[test_data_idx, :]
            
            print ("#high-income Female", len(np.where((self.data[:, split] == 0) & (self.data[:, 0] == 1))[0]))
            print ("#high-income Male", len(np.where((self.data[:, split] == 1) & (self.data[:, 0] == 1))[0]))
            self.num_data, self.dim_prob = self.data.shape
            print ("# Testing data size: ", self.num_data)
        elif train_or_test == "test":
            w = np.random.choice(NUM_Alldata, NumData, replace=False)
            test_data_idx = np.delete(np.arange(NUM_Alldata), w)
            self.data = self.Alldata[test_data_idx, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("# Testing data size: ", self.num_data)
        else:
            self.data = self.Alldata
            self.num_data, self.dim_prob = self.data.shape
            print ("# All data size: ", self.num_data)
        
        self.data_name = dataset
        self.n = self.dim_prob - 2
        self.lambda_ = 1.0/1000
        self.split = split
        self.idx = np.delete(np.arange(self.dim_prob), [0, split])
        print ('split', self.split)
        print ('idx', self.idx)
        
        # compute z bar
        self.zbar = np.sum(self.data[:, split])*1.0/self.num_data
        print ('sum of sensitive', np.sum(self.data[:, split]))
        
        # data size for function value evaluation
        self.eval_size = self.num_data 
         
    
    ## logistic regression loss only
    def loss(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+np.exp(- data1[:,0].reshape(1, sizek)*np.matmul(x, A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek
        return f1   
    
    ## first objective is to minimize reguarized logistic regression loss
    def f1(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+ np.exp(-data1[:,0].reshape(1,sizek)*np.matmul(x,A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek + self.lambda_/2*np.linalg.norm(x,axis = 1)**2
        return f1
    
    ## second objective is to minimize the fairness w.r.t. Disparate impact
    def f2(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_each = 1.0/sizek*np.matmul(x,A.T)*(data1[:, self.split] - self.zbar).reshape(1, sizek)
        f2 = np.sum(f2_each,axis = 1)
        f2 = f2**2
        return f2
    
    ## stochastic gradients for f1: k is the random index set
    def g1(self, x, k=[1]):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        part1 = - np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T))/(1 + np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T)))
        part2 = data1[:, 0].reshape(1,sizek)*A.T
        g1_each = part1.reshape(1,1,sizek)*part2.reshape(1,self.n,sizek)
        g1 = np.sum(g1_each,axis = 2)
        g1 = g1/sizek + self.lambda_*x
        return g1[0]
    
    ## stochastic gradients for f2: use a mini-batch
    def g2(self, x, k=range(10)): 
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        g2_each = (2.0/sizek)*(data1[:,self.split]- self.zbar).reshape(1,sizek)*np.matmul(x, A.T)
        g3_each = (1.0/sizek)*(data1[:, self.split] - self.zbar).reshape(sizek,1)*A
        g23 = (np.sum(g2_each,axis = 1).reshape(1,1)) * (np.sum(g3_each,axis = 0).reshape(1,self.n))
        return g23[0]
        
    def predict_accuracy(self, x):  
        num_group = self.num_group
        count_group = np.zeros(num_group).astype(float)
        sum_correct = np.zeros(num_group).astype(float)
        sum_one = np.zeros(num_group).astype(float)
        sum_zero = np.zeros(num_group).astype(float)
        sum_FPR = np.zeros(num_group).astype(float)
        sum_FNR = np.zeros(num_group).astype(float)
        Accuracy = np.zeros(num_group).astype(float)
                
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i] # split to group
            count_group[i] = len(A)
            target = A[:, 0]
            APos = A[A[:, 0] == 1]
            ANeg = A[A[:, 0] == -1]
            A = A[:, self.idx]
            
            sum_correct[i] =  np.sum(target*np.matmul(x, A.T) >= 0)
            sum_one[i] = np.sum(np.matmul(x, A.T) >= 0)
            sum_zero[i] = count_group[i] - sum_one[i]            
            
            sum_FPR[i] = np.sum(np.matmul(x, ANeg[:, self.idx].T) > 0)
            sum_FNR[i] = np.sum(np.matmul(x, APos[:, self.idx].T) < 0)
            
        Accuracy = sum_correct/count_group
        FPR = sum_FPR/sum_zero
        FNR = sum_FNR/sum_one
        total_accuracy = np.sum(sum_correct)*1.0/self.num_data
        
        return total_accuracy, Accuracy.reshape(1, num_group), FPR.reshape(1, num_group), FNR.reshape(1, num_group)
    
    
    def disparate_impact(self, x):
        num_group = 2
        sum_positive = np.zeros(num_group).astype(float)
        count_group = np.zeros(num_group).astype(float)
        
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i]
            A = A[:, self.idx]
            count_group[i] = len(A)
            sum_positive[i] = np.sum(np.matmul(x, A.T) >= 0)
                           
        ratio = sum_positive/count_group
        CV = np.max(ratio) - np.min(ratio)
        pvalue = np.min(ratio)/np.max(ratio)
        return ratio, CV, pvalue
    
    def compute_accuracy(self, list_f1, list_f2, list_pts, num_pts): 
        '''
        This function is to compute training/testing loss/accuracy, disparate impact for a number of nondominated solutions.
        Input
        list_f1: The list of first objective function values
        list_f2: The list of second objective function values
        list_pts: The list of nondominated solutions
        num_pts: Number of solutions to be measure
        Output
        disparate_impact: CV scores
        percentage: Demographic decomposition in positive prediction class
        pvalue: percent of minorty group of positive prediction/percent of majority group of positive prediction. when pvalue is greater 0.8, we say it fair. 
        total_accuracy: Training accuracy of the entire dataset
        training_accuracy: Training accuracy of each demograpic group
        training_FPR: FPR of each demograpic group
        training_FNR: FNR of each demograpic group
        training_loss: Training loss of the entire dataset
        training_obj1: Regularized training loss of the entire dataset
        training_obj2: Second objective value, i.e., square of covariance
        '''
        ## get the set of indices according to which the disparate_impact   
        temp = np.copy(list_f1)
        sort_acc_index = np.argsort(temp)
        idx = range(0, len(list_f1), int(math.floor(len(list_f1)/num_pts)))
        sort_index = sort_acc_index[idx]
        num_pts = len(sort_index)
        
        num_group = 2
        total_accuracy = np.zeros(num_pts)
        training_accuracy = np.zeros([num_pts, num_group])
        training_FPR = np.zeros([num_pts, num_group])
        training_FNR = np.zeros([num_pts, num_group])
        disparate_impact = np.zeros(num_pts)
        percentage = np.zeros([num_pts, num_group])
        pvalue = np.zeros(num_pts)
        
        ## Always evaluate each objectives using the whole set of datas
        training_loss = self.loss(list_pts[sort_index])
        training_obj1 = self.f1(list_pts[sort_index])
        training_obj2 = self.f2(list_pts[sort_index])
        
        for i in range(num_pts):   
            total_accuracy[i], training_accuracy[i, :], training_FPR[i, :], training_FNR[i, :] = self.predict_accuracy(list_pts[sort_index[i]])
            percentage[i, :], disparate_impact[i], pvalue[i] = self.disparate_impact(list_pts[sort_index[i]])

        return disparate_impact, percentage, pvalue, total_accuracy, training_accuracy, training_FPR, training_FNR,                training_loss, training_obj1, training_obj2
   
   

# Biobjective formulation: works for any dataset with multi-valued categorical sensitive attribute, minimize prediction loss and disparate impact, including info of functions and gradients

class Fairness_LogRe_DI_multi:
    setting = "finite_sum"
    projection = 0
    name = "Fairness_LogRe"
    lb = -5
    ub = 5
    m = 2
    num_group = 5
    
    
    def __init__(self, file_path, dataset, split, SEED, NumData, train_or_test):
        self.Alldata = np.loadtxt(file_path)
        NUM_Alldata = len(self.Alldata)
        
        np.random.seed(SEED)
        w = np.random.choice(NUM_Alldata, NumData, replace=False)
        
        if train_or_test == "train":
            self.data = self.Alldata[w, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Training data size: ", self.num_data)
        else: 
            test_data_idx = np.delete(np.arange(NUM_Alldata), w)
            self.data = self.Alldata[test_data_idx, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("# Testing data size: ", self.num_data)
        
        
        self.data_name = dataset
        self.n = self.dim_prob - self.num_group - 1
        self.lambda_ = 1.0/1000
        self.split = split
        self.idx = np.delete(np.arange(self.dim_prob), [0] + split)
        print ('Sensitive feature index: ', self.split)
        
        # compute z bar
        self.zbar = np.mean(self.data[:, split], axis=0)
        print ('Mean of sensitive values: ', self.zbar)
        
        # data size for function value evaluation
        self.eval_size = self.num_data 
    
    ## logistic loss only
    def loss(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+np.exp(- data1[:,0].reshape(1, sizek)*np.matmul(x, A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek
        return f1   
    
    ## first objective is to minimize logistic regression loss
    def f1(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+ np.exp(-data1[:,0].reshape(1,sizek)*np.matmul(x,A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek + self.lambda_/2*np.linalg.norm(x,axis = 1)**2
        return f1

    ## second objective is to minimize the fairness w.r.t. Disparate impact
    def f2(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_each = 1.0/sizek*np.matmul(np.matmul(x, A.T), (data1[:, self.split] - self.zbar))
        f2 = np.max(f2_each**2, axis = 1)
        return f2
    
    def f2_index(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_each = 1.0/sizek*np.matmul(np.matmul(x, A.T), (data1[:, self.split] - self.zbar))
        idx = np.argmax(f2_each**2)
        return idx
    
    ## stochastic gradients: k is the random index set
    def g1(self, x, k=[1]):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        part1 = - np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T))/(1 + np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T)))
        part2 = data1[:, 0].reshape(1,sizek)*A.T
        g1_each = part1.reshape(1,1,sizek)*part2.reshape(1,self.n,sizek)
        g1 = np.sum(g1_each,axis = 2)
        g1 = g1/sizek #+ self.lambda_*x
        return g1[0]
    
    ## stochastic gradients for f2: use a mini-batch
    def g2(self, x, k=range(10)): 
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        idx = self.f2_index(x)
        g2_each = (2.0/sizek)*np.matmul(np.matmul(x, A.T), (data1[:,self.split[idx]] - self.zbar[idx]))
        g3_each = (1.0/sizek)*np.matmul((data1[:, self.split[idx]] - self.zbar[idx]).T, A)
        g23 = g2_each*g3_each.reshape(1, self.n)
        return g23[0]
        
    def predict_accuracy(self, x):  
        num_group = self.num_group
        count_group = np.zeros(num_group).astype(float)
        sum_correct = np.zeros(num_group).astype(float)
        sum_one = np.zeros(num_group).astype(float)
        sum_zero = np.zeros(num_group).astype(float)
        sum_FPR = np.zeros(num_group).astype(float)
        sum_FNR = np.zeros(num_group).astype(float)
        Accuracy = np.zeros(num_group).astype(float)
                
        for i in range(num_group):
            A = self.data[self.data[:, self.split[i]] == 1] # split to group
            count_group[i] = len(A)
            target = A[:, 0]
            APos = A[A[:, 0] == 1]
            ANeg = A[A[:, 0] == -1]
            A = A[:, self.idx]
            
            sum_correct[i] =  np.sum(target*np.matmul(x, A.T) >= 0)
            sum_one[i] = len(APos)
            sum_zero[i] = len(ANeg)            
            
            sum_FPR[i] = np.sum(np.matmul(x, ANeg[:, self.idx].T) > 0)
            sum_FNR[i] = np.sum(np.matmul(x, APos[:, self.idx].T) < 0)
        
        Accuracy = sum_correct/count_group
        FPR = sum_FPR/sum_zero
        FNR = sum_FNR/sum_one
        total_accuracy = np.sum(sum_correct)*1.0/self.num_data
        
        return total_accuracy, Accuracy.reshape(1, num_group), FPR.reshape(1, num_group), FNR.reshape(1, num_group)
    
    
    def disparate_impact(self, x):
        num_group = self.num_group
        sum_positive = np.zeros([num_group,2]).astype(float)
        count_group = np.zeros([num_group,2]).astype(float)
        
        for i in range(num_group):
            A = self.data[self.data[:, self.split[i]] == 1]
            A = A[:, self.idx]
            count_group[i, 0] = len(A)
            sum_positive[i, 0] = np.sum(np.matmul(x, A.T) >= 0)
            
            A = self.data[self.data[:, self.split[i]] == 0]
            A = A[:, self.idx]
            count_group[i, 1] = len(A)
            sum_positive[i, 1] = np.sum(np.matmul(x, A.T) >= 0)
                           
        ratio = sum_positive/count_group
        CV = np.max(np.max(ratio, axis = 1) - np.min(ratio, axis = 1))
        pvalue = np.min(ratio)/np.max(ratio)
        return ratio[:, 0], CV, pvalue
    
    def compute_accuracy(self, list_f1, list_f2, list_pts, num_pts): 
        '''
        This function is to compute training/testing loss/accuracy, disparate impact for a number of nondominated solutions.
        Input
        list_f1: The list of first objective function values
        list_f2: The list of second objective function values
        list_pts: The list of nondominated solutions
        num_pts: Number of solutions to be measure
        Output
        disparate_impact: CV scores
        percentage: Demographic decomposition in positive prediction class
        pvalue: percent of minorty group of positive prediction/percent of majority group of positive prediction. when pvalue is greater 0.8, we say it fair. 
        total_accuracy: Training accuracy of the entire dataset
        training_accuracy: Training accuracy of each demograpic group
        training_FPR: FPR of each demograpic group
        training_FNR: FNR of each demograpic group
        training_loss: Training loss of the entire dataset
        training_obj1: Regularized training loss of the entire dataset
        training_obj2: Second objective value, i.e., square of covariance
        '''
        temp = np.copy(list_f1)
        sort_acc_index = np.argsort(temp)
        idx = range(0, len(list_f1), int(math.floor(len(list_f1)/num_pts)))
        sort_index = sort_acc_index[idx]
        num_pts = len(sort_index)
        
        num_group = self.num_group
        total_accuracy = np.zeros(num_pts)
        training_accuracy = np.zeros([num_pts, num_group])
        training_FPR = np.zeros([num_pts, num_group])
        training_FNR = np.zeros([num_pts, num_group])
        disparate_impact = np.zeros(num_pts)
        percentage = np.zeros([num_pts, num_group])
        pvalue = np.zeros(num_pts)
        
        training_loss = self.loss(list_pts[sort_index])
        training_obj1 = self.f1(list_pts[sort_index])
        training_obj2 = self.f2(list_pts[sort_index])
        
        for i in range(num_pts):   
            total_accuracy[i], training_accuracy[i, :], training_FPR[i, :], training_FNR[i, :] = self.predict_accuracy(list_pts[sort_index[i]])
            percentage[i, :], disparate_impact[i], pvalue[i] = self.disparate_impact(list_pts[sort_index[i]])

        return disparate_impact, percentage, pvalue, total_accuracy, training_accuracy, training_FPR, training_FNR,                training_loss, training_obj1, training_obj2
    
    # Biobjective formulation: works for any dataset with multi-valued categorical sensitive attribute, minimize prediction loss and smoothed approximated disparate impact, including info of functions and gradients

class Fairness_LogRe_DI_multi_smoothed:
    setting = "finite_sum"
    projection = 0
    name = "Fairness_LogRe"
    lb = -5
    ub = 5
    m = 2
    num_group = 5
    
    
    def __init__(self, file_path, dataset, split, SEED, NumData, train_or_test):
        self.Alldata = np.loadtxt(file_path)
        NUM_Alldata = len(self.Alldata)
        
        np.random.seed(SEED)
        w = np.random.choice(NUM_Alldata, NumData, replace=False)
        
        if train_or_test == "train":
            self.data = self.Alldata[w, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Training data size: ", self.num_data)
        else: 
            test_data_idx = np.delete(np.arange(NUM_Alldata), w)
            self.data = self.Alldata[test_data_idx, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("# Testing data size: ", self.num_data)
        
        
        self.data_name = dataset
        self.n = self.dim_prob - self.num_group - 1
        self.lambda_ = 1.0/1000
        self.split = split
        self.idx = np.delete(np.arange(self.dim_prob), [0] + split)
        print ('Sensitive feature index: ', self.split)
        
        # compute z bar
        self.zbar = np.mean(self.data[:, split], axis=0)
        print ('Mean of sensitive values: ', self.zbar)
        
        # data size for function value evaluation
        self.eval_size = self.num_data 
    
    ## logistic loss only
    def loss(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+np.exp(- data1[:,0].reshape(1, sizek)*np.matmul(x, A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek
        return f1   
    
    ## first objective is to minimize logistic regression loss
    def f1(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+ np.exp(-data1[:,0].reshape(1,sizek)*np.matmul(x,A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek + self.lambda_/2*np.linalg.norm(x,axis = 1)**2
        return f1

    ## second objective is to minimize the fairness w.r.t. Disparate impact
    def f2(self, x, k = np.array([]), alpha = 8.0):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_each = 1.0/sizek*np.matmul(np.matmul(x, A.T), (data1[:, self.split] - self.zbar))
        f2 = np.sum(f2_each**2*np.exp(alpha*f2_each**2), axis = 1)/np.sum(np.exp(alpha*f2_each**2), axis = 1)
        return f2
    
    ## second objective is to minimize the fairness w.r.t. Disparate impact
    def f2_true(self, x, k = np.array([]), alpha = 8.0):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_each = 1.0/sizek*np.matmul(np.matmul(x, A.T), (data1[:, self.split] - self.zbar))
        f2 = np.max(f2_each**2, axis = 1)
        return f2
    
    def f2_index(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_each = 1.0/sizek*np.matmul(np.matmul(x, A.T), (data1[:, self.split] - self.zbar))
        idx = np.argmax(f2_each**2)
        return idx
    
    ## stochastic gradients: k is the random index set
    def g1(self, x, k=[1]):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        part1 = - np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T))/(1 + np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T)))
        part2 = data1[:, 0].reshape(1,sizek)*A.T
        g1_each = part1.reshape(1,1,sizek)*part2.reshape(1,self.n,sizek)
        g1 = np.sum(g1_each,axis = 2)
        g1 = g1/sizek + self.lambda_*x
        return g1[0]
    
    ## stochastic gradients for f2: use a mini-batch
    def g2(self, x, k=range(10), alpha = 8.0): 
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        
        idx = self.f2_index(x)
        g2_each = (2.0/sizek)*np.matmul(np.matmul(x, A.T), (data1[:,self.split] - self.zbar))
        g3_each = (1.0/sizek)*np.matmul((data1[:, self.split] - self.zbar).T, A)
        g23 = g2_each.reshape(-1, 1)*g3_each
        
        f2_each = 1.0/sizek*np.matmul(np.matmul(x, A.T), (data1[:, self.split] - self.zbar))
        f2 = np.sum(f2_each**2*np.exp(alpha*f2_each**2))/np.sum(np.exp(alpha*f2_each**2))
        g2 = np.matmul((np.exp(alpha*f2_each**2)*(1 + alpha*(f2_each**2 - f2))).reshape(1, -1), g23)/np.sum(np.exp(alpha*f2_each**2))
        return g2[0]
        
    def predict_accuracy(self, x):  
        num_group = self.num_group
        count_group = np.zeros(num_group).astype(float)
        sum_correct = np.zeros(num_group).astype(float)
        sum_one = np.zeros(num_group).astype(float)
        sum_zero = np.zeros(num_group).astype(float)
        sum_FPR = np.zeros(num_group).astype(float)
        sum_FNR = np.zeros(num_group).astype(float)
        Accuracy = np.zeros(num_group).astype(float)
                
        for i in range(num_group):
            A = self.data[self.data[:, self.split[i]] == 1] # split to group
            count_group[i] = len(A)
            target = A[:, 0]
            APos = A[A[:, 0] == 1]
            ANeg = A[A[:, 0] == -1]
            A = A[:, self.idx]
            
            sum_correct[i] =  np.sum(target*np.matmul(x, A.T) >= 0)
            sum_one[i] = len(APos)
            sum_zero[i] = len(ANeg)            
            
            sum_FPR[i] = np.sum(np.matmul(x, ANeg[:, self.idx].T) > 0)
            sum_FNR[i] = np.sum(np.matmul(x, APos[:, self.idx].T) < 0)
        
        Accuracy = sum_correct/count_group
        FPR = sum_FPR/sum_zero
        FNR = sum_FNR/sum_one
        total_accuracy = np.sum(sum_correct)*1.0/self.num_data
        
        return total_accuracy, Accuracy.reshape(1, num_group), FPR.reshape(1, num_group), FNR.reshape(1, num_group)
    
    
    def disparate_impact(self, x):
        num_group = self.num_group
        sum_positive = np.zeros([num_group,2]).astype(float)
        count_group = np.zeros([num_group,2]).astype(float)
        
        for i in range(num_group):
            A = self.data[self.data[:, self.split[i]] == 1]
            A = A[:, self.idx]
            count_group[i, 0] = len(A)
            sum_positive[i, 0] = np.sum(np.matmul(x, A.T) >= 0)
            
            A = self.data[self.data[:, self.split[i]] == 0]
            A = A[:, self.idx]
            count_group[i, 1] = len(A)
            sum_positive[i, 1] = np.sum(np.matmul(x, A.T) >= 0)
                           
        ratio = sum_positive/count_group
        CV = np.max(np.max(ratio, axis = 1) - np.min(ratio, axis = 1))
        pvalue = np.min(ratio)/np.max(ratio)
        return ratio[:, 0], CV, pvalue
    
    def compute_accuracy(self, list_f1, list_f2, list_pts, num_pts):
        '''
        This function is to compute training/testing loss/accuracy, disparate impact for a number of nondominated solutions.
        Input
        list_f1: The list of first objective function values
        list_f2: The list of second objective function values
        list_pts: The list of nondominated solutions
        num_pts: Number of solutions to be measure
        Output
        disparate_impact: CV scores
        percentage: Demographic decomposition in positive prediction class
        pvalue: percent of minorty group of positive prediction/percent of majority group of positive prediction. when pvalue is greater 0.8, we say it fair. 
        total_accuracy: Training accuracy of the entire dataset
        training_accuracy: Training accuracy of each demograpic group
        training_FPR: FPR of each demograpic group
        training_FNR: FNR of each demograpic group
        training_loss: Training loss of the entire dataset
        training_obj1: Regularized training loss of the entire dataset
        training_obj2: Second objective value, i.e., square of covariance
        '''
        temp = np.copy(list_f1)
        sort_acc_index = np.argsort(temp)
        idx = range(0, len(list_f1), int(math.floor(len(list_f1)/num_pts)))
        sort_index = sort_acc_index[idx]
        num_pts = len(sort_index)
        
        num_group = self.num_group
        total_accuracy = np.zeros(num_pts)
        training_accuracy = np.zeros([num_pts, num_group])
        training_FPR = np.zeros([num_pts, num_group])
        training_FNR = np.zeros([num_pts, num_group])
        disparate_impact = np.zeros(num_pts)
        percentage = np.zeros([num_pts, num_group])
        pvalue = np.zeros(num_pts)
        
        training_loss = self.loss(list_pts[sort_index])
        training_obj1 = self.f1(list_pts[sort_index])
        training_obj2 = self.f2(list_pts[sort_index])
        
        for i in range(num_pts):   
            total_accuracy[i], training_accuracy[i, :], training_FPR[i, :], training_FNR[i, :] = self.predict_accuracy(list_pts[sort_index[i]])
            percentage[i, :], disparate_impact[i], pvalue[i] = self.disparate_impact(list_pts[sort_index[i]])

        return disparate_impact, percentage, pvalue, total_accuracy, training_accuracy, training_FPR, training_FNR, training_loss, training_obj1, training_obj2
    
    
    # Three-objective formulation: handle both binary sensitive attribute and multi-valued categorical sensitive attribute, minimize prediction loss and disparate impact for two different sensitive attributes, including info of functions and gradients

class Fairness_LogRe_DI_multi_attributes:
    setting = "finite_sum"
    projection = 0
    name = "Fairness_LogRe"
    lb = -5
    ub = 5
    m = 3
    num_group_attr1 = 5
    num_group_attr2 = 2
    
    
    def __init__(self, file_path, dataset, split1, split2, SEED, NumData, train_or_test):
        self.Alldata = np.loadtxt(file_path)
        NUM_Alldata = len(self.Alldata)
        
        np.random.seed(SEED)
        w = np.random.choice(NUM_Alldata, NumData, replace=False)
        
        if train_or_test == "train":
            self.data = self.Alldata[w, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Training data size: ", self.num_data)
        else: 
            test_data_idx = np.delete(np.arange(NUM_Alldata), w)
            print (len(test_data_idx), NUM_Alldata)
            self.data = self.Alldata[test_data_idx, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("# Testing data size: ", self.num_data)
        
        
        self.data_name = dataset
        self.n = self.dim_prob - self.num_group_attr1 - self.num_group_attr2
        self.lambda_ = 1.0/1000
        self.split1 = split1
        self.split2 = split2[0]
        self.idx = np.delete(np.arange(self.dim_prob), [0] + split1 + split2)
        print ('Sensitive feature index: ', split1 + split2)
        
        # compute z bar
        self.zbar1 = np.mean(self.data[:, self.split1], axis=0)
        self.zbar2 = np.mean(self.data[:, self.split2], axis=0)
        print ('Mean of sensitive values for 1st attribute: ', self.zbar1)
        print ('Mean of sensitive values for 2nd attribute: ', self.zbar2)
        
        # data size for function value evaluation
        self.eval_size = self.num_data 
    
    ## logistic loss only
    def loss(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+np.exp(- data1[:,0].reshape(1, sizek)*np.matmul(x, A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek
        return f1   
    
    ## first objective is to minimize logistic regression loss
    def f1(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+ np.exp(-data1[:,0].reshape(1,sizek)*np.matmul(x,A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek + self.lambda_/2*np.linalg.norm(x,axis = 1)**2
        return f1

    ## second objective is to minimize the fairness w.r.t. Disparate impact for a multi-category sensitive attribute
    def f2(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_each = 1.0/sizek*np.matmul(np.matmul(x, A.T), (data1[:, self.split1] - self.zbar1))
        f2 = np.max(f2_each**2, axis = 1)
        return f2
    
    def f2_index(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_each = 1.0/sizek*np.matmul(np.matmul(x, A.T), (data1[:, self.split1] - self.zbar1))
        idx = np.argmax(f2_each**2)
        return idx
    
    ## third objective is to minimize the fairness w.r.t. Disparate impact for a binary sensitive attribute
    def f3(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f3_each = 1.0/sizek*np.matmul(x,A.T)*(data1[:, self.split2] - self.zbar2).reshape(1, sizek)
        f3 = np.sum(f3_each,axis = 1)
        f3 = f3**2
        return f3
    
    ## stochastic gradients: k is the random index set
    def g1(self, x, k=[1]):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        part1 = - np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T))/(1 + np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T)))
        part2 = data1[:, 0].reshape(1,sizek)*A.T
        g1_each = part1.reshape(1,1,sizek)*part2.reshape(1,self.n,sizek)
        g1 = np.sum(g1_each,axis = 2)
        g1 = g1/sizek + self.lambda_*x
        return g1[0]
    
    ## stochastic gradients for f2: use a mini-batch
    def g2(self, x, k=range(10)): 
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        idx = self.f2_index(x)
        g2_each = (2.0/sizek)*np.matmul(np.matmul(x, A.T), (data1[:,self.split1[idx]] - self.zbar1[idx]))
        g3_each = (1.0/sizek)*np.matmul((data1[:, self.split1[idx]] - self.zbar1[idx]).T, A)
        g23 = g2_each*g3_each.reshape(1, self.n)
        return g23[0]
    
    ## stochastic gradients for f3: use a mini-batch
    def g3(self, x, k=range(10)): 
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        g2_each = (2.0/sizek)*(data1[:, self.split2] - self.zbar2).reshape(1,sizek)*np.matmul(x, A.T)
        g3_each = (1.0/sizek)*(data1[:, self.split2] - self.zbar2).reshape(sizek,1)*A
        g23 = (np.sum(g2_each,axis = 1).reshape(1,1)) * (np.sum(g3_each,axis = 0).reshape(1,self.n))
        return g23[0]
        
    def predict_accuracy(self, x):  
        num_group = self.num_group_attr1 + self.num_group_attr2
        count_group = np.zeros(num_group).astype(float)
        sum_correct = np.zeros(num_group).astype(float)
        sum_one = np.zeros(num_group).astype(float)
        sum_zero = np.zeros(num_group).astype(float)
        sum_FPR = np.zeros(num_group).astype(float)
        sum_FNR = np.zeros(num_group).astype(float)
        Accuracy = np.zeros(num_group).astype(float)
                
        for i in range(self.num_group_attr1):
            A = self.data[self.data[:, self.split1[i]] == 1] # split to group
            count_group[i] = len(A)
            target = A[:, 0]
            APos = A[A[:, 0] == 1]
            ANeg = A[A[:, 0] == -1]
            A = A[:, self.idx]
            
            sum_correct[i] =  np.sum(target*np.matmul(x, A.T) >= 0)
            sum_one[i] = len(APos)
            sum_zero[i] = len(ANeg)            
            
            sum_FPR[i] = np.sum(np.matmul(x, ANeg[:, self.idx].T) > 0)
            sum_FNR[i] = np.sum(np.matmul(x, APos[:, self.idx].T) < 0)
            
        for i in range(self.num_group_attr2):
            A = self.data[self.data[:, self.split2] == i] # split to group
            i = self.num_group_attr1 + i
            count_group[i] = len(A)
            target = A[:, 0]
            APos = A[A[:, 0] == 1]
            ANeg = A[A[:, 0] == -1]
            A = A[:, self.idx]
            
            sum_correct[i] =  np.sum(target*np.matmul(x, A.T) >= 0)
            sum_one[i] = len(APos)
            sum_zero[i] = len(ANeg)            
            
            sum_FPR[i] = np.sum(np.matmul(x, ANeg[:, self.idx].T) > 0)
            sum_FNR[i] = np.sum(np.matmul(x, APos[:, self.idx].T) < 0)
        
        Accuracy = sum_correct/count_group
        FPR = sum_FPR/sum_zero
        FNR = sum_FNR/sum_one
        total_accuracy = np.sum(sum_correct)*1.0/self.num_data
        
        return total_accuracy, Accuracy.reshape(1, num_group), FPR.reshape(1, num_group), FNR.reshape(1, num_group)
    
    
    def disparate_impact(self, x):
        num_group = self.num_group_attr1 + self.num_group_attr2
        sum_positive = np.zeros(num_group).astype(float)
        count_group = np.zeros(num_group).astype(float)
        
        for i in range(self.num_group_attr1):
            A = self.data[self.data[:, self.split1[i]] == 1]
            A = A[:, self.idx]
            count_group[i] = len(A)
            sum_positive[i] = np.sum(np.matmul(x, A.T) >= 0)
            
        for i in range(self.num_group_attr2):
            A = self.data[self.data[:, self.split2] == i]
            i = self.num_group_attr1 + i
            A = A[:, self.idx]
            count_group[i] = len(A)
            sum_positive[i] = np.sum(np.matmul(x, A.T) >= 0)
                           
        ratio = sum_positive/count_group
        
        CV = np.max(ratio) - np.min(ratio)
        pvalue = np.min(ratio)/np.max(ratio)
        return ratio, CV, pvalue
    
    def compute_accuracy(self, list_f1, list_f2, list_pts, num_pts):  
        '''
        This function is to compute training/testing loss/accuracy, disparate impact for a number of nondominated solutions.
        Input
        list_f1: The list of first objective function values
        list_f2: The list of second objective function values
        list_pts: The list of nondominated solutions
        num_pts: Number of solutions to be measure
        Output
        disparate_impact: CV scores
        percentage: Demographic decomposition in positive prediction class
        pvalue: percent of minorty group of positive prediction/percent of majority group of positive prediction. when pvalue is greater 0.8, we say it fair. 
        total_accuracy: Training accuracy of the entire dataset
        training_accuracy: Training accuracy of each demograpic group
        training_FPR: FPR of each demograpic group
        training_FNR: FNR of each demograpic group
        training_loss: Training loss of the entire dataset
        training_obj1: Regularized training loss of the entire dataset
        training_obj2: Second objective value, i.e., square of covariance
        '''
        temp = np.copy(list_f1)
        sort_acc_index = np.argsort(temp)
        idx = range(0, len(list_f1), int(math.floor(len(list_f1)/num_pts)))
        sort_index = sort_acc_index[idx]
        num_pts = len(sort_index)
        
        num_group = self.num_group_attr1 + self.num_group_attr2
        total_accuracy = np.zeros(num_pts)
        training_accuracy = np.zeros([num_pts, num_group])
        training_FPR = np.zeros([num_pts, num_group])
        training_FNR = np.zeros([num_pts, num_group])
        disparate_impact = np.zeros(num_pts)
        percentage = np.zeros([num_pts, num_group])
        pvalue = np.zeros(num_pts)
        
        training_loss = self.loss(list_pts[sort_index])
        training_obj1 = self.f1(list_pts[sort_index])
        training_obj2 = self.f2(list_pts[sort_index])
        training_obj3 = self.f3(list_pts[sort_index])
        
        for i in range(num_pts):   
            total_accuracy[i], training_accuracy[i, :], training_FPR[i, :], training_FNR[i, :] = self.predict_accuracy(list_pts[sort_index[i]])
            percentage[i, :], disparate_impact[i], pvalue[i] = self.disparate_impact(list_pts[sort_index[i]])

        return disparate_impact, percentage, pvalue, total_accuracy, training_accuracy, training_FPR, training_FNR,                training_loss, training_obj1, training_obj2, training_obj3
    


# Bi-objective formulation: works for datasets with consider binary sensitive attribute, minimize prediction loss and maximize equal opportunity (Equalized FNR), include info of functions and gradients 

class Fairness_LogRe_EO:
    setting = "finite_sum"
    projection = 0
    name = "Fairness_LogRe"
    lb = - 5
    ub = 5
    m = 2
    num_group = 2
    
    
    def __init__(self, file_path, dataset, split, SEED, NumData, train_or_test):
        self.Alldata = np.loadtxt(file_path)
        NUM_Alldata = len(self.Alldata)
        
        np.random.seed(SEED)
        w = np.random.choice(NUM_Alldata, NumData, replace=False)
        
        if train_or_test == "train":
            self.data = self.Alldata[w, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Training data size: ", self.num_data)
        elif train_or_test == "test":
            test_data_idx = np.delete(np.arange(NUM_Alldata), w)
            self.data = self.Alldata[test_data_idx, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Testing data size: ", self.num_data)
        else:
            self.data = self.Alldata
            self.num_data, self.dim_prob = self.data.shape
            print ("#All data size: ", self.num_data)
        
        self.data_name = dataset
        self.n = self.dim_prob - 2
        self.lambda_ = 1.0/1000 
        self.split = split
        self.idx = np.delete(np.arange(self.dim_prob), [0, split])
        print ('Sensitive feature index: ', self.split)
        
        # compute z bar
        self.zbar = np.sum(self.data[:, split])*1.0/self.num_data
        print ('Sum of positive ones: ', np.sum(self.data[:, split]))
        
        # data size for function value evaluation
        self.eval_size = self.num_data 

    ## logistic loss only
    def loss(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+np.exp(- data1[:,0].reshape(1, sizek)*np.matmul(x, A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek
        return f1   
    
    ## first objective is to minimize logistic regression loss
    def f1(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+ np.exp(-data1[:,0].reshape(1,sizek)*np.matmul(x,A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek + self.lambda_/2*np.linalg.norm(x,axis = 1)**2
        return f1
    
    ## second objective is to minimize the CV score
    def f2(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_subpart = (1.0 + data1[:, 0])*data1[:, 0]*x.dot(A.T)/2.0
        f2_each = 1.0/sizek*(f2_subpart*((f2_subpart < 0)*1.0)).reshape(-1, sizek)*(data1[:, self.split] - self.zbar).reshape(1, sizek)
        f2 = np.sum(f2_each, axis = 1)
        f2 = f2**2
        return f2
    
    ## stochastic gradients for f1: k is the random index set
    def g1(self, x, k=[1]):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        part1 = - np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T))/(1 + np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T)))
        part2 = data1[:, 0].reshape(1,sizek)*A.T
        g1_each = part1.reshape(1,1,sizek)*part2.reshape(1,self.n,sizek)
        g1 = np.sum(g1_each,axis = 2)
        g1 = g1/sizek # + self.lambda_*x
        return g1[0]
    
    ## stochastic gradients for f2: use a mini-batch
    def g2(self, x, k = range(10)):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        
        g2_subpart = (1.0 + data1[:, 0])*data1[:, 0]*x.dot(A.T)/2.0
        g2_each = (2.0/sizek)*(data1[:,self.split]- self.zbar).reshape(1, sizek)*(g2_subpart*((g2_subpart < 0)*1.0)).reshape(1, sizek)
        g2 = np.sum(g2_each, axis = 1).reshape(1,1)
        
        g3_each = (1.0/sizek)*(1.0 + data1[:, 0])*data1[:, 0]/2.0*((g2_subpart < 0)*1.0)*(data1[:, self.split] - self.zbar)
        g3 = g3_each.reshape(sizek, 1)*A
        
        g23 = g2 * (np.sum(g3,axis = 0).reshape(1, self.n))
        return g23[0]
        
    def predict_accuracy(self, x):  
        num_group = self.num_group
        count_group = np.zeros(num_group).astype(float)
        sum_correct = np.zeros(num_group).astype(float)
        sum_one = np.zeros(num_group).astype(float)
        sum_zero = np.zeros(num_group).astype(float)
        sum_FPR = np.zeros(num_group).astype(float)
        sum_FNR = np.zeros(num_group).astype(float)
        Accuracy = np.zeros(num_group).astype(float)
                
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i] # split to group
            count_group[i] = len(A)
            target = A[:, 0]
            APos = A[A[:, 0] == 1]
            ANeg = A[A[:, 0] == -1]
            A = A[:, self.idx]
            
            sum_correct[i] =  np.sum(target*np.matmul(x, A.T) >= 0)
            sum_one[i] = len(APos)
            sum_zero[i] = len(ANeg)            
            
            sum_FPR[i] = np.sum(np.matmul(x, ANeg[:, self.idx].T) > 0)
            sum_FNR[i] = np.sum(np.matmul(x, APos[:, self.idx].T) < 0)
        
        Accuracy = sum_correct/count_group
        FPR = sum_FPR/sum_zero
        FNR = sum_FNR/sum_one
        total_accuracy = np.sum(sum_correct)*1.0/self.num_data
        
        ## only works for binary
        CV_FPR = np.abs(FPR[0] - FPR[1])
        CV_FNR = np.abs(FNR[0] - FNR[1])
        return CV_FPR, CV_FNR, total_accuracy, Accuracy.reshape(1, num_group), FPR.reshape(1, num_group), FNR.reshape(1, num_group)
    
    
    def disparate_impact(self, x):
        num_group = self.num_group
        sum_positive = np.zeros(num_group).astype(float)
        count_group = np.zeros(num_group).astype(float)
        
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i]
            A = A[:, self.idx]
            count_group[i] = len(A)
            sum_positive[i, 0] = np.sum(np.matmul(x, A.T) >= 0)
        
        ratio = sum_positive/count_group
        CV = np.max(ratio) - np.min(ratio)
        pvalue = np.min(ratio)/np.max(ratio)
        return ratio, CV, pvalue
    
    ## This function is to compute training/testing loss/accuracy, FNR, FPR, and CV score
    ## for a number of nondominated solutions.
    def compute_accuracy(self, list_f1, list_f2, list_pts, num_pts):        
        temp = np.copy(list_f1)
        sort_acc_index = np.argsort(temp)
        idx = range(0, len(list_f1), int(math.floor(len(list_f1)/num_pts)))
        sort_index = sort_acc_index[idx]
        num_pts = len(sort_index)
        
        num_group = self.num_group
        total_accuracy = np.zeros(num_pts)
        training_accuracy = np.zeros([num_pts, num_group])
        training_FPR = np.zeros([num_pts, num_group])
        training_FNR = np.zeros([num_pts, num_group])
        disparate_impact = np.zeros(num_pts)
        percentage = np.zeros([num_pts, num_group])
        pvalue = np.zeros(num_pts)
        
        CV_FPR = np.zeros(num_pts)
        CV_FNR = np.zeros(num_pts)
        
        training_loss = self.loss(list_pts[sort_index])
        training_obj1 = self.f1(list_pts[sort_index])
        training_obj2 = self.f2(list_pts[sort_index])
        
        for i in range(num_pts):   
            CV_FPR[i], CV_FNR[i], total_accuracy[i], training_accuracy[i, :], training_FPR[i, :], training_FNR[i, :] = self.predict_accuracy(list_pts[sort_index[i]])
            percentage[i, :], disparate_impact[i], pvalue[i] = self.disparate_impact(list_pts[sort_index[i]])

        return CV_FPR, CV_FNR, disparate_impact, percentage, pvalue, total_accuracy, training_accuracy, training_FPR, training_FNR,                training_loss, training_obj1, training_obj2      
    
    
    # Bi-objective formulation: works for datasets with consider binary sensitive attribute, minimize prediction loss and maximize equal opportunity (Equalized FNR), include info of functions and gradients 

class Fairness_LogRe_EO_smoothed:
    setting = "finite_sum"
    projection = 0
    name = "Fairness_LogRe"
    lb = - 5
    ub = 5
    m = 2
    num_group = 2
    
    
    def __init__(self, file_path, dataset, split, SEED, NumData, train_or_test):
        self.Alldata = np.loadtxt(file_path)
        NUM_Alldata = len(self.Alldata)
        
        np.random.seed(SEED)
        w = np.random.choice(NUM_Alldata, NumData, replace=False)
        
        if train_or_test == "train":
            self.data = self.Alldata[w, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Training data size: ", self.num_data)
        elif train_or_test == "test":
            test_data_idx = np.delete(np.arange(NUM_Alldata), w)
            self.data = self.Alldata[test_data_idx, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Testing data size: ", self.num_data)
        else:
            self.data = self.Alldata
            self.num_data, self.dim_prob = self.data.shape
            print ("#All data size: ", self.num_data)
        
        self.data_name = dataset
        self.n = self.dim_prob - 2
        self.lambda_ = 1.0/1000 
        self.split = split
        self.idx = np.delete(np.arange(self.dim_prob), [0, split])
        print ('Sensitive feature index: ', self.split)
        
        # compute z bar
        self.zbar = np.sum(self.data[:, split])*1.0/self.num_data
        print ('Sum of positive ones: ', np.sum(self.data[:, split]))
        
        # data size for function value evaluation
        self.eval_size = self.num_data 

    ## logistic loss only
    def loss(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+np.exp(- data1[:,0].reshape(1, sizek)*np.matmul(x, A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek
        return f1   
    
    ## first objective is to minimize logistic regression loss
    def f1(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+ np.exp(-data1[:,0].reshape(1,sizek)*np.matmul(x,A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek + self.lambda_/2*np.linalg.norm(x,axis = 1)**2
        return f1
    
    ## second objective is to minimize the CV score
    def f2(self, x, k = np.array([]), alpha = -30.0):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_subpart = (1.0 + data1[:, 0])*data1[:, 0]*x.dot(A.T)/2.0 # (1 + y)y cx/2
        f2_each = 1.0/sizek*(f2_subpart*np.exp(alpha*f2_subpart)/(1.0 + np.exp(alpha*f2_subpart))).reshape(-1, sizek)*(data1[:, self.split] - self.zbar).reshape(1, sizek)
        f2 = np.sum(f2_each, axis = 1)
        f2 = f2**2
        return f2
    
    ## stochastic gradients for f1: k is the random index set
    def g1(self, x, k=[1]):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        part1 = - np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T))/(1 + np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T)))
        part2 = data1[:, 0].reshape(1,sizek)*A.T
        g1_each = part1.reshape(1,1,sizek)*part2.reshape(1,self.n,sizek)
        g1 = np.sum(g1_each,axis = 2)
        g1 = g1/sizek # + self.lambda_*x
        return g1[0]
    
    ## stochastic gradients for f2: use a mini-batch
    def g2(self, x, k = range(10), alpha = -30.0):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        
        g2_subpart = (1.0 + data1[:, 0])*data1[:, 0]*x.dot(A.T)/2.0 # (1+y)y/2*cx
        smoothed_min = g2_subpart*np.exp(alpha*g2_subpart)/(1.0 + np.exp(alpha*g2_subpart))
        
        g2_each = (2.0/sizek)*(data1[:,self.split]- self.zbar).reshape(1, sizek)*smoothed_min.reshape(1, sizek)
        g2 = np.sum(g2_each, axis = 1).reshape(1,1)
        
        g3_each = (1.0/sizek)*(data1[:, self.split] - self.zbar)*(1.0 + data1[:, 0])*data1[:,0]/2.0 *np.exp(alpha*g2_subpart)/(1.0 + np.exp(alpha*g2_subpart))*(1.0 + alpha*g2_subpart - alpha*smoothed_min)
        g3 = g3_each.reshape(sizek, 1)*A
        
        g23 = g2 * (np.sum(g3,axis = 0).reshape(1, self.n))
        
        return g23[0]
        
    def predict_accuracy(self, x):  
        num_group = self.num_group
        count_group = np.zeros(num_group).astype(float)
        sum_correct = np.zeros(num_group).astype(float)
        sum_one = np.zeros(num_group).astype(float)
        sum_zero = np.zeros(num_group).astype(float)
        sum_FPR = np.zeros(num_group).astype(float)
        sum_FNR = np.zeros(num_group).astype(float)
        Accuracy = np.zeros(num_group).astype(float)
                
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i] # split to group
            count_group[i] = len(A)
            target = A[:, 0]
            APos = A[A[:, 0] == 1]
            ANeg = A[A[:, 0] == -1]
            A = A[:, self.idx]
            
            sum_correct[i] =  np.sum(target*np.matmul(x, A.T) >= 0)
            sum_one[i] = len(APos)
            sum_zero[i] = len(ANeg)            
            
            sum_FPR[i] = np.sum(np.matmul(x, ANeg[:, self.idx].T) > 0)
            sum_FNR[i] = np.sum(np.matmul(x, APos[:, self.idx].T) < 0)
        
        Accuracy = sum_correct/count_group
        FPR = sum_FPR/sum_zero
        FNR = sum_FNR/sum_one
        total_accuracy = np.sum(sum_correct)*1.0/self.num_data
        
        ## only works for binary
        CV_FPR = np.abs(FPR[0] - FPR[1])
        CV_FNR = np.abs(FNR[0] - FNR[1])
        return CV_FPR, CV_FNR, total_accuracy, Accuracy.reshape(1, num_group), FPR.reshape(1, num_group), FNR.reshape(1, num_group)
    
    
    def disparate_impact(self, x):
        num_group = self.num_group
        sum_positive = np.zeros(num_group).astype(float)
        count_group = np.zeros(num_group).astype(float)
        
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i]
            A = A[:, self.idx]
            count_group[i] = len(A)
            sum_positive[i] = np.sum(np.matmul(x, A.T) >= 0)
                           
        ratio = sum_positive/count_group
        CV = np.max(ratio) - np.min(ratio)
        pvalue = np.min(ratio)/np.max(ratio)
        
        return ratio, CV, pvalue
    
    ## This function is to compute training/testing loss/accuracy, FNR, FPR, and CV score
    ## for a number of nondominated solutions.
    def compute_accuracy(self, list_f1, list_f2, list_pts, num_pts):        
        temp = np.copy(list_f1)
        sort_acc_index = np.argsort(temp)
        idx = range(0, len(list_f1), int(math.floor(len(list_f1)/num_pts)))
        sort_index = sort_acc_index[idx]
        num_pts = len(sort_index)
        
        num_group = self.num_group
        total_accuracy = np.zeros(num_pts)
        training_accuracy = np.zeros([num_pts, num_group])
        training_FPR = np.zeros([num_pts, num_group])
        training_FNR = np.zeros([num_pts, num_group])
        disparate_impact = np.zeros(num_pts)
        percentage = np.zeros([num_pts, num_group])
        pvalue = np.zeros(num_pts)
        
        CV_FPR = np.zeros(num_pts)
        CV_FNR = np.zeros(num_pts)
        
        training_loss = self.loss(list_pts[sort_index])
        training_obj1 = self.f1(list_pts[sort_index])
        training_obj2 = self.f2(list_pts[sort_index])
        
        for i in range(num_pts):   
            CV_FPR[i], CV_FNR[i], total_accuracy[i], training_accuracy[i, :], training_FPR[i, :], training_FNR[i, :] = self.predict_accuracy(list_pts[sort_index[i]])
            percentage[i, :], disparate_impact[i], pvalue[i] = self.disparate_impact(list_pts[sort_index[i]])

        return CV_FPR, CV_FNR, disparate_impact, percentage, pvalue, total_accuracy, training_accuracy, training_FPR, training_FNR,                training_loss, training_obj1, training_obj2  
    
    
    # Bi-objective formulation: works for datasets with consider binary sensitive attribute, minimize prediction loss and maximize smoothed equal opportunity (Equalized FNR), include info of functions and gradients 

class Fairness_LogRe_EO_smoothed_v2:
    setting = "finite_sum"
    projection = 0
    name = "Fairness_LogRe"
    lb = - 5
    ub = 5
    m = 2
    num_group = 2
    
    
    def __init__(self, file_path, dataset, split, SEED, NumData, train_or_test):
        self.Alldata = np.loadtxt(file_path)
        NUM_Alldata = len(self.Alldata)
        
        np.random.seed(SEED)
        w = np.random.choice(NUM_Alldata, NumData, replace=False)
        
        if train_or_test == "train":
            self.data = self.Alldata[w, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Training data size: ", self.num_data)
        elif train_or_test == "test":
            test_data_idx = np.delete(np.arange(NUM_Alldata), w)
            self.data = self.Alldata[test_data_idx, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Testing data size: ", self.num_data)
        else:
            self.data = self.Alldata
            self.num_data, self.dim_prob = self.data.shape
            print ("#All data size: ", self.num_data)
        
        self.data_name = dataset
        self.n = self.dim_prob - 2
        self.lambda_ = 1.0/1000 
        self.split = split
        self.idx = np.delete(np.arange(self.dim_prob), [0, split])
        print ('Sensitive feature index: ', self.split)
        
        # compute z bar
        self.zbar = np.sum(self.data[:, split])*1.0/self.num_data
        print ('Sum of positive ones: ', np.sum(self.data[:, split]))
        
        # data size for function value evaluation
        self.eval_size = self.num_data 

    ## logistic loss only
    def loss(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+np.exp(- data1[:,0].reshape(1, sizek)*np.matmul(x, A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek
        return f1   
    
    ## first objective is to minimize logistic regression loss
    def f1(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+ np.exp(-data1[:,0].reshape(1,sizek)*np.matmul(x,A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek + self.lambda_/2*np.linalg.norm(x,axis = 1)**2
        return f1
    
    ## second objective is to minimize the CV score
    def f2(self, x, k = np.array([]), mu = 1e-5):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_subpart = (1.0 + data1[:, 0])*data1[:, 0]*x.dot(A.T)/2.0 # (1 + y)y cx/2
        smoothedOrNot = ((f2_subpart >= -mu) & (f2_subpart <= mu))*1.0
        f2_smoothed_subpart = f2_subpart*((f2_subpart < -mu)*1.0) + 0.25*(f2_subpart - mu)*(1.0 - 1.0/mu*f2_subpart)*smoothedOrNot
        
        f2_each = 1.0/sizek*f2_smoothed_subpart.reshape(-1, sizek)*(data1[:, self.split] - self.zbar).reshape(1, sizek)
        f2 = np.sum(f2_each, axis = 1)
        f2 = f2**2
        return f2
    
    ## stochastic gradients for f1: k is the random index set
    def g1(self, x, k=[1]):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        part1 = - np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T))/(1 + np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T)))
        part2 = data1[:, 0].reshape(1,sizek)*A.T
        g1_each = part1.reshape(1,1,sizek)*part2.reshape(1,self.n,sizek)
        g1 = np.sum(g1_each,axis = 2)
        g1 = g1/sizek # + self.lambda_*x
        return g1[0]
    
    ## stochastic gradients for f2: use a mini-batch
    def g2(self, x, k = range(10), mu = 1e-5):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        
        g2_subpart = (1.0 + data1[:, 0])*data1[:, 0]*x.dot(A.T)/2.0 # (1+y)y/2*cx
        smoothedOrNot = ((g2_subpart >= -mu) & (g2_subpart <= mu))*1.0 # indicate if it's in [-mu, mu]
        smoothed_min = g2_subpart*((g2_subpart < -mu)*1.0) + 0.25*(g2_subpart - mu)*(1.0 - 1.0/mu*g2_subpart)*smoothedOrNot
        
        g2_each = (2.0/sizek)*(data1[:, self.split]- self.zbar).reshape(1, sizek)*smoothed_min.reshape(1, sizek)
        g2 = np.sum(g2_each, axis = 1).reshape(1,1)
        
        g3_each = (1.0/sizek)*(data1[:, self.split] - self.zbar)*(1.0 + data1[:, 0])*data1[:,0]/2.0*((g2_subpart < -mu)*1.0 + smoothedOrNot*0.25*(2.0 - 2.0/mu*g2_subpart))
        g3 = g3_each.reshape(sizek, 1)*A
        
        g23 = g2 * (np.sum(g3,axis = 0).reshape(1, self.n))
        
        return g23[0]
        
    def predict_accuracy(self, x):  
        num_group = self.num_group
        count_group = np.zeros(num_group).astype(float)
        sum_correct = np.zeros(num_group).astype(float)
        sum_one = np.zeros(num_group).astype(float)
        sum_zero = np.zeros(num_group).astype(float)
        sum_FPR = np.zeros(num_group).astype(float)
        sum_FNR = np.zeros(num_group).astype(float)
        Accuracy = np.zeros(num_group).astype(float)
                
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i] # split to group
            count_group[i] = len(A)
            target = A[:, 0]
            APos = A[A[:, 0] == 1]
            ANeg = A[A[:, 0] == -1]
            A = A[:, self.idx]
            
            sum_correct[i] =  np.sum(target*np.matmul(x, A.T) >= 0)
            sum_one[i] = len(APos)
            sum_zero[i] = len(ANeg)            
            
            sum_FPR[i] = np.sum(np.matmul(x, ANeg[:, self.idx].T) > 0)
            sum_FNR[i] = np.sum(np.matmul(x, APos[:, self.idx].T) < 0)
        
        Accuracy = sum_correct/count_group
        FPR = sum_FPR/sum_zero
        FNR = sum_FNR/sum_one
        total_accuracy = np.sum(sum_correct)*1.0/self.num_data
        
        ## only works for binary
        CV_FPR = np.abs(FPR[0] - FPR[1])
        CV_FNR = np.abs(FNR[0] - FNR[1])
        return CV_FPR, CV_FNR, total_accuracy, Accuracy.reshape(1, num_group), FPR.reshape(1, num_group), FNR.reshape(1, num_group)
    
    
    def disparate_impact(self, x):
        num_group = self.num_group
        sum_positive = np.zeros(num_group).astype(float)
        count_group = np.zeros(num_group).astype(float)
        
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i]
            A = A[:, self.idx]
            count_group[i] = len(A)
            sum_positive[i] = np.sum(np.matmul(x, A.T) >= 0)
                           
        ratio = sum_positive/count_group
        CV = np.max(ratio) - np.min(ratio)
        pvalue = np.min(ratio)/np.max(ratio)
        return ratio, CV, pvalue
    
    ## This function is to compute training/testing loss/accuracy, FNR, FPR, and CV score
    ## for a number of nondominated solutions.
    def compute_accuracy(self, list_f1, list_f2, list_pts, num_pts):        
        temp = np.copy(list_f1)
        sort_acc_index = np.argsort(temp)
        idx = range(0, len(list_f1), int(math.floor(len(list_f1)/num_pts)))
        sort_index = sort_acc_index[idx]
        num_pts = len(sort_index)
        
        num_group = self.num_group
        total_accuracy = np.zeros(num_pts)
        training_accuracy = np.zeros([num_pts, num_group])
        training_FPR = np.zeros([num_pts, num_group])
        training_FNR = np.zeros([num_pts, num_group])
        disparate_impact = np.zeros(num_pts)
        percentage = np.zeros([num_pts, num_group])
        pvalue = np.zeros(num_pts)
        
        CV_FPR = np.zeros(num_pts)
        CV_FNR = np.zeros(num_pts)
        
        training_loss = self.loss(list_pts[sort_index])
        training_obj1 = self.f1(list_pts[sort_index])
        training_obj2 = self.f2(list_pts[sort_index])
        
        for i in range(num_pts):   
            CV_FPR[i], CV_FNR[i], total_accuracy[i], training_accuracy[i, :], training_FPR[i, :], training_FNR[i, :] = self.predict_accuracy(list_pts[sort_index[i]])
            percentage[i, :], disparate_impact[i], pvalue[i] = self.disparate_impact(list_pts[sort_index[i]])

        return CV_FPR, CV_FNR, disparate_impact, percentage, pvalue, total_accuracy, training_accuracy, training_FPR, training_FNR,                training_loss, training_obj1, training_obj2  

    # Three objectives: Consider binary sensitive attribute, minimize prediction loss, maximize equal opportunity (Equalized FNR), and minimize disparate impact

class Fairness_LogRe_DIEO_m3:
    setting = "finite_sum"
    projection = 0
    name = "Fairness_LogRe"
    lb = - 5
    ub = 5
    m = 3
    num_group = 2
    
    
    def __init__(self, file_path, dataset, split, SEED, NumData, train_or_test):
        self.Alldata = np.loadtxt(file_path)
        NUM_Alldata = len(self.Alldata)
        
        np.random.seed(SEED)
        w = np.random.choice(NUM_Alldata, NumData, replace=False)
        
        if train_or_test == "train":
            self.data = self.Alldata[w, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Training data size: ", self.num_data)
        elif train_or_test == "test":
            test_data_idx = np.delete(np.arange(NUM_Alldata), w)
            self.data = self.Alldata[test_data_idx, :]
            self.num_data, self.dim_prob = self.data.shape
            print ("#Testing data size: ", self.num_data)
        else:
            self.data = self.Alldata
            self.num_data, self.dim_prob = self.data.shape
            print ("#All data size: ", self.num_data)
        
        self.data_name = dataset
        self.n = self.dim_prob - 2
        self.lambda_ = 1.0/1000 
        self.split = split
        self.idx = np.delete(np.arange(self.dim_prob), [0, split])
        print ('Sensitive feature index: ', self.split)
        
        # compute z bar
        self.zbar = np.sum(self.data[:, split])*1.0/self.num_data
        print ('Sum of positive ones: ', np.sum(self.data[:, split]))
        
        # data size for function value evaluation
        self.eval_size = self.num_data 

    ## logistic loss only
    def loss(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+np.exp(- data1[:,0].reshape(1, sizek)*np.matmul(x, A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek
        return f1   
    
    ## first objective is to minimize logistic regression loss
    def f1(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f1_each = np.log(1+ np.exp(-data1[:,0].reshape(1,sizek)*np.matmul(x,A.T)))
        f1 = np.sum(f1_each,axis = 1)
        f1 = f1/sizek + self.lambda_/2*np.linalg.norm(x,axis = 1)**2
        return f1
    
    ## second objective is to minimize the EO
    def f2(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f2_subpart = (1.0 + data1[:, 0])*data1[:, 0]*x.dot(A.T)/2.0
        f2_each = 1.0/sizek*(f2_subpart*((f2_subpart < 0)*1.0)).reshape(-1, sizek)*(data1[:, self.split] - self.zbar).reshape(1, sizek)
        f2 = np.sum(f2_each, axis = 1)
        f2 = f2**2
        return f2
    
    ## third objective is to minimize the fairness w.r.t. Disparate impact
    def f3(self, x, k = np.array([])):
        sizek = len(k)
        if sizek == 0:
            k = np.arange(self.eval_size)
            sizek = self.eval_size
        data1 = self.data[k,:]
        
        A = data1[:, self.idx]
        f3_each = 1.0/sizek*np.matmul(x,A.T)*(data1[:, self.split] - self.zbar).reshape(1, sizek)
        f3 = np.sum(f3_each,axis = 1)
        f3 = f3**2
        return f3
    
    ## stochastic gradients for f1: k is the random index set
    def g1(self, x, k=[1]):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        part1 = - np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T))/(1 + np.exp(- data1[:, 0].reshape(1,sizek)*np.matmul(x, A.T)))
        part2 = data1[:, 0].reshape(1,sizek)*A.T
        g1_each = part1.reshape(1,1,sizek)*part2.reshape(1,self.n,sizek)
        g1 = np.sum(g1_each,axis = 2)
        g1 = g1/sizek + self.lambda_*x
        return g1[0]
    
    ## stochastic gradients for f2: use a mini-batch
    def g2(self, x, k = range(10)):
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        
        g2_subpart = (1.0 + data1[:, 0])*data1[:, 0]*x.dot(A.T)/2.0
        g2_each = (2.0/sizek)*(data1[:,self.split]- self.zbar).reshape(1, sizek)*(g2_subpart*((g2_subpart < 0)*1.0)).reshape(1, sizek)
        g2 = np.sum(g2_each, axis = 1).reshape(1,1)
        
        g3_each = (1.0/sizek)*(1.0 + data1[:, 0])*data1[:, 0]/2.0*((g2_subpart < 0)*1.0)*(data1[:, self.split] - self.zbar)
        g3 = g3_each.reshape(sizek, 1)*A
        
        g23 = g2 * (np.sum(g3,axis = 0).reshape(1, self.n))
        return g23[0]
    
    ## stochastic gradients for f3: use a mini-batch
    def g3(self, x, k=range(10)): 
        data1 = self.data[k,:]
        sizek = len(k)
        A = data1[:, self.idx]
        g2_each = (2.0/sizek)*(data1[:,self.split]- self.zbar).reshape(1,sizek)*np.matmul(x, A.T)
        g3_each = (1.0/sizek)*(data1[:, self.split] - self.zbar).reshape(sizek,1)*A
        g23 = (np.sum(g2_each,axis = 1).reshape(1,1)) * (np.sum(g3_each,axis = 0).reshape(1,self.n))
        return g23[0]
        
    def predict_accuracy(self, x):  
        num_group = self.num_group
        count_group = np.zeros(num_group).astype(float)
        sum_correct = np.zeros(num_group).astype(float)
        sum_one = np.zeros(num_group).astype(float)
        sum_zero = np.zeros(num_group).astype(float)
        sum_FPR = np.zeros(num_group).astype(float)
        sum_FNR = np.zeros(num_group).astype(float)
        Accuracy = np.zeros(num_group).astype(float)
                
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i] # split to group
            count_group[i] = len(A)
            target = A[:, 0]
            APos = A[A[:, 0] == 1]
            ANeg = A[A[:, 0] == -1]
            A = A[:, self.idx]
            
            sum_correct[i] =  np.sum(target*np.matmul(x, A.T) >= 0)
            sum_one[i] = len(APos)
            sum_zero[i] = len(ANeg)            
            
            sum_FPR[i] = np.sum(np.matmul(x, ANeg[:, self.idx].T) > 0)
            sum_FNR[i] = np.sum(np.matmul(x, APos[:, self.idx].T) < 0)
        
        Accuracy = sum_correct/count_group
        FPR = sum_FPR/sum_zero
        FNR = sum_FNR/sum_one
        total_accuracy = np.sum(sum_correct)*1.0/self.num_data
        
        ## only works for binary
        CV_FPR = np.abs(FPR[0] - FPR[1])
        CV_FNR = np.abs(FNR[0] - FNR[1])
        return CV_FPR, CV_FNR, total_accuracy, Accuracy.reshape(1, num_group), FPR.reshape(1, num_group), FNR.reshape(1, num_group)
    
    
    def disparate_impact(self, x):
        num_group = self.num_group
        sum_positive = np.zeros(num_group).astype(float)
        count_group = np.zeros(num_group).astype(float)
        
        for i in range(num_group):
            A = self.data[self.data[:, self.split] == i]
            A = A[:, self.idx]
            count_group[i] = len(A)
            sum_positive[i] = np.sum(np.matmul(x, A.T) >= 0)
                           
        ratio = sum_positive/count_group
        CV = np.max(ratio) - np.min(ratio)
        pvalue = np.min(ratio)/np.max(ratio)
        return ratio, CV, pvalue
    
    ## This function is to compute training/testing loss/accuracy, FNR, FPR, and CV score
    ## for a number of nondominated solutions.
    def compute_accuracy(self, list_f1, list_f2, list_pts, num_pts):        
        temp = np.copy(list_f1)
        sort_acc_index = np.argsort(temp)
        idx = range(0, len(list_f1), int(math.floor(len(list_f1)/num_pts)))
        sort_index = sort_acc_index[idx]
        num_pts = len(sort_index)
        
        num_group = self.num_group
        total_accuracy = np.zeros(num_pts)
        training_accuracy = np.zeros([num_pts, num_group])
        training_FPR = np.zeros([num_pts, num_group])
        training_FNR = np.zeros([num_pts, num_group])
        disparate_impact = np.zeros(num_pts)
        percentage = np.zeros([num_pts, num_group])
        pvalue = np.zeros(num_pts)
        
        CV_FPR = np.zeros(num_pts)
        CV_FNR = np.zeros(num_pts)
        
        training_loss = self.loss(list_pts[sort_index])
        training_obj1 = self.f1(list_pts[sort_index])
        training_obj2 = self.f2(list_pts[sort_index])
        training_obj3 = self.f3(list_pts[sort_index])
        
        for i in range(num_pts):   
            CV_FPR[i], CV_FNR[i], total_accuracy[i], training_accuracy[i, :], training_FPR[i, :], training_FNR[i, :] = self.predict_accuracy(list_pts[sort_index[i]])
            percentage[i, :], disparate_impact[i], pvalue[i] = self.disparate_impact(list_pts[sort_index[i]])

        return CV_FPR, CV_FNR, disparate_impact, percentage, pvalue, total_accuracy, training_accuracy, training_FPR, training_FNR, training_loss, training_obj1, training_obj2, training_obj3

