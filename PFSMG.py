#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import time
# from quadprog import solve_qp # Import this QP solver if dealing with three-objective problem


class Main_SMG:
    ## general parameters setting
    epsion = 1e-4
    max_iter = 1000
    max_len_pareto_front = 1500
    
    # for step size
    stepsize = 0.3
    alpha = 0.5
    step_scheme = 3
    discount_iter_interval = 400 # discount step size every certain iterations

    num_starting_pts = 5
    has_starting_list = False
    point_per_iteration = 2 #p1
    num_steps_per_point = 2 #p2
    
    # parameters for getting better spread: explore along first objective, second objective, and maximum hole region
    percent_explore = 0.3 # only explore along single objective for first 30% iterations
    f1_explore_interval = 6 # explore first objective every 6 iterations
    f2_explore_interval = 4 # explore first objective every 4 iterations
    f1_explore_pt_per_iter = 1 # do once 2-step stochastic gradient for each point
    f2_explore_pt_per_iter = 1
    f1_num_steps_per_point = 2
    f2_num_steps_per_point = 2
    max_hole_only = False # indicate whether or not only do SMG for maximum hole points
    num_max_hole_points = 3 # select three pairs of largest hole points
    max_hole_explore_pt_per_iter = 2 # do twice 2-step SMG for each point
    max_hole_num_steps_per_point = 2
    dense_threshold = 1.0/1000  # the maximum allowed gap between two non-dominated points
    
    # gradient sample batch parameters: batch size will grow exponetially with rate 1.01
    batch1_init = 5
    batch1_factor = 1.01
    batch2_init = 200
    batch2_factor = 1.01
    batch1_max = 1.0/3 # indicate maximum batch size, control the computational expense
    batch2_max = 1.0/3  
    
    #samples for pareto_shape function
    num_samples = 1500 
    
    # counter of iterations and gradient evaluation
    num_iter = 0
    num_grad_eval_f1 = 0
    num_grad_eval_f2 = 0
    
    # construct the class based on a multi-objective formulation
    def __init__(self, func):
        self.func = func
        self.starting_point_list = np.zeros((self.num_starting_pts, self.func.n))
        

    # function to draw the shape of two function value
    def pareto_shape(self):
        f1_value_list = np.zeros(self.num_samples)
        f2_value_list = np.zeros(self.num_samples)
        if self.func.setting == "finite_sum":
            x = np.random.uniform(self.func.lb, self.func.ub, [self.num_samples,self.func.n])
            f1_value_list = self.func.f1(x)
            f2_value_list = self.func.f2(x)
        return f1_value_list, f2_value_list
    
    # main function of Pareto front stochastic multi-gradient
    def main_SMG(self):
        self.num_iter = 0        
        extreme_index = {"f1": [], "f2": [], "f3": []}
        
        # generate random starting list
        if not self.has_starting_list:
            x = np.random.uniform(self.func.lb, self.func.ub, [self.num_starting_pts,self.func.n])
            self.starting_point_list = x
            f1_value_list = self.func.loss(x)
            f2_value_list = self.func.f2(x)
        else:
            f1_value_list = self.func.loss(self.starting_point_list)
            f2_value_list = self.func.f2(self.starting_point_list)
             
        updating_point_list = self.starting_point_list        
        begin = time.time()
        
        # PFSMG is stopped it reaches either maximum iteration or maximum number of nondominated points
        while len(updating_point_list) < self.max_len_pareto_front and self.num_iter < self.max_iter:
            if self.num_iter <= self.percent_explore*self.max_iter and self.num_iter % self.f1_explore_interval == 0:
                extreme_index["f1"] = range(len(updating_point_list))
            else:
                extreme_index["f1"] = [np.argmin(f1_value_list), np.argmax(f1_value_list)]
                
            if self.num_iter <= self.percent_explore*self.max_iter and self.num_iter % self.f2_explore_interval == 0:
                extreme_index["f2"] = range(len(updating_point_list))
            else:
                extreme_index["f2"] = [np.argmin(f2_value_list), np.argmax(f2_value_list)]

            start = time.time()
            pts_considered = self.max_hole_index(f1_value_list, f2_value_list, self.num_max_hole_points)
            for i in range(len(updating_point_list)):
                curr_x = updating_point_list[i, :]
                if i in extreme_index["f1"]:
                    for j in range(self.f1_explore_pt_per_iter):
                        f1, f2, pt = self.LSMG(curr_x, self.f1_num_steps_per_point, "f1")
                        f1_value_list = np.append(f1_value_list, f1)
                        f2_value_list = np.append(f2_value_list, f2)
                        updating_point_list = np.vstack([updating_point_list, pt.T])
                if i in extreme_index["f2"]:
                    for j in range(self.f2_explore_pt_per_iter):
                        f1, f2, pt = self.LSMG(curr_x, self.f2_num_steps_per_point, "f2")
                        f1_value_list = np.append(f1_value_list, f1)
                        f2_value_list = np.append(f2_value_list, f2)
                        updating_point_list = np.vstack([updating_point_list, pt.T])
                if i in pts_considered: 
                    for j in range(self.max_hole_explore_pt_per_iter):
                        f1, f2, pt = self.LSMG(curr_x, self.max_hole_num_steps_per_point, "")
                        f1_value_list = np.append(f1_value_list, f1)
                        f2_value_list = np.append(f2_value_list, f2)
                        updating_point_list = np.vstack([updating_point_list, pt.T])
                else:
                    if not self.max_hole_only:
                        for j in range(self.point_per_iteration):
#                             temp = 1.0/1000*np.random.uniform(self.func.lb, self.func.ub, self.func.n)  ## uncomment this line and add temp to current point can add more stochasticity to the algorithm
                            curr_x_perturbed = curr_x #+ temp
                            f1, f2, pt = self.LSMG(curr_x_perturbed, self.num_steps_per_point, "")
                            f1_value_list = np.append(f1_value_list, f1)
                            f2_value_list = np.append(f2_value_list, f2)
                            updating_point_list = np.vstack([updating_point_list, pt.T])


            f1_value_list, f2_value_list, updating_point_list = self.clear(f1_value_list, f2_value_list, updating_point_list)
            f1_value_list, f2_value_list, updating_point_list = self.remove_dense(f1_value_list, f2_value_list, updating_point_list)
            dur = time.time() - start
            print ("time: ",  dur)
            self.num_iter += self.num_steps_per_point

            # discount every certain iterations
            if self.step_scheme == 3 and self.num_iter % self.discount_iter_interval == 0:
                self.stepsize = self.stepsize*self.alpha
            print ("#Pts: ", len(f1_value_list), " #Iter: ", self.num_iter)
            
        total_time = time.time() - begin
        print ("Total time: ",  total_time)
                
        return f1_value_list, f2_value_list, updating_point_list, total_time  
    
    # Stochastic multi-gradient steps
    def LSMG(self, x_k, num_steps, flag):
        beta = 0.02
        gama = 0.5
        normalization = 0
        
        # specify batch size for two functions
        batch = min(int(self.batch1_init*self.batch1_factor**self.num_iter), int(self.batch1_max*self.func.num_data))
        batch2 = min(int(self.batch2_init*self.batch2_factor**self.num_iter), int(self.batch2_max*self.func.num_data))
        
        for t in range(num_steps):
            w1 = np.random.choice(self.func.num_data, batch, replace=False)
            w2 = np.random.choice(self.func.num_data, batch2, replace=False)
            
            if flag == "":
                g1 = self.func.g1(x_k, w1)
                g2 = self.func.g2(x_k, w2)
                self.num_grad_eval_f1 += batch
                self.num_grad_eval_f2 += batch2  
                
                ## compute common descent direction
                if self.func.n == 1: 
                    if g1*g2 > 0:
                        if abs(g1) < abs(g2):
                            g_k = g1
                        else:
                            g_k = g2
                    else:
                        g_k = 0
                else: 
                    c1 = np.dot(g2 - g1, g2)/np.dot(g2 - g1, g2 - g1) 
                    c2 = np.dot(g1 - g2, g1)/np.dot(g2 - g1, g2 - g1)
                    if c1 < 0:
                        c1 = 0
                        c2 = 1
                    if c2 < 0:
                        c2 = 0
                        c1 = 1
                    g_k = c1*g1 + c2*g2
            elif flag == "f1":
                g1 = self.func.g1(x_k, w1)
                self.num_grad_eval_f1 += batch
                
                g_k = g1
            elif flag == "f2":
                g2 = self.func.g2(x_k, w2)
                self.num_grad_eval_f2 += batch2
                
                g_k = g2
            
            ## take a step
            if self.step_scheme == 0: # fixed step size
                x_k = x_k - self.stepsize*g_k
            elif self.step_scheme == 1: # const/t
                self.stepsize = min([1.0, 2.0/(self.num_iter + 1)])   # min([1 4/np.sqrt(t + 1)]);
                x_k = x_k - self.stepsize*g_k
            elif self.step_scheme == 2: # backtracking strategy
                step = 1
                while step >= self.epsion and (((self.func.f1(x_k - step*g_k) > self.func.f1(x_k) - beta*step*np.dot(g1, g_k)) or (self.func.f2(x_k - step*g_k) > self.func.f2(x_k)))                      or (((self.func.f1(x_k - step*g_k) > self.func.f1(x_k)) or self.func.f2(x_k - step*g_k) > self.func.f2(x_k) - beta*step*np.dot(g2, g_k)))):
                    step = gama*step
                if step < self.epsion:
                    break
                else:
                    self.stepsize = step
                    x_k = x_k - self.stepsize*g_k
            elif self.step_scheme == 3: # discount every certain iterations
                x_k = x_k - self.stepsize*g_k
            
            ## project if a simple bound exists
            if self.func.projection:
                x_k = np.where(x_k <= self.func.ub, x_k, self.func.ub)
                x_k = np.where(x_k >= self.func.lb, x_k, self.func.lb)
            
        return self.func.loss(x_k.reshape(1,-1)), self.func.f2(x_k.reshape(1,-1)), x_k
    
    # clean dominated points at the end of each iteration
    def clear(self, list_f1, list_f2, list_pt):
        
        x_bar = np.repeat(list_f1.reshape(-1,1),len(list_f1),axis = 1)
        y_bar = np.repeat(list_f2.reshape(-1,1),len(list_f2),axis = 1)
        
        array_f1 = list_f1
        array_f2 = list_f2
        array_pt = list_pt        
        
        x_check1 = (array_f1 <= x_bar)
        x_check2 = (array_f1 < x_bar)
        
        y_check1 = (array_f2 <= y_bar)
        y_check2 = (array_f2 < y_bar)
        
        all_check1 = (x_check1 & y_check2)
        all_check2 = (x_check2 & y_check1)
        
        sum1 = all_check1.sum(axis = 1)
        sum2 = all_check2.sum(axis = 1)
        
        rest_index = np.array([i for i in range(len(list_f1)) if (sum1[i] < 1 or sum2[i] < 1)])
        return array_f1[rest_index], array_f2[rest_index], array_pt[rest_index]
    
    # remove non-dominated points from current list from dense region
    def remove_dense(self, list_f1, list_f2, list_pt):
        num_total_pts = len(list_f1)
        if self.dense_threshold == 0:
            dense_threshold = 1.0/(800 + self.num_iter/2.0)
        else:
            dense_threshold = self.dense_threshold
    
        index_f1 = np.argsort(list_f1)
        index_f2 = np.argsort(list_f2)
        temp_list_f1 = np.sort(list_f1)
        temp_list_f2 = np.sort(list_f2)
        
        min_gap_f1 = (temp_list_f1[-1] - temp_list_f1[0])*dense_threshold
        max_gap_f2 = (temp_list_f2[-1] - temp_list_f2[0])*dense_threshold

        diff_list_f1 = np.diff(temp_list_f1)
        diff_list_f2 = np.diff(temp_list_f2)
        keep_index_f1 = [0, num_total_pts - 1]
        keep_index_f2 = [0, num_total_pts - 1]
        
        i = 1
        while i < num_total_pts - 2:# keep the first and last ones
            j = i
            curr_gap = diff_list_f1[j]
            while curr_gap < min_gap_f1 and j < num_total_pts - 2:
                j += 1
                curr_gap += diff_list_f1[j]
            keep_index_f1.append((j - i)/2 + i)

            if i < j:
                i = j
            else:
                i += 1
        i = 1
        while i < num_total_pts - 2:# keep the first and last ones
            j = i
            curr_gap = diff_list_f2[j]
            while curr_gap < max_gap_f2 and j < num_total_pts - 2:
                j += 1
                curr_gap += diff_list_f2[j]
            keep_index_f2.append((j - i)/2 + i)

            if i < j:
                i = j
            else:
                i += 1
        keep_index_f1 = np.unique(keep_index_f1).astype('int')
        keep_index_f2 = np.unique(keep_index_f2).astype('int')
     
        keep_index = np.union1d(index_f1[keep_index_f1], index_f2[keep_index_f2]).astype('int')
        return list_f1[keep_index], list_f2[keep_index], list_pt[keep_index]
    
    # This function is used to select a certain number of points evenly from a point list
    def select_n_pts(self, list_f1, list_f2, list_pt, num):
        num_total_pts = len(list_f1)
        if num_total_pts < num:
            return list_f1, list_f2, list_pt
        else:
            index_f1 = np.argsort(list_f1)
            sub_index = np.arange(0, num_total_pts, np.around(num_total_pts/num))
            sub_index_f1 = index_f1[sub_index]
            return list_f1[sub_index_f1], list_f2[sub_index_f1], list_pt[sub_index_f1, :]
    
    # function to return points around region of large hole
    def max_hole_index(self, list_f1, list_f2, num):
        num_pts = len(list_f1)
        if num_pts <= 2*num:
            return range(num_pts)
        
        index_f1 = np.argsort(list_f1)
        index_f2 = np.argsort(list_f2)
        temp_list_f1 = np.sort(list_f1)
        temp_list_f2 = np.sort(list_f2)

        diff_list_f1 = np.diff(temp_list_f1)
        diff_list_f2 = np.diff(temp_list_f2)
        
        diff_index_f1 = np.argsort(-diff_list_f1)
        diff_index_f2 = np.argsort(-diff_list_f2)
        
        count_index = 0
        index = []
        for i in range(num):
            index.append(index_f1[diff_index_f1[i]])
            index.append(index_f1[diff_index_f1[i] + 1])
            index.append(index_f2[diff_index_f2[i]])
            index.append(index_f2[diff_index_f2[i] + 1])

        index = np.unique(index)
        return index
    
## PFSMG for three objectives problem
class Main_SMG_m3:
    ## general parameters setting
    epsion = 1e-4
    max_iter = 1000
    max_len_pareto_front = 1500
    
    # for step size
    stepsize = 0.3
    alpha = 0.5
    step_scheme = 3
    discount_iter_interval = 400 # discount step size every certain iterations

    num_starting_pts = 5
    point_per_iteration = 2 #p1
    num_steps_per_point = 2 #p2
    
    # parameters for getting better spread: explore along first objective, second objective, and maximum hole region
    percent_explore = 0.3 # only explore along single objective for first 30% iterations
    f1_explore_interval = 6 # explore first objective every 6 iterations
    f2_explore_interval = 4 # explore second objective every 4 iterations
    f3_explore_interval = 4 # explore third objective every 4 iterations
    
    f1_explore_pt_per_iter = 1 # do once 2-step stochastic gradient for each point
    f2_explore_pt_per_iter = 1
    f3_explore_pt_per_iter = 1
    
    f1_num_steps_per_point = 2
    f2_num_steps_per_point = 2
    f3_num_steps_per_point = 2
    
    max_hole_only = False # indicate whether or not only do SMG for maximum hole points
    num_max_hole_points = 3 # select three pairs of largest hole points
    max_hole_explore_pt_per_iter = 2 # do twice 2-step SMG for each point
    max_hole_num_steps_per_point = 2
    dense_threshold = 1.0/1000  # the maximum allowed gap between two non-dominated points
    
    # gradient sample batch parameters: batch size will grow exponetially with rate 1.01
    batch1_init = 5
    batch1_factor = 1.01
    batch2_init = 200
    batch2_factor = 1.01
    batch3_init = 200
    batch3_factor = 1.01
    batch1_max = 1.0/3 # indicate maximum batch size, control the computational expense
    batch2_max = 1.0/3  
    batch3_max = 1.0/3  
    
    #samples for pareto_shape function
    num_samples = 1500 
    
    # counter of iterations and gradient evaluation
    num_iter = 0
    num_grad_eval_f1 = 0
    num_grad_eval_f2 = 0
    num_grad_eval_f3 = 0
    
    # construct the class based on a multi-objective formulation
    def __init__(self, func):
        self.func = func
        self.starting_point_list = np.zeros((self.num_starting_pts, self.func.n))
        
    # function to draw the shape of two function value
    def pareto_shape(self):
        f1_value_list = np.zeros(self.num_samples)
        f2_value_list = np.zeros(self.num_samples)
        f3_value_list = np.zeros(self.num_samples)
        
        if self.func.setting == "finite_sum":
            x = np.random.uniform(self.func.lb, self.func.ub, [self.num_samples,self.func.n])
            f1_value_list = self.func.f1(x)
            f2_value_list = self.func.f2(x)
            f3_value_list = self.func.f3(x)
        return f1_value_list, f2_value_list, f3_value_list
    
    # main function of Pareto front stochastic multi-gradient
    def main_SMG(self):
        self.num_iter = 0        
        extreme_index = {"f1": [], "f2": [], "f3": []} 
        
        # generate random starting list
        if self.func.setting == "finite_sum":
            x = np.random.uniform(self.func.lb, self.func.ub, [self.num_starting_pts,self.func.n])
            self.starting_point_list = x
            f1_value_list = self.func.loss(x)
            f2_value_list = self.func.f2(x)
            f3_value_list = self.func.f3(x)
             
        updating_point_list = self.starting_point_list        
        begin = time.time()
        
        # PFSMG is stopped it reaches either maximum iteration or maximum number of nondominated points
        while len(updating_point_list) < self.max_len_pareto_front and self.num_iter < self.max_iter:
            if self.num_iter <= self.percent_explore*self.max_iter and self.num_iter % self.f1_explore_interval == 0:
                extreme_index["f1"] = range(len(updating_point_list))
            else:
                extreme_index["f1"] = [np.argmin(f1_value_list), np.argmax(f1_value_list)]
                
            if self.num_iter <= self.percent_explore*self.max_iter and self.num_iter % self.f2_explore_interval == 0:
                extreme_index["f2"] = range(len(updating_point_list))
            else:
                extreme_index["f2"] = [np.argmin(f2_value_list), np.argmax(f2_value_list)]
                
            if self.num_iter <= self.percent_explore*self.max_iter and self.num_iter % self.f3_explore_interval == 0:
                extreme_index["f3"] = range(len(updating_point_list))
            else:
                extreme_index["f3"] = [np.argmin(f3_value_list), np.argmax(f3_value_list)]
            
            start = time.time()
            pts_considered = self.max_hole_index(f1_value_list, f2_value_list, f3_value_list, self.num_max_hole_points)
            for i in range(len(updating_point_list)):
                curr_x = updating_point_list[i, :]
                if i in extreme_index["f1"]:
                    for j in range(self.f1_explore_pt_per_iter):
                        f1, f2, f3, pt = self.LSMG(curr_x, self.f1_num_steps_per_point, "f1")
                        f1_value_list = np.append(f1_value_list, f1)
                        f2_value_list = np.append(f2_value_list, f2)
                        f3_value_list = np.append(f3_value_list, f3)
                        updating_point_list = np.vstack([updating_point_list, pt.T])
                if i in extreme_index["f2"]:
                    for j in range(self.f2_explore_pt_per_iter):
                        f1, f2, f3, pt = self.LSMG(curr_x, self.f2_num_steps_per_point, "f2")
                        f1_value_list = np.append(f1_value_list, f1)
                        f2_value_list = np.append(f2_value_list, f2)
                        f3_value_list = np.append(f3_value_list, f3)
                        updating_point_list = np.vstack([updating_point_list, pt.T])
                if i in extreme_index["f3"]:
                    for j in range(self.f3_explore_pt_per_iter):
                        f1, f2, f3, pt = self.LSMG(curr_x, self.f3_num_steps_per_point, "f3")
                        f1_value_list = np.append(f1_value_list, f1)
                        f2_value_list = np.append(f2_value_list, f2)
                        f3_value_list = np.append(f3_value_list, f3)
                        updating_point_list = np.vstack([updating_point_list, pt.T])
                if i in pts_considered: 
                    for j in range(self.max_hole_explore_pt_per_iter):
                        f1, f2, f3, pt = self.LSMG(curr_x, self.max_hole_num_steps_per_point, "")
                        f1_value_list = np.append(f1_value_list, f1)
                        f2_value_list = np.append(f2_value_list, f2)
                        f3_value_list = np.append(f3_value_list, f3)
                        updating_point_list = np.vstack([updating_point_list, pt.T])
                else:
                    if not self.max_hole_only:
                        for j in range(self.point_per_iteration):
                            temp = 1.0/100*np.random.uniform(self.func.lb, self.func.ub, self.func.n) 
                            curr_x_perturbed = curr_x + temp
                            f1, f2, f3, pt = self.LSMG(curr_x_perturbed, self.num_steps_per_point, "")
                            f1_value_list = np.append(f1_value_list, f1)
                            f2_value_list = np.append(f2_value_list, f2)
                            f3_value_list = np.append(f3_value_list, f3)
                            updating_point_list = np.vstack([updating_point_list, pt.T])

            f1_value_list, f2_value_list, f3_value_list, updating_point_list = self.clear(f1_value_list, f2_value_list, f3_value_list, updating_point_list)
            f1_value_list, f2_value_list, f3_value_list, updating_point_list = self.remove_dense(f1_value_list, f2_value_list, f3_value_list, updating_point_list)
            dur = time.time() - start
            print ("time: ",  dur)
            self.num_iter += self.num_steps_per_point

            # discount every certain iterations
            if self.step_scheme == 3 and self.num_iter % self.discount_iter_interval == 0:
                self.stepsize = self.stepsize*self.alpha
            print ("#Pts: ", len(f1_value_list), " #Iter: ", self.num_iter)
            
        total_time = time.time() - begin
        print ("Total time: ",  total_time)
                
        return f1_value_list, f2_value_list, f3_value_list, updating_point_list, total_time  
    
    # Stochastic multi-gradient steps
    def LSMG(self, x_k, num_steps, flag):
        beta = 0.02
        gama = 0.5
        normalization = 0
        
        # specify batch size for two functions
        batch = min(int(self.batch1_init*self.batch1_factor**self.num_iter), int(self.batch1_max*self.func.num_data))
        batch2 = min(int(self.batch2_init*self.batch2_factor**self.num_iter), int(self.batch2_max*self.func.num_data))
        batch3 = min(int(self.batch3_init*self.batch3_factor**self.num_iter), int(self.batch3_max*self.func.num_data))
        
        for t in range(num_steps):
            w1 = np.random.choice(self.func.num_data, batch, replace=False)
            w2 = np.random.choice(self.func.num_data, batch2, replace=False)
            w3 = np.random.choice(self.func.num_data, batch3, replace=False)
            
            if flag == "":
                g1 = self.func.g1(x_k, w1)
                g2 = self.func.g2(x_k, w2)
                g3 = self.func.g3(x_k, w3)
                
                self.num_grad_eval_f1 += batch
                self.num_grad_eval_f2 += batch2
                self.num_grad_eval_f3 += batch3
                
                ## compute common descent direction
                if self.func.n == 1: 
                    if g1*g2*g3 > 0:
                        if abs(g1) == np.max([abs(g1), abs(g2), abs(g3)]):
                            g_k = g1
                        elif abs(g2) == np.max([abs(g1), abs(g2), abs(g3)]):
                            g_k = g2
                        else:
                            g_k = g3
                    else:
                        g_k = 0
                else: 
                    G = np.matmul(np.vstack((g1, g2, g3)), np.vstack((g1, g2, g3)).T)
                    a = np.zeros(self.func.m, dtype=np.double)
                    C = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.double).T
                    b = np.array([1, 0, 0, 0], dtype=np.double)
                    xf, f, xu, iters, lagr, iact = solve_qp(G, a, C, b, meq=1)
                    g_k = xf[0]*g1 + xf[1]*g2 + xf[2]*g3
                    
            elif flag == "f1":
                g1 = self.func.g1(x_k, w1)
                self.num_grad_eval_f1 += batch
                g_k = g1
                
            elif flag == "f2":
                g2 = self.func.g2(x_k, w2)
                self.num_grad_eval_f2 += batch2
                g_k = g2
                
            elif flag == "f3":
                g3 = self.func.g3(x_k, w3)
                self.num_grad_eval_f3 += batch3
                g_k = g3
            
            ## take a step
            if self.step_scheme == 0: # fixed step size
                x_k = x_k - self.stepsize*g_k
            elif self.step_scheme == 1: # const/t
                self.stepsize = min([1.0, 2.0/(self.num_iter + 1)])   # min([1 4/np.sqrt(t + 1)]);
                x_k = x_k - self.stepsize*g_k
            elif self.step_scheme == 2: # backtracking strategy
                step = 1
                while step >= self.epsion and (((self.func.f1(x_k - step*g_k) > self.func.f1(x_k) - beta*step*np.dot(g1, g_k)) or (self.func.f2(x_k - step*g_k) > self.func.f2(x_k)))                      or (((self.func.f1(x_k - step*g_k) > self.func.f1(x_k)) or self.func.f2(x_k - step*g_k) > self.func.f2(x_k) - beta*step*np.dot(g2, g_k)))):
                    step = gama*step
                if step < self.epsion:
                    break
                else:
                    self.stepsize = step
                    x_k = x_k - self.stepsize*g_k
            elif self.step_scheme == 3: # discount every certain iterations
                x_k = x_k - self.stepsize*g_k
            
            ## project if a simple bound exists
            if self.func.projection:
                x_k = np.where(x_k <= self.func.ub, x_k, self.func.ub)
                x_k = np.where(x_k >= self.func.lb, x_k, self.func.lb)
            
        return self.func.loss(x_k.reshape(1,-1)), self.func.f2(x_k.reshape(1,-1)), self.func.f3(x_k.reshape(1,-1)), x_k
    
    # clear dominated points at the end of each iteration
    def clear(self, list_f1, list_f2, list_f3, list_pt):
        
        x_bar = np.repeat(list_f1.reshape(-1,1),len(list_f1),axis = 1)
        y_bar = np.repeat(list_f2.reshape(-1,1),len(list_f2),axis = 1)
        z_bar = np.repeat(list_f3.reshape(-1,1),len(list_f3),axis = 1)
        
        array_f1 = list_f1
        array_f2 = list_f2
        array_f3 = list_f3
        array_pt = list_pt        
        
        x_check1 = (array_f1 <= x_bar)
        x_check2 = (array_f1 < x_bar)
        
        y_check1 = (array_f2 <= y_bar)
        y_check2 = (array_f2 < y_bar)
        
        z_check1 = (array_f3 <= z_bar)
        z_check2 = (array_f3 < z_bar)
        
        all_check1 = (x_check1 & y_check1 & z_check2)
        all_check2 = (x_check1 & y_check2 & z_check1)
        all_check3 = (x_check2 & y_check1 & z_check1)
        
        sum1 = all_check1.sum(axis = 1)
        sum2 = all_check2.sum(axis = 1)
        sum3 = all_check3.sum(axis = 1)
        
        rest_index = np.array([i for i in range(len(list_f1)) if (sum1[i] < 1 or sum2[i] < 1 or sum3[i] < 1)])
        return array_f1[rest_index], array_f2[rest_index], array_f3[rest_index], array_pt[rest_index]
    
    # remove non-dominated points from current list from dense region
    def remove_dense(self, list_f1, list_f2, list_f3, list_pt):
        num_total_pts = len(list_f1)
        if self.dense_threshold == 0:
            dense_threshold = 1.0/(800 + self.num_iter/2.0)
        else:
            dense_threshold = self.dense_threshold
    
        index_f1 = np.argsort(list_f1)
        index_f2 = np.argsort(list_f2)
        index_f3 = np.argsort(list_f3)
        
        temp_list_f1 = np.sort(list_f1)
        temp_list_f2 = np.sort(list_f2)
        temp_list_f3 = np.sort(list_f3)
        
        min_gap_f1 = (temp_list_f1[-1] - temp_list_f1[0])*dense_threshold
        max_gap_f2 = (temp_list_f2[-1] - temp_list_f2[0])*dense_threshold
        max_gap_f3 = (temp_list_f3[-1] - temp_list_f3[0])*dense_threshold

        diff_list_f1 = np.diff(temp_list_f1)
        diff_list_f2 = np.diff(temp_list_f2)
        diff_list_f3 = np.diff(temp_list_f3)
        
        keep_index_f1 = [0, num_total_pts - 1]
        keep_index_f2 = [0, num_total_pts - 1]
        keep_index_f3 = [0, num_total_pts - 1]
        
        i = 1
        while i < num_total_pts - 2:# keep the first and last ones
            j = i
            curr_gap = diff_list_f1[j]
            while curr_gap < min_gap_f1 and j < num_total_pts - 2:
                j += 1
                curr_gap += diff_list_f1[j]
            keep_index_f1.append((j - i)/2 + i)

            if i < j:
                i = j
            else:
                i += 1
        i = 1
        while i < num_total_pts - 2:# keep the first and last ones
            j = i
            curr_gap = diff_list_f2[j]
            while curr_gap < max_gap_f2 and j < num_total_pts - 2:
                j += 1
                curr_gap += diff_list_f2[j]
            keep_index_f2.append((j - i)/2 + i)

            if i < j:
                i = j
            else:
                i += 1
        i = 1
        while i < num_total_pts - 2:# keep the first and last ones
            j = i
            curr_gap = diff_list_f3[j]
            while curr_gap < max_gap_f3 and j < num_total_pts - 2:
                j += 1
                curr_gap += diff_list_f3[j]
            keep_index_f3.append((j - i)/2 + i)

            if i < j:
                i = j
            else:
                i += 1
                
        keep_index_f1 = np.unique(keep_index_f1).astype('int')
        keep_index_f2 = np.unique(keep_index_f2).astype('int')
        keep_index_f3 = np.unique(keep_index_f3).astype('int')
        
        keep_index = np.union1d(index_f1[keep_index_f1], index_f2[keep_index_f2]).astype('int')
        keep_index = np.union1d(keep_index, index_f3[keep_index_f3]).astype('int')
        
        return list_f1[keep_index], list_f2[keep_index], list_f3[keep_index], list_pt[keep_index]
    
    # This function is used to select a certain number of points evenly from a point list
    def select_n_pts(self, list_f1, list_f2, list_f3, list_pt, num):
        num_total_pts = len(list_f1)
        if num_total_pts < num:
            return list_f1, list_f2, list_f3, list_pt
        else:
            index_f1 = np.argsort(list_f1)
            sub_index = np.arange(0, num_total_pts, np.around(num_total_pts/num))
            sub_index_f1 = index_f1[sub_index]
            return list_f1[sub_index_f1], list_f2[sub_index_f1], list_f3[sub_index_f1], list_pt[sub_index_f1, :]
    
    # function to return points around region of large hole
    def max_hole_index(self, list_f1, list_f2, list_f3, num):
        num_pts = len(list_f1)
        if num_pts <= 2*num:
            return range(num_pts)
        
        index_f1 = np.argsort(list_f1)
        index_f2 = np.argsort(list_f2)
        index_f3 = np.argsort(list_f3)
        
        temp_list_f1 = np.sort(list_f1)
        temp_list_f2 = np.sort(list_f2)
        temp_list_f3 = np.sort(list_f3)

        diff_list_f1 = np.diff(temp_list_f1)
        diff_list_f2 = np.diff(temp_list_f2)
        diff_list_f3 = np.diff(temp_list_f3)
        
        diff_index_f1 = np.argsort(-diff_list_f1)
        diff_index_f2 = np.argsort(-diff_list_f2)
        diff_index_f3 = np.argsort(-diff_list_f3)
        
        count_index = 0
        index = []
        for i in range(num):
            index.append(index_f1[diff_index_f1[i]])
            index.append(index_f1[diff_index_f1[i] + 1])
            index.append(index_f2[diff_index_f2[i]])
            index.append(index_f2[diff_index_f2[i] + 1])
            index.append(index_f3[diff_index_f3[i]])
            index.append(index_f3[diff_index_f3[i] + 1])

        index = np.unique(index)
        return index
