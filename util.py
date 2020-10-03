#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:15:19 2019

@author: carter
"""

import numpy as np
import shutil
import os
import pickle
import re
import pandas as pd
import collections
import scipy.stats as stats
from scipy.stats import norm
import subprocess
import psutil
import time
import datetime

### rectified linear
def relu(x):
    return x * (x > 0)

### sample random normal
def rmnorm(mean, var_chol, samples = 1):
    N = mean.size
    return (mean.reshape(-1, 1) + var_chol @ np.random.normal(size = (N, samples))).transpose()

def add_constant(x):
    x = [np.ones(x.shape[0]), x]
    return np.column_stack(x)

def ols(X, Y):
    return np.linalg.lstsq(X, Y, rcond = None)[0] # , rcond = None

def ridge_reg(X, Y, lam):
    K = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(K), X.T @ Y)

def big_ridge_reg(X, Y, lam):
    K = X[0].shape[1]
    T = len(X)
    XX = np.zeros((K, K))
    XY = np.zeros((K, 1))
    for tt in range(T):
        XX += X[tt].T @ X[tt] / T
        XY += X[tt].T @ Y[tt] / T
    return np.linalg.solve(XX + lam * np.eye(K), XY).flatten()

### break into roughly n equal pieces
def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]

### prepend string to keys
def prepend2keys(x, string):
    out = {}
    for key, value in x.items():
        out[string + key] = value
    return out

### add slash on back of directory
def add_slash(dirname):
    if dirname[-1] != '/':
        return dirname + '/'
    return dirname

### create directory if not already created
def createDir(dirname):
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except:
            pass

### get pickled data
def read_pickle(file):
    with open(file, 'rb') as fptr:
        #fptr.seek(0)
        data = pickle.load(fptr)
    return data

def write_pickle(x, file):
    with open(file, 'wb') as fptr:
        pickle.dump(x, fptr)

### create check point directory
def create_new_checkpoint_dir(checkdir, epoch):
    checkdir = add_slash(checkdir)
    check = checkdir + 'checkpoint-%d/' % epoch
    createDir(checkdir)
    createDir(check)
    createDir(check + 'working')
    createDir(check + 'parameters')
    createDir(check + 'cov_parameters')
    createDir(check + 'xi_values')
    createDir(check + 'intermediate-results')
    createDir(check + 'results')

def get_current_epoch(checkdir):
    createDir(checkdir)
    files = os.listdir(checkdir)
    if files == []:
        return -1
    eps = []
    for f in files:
        check_match = re.findall(r'checkpoint-\d+', f)
        if len(check_match) > 0:
            eps.append(int(re.findall(r'\d+', f)[0]))
    return np.max(eps)

def reserve_batch(check, batches):
    workdir = check + 'working/'
    findir = check + 'xi_values/'
    working_batches = []
    complete_batches = []
    files = os.listdir(workdir)
    for f in files:
        match = re.findall(r'batch-\d+', f)
        if len(match) > 0:
            working_batches.append(int(re.findall(r'\d+', f)[0]))
    files = os.listdir(findir)
    for f in files:
        match = re.findall(r'batch-\d+', f)
        if len(match) > 0:
            complete_batches.append(int(re.findall(r'\d+', f)[0]))
    if len(working_batches) == batches and len(complete_batches) == batches:
        return -1
    if len(working_batches) == batches:
        possible_batches = list(set(range(batches)) - set(complete_batches))
    else:
        possible_batches = list(set(range(batches)) - set(working_batches))
    batch = np.random.choice(possible_batches)
    with open(workdir + 'batch-%d.txt' % batch, 'w') as fptr:
        fptr.write('\n')
    return batch

def summarize_epoch(check, batches):
    data = pd.DataFrame([read_pickle(check + 'intermediate-results/batch-%d.pkl' % batch)
                        for batch in range(batches)])
    sharpe = np.mean(data['tanexret']) / np.std(data['tanexret'])
    data['sharpe'] = sharpe
    data = dict(data.mean())
    write_pickle(data, check + 'results/epoch-summary.pkl')

def sync_params(check, ptype, batches):
    data = [read_pickle(check + ptype + '/batch-%d.pkl' % batch)
            for batch in range(batches)]
    N = len(data[0])
    outdata = []
    for ii in range(N):
        outdata.append(np.mean([x[ii] for x in data], axis = 0).astype(np.float32))
    write_pickle(outdata, check + 'results/' + ptype + '.pkl')

def clean_data(check, look_back):
    epoch = int(re.findall(r'\d+', check)[-1])
    updir = check + '../'
    files = os.listdir(updir)
    eps = []
    for f in files:
        check_match = re.findall(r'checkpoint-\d+', f)
        if len(check_match) > 0:
            eps.append(int(re.findall(r'\d+', f)[0]))
    for ep in eps:
        if ep < epoch - look_back:
            filedir = updir + 'checkpoint-%d/' % ep
            for addon in ['working', 'parameters', 'cov_parameters', 'xi_values', 'intermediate-results']:
                try:
                    shutil.rmtree(filedir + addon)
                except:
                    pass

def finish_epoch(check, batches):
    summarize_epoch(check, batches)
    sync_params(check, 'parameters', batches)
    sync_params(check, 'cov_parameters', batches)
    clean_data(check, 5)

def get_job(checkdir, epochs, batches):
    epoch = get_current_epoch(checkdir)
    if epoch == -1:
        return 'initialize', 0, 0
    checkdir = add_slash(checkdir)
    check = checkdir + 'checkpoint-%d/' % epoch
    batch = reserve_batch(check, batches)
    if batch > -1:
        return 'next_batch', epoch, batch
    finish_epoch(check, batches)
    epoch += 1
    if epoch > epochs:
        return 'finished', epoch, 0
    create_new_checkpoint_dir(checkdir, epoch)
    check = checkdir + 'checkpoint-%d/' % epoch
    batch = reserve_batch(check, batches)
    return 'next_batch', epoch, batch

def initialize_checkpoints(checkdir, param_list, cov_param_list, xi_vars):
    checkdir = add_slash(checkdir)
    create_new_checkpoint_dir(checkdir, 0)
    check = checkdir + 'checkpoint-0/'
    write_pickle(param_list, check + 'results/' + 'parameters' + '.pkl')
    write_pickle(cov_param_list, check + 'results/' + 'cov_parameters' + '.pkl')
    for batch, xi_arr in enumerate(xi_vars):
        write_pickle(xi_arr, check + 'xi_values/batch-%d.pkl' % batch)
    create_new_checkpoint_dir(checkdir, 1)

def get_last_checkpoint(checkdir, epoch, batch):
    checkdir = add_slash(checkdir)
    check = checkdir + 'checkpoint-%d/' % (epoch-1)
    param_list = read_pickle(check + 'results/' + 'parameters' + '.pkl')
    cov_param_list = read_pickle(check + 'results/' + 'cov_parameters' + '.pkl')
    xi = read_pickle(check + 'xi_values/batch-%d.pkl' % batch)
    return param_list, cov_param_list, xi

def save_batch(checkdir, epoch, batch, param_list, cov_param_list, xi, summary):
    checkdir = add_slash(checkdir)
    check = checkdir + 'checkpoint-%d/' % epoch
    write_pickle(param_list, check + 'parameters/batch-%d.pkl' % batch)
    write_pickle(cov_param_list, check + 'cov_parameters/batch-%d.pkl' % batch)
    write_pickle(xi, check + 'xi_values/batch-%d.pkl' % batch)
    write_pickle(summary, check + 'intermediate-results/batch-%d.pkl' % batch)

### process data
class rz_data(object):
    def __init__(self):
        self.initialized = False
        self.Ns = []
        self.N = 0
        self.T = 0
        self.D = 0
        self.dates = []
        self.ids = []
        self.Z = []
        self.R = []
    def initialize_pandas(self, data, date_name, id_name, return_name, z_names):
        self.Z = []
        self.R = []
        self.ids = []
        self.D = len(z_names)
        self.dates = sorted(list(data.loc[:, date_name].drop_duplicates()))
        self.T = len(self.dates)
        self.Nmax = 0
        self.Ns = []
        for tt, date in enumerate(self.dates):
            select = date == data[date_name]
            N = np.sum(select)
            self.R.append(data.loc[select, return_name].values.astype(np.float32))
            self.Z.append(data.loc[select, z_names].values.astype(np.float32).reshape((-1, self.D)))
            self.ids.append(data.loc[select, id_name].values)
            self.Ns.append(N)
            if N > self.Nmax:
                self.Nmax = N
        self.balanced = False
        self.initialized = True
    def initialize_list(self, Z, R, dates, ids):
        self.Z = Z
        self.R = R
        self.ids = ids
        self.dates = dates
        self.D = Z[0].shape[1]
        self.T = len(Z)
        self.Nmax = 0
        self.Ns = []
        for tt, date in enumerate(self.dates):
            N = len(R[tt])
            self.Ns.append(N)
            if N > self.Nmax:
                self.Nmax = N
        self.balanced = False
        self.initialized = True
    def balance_jagged(self):
        if not self.balanced:
            self.balanced = True
            self.A = []
            for tt, date in enumerate(self.dates):
                addon = self.Nmax - self.Ns[tt]
                self.R[tt] = np.pad(self.R[tt], [(0, addon)], mode = 'constant')
                self.Z[tt] = np.pad(self.Z[tt], [(0, addon), (0, 0)], mode = 'constant')
                self.A.append(np.concatenate((np.ones(self.Ns[tt]), np.zeros(addon))).astype(np.float32))
    def make_jagged(self):
        if self.balanced:
            for tt, date in enumerate(self.dates):
                select = self.A[tt] > 0.5
                self.R[tt] = self.R[tt][select]
                self.Z[tt] = self.Z[tt][select]
            del self.A
            self.balanced = False
    def append_list(self, Z, R, dates, ids):
        if self.initialized == False:
            self.initialize_list(Z, R, dates, ids)
        else:
            self.T += len(Z)
            self.dates += dates
            self.ids += ids
            self.Z += Z
            self.R += R
            for tt in range(len(Z)):
                N = len(R[tt])
                self.Ns.append(N)
                if N > self.Nmax:
                    self.Nmax = N
    def append_list_balanced(self, Z, R, A, dates, ids):
        self.append_list(Z, R, dates, ids)
        self.A += A
    def split_data(self, split_date, strictly_less_than = False):
        first = rz_data()
        second = rz_data()
        if strictly_less_than == False:
            first_dates = [ii for ii, dd in enumerate(self.dates) if dd <= split_date]
        else:
            first_dates = [ii for ii, dd in enumerate(self.dates) if dd < split_date]
        if len(first_dates) == 0:
            return first, self
        split_index = np.max(first_dates)+1
        if split_index == len(self.dates):
            return self, second
        first.initialize_list(self.Z[:split_index], self.R[:split_index], 
                              self.dates[:split_index], self.ids[:split_index])
        second.initialize_list(self.Z[split_index:], self.R[split_index:], 
                               self.dates[split_index:], self.ids[split_index:])
        return first, second
    def append_data(self, data):
        self.append_list(data.Z, data.R, data.dates, data.ids)
    def extract_fold(self, fold, folds):
        xv_sample_beg = self.dates[(fold-1) * self.T // folds]
        xv_sample_end = self.dates[fold * self.T // folds - 1]
        train1, second = self.split_data(xv_sample_beg, strictly_less_than = True)
        xv, train2 = second.split_data(xv_sample_end)
        train1.append_data(train2)
        return train1, xv

### process data
class rzw_data(object):
    def __init__(self):
        self.initialized = False
        self.Ns = []
        self.N = 0
        self.T = 0
        self.D = 0
        self.dates = []
        self.ids = []
        self.Z = []
        self.R = []
        self.W = []
    def initialize_pandas(self, data, date_name, id_name, return_name, z_names, w_name):
        self.Z = []
        self.R = []
        self.ids = []
        self.D = len(z_names)
        self.dates = sorted(list(data.loc[:, date_name].drop_duplicates()))
        self.T = len(self.dates)
        self.Nmax = 0
        self.Ns = []
        for tt, date in enumerate(self.dates):
            select = date == data[date_name]
            N = np.sum(select)
            self.W.append(data.loc[select, w_name].values.astype(np.float32))
            self.R.append(data.loc[select, return_name].values.astype(np.float32))
            self.Z.append(data.loc[select, z_names].values.astype(np.float32).reshape((-1, self.D)))
            self.ids.append(data.loc[select, id_name].values)
            self.Ns.append(N)
            if N > self.Nmax:
                self.Nmax = N
        self.balanced = False
        self.initialized = True
    def initialize_list(self, Z, R, W, dates, ids):
        self.Z = Z
        self.R = R
        self.W = W
        self.ids = ids
        self.dates = dates
        self.D = Z[0].shape[1]
        self.T = len(Z)
        self.Nmax = 0
        self.Ns = []
        for tt, date in enumerate(self.dates):
            N = len(R[tt])
            self.Ns.append(N)
            if N > self.Nmax:
                self.Nmax = N
        self.balanced = False
        self.initialized = True
    def balance_jagged(self):
        if not self.balanced:
            self.balanced = True
            self.A = []
            for tt, date in enumerate(self.dates):
                addon = self.Nmax - self.Ns[tt]
                self.W[tt] = np.pad(self.W[tt], [(0, addon)], mode = 'constant')
                self.R[tt] = np.pad(self.R[tt], [(0, addon)], mode = 'constant')
                self.Z[tt] = np.pad(self.Z[tt], [(0, addon), (0, 0)], mode = 'constant')
                self.A.append(np.concatenate((np.ones(self.Ns[tt]), np.zeros(addon))).astype(np.float32))
    def make_jagged(self):
        if self.balanced:
            for tt, date in enumerate(self.dates):
                select = self.A[tt] > 0.5
                self.R[tt] = self.R[tt][select]
                self.W[tt] = self.W[tt][select]
                self.Z[tt] = self.Z[tt][select]
            del self.A
            self.balanced = False
    def append_list(self, Z, R, W, dates, ids):
        if self.initialized == False:
            self.initialize_list(Z, R, dates, ids)
        else:
            self.T += len(Z)
            self.dates += dates
            self.ids += ids
            self.Z += Z
            self.R += R
            self.W += W
            for tt in range(len(Z)):
                N = len(R[tt])
                self.Ns.append(N)
                if N > self.Nmax:
                    self.Nmax = N
    def split_data(self, split_date, strictly_less_than = False):
        first = rz_data()
        second = rz_data()
        if strictly_less_than == False:
            first_dates = [ii for ii, dd in enumerate(self.dates) if dd <= split_date]
        else:
            first_dates = [ii for ii, dd in enumerate(self.dates) if dd < split_date]
        if len(first_dates) == 0:
            return first, self
        split_index = np.max(first_dates)+1
        if split_index == len(self.dates):
            return self, second
        first.initialize_list(self.Z[:split_index], self.R[:split_index], 
                              self.dates[:split_index], self.ids[:split_index])
        second.initialize_list(self.Z[split_index:], self.R[split_index:], 
                               self.dates[split_index:], self.ids[split_index:])
        return first, second
    def append_data(self, data):
        self.append_list(data.Z, data.R, data.dates, data.ids)
    def extract_fold(self, fold, folds):
        xv_sample_beg = self.dates[(fold-1) * self.T // folds]
        xv_sample_end = self.dates[fold * self.T // folds - 1]
        train1, second = self.split_data(xv_sample_beg, strictly_less_than = True)
        xv, train2 = second.split_data(xv_sample_end)
        train1.append_data(train2)
        return train1, xv

### continuous percentile function
def kernel_ranking(x):
    xstd = np.std(x)
    N = len(x)
    bandwidth = (4. / (3. * N))**(1. / 5.) * xstd
    y = x.reshape(-1, 1)
    return np.mean(stats.norm.cdf((y - y.T) / bandwidth), 1) - 0.5

def percentile(x):
    y = x.reshape(-1, 1)
    return np.mean((y - y.T) > 0, 1)

### take percentiles of rows
def percentile_rows(x):
    return np.apply_along_axis(percentile, 1, x)

def char2weights(Z, first_market = True):
    market = Z[:, 0]
    Z = (Z - np.mean(Z, 0).reshape(1, -1)) / np.std(Z, 0)
    Z = norm.cdf(Z) - 0.5
    Z = Z / np.sum(np.abs(Z)).reshape(1, -1)
    if first_market:
        Z[:, 0] = market / np.sum(market)
    return Z

def summary_percentiles(x):
    return dict(zip([1, 2, 3, 4, 5, 25, 50, 75, 95, 96, 97, 98, 99], 
                       np.percentile(x, [1, 2, 3, 4, 5, 25, 50, 75, 95, 96, 97, 98, 99])))

def get_hostname():
    process = subprocess.Popen('hostname', stdout=subprocess.PIPE)
    output, error = process.communicate()
    return str(output)

def is_readable_pickle(x):
    if not os.path.exists(x):
        return False
    try:
        read_pickle(x)
        return True
    except:
        return False
    return False

def memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30
    return memoryUse

def chunk_map(total_chunks, reps_each = 1000, parn = 1):
    int_div = reps_each * parn // total_chunks
    reps = np.repeat(int_div, total_chunks)
    modchnks = np.mod(np.arange(total_chunks), parn)
    un, cnts = np.unique(modchnks, return_counts = True)
    sums = cnts * int_div
    for chnk, ss in zip(un, sums):
        adds = reps_each - ss
        if adds > 0:
            reps[np.where(modchnks == chnk)[0][:adds]] += 1
        elif adds < 0:
            reps[np.where(modchnks == chnk)[0][:np.abs(adds)]] -= 1  
    return modchnks, reps

class timekeeper(object):
    def __init__(self, iterations):
        self.iterations = iterations
        self.iters_complete = 0
        self.start = time.time()
    def restart(self):
        self.start = time.time()
    def increment(self):
        self.iters_complete += 1
        self.iters_left = self.iterations - self.iters_complete
        self.now = time.time()
        self.avg_time_per_iter = (self.now - self.start) / self.iters_complete
        self.seconds_left = self.avg_time_per_iter * self.iters_left
        self.finish_time = datetime.datetime.fromtimestamp(self.now + self.seconds_left)
        self.finish_string = f"{self.finish_time:%Y-%m-%d, %H:%M:%S}"
    def print_time(self):
        print('Time Left:')
        print(datetime.timedelta(seconds = self.seconds_left))
        print('Estimated Finish Time:')
        print(self.finish_string)
        print('\n')

