#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:03:43 2020

@author: carter
"""

import numpy as np
import scipy.optimize as so
from collections import Sequence
from itertools import chain, count
from sklearn.ensemble import RandomForestRegressor

def list_depth(seq):
    seq = iter(seq)
    try:
        for level in count():
            seq = chain([next(seq)], seq)
            seq = chain.from_iterable(s for s in seq if isinstance(s, Sequence))
    except StopIteration:
        return level

def list_2deep(lst):
    depth = list_depth(lst)
    if depth == 1:
        return [lst]
    elif depth == 2:
        return lst
    return list_2deep([item for sublist in lst for item in sublist])

class covariance_model(object):
    def __init__(self, left_sample, right_sample, not_missing, 
                 fixed_betas, y):
        self.left_sample = left_sample
        self.right_sample = right_sample
        self.not_missing = not_missing
        self.fixed_betas = self.not_missing * (1 - self.left_sample) * (1 - self.right_sample) * fixed_betas
        self.y = y
        self.T, self.N = self.y.shape
        self.xbf = np.sum(self.not_missing * self.y * self.fixed_betas, 1)
        self.xl = np.sum(self.not_missing * self.y * self.left_sample, 1)
        self.xr = np.sum(self.not_missing * self.y * self.right_sample, 1)
        self.Nt = np.sum(self.not_missing, 1)
        self.Nl = np.sum(self.not_missing * self.left_sample, 1)
        self.Nr = np.sum(self.not_missing * self.right_sample, 1)
        self.xx = np.sum(self.not_missing * self.y**2, 1)
        self.bfbf = np.sum(self.not_missing * self.fixed_betas**2, 1)
        self.NT = np.sum(self.Nt)
        self.N2T = np.sum(self.Nt**2)
        self.af = np.sum(self.xx - self.bfbf) / self.NT
        self.al = -np.sum(self.Nl) / self.NT
        self.ar = -np.sum(self.Nr) / self.NT
    def calc_zeta(self, beta):
        betal, betar = beta
        return self.af + self.al * betal**2 + self.ar * betar**2
    def loss(self, beta):
        betal, betar = beta
        bb = self.bfbf + self.Nl * betal**2 + self.Nr * betar**2
        bx = self.xbf + self.xl * betal + self.xr * betar
        zeta = self.af + self.al * betal**2 + self.ar * betar**2
        return np.sum(self.xx**2 - 2 * bx**2 + bb**2 + zeta**2 * self.Nt - 2 * zeta * self.xx
                      + 2 * zeta * bb) / self.N2T 
    def loss_deriv(self, beta):
        betal, betar = beta
        bb = self.bfbf + self.Nl * betal**2 + self.Nr * betar**2
        bx = self.xbf + self.xl * betal + self.xr * betar
        zeta = self.af + self.al * betal**2 + self.ar * betar**2
        dzeta_dbetal = 2 * self.al * betal
        dzeta_dbetar = 2 * self.ar * betar
        df_dl = np.sum(-4 * bx * self.xl + 4 * bb * betal * self.Nl + 2 * zeta * dzeta_dbetal * self.Nt
                       - 2 * self.xx * dzeta_dbetal
                       + 2 * dzeta_dbetal * bb + 4 * zeta * betal * self.Nl) 
        df_dr = np.sum(-4 * bx * self.xr + 4 * bb * betar * self.Nr + 2 * zeta * dzeta_dbetar * self.Nt
                       - 2 * self.xx * dzeta_dbetar
                       + 2 * dzeta_dbetar * bb + 4 * zeta * betar * self.Nr) 
        return np.array([df_dl, df_dr]) / self.N2T
    def fit(self, x0):
        #methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 
        #           'SLSQP', 'Newton-CG']
        methods = ['Nelder-Mead', 'BFGS']
        jac_methods = ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP', 'Newton-CG']
        losses = [np.inf]
        sols = [[x0, x0]]
        for method in methods:
            jac = None
            options = None
            if method == 'Nelder-Mead':
                options = {'maxiter': 200}
            if method in jac_methods:
                jac = self.loss_deriv
            res = so.minimize(self.loss, 
                              x0 = sols[np.argmin(losses)], 
                              jac = jac,
                              method = method, 
                              options = options)
            losses.append(res.fun)
            sols.append(res.x)
        solution = sols[np.argmin(losses)]
        #print()
        #print(methods)
        #print(losses)
        #print(sols)
        #print('Best method:', methods[np.argmin(losses[1:])])
        #print()
        #print('\n')
        #print('Optimization Complete:')
        #print(solution)
        #print(self.loss(solution))
        #print(('zeta:', self.calc_zeta(solution)))
        #print([x0, x0])
        #print(self.loss([x0, x0]))
        #print(('zeta:', self.calc_zeta([x0, x0])))
        #print('\n\n')
        return solution

class Node(object):
    def __init__(self, depth = 0):
        self.depth = depth
        self.split_variable = None
        self.split_value = None
        self.left_beta = None
        self.right_beta = None
        self.left_child = None
        self.right_child = None
    def fit(self, sample, not_missing, fixed_betas, X, y, max_depth,
            features = None, above_beta = 0.):
        K = X.shape[2]
        if features is None:
            features = np.arange(K)
        losses = np.zeros(len(features))
        left_betas = np.zeros(len(features))
        right_betas = np.zeros(len(features))
        for ii, kk in enumerate(features):
            xvals = X[:, :, kk][sample]
            median = np.median(xvals)
            left_sample = np.logical_and(sample, X[:, :, kk] <= median)
            right_sample = np.logical_and(sample, X[:, :, kk] > median)
            cov_mod = covariance_model(left_sample, right_sample, not_missing, 
                                           fixed_betas, y)
            if np.sum(left_sample) > 0 and np.sum(right_sample) > 0:
                left_beta, right_beta = cov_mod.fit(above_beta)
                left_betas[ii] = left_beta
                right_betas[ii] = right_beta
            else:
                left_betas[ii] = above_beta
                right_betas[ii] = above_beta
            losses[ii] = cov_mod.loss([left_betas[ii], right_betas[ii]])
        argmin = np.argmin(losses)
        self.split_variable = features[argmin]
        xvals = X[:, :, self.split_variable][sample]
        self.split_value = np.median(xvals)
        self.left_beta = left_betas[argmin]
        self.right_beta = right_betas[argmin]
        if self.depth < max_depth:
            left_sample = np.logical_and(sample, X[:, :, self.split_variable] <= self.split_value)
            right_sample = np.logical_and(sample, X[:, :, self.split_variable] > self.split_value)
            left_fixed_betas = not_missing * (1 - right_sample) * fixed_betas + right_sample * self.right_beta
            right_fixed_betas = not_missing * (1 - left_sample) * fixed_betas + left_sample * self.left_beta
            self.left_child = Node(self.depth + 1)
            self.left_child.fit(left_sample, not_missing, left_fixed_betas, X, y, 
                                max_depth, features, self.left_beta)
            self.right_child = Node(self.depth + 1)
            self.right_child.fit(right_sample, not_missing, right_fixed_betas, X, y, 
                                 max_depth, features, self.right_beta)
    def build_empty(self, max_depth):
        if self.depth < max_depth:
            self.left_child = Node(self.depth + 1)
            self.right_child = Node(self.depth + 1)
            self.left_child.build_empty(max_depth)
            self.right_child.build_empty(max_depth)
    def get_paths(self, paths = []):
        if self.left_child is not None:
            left_path = self.left_child.get_paths(paths + [0])
            right_path = self.right_child.get_paths(paths + [1])
            left_path = list_2deep(left_path)
            right_path = list_2deep(right_path)
            return left_path + right_path
        return paths
    def walk_through(self, path):
        if self.depth == len(path):
            return (path, self.left_beta, self.right_beta)
        if path[self.depth] == 0:
            return self.left_child.walk_through(path)
        return self.right_child.walk_through(path)
    def predict_path(self, sample, X, path):
        left_sample = np.logical_and(sample, X[:, :, self.split_variable] <= self.split_value)
        right_sample = np.logical_and(sample, X[:, :, self.split_variable] > self.split_value)
        if self.depth == len(path):
            return left_sample * self.left_beta + right_sample * self.right_beta
        if path[self.depth] == 0:
            return self.left_child.predict_path(left_sample, X, path)
        return self.right_child.predict_path(right_sample, X, path)
    def predict(self, not_missing, X):
        T, N, K = X.shape
        preds = np.zeros((T, N))
        paths = self.get_paths()
        if paths == []:
            return self.predict_path(not_missing, X, paths)
        for path in paths:
            preds = preds + self.predict_path(not_missing, X, path)
        return preds

class covariance_random_forest:
    def __init__(self, n_estimators = 100, max_depth = 2, random_state = None,
                 print_iter = False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.print_iter = print_iter
    def fit(self, Z, r, not_missing):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        T, N, K = Z.shape
        self.roots = []
        self.zetas = []
        for ii in range(self.n_estimators):
            if self.print_iter:
                print('Working on Covariance Model: %d' % (ii+1))
            features = np.random.choice(K, size = max(1, int(round(np.sqrt(K)))), 
                                        replace = False)
            Ts = np.random.choice(T, size = T, replace = True)
            sample_nm = not_missing[Ts, :]
            sample_Z = Z[Ts, :, :]
            sample_r = r[Ts, :]
            root = Node()
            root.fit(sample_nm, sample_nm, np.zeros((T, N)),
                     sample_Z, sample_r, self.max_depth, features)
            cov_mod = covariance_model(np.zeros(r.shape), np.zeros(r.shape),
                                       not_missing, root.predict(sample_nm, sample_Z), sample_r)
            zeta = np.maximum(1e-04, cov_mod.calc_zeta([0, 0]))
            self.roots.append(root)
            self.zetas.append(zeta)
            if self.print_iter:
                print('Covariance Fit Number: %d' % (ii+1))
        self.mean_zeta = np.mean(self.zetas)
    def predict(self, Z, not_missing):
        beta = np.zeros(not_missing.shape)
        for ii in range(self.n_estimators):
            beta += not_missing * self.roots[ii].predict(not_missing, Z) / self.n_estimators
        return beta, self.mean_zeta

class PortfolioRandomForest:
    def __init__(self, Z = None, r = None, not_missing = None,
                 mean_args = {'n_estimators': 100, 'max_depth': 8, 'max_features': 'sqrt'}, # 'max_depth': 3, 'max_features': 47
                 cov_n_estimators = 100, cov_max_depth = 3, use_oob = True,
                 print_iter = True):
        self.Z = Z
        self.r = r
        self.not_missing = not_missing
        if self.Z is not None:
            self.T, self.N, self.K = self.Z.shape
        self.mean_args = mean_args
        self.cov_n_estimators = cov_n_estimators
        self.cov_max_depth = cov_max_depth
        self.use_oob = use_oob
        self.print_iter = print_iter
    def set_data(self, Z, r, not_missing):
        self.Z = Z
        self.r = r
        self.not_missing = not_missing
        self.T, self.N, self.K = self.Z.shape
    def purge_data(self):
        #del self.Z
        #del self.r
        #del self.not_missing
        #del self.T
        #del self.N
        self.Z = None
        self.r = None
        self.not_missing = None
    def fit_mean(self):
        self.mean_model = RandomForestRegressor(**self.mean_args, oob_score = self.use_oob)
        if self.print_iter:
            print('Fitting Mean')
        self.mean_model.fit(self.Z[self.not_missing, :], self.r[self.not_missing])
        if self.print_iter:
            print('Mean Fitting Complete')
        if self.use_oob:
            mean_prediction = self.mean_model.oob_prediction_
        else:
            mean_prediction = self.mean_model.predict(self.Z[self.not_missing, :])
        self.mean_prediction = np.zeros((self.T, self.N))
        self.mean_prediction[self.not_missing] = mean_prediction
        self.r_demean = self.not_missing * (self.r - self.mean_prediction)
    def fit_cov(self):
        self.cov_model = covariance_random_forest(self.cov_n_estimators, self.cov_max_depth, 
                                                  print_iter = self.print_iter)
        self.cov_model.fit(self.Z, self.r, self.not_missing)
    def fit(self, Z = None, r = None, not_missing = None,
            purge_data = True):
        if Z is not None:
            self.set_data(Z, r, not_missing)
        self.fit_mean()
        self.fit_cov()
        if purge_data:
            self.purge_data()
    def parameter_predict(self, Z, not_missing = None):
        if len(Z.shape) == 2:
            Z = np.expand_dims(Z, 0)
            if not_missing is not None:
                not_missing = np.expand_dims(not_missing, 0)
        T, N, K = Z.shape
        if not_missing is None:
            not_missing = np.ones((T, N))
        not_missing = not_missing > 0.5
        mu = np.zeros((T, N))
        mu[not_missing] = self.mean_model.predict(Z[not_missing, :])
        Gamma, zeta = self.cov_model.predict(Z, not_missing)
        return mu, Gamma, zeta
    def predict_weights(self, Z, not_missing):
        mu, Gamma, zeta = self.parameter_predict(Z, not_missing)
        denoms = np.sum(Gamma**2, 1, keepdims = True) + zeta
        Gamma_mu = np.sum(Gamma * mu, 1, keepdims = True)
        return mu / zeta - Gamma * Gamma_mu / (zeta * denoms)
    def predict(self, Z, r, not_missing):
        weights = self.predict_weights(Z, not_missing)
        f = np.sum(weights * r, 1)
        return {'w': weights, 'tangency': f}
    def gen_random_sample(self, Z, not_missing, random_state = None):
        T, N, K = Z.shape
        if random_state is not None:
            np.random.normal(random_state)
        mu, Gamma, zeta = self.parameter_predict(Z, not_missing)
        X = mu + Gamma * np.random.normal(size = (T, 1)) + np.sqrt(zeta) * np.random.normal(size = (T, N))
        X = X * not_missing
        return X

