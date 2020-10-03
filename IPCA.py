#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 09:53:58 2018

@author: carter
"""

import numpy as np
import numpy.linalg as linalg
import scipy.linalg as sl

def add_constant(x):
    x = [np.ones(x.shape[0]), x]
    return np.column_stack(x)

def solve(A, b):
    if b.size == 1:
        return np.array([float(b) / float(A)])
    try:
        return sl.solve(A, b)
    except:
        pass
    x, _, _, _ = sl.lstsq(A, b)
    return x

def add_tensor_constant(X):
    T, N, K = X.shape
    const = np.ones((T, N, 1))
    return np.concatenate((const, X), axis = 2)

def initial_pca(Z, r, nfac = 5):
    X = np.matmul(np.transpose(Z, [0, 2, 1]), r)
    second_moment = np.mean(np.matmul(X, np.transpose(X, [0, 2, 1])), 0)
    lam, v = linalg.eig(second_moment)
    v = np.real(v)
    sorted_indices = np.argsort(lam)[::-1]
    v = v[:, sorted_indices]
    return v[:, :nfac]

def estimate_factors(Z, r, Gamma):
    T, N, K = Z.shape
    Gamma = np.tile(np.expand_dims(Gamma, 0), (T, 1, 1))
    ZG = np.matmul(Z, Gamma)
    ZGpZG = np.matmul(np.transpose(ZG, [0, 2, 1]), ZG)
    ZGpY = np.matmul(np.transpose(ZG, [0, 2, 1]), r)
    return linalg.solve(ZGpZG, ZGpY)

def estimate_Gamma(Z, r, f):
    T, N, K = Z.shape
    ZpZ = np.matmul(np.transpose(Z, [0, 2, 1]), Z)
    ffp = np.matmul(f, np.transpose(f, [0, 2, 1]))
    Zr = np.matmul(np.transpose(Z, [0, 2, 1]), r)
    ZZff = np.mean(np.stack([np.kron(ZpZ[t, :, :], ffp[t, :, :]) for t in range(T)]), 0)
    Zfr = np.mean(np.stack([np.kron(Zr[t, :, :], f[t, :, :]) for t in range(T)]), 0)
    vecGamma = solve(ZZff, Zfr)
    return vecGamma.reshape((K, -1))

def estimate_w(Z, Gamma):
    T, N, K = Z.shape
    Gamma = np.tile(np.expand_dims(Gamma, 0), (T, 1, 1))
    ZG = np.matmul(Z, Gamma)
    ZGpZG = np.matmul(np.transpose(ZG, [0, 2, 1]), ZG)
    return linalg.solve(ZGpZG, np.transpose(ZG, [0, 2, 1]))

def calc_loss(Z, r, not_missing, Gamma, f):
    T, N, K = Z.shape
    Gamma = np.tile(np.expand_dims(Gamma, 0), (T, 1, 1))
    pred = np.matmul(np.matmul(Z, Gamma), f)
    squared_error = not_missing * np.reshape((r - pred)**2, (T, N))
    return np.mean(np.sum(squared_error, 1) / np.sum(not_missing, 1))

def naive_tangency(R):
    Sigma = np.cov(R, rowvar = False)
    mu = np.mean(R, axis = 0)
    b = solve(Sigma, mu)
    b = b.flatten()
    return b #/ np.sum(b)

def Zr_mult(Z, r):
    T, N, K = Z.shape
    return np.matmul(np.transpose(Z, [0, 2, 1]), r).reshape((T, K))

def calc_var(Z, r):
    F = Zr_mult(Z, r)
    return np.cov(F, rowvar=False)

class IPCA:
    def __init__(self, nfac = 5, verbose = False, 
                 tol = 1e-08, maxiter = 10000, add_cons = True):
        self.nfac = nfac
        self.verbose = verbose
        self.tol = tol
        self.maxiter = maxiter
        self.add_cons = add_cons
    def set_data(self, Z, r, not_missing):
        if self.add_cons:
            self.Z = add_tensor_constant(Z) * np.expand_dims(not_missing, axis = 2)
        else:
            self.Z = Z * np.expand_dims(not_missing, axis = 2)
        self.not_missing = not_missing
        if len(r.shape) == 2:
            self.r = np.expand_dims(r * not_missing, axis = 2)
        else:
            self.r = r * np.expand_dims(not_missing, axis = 2)
        self.T, self.N, self.K = self.Z.shape
    def fit(self, Z = None, r = None, not_missing = None):
        if Z is not None and r is not None and not_missing is not None:
            self.set_data(Z, r, not_missing)
        self.Gamma = initial_pca(self.Z, self.r, self.nfac)
        self.f = estimate_factors(self.Z, self.r, self.Gamma)
        lastloss = calc_loss(self.Z, self.r, self.not_missing, self.Gamma, self.f)
        self.loss = [lastloss]
        for ii in range(self.maxiter):
            if self.verbose:
                print('\t\tCalculating Gamma on Iteration: %d' % (ii+1))
            self.Gamma = estimate_Gamma(self.Z, self.r, self.f)
            if self.verbose:
                print('\t\tCalculating Factors on Iteration: %d' % (ii+1))
            self.f = estimate_factors(self.Z, self.r, self.Gamma)
            if self.verbose:
                print('\t\tCalculating Loss on Iteration: %d' % (ii+1))
            iterloss = calc_loss(self.Z, self.r, self.not_missing, self.Gamma, self.f)
            self.loss.append(iterloss)
            if self.verbose:
                print('Loss on Iteration %d: %f' % (ii+1, iterloss))
            if lastloss - iterloss < self.tol:
                break
            lastloss = iterloss
        self.fit_factor_tangency()
        return self.Gamma, self.tangency_factor_weights
    def fit_factor_tangency(self):
        T = self.f.shape[0]
        self.tangency_factor_weights = naive_tangency(np.reshape(self.f, (T, -1)))
    def set_model_parameters(self, Gamma):
        self.Gamma = Gamma
    def set_model_variables(self, Gamma):
        self.Gamma = Gamma
        self.f = estimate_factors(self.Z, self.r, self.Gamma)
        lastloss = calc_loss(self.Z, self.r, self.not_missing, self.Gamma, self.f)
        self.loss = [lastloss]
    def get_model_variables(self):
        return self.Gamma
    def predict(self, Z, r, not_missing):
        if self.add_cons:
            Z = add_tensor_constant(Z) * np.expand_dims(not_missing, axis = 2)
        if len(r.shape) == 2:
            r = np.expand_dims(r * not_missing, axis = 2)
        T, N, K = Z.shape
        w = estimate_w(Z, self.Gamma)
        tangency_factor_weights = np.tile(np.expand_dims(np.expand_dims(self.tangency_factor_weights, 0), 0), (T, 1, 1))
        tangency_weights = np.reshape(np.matmul(tangency_factor_weights, w), (T, N))
        tangency_weights = tangency_weights * not_missing
        #tangency_weights = tangency_weights / np.abs(np.sum(tangency_weights, axis = 1, keepdims = True))
        tangency = np.sum(tangency_weights * np.reshape(r, (T, N)), 1)
        factors = np.matmul(w, r)
        Gamma = np.tile(np.expand_dims(self.Gamma, 0), (T, 1, 1))
        ZG = np.matmul(Z, Gamma)
        betas = np.transpose(ZG, [0, 2, 1])
        rpred = np.matmul(ZG, factors)
        return {'rpred': rpred, 'factors': factors, 'factor_w': w, 
                'betas': betas, 'w': tangency_weights, 'tangency': tangency}
    def cross_validate(self, Z = None, r = None, not_missing = None,
                       folds = 4, maxfac = 20,
                       print_iter = False,
                       tol = None, maxiter = None):
        if Z is not None and r is not None and not_missing is not None:
            self.set_data(Z, r, not_missing)
        if tol is None:
            tol = self.tol
        if maxiter is None:
            maxiter = self.maxiter
        rank = np.linalg.matrix_rank(calc_var(self.Z, self.r))
        maxfac = min(maxfac, rank)
        if maxfac == 1:
            self.nfac = 1
            return self.nfac
        nfacs = list(range(1, maxfac+1))
        means = np.zeros((folds, maxfac))
        variances = np.zeros((folds, maxfac))
        samples = [[tt for tt in range(self.T) 
                    if int(folds * tt / self.T) == fold]
                    for fold in range(folds)]
        for fold in range(folds):
            fold_sample = np.concatenate([samp for kk, samp in enumerate(samples) if kk != fold])
            xv_sample = samples[fold]
            fold_model = IPCA(nfac = 1, verbose = False, 
                              tol = tol, maxiter = maxiter, 
                              add_cons = False)
            fold_model.set_data(self.Z[fold_sample, :, :], self.r[fold_sample, :],
                                self.not_missing[fold_sample, :])
            for ii, nfac in enumerate(nfacs):
                fold_model.nfac = nfac
                fold_model.fit()
                fold_fits = fold_model.predict(self.Z[xv_sample, :, :],
                       self.r[xv_sample, :], self.not_missing[xv_sample, :])
                fold_tan = fold_fits['tangency']
                means[fold, ii] = np.mean(fold_tan)
                variances[fold, ii] = np.var(fold_tan)
                if print_iter:
                    print(('Cross Validation:', fold, ii, means[fold, ii] / np.sqrt(variances[fold, ii])))
            del fold_model
        means = np.mean(means, axis = 0)
        variances = np.mean(variances, axis = 0)
        sharpes = means / np.sqrt(variances)
        self.nfac = nfacs[np.argmax(sharpes)]
        if print_iter:
            print('Cross Validation Number of Factors:', self.nfac)
        return self.nfac



