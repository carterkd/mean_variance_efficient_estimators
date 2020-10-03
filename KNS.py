#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:15:11 2019

@author: carter
"""

import numpy as np
import numpy.linalg as linalg
import scipy.optimize as so
import scipy.linalg as sl
import util

def solve(a, b):
    try:
        return sl.solve(a, b)
    except:
        sol, _, _, _ = sl.lstsq(a, b)
        return sol

def add_tensor_constant(Z, not_missing = None, normalize_const = True):
    T, N, K = Z.shape
    if not_missing is None:
        const = np.ones((T, N, 1))
    else:
        const = np.expand_dims(not_missing, axis = 2)
    if normalize_const == True:
        const = const / np.sum(const, axis = 1, keepdims = True)
    return np.concatenate((const, Z), axis = 2)

def loss_function(b, mu, Lambda, LambdaInv, gamma1, gamma2):
    b = b.reshape(-1, 1)
    return ((mu - Lambda @ b).T @ LambdaInv @ (mu - Lambda @ b))[0, 0] + gamma2 * np.sum(b**2) + gamma1 * np.sum(np.abs(b))

def loss_deriv(b, mu, Lambda, LambdaInv, gamma1, gamma2):
    b = b.reshape(-1, 1)
    return (2 * Lambda @ b - 2 * mu + 2 * gamma2 * b + gamma1 * np.sign(b)).flatten()

def solve_l2shrink(mu, Lambda, gamma2):
    N = Lambda.shape[0]
    return solve(Lambda + gamma2 * np.eye(N), mu)

def loss_function_bm(b, mu, Lambda, LambdaInv, gamma1, gamma2, iota_no):
    b = b.reshape(-1, 1)
    return ((mu - Lambda @ b).T @ LambdaInv @ (mu - Lambda @ b))[0, 0] + gamma2 * np.sum((b*iota_no)**2) + gamma1 * np.sum(np.abs(b*iota_no))

def loss_deriv_bm(b, mu, Lambda, LambdaInv, gamma1, gamma2, iota_no):
    b = b.reshape(-1, 1)
    return (2 * (Lambda @ b - mu) + 2 * gamma2 * b * iota_no + gamma1 * np.sign(b * iota_no)).flatten()

def solve_l2shrink_bm(mu, Lambda, gamma2):
    N = Lambda.shape[0]
    I = np.eye(N)
    I[0, 0] = 0.
    return solve(Lambda + gamma2 * I, mu)

def zr_pca(Z, r, nfac = 5):
    T, N, K = Z.shape
    F = np.matmul(np.transpose(Z, [0, 2, 1]), r).reshape((T, K))
    Sigma = np.cov(F, rowvar=False)
    lam, v = linalg.eig(Sigma)
    sorted_indices = np.argsort(lam)[::-1]
    v = v[:, sorted_indices]
    weights = v[:, :nfac]
    return weights

def Zr_mult(Z, r):
    T, N, K = Z.shape
    return np.matmul(np.transpose(Z, [0, 2, 1]), r).reshape((T, K))

def calc_moments(Z, r):
    F = Zr_mult(Z, r)
    Sigma = np.cov(F, rowvar=False)
    mu = np.mean(F, 0).reshape(-1, 1)
    return mu, Sigma

def calc_var(Z, r):
    F = Zr_mult(Z, r)
    return np.cov(F, rowvar=False)

def ndargmax(a):
    return np.unravel_index(np.argmax(a, axis=None), a.shape)

def all_except_column(A, ii):
    select = A.shape[-1] * [True]
    select[ii] = False
    return A[..., select]

### condition number of matrix
def cond(A):
    eigs = sl.eigvals(A)
    abs_eigs = np.abs(eigs)
    return np.max(abs_eigs) / np.min(abs_eigs)

### get r2
def get_r2(X, Y):
    error = Y - (X @ util.ols(X, Y))
    var_error = np.var(error.flatten())
    var_y = np.var(Y.flatten())
    if var_error == 0:
        return 1.
    return 1. - var_error / var_y 

class KNS:
    def __init__(self, Z = None, r = None, not_missing = None, 
                 gamma1 = 0., gamma2 = 0., 
                 add_cons = False, 
                 benchmark_portfolio = 0):
        self.add_cons = add_cons
        self.benchmark_portfolio = benchmark_portfolio
        self.set_gamma(gamma1, gamma2)
        self.naive_fitted = False
        self.exclude_bms = None
        if Z is not None:
            self.set_data(Z, r, not_missing)
    def set_gamma(self, gamma1, gamma2):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
    def check_benchmark(self):
        if self.benchmark_portfolio > 1:
            self.exclude_bms = np.zeros(self.benchmark_portfolio)
            for ii in range(self.benchmark_portfolio):
                X = []
                Y = []
                exind = [jj for jj in range(self.benchmark_portfolio) if jj != ii and self.exclude_bms[jj] != 1]
                exlen = len(exind)
                if exlen == 0:
                    continue
                for tt in range(self.T):
                    Z = self.Z[tt, self.not_missing[tt, :] > 0, :]
                    X.append(Z[:, exind].reshape((-1, exlen)))
                    Y.append(Z[:, ii].reshape((-1, 1)))
                X = np.concatenate(X, axis = 0)
                Y = np.concatenate(Y, axis = 0)
                r2 = get_r2(X, Y)
                if r2 > 0.999:
                    self.exclude_bms[ii] = 1
    def set_data(self, Z, r, not_missing):
        if self.add_cons:
            self.Z = add_tensor_constant(Z, not_missing) * np.expand_dims(not_missing, axis = 2)
        else:
            self.Z = Z * np.expand_dims(not_missing, axis = 2)
        self.not_missing = not_missing
        if len(r.shape) == 2:
            self.r = np.expand_dims(r * not_missing, axis = 2)
        else:
            self.r = r * np.expand_dims(not_missing, axis = 2)
        self.T, self.N, self.K = self.Z.shape
        self.pca = False
        self.nfac = self.K - self.benchmark_portfolio
        self.check_benchmark()
        zw = np.eye(self.K)
        if not self.exclude_bms is None:
                keep = [kk for kk in range(self.nfac + self.benchmark_portfolio) 
                        if kk >= self.benchmark_portfolio or self.exclude_bms[kk] != 1]
                zw = zw[:, keep]
        self.zw = zw
        self.expanded_zw = np.tile(np.expand_dims(self.zw, 0), (self.T, 1, 1))
        self.nfac = self.K
    def fit_pca(self, nfac = 20):
        self.nfac = nfac
        if self.benchmark_portfolio > 0:
            zw = zr_pca(self.Z[:, :, self.benchmark_portfolio:], self.r, self.nfac)
            zw = np.pad(zw, pad_width = [(self.benchmark_portfolio, 0), (self.benchmark_portfolio, 0)], 
                                         mode = 'constant')
            for ii in range(self.benchmark_portfolio):
                zw[ii, ii] = 1.
            if not self.exclude_bms is None:
                keep = [kk for kk in range(self.nfac + self.benchmark_portfolio) 
                        if kk >= self.benchmark_portfolio or self.exclude_bms[kk] != 1]
                zw = zw[:, keep]
            self.zw = zw
        else:
            self.zw = zr_pca(self.Z, self.r, self.nfac)
        self.expanded_zw = np.tile(np.expand_dims(self.zw, 0), (self.T, 1, 1))
        self.pca = True
    def fit_naive(self):
        Zw = np.matmul(self.Z, self.expanded_zw)
        self.mu, self.Lambda = calc_moments(Zw, self.r)
        try:
            self.LambdaInv = sl.inv(self.Lambda)
        except:
            self.LambdaInv = np.linalg.pinv(self.Lambda)
        self.naive_b = (self.LambdaInv @ self.mu).flatten()
        self.b = self.naive_b.reshape(-1, 1)
        expanded_b = np.tile(np.expand_dims(self.b, 0), (self.T, 1, 1))
        self.weights = np.matmul(Zw, expanded_b).reshape((self.T, self.N))
        self.weights = self.weights / np.sum(self.weights, axis = 1, keepdims = True)
        self.zwb = np.matmul(self.zw, self.b)
        self.naive_fitted = True
    def reset_zw(self):
        self.pca = False
        self.zw = np.eye(self.K)
        self.expanded_zw = np.tile(np.expand_dims(self.zw, 0), (self.T, 1, 1))
        self.nfac = self.K
    def correlation_stepdown(self, threshold):
        Zw = np.matmul(self.Z, self.expanded_zw)
        Lambda = calc_var(Zw, self.r)
        sigmas = np.sqrt(np.diag(Lambda))
        corr = np.diag(1 / sigmas) @ Lambda @ np.diag(1 / sigmas)
        abscorr = np.abs(corr - np.eye(len(sigmas)))
        maxcorr = np.max(abscorr)
        if maxcorr < threshold or self.zw.shape[1] <= 2:
            return
        xi, yi = ndargmax(abscorr)
        Lambda_x = calc_var(all_except_column(Zw, xi), self.r)
        Lambda_y = calc_var(all_except_column(Zw, yi), self.r)
        xcond = cond(Lambda_x)
        ycond = cond(Lambda_y)
        di = xi
        if ycond < xcond:
            di = yi
        self.zw = all_except_column(self.zw, di)
        self.nfac = self.zw.shape[1]
        self.expanded_zw = np.tile(np.expand_dims(self.zw, 0), (self.T, 1, 1))
        return self.correlation_stepdown(threshold)
    def calc_cond(self):
        Zw = np.matmul(self.Z, self.expanded_zw)
        Lambda = calc_var(Zw, self.r)
        return cond(Lambda)
    def cond_stepdown(self, threshold):
        if self.benchmark_portfolio == True:
            rank = np.linalg.matrix_rank(self.Lambda[1:, 1:])
            nfacs = np.arange(1, rank+1)[::-1]
        else:
            rank = rank = np.linalg.matrix_rank(self.Lambda)
            nfacs = np.arange(2, rank+1)[::-1]
        for nfac in nfacs:
            self.reset_zw()
            self.fit_pca(nfac)
            cond = self.calc_cond()
            if cond < threshold:
                if nfac == self.K:
                    self.reset_zw()
                break
    def fit(self, Z = None, r = None, not_missing = None,
            gamma1 = None, gamma2 = None, threshold = None):
        if gamma1 is not None and gamma2 is not None:
            self.set_gamma(gamma1, gamma2)
        elif gamma1 is not None:
            self.set_gamma(gamma1, self.gamma2)
        elif gamma2 is not None:
            self.set_gamma(self.gamma1, gamma2)
        if Z is not None:
            self.set_data(Z, r, not_missing)
            self.fit_naive()
        if self.naive_fitted == False:
            self.fit_naive()
        if threshold is not None:
            self.reset_zw()
            #self.correlation_stepdown(threshold)
            self.cond_stepdown(threshold)
            self.fit_naive()
        Zw = np.matmul(self.Z, self.expanded_zw)
        if self.benchmark_portfolio > 0:
            iota_no = np.ones((len(self.naive_b), 1))
            if self.exclude_bms is None:
                iota_no[:self.benchmark_portfolio, 0] = 0.
            else:
                iota_no[:np.sum(self.exclude_bms == 0), 0] = 0.
            if self.gamma1 == 0.:
                self.b = solve_l2shrink_bm(self.mu, self.Lambda, self.gamma2).reshape(-1, 1)
            else:
                self.op_res = so.minimize(loss_function_bm, self.naive_b, 
                                          args = (self.mu, self.Lambda, 
                                                  self.LambdaInv, self.gamma1, 
                                                  self.gamma2, iota_no),
                                                  jac = loss_deriv_bm)
                self.b = self.op_res.x.reshape(-1, 1)
        else:
            if self.gamma1 == 0.:
                self.b = solve_l2shrink(self.mu, self.Lambda, self.gamma2).reshape(-1, 1)
            else:
                self.op_res = so.minimize(loss_function, self.naive_b, 
                                          args = (self.mu, self.Lambda, 
                                                  self.LambdaInv, self.gamma1, 
                                                  self.gamma2), jac = loss_deriv)
                self.b = self.op_res.x.reshape(-1, 1)
        expanded_b = np.tile(np.expand_dims(self.b, 0), (self.T, 1, 1))
        self.weights = np.matmul(Zw, expanded_b).reshape((self.T, self.N))
        #self.weights = self.weights / np.sum(self.weights, axis = 1, keepdims = True)
        self.zwb = np.matmul(self.zw, self.b)
        return self.zwb.flatten()
    def r2(self, Z, r, not_missing):
        T, N, K = Z.shape
        expanded_zw = np.tile(np.expand_dims(self.zw, 0), (T, 1, 1))
        if self.add_cons:
            Z = add_tensor_constant(Z, not_missing) * np.expand_dims(not_missing, axis = 2)
        else:
            Z = Z * np.expand_dims(not_missing, axis = 2)
        if self.pca:
            Z = np.matmul(Z, expanded_zw)
        if len(r.shape) == 2:
            r = np.expand_dims(r * not_missing, axis = 2)
        mu, Lambda = calc_moments(Z, r)
        r2 = 1. - np.sum((mu - Lambda @ self.b)**2) / np.sum(mu**2)
        return r2
    def set_model_variables(self, model_variables):
        zw, b = model_variables
        self.zw = zw
        self.b = b
        if len(self.zw) == len(self.b):
            self.pca = False
        else:
            self.pca = True
        self.zwb = self.zw @ self.b
    def get_model_variables(self):
        return self.zw, self.b
    def predict(self, Z, r, not_missing):
        T, N, K = Z.shape
        if self.add_cons:
            Z = add_tensor_constant(Z, not_missing) * np.expand_dims(not_missing, axis = 2)
        else:
            Z = Z * np.expand_dims(not_missing, axis = 2)
        if len(r.shape) == 2:
            r = r * not_missing
        expanded_zwb = np.tile(np.expand_dims(self.zwb, 0), (T, 1, 1))
        weights = np.matmul(Z, expanded_zwb).reshape((T, N))
        #weights = weights / np.sum(weights, axis = 1, keepdims = True)
        f = np.sum(weights * r, 1)
        return {'w': weights, 'tangency': f}
    def cross_validate(self, folds = 4, ncount = 9, 
                       gamma_min = 0., gamma_max = 2.,
                       print_iter = False):
        r2s = np.zeros((folds, ncount, ncount))
        gammas = np.linspace(gamma_min, gamma_max, ncount)
        samples = [[tt for tt in range(self.T) 
                    if int(folds * tt / self.T) == fold]
                    for fold in range(folds)]
        for fold in range(folds):
            fold_sample = np.concatenate([samp for kk, samp in enumerate(samples) if kk != fold])
            xv_sample = samples[fold]
            fold_model = KNS(self.Z[fold_sample, :, :], self.r[fold_sample, :],
                             self.not_missing[fold_sample, :],
                             add_cons = False, 
                             benchmark_portfolio = self.benchmark_portfolio)
            if self.nfac < self.K:
                fold_model.fit_pca(self.nfac)
            fold_model.fit_naive()
            for ii, gamma1 in enumerate(gammas):
                for jj, gamma2 in enumerate(gammas):
                    fold_model.gamma1 = gamma1
                    fold_model.gamma2 = gamma2
                    fold_model.fit()
                    r2s[fold, ii, jj] = fold_model.r2(self.Z[xv_sample, :, :],
                       self.r[xv_sample, :], self.not_missing[xv_sample, :])
                    if print_iter:
                        print(('Cross Validation:', fold, ii, jj, r2s[fold, ii, jj]))
            del fold_model
        r2s_avg = np.mean(r2s, axis = 0)
        g1ind, g2ind = np.unravel_index(np.argmin(r2s_avg), r2s_avg.shape)
        self.gamma1, self.gamma2 = gammas[g1ind], gammas[g2ind]
        return self.gamma1, self.gamma2
            




