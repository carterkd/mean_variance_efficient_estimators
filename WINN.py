#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:48:05 2019

@author: carter
"""

import numpy as np
import tensorflow as tf
import util
import scipy.linalg as sl

### extract paramters from var_list
def extract_params(var_list, multiply_x = 100):
    weights, biases, phi = var_list
    beta_mu0 = biases[0]
    beta_Gamma0 = biases[1]
    beta_mu1 = weights[:, 0]
    beta_Gamma1 = weights[:, 1]
    phi = np.exp(phi)
    beta_mu0 = beta_mu0 / multiply_x
    beta_mu1 = beta_mu1 / multiply_x
    beta_Gamma0 = beta_Gamma0 / multiply_x
    beta_Gamma1 = beta_Gamma1 / multiply_x
    phi = phi / multiply_x**2
    return beta_mu0, beta_mu1, beta_Gamma0, beta_Gamma1, phi

### pack parameters back into var_list
def pack_params(beta_mu0, beta_mu1, beta_Gamma0, beta_Gamma1, phi, multiply_x = 100):
    beta_mu0 = beta_mu0 * multiply_x
    beta_mu1 = beta_mu1 * multiply_x
    beta_Gamma0 = beta_Gamma0 * multiply_x
    beta_Gamma1 = beta_Gamma1 * multiply_x
    phi = phi * multiply_x**2
    phi = np.log(phi)
    weights = np.stack((beta_mu1, beta_Gamma1), axis = 1)
    biases = np.array([beta_mu0, beta_Gamma0])
    var_list = [weights, biases, phi]
    return var_list

def simple_woodbury_inversion(Ainv, U, c):
    U = U.reshape((-1, 1))
    AinvU = Ainv @ U
    return Ainv - AinvU @ AinvU.T / (1. / c + np.sum(U.T @ AinvU))

def calc_cov(phi, Gamma):
    Gamma = Gamma.reshape(-1, 1)
    return (phi * np.eye(Gamma.size) + Gamma @ Gamma.T)

def calc_cov_inverse(phi, Gamma):
    Gamma = Gamma.reshape(-1, 1)
    return simple_woodbury_inversion(np.eye(Gamma.size) / phi, Gamma, 1.)

def ssimu(Gamma1, Gamma2, phi1, phi2, mu):
    Gamma1 = Gamma1.flatten()
    Gamma2 = Gamma2.flatten()
    Gamma2_mu = np.sum(Gamma2 * mu)
    Gamma1_mu = np.sum(Gamma1 * mu)
    Gamma1_Gamma2 = np.sum(Gamma1 * Gamma2)
    k = 1. / (phi2 * (phi2 + np.sum(Gamma2**2)))
    return (phi1 / phi2) * mu - phi1 * k * Gamma2_mu * Gamma2 + (1 / phi2) * Gamma1_mu * Gamma1 - k * Gamma1_Gamma2 * Gamma2_mu * Gamma1

def calc_linear_weights(mu, Gamma, phi):
    Gamma = Gamma.flatten()
    k = 1. / (phi * (phi + np.sum(Gamma**2)))
    return mu / phi - k * np.sum(Gamma * mu) * Gamma

### main class
class WINN:
    """WINN = Woodbury Identity Neural Network
    Parameters
    ----------
    Tmax : {int}, 
        Maximum number of periods.
    Nmax : {int},
        Maximum number of assets.
    K : {int},
        Number of characteristics or features.
    J : {int},
        Second dimension of Gamma_t.
    num_nodes : {list of ints},
        Number of neurons in each hidden layer.
    initial_range : {float},
        Number used to initialize parameters.
    learning_rate_decrease : {float},
        Rate of learning rate decrease.
    initial_learning_rate : {float},
        Initial learning rate.
    no_improvement_limit : {int},
        Maximum number of iterations without a loss function decrease before termination.
    batch_T : {int},
        Batch size for Adam optimization.
    model_numb : {int},
        Differentiates between multiple instances of this class in memory at the same time.
    look_back : {int},
        Number of iterations to look back and comapre against during fitting.
    max_epochs : {int},
        Maximum number of epochs before termination during learning.
    conv_tol : {float},
        Convergence tolerance.
    multiply_x : {float},
        Term to multiply returns by to help numerical calcualtions.
    no_improve_decrease_lr : {float},
        Number of epochs of without decreases in the loss function where the learning rate is increased.
    cpu_limit : int or None,
        if None, no CPU limit is used, else cpu_limit is used as the limit.
    """
    def __init__(self, Tmax, Nmax, K, J = 1, 
                 num_nodes = [], 
                 initial_range = 0.1, learning_rate_decrease = 0.5,
                 initial_learning_rate = 0.1,
                 no_improvement_limit = 50,
                 batch_T = 6, model_numb = 1, 
                 look_back = 10, max_epochs = 5000, conv_tol = 1e-08,
                 multiply_x = 100.,
                 no_improve_decrease_lr = 5, cpu_limit = None):
        self.Tmax = Tmax
        self.Nmax = Nmax
        self.K = K
        self.J = J
        self.num_nodes = num_nodes
        self.initial_range = initial_range
        self.active_session = False
        self.epoch = 0
        self.initial_learning_rate = initial_learning_rate
        self.current_learning_rate = initial_learning_rate
        self.learning_rate_decrease = learning_rate_decrease
        self.no_improvement_limit = no_improvement_limit
        self.batch_T = batch_T
        self.look_back = look_back
        self.max_epochs = max_epochs
        self.conv_tol = conv_tol
        self.multiply_x = multiply_x
        self.model_numb = model_numb
        self.no_improve_decrease_lr = no_improve_decrease_lr
        self.cpu_limit = cpu_limit
        
        ### declare main variables
        tf.compat.v1.disable_eager_execution()
        self.sample_ratio = tf.compat.v1.placeholder(tf.float64, shape = (), name = 'sample_ratio%d' % self.model_numb)
        self.predictors_big = tf.compat.v1.get_variable('predictors_big%d'%self.model_numb, 
                                              shape = [self.Tmax, self.Nmax, self.K],
                                              initializer=tf.initializers.constant(0.),
                                              dtype = tf.float64)
        self.not_missing_big = tf.compat.v1.get_variable('not_missing_big%d'%self.model_numb, 
                                               shape = [self.Tmax, self.Nmax],
                                               initializer=tf.initializers.constant(0.),
                                               dtype = tf.float64)
        self.exrets_big = tf.compat.v1.get_variable('exrets_big%d'%self.model_numb, 
                                          shape = [self.Tmax, self.Nmax],
                                          initializer=tf.initializers.constant(0.),
                                          dtype = tf.float64)
        
        ### declare placeholders
        self.T = tf.compat.v1.placeholder(tf.int32, shape = (), name = 'T%d' % self.model_numb)
        self.prior_precision_ph = tf.compat.v1.placeholder(tf.float64, shape = (), name = 'prior_precision_ph%d' % self.model_numb)
        self.mean_prior_precision_ph = tf.compat.v1.placeholder(tf.float64, shape = (), name = 'mean_prior_precision_ph%d' % self.model_numb)
        self.predictors_ph = tf.compat.v1.placeholder(tf.float64, shape = [self.Tmax, self.Nmax, self.K],
                                    name = 'predictors_ph%d' % self.model_numb)
        self.not_missing_ph = tf.compat.v1.placeholder(tf.float64, shape = [self.Tmax, self.Nmax],
                                   name = 'not_missing_ph%d' % self.model_numb)
        self.exrets_ph = tf.compat.v1.placeholder(tf.float64, shape = [self.Tmax, self.Nmax],
                                 name = 'exrets_ph%d' % self.model_numb)
        self.use_indices = tf.compat.v1.placeholder(tf.int32, shape = [None],
                                 name = 'use_indices%d' % self.model_numb)
        
        ### get slices
        self.predictors = tf.gather(self.predictors_big, self.use_indices, axis = 0)
        self.not_missing = tf.gather(self.not_missing_big, self.use_indices, axis = 0)
        self.exrets = tf.gather(self.exrets_big, self.use_indices, axis = 0)
        
        ####################### conditional expectation predictions ##################
        if self.num_nodes == []:
            self.layers_dims = [(self.K, 1 + self.J)]
        else:
            self.layers_dims = ([(self.K, self.num_nodes[0])]
                            + [(self.num_nodes[ii-1], self.num_nodes[ii])
                                for ii in range(1, len(self.num_nodes))]
                            + [(self.num_nodes[-1], 1 + self.J)])
        
        ### define weights and biases
        self.weights_ph = [tf.compat.v1.placeholder(name = 'weights_ph_%d_%d'%(ii+1, self.model_numb), shape = ss,
                                      dtype = tf.float64)
                            for ii, ss in enumerate(self.layers_dims)]
        self.biases_ph = [tf.compat.v1.placeholder(name = 'biases_ph_%d_%d'%(ii+1, self.model_numb), shape = ss[1],
                                      dtype = tf.float64)
                            for ii, ss in enumerate(self.layers_dims)]
        self.phi_ph = tf.compat.v1.placeholder(name = 'phi_ph%d' % self.model_numb, shape = (), 
                                  dtype = tf.float64)
        
        self.weights = [tf.compat.v1.get_variable('weights_%d_%d'%(ii+1, self.model_numb), shape = ss,
                                      initializer=tf.compat.v1.initializers.random_uniform(minval = 0., maxval = self.initial_range),
                                      dtype = tf.float64)
                            for ii, ss in enumerate(self.layers_dims)]
        self.biases = [tf.compat.v1.get_variable('biases_%d_%d'%(ii+1, self.model_numb), shape = ss[1],
                                      initializer=tf.compat.v1.initializers.random_uniform(minval = 0., maxval = self.initial_range),
                                      dtype = tf.float64)
                            for ii, ss in enumerate(self.layers_dims)]
        self.phi = tf.compat.v1.get_variable('phi%d' % self.model_numb, shape = (), 
                                  initializer=tf.compat.v1.initializers.random_uniform(minval = 0., maxval = 0. + self.initial_range),
                                  dtype = tf.float64)
        self.prior_precision = tf.compat.v1.get_variable('prior_precision%d' % self.model_numb, shape = (),
                                               initializer=tf.initializers.constant(0.),
                                               dtype = tf.float64, trainable = False)
        self.mean_prior_precision = tf.compat.v1.get_variable('mean_prior_precision%d' % self.model_numb, shape = (),
                                               initializer=tf.initializers.constant(0.),
                                               dtype = tf.float64, trainable = False)
        
        self.var_list = self.weights + self.biases + [self.phi]
        self.var_ph_list = self.weights_ph + self.biases_ph + [self.phi_ph]
        
        ### feed data through layers
        indata = tf.reshape(self.predictors, [-1, self.K])
        self.out_layers = [indata]
        self.prior = 0.
        for ii, ss in enumerate(self.layers_dims):
            if ii+1 == len(self.layers_dims):
                self.prior = self.prior + 0.5 * self.prior_precision * (tf.reduce_sum(self.weights[ii][:, 1:]**2)) # + tf.reduce_sum(self.biases[ii][1:]**2)
                self.prior = self.prior + 0.5 * self.mean_prior_precision * (tf.reduce_sum(self.weights[ii][:, 0]**2)) # + tf.reduce_sum(self.biases[ii][0]**2)
            else:
                self.prior = self.prior + 0.5 * self.prior_precision * (tf.reduce_sum(self.weights[ii]**2)) #  + tf.reduce_sum(self.biases[ii]**2)
            if ii > 0:
                self.out_layers.append(tf.compat.v1.nn.xw_plus_b(tf.nn.relu(self.out_layers[ii]), 
                                self.weights[ii], self.biases[ii]))
            else:
                self.out_layers.append(tf.compat.v1.nn.xw_plus_b(self.out_layers[ii], 
                                self.weights[ii], self.biases[ii]))
        
        ### Split Output into Relevent Parts
        nn_output = tf.unstack(tf.reshape(self.out_layers[-1], [self.T, self.Nmax, self.J+1]), axis = 2)
        self.mean_predictions = nn_output[0]
        self.var_predictions = tf.stack(nn_output[1:])
        
        ### calculate phi stuff
        phi = tf.exp(self.phi)
        
        ### calculate key stuff
        self.Gamma = tf.expand_dims(self.not_missing, axis = 2) * tf.transpose(self.var_predictions, [1, 2, 0])
        self.IJ = tf.expand_dims(tf.eye(self.J, dtype = tf.float64), axis = 0)
        self.GammaGamma = tf.matmul(tf.transpose(self.Gamma, [0, 2, 1]), self.Gamma)
        self.middle = phi * self.IJ + self.GammaGamma
        self.average_mean = tf.reduce_sum(self.mean_predictions * self.not_missing) / tf.reduce_sum(self.not_missing)
        self.average_exrets = tf.reduce_sum(self.exrets * self.not_missing) / tf.reduce_sum(self.not_missing)
        self.ssr = tf.reduce_sum(self.not_missing * self.exrets**2)
        self.sse = tf.reduce_sum(self.not_missing * (self.exrets - self.mean_predictions)**2)
        self.varis = tf.reduce_sum(self.Gamma**2, 2) + phi
        self.xs_vari_mean = tf.reduce_sum(self.not_missing * self.varis, 1) / tf.reduce_sum(self.not_missing, 1)
        self.vari_mean = tf.reduce_mean(self.xs_vari_mean)
        
        ### calculate log likelihood variables
        self.error = self.not_missing * (self.exrets - self.mean_predictions)
        self.errGamma = tf.matmul(tf.expand_dims(self.error, axis = 1), self.Gamma)
        self.err_part = tf.reduce_sum(self.error**2, 1) / phi
        self.invmGammaErr = tf.linalg.solve(self.middle, tf.transpose(self.errGamma, [0, 2, 1]))
        self.v_part = tf.reduce_sum(tf.matmul(self.errGamma, self.invmGammaErr), [1, 2]) / phi
        self.Ns = tf.reduce_sum(self.not_missing, 1)
        self.log_determinant = self.Ns * tf.math.log(phi) + tf.linalg.logdet(self.IJ + self.GammaGamma / phi)
        
        ### calculate loss
        self.nll = 0.5 * (self.log_determinant + self.err_part - self.v_part)
        self.loss = tf.reduce_sum(self.nll) + self.sample_ratio * self.prior
        
        ### optimal portfolio weights -- unscaled
        self.GammaMu = tf.matmul(tf.transpose(self.Gamma, [0, 2, 1]), tf.expand_dims(self.not_missing * self.mean_predictions, axis = 2))
        self.invmGammaMu = tf.linalg.solve(self.middle, self.GammaMu)
        self.portfolio_weights = self.not_missing * (self.mean_predictions - tf.reduce_sum(tf.matmul(self.Gamma, self.invmGammaMu), axis = 2)) / phi
        #self.portfolio_weights = self.portfolio_weights / tf.reduce_sum(self.portfolio_weights, axis = 1, keepdims = True)
        self.tangency_exrets = tf.reduce_sum(self.portfolio_weights * self.exrets, axis = 1) # keepdims = True
        
        ### assign things
        self.assign_parameters = tf.group([tf.compat.v1.assign(self.var_list[ii], self.var_ph_list[ii]) for ii in range(len(self.var_list))])
        self.assign_prior_precision = tf.group([tf.compat.v1.assign(self.prior_precision, self.prior_precision_ph),
                                                tf.compat.v1.assign(self.mean_prior_precision, self.mean_prior_precision_ph)])
        self.assign_data = tf.group([tf.compat.v1.assign(self.predictors_big, self.predictors_ph),
                                     tf.compat.v1.assign(self.not_missing_big, self.not_missing_ph),
                                     tf.compat.v1.assign(self.exrets_big, self.exrets_ph)])
        
        ################## learning rate ###################################
        self.learning_rate = tf.compat.v1.placeholder_with_default(1e-06, shape = ())
        
        ###################### Optimizers ##################################
        # Optimizer.
        self.optim = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.gradients, v = zip(*self.optim.compute_gradients(self.loss, var_list = self.var_list))
        #self.gradients, _ = tf.clip_by_global_norm(self.gradients, 5.0)
        self.optimizer = self.optim.apply_gradients(zip(self.gradients, v))
        
        ### Random Sample Generator
        self.sample_size = tf.compat.v1.placeholder(tf.int32, shape = (), name = 'sample_size')
        self.GammaSimX = tf.random.normal([self.J, self.sample_size], dtype = tf.float64)
        self.phiX = tf.random.normal([self.Nmax, self.sample_size], dtype = tf.float64)
        self.random_sample = tf.transpose(self.mean_predictions) + tf.matmul(tf.reshape(self.Gamma, [self.Nmax, self.J]), self.GammaSimX) + tf.math.sqrt(phi) * self.phiX
        
        ### initialize variables
        self.data_variables = [self.predictors_big, self.not_missing_big, self.exrets_big]
        self.initialize_variables = tf.compat.v1.initializers.variables(self.optim.variables() + self.var_list + [self.prior_precision, self.mean_prior_precision])
        self.restart_optimizer = tf.compat.v1.initializers.variables(self.optim.variables())
        self.initialize_data = tf.compat.v1.initializers.variables(self.data_variables)
        
    def initialize_session(self):
        if self.cpu_limit is not None:
            session_conf = tf.compat.v1.ConfigProto(device_count={'CPU': self.cpu_limit})
            self.session = tf.compat.v1.Session(config = session_conf)
        else:
            self.session = tf.compat.v1.Session()
        self.session.run(self.initialize_variables)
        self.active_session = True
        self.epoch = 0
        self.last_update_index = 0
        self.lls = []
    
    def close_session(self):
        self.session.close()
        self.active_session = False
    
    def restart(self):
        if self.active_session == False:
            self.initialize_session()
        self.epoch = 0
        self.last_update_index = 0
        self.lls = []
        self.current_learning_rate = self.initial_learning_rate
        precision = self.get_prior_precision()
        params = self.get_model_variables()
        self.session.run(self.initialize_variables)
        self.session.run(self.initialize_data)
        self.set_prior_precision(precision[0], precision[1])
        self.set_model_variables(params)
    
    def get_model_variables(self):
        if self.active_session == False:
            self.initialize_session()
        return self.session.run(self.var_list)
    
    def set_model_variables(self, layer_variables):
        if self.active_session == False:
            self.initialize_session()
        feed_dict = {}
        for ii in range(len(layer_variables)):
            feed_dict[self.var_ph_list[ii]] = layer_variables[ii]
        self.session.run(self.assign_parameters,
                         feed_dict = feed_dict)
    
    def set_prior_precision(self, prior_precision, mean_prior_precision):
        if self.active_session == False:
            self.initialize_session()
        self.session.run(self.assign_prior_precision, 
                         feed_dict = {self.prior_precision_ph: prior_precision,
                                      self.mean_prior_precision_ph: mean_prior_precision})
    
    def get_prior_precision(self):
        if self.active_session == False:
            self.initialize_session()
        return tuple(self.session.run([self.prior_precision, self.mean_prior_precision]))
    
    def set_data(self, Z, r, not_missing):
        if self.active_session == False:
            self.initialize_session()
        T, N, K = Z.shape
        add_T = max(self.Tmax - T, 0)
        add_N = max(self.Nmax - N, 0)
        Z_big = np.pad(Z, [(0, add_T), (0, add_N), (0, 0)],
                           mode = 'constant')
        r_big = np.pad(r, [(0, add_T), (0, add_N)],
                           mode = 'constant')
        not_missing_big = np.pad(not_missing, [(0, add_T), (0, add_N)],
                                               mode = 'constant')
        if self.multiply_x is not None:
            r_big = self.multiply_x * r_big
        feed_dict = {self.predictors_ph: Z_big,
                    self.not_missing_ph: not_missing_big,
                    self.exrets_ph: r_big}
        self.session.run(self.assign_data, feed_dict = feed_dict)
        return T
    
    def print_ll(self, feed_dict):
        #self.log_determinant + self.err_part - self.v_part
        print('printing ll')
        logdet, ep, vp, exrets, mu, Gamma, tphi, notmiss, errG, invGerr = self.session.run([self.log_determinant, self.err_part, 
                                                                    self.v_part, self.exrets, self.mean_predictions,
                                                                    self.Gamma, self.phi, self.not_missing,
                                                                    self.errGamma, self.invmGammaErr], feed_dict = feed_dict)
        phi = np.exp(tphi)
        mu = list(mu)
        exrets = list(exrets)
        Gamma = list(Gamma)
        notmiss = list(notmiss > 0.5)
        T = len(mu)
        Sigma = T * [None]
        quadratic = T * [None]
        logdets = T * [None]
        for tt in range(T):
            print('beginning of step', tt)
            N = np.sum(notmiss[tt])
            Gamma[tt] = Gamma[tt][notmiss[tt], :]
            mu[tt] = mu[tt][notmiss[tt]]
            exrets[tt] = exrets[tt][notmiss[tt]]
            Sigma[tt] = Gamma[tt] @ Gamma[tt].T + phi * np.eye(N)
            diff = exrets[tt] - mu[tt]
            quadratic[tt] = np.sum(diff * sl.solve(Sigma[tt], diff))
            _, logdets[tt] = np.linalg.slogdet(Sigma[tt])
            print('end of step', tt)
        print(errG.shape)
        print(invGerr.shape)
        print('tensorflow quadratic:')
        print(ep - vp)
        print('numpy quadratic:')
        print(quadratic)
        print('tensorflow log determinants')
        print(logdet)
        print('numpy log determinants:')
        print(logdets)
    
    def fit_epoch(self, T, batch_T, 
                  learning_rate = None, 
                  print_iter = False,
                  iter_indices = None):
        if self.active_session == False:
            self.initialize_session()
        if iter_indices is None:
            obs = T
            iter_indices = np.arange(obs)
        else:
            obs = len(iter_indices)
        np.random.shuffle(iter_indices)
        iter_batches = util.chunkify(iter_indices, max(len(iter_indices) // batch_T, 1))
        itern = len(iter_batches)
        average_ll = 0
        tan_exrets = []
        average_mean = 0
        average_vari = 0
        average_exrets = 0
        sse = 0.
        ssr = 0.
        for step, indices in enumerate(iter_batches):
            
            Tval = len(indices)
            sample_ratio = Tval / obs
            feed_dict = {self.T: Tval,  
                         self.sample_ratio: sample_ratio,
                         self.use_indices: np.array(indices).astype(np.int32)}
            
            if learning_rate is not None:
                feed_dict.update({self.learning_rate: learning_rate})
            else:
                feed_dict.update({self.learning_rate: self.current_learning_rate})
            
            session_output = self.session.run([self.optimizer, self.loss, 
                                               self.tangency_exrets, 
                                               self.average_mean,
                                               self.average_exrets,
                                               self.sse, self.ssr,
                                               self.vari_mean], 
                                                feed_dict = feed_dict)
            _, nll, rr, amn, aex, ssep, ssrp, vm = session_output
            
            ### print the log-likelihood stuff
            #self.print_ll(feed_dict)
            
            average_ll -= nll / obs
            tan_exrets += list(rr)
            average_mean += amn / itern
            average_exrets += aex / itern
            average_vari += vm / itern
            ssr += ssrp
            sse += ssep
        
        ### save data to output
        mean_rr = np.mean(tan_exrets)
        var_rr = np.var(tan_exrets)
        average_rr = mean_rr / np.sqrt(var_rr)
        desc_str = 'Cross Validation'
        self.epoch += 1
        desc_str = 'Training'
        if print_iter:
            print('Epoch %d' % (self.epoch))
            print('\t%s LL at step %d: %f' % (desc_str, self.epoch, average_ll))
            print('\tR2: %f' % (1. - sse / ssr))
            print('\t%s Sharpe at step %d: %f' % (desc_str, self.epoch, average_rr))
            print('\t%s Average mu at step %d: %f' % (desc_str, self.epoch, average_mean))
            print('\t%s Average variance at step %d: %f' % (desc_str, self.epoch, average_vari))
            print('\t%s Average excess returns at step %d: %f' % (desc_str, self.epoch, average_exrets))
            print('\t\t\t%s Learning Rate at step %d: %f' % (desc_str, self.epoch, self.current_learning_rate))
        
        outdata = {'loglikelihood': average_ll, 
                   'sharpe_ratio': average_rr,
                   'learning_rate': self.current_learning_rate,
                   'mean': mean_rr,
                   'variance': var_rr}
        
        return outdata
    
    def gen_random_sample(self, Z, sample_size = 1):
        if self.active_session == False:
            self.initialize_session()
        feed_dict = {self.T: 1, self.use_indices: [0]}
        N = Z.shape[0]
        feed_dict[self.predictors] = np.expand_dims(Z, 0).astype(np.float64)
        feed_dict[self.not_missing] = np.ones((1, N), dtype = np.float64)
        return self.session.run(self.random_sample, feed_dict = feed_dict)
    
    def predict(self, Z, r, not_missing, 
                print_iter = False):
        T = self.set_data(Z, r, not_missing)
        use_indices = np.arange(T).astype(np.int32)
        feed_dict = {self.T: T, self.use_indices: use_indices}
        mu, Gamma, tphi, w, tangency = self.session.run([self.mean_predictions, self.Gamma,
                                                         self.phi, 
                                                         self.portfolio_weights,
                                                         self.tangency_exrets], 
                                                        feed_dict = feed_dict)
        phi = np.exp(tphi)
        if self.multiply_x is not None:
            mu = mu / self.multiply_x
            Gamma = Gamma / self.multiply_x
            phi = phi / self.multiply_x**2
            w = w * self.multiply_x
        outdata = {'mu': mu, 'Gamma': Gamma, 'phi': phi,
                   'w': w, 'tangency': tangency}
        return outdata
    
    def _fit(self, Z, r, not_missing, 
            batch_T = None,
            learning_rate = None, 
            print_iter = False,
            max_epochs = None, 
            no_improvement_limit = None,
            iter_indices = None,
            no_improve_decrease_lr = None,
            learning_rate_decrease = None):
        self.epoch = 0
        T = self.set_data(Z, r, not_missing)
        if iter_indices is not None:
            T = len(iter_indices)
        no_improve_ll = 0
        learning_rate_decreases = 0
        best_state = None
        best_ll = -np.inf
        lls = []
        original_state = self.get_model_variables()
        if batch_T is None:
            batch_T = self.batch_T
        else:
            batch_T = min(T, batch_T)
        if learning_rate is None:
            learning_rate = self.initial_learning_rate
        if max_epochs is None:
            max_epochs = self.max_epochs
        if no_improvement_limit is None:
            no_improvement_limit = self.no_improvement_limit
        if no_improve_decrease_lr is None:
            no_improve_decrease_lr = self.no_improve_decrease_lr
        if learning_rate_decrease is None:
            learning_rate_decrease = self.learning_rate_decrease
        for ii in range(max_epochs):
            try:
                current_state = self.get_model_variables()
                res = self.fit_epoch(T, batch_T, 
                                     learning_rate = learning_rate, 
                                     print_iter = print_iter, 
                                     iter_indices = iter_indices)
            except:
                self.close_session()
                self.set_model_variables(original_state)
                learning_rate_decreases += 1
                learning_rate = learning_rate_decrease * learning_rate
                no_improve_ll = 0
                continue
            lls = lls[-(self.look_back-1):] + [res['loglikelihood']]
            if res['loglikelihood'] > best_ll:
                best_ll = res['loglikelihood']
                no_improve_ll = 0
                best_state = current_state
            else:
                no_improve_ll += 1
            if no_improve_ll >= no_improve_decrease_lr:
                learning_rate = learning_rate_decrease * learning_rate
                no_improve_ll = 0
                learning_rate_decreases += 1
            if learning_rate == 0.:
                break
            if learning_rate_decreases >= self.no_improvement_limit:
                break
            if len(lls) >= self.look_back and np.mean(np.abs(np.diff(lls))) < self.conv_tol:
                break
            
        self.set_model_variables(best_state)
        res = self.fit_epoch(T, T, learning_rate = 0., 
                             print_iter = print_iter,
                             iter_indices = iter_indices)
        res['model_parameters'] = self.get_model_variables()
        res['iterations'] = ii+1
        return res
    
    def fit(self, Z, r, not_missing, 
            learning_rate = None, 
            print_iter = False,
            max_epochs = None, 
            no_improvement_limit = None,
            iter_indices = None,
            no_improve_decrease_lr = None,
            learning_rate_decrease = None,
            reps = 1, restart = True):
        """ train or fit the model
        Parameters
        ----------
        Z : {array-like} of shape (t_periods, n_assets, k_characteristics),
            Characteristics or features.
            Note that it must be the case that t_periods <= Tmax, n_assets <= Nmax, and k_characteristics == K.
        r : {array-like} of shape (t_periods, n_assets),
            Matrix of excess returns.
        not_missing : {array-like} of shape (t_period, n_assets),
            Matrix of zeros and ones, where a one indicates that the asset is not missing, and the zero indicates missing
        print_iter : {boolean},
            Indicates if the output should be printed out during learning.
        reps : {int},
            Number of times learning should be done with different starting values.
        restart : {boolean},
            Indicates if the parameters should be re-initialized after each repeat (reps number of times) training.
        """
        if not restart:
            return self._fit(Z, r, not_missing, 
                             learning_rate = learning_rate, 
                             print_iter = print_iter,
                             max_epochs = max_epochs, 
                             no_improvement_limit = no_improvement_limit,
                             iter_indices = iter_indices,
                             no_improve_decrease_lr = no_improve_decrease_lr,
                             learning_rate_decrease = learning_rate_decrease)
        all_fits = reps * [0]
        for ii in range(reps):
            self.restart()
            fits = self._fit(Z, r, not_missing, 
                            learning_rate = learning_rate, 
                            print_iter = print_iter,
                            max_epochs = max_epochs, 
                            no_improvement_limit = no_improvement_limit,
                            iter_indices = iter_indices,
                            no_improve_decrease_lr = no_improve_decrease_lr,
                            learning_rate_decrease = learning_rate_decrease)
            all_fits[ii] = fits
        all_fits = sorted(all_fits, key = lambda x: x['loglikelihood'])
        self.set_model_variables(all_fits[-1]['model_parameters'])
        return all_fits[-1]
    
    def get_portfolio_weights(self, Z):
        T, N, D = Z.shape
        return self.session.run(self.portfolio_weights,
                                feed_dict = {self.predictors: Z,
                                             self.not_missing: np.ones((T, N)),
                                             self.T: 1, self.use_indices: [0]}).flatten()
    
    def cross_validate(self, Z, r, not_missing, 
                       folds = 4, ncount = 11, 
                       penalty_min = 0., penalty_max = 1000.,
                       xv_chunk = 1, xv_chunks = 1,
                       print_iter = False,
                       max_epochs = None,
                       reps = 1):
        T = Z.shape[0]
        uni_pen = np.linspace(penalty_min, penalty_max, ncount)
        pens = [(x, y) for x in uni_pen for y in uni_pen]
        pens = util.chunkify(pens, xv_chunks)[xv_chunk-1]
        xvN = len(pens)
        means = np.zeros((reps, folds, xvN))
        variances = np.zeros((reps, folds, xvN))
        iter_sharpe = np.zeros(means.size)
        iter_params = means.size * [0]
        samples = [[tt for tt in range(T) 
                    if int(folds * tt / T) == fold]
                    for fold in range(folds)]
        iterii = 0
        for jj in range(reps):
            for fold in range(folds):
                if folds > 1:
                    fold_sample = np.concatenate([samp for kk, samp in enumerate(samples) if kk != fold])
                    xv_sample = samples[fold]
                else:
                    fold_sample = list(range(T))
                    xv_sample = fold_sample
                for ii, penvals in enumerate(pens):
                    prior_prec, mean_prec = penvals
                    self.restart()
                    self.set_prior_precision(prior_prec, mean_prec)
                    self._fit(Z, r, not_missing, iter_indices = fold_sample, print_iter = print_iter)
                    xv_fit = self._fit(Z, r, not_missing, learning_rate = 0., iter_indices = xv_sample,
                                      print_iter = print_iter, max_epochs = max_epochs)
                    means[jj, fold, ii] = xv_fit['mean']
                    variances[jj, fold, ii] = xv_fit['variance']
                    iter_sharpe[iterii] = xv_fit['sharpe_ratio']
                    iter_params[iterii] = xv_fit['model_parameters']
                    if print_iter:
                        print(('Cross Validation:', fold, ii, means[jj, fold, ii] / np.sqrt(variances[jj, fold, ii])))
                    iterii += 1
        means = np.mean(np.mean(means, axis = 0), axis = 0)
        variances = np.mean(np.mean(variances, axis = 0), axis = 0)
        sharpes = means / np.sqrt(variances)
        best_prec, best_mean_prec = pens[np.argmax(sharpes)]
        xv_results = np.zeros((xvN, len(pens[0]) + 1))
        xv_results[:, :-1] = np.array(pens)
        xv_results[:, -1] = sharpes
        self.set_prior_precision(best_prec, best_mean_prec)
        self.set_model_variables(iter_params[np.argmax(iter_sharpe)])
        return xv_results
    
    def get_matrices(self, Z, not_missing = None, weights = None, rf = None, 
                     gamma = None, price_sum = None):
        if gamma is None and price_sum is None:
            price_sum = 1.
        if self.active_session == False:
            self.initialize_session()
        if len(Z.shape) == 2:
            Z = np.expand_dims(Z, 0)
            if not_missing is not None:
                not_missing = np.expand_dims(not_missing, 0)
        T, N, K = Z.shape
        if not_missing is None:
            not_missing = np.ones((T, N))
        self.set_data(Z, np.zeros((T, N)), not_missing)
        mu, Gamma, tphi = self.session.run([self.mean_predictions, self.Gamma, self.phi],
                                           feed_dict = {self.T: T, self.use_indices: list(range(T))})
        phi = np.exp(tphi).astype(np.float64)
        if self.multiply_x is not None:
            mu = mu / self.multiply_x
            Gamma = Gamma / self.multiply_x
            phi = phi / self.multiply_x**2
        Tmax, Nmax = mu.shape
        mus = T * [None]
        Gammas = T * [None]
        xis = T * [None]
        mispricing = T * [None]
        for tt in range(T):
            keep = np.pad(not_missing[tt, :], (0, Nmax - N), mode = 'constant') > 0
            mus[tt] = mu[tt, keep]
            Gammas[tt] = Gamma[tt, keep, :].flatten()
            if weights is not None:
                xis[tt] = weights[tt, keep]
                Sigma_xi = phi * xis[tt] + Gammas[tt] * np.sum(Gammas[tt] * xis[tt])
                alpha_p = mus[tt] - Sigma_xi * np.sum(xis[tt] * mus[tt]) / np.sum(xis[tt] * Sigma_xi)
                Gamma2 = np.sum(Gammas[tt]**2)
                Gamma_alpha_p = np.sum(Gammas[tt] * alpha_p)
                alpha_p2 = np.sum(alpha_p**2)
                mispricing[tt] = alpha_p2 / phi - Gamma_alpha_p**2 / (phi * (phi + Gamma2))
        return {'phi': phi, 'mus': mus, 'Gammas': Gammas, 'mispricing': mispricing}
    
    """
    def get_matrices(self, Z, weights, rf, gamma = None, price_sum = None):
        if gamma is None and price_sum is None:
            price_sum = 1.
        if self.active_session == False:
            self.initialize_session()
        N, K = Z.shape
        self.set_data(np.expand_dims(Z, axis = 0),
                      np.zeros((1, N)), np.ones((1, N)))
        mu, Gamma, tphi = self.session.run([self.mean_predictions, self.Gamma, self.phi],
                                           feed_dict = {self.T: 1, self.use_indices: [0]})
        mu = mu.flatten()[:N].astype(np.float64)
        Gamma = Gamma[0, :N, :].astype(np.float64) 
        phi = np.exp(tphi).astype(np.float64)
        if self.multiply_x is not None:
            mu = mu / self.multiply_x
            Gamma = Gamma / self.multiply_x
            phi = phi / self.multiply_x**2
        IN = np.identity(N)
        IJ = np.identity(self.J)
        Sigma = phi * IN + Gamma @ Gamma.T
        market_cov = Sigma @ weights
        market_var = np.sum(weights * market_cov)
        beta = market_cov / market_var
        market_return = np.sum(weights * mu)
        alpha = mu - beta * market_return
        SigmaInv = (IN - Gamma @ np.linalg.solve(phi * IJ + Gamma.T @ Gamma, Gamma.T)) / phi
        if price_sum is None:
            price_sum = market_return / (market_var * gamma)
        if gamma is None:
            gamma = market_return / (market_var * price_sum)
        price = price_sum * weights
        Lambda = price.reshape(-1, 1) * price.reshape(1, -1) * Sigma
        LambdaInv = SigmaInv * (1. / price.reshape(-1, 1)) * (1. / price.reshape(1, -1))
        Ed = price * (1 + rf) + price * mu
        delta = - price * alpha
        incumbent_demand = LambdaInv @ (Ed - price * (1 + rf + alpha)) / gamma
        rational_price = (Ed - gamma * np.sum(Lambda, 0)) / (1. + rf)
        try:
            np.linalg.cholesky(LambdaInv * (1 + rf + alpha).reshape(-1, 1) + LambdaInv * (1 + rf + alpha).reshape(1, -1))
            is_positive_definite = True
        except:
            is_positive_definite = False
        return {'alpha': alpha, 'mu': mu, 'Sigma': Sigma, 'SigmaInv': SigmaInv,
                'Lambda': Lambda, 'LambdaInv': LambdaInv, 'Ed': Ed, 'delta': delta,
                'incumbent_demand': incumbent_demand, 'rational_price': rational_price,
                'price': price, 'gamma': gamma, 'price_sum': price_sum,
                'is_positive_definite': is_positive_definite,
                'market_return': market_return, 'market_var': market_var}
    """
        
    def linear_r2(self, Z, not_missing, params):
        train_beta_mu0, train_beta_mu1, train_beta_Gamma0, train_beta_Gamma1, train_phi = extract_params(params)
        test_beta_mu0, test_beta_mu1, test_beta_Gamma0, test_beta_Gamma1, test_phi = extract_params(self.get_model_variables())
        T, Nmax, K = Z.shape
        num = 0.
        denom = 0.
        for tt in range(T):
            Zt = Z[tt, not_missing[tt, :] > 0.5, :]
            train_mu = train_beta_mu0 + Zt @ train_beta_mu1
            train_Gamma = train_beta_Gamma0 + Zt @ train_beta_Gamma1
            test_mu = test_beta_mu0 + Zt @ test_beta_mu1
            test_Gamma = test_beta_Gamma0 + Zt @ test_beta_Gamma1
            error = test_mu - ssimu(test_Gamma, train_Gamma, test_phi, train_phi, train_mu)
            #error = test_mu - (calc_cov(test_phi, test_Gamma) @ calc_cov_inverse(train_phi, train_Gamma) @ train_mu)
            num += np.sum(error**2)
            denom += np.sum(test_mu**2)
        return 1. - num / denom
    
    def calc_data_sharpe(self, Z, r, not_missing, params):
        if params is not None:
            beta_mu0, beta_mu1, beta_Gamma0, beta_Gamma1, phi = extract_params(params)
        else:
            beta_mu0, beta_mu1, beta_Gamma0, beta_Gamma1, phi = extract_params(self.get_model_variables())
        T, Nmax, K = Z.shape
        port_returns = T * [0.]
        for tt in range(T):
            Zt = Z[tt, not_missing[tt, :] > 0.5, :]
            mu = beta_mu0 + Zt @ beta_mu1
            Gamma = beta_Gamma0 + Zt @ beta_Gamma1
            w = calc_linear_weights(mu, Gamma, phi)
            port_returns[tt] = np.sum(r[tt, not_missing[tt, :] > 0] * w)
        sharpe = np.mean(port_returns) / np.std(port_returns)
        return sharpe
        












