"""
@author: Maziar Raissi
"""

import numpy as np
import tensorflow as tf
import time
from abc import ABC, abstractmethod

class FBSNN(ABC): # Forward-Backward Stochastic Neural Network
    def __init__(self, Xi, T,
                       M, N, D,
                       layers):
        
        self.Xi = Xi # initial point
        self.T = T # terminal time
        
        self.M = M # number of trajectories
        self.N = N # number of time snapshots
        self.D = D # number of dimensions
        
        # layers
        self.layers = layers # (D+1) --> 1
        
        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        # optimizers
        self.optimizer = tf.keras.optimizers.Adam()

    
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim],
                                               stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_u(self, t, X): # M x 1, M x D
        with tf.GradientTape() as tape:
            tape.watch(X)
            u = self.neural_net(tf.concat([t,X], 1), self.weights, self.biases) # M x 1
        Du = tape.gradient(u, X) # M x D
        
        return u, Du

    def Dg_tf(self, X): # M x D
        with tf.GradientTape() as tape:
            tape.watch(X)
            g = self.g_tf(X)
        return tape.gradient(g, X)# M x D
        
    def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        loss = 0
        X_list = []
        Y_list = []
        
        t0 = t[:,0,:]
        W0 = W[:,0,:]
        X0 = tf.tile(Xi,[self.M,1]) # M x D
        Y0, Z0 = self.net_u(t0,X0) # M x 1, M x D
        
        X_list.append(X0)
        Y_list.append(Y0)
        
        for n in range(0,self.N):
            t1 = t[:,n+1,:]
            W1 = W[:,n+1,:]
            X1 = X0 + self.mu_tf(t0,X0,Y0,Z0)*(t1-t0) + tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1)), axis=[-1])
            Y1_tilde = Y0 + self.phi_tf(t0,X0,Y0,Z0)*(t1-t0) + tf.reduce_sum(Z0*tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1))), axis=1, keepdims = True)
            Y1, Z1 = self.net_u(t1,X1)
            
            loss += tf.reduce_sum(tf.square(Y1 - Y1_tilde))
            
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1
            
            X_list.append(X0)
            Y_list.append(Y0)
            
        loss += tf.reduce_sum(tf.square(Y1 - self.g_tf(X1)))
        loss += tf.reduce_sum(tf.square(Z1 - self.Dg_tf(X1)))

        X = tf.stack(X_list,axis=1)
        Y = tf.stack(Y_list,axis=1)
        
        return loss, X, Y, Y[0,0,0]

    def fetch_minibatch(self):
        T = self.T
        
        M = self.M
        N = self.N
        D = self.D
        
        Dt = np.zeros((M,N+1,1)) # M x (N+1) x 1
        DW = np.zeros((M,N+1,D)) # M x (N+1) x D
        
        dt = T/N
        
        Dt[:,1:,:] = dt
        DW[:,1:,:] = np.sqrt(dt)*np.random.normal(size=(M,N,D))
        
        t = np.cumsum(Dt,axis=1) # M x (N+1) x 1
        W = np.cumsum(DW,axis=1) # M x (N+1) x D
        
        return t, W
    
    def train(self, N_Iter, learning_rate):
        self.optimizer.learning_rate = learning_rate
        start_time = time.time()
        for it in range(N_Iter):

            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D
            t_batch = tf.convert_to_tensor(t_batch, dtype=tf.float32)
            W_batch = tf.convert_to_tensor(W_batch, dtype=tf.float32)
            Xi = tf.convert_to_tensor(self.Xi, dtype=tf.float32)
            with tf.GradientTape() as tape:
                loss, _, _, Y0_pred = self.loss_function(t_batch, W_batch, Xi)
            gradients = tape.gradient(loss, self.weights + self.biases)
            self.optimizer.apply_gradients(zip(gradients, self.weights + self.biases))

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                _, _, _, Y0_value = self.loss_function(t_batch, W_batch, Xi)
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' % 
                      (it, loss.numpy(), Y0_value.numpy(), elapsed, learning_rate))
                start_time = time.time()
                
    
    def predict(self, Xi_star, t_star, W_star):

        Xi_star = tf.convert_to_tensor(Xi_star, dtype=tf.float32)
        t_star = tf.convert_to_tensor(t_star, dtype=tf.float32)
        W_star = tf.convert_to_tensor(W_star, dtype=tf.float32)
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)
        
        return X_star.numpy(), Y_star.numpy()
    
    ###########################################################################
    ############################# Change Here! ################################
    ###########################################################################
    @abstractmethod
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        pass # M x1
    
    @abstractmethod
    def g_tf(self, X): # M x D
        pass # M x 1
    
    @abstractmethod
    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return np.zeros([M,D]) # M x D
    
    @abstractmethod
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return tf.linalg.diag(tf.ones([M,D])) # M x D x D
    ###########################################################################