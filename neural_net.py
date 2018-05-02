#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 00:08:27 2018

@author: NikithaShravan
"""
import os
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
import time
import random

params = {}
params['batch_size'] = 100
params['num_epochs'] = 1000
params['learning_rate'] = 1e-6
params['activation'] = 'relu'
params['optimizer'] = 'sgd'
#params['layers'] = [4096, 4096]
params['output_dim'] = 10
params['input_dim'] = 28*28

def relu_function(z):
    return np.maximum(z,0)

def tanh_function(z):
    return np.tanh(z)

def sigmoid_function(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_backward(z):
    return sigmoid_function(z)*(1.0 - sigmoid_function(z))
def tanh_backward(z):
    return np.arctanh(z)
def linear_forward(X, W, b):
    #  X: NxD, W1: dimensions: DxL1, b: 1xL1
    return np.dot(X ,W ) + b

def non_linear_activation(z,activation):
    if activation == "relu":
        return relu_function(z)
    if activation == "tanh":
        return tanh_function(z)
    if activation == "sigmoid":
        return sigmoid_function(z)
def relu_backward(z):
    r = relu_function(z)
    if np.max(r) == 0:
        pass
    else :
        r = np.ceil(r/np.max(r))
    return r
    
def non_linear_backward(z,activation):
    if activation == "relu":
        return relu_backward(z)
    if activation == "tanh":
        return tanh_backward(z)
    if activation == "sigmoid":
        return sigmoid_backward(z)
    
def compute_loss(z, y):  
#    print("np.max(np.absolute(z): ", np.max(np.absolute(z)))
    z = z / np.max(np.absolute(z))
    z = z + 1e-12
    prob = np.exp(z)/np.sum(np.exp(z),axis=1,keepdims=True) 
    p_i = prob[np.arange(len(y)),list(np.squeeze(y))]
#    print(p_i[:3])
#    print("pi shape: ", p_i.shape," y.shape", np.squeeze(y)[:4])
    loss=0
    loss = -np.sum(np.log(p_i))
#    print(np.sum(np.log(p_i) <  -100)) 
    #print(prob.shape)      
    return loss/prob.shape[0]

def compute_loss_grad(z, y):
#    print("np.max(np.absolute(z): ", np.max(np.absolute(z)))

    z = z / np.max(np.absolute(z)) 
    z = z + 1e-12
    dz =  np.exp(z)/np.sum(np.exp(z),1, keepdims=True)    
    dz[range(len(y)),np.array(y)] -= 1.0     
    return dz/z.shape[0]
def linear_backward(A,dZ):
    return np.dot(dZ.T,A).T

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g
class neural_net(object):
    def __init__(self, layers):
        
        self.params = {}
        self.layers = layers
        for i in range(len(self.layers)-1):
            self.params['W'+str(i+1)] = np.random.randn(layers[i],layers[i+1])
            self.params['b'+str(i+1)] = np.random.randn(layers[i+1])
        self.gini_coeff = []
            
    def update_parameters(self):
        pass
    def epoch(self,X_train, y_train,learning_rate, activation):
        batch_size = params['batch_size']
#        print(np.floor(X_train.shape[0]/batch_size))
        for i in range(int(np.floor(X_train.shape[0]/batch_size))):
#            print("i: ", i)
            idx_rand = np.random.randint(0,X_train.shape[0], size=(batch_size))
            x_batch = X_train[idx_rand,:]
            y_batch = y_train[idx_rand]
            # X_train dim: NxD
            A = x_batch
            N = x_batch.shape[0]
            H = len(self.params)//2
#        assert(y_train.shape == (N,1))
            cache = []               
            #Z: NxL1  
            for i in range(H-1):
            
                # Z: Nxlayer(i+1),A: Nxlayer(i), W: layer(i)xlayer(i+1) 
                Z = linear_forward(A, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
                cache.append((A,Z))
                # non linear activation: A:Nxlayer(i+1)
                A = non_linear_activation(Z, activation)
                assert(A.shape == (N,self.params['W'+str(i+1)].shape[1]))
            
        
            Z = linear_forward(A, self.params['W'+str(H)], self.params['b'+str(H)])
            cache.append((A,Z))
            assert(Z.shape == (N, self.layers[-1]))
        
        #        y = non_linear_activation(Z, "sigmoid")
        #        print(Z)
        #        print(y)
        
            loss = compute_loss(Z, y_batch)#, "cross-entropy")
#            print("loss = ", loss)
        
            grads = {}
       
            dZ = compute_loss_grad(Z, y_batch)#, "cross-entropy")
            #        print(len(cache))
            #        dZ = dy*non_linear_backward(Z,"sigmoid")
        
            for i in reversed(range(H)):
                #A : Nxlayer(i+1)
                #            print(dZ.shape, cache[i][0].shape )
                dW = np.dot(cache[i][0].T, dZ)
                dA = np.dot(dZ,self.params['W'+str(i+1)].T)
                db = np.squeeze(np.sum(dZ,axis=0))
            
                grads['W'+str(i+1)] = dW
                grads['b'+str(i+1)] = db
                if i == 0:
                    break
    #            print("cache size: ", cache[i-1][1].shape)
    #            print("A: ",dA.shape )
    #            print("i: ", i)
    #            print("H: ", H)
                dZ = dA*non_linear_backward(cache[i-1][1],activation)
        
            for i in range(H):
                self.params['W'+str(i+1)] -= learning_rate*grads['W'+str(i+1)]
                self.params['b'+str(i+1)] -= learning_rate*grads['b'+str(i+1)]
            """ =================== GINI INDEX ======================= """                
#            dl_dx = np.zeros((x_batch.shape[0], x_batch.shape[1]))
#            g_x = np.zeros(x_batch.shape[0])
#            dl_dx = 
            g_x = np.sum(np.absolute(np.dot(dZ,self.params['W1'].T)),axis=1)
#            for i in range(1000):
#                print(model.trainable_weights[0].shape)
#                print(np.sum(np.divide(dl_dw, X_train[i].reshape(784,1)).shape))
#                dl_dx[i] = np.sum(np.divide(dl_dw, X_train[i].reshape(784,1)) * model.layers[0].get_weights()[0], 1)
#                g_x[i] = np.linalg.norm(dl_dx[i])
#                print(g_x[i])
#            DL_DX += g_x[i] /(i+1)
            gini_x = gini(g_x)
            self.gini_coeff.append(gini(g_x))
            
            """ ==================== """
        acc = self.test(X_train[:600], y_train[:600])        
        return np.sum(loss), acc
    def train(self,X_train, y_train,num_epochs,learning_rate):
        loss = []
        for i in range(num_epochs):
            print("Epoch ", i,"/",num_epochs)
            l, acc = self.epoch(X_train[:600],y_train[:600],learning_rate,"relu")
            print("       loss: ", l," accuracy: ", acc)
            loss.append(l)
        return loss, acc
    
    def test(self, XTest, yTest):
#        z1 = np.dot(XTest, self.W1) + self.b1
#                   
#        if self.activation == 'relu':
#            a1 = relu(z1)
#        elif self.activation == 'sigmoid':
#            a1 = sigmoid(z1)
#        elif self.activation == 'tanh':
#            a1 = tanh(z1)
#        
#        z2 = np.dot(a1, self.W2) + self.b2
        activation = 'relu'
        A = XTest
        N = XTest.shape[0]
        H = len(self.params)//2
        cache = []               
            #Z: NxL1  
        for i in range(H-1):
            
            # Z: Nxlayer(i+1),A: Nxlayer(i), W: layer(i)xlayer(i+1) 
            Z = linear_forward(A, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            cache.append((A,Z))
            # non linear activation: A:Nxlayer(i+1)
            A = non_linear_activation(Z, activation)
            assert(A.shape == (N,self.params['W'+str(i+1)].shape[1]))
            
        
        Z = linear_forward(A, self.params['W'+str(H)], self.params['b'+str(H)])
        cache.append((A,Z))        
        z = Z/ np.max(np.absolute(Z))
        z = z + 1e-12
        prob = np.exp(z)/np.sum(np.exp(z),axis=1,keepdims=True)         
        yPred = np.argmax(prob,axis=1)
        acc = yPred - yTest

        return np.sum(acc==0)/acc.shape[0]

num_inputs = 1000
input_dims = 2
X_train = 1*np.random.randn(num_inputs, input_dims)
#print(X_train)
y_train = np.random.randint(10,size=(1,num_inputs)).T 
#print(y_train)
#np.random.seed(1)

def loadData():
    
    num_classes = 10

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    #Y_train = keras.utils.to_categorical(Y_train, num_classes)
    #Y_test = keras.utils.to_categorical(Y_test, num_classes)
    
    num_rows = X_train.shape[1]
    num_cols = X_train.shape[2]
    #num_channels = X_train.shape[3]
    input_dims = num_rows*num_cols 
    print(X_train.shape)
    
    X_train = X_train.reshape(X_train.shape[0], input_dims)
    X_test = X_test.reshape(X_test.shape[0], input_dims)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
                          
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = loadData()

print("mean before normalization:", np.mean(X_train))
print("std before normalization:", np.std(X_train))

mean=[0,0,0]
std=[0,0,0]
newX_train = np.ones(X_train.shape)
newX_test = np.ones(X_test.shape)

mean = np.mean(X_train)
std = np.std(X_train)
    

newX_train = X_train - mean
newX_train = newX_train / std
newX_test = X_test - mean
newX_test = newX_test / std
        
    
X_train = newX_train
X_test = newX_test

print("mean after normalization:", np.mean(X_train))
print("std after normalization:", np.std(X_train))

def Add_noise(p, x):
        data = x
        num = int(p*len(x))
        ix = random.sample(range(0, len(x)), num)
        for i in ix:
            randX = np.std(X_train)*np.random.randn(1, x.shape[1]) + np.mean(X_train)
            data[i] = randX
        return data
x_data = np.std(X_train)*np.random.randn(X_train.shape[0], X_train.shape[1]) + np.mean(X_train)
num_classes = 10
params['layers'] = [X_train.shape[1], 4096, 4096, num_classes ]
net = neural_net(params['layers'])
net.train(x_data, Y_train, params['num_epochs'], params['learning_rate'])
gini_vals = net.gini_coeff