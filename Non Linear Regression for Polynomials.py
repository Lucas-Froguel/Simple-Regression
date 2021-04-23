# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:21:24 2021

@author: Lucas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre




def points(n, e=0.1, a=-1, b=1, poly=legendre(3)):
    x = (b-a)*np.random.rand(n) - (b-a)/2
    y = np.zeros(len(x))
    for i in range(0, len(poly)+1):
        y += poly[i]*x**(i)
    y += e*(2*np.random.rand(n)-1)
    return x, y

def polynomial(x, coeff):
    y = 0
    for i in range(len(coeff)):
        y += coeff[i]*(x**i)
    return y

def leg_polynomial(x, poly):
    y = 0
    for i in range(len(poly)+1):
        y += poly[i]*(x**i)
    return y

def transform(x, d):
    X = np.zeros((len(x), d+1))
    for i in range(X.shape[1]):
        X[:,i] = x**i
    return X

def regression_with_regularization(X, Y, l=0.1):
    Xt = np.transpose(X)
#    l = 10**l
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(Xt, X)+l*np.identity(X.shape[1])), Xt), Y)

def E_in(w, x, y):
    return sum((polynomial(x, w) - y)**2)/len(y)

def E_out(w, poly):
    X = np.random.rand(10000)
    y = leg_polynomial(X, poly)
    return E_in(w, X, y)

class Non_Linear_Regression:
    def __init__(self, l=0):
        self.l = l
        self.w = 0 
    
    def transform(self, x, d):
        X = np.ones((len(x), d+1))
        for i in range(1, X.shape[1]):
            X[:,i] = x**i
        return X
    
    def fit(self, X, Y, in_place=True, l = -1):
        Xt = np.transpose(X)
        if l == -1:
            l = self.l
        w = np.matmul(np.matmul(np.linalg.inv(np.matmul(Xt, X)+l*np.identity(X.shape[1])), Xt), Y)
        if in_place:
            self.w = w
        return w
    
    def E_in(self, x, y, w=0):
        return sum((self.fit_polynomial(x, w=w) - y)**2)/len(y)
    
    def E_in_new(self, X, y, w=0):
        return sum((np.dot(X, w)-y)**2)/len(y)

    def E_out(self, poly):
        X = np.random.rand(10000)
        y = leg_polynomial(X, poly)+0.1*np.random.rand(10000)
        return E_in(self.w, X, y)
    
    def fit_polynomial(self, x, w=0):
        y = 0
        if np.isscalar(w):
            w = self.w
        for i in range(len(self.w)):
            y += w[i]*(x**i)
        return y
    
    def cross_validation_lambda(self, X, y, P=10):
        part = np.random.choice(range(P), len(x))
        num = 1000
        E_cv = np.zeros(num)
        t = 0
        for l in np.linspace(0, 1, num=num):
            for i in range(P):
                ind = np.where(part == i, True, False)
                w = self.fit(X[~ind], y[~ind], in_place=False, l=l)
                #print(w, X.shape, X[~ind].shape, X[ind].shape, y[ind].shape)
                E_cv[t] += self.E_in_new(X[ind], y[ind], w=w)
            E_cv[t] = E_cv[t]/P
            t += 1
        return np.where(min(E_cv)==E_cv)[0][0]/num
    
    def cross_validation_degree(self, x, y, degrees, lambdas, P=10):
        part = np.random.choice(range(P), len(x))
        E_cv = np.zeros(degrees)
        t = 0
        for k in range(degrees):
            X = self.transform(x, k)
            for i in range(P):
                ind = np.where(part == i, True, False)
                w = self.fit(X[~ind], y[~ind], in_place=False, l=lambdas[k])
                
                E_cv[t] += self.E_in_new(X[ind], y[ind], w=w)
            E_cv[t] = E_cv[t]/P
            t += 1
        return np.where(min(E_cv)==E_cv)[0][0], min(E_cv)



k = 40 #highest degree to be tested
n = 100 #number of points on the data set
Ein = np.zeros(k)
Eout = np.zeros(k)

plt.figure()


#creates some random points with noise out from an legendre polynomial for testing the code
d = 20
poly = legendre(d)
val_x = np.linspace(-1, 1, num=500)
val_y = leg_polynomial(val_x, poly)

x, y = points(n, e=0.1, a=-1, b=1, poly=poly)
lambdas = np.zeros(k)

#cross validates the regularization
for i in range(k):
    Reg = Non_Linear_Regression()
    X = Reg.transform(x, i)
    
    lambdas[i] = Reg.cross_validation_lambda(X, y)

Reg = Non_Linear_Regression()
#cross validates the degree of the polynomial
K, Ecv = Reg.cross_validation_degree(x, y, k, lambdas)

Reg = Non_Linear_Regression(l=lambdas[K])
X = Reg.transform(x, K)
w = Reg.fit(X, y)
Y = Reg.fit_polynomial(val_x)

Eout = Reg.E_out(poly)

#plot the errors
print("Degree: ", K, "\nRegularization: ", lambdas[K], "\nE_out: ", Eout, "\nE_cv: ", Ecv)

#plot everything
plt.plot(x, y, 'o')
plt.plot(val_x, val_y, 'r', label="Target")
plt.plot(val_x, Y, 'g', label="Best")
plt.xlim(-1, 1)
plt.ylim(min(val_y), max(val_y))













