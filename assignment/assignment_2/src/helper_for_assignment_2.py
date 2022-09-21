from math import sqrt, exp, pi
from pyexpat import model
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from random import randint
import scipy


def modal_gaussian_sampling(mu:list, sd:list, size:int):

    res = [] 
    n = size//3

    for (m, s) in zip(mu, sd):
        tmp = np.random.normal(loc=m, scale=s, size=n)
        tmp = list(tmp)
        res += tmp 
    
    return res 

def plot_poly_regression(degree:list, y:list, fig_size:tuple):

    plt.figure(figsize=fig_size)
    dummy_x = list(range(len(y)))

    for (i, d) in enumerate(degree):

        plt.subplot(len(degree) + 1, 2, i+1)

        model = np.poly1d( np.polyfit(dummy_x, y, deg = d))

        plt.scatter(dummy_x, y, marker='.', alpha=0.5)
        plt.plot(dummy_x, model(dummy_x), 'r-')
        plt.title(f"degree = {d}")

def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    p = 1-scipy.stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p