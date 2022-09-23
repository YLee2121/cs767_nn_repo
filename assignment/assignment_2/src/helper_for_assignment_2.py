import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random
import scipy
from sklearn.metrics import * 

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

class Pwr_linear:
    "piece wise linear regression"
    # init
    def __init__(self, n_knots, x, y):
        self.n_knots = n_knots
        self.x = x 
        self.y = y 
        self.knot = []
        self.train_x = [ [] for _ in range(n_knots + 1)] 
        self.train_y = [ [] for _ in range(n_knots + 1)]
        self.lowest_mse = float("inf")
        self.best_knot = [] 
    
    


    # method
    def best_prediction(self):
    
        self.knot = self.best_knot
        self.cut_data()
        yhat = [] 
        for tx, ty in zip(self.train_x, self.train_y):
            
            lr = LinearRegression()
            lr.fit(tx, ty)
            foo = lr.predict(tx)
            foo.reshape((1, -1))
            yhat += foo.tolist() 

        return yhat, mean_squared_error(yhat, self.y)

    def train(self):

        self.random_knot_idx()
        self.cut_data()

        mse = 0

        for tx, ty in zip(self.train_x, self.train_y):
            lr = LinearRegression()
            lr.fit(tx, ty)
            yh = lr.predict(tx) 
            mse += mean_squared_error(yh, ty)

        if mse < self.lowest_mse:
            self.lowest_mse = mse 
            self.best_knot = self.knot

    def train_iter(self, n:int):
        for _ in range(n):
            self.train()
        return self.best_knot, self.lowest_mse

    def plot_knot_spine(self, knot:list):
        plt.scatter(self.x, self.y, alpha=0.3)
        plt.scatter(self.x[ knot ], self.y[ knot ], c='red', label='knot')
        for l in knot:
            plt.axvline(x=self.x[l], color='b', linestyle="--")
        plt.legend()
        plt.show()

    def random_knot_idx(self):
        def valid(knot):
            k_set = set() 

            for i in range(len(knot)-1):

                k = knot[i]

                if k in k_set:
                    return False 
                
                k_set.add(k)

                if knot[i+1] - knot[i] <= 2:
                    return False 
        
            return True 

        while True:
            knot = [random.randrange(2, len(self.x) - 1) for _ in range(self.n_knots)]
            knot.sort()

            if valid(knot):
                self.knot = knot
                break
    

    def cut_data(self):

        bin_range = [0] + self.knot + [len(self.x)]

        for i in range(len(bin_range)-1):
            
            l_i = bin_range[i]
            r_i = bin_range[i+1]

            self.train_x[i] = self.x[l_i: r_i]
            self.train_y[i] = self.y[l_i: r_i]
        

        return self.train_x

class Pwr_poly(Pwr_linear):
    # init
    def __init__(self, n_knots, x, y, degree):
        self.n_knots = n_knots
        self.x = x 
        self.y = y 
        self.degree = degree
        self.knot = []
        self.train_x = [ [] for _ in range(n_knots + 1)] 
        self.train_y = [ [] for _ in range(n_knots + 1)]
        self.lowest_mse = float("inf")
        self.best_knot = [] 
    def best_prediction(self):
    
        self.knot = self.best_knot
        self.cut_data()
        yhat = [] 
        for tx, ty in zip(self.train_x, self.train_y):

            model = np.poly1d( np.polyfit(tx, ty, deg=self.degree))
            foo = model(tx)
            foo.reshape((1, -4))
            yhat += foo.tolist()
        
        return yhat , mean_squared_error(yhat, self.y)


        
    def train(self):

        self.random_knot_idx()
        self.cut_data()

        mse = 0

        for tx, ty in zip(self.train_x, self.train_y):

            model = np.poly1d( np.polyfit(tx, ty, deg=self.degree))
            yh=model(tx)
            mse += mean_squared_error(yh, ty)

        if mse < self.lowest_mse:
            self.lowest_mse = mse 
            self.best_knot = self.knot

class Pwr_experiment:
    @staticmethod
    def pwr(x, y):
        x_idx = list(range(len(x)))
        best_knot = None 
        lowest_error = float("inf")
        best_knot_f = None 
        lowest_error_f = float("inf")
        for knot_idx in x_idx[2:-2]:

            l_x = x[:knot_idx]
            r_x = x[knot_idx:]
            l_y = y[:knot_idx]
            r_y = y[knot_idx:]

            f_total = 0 
            mse = 0

            l1 = LinearRegression()
            l1.fit(l_x, l_y)
            y_hat = l1.predict(l_x)
            f_total += f_test(y_hat, l_y)[0]
            mse += mean_squared_error(y_hat, l_y)

            l2 = LinearRegression()
            l2.fit(r_x, r_y)
            y_hat = l2.predict(r_x)
            f_total += f_test(y_hat, r_y)[0]
            mse += mean_squared_error(y_hat, r_y)

            if mse < lowest_error:
                lowest_error = mse 
                best_knot = knot_idx 

            if f_total < lowest_error_f:
                lowest_error_f = f_total 
                best_knot_f = knot_idx

        return best_knot, best_knot_f
    @staticmethod
    def plot_pwr(x, y, best_knot, best_knot_f):
        plt.scatter(x, y, marker='.')

        plt.scatter(x[best_knot], y[best_knot],c='red')
        plt.scatter(x[best_knot_f], y[best_knot_f], c='green')
        print("yellow for f, red for mse")
        plt.show()