import numpy as np 
import pandas as pd 
from scipy.special import kl_div
from math import sqrt, log, exp, pi
from random import uniform
import matplotlib.pyplot as plt


class helper_for_assign_1:

    @staticmethod
    def gmm_train(data, n_iter, verbose=False):
        best_gmm = None 
        best_loglike = float("-inf")
        gmm = GMM_for_3_cluster(data)
        for _ in range(n_iter):
            try:
                # train 
                gmm.iterate(verbose=verbose)
                if gmm.loglike > best_loglike:
                    best_loglike = gmm.loglike 
                    best_gmm = gmm

            except (ZeroDivisionError, ValueError, RuntimeWarning): # Catch division errors from bad starts, and just throw them out...
                print("one less")
                pass
        
        return best_gmm, gmm

    @staticmethod
    def plot_gmm_mixture(gmm, data):
        Min_graph = min(data)
        Max_graph = max(data)
        x = np.linspace(Min_graph, Max_graph, 2000) # for ploting the Gaussian pdf

        # plot the data hist
        foo = plt.hist(data, bins=60, density=True, alpha=0.5)

        # plot Gaussian pdf line
        g_all = [gmm.pdf(e) for e in x]
        plt.plot(x, g_all, label='G mixture')

        g_1 = [gmm.g1.pdf(e) * gmm.m1 for e in x]
        plt.plot(x, g_1, label='G1', alpha=0.5)

        g_2 = [gmm.g2.pdf(e) * gmm.m2 for e in x]
        plt.plot(x, g_2, label='G2', alpha=0.5)

        g_3 = [gmm.g3.pdf(e) * gmm.m3 for e in x]
        plt.plot(x, g_3, label='G3', alpha=0.5)

        plt.legend(loc=1, prop={'size': 5})



    @staticmethod
    def cum_sum(l:list):
        res = [] 
        cur_sum = 0 
        for val in l:
            cur_sum += val 
            res.append(cur_sum)
        return res

    @staticmethod
    def combine_df(frame:list):
        df_combine = pd.concat(frame)
        df_combine = df_combine.reset_index()
        return df_combine

    @staticmethod
    def cluster_data_3d(x, y, z, sigma, n, cluster_name):
        x_tmp = np.random.normal(x, sigma, n)
        y_tmp = np.random.normal(y, sigma, n)
        z_tmp = np.random.normal(z, sigma, n)
        cluster = [cluster_name for _ in range(n)]
        df = pd.DataFrame(data=[x_tmp, y_tmp, z_tmp, cluster]).T 
        df.columns = ['x', 'y', 'z', 'label']
        return df

    @staticmethod
    def kl_divergence(f1:list, f2:list):
        kl = kl_div(f1, f2)
        kl = sum(kl)
        return kl 

    @staticmethod
    def js_divergence(f1:list, f2:list):
        
        foo1 = helper_for_assign_1.kl_divergence(f1, f2)
        foo2 = helper_for_assign_1.kl_divergence(f2, f1)
        return (foo1 + foo2) / 2  


class Gaussian:
    "Model univariate Gaussian"
    def __init__(self, mu, sigma):
        #mean and standard deviation
        self.mu = mu
        self.sigma = sigma

    #probability density function
    def pdf(self, datum):
        "Probability of a data point given the current parameters"
        u = (datum - self.mu) / abs(self.sigma)
        return (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
    
    def __repr__(self):
        return f'Gaussian({self.mu}, {self.sigma})'

class GMM_for_3_cluster:
    "GMM for three clusters"

    def __init__(self, data, sig_min=1, sig_max=1, m1=0.3, m2=0.3, m3=0.4):
        
        mu_min = min(data)
        mu_max = max(data)
        self.data = data 
        self.g1 = Gaussian(uniform(mu_min, mu_max), uniform(sig_min, sig_max))
        self.g2 = Gaussian(uniform(mu_min, mu_max), uniform(sig_min, sig_max))
        self.g3 = Gaussian(uniform(mu_min, mu_max), uniform(sig_min, sig_max))
        self.m1 = m1 
        self.m2 = m2 
        self.m3 = m3 
    
    def E(self):
        "Perform an E(stimation)-step, assign each point to gaussian 1 or 2 or 3 with a probability" 
        # computer weights 
        self.loglike = 0 
        for datum in self.data:

            # init weight 
            w1 = self.g1.pdf(datum) * self.m1 
            w2 = self.g2.pdf(datum) * self.m2
            w3 = self.g3.pdf(datum) * self.m3 

            # compute denominator 
            d = w1 + w2 + w3 

            # normalized the weight 
            w1 /= d 
            w2 /= d 
            w3 /= d 
            
            # add into loglike 
            self.loglike += log(d)

            # yield 
            yield(w1, w2, w3)

    def M(self, weights):
        "Perform an M step"
        # compute denominators 
        (w1, w2, w3) = zip(*weights)
        den1 = sum(w1)
        den2 = sum(w2)
        den3 = sum(w3)

        # compute new mean 
        self.g1.mu = sum(w * d for (w, d) in zip(w1, self.data)) / den1 
        self.g2.mu = sum(w * d for (w, d) in zip(w2, self.data)) / den2 
        self.g3.mu = sum(w * d for (w, d) in zip(w3, self.data)) / den3 

        # compute new sigma 
        foo = sum(w * ((d - self.g1.mu) ** 2) for (w, d) in zip(w1, self.data)) / den1
        self.g1.sigma = sqrt(foo)

        foo = sum(w * ((d - self.g2.mu) ** 2) for (w, d) in zip(w2, self.data)) / den2
        self.g2.sigma = sqrt(foo)

        foo = sum(w * ((d - self.g3.mu) ** 2) for (w, d) in zip(w3, self.data)) / den3
        self.g3.sigma = sqrt(foo)

        # compute new mix ratio 
        self.m1 = den1 / len(self.data)
        self.m2 = den2 / len(self.data)
        self.m3 = 1 - (self.m1 + self.m2)

    def iterate(self, N=1, verbose=False):
        "Perform N iterations, then computer log-likeihood"
        for i in range(1, N+1):
            self.M(self.E())
            if verbose:
                print(f'{i} {self}')
        self.E() 

    def pdf(self, datum):
        
        p1 = self.m1 * self.g1.pdf(datum)
        p2 = self.m2 * self.g2.pdf(datum)
        p3 = self.m3 * self.g3.pdf(datum)
        
        return p1 + p2 + p3 
    
    def __repr__(self):
        return f'GMM({self.g1}, {self.g2}, {self.g3}, {self.m1}, {self.m2}, {self.m3})'
    
    def __str__(self) -> str: 
        return f'Mixture: {self.g1}, {self.g2}, {self.g3}, {self.m1}, {self.m2}, {self.m3}'