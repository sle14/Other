from datetime import datetime, timedelta
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import utils

#Apply spectral decomposition on the covariance matrix    
def decompose(data):
    cov_mat = utils.cov(data)
    eig_val, eig_vec  = np.linalg.eig(cov_mat)
    return eig_val, eig_vec

#Determine number of principal components such that 0.9 variation in data is explained 
def prob_comp(eig_val, prob, plot=False):
    i,p_,py = 0,0,[]
    while p_ < prob:
        p_ = sum(eig_val[0:i+1]) / sum(eig_val)
        py.append(eig_val[i] / sum(eig_val))
        i += 1
    if plot == True:
        plt.bar(list(range(0,i)), py)
        plt.show()
    return i

#Determine the amount of variation in data is explained by selecting number of principal components
def comp_prob(eig_val, pcs):
    i = 0
    while i < pcs:
        p_ = sum(eig_val[0:i+1]) / sum(eig_val)
        i += 1
    return p_

#Correlation between the original variables and the principal components
def loadings(eig_val, eig_vec):
    return eig_vec * np.sqrt(eig_val)

#Proportion of variance in a variable that is captured by principal components
#So, for each variable the sum of its squared loading across all PCs equals to 1
def cum_loadings(loadings):
    return np.sum(loadings**2, axis=1)
    
#Projection of the data in the new space
def scores(eig_vec, data):
    return np.inner(eig_vec.T, data).T

#Reconstruct original data with selected number of principal components
def reconstruct(scores, eig_vec, data, pcs):
    return np.dot(scores[:,:pcs], eig_vec.T[:pcs,:]) + np.mean(data, axis=0)

# i = 1  
pp = 0.9 #we want pp amount of variation in data to be explained
tickers = utils.get_tickers()

lr = utils.get_log_returns(tickers).to_numpy()
X = utils.stan(lr)
    
#How many principal components can explain 0.9 of variation in data?
e,V = decompose(X)
I = prob_comp(e,pp,True)
e,V = e[:I],V[:,:I]


# #Using 6 principal components, we replicate normalised returns of the index
# L = loadings(e,V)
# S = scores(V,X)
# X_hat = reconstruct(S,V,X,I)
# L_cum = cum_loadings(L)


#Determine power of each log returns vector
dct = {"Ticker":[],"Power":[]}
for ticker in tickers:
    
    tickers_ = tickers.copy()
    tickers_.remove(ticker)
    lr_ = utils.get_log_returns(tickers_).to_numpy()
    lr_ = utils.stan(lr_)
    
    e_,V_ = decompose(lr_)
    power = pp - comp_prob(e_,I) #
    
    print(ticker, power)
    
    dct["Ticker"].append(ticker)
    dct["Power"].append(power)
 
df = pd.DataFrame(dct).set_index("Ticker")
print(df)

# #Sanity check
# pca = PCA(pp)
# pca.fit(X)
# e2 = pca.explained_variance_ 
# V2 = pca.components_.T
# I2 = pca.n_components_

# L2 = pca.components_.T * np.sqrt(pca.explained_variance_)
# S2 = pca.transform(X)
# X_hat2 = np.dot(S2[:,:6], V2.T[:6,:]) + np.mean(X, axis=0)



