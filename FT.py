from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
import numpy as np
import utils


#y = ArmaProcess(ar=np.r_[1, -0.5]).generate_sample(1000)
#Simulate MA1!!

def ar1(b,T):
    y = np.zeros(T)
    y[0] = norm.rvs(0, 1)
    for t in range(1,T): 
        y[t] = b*y[t-1] + norm.rvs(0, 1)
    return y

#--------------------------------------------------------------------

# df = utils.get_stock_HF("ABNB")
# df["Spread"] = (df["Ask"] - df["Bid"]) / df["Mid"]
# df = df.reset_index().groupby(by=["Date","Hour"])["Spread"].mean()
# s = df.reset_index(drop=True) 
# y = np.log(censor(s,0.5))

#--------------------------------------------------------------------
days = 360
y = ar1(0,days)

# plt.plot(y); plt.show()
# plt.hist(y); plt.show()
# acf(y)
# pacf(y)

# res = ARIMA(y,order=(1,0,0)).fit()
# print(res.summary())

#--------------------------------------------------------------------
#Apply FT, convert to frequency domain

#https://www.asc.ohio-state.edu/de-jong.8/note3.pdf