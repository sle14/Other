from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
import pandas as pd
import numpy as np

def acf(x):
    sm.graphics.tsa.plot_acf(x)
    plt.show()

def pacf(x):
    sm.graphics.tsa.plot_pacf(x, method='ywm')
    plt.show()

def stan(x):
    return (x-np.mean(x))/np.std(x)

def cov(x):
    return np.cov(x.T)

def censor(x, p):
    T = len(x)
    lst = [0, *np.random.choice([0,1],size=T-1,p=[1-p,p]).tolist()]
    idx = np.nonzero(np.array(lst))[0]
    x[idx] = np.nan
    return x.dropna()

#------------------------------------------------------------------------
#Data

PATH_DATA = "C:/Users/Aigars/Desktop/Citadel/Stats/DATA"

def get_stock_HF(ticker):   
    df = pd.read_csv(f"{PATH_DATA}/stock_HF/{ticker}.csv")
    df["Date"] = pd.to_datetime(df["Date"],format="%Y-%m-%d").dt.date
    # df = df[df["Hour"] > 9]
    df = df.set_index(["Date","Hour","Min"])
    return df

def get_tickers(drop=["CEG","EXC"]):
    df = pd.read_csv(f"{PATH_DATA}/members/NDX.csv")
    df = df[df["Part2022"]==1]
    df = df["Ticker"].to_list()
    if drop is not None:
        df = [x for x in df if x not in drop]
    return df

def get_stock(ticker,cols=["Date","Open","Close","High","Low"]):
    df = pd.read_csv(f"{PATH_DATA}/stock/{ticker}.csv")
    df["Date"] = pd.to_datetime(df["Date"],format="%Y-%m-%d")
    if len(cols) > 0:
        df = df[cols]
    return df

def get_log_returns(tickers):
    df = get_stock(tickers[0],["Date","Close"]).set_index("Date")
    df.columns = [tickers[0]]
    for ticker in tickers[1:]:
        df_ = get_stock(ticker,["Date","Close"]).set_index("Date")
        df_.columns = [ticker]
        df = df.join(df_)
    df = np.log(1 + df.pct_change().dropna())
    return df




