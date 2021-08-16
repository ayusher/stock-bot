import alpaca_trade_api as tradeapi
import yfinance as yf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import coloredlogs
import tensorflow as tf
import numpy as np
import time
import sys
from agent import Agent
from methods import eval_model_new
from utils import get_stock_data, get_state
from get_all_tickers import get_tickers as gt
import matplotlib.pyplot as plt
#tf.get_logger().setLevel("ERROR")
#import logging
#tf.get_logger().setLevel(logging.ERROR)
buy_agg = .5
sell_agg = .5
getters = gt
ts = [input("stock ticker: ")]
#ts = ["CIDM", "LI", "AAPL", "BA", "IVR", "XAN", "MSFT", "TSLA", "GE", "F", "GM", "TWTR", "GEVO", "H", "FB", "GOOG", "LXRX", "OCGN", "CTRM", "GOLD", "AMZN", "CANG", "BNGO", "SPCE", "SRPT", "NAKD", "NFLX", "SNDL"]
#ts = gt.get_tickers_filtered(mktcap_min=1000)
ts = sorted(list(set(ts)), key=lambda x: len(x))
d = {0: "HOLD", 1: "BUY", 2: "SELL"}
tickers = [yf.Ticker(a) for a in ts]
agent = Agent(30, model_name=sys.argv[1])
agent.first_iter = False


buys = []
sells = []
for t in range(len(tickers)):
    hist = tickers[t].history(period="3mo")
    #tup = agent.act(hist, True)
    dc = {}
    for key in hist.keys():
        #print(hist[key])
        #print(key)
        dc[key] = list(hist[key])[-30:]
    hist = get_stock_data("", d=dc)
    #print(hist)
    state = get_state(hist, 30, 30)
    #print(state)
    tup = agent.act(state, True)
    mod = tf.keras.models.Model(inputs = agent.model.inputs, outputs=agent.model.inputs) #, outputs=agent.model.layers[2].output)
    o = mod.predict(state)[0]
    print(o.shape)
    #o_min, o_max = o.min(), o.max()
    #plt.imshow((o - o_min)/(o_max-o_min), cmap='gray')
    plt.imshow(o)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('conv_output.png', dpi=400)
    #print(state)
    print("ticker: {} | {} | {}".format(ts[t], d[tup[0]], tup[1]))
    if tup[0]==1:
        buys.append((ts[t], tup[1]))
    elif tup[0]==2:
        sells.append((ts[t], tup[1]))

buys.sort(key=lambda q: q[1], reverse=True)
