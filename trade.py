import alpaca_trade_api as tradeapi
import yfinance as yf
import os
import tensorflow as tf
import numpy as np
import time
import sys
from agent import Agent
from methods import eval_model_new
from utils import get_stock_data, get_state
from get_all_tickers import get_tickers as gt
from puller import get_tickers

buy_agg = .5
sell_agg = .5
getters = gt

ts, mktcap = get_tickers()
#print(mktcap)
tdict = dict(zip(ts, [float(x) if x!='' else 0 for x in mktcap]))
#ts = gt.get_tickers_filtered(mktcap_min=1000)
print(len(ts))
ts = sorted(ts, key=lambda x: tdict[x], reverse=True)[:500]
d = {0: "HOLD", 1: "BUY", 2: "SELL"}
tickers = [yf.Ticker(a) for a in ts]
agent = Agent(30, model_name=sys.argv[1])
agent.first_iter = False


buys = []
sells = []
for t in range(len(tickers)):
    try:
        if "^" in ts[t]:
            print("ticker: {} | NOT TRADEABLE".format(ts[t]))
            continue
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
        #print(state)
        print("ticker: {} | {} | {}".format(ts[t], d[tup[0]], tup[1]))
        if tup[0]==1:
            buys.append((ts[t], tup[1]))
        elif tup[0]==2:
            sells.append((ts[t], tup[1]))
    except Exception as e:
        if e is KeyboardInterrupt: exit()
        print("ticker: {} | ERROR".format(ts[t]))

buys.sort(key=lambda q: q[1], reverse=True)
print(buys)

if len(sys.argv)>2 and sys.argv[2]=="trade":

    with open("keys.txt", "r") as keys:
        ks = keys.read().split()
        api = tradeapi.REST(ks[0], ks[1], 'https://paper-api.alpaca.markets', api_version='v2')
    for item in sells:
        acc = api.get_account()
        try:
            pos = api.get_position(item[0])
        except:
            continue
        if int(pos.qty)>0:
            api.submit_order(
                symbol=item[0],
                qty=min(int(sell_agg*float(acc.equity)*item[1]//tickers[ts.index(item[0])].info["bid"]), int(pos.qty)),
                side='sell',
                type='market',
                time_in_force='gtc'
            )

    time.sleep(5)

    for item in buys:
        acc = api.get_account()
        if int(buy_agg*float(acc.buying_power)*item[1]//tickers[ts.index(item[0])].info["ask"])>0:
            api.submit_order(
                symbol=item[0],
                qty=int(buy_agg*float(acc.buying_power)*item[1]//tickers[ts.index(item[0])].info["ask"]),
                side='buy',
                type='market',
                time_in_force='gtc'
            )

    positions = api.list_positions()
    orders = api.list_orders()
    print(positions)
    print(orders)

