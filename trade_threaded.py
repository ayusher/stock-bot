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
import threading

buy_agg = .5
sell_agg = .5
getters = gt

ts, mktcap = get_tickers()
#ts = gt.get_tickers_filtered(mktcap_min=1000)
tdict = dict(zip(ts, [float(x) if x!='' else 0 for x in mktcap]))
print(len(ts))
ts = sorted(ts, key=lambda x: tdict[x], reverse=True)
d = {0: "HOLD", 1: "BUY", 2: "SELL"}
tickers = [yf.Ticker(a) for a in ts]
agent = Agent(30, model_name=sys.argv[1])
agent.first_iter = False


buys = []
sells = []
states, names = [-1 for _ in range(len(tickers))], [-1 for _ in range(len(tickers))]
def get_hist(part):
	for t in range(part[0], part[1]):
		try:
			if "^" in ts[t]:
				#print("ticker: {} | NOT TRADEABLE".format(ts[t]))
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
			#if t%50==0: print("{}/{}".format(t, len(tickers)), end='\r')
			states[t] = get_state(hist, 30, 30)
			names[t] = ts[t]
			#print(state)
		except: pass


threads = []
tnum = 8
for part in range(tnum):
	if part==tnum-1: threads.append(threading.Thread(target=get_hist, args=((part*len(tickers)//tnum, len(tickers)),)))
	else: threads.append(threading.Thread(target=get_hist, args=((part*len(tickers)//tnum, (part+1)*len(tickers)//tnum),)))
#names = [i for i in names if type(i)!=int]
#states = [i for i in states if type(i)!=int]

for thread in threads: thread.start()
for thread in threads: thread.join()

names = [i for i in names if type(i)!=int]
states = [i for i in states if type(i)!=int]

print(np.array(states).shape)

tups = agent.act_bulk(states, True)
for t in range(len(tups[0])):
	#print(state)
	tup = (tups[0][t], tups[1][t])
	print("ticker: {} | {} | {}".format(names[t], d[tup[0]], tup[1]))
	#print((names[t], tup[1]))
	if tup[0]==1:
		buys.append((names[t], tup[1]))
	elif tup[0]==2:
		sells.append((names[t], tup[1]))


buys.sort(key=lambda q: q[1], reverse=True)
buys = buys[:5]
sumbuys = sum([i[1] for i in buys])
print(buys)
if len(sys.argv)>2 and sys.argv[2]=="trade":
	with open("keys.txt", "r") as keys:
		ks = keys.read().split()
		api = tradeapi.REST(ks[0], ks[1], 'https://paper-api.alpaca.markets', api_version='v2')
	clock = api.get_clock()
	if not clock.is_open: exit()
	poss = [position.symbol.upper() for position in api.list_positions()]
	for item in sells:
		if item[0].upper() not in poss: continue
		#acc = api.get_account()
		pos = api.get_position(item[0])
		acc = api.get_account()
		if int(pos.qty)>0:
			api.submit_order(
			symbol=item[0],
			qty=pos.qty,
			side='sell',
			type='market',
			time_in_force='day'
			)

	time.sleep(5)

	acc = api.get_account()
	obuy = acc.buying_power
	for item in buys:
		acc = api.get_account()
		#print(acc.buying_power)
		#print(item[1]/sumbuys)
		#print((tickers[ts.index(item[0])].info["ask"]))
		#print((float(item[1]/sumbuys)*float(acc.buying_power))/(tickers[ts.index(item[0])].info["ask"]))
		try:
			if (float(item[1]/sumbuys)*float(obuy))/(tickers[ts.index(item[0])].info["ask"])>0:
				api.submit_order(
				  symbol=item[0],
				  qty=int((float(item[1]/sumbuys)*float(obuy))/(tickers[ts.index(item[0])].info["ask"])),
				  side='buy',
				  type='market',
				  time_in_force='day'
				)
		except: pass

	positions = api.list_positions()
	orders = api.list_orders()
	print(positions)
	print(orders)

