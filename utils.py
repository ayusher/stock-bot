import os
import math
import logging
import time
import pickle

import pandas as pd
import numpy as np
import datetime

from pygooglenews import GoogleNews

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return t

def get_state(data, t, n_days):
    if t<n_days:
        block = [data[0]]*(n_days-t)+data[:t]
    else:
        block = data[t-n_days:t]

    res = [[0 for _ in range(len(data[0]))]]
    #print(block[0])
    for i in range(n_days-1):
        #print([10*(block[i+1][x] - block[i][x])/block[i][x] for x in range(len(block[0]))])
        #time.sleep(1)
        res.append([10*(block[i+1][x] - block[i][x])/block[0][x] for x in range(len(block[0]))])
    #print(res)

    return np.array([res])

def get_stock_data_sentiment(stock_file, e=False):
    df = pd.read_csv(stock_file)
    #stockdat = list(df['Close'])
    datedat = list(df['Date'])
    #nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    output = []
    name = stock_file.split("/")[2][:-4]
    gn = GoogleNews(lang = 'en')
    lastdate = datetime.datetime.strptime(datedat[-1], "%Y-%m-%d")
    lastdate = lastdate+datetime.timedelta(days=1)
    datedat.append(lastdate.strftime("%Y-%m-%d"))
    for date in range(len(datedat)-1):
        search = gn.search("{} stock".format(name), from_=datedat[date], to_=datedat[date+1])
        #time.sleep(.001)
        search = [i["title"] for i in search["entries"]]
        d = sid.polarity_scores(". ".join(search))
        print("{} stock".format(name), int(100*date/(len(datedat)-1)))
        output.append([d["neu"], d["pos"], d["neg"]])
    with open(stock_file[:-4]+'.pkl', 'wb') as f:
        pickle.dump(output, f)
    return output

def get_stock_data(stock_file, e=False, d = None):
    if d is not None: df = d
    else: df = pd.read_csv(stock_file)
    out = np.dstack((np.array(df["Open"]), np.array(df["High"]), np.array(df["Low"]), np.array(df["Close"]), np.array(df["Volume"]))).tolist()[0]
    latest_nonzero = out[0]
    for i in range(len(out)):
        #if 0 in latest_nonzero: print(latest_nonzero)
        #if 0 in out[i]: print(latest_nonzero)
        for feature in range(len(out[i])):
            if out[i][feature] == 0 or math.isnan(out[i][feature]):
                #print(latest_nonzero[0]==0)
                #print(latest_nonzero[feature])
                if feature == 0:
                    try: out[i][feature]=out[i-1][3]
                    except: out[i][feature]=out[i][3]
                    #print(out[i][feature], out[i][3])
                else: out[i][feature] = latest_nonzero[feature]
                #print(out[i])
                #print(i, out[i], latest_nonzero[feature])
            else:
                latest_nonzero[feature] = out[i][feature]
    return out

def get_stock_price(stock_file, e=False):
    df = pd.read_csv(stock_file)
    return list(df['Close'])

def get_stock_volume(stock_file, e=False):
    df = pd.read_csv(stock_file)
    return list(df['Volume'])

def get_dates_data(stock_file):
    df = pd.read_csv(stock_file)
    return list(df['Date'])
