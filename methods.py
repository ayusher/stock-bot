import os
import math

import numpy as np
from collections import deque

from utils import get_state
import time

def train_model(agent, episode, data, ep_count, batch_size, window_size, states):
    total_profit = 0
    data_length = len(data) - 1
    agent.inventory = []
    avg_loss = []
    state = states[0]
    prices = [y[-2] for y in data]
    #print("length of data is".format(len(prices)))
    print(len(data))
    start = time.time()
    actions = agent.act_bulk(states)
    counter = 0
    for t in range(data_length):
        reward = 0
        next_state = states[t+1]
        action = actions[t]
        #print(action)
        # BUY
        if action == 1:
            agent.inventory.append(prices[t])
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            #rew = sum(agent.inventory)/len(agent.inventory)
            bought_price = agent.inventory.pop(0)
            delta = (prices[t] - bought_price)/bought_price
            #delta2 = (prices[t] - rew)/rew
            #reward = delta**3 TODO: change this where applicable
            if delta<0: reward = 2*math.tan(delta)
            else: reward = math.tanh(delta/2)
            #if abs(reward)>1: print(reward, delta)
            total_profit += delta
            counter += 1
        # HOLD
        else:
            pass

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        state = next_state
    print("time taken replaying {}".format(time.time()-start))
    if counter==0: print("made no trades")
    else: print("average realized profit {:.2f}% on {} trades".format(total_profit/counter*100, counter))
    #print("total average profit {:.2f}%".format((total_profit+len(agent.inventory)*(prices[-1]-sum(agent.inventory)/len(agent.inventory))/(sum(agent.inventory)/len(agent.inventory)) ))
    if counter>0 and len(agent.inventory)>0: print("total average profit {:.2f}%".format( 100*(total_profit+len(agent.inventory)*(prices[-1]-sum(agent.inventory)/len(agent.inventory))/(sum(agent.inventory)/len(agent.inventory)))/(counter+len(agent.inventory)) ))
    if len(agent.memory) > batch_size:
        loss = agent.train_experience_replay(batch_size)
        avg_loss.append(loss)

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))

def eval_model_new(agent, data, window_size, debug, dates = None, safety = 0, yolo = False, SL = False):
    bank = 100000
    init_bank = bank
    data_length = len(data) - 1
    agg_b = .5
    agg_s = .5

    history = []
    agent.inventory = []

    state = get_state(data, 0, window_size)
    bd, sd = [], []
    prices = [y[-2] for y in data]
    for t in range(data_length):
        #print("{}/{}".format(t+1, data_length+1), end="\r")
        next_state = get_state(data, t + 1, window_size)

        # select an action
        action, prob = agent.act(state, is_eval=True)
        #print(prob)
        # BUY
        if action == 1 and bank>0 and prob>safety:
            bd.append(t)
            try: numshares = int(bank//prices[t]*agg_b*prob)
            except:
                print(prices[t])
                exit()
            if yolo: numshares = int(bank//prices[t])
            for _ in range(numshares):
                bank -= prices[t]
                agent.inventory.append(prices[t])
                history.append((prices[t], "BUY"))
            if debug and numshares>0:
                if dates!=None:
                    print(prices[t])
                #print(prob)
                print("Buy {} shares at: {}".format(numshares, (prices[t])))

        # SELL
        elif action == 2 and len(agent.inventory) > 0 or (SL and len(agent.inventory)>0 and sum(agent.inventory)/len(agent.inventory)*.85 > prices[t]):
            print("{}/{}".format(t+1, data_length+1), end="\r")
            sd.append(t)
            numshares = min(int(init_bank//prices[t]*agg_s*prob), len(agent.inventory))
            if yolo: numshares = len(agent.inventory)
            for _ in range(numshares):
                bought_price = agent.inventory.pop(0)
                bank += prices[t]
                history.append((prices[t], "SELL"))
            if debug and numshares>0:
                if dates!=None:
                    print(dates[t])
                print("Sell {} shares at: {} | Position: {}".format(
                    numshares, (prices[t]), (prices[t] - bought_price)))
        # HOLD
        else:
            history.append((prices[t], "HOLD"))

        done = (t == data_length - 1)

        state = next_state
        if done:
            return bank+len(agent.inventory)*prices[t]-init_bank, history, (bd, sd)

