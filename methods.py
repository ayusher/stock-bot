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
    buylocs = []
    for t in range(data_length):
        reward = 0
        next_state = states[t+1]
        action = actions[t]
        #print(action)
        if len(agent.inventory)==0 and action==2: action = 0
        elif len(agent.inventory)>0 and t == data_length-1: action = 2
        # BUY
        if action == 1:
            buylocs.append(t)
            agent.inventory.append((prices[t], t))
        # SELL
        elif action == 2:
            #rew = sum(agent.inventory)/len(agent.inventory)
            bought_price = agent.inventory.pop(0)
            delta = (prices[t] - bought_price[0])/bought_price[0]
            #delta2 = (prices[t] - rew)/rew
            #reward = delta**3 TODO: change this where applicable
            if delta<0: reward = math.tan(delta)
            else: reward = math.tanh(delta)
            #if reward>0: reward = reward/(t-bought_price[1])
            #if abs(reward)>1: print(reward, delta)
            total_profit += delta
            counter += 1
        # HOLD
        else:
            pass

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        state = next_state

    '''
    for loc in buylocs:
        newmin = prices[loc]
        for i in range(loc+1, data_length):
            if prices[i]>prices[loc]: break
            newmin = prices[i]
        newmax = prices[loc]
        for i in range(loc+1, data_length):
            if prices[i]<prices[loc]: break
            newmax = prices[i]
        if newmin != prices[loc]:
            cop = list(agent.memory[-(data_length-loc)])
            cop[2] = -abs(prices[loc]-newmin)/prices[loc]
            #print(cop[2])
            cop[2] = math.tan(cop[2])
            agent.memory[-(data_length-loc)] = tuple(cop)
        else:
            cop = list(agent.memory[-(data_length-loc)])
            cop[2] = (newmax-prices[loc])/prices[loc]
            #print(cop[2])
            cop[2] = math.tanh(cop[2])
            agent.memory[-(data_length-loc)] = tuple(cop)
    '''

    print("time taken replaying {}".format(time.time()-start))
    if counter==0: print("made no trades")
    else: print("average realized profit {:.2f}% on {} trades".format(total_profit/counter*100, counter))
    #print("total average profit {:.2f}%".format((total_profit+len(agent.inventory)*(prices[-1]-sum(agent.inventory)/len(agent.inventory))/(sum(agent.inventory)/len(agent.inventory)) ))
    s = sum([i[0] for i in agent.inventory])
    if counter>0 and len(agent.inventory)>0: print("total average profit {:.2f}%".format( 100*(total_profit+len(agent.inventory)*(prices[-1]-s/len(agent.inventory))/(s/len(agent.inventory)))/(counter+len(agent.inventory)) ))
    if len(agent.memory) > batch_size:
        loss = agent.train_experience_replay(batch_size)
        avg_loss.append(loss)

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))

def eval_model_new(agent, data, window_size, debug, dates = None, safety = 0, yolo = False, SL = True):
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

