import random
import math

from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import Dense, GaussianNoise, InputLayer, Dropout, Add, Input, LeakyReLU, GRU, SimpleRNN, Embedding, Flatten, Conv1D, Reshape, LSTM, Activation, MaxPooling1D, BatchNormalization, AveragePooling1D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import Huber
#tf.get_logger().setLevel('WARNING')
#"ops"
def normalize(data):
    #print(data)
    temp = data-min(data)
    return temp/sum(temp)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    #print(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Agent:

    def __init__(self, state_size, reset_every=10, model_name=None):
        #physical_devices = tf.config.list_physical_devices('GPU')
        #tf.config.experimental.set_memory_growth(physical_devices[0], True)

        self.state_size = state_size
        self.action_size = 3
        self.inventory = []
        self.memory = deque(maxlen=1000000)
        self.first_iter = True

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 1e-4
        self.loss = "mse"

        self.optimizer = Adam(learning_rate=self.learning_rate)

        if model_name==None:
            self.model = self.create_timeseries_model()
            self.model_name = "model"
        else:
            self.model_name = model_name
            self.model = self.load()

        self.n_iter = 0
        self.reset_every = reset_every
        self.am = np.vectorize(np.argmax)
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def create_timeseries_model(self):
        model = Sequential([
            InputLayer(input_shape=(self.state_size, 5,)),
            #Activation('tanh'),

            Conv1D(128, 3, padding="same"),
            LeakyReLU(alpha=0.2),

            Conv1D(128, 3, padding="same"),
            LeakyReLU(alpha=0.2),

            Flatten(),
            Dense(1024, activation="tanh"),
            Dense(self.action_size, activation="linear")
        ])


        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.first_iter:
            self.first_iter = False
            if not is_eval: return 1
            #return 1

        #with tf.device('/cpu:0'):
        action_probs = self.model(state, training=False).numpy()
        if is_eval:
            return np.argmax(action_probs[0]), max(softmax(action_probs[0]))
        return np.argmax(action_probs[0])

    def act_bulk(self, states, is_eval=False):
        action_probs = self.model.predict(np.reshape(np.array(states), (-1, 30, 5)))
        if is_eval:
            return np.argmax(action_probs, axis=1), [max(softmax(action_probs[x])) for x in range(len(action_probs))]
        else:
            outs = np.argmax(action_probs, axis=1)
            #print(outs)
            return [i if random.random()>self.epsilon else random.randrange(self.action_size) for i in outs]

    def train_experience_replay(self, batch_size):
        print("memory size", len(self.memory))
        if random.random()<.1:
            print("error-based")
            #mini_batch = random.sample(self.memory, batch_size)
            #dst = lambda x, y: return sum([(x[i]-y[i])**2 for i in range(len(x))])**.5
            pred = self.model(np.array([i[0][0] for i in self.memory]), training=False).numpy()
            acs = [np.argmax(x) for x in pred]
            acts = [i[1] for i in self.memory]
            #print(acs)
            mini = [self.memory[j] for j in range(len(self.memory)) if acts[j]!=acs[j]]
            if len(mini)>=batch_size: mini_batch = random.sample(mini, batch_size)
            else: mini_batch = mini+random.sample(self.memory, batch_size-len(mini))
            #mini_batch = random.choices(self.memory, batch_size, weights)
        else:
            print("random")
            mini_batch = random.sample(self.memory, batch_size)
            #print(max([i[2] for i in mini_batch], key=lambda x: abs(x)))
        X_train, y_train = [], []

        self.n_iter += 1

        if self.n_iter % self.reset_every == 0:
            print("move target")
            self.n_iter = 0
            self.target_model.set_weights(self.model.get_weights())

        q_values = self.model(np.array([g[0][0] for g in mini_batch]), training=False).numpy()
        targs = self.target_model(np.array([g[3][0] for g in mini_batch]), training=False).numpy()
        #print(len(q_values))
        ind = 0
        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(targs[ind])
            #print(target)
            #print(ind, target, action)
            q_values[ind][action] = target
            #print(state)
            X_train.append(state[0])
            y_train.append(q_values[ind])
            ind+=1

        print(self.epsilon)
        #print("END REPLAY")
        #print(y_train)
        print(np.array(X_train).shape)
        #for i in y_train:
        #    if any([math.isnan(h) for h in i]): exit()
        #loss = -1
        #while loss==-1 or loss>1:
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1
        ).history["loss"][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        print("saving model to models/{}_{}".format(self.model_name, episode))
        self.model.save("models/{}_{}".format(self.model_name, episode))

    def load(self):
        return load_model("models/" + self.model_name)
