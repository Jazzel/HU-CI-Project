import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import math
import numpy as np
import random
from collections import deque

# import csv

# lst = []
# with open("data/ADANIPORTS.csv", "r") as file:
#     csv_reader = csv.reader(file, delimiter=",")
#     c = 0
#     for r in csv_reader:
#         if c ==0:
#             print(r)
#             c += 1
#             continue
#     lst.append((c, float(r[4]), float(r[8])))

# eps = []
# t = 1
# for i in range()


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = load_model(model_name) if is_eval else self._model()
        self.policy = load_model(model_name) if is_eval else self._policy()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model

    def _policy(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.policy.predict(state)
        options = options[0]
        for i in range(len(options)):
            options[i] = options[i] / sum(options)
        r = np.random.rand()
        if 0 <= r < options[0]:
            return 0
        elif options[0] <= r < options[0] + options[1]:
            return 1
        else:
            return 2

    def expReplay(self, batch_size):
        # print("-------------")
        # print("Training Start")
        # print("-------------")
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0]
                )
            target_f = self.model.predict(state)
            target_f[0][0] = target
            a = np.array([[0, 0, 0]])
            a[0][action] = 1
            # print("Action", a)
            self.model.fit(state, target_f, epochs=1, verbose=0)
            self.policy.fit(state, a, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def formatPrice(n):
    return ("-$." if n < 0 else "$.") + "{0:.2f}".format(abs(n))


def getStockDataVec(key):
    vec = []
    lines = open(key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        # print(line)
        # print(float(line.split(",")[4]))
        vec.append(float(line.split(",")[4]))
        # print(vec)
    return vec


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def getState(data, t, n):
    d = t - n + 1
    block = (
        data[d : t + 1] if d >= 0 else -d * [data[0]] + data[0 : t + 1]
    )  # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])


import sys

stock_name = "data/NVDA"
window_size = 1
episode_count = 10
stock_name = str(stock_name)
window_size = int(window_size)
episode_count = int(episode_count)
agent = Agent(window_size)
data = getStockDataVec(stock_name)

RL_data = {
    i: {"data": [], "sell": [], "buy": [], "profit": [], "total_profit": 0}
    for i in range(11)
}

l = len(data) - 1
batch_size = 128
for e in range(episode_count + 1):
    # print("Episode " + str(e) + "/" + str(episode_count))
    RL_data[e]["data"] = data
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    for t in range(l):
        print("Epsiode, Step:", e, t)
        action = agent.act(state)
        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        if action == 1:  # buy
            agent.inventory.append(data[t])
            RL_data[e]["buy"].append(t)
            # print("Buy: " + formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0:  # sell
            bought_price = window_size_price = agent.inventory.pop(
                0
            )  # Selling stock  # [44.3]
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            RL_data[e]["sell"].append(t)
            # print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        RL_data[e]["profit"].append(total_profit)
        if done:
            RL_data[e]["total_profit"] = total_profit
            print("-------------    -------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
        if len(agent.memory) % batch_size == 0 and len(agent.memory) > 0:
            agent.expReplay(batch_size)
    if e % 10 == 0:
        agent.model.save(str(e) + ".keras")


import json

file = "actorcritic_data.json"
with open(file, "w") as f:
    json.dump(RL_data, f)
