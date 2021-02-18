# Stock Bot

This repo contains a DQN-based stock bot that uses 1D convolution and LSTMs for Q-estimation. The goal of this was to learn about basic reinforcement learning, while pursuing a challenging (and unsolved) goal. Do not use this bot to make financial decisions!

# Features
- [x] Conv1D and LSTM Q-estimation
- [x] Complete train, test, and deploy pipeline
- [x] Model trained on a handful of stocks, see **data/test/** and **models/**
- [x] Model evaluation code that pulls from live Yahoo Finance data, see **trade.py** and **input.py**
- [x] Model convolutional filter visualization, see **vis.py** and **experiment.py**
- [ ] Model trained on large and diverse testing and training datasets

# Usage
* In progress

# Notes
* To link the code to a paper Alpaca trading account, you must add your keys to a file named keys.txt in plaintext. For accounts with real money, remember that it is not ever a good idea to store important keys without encryption!
  * Example keys.txt: `aBcDeFgHiJkL AbCdEfGhIjKl`
