# Stock Bot

This repo contains a DQN-based stock bot that uses 1D convolution and LSTMs for Q-estimation. The goal of this was to learn about basic reinforcement learning, while pursuing a challenging (and unsolved) goal. Do not use this bot to make financial decisions!

# Features
- [x] Conv1D and LSTM Q-estimation
- [x] Complete train, test, and deploy pipeline
- [x] Model trained on a handful of stocks, see **data/test/** and **models/**
- [x] Model evaluation code that pulls from live Yahoo Finance data, see **trade.py** and **input.py**
- [x] Model convolutional filter visualization, see **vis.py** and **experiment.py**
- [ ] Model trained on large and diverse testing and training datasets
- [ ] Google News sentiment analysis as a model input (initial results have been iffy)

# Setup

Run `pip3 install -r requirements.txt` to install all of the required dependencies.

# Usage

Models are stored in **models/** and data is stored in subfolders in **data/test/**.

## Training

Start training the model from scratch by running `python3 train.py data/test/XERS.csv`, where the argument can be replaced by the path to any CSV file to be used for validation. This will train the model on all stock data found in **data/test/**.

To resume training from a pretrained model, run `python3 train.py data/test/XERS.csv model_2300`, where the last argument is the name of the model as stored in the **models/** folder.

## Evaluation

Evaluate a model by running `python3 eval.py model_2300`. This will test the model against all stock data found in **data/test/**.

Evaluate the model "YOLO" style (risking the entire portfolio on each trade) by running `python3 eval.py model_2300 -1`.

Evaluate the model with a minimum buy confidence of .5 by running `python3 eval.py model_2300 .5`, where the last argument can be changed to any buy confidence between 0 and 1.

## Deployment

To make trades on an Alpaca ccount using this bot, add your keys to keys.txt in the format "(API key) (secret key)". Example keys.txt: `aBcDeFgHiJkL AbCdEfGhIjKl`.

Then, run `python3 trade.py model_2300` to see the trades that you expect the bot to make. To actually execute trades, include a final argument "trade" that acts as verification to prevent accidental trading. The complete command would be `python3 trade.py model_2300 trade`. 
* See the first bullet point in the Notes section below for a warning about storing keys in plaintext. 

## Visualization

Visualize the model's convolutional filters by running `python3 vis.py model_2300`. The resulting image will be stored in **vis.png**.

<img src="https://github.com/ayusher/stock-bot/blob/main/vis.png" width="100%">

Visualize the output of the convolutional layer by running `python3 experiment.py model_2300`. The resulting image will be stored in **conv_output.png**.

<img src="https://github.com/ayusher/stock-bot/blob/main/conv_output.png" width="100%">

# Notes
* To link the code to a paper Alpaca trading account, you must add your keys to a file named keys.txt in plaintext. For accounts with real money, remember that it is not ever a good idea to store important keys without encryption!
* Once again, this bot is not intended to provide financial advice, please invest wisely.
