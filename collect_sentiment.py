from utils import get_stock_data_sentiment
from multiprocessing import Process
import os

ps = []
for file in os.listdir("data/train/"):
	if ".csv" in file:
		ps.append(Process(target=get_stock_data_sentiment, args=("data/train/"+file,)))

for p in ps:
	p.start()
for p in ps:
	p.join()
