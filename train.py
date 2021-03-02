import sys, os

from agent import Agent
from methods import train_model, eval_model_new
from utils import get_stock_data, get_state

def train(val_stock, window_size, batch_size, ep_count, name):
	trainer = os.listdir("data/train/")
	files = sorted([i for i in trainer if ".csv" in i])
	agent = Agent(window_size, model_name=name)
	val_data = get_stock_data(val_stock)
	epic = 1
	for _ in range(50):
		for filename in files:
			print("training on {}".format(filename))
			train_data = get_stock_data("data/train/"+filename)
			states = [get_state(train_data, x, window_size) for x in range(len(train_data))]

			for episode in range(1, ep_count + 1):
				train_model(agent, epic, train_data, ep_count, batch_size, window_size, states)
				epic+=1
		val_result, _, tup = eval_model_new(agent, val_data, window_size, debug=False)
		print("total profit: ${}".format(val_result))

if __name__ == "__main__":
    val_stock = sys.argv[1]
    try:
      name = sys.argv[2]
    except:
      name = None
    window_size = 30
    batch_size = 2048
    ep_count = 5

    try:
        train(val_stock, window_size, batch_size, ep_count, name)
    except KeyboardInterrupt:
        print("Aborted!")
