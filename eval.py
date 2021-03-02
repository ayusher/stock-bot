import sys, os

from agent import Agent
from methods import train_model, eval_model_new
from utils import get_stock_data, get_dates_data
import matplotlib.pyplot as plt

def eval(window_size, name, safety, yolo):
    l = sorted([i for i in os.listdir("data/test/") if ".csv" in i])
    fig, axes = plt.subplots(len(l), 1)
    a = 0
    for file in l:
        ax = axes.flatten()[a]
        data = get_stock_data("data/test/"+file)
        dates = get_dates_data("data/test/"+file)
        agent = Agent(window_size, model_name=name)

        print("evaluating on {} of length {}".format(file, len(data)))
        profit, _, tup = eval_model_new(agent, data, window_size, debug=False, dates=dates, safety = safety, yolo = yolo)
        ax.plot([i[-2] for i in data], zorder=1)
        for item in tup[0]:
            #print(item)
            ax.plot(item, data[item][-2], "g", zorder=2, markersize=6, marker='o')
        for item in tup[1]:
            #print(item)
            ax.plot(item, data[item][-2], "r", zorder=2, markersize=6, marker='o')
        #fig.savefig("performance_{}.png".format(file[:-4]))
        #plt.figure().clear()
        #fig.close()
        a+=1
        print("Total profit: ${}".format(profit))
    fig.savefig("performance.png")
    return

if __name__ == "__main__":
    name = sys.argv[1]

    try: safety = float(sys.argv[2])
    except: safety = 0

    window_size = 30
    if safety == -1: yolo = True
    else: yolo = False

    try:
        eval(window_size, name, safety, yolo)
    except KeyboardInterrupt:
        print("Aborted")
