import csv
import matplotlib.pyplot as plt
import ast
import numpy as np


def create_graph(filenames):
    y_values = []
    cur_min = 1
    cur_max = 0
    for name in filenames:
        file = open(name)
        file = csv.reader(file)
        values = ast.literal_eval(next(file)[0])
        y_values.append(values)
        if min(values) < cur_min:
            cur_min = min(values)
        if max(values) > cur_max:
            cur_max = max(values)
    for i, valuelist in enumerate(y_values):
        plt.clf()
        _, bins, _ = plt.hist(valuelist, bins=8, range=(cur_min, cur_max), color="tab:blue")
        bins_2 = [round(elem, 3) for elem in bins]
        plt.xticks(bins, bins_2)
        plt.ylim(0, 255)
        plt.title(filenames[i].split("_")[0] + " keskmine absoluutne viga")
        plt.savefig(filenames[i] + ".png")
        with open(filenames[i] + ".txt", "w") as f:
            f.write("MAE: " + str(round(sum(valuelist)/len(valuelist))))


create_graph(["ELU_final.csv", "Swishi_final.csv", "Sigmoidi_final.csv", "ReLU_final.csv", "Tanh funktsiooni_final.csv"])