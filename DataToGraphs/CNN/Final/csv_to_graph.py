import csv
import matplotlib.pyplot as plt
import ast
import numpy as np

# def create_graph(filename):
#     file = open(filename)
#     file = csv.reader(file)
#     y = ast.literal_eval(next(file)[0])
#     print(y)
#     _, bins, _ = plt.hist(y, bins=3)
#     plt.xticks(((bins[0] + bins[1])/2, (bins[1]+bins[2])/2, (bins[2]+bins[3])/2), (str(bins[0])[0:6] + "..." + str(bins[1])[0:6], str(bins[1])[0:6] + "..." + str(bins[2])[0:6], str(bins[2])[0:6] + "..." + str(bins[3])))
#     plt.savefig(filename + ".png")

# def create_graph(filename):
#     file = open(filename)
#     file = csv.reader(file)
#     y = ast.literal_eval(next(file)[0])
#     _, bins, _ = plt.hist(y, (0.982, 0.984, 0.986, 0.988, 0.99, 0.992, 0.994, 0.996, 0.998, 1))
#     plt.xticks((0.982, 0.984, 0.986, 0.988, 0.99, 0.992, 0.994, 0.996, 0.998, 1))
#     plt.savefig(filename + ".png")

def create_graph(filenames):
    y_values = []
    length = []
    cur_min = 1
    cur_max = 0
    for name in filenames:
        file = open(name)
        file = csv.reader(file)
        line = next(file)
        values = ast.literal_eval(line[0])
        y_values.append(values)
        length.append(int(line[1]))
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
        plt.title(filenames[i].split("_")[0] + " kõik tulemused")
        plt.savefig(filenames[i] + ".png")
        with open(filenames[i] + ".txt", "w") as f:
            f.write("Keskmine täpsus: " + str(round(sum(valuelist)/len(valuelist))) + "\nKeskmine treeningupikkus: " + str(round(length[i]/len(valuelist))))


create_graph(["ELU_final.csv", "Swishi_final.csv", "Sigmoidi_final.csv", "ReLU_final.csv", "Tanh funktsiooni_final.csv"])