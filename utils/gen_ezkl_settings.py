import numpy as np
import json


paths = ['../data/1.csv', '../data/2.csv', '../data/3.csv']



for path in paths:
    data = np.genfromtxt(path, delimiter=',')
    data = data[1:]
    data = data[:, 1:]
    data = data.T
    data = data.tolist()
    with open(path[:-4] + '.json', 'w') as f:
        json.dump(data, f)
