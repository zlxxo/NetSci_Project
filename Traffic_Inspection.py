import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nodes_filepath = "data/nodes.txt"
edges_filepath = "data/edges.txt"

# load data
df_nodes = pd.read_csv(nodes_filepath, sep=' ')
df_edges = pd.read_csv(edges_filepath, sep=' ')

nodes = {}
for i, line in df_nodes.iterrows():
    id = int(line['Node_ID'])
    x = line['X']
    y = line['Y']
    nodes[id] = (x, y)

edges = {}
for i, line in df_edges.iterrows():
    id = int(line['Edge_ID'])
    start = int(line['Start_ID'])
    end = int(line['End_ID'])
    if start in nodes.keys() and end in nodes.keys():
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        edges[(start, end)] = (x1, y1, x2, y2, line['L2'])

plt.figure(figsize=(8, 8))
#for node, (x, y) in nodes.items():
#    plt.scatter(x, y, color='blue', s=20)

for (s, e), (x1, y1, x2, y2, l2) in edges.items():
    plt.plot([x1, x2], [y1, y2], 'k-', lw=2)

plt.title("City of Oldenburg")
plt.savefig('plots/city.png')
plt.show()