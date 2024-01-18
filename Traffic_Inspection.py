import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


nodes_filepath = "data/nodes.txt"
edges_filepath = "data/edges.txt"

# load data
df_nodes = pd.read_csv(nodes_filepath, sep=' ')
df_edges = pd.read_csv(edges_filepath, sep=' ')

city_graph = nx.Graph()

nodes = {}
for i, line in df_nodes.iterrows():
    id = int(line['Node_ID'])
    x = line['X']
    y = line['Y']
    nodes[id] = (x, y)
    city_graph.add_node(id, x=x, y=y)

edges = {}
for i, line in df_edges.iterrows():
    id = int(line['Edge_ID'])
    start = int(line['Start_ID'])
    end = int(line['End_ID'])
    length = line['L2']
    if start in nodes.keys() and end in nodes.keys():
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        edges[(start, end)] = (x1, y1, x2, y2, length)
        city_graph.add_edge(start, end, length=length)


k = dict(city_graph.degree())

print(f'In the city graph the nodes have following degrees {np.unique(list(k.values()))}')

# plot graph
# adjust node size to show or not show nodes
#"""
node_colors = [k[node] for node in city_graph.nodes]
cmap = plt.cm.viridis
norm = plt.Normalize(min(node_colors), max(node_colors))
node_colors = [cmap(norm(degree)) for degree in node_colors]

fig, ax = plt.subplots(figsize=(8, 8))
pos = {node: (city_graph.nodes[node]['x'], city_graph.nodes[node]['y']) for node in city_graph.nodes}
nx.draw(city_graph, pos, node_size=5, node_color=node_colors)

# add legend
sm = ScalarMappable(cmap=cmap, norm=norm)
cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])  # Adjust these values to position the colorbar
cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', label='Node Degree')

ax.set_title("City of Oldenburg")
plt.savefig('plots/city.png')
plt.show()


fig, ax = plt.subplots(figsize=(8, 8))
pos = {node: (city_graph.nodes[node]['x'], city_graph.nodes[node]['y']) for node in city_graph.nodes}
nx.draw(city_graph, pos, node_size=0, node_color=node_colors)

ax.set_title("City of Oldenburg")
plt.savefig('plots/city_egdes.png')
plt.show()
#"""