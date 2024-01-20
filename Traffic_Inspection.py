import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from scipy.spatial.distance import pdist, squareform

def build_graph(df_nodes, df_edges):
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

    return city_graph

def plot_city(city_graph, k):
    # plot graph
    # adjust node size to show or not show nodes
    # """
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
    # """

def furthest_n_nodes(symmetric_matrix, n):
    flattened_upper_triangle = np.triu(symmetric_matrix).flatten()
    top_n_indices = np.argpartition(flattened_upper_triangle, -n)[-n:]
    rows, cols = np.unravel_index(top_n_indices, symmetric_matrix.shape)

    return list(zip(rows, cols))

def find_shortest_path(city_graph, node_pair, plot=False):
    # Find the shortest path
    shortest_path = nx.shortest_path(city_graph, source=node_pair[0], target=node_pair[1])

    #print(f"Shortest path from {node_pair[0]} to {node_pair[1]}: {shortest_path}")
    if plot:
        k = dict(city_graph.degree())

        # Optionally, you can visualize the graph and the shortest path
        node_colors = [k[node] for node in city_graph.nodes]
        cmap = plt.cm.viridis
        norm = plt.Normalize(min(node_colors), max(node_colors))
        node_colors = [cmap(norm(degree)) for degree in node_colors]

        fig, ax = plt.subplots(figsize=(8, 8))
        pos = {node: (city_graph.nodes[node]['x'], city_graph.nodes[node]['y']) for node in city_graph.nodes}
        nx.draw(city_graph, pos, node_size=1, node_color=node_colors)
        nx.draw_networkx_edges(city_graph, pos,
                               edgelist=[(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)],
                               edge_color='red', width=2)
        plt.show()

def main(plot_graph=True):
    nodes_filepath = "data/nodes.txt"
    edges_filepath = "data/edges.txt"

    # load data
    df_nodes = pd.read_csv(nodes_filepath, sep=' ')
    df_edges = pd.read_csv(edges_filepath, sep=' ')

    city_graph = build_graph(df_nodes, df_edges)

    if plot_graph:
        k = dict(city_graph.degree())
        print(f'In the city graph the nodes have following degrees {np.unique(list(k.values()))}')
        plot_city(city_graph, k)

    arr_XY = np.array([df_nodes['X'],df_nodes['Y']]).T
    pairwise_dist = squareform(pdist(arr_XY))

    # number of furthest nodes to find
    nr_furthest_nodes = 20
    furthest_nodes = furthest_n_nodes(pairwise_dist, nr_furthest_nodes)

    # finds shortest path through city between two nodes and plots the path
    find_shortest_path(city_graph, furthest_nodes[0], True)

    


if __name__ == "__main__":
    main(False)
