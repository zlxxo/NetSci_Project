import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from scipy.spatial.distance import pdist, squareform
from Metrics import node_betweenness, edge_betweenness, closeness, avg_shortest_path
from Analysis import load_metrics, calculate_resilience
import json
import heapq

np.random.seed(1234)

nr_furthest_nodes = 1000
n_edges_to_highway = 200 # top n busiest roads to be found


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
            city_graph.add_edge(start, end, length=length, weight=1) # add default weight

    return city_graph

def plot_city(city_graph, k):
    # plot graph
    # adjust node size to show or not show nodes
    # """
    node_colors = [k[node] for node in city_graph.nodes]
    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(min(node_colors), max(node_colors))
    node_colors = [cmap(norm(degree)) for degree in node_colors]

    edge_colors = [city_graph[u][v]['length'] for u, v in city_graph.edges()]
    cmap2 = plt.cm.YlOrRd
    norm2 = plt.Normalize(min(edge_colors), max(edge_colors))
    edge_colors = [cmap2(norm2(length)) for length in edge_colors]

    # draw nodes and their degree
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = {node: (city_graph.nodes[node]['x'], city_graph.nodes[node]['y']) for node in city_graph.nodes}
    nx.draw_networkx_nodes(city_graph, pos, node_size=5, node_color=node_colors)

    # add legend
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])  # Adjust these values to position the colorbar
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', label='Node Degree')

    ax.set_title("City of Oldenburg")
    plt.savefig('plots/city-nodes.png')
    plt.show()

    # draw edges color coded with respect to lenght
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(city_graph, pos, edge_color=edge_colors)

    # add legend
    sm = ScalarMappable(cmap=cmap2, norm=norm2)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])  # Adjust these values to position the colorbar
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', label='Edge length')

    ax.set_title("City of Oldenburg")
    plt.savefig('plots/city_egdes.png')
    plt.show()

    # draw edges balck
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(city_graph, pos)

    ax.set_title("City of Oldenburg")
    plt.savefig('plots/city.png')
    plt.show()
    # """

def high_value_indices(symmetric_matrix, top_n=100): #threshold=100):
    # Ensure the matrix is symmetric
    assert (symmetric_matrix == symmetric_matrix.T).all(), "Matrix must be symmetric"

    # Find indices where values are higher than the threshold
    #high_value_indices = np.where(np.triu(symmetric_matrix) > threshold)

    # Find highest n indices
    top_indices = np.argpartition(symmetric_matrix.flatten(), -top_n)[-top_n:]

    # Convert the flattened indices back to 2D indices
    top_indices = np.unravel_index(top_indices, symmetric_matrix.shape)

    # Convert indices to row and column indices
    rows, cols = top_indices

    return list(zip(rows, cols))


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

def calculate_edge_heatmap(paths, city_graph, weight='weight'):
    num_nodes = len(city_graph.nodes)
    # Initialize a matrix to store the number of paths passing through each edge
    heatmap_matrix = np.zeros((num_nodes, num_nodes))

    for path in paths:
        for i in range(len(path) - 1):
            heatmap_matrix[path[i], path[i + 1]] += 1
            heatmap_matrix[path[i + 1], path[i]] += 1

    # Find indices where values are non-zero
    rows, cols = np.nonzero(heatmap_matrix)

    # divide heatmap values through weight to dampen colors with streets resistance
    """
    for node1, node2 in zip(rows, cols):
        if city_graph.has_edge(node1, node2):
            edge_data = city_graph[node1][node2]
            heatmap_matrix[node1][node2] = heatmap_matrix[node1][node2] / edge_data[weight]
    #"""

    return heatmap_matrix

def plot_paths_as_heatmap(graph, heatmap_matrix, title=None, weight='weight'):
    fig, ax = plt.subplots(figsize=(8, 8))

    pos = {node: (graph.nodes[node]['x'], graph.nodes[node]['y']) for node in graph.nodes}

    # Draw the graph
    #nx.draw_networkx_nodes(graph, pos, node_size=2)

    # Normalize the heatmap matrix for colormap intensity
    normalized_heatmap = heatmap_matrix / np.max(heatmap_matrix)

    # Draw edges with colormap intensity
    edges = graph.edges()
    colors = [normalized_heatmap[u][v] for u, v in edges]
    nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=colors, edge_cmap=plt.cm.YlOrRd, width=2)

    if title:
        #ax.set_title(title)
        plt.savefig(f'plots/{title.lower().replace(" ", "_") + "_" + weight + "_" + str(nr_furthest_nodes) + "_" + str(n_edges_to_highway)}.png')
    plt.show()

def generate_n_agent(furthest_node_pairs, nr_of_agents):
    indices = np.arange(len(furthest_node_pairs))
    # sample uniformly
    sampled_indices = np.random.choice(indices, size=nr_of_agents, replace=True)

    # Choose pairs based on sampled indices
    sampled_pairs = np.array(furthest_node_pairs)[sampled_indices]

    return sampled_pairs

def find_n_shortest_paths(city_graph, node_pairs, plot=False, title=None, weight='weight'):
    all_paths = []

    #"""
    # Find and store the shortest paths
    for source, target in node_pairs:
        shortest_path = nx.shortest_path(city_graph, source=source, target=target, weight=weight) # djjikstra because we only have positive weights
        all_paths.append(shortest_path)
   #"""


    # Calculate the heatmap matrix
    heatmap_matrix = calculate_edge_heatmap(all_paths, city_graph)

    # Plot the graph with a heatmap-like effect
    if plot:
        plot_paths_as_heatmap(city_graph, heatmap_matrix, title=title, weight=weight)
    return heatmap_matrix


def generate_graph_highways(graph, heatmap_edges, weight='weight'):
    #hw_weight = 4 # default highway weight


    city_graph = graph.copy()

    nodes_busiest_streets = high_value_indices(heatmap_edges, n_edges_to_highway)

    for node_tuple in nodes_busiest_streets:
        # Check if the edge exists
        if city_graph.has_edge(node_tuple[0], node_tuple[1]):
            # Retrieve the existing edge data
            edge_data = city_graph[node_tuple[0]][node_tuple[1]]

            # Add or update the 'weight' attribute
            #edge_data[weight] = hw_weight  # Replace 5 with the desired weight value
            edge_data['weight'] /= 4.
            #edge_data['length'] /= 4.

    return city_graph

def main(plot_graph=True):
    nodes_filepath = "data/nodes.txt"
    edges_filepath = "data/edges.txt"
    metrics_filepath = "results/metrics.json"

    # load data
    df_nodes = pd.read_csv(nodes_filepath, sep=' ')
    df_edges = pd.read_csv(edges_filepath, sep=' ')
    metrics_data = load_metrics(metrics_filepath)

    city_graph = build_graph(df_nodes, df_edges)



    if plot_graph:
        k = dict(city_graph.degree())
        print(f'In the city graph the nodes have following degrees {np.unique(list(k.values()))}')
        plot_city(city_graph, k)

        print('Calculating average path 1')
        avg_path = avg_shortest_path(city_graph, weight_attr_name='weight')
        metrics_data['orig_avg_path_w'] = avg_path

        print('Calculating average path 2')
        avg_path = avg_shortest_path(city_graph, weight_attr_name='length')
        metrics_data['orig_avg_path_l'] = avg_path

        print('Calculating resilience')
        resilience = calculate_resilience(city_graph)
        metrics_data[f'orig_resilience'] = resilience

    arr_XY = np.array([df_nodes['X'],df_nodes['Y']]).T
    pairwise_dist = squareform(pdist(arr_XY))

    # number of furthest nodes to find
    furthest_nodes = furthest_n_nodes(pairwise_dist, nr_furthest_nodes)

    # generate a number of agents, sampled from furthest node pairs
    #nr_of_agents = 5000
    #list_agents = generate_n_agent(furthest_nodes, nr_of_agents)

    for weight in ["weight", "length"]:

        print(f"Finding highways by {weight}")

        # find n shortest paths through city between two nodes and plot the paths
        heatmap_edges = find_n_shortest_paths(city_graph, furthest_nodes, weight=weight, plot=True, title='Shortest paths2')

        # use heatmap_edges to choose which streets to make highways
        # simulate highways by adding weights to edges (low weight -> highway, higher weight -> street)
        city_graph_hw = generate_graph_highways(city_graph, heatmap_edges, weight=weight)
        heatmap_edges = find_n_shortest_paths(city_graph_hw, furthest_nodes, weight="weight", plot=True, title=f'Highways2 {weight}')

        print(f'Calculating average path by weight')
        avg_path = avg_shortest_path(city_graph_hw, weight_attr_name="weight")
        metrics_data[f'avg_path_{nr_furthest_nodes}_{n_edges_to_highway}_{weight}_w2'] = avg_path

        print(f'Calculating average path by length')
        avg_path = avg_shortest_path(city_graph_hw, weight_attr_name="length")
        metrics_data[f'avg_path_{nr_furthest_nodes}_{n_edges_to_highway}_{weight}_l2'] = avg_path

        print('Calculating resilience')
        resilience = calculate_resilience(city_graph_hw)
        metrics_data[f'resilience_{nr_furthest_nodes}_{n_edges_to_highway}_{weight}2'] = resilience

        print()



    with open(metrics_filepath, "w") as json_file:
        json.dump(metrics_data, json_file, indent=2)


if __name__ == "__main__":
    main(False)
