import pandas as pd
#from Traffic_Inspection import build_graph
import networkx as nx
import json
import os



def calculate_degree_distribution(graph):
    """
    Calculate the degree distribution of a graph.

    Nodes with high degrees are important intersections or hubs. 
    Interesting to see how intersections are built up in the city.
    Ratio of 4 way intersections to 3 way intersections could be interesting.
    The number of endpoints (degree 1), could also be interesting

    Parameters:
    - graph (NetworkX Graph): The input graph.

    Returns:
    - dict: A dictionary where keys are nodes and values are their degrees.
    """
    return dict(graph.degree())


def calculate_betweenness_centrality(graph):
    """
    Calculate the betweenness centrality of nodes in a graph.

    Identify any super critical nodes that acts as a bridge between different parts of the network
    -> Critical Nodes that should be converted to highways 
    -> Interesting to see if this is the case with our simulated example

    Parameters:
    - graph (NetworkX Graph): The input graph.

    Returns:
    - dict: A dictionary where keys are nodes and values are their betweenness centrality scores.
    """
    return nx.betweenness_centrality(graph)


def calculate_closeness_centrality(graph):
    """
    Calculate the closeness centrality of nodes in a graph.

    Identifies nodes that are central in terms of travel time or distance to other nodes.

    Parameters:
    - graph (NetworkX Graph): The input graph.

    Returns:
    - dict: A dictionary where keys are nodes and values are their closeness centrality scores.
    """
    return nx.closeness_centrality(graph)


def calculate_density(graph):
    """
    Calculate the density of a graph.

    Low density means that the graph is not well connected
    -> Trivial, we can't build roads from every intersevtion to every other intersection
       That's just not how roads work.
    -> Road Network could be given as an example for a low-density network  

    Parameters:
    - graph (NetworkX Graph): The input graph.

    Returns:
    - float: The density of the graph.
    """
    return nx.density(graph)


def calculate_clustering_coefficient(graph):
    """
    Calculate the average clustering coefficient of a graph.

    Clusters indicate neighbourhoods/ local communities
    -> Higher clustering coefficient would indicate a more community based construction

    Parameters:
    - graph (NetworkX Graph): The input graph.

    Returns:
    - float: The average clustering coefficient.
    """
    return nx.average_clustering(graph)


def calculate_average_shortest_path_length(graph):
    """
    Calculate the average shortest path length of a graph.

    Interesting to determine the efficiency of the network to drive from any point to all other points.
    Possibly a metric we could compare between the original and the highway version

    Parameters:
    - graph (NetworkX Graph): The input graph.

    Returns:
    - float: The average shortest path length.
    """
    return nx.average_shortest_path_length(graph)


def check_connectivity(graph):
    """
    Check if a graph is connected.

    See if the transportation network is connected.
    Possible non-connectedness would mean that people have no way of getting to the other places by the road network.

    Parameters:
    - graph (NetworkX Graph): The input graph.

    Returns:
    - bool: True if the graph is connected, False otherwise.
    """
    return nx.is_connected(graph)


def calculate_resilience(graph):
    """
    Calculate the node connectivity of a graph, indicating its resilience.

    How resilient is te network against attacks (traffic jams, closing of street, climate protest).
    Would also be intereting to compare this with the value of the highway version

    Note: The paths with only 1 edge need to be removed, else the result is trivial (1).
    As it would only take 1 edge to completely disconnect a Node from the Graph.
    It takes quite long to compute, the value for the unmodified data is still only 1. 

    Parameters:
    - graph (NetworkX Graph): The input graph.

    Returns:
    - int: The node connectivity of the graph.
    """

    # Create a copy of the graph to avoid modifying the original
    modified_graph = graph.copy()

    # Remove nodes with degree 1, else resilience is 1
    nodes_to_remove = [node for node, degree in graph.degree() if degree == 1]
    modified_graph.remove_nodes_from(nodes_to_remove)

    # Calculate node connectivity
    resilience = nx.node_connectivity(modified_graph)

    return resilience


def calc_metrics(city_graph):
    degree_distribution = calculate_degree_distribution(city_graph)
    betweenness_centrality = calculate_betweenness_centrality(city_graph)
    closeness_centrality = calculate_closeness_centrality(city_graph)
    density = calculate_density(city_graph)
    clustering_coefficient = calculate_clustering_coefficient(city_graph)
    average_shortest_path_length = calculate_average_shortest_path_length(city_graph)
    connectivity = check_connectivity(city_graph)
    resilience = calculate_resilience(city_graph)

    # Save metrics to a JSON file
    metrics_data = {
        "degree_distribution": degree_distribution,
        "betweenness_centrality": betweenness_centrality,
        "closeness_centrality": closeness_centrality,
        "density": density,
        "clustering_coefficient": clustering_coefficient,
        "average_shortest_path_length": average_shortest_path_length,
        "connectivity": connectivity,
        "resilience": resilience
    }

    with open("network_metrics.json", "w") as json_file:
        json.dump(metrics_data, json_file, indent=2)

    print("Metrics saved to network_metrics.json")

def load_metrics(file_path="network_metrics.json"):
    metrics_data = {}

    if not os.path.isfile(file_path):
        with open(file_path, "w") as json_file:
            json.dump(metrics_data, json_file, indent=2)

    with open(file_path, "r") as json_file:
        metrics_data = json.load(json_file)
    
    return metrics_data

def print_metrics(network_metrics):
    degree_distribution = network_metrics["degree_distribution"]
    betweenness_centrality = network_metrics["betweenness_centrality"]
    closeness_centrality = network_metrics["closeness_centrality"]
    density = network_metrics["density"]
    clustering_coefficient = network_metrics["clustering_coefficient"]
    average_shortest_path_length = network_metrics["average_shortest_path_length"]
    connectivity = network_metrics["connectivity"]
    resilience = network_metrics["resilience"]

    # Printing metrics
    print("Degree Distribution:", degree_distribution)
    print("Betweenness Centrality:", betweenness_centrality)
    print("Closeness Centrality:", closeness_centrality)
    print("Density:", density)
    print("Clustering Coefficient:", clustering_coefficient)
    print("Average Shortest Path Length:", average_shortest_path_length)
    print("Connectivity:", connectivity)
    print("Resilience:", resilience)

def main():
    nodes_filepath = "data/nodes.txt"
    edges_filepath = "data/edges.txt"

    # load data
    df_nodes = pd.read_csv(nodes_filepath, sep=' ')
    df_edges = pd.read_csv(edges_filepath, sep=' ')

    city_graph = build_graph(df_nodes, df_edges)

    # Calculate the metrics
    #calc_metrics(city_graph)

    # load metrics
    network_metrics = load_metrics()

    # print metrics
    print_metrics(network_metrics)



if __name__ == "__main__":
    main()