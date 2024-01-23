import networkx as nx

"""
Note: If graph includes 'weight' attribute it is automatically considered

High-degree crossings/nodes can be seen as potential bottlenecks.
"""
def degree_centrality(graph):
    return nx.degree_centrality(graph)

"""
Note: To calculate it for weighted graph use weight_attr_name to set the name of the attribute to consider as weight

High betweenness -> key crossing connecting parts of the city
"""
def node_betweenness(graph, weight_attr_name=None):
    return nx.betweenness_centrality(graph, weight=weight_attr_name)

"""
Note: To calculate it for weighted graph use weight_attr_name to set the name of the attribute to consider as weight

High betweenness -> key road connecting parts of the city
"""
def edge_betweenness(graph, weight_attr_name=None):
    return nx.edge_betweenness_centrality(graph, weight=weight_attr_name)

"""
Nodes with high closeness centrality are more accessible and may play a crucial role in traffic redistribution.
"""
def closeness(graph, distance_attr):
    return nx.closeness_centrality(graph, distance_attr)

"""
Low transitivity might indicate more grid-like structure
High transitivity might indicate more interconnected road system
"""
def transitivity(graph):
    return nx.transitivity(graph)

"""
Note: To calculate it for weighted graph use weight_attr_name to set the name of the attribute to consider as weight

A shorter average path length can indicate better accessibility.
"""
def avg_shortest_path(graph, weight_attr_name=None):
    return nx.average_shortest_path_length(graph, weight=weight_attr_name)

"""
Resilience Measures -> Robustness to Edge Removal
Simulate the impact of road closures by removing nodes or edges and observing the effect on the network's connectivity.

weighted?
"""
#def edge_resilience(graph):

#def node_resilience(graph):
