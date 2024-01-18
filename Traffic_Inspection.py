import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nodes_filepath = "data/nodes.txt"
edges_filepath = "data/edges.txt"

df_nodes = pd.read_csv(nodes_filepath, sep=' ')
print(df_nodes)
df_edges = pd.read_csv(edges_filepath, sep=' ')
print(df_edges)