import numpy as np
import networkx as nx
import pickle

# Read the network graph from a GML file
graph = nx.read_gml('nsfnet.gml')
num_nodes = len(graph.nodes)

def generate_c1_request():
    """
    Generate a random connection request for Case I
    where the source and target nodes are randomly selected.
    """
    source = np.random.randint(0, num_nodes) # Randomly select a source node
    target = source
    while target == source:  # Ensure the target is different from the source
        target = np.random.randint(0, num_nodes)
    holding_time = np.random.randint(10, 21) # Randomly generate holding time between 10 and 20
    return source, target, holding_time

def generate_c2_request():
    """
    Generate a fixed connection request for Case II
    where the source is 'San Diego Supercomputer Center' and 
    the target is 'Jon Von Neumann Center, Princeton, NJ'.
    """
    node_list = list(graph.nodes)
    source = node_list.index('San Diego Supercomputer Center') # Fixed source node
    target = node_list.index('Jon Von Neumann Center, Princeton, NJ') # Fixed target node
    holding_time = np.random.randint(10, 21) # Randomly generate holding time between 10 and 20
    return source, target, holding_time

# Generate 100 requests for Case I
out1 = [generate_c1_request() for _ in range(100)]

# Save the Case I requests to a file
with open('request_c1', 'wb') as fp:
    pickle.dump(out1, fp)

# Generate 100 requests for Case II
out2 = [generate_c2_request() for _ in range(100)]

# Save the Case II requests to a file
with open('request_c2', 'wb') as fp:
    pickle.dump(out2, fp)