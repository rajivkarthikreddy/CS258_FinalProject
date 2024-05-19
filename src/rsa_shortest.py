import matplotlib.pyplot as plt
import networkx as nx
import pickle
class Request:
    """This class represents a request. Each request is characterized by source and destination nodes and holding time (represented by an integer).

    The holding time of a request is the number of time slots for this request in the network. You should remove a request that exhausted its holding time.
    """
    def __init__(self, s, t, ht):
        self.s = s
        self.t = t
        self.ht = ht

    def __str__(self) -> str:
        return f'req({self.s}, {self.t}, {self.ht})'
    
    def __repr__(self) -> str:
        return self.__str__()
        
    def __hash__(self) -> int:
        # used by set()
        return self.id


class EdgeStats:
    """This class saves all state information of the system. In particular, the remaining capacity after request mapping and a list of mapped requests should be stored.
    """
    def __init__(self, u, v, cap) -> None:
        self.id = (u,v)
        self.u = u
        self.v = v 
        # remaining capacity
        self.cap = cap 

        # spectrum state (a list of requests, showing color <-> request mapping). Each index of this list represents a color
        self.__slots = [None] * cap
        # a list of the remaining holding times corresponding to the mapped requests
        self.__hts = [0] * cap

    def __str__(self) -> str:
        return f'{self.id}, cap = {self.cap}: {self.reqs}'
    
    def add_request(self, req: Request, color:int):
        """update self.__slots by adding a request to the specific color slot

        Args:
            req (Request): a request
            color (int): a color to be used for the request
        """

        self.__slots[color] = req
        self.__hts[color] = req.ht

    def remove_requests(self):
        """update self.__slots by removing the leaving requests based on self.__hts; Also, self.__hts should be updated in this function.
        """
        for i in range(len(self.__slots)):
            self.__hts[i] -= 1
            if self.__hts[i] <= 0: 
                self.__slots[i] = None
                print(f"Request in slot {i} on edge removed")

    def get_available_colors(self) -> list[int]:
        """return a list of integers available to accept requests
        """
        available_colors = []
        for index, request in enumerate(self.__slots):
            if request is None:
                available_colors.append(index)
        return available_colors
    
    def show_spectrum_state(self):
        """Come up with a representation to show the utilization state of a link (by colors)
        """
        state_representation = []
        for slot in self.__slots:
            if slot is None:
                state_representation.append('-')  # Represent an available slot
            else:
                state_representation.append('X')  # Represent an occupied slot
            print(' '.join(state_representation))
    
    def get_utilization(self):
        occupied_colors = sum(1 for req in self.__slots if req is not None)
        return occupied_colors / self.cap

def generate_requests() -> list[Request]:
    """Generate a set of requests, 

    Args:
        num_reqs (int): the number of requests
        g (nx.Graph): network topology

    Returns:
        list[Request]: a list of request instances
    """
    requests = []
    raw_request = []

    #change input file request_c1 or request_c2 for case I or II
    with open ('request_c2', 'rb') as fp:
        raw_requests = pickle.load(fp)

    for raw_request in raw_requests:
        # print(type(raw_request))
        requests.append(Request(*raw_request))

    return requests


def generate_graph() -> nx.Graph:
    """Generate a networkx graph instance importing a GML file. Set the capacity attribute to all links based on a random distribution.
    
    Returns:
        nx.Graph: a weighted graph
    """
    graph = nx.read_gml('nsfnet.gml')

    # Define the capacity value
    capacity_value = 10

    # Add capacity attribute to each edge in the graph
    for edge in graph.edges():
        graph.edges[edge]['capacity'] = capacity_value
    return graph


## Since estats is passed by reference, the function now simply returns a boolean indicating if a request is blocked
def route(g: nx.Graph, estats: list[EdgeStats], req:Request):
    """Use a routing algorithm to decide a mapping of requests onto a network topology. The results of mapping should be reflected. Consider available colors on links on a path. 

    Args:
        g (nx.Graph): a network topology
        req (Request): a request to map

    Returns:
        list[EdgeStats]: updated EdgeStats
    """

    node_index_to_name = {i: name for i, name in enumerate(g.nodes)}
    # node_name_to_index = {name: i for i, name in enumerate(g.nodes)}

    source_node = node_index_to_name[req.s]
    target_node = node_index_to_name[req.t]
    path = nx.shortest_path(g, source=source_node, target=target_node)
    # path = nx.shortest_path(g, source=req.s, target=req.t)
    print('Path = ', path)
    
    # Collect available colors for each edge in the path
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    
    # Function to find EdgeStats for a given edge
    def find_edge_stats(u, v):
        return next((edge for edge in estats if (edge.u, edge.v) == (u, v) or (edge.u, edge.v) == (v, u)), None)

    # Get sets of available colors for each edge in the path
    available_colors = []
    for u, v in path_edges:
        edge_stats = find_edge_stats(u, v)
        if edge_stats:
            available_colors.append(set(edge_stats.get_available_colors()))
        else:
            print(f"No EdgeStats found for edge ({u}, {v}), blocking the request.")
            return False

    # Find a common color available across all edges
    common_colors = set.intersection(*available_colors)
    if not common_colors:
        print("No common color available, blocking the request.")
        return False  # No common color available, the request is blocked
    
    # Sort the common colors and select the lowest
    lowest_common_color = sorted(common_colors)[0]  # Get the lowest available common color

    # Assign the lowest common color to the request on all edges
    for u, v in path_edges:
        edge_stats = find_edge_stats(u, v)
        if edge_stats:
            edge_stats.add_request(req, lowest_common_color)
        else:
            print(f"Failed to update EdgeStats for edge ({u}, {v}) after initial check.")
    
    return True


def plot_data(data, title, xlabel, ylabel, line_label=None):
    """
    General purpose function to plot given data.
    """
    plt.figure()
    plt.plot(data, label=line_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if line_label:
        plt.legend()
    plt.grid(True)
    plt.show()

        
if __name__ == "__main__":
    # generate a network

    G = generate_graph()

    requests = []

    # Generate/retrieve requests
    requests = generate_requests()

    # Initialize edge statistics
    estats = []
    for u, v in G.edges():
        estats.append(EdgeStats(u, v, G[u][v]['capacity']))


    blocked_requests_cumulative = 0
    blocked_requests_at_each_round = []
    link_utilization = {}  # Dictionary to store link utilizations
    network_wide_utilization = []
    # this simulation follows the discrete event simulation concept. Each time slot is defined by an arrival of a request
    count = 0
    for req in requests:
        # use the route function to map the request onto the topology (update EdgeStats)
        print(f"received request {req}")
        # Route the request and check if it was successful

        blocked_requests_each_time_round = 0

        success = route(G, estats, req)
        
        if not success :
            print(f"blocked req {req}")
            # blocked_requests_cumulative += 1              # use this if cumulative is needed
            blocked_requests_each_time_round += 1           # use this variable if at each instance of time is needed
        else:
            # Only if the routing is successful, we show the spectrum state
            for edge in estats:
                edge.show_spectrum_state()

        
        # Remove all requests that exhausted their holding times (use remove_requests)
        for estat in estats:
            estat.remove_requests()
        
        blocked_requests_at_each_round.append(blocked_requests_each_time_round)         ## change this accordingly cumulative or non-cumulative

        count = count + 1

        # Calculate and store the utilization of each link
        temp_network_wide_utilization = 0
        for estat in estats:
            link = (estat.u, estat.v)
            utilization = estat.get_utilization()  # Assuming get_utilization method exists
            if link not in link_utilization:
                link_utilization[link] = 0
            link_utilization[link] = (link_utilization[link] * (count - 1) + utilization) / count
            temp_network_wide_utilization = temp_network_wide_utilization + link_utilization[link]

        network_wide_utilization.append(temp_network_wide_utilization/len(G.edges()))       

    # print(network_wide_utilization)

    # Plot network-wide utilization over episodes
    plot_data(network_wide_utilization, 'Network Wide Utilization over T episodes', 'Time Round', 'Network Utilization')
    
    # Print the utilization of each link at each round
    for link, utilization in link_utilization.items():
        print(f"Link {link}: {utilization}")

    # print("blocked requests : ", blocked_requests_at_each_round)

    # Plot the number of blocked requests over time
    plot_data(blocked_requests_at_each_round, 'Trend of Blocked Requests Over Time', 'Time Round', 'Number of Blocked Requests')
