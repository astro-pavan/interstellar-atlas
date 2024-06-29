import numpy as np
from numba import jit
from scipy.spatial import KDTree
from scipy.optimize import minimize, basinhopping, dual_annealing
from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx

call_count = 0


@jit(nopython=True)
def distance_between_2_points(x1, x2):
    return np.sqrt(np.dot(x1 - x2, x1 - x2))


class graph_node:

    def __init__(self, name, position):
        self.name = name
        self.position = position
        self.map_position = np.array((position[0], position[2]))

        self.neighbours, self.neighbour_names, self.distances = [], [], []
        self.neighbour_count = 0
        self.map_distances = []

    def link(self, node, n=None):

        if node.name not in self.neighbour_names:

            self.neighbours.append(node)
            node.neighbours.append(self)

            self.neighbour_names.append(node.name)
            node.neighbour_names.append(self.name)

            d3 = distance_between_2_points(self.position, node.position)

            self.distances.append(d3)
            node.distances.append(d3)

            self.neighbour_count += 1
            node.neighbour_count += 1

            if n is not None:

                if self.neighbour_count > 7:

                    def objective_function(x):
                        self.map_position = x
                        return self.map_error(triple=False)

                    def objective_function_v2(x):
                        self.set_neighbour_map_positions(x)
                        return self.map_error(triple=False)

                    #basinhopping(objective_function, self.map_position)
                    #dual_annealing(objective_function, ((-20, 20), (-20, 20)))
                    #print('Minimizing...')
                    minimize(objective_function, self.map_position)
                    #minimize(objective_function_v2, self.get_neighbour_map_positions())

    def map_error(self, triple=True):

        total, count = 0, 0

        if triple:
            for i_1 in range(self.neighbour_count):
                for i_2 in range(self.neighbour_count):
                    if i_1 != i_2:
                        dist_3d_1, dist_3d_2 = self.distances[i_1], self.distances[i_2]
                        dist_2d_1 = distance_between_2_points(self.map_position, self.neighbours[i_1].map_position)
                        dist_2d_2 = distance_between_2_points(self.map_position, self.neighbours[i_2].map_position)

                        ratio3 = dist_3d_2 / dist_3d_1
                        ratio2 = dist_2d_2 / dist_2d_1

                        total += (ratio2 - ratio3) ** 2
                        count += 1
        else:
            for i in range(self.neighbour_count):
                dist_3d = self.distances[i]
                dist_2d = distance_between_2_points(self.map_position, self.neighbours[i].map_position)
                error = np.abs(dist_3d - dist_2d)
                total += error
                count += 1

        return total

    def update_map_distances(self):
        self.map_distances = []
        for i in range(self.neighbour_count):
            dist = distance_between_2_points(self.map_position, self.neighbours[i].map_position)
            self.map_distances.append(dist)

    def get_neighbour_map_positions(self):
        res = np.zeros((self.neighbour_count + 1, 2))
        res[0, :] = self.map_position
        for i in range(self.neighbour_count):
            res[i + 1, :] = self.neighbours[i].map_position
        return np.array(res).flatten()

    def set_neighbour_map_positions(self, pos):
        pos = pos.reshape(-1, 2)
        self.map_position = pos[0, :]
        for i in range(self.neighbour_count):
            self.neighbours[i].map_position = pos[i + 1, :]


def construct_graph(names, positions, n=5, find_map_positions=False):

    node_list = []

    for i in range(len(names)):
        node = graph_node(names[i], positions[i])
        node_list.append(node)

    tree = KDTree(positions)
    graph_size = len(node_list)

    for i in range(len(node_list)):

        if n is None and find_map_positions:
            n = graph_size

        node = node_list[i]
        neighbour_distances, neighbour_indexes = tree.query(node.position, k=n)

        for j in neighbour_indexes[1:]:
            neighbour_node = node_list[j]
            node.link(neighbour_node, n=n)

    if find_map_positions:
        for n in node_list:
            n.update_map_distances()

    return node_list


def assign_map_positions(x2, nodes):
    vals = x2.reshape(-1, 2)  # turns it into 2d array from flat array
    for i in range(len(nodes)):
        nodes[i].map_position = vals[i, :]


def get_map_positions_list(nodes):
    x2 = np.zeros((len(nodes), 2))
    for i in range(len(nodes)):
        x2[i, :] = nodes[i].map_position
    return x2.flatten()


def error_function(x2, nodes):
    global call_count
    call_count += 1
    assign_map_positions(x2, nodes)
    total = 0
    for n in nodes:
        total += n.map_error()
    return total / len(nodes)


def make_map(nodes):
    global call_count
    call_count = 0

    print(f'Making map out of {len(nodes)} points...')

    initial_guess = get_map_positions_list(nodes)
    objective_function = lambda x2: error_function(x2, nodes)

    # result = basinhopping(objective_function, initial_guess.flatten())
    result = minimize(objective_function, initial_guess.flatten())
    assign_map_positions(result.x, nodes)
    print(result.message)

    print('Map completed')
    print(f'Call count: {call_count}')

    dist3 = distance_between_2_points(nodes[0].position, nodes[1].position)
    dist2 = distance_between_2_points(nodes[0].map_position, nodes[1].map_position)

    for n in nodes:
        n.map_position = n.map_position * (dist3 / dist2)

    for n in nodes:
        n.update_map_distances()


def error_summary(nodes):
    errors = []
    for n in nodes:
        for i in range(n.neighbour_count):
            e = np.abs(np.log10(n.map_distances[i]/n.distances[i]))
            errors.append(e)

    errors = np.array(errors)
    mean_error = np.mean(errors)

    print(mean_error)

    return mean_error


# Function to convert graph nodes to a NetworkX graph
def create_networkx_graph(nodes):
    G = nx.Graph()
    edge_labels = {}
    for node in nodes:
        G.add_node(node.name)
        for idx, neighbour in enumerate(node.neighbour_names):
            if (node.name, neighbour) not in edge_labels and (neighbour, node.name) not in edge_labels:
                G.add_edge(node.name, neighbour)
                # edge_labels[(node.name, neighbour)] = f"3D:{node.distances[idx]:.1f} 2D:{node.map_distances[idx]:.1f}"
                edge_labels[(node.name, neighbour)] = f"{node.map_distances[idx]/node.distances[idx]:.4f}"
    return G, edge_labels


# Function to plot the graph using matplotlib
def plot_graph(nodes, labels=True):
    G, edge_labels = create_networkx_graph(nodes)
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold',
            edge_color='gray')
    if labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.show()
