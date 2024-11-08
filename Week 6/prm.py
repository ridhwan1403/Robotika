import yaml
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from scipy.spatial import KDTree

# Muat parameter dari params.yaml
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

NUM_NODES = params["num_nodes"]
MAP_LIMITS = params["map_limits"]
CONNECTION_RADIUS = params["connection_radius"]
START = tuple(params["start"])
GOAL = tuple(params["goal"])

# Fungsi untuk membuat node acak
def generate_random_nodes(num_nodes, map_limits):
    nodes = []
    for _ in range(num_nodes):
        x = random.uniform(map_limits[0], map_limits[1])
        y = random.uniform(map_limits[0], map_limits[1])
        nodes.append((x, y))
    return nodes

# Bangun graf dengan koneksi node dalam radius tertentu
def build_roadmap(nodes, radius):
    tree = KDTree(nodes)
    graph = nx.Graph()
    for i, node in enumerate(nodes):
        neighbors = tree.query_ball_point(node, radius)
        for neighbor in neighbors:
            if neighbor != i:
                distance = np.linalg.norm(np.array(node) - np.array(nodes[neighbor]))
                graph.add_edge(i, neighbor, weight=distance)
    return graph

# Cari jalur terpendek dengan A*
def find_shortest_path(graph, start_idx, goal_idx):
    try:
        return nx.astar_path(graph, start_idx, goal_idx, heuristic=lambda a, b: np.linalg.norm(np.array(a) - np.array(b)))
    except nx.NetworkXNoPath:
        return None

# Visualisasi PRM dan hasil jalur
def visualize(nodes, graph, path=None):
    plt.figure(figsize=(10, 10))
    for edge in graph.edges:
        x = [nodes[edge[0]][0], nodes[edge[1]][0]]
        y = [nodes[edge[0]][1], nodes[edge[1]][1]]
        plt.plot(x, y, "gray")

    # Plot node dan jalur yang ditemukan
    for node in nodes:
        plt.plot(node[0], node[1], "bo", markersize=3)
    if path:
        for i in range(len(path) - 1):
            x = [nodes[path[i]][0], nodes[path[i + 1]][0]]
            y = [nodes[path[i]][1], nodes[path[i + 1]][1]]
            plt.plot(x, y, "r", linewidth=2)

    plt.plot(START[0], START[1], "go", markersize=10, label="Start")
    plt.plot(GOAL[0], GOAL[1], "ro", markersize=10, label="Goal")
    plt.legend()
    plt.show()

# Implementasi utama
def main():
    nodes = generate_random_nodes(NUM_NODES, MAP_LIMITS)
    nodes.append(START)
    nodes.append(GOAL)

    graph = build_roadmap(nodes, CONNECTION_RADIUS)
    
    # Index untuk start dan goal
    start_idx = len(nodes) - 2
    goal_idx = len(nodes) - 1
    
    path = find_shortest_path(graph, start_idx, goal_idx)
    if path:
        print("Path found!")
    else:
        print("No path found.")
    visualize(nodes, graph, path)

if __name__ == "__main__":
    main()
