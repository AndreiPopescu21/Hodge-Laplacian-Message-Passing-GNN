import seaborn as sns
import matplotlib.pyplot as plt
import os
import colorsys

from Flow_Dataset import Flow_Dataset

sns.set_style('white')
sns.color_palette("tab10")


def plot_arrow(p1, p2, color='red'):
    plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color=color,
        shape='full', lw=3, length_includes_head=True, head_width=.01, zorder=10)

def get_color_scale(r):
    g = 50
    b = 255 
    return (r / 255, g / 255, b / 255)

def visualise_trajectories():
    dataset = Flow_Dataset()
    G = dataset.G
    edge_to_tuple = G.graph['edge_to_tuple']
    triangles = G.graph['triangles']
    points = G.graph['points']

    plt.figure(figsize=(10, 8))
    plt.triplot(points[:, 0], points[:, 1], triangles)
    plt.plot(points[:, 0], points[:, 1], 'o')

    upper_trajectory, lower_trajectory = None, None

    for i, data in enumerate(dataset):
        graph, label, graph_info = data
        
        if upper_trajectory and lower_trajectory:
            break
        elif label.item() == 1:
            upper_trajectory = graph_info['trajectory']
        else:
            lower_trajectory = graph_info['trajectory']
    
    for i in range(1, len(upper_trajectory)):
        color='red'

        p1, p2 = points[upper_trajectory[i-1]], points[upper_trajectory[i]]

        plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color=color,
            shape='full', lw=3, length_includes_head=True, head_width=.01, zorder=10)

    for i in range(1, len(lower_trajectory)):
        color='blue'

        p1, p2 = points[lower_trajectory[i-1]], points[lower_trajectory[i]]

        plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color=color,
            shape='full', lw=3, length_includes_head=True, head_width=.01, zorder=10)

    plt.show()

def visualize_eigenvector():
    dataset = Flow_Dataset()
    G = dataset.G
    edge_to_tuple = G.graph['edge_to_tuple']
    triangles = G.graph['triangles']
    points = G.graph['points']
    
    graph, _, _ = dataset[0]
    eigenvector = graph.edata['edge_features']
    max_value = max(eigenvector).item()
    min_value = min(eigenvector).item()

    plt.figure(figsize=(10, 8))
    plt.triplot(points[:, 0], points[:, 1], triangles)
    plt.plot(points[:, 0], points[:, 1], 'o')

    for i, _ in enumerate(G.graph['tuple_to_edge']):
        nodes1 = edge_to_tuple[i]
        p1, p2 = points[nodes1[0]], points[nodes1[1]]

        value = eigenvector[i].item()
        mapped_value = (value - min_value) / (max_value - min_value) * 255
        color = get_color_scale(mapped_value)

        plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color=color,
            shape='full', lw=3, length_includes_head=True, head_width=0, zorder=10)

    plt.show()


if __name__ == "__main__":
    visualise_trajectories()
    visualize_eigenvector()