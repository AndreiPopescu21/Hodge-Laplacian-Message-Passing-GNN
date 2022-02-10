import seaborn as sns
import matplotlib.pyplot as plt
import os
import colorsys

from Flow2_Dataset import Flow_Dataset

sns.set_style('white')
sns.color_palette("tab10")


def plot_arrow(p1, p2, color='red'):
    plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color=color,
        shape='full', lw=3, length_includes_head=True, head_width=.01, zorder=10)

def get_color_scale(fraction):
    R =  (20-79) * fraction + 79
    G =  (255-79) * fraction + 79
    B =  (0-79) * fraction + 79
    return (int(R) / 255, int(G) / 255, int(B) / 255)

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

    plt.savefig("flow_trajectories.png")
    plt.show()

def show_eigenvector_visualisation(G, eigenvector):
    edge_to_tuple = G.graph['edge_to_tuple']
    triangles = G.graph['triangles']
    points = G.graph['points']

    plt.figure(figsize=(10, 8))
    plt.triplot(points[:, 0], points[:, 1], triangles)
    plt.plot(points[:, 0], points[:, 1], 'o')

    max_value = max(eigenvector).item()
    min_value = min(eigenvector).item()

    for i, _ in enumerate(G.graph['tuple_to_edge']):
        nodes1 = edge_to_tuple[i]
        p1, p2 = points[nodes1[0]], points[nodes1[1]]

        value = eigenvector[i].item()
        mapped_value = (value - min_value) / (max_value - min_value)
        color = get_color_scale(mapped_value)

        plt.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color=color,
            shape='full', lw=3, length_includes_head=True, head_width=0, zorder=10)

    # plt.savefig("flow_eigenvector_colormap.png")
    plt.show()

def visualize_eigenvector():
    dataset = Flow_Dataset()
    G = dataset.G
    graph, _, _ = dataset[0]

    eigenvector1 = []
    eigenvector2 = []
    
    for element in graph.edata['edge_features']:
        eigenvector1.append(element[0])
        eigenvector2.append(element[1])

    show_eigenvector_visualisation(G, eigenvector1)
    show_eigenvector_visualisation(G, eigenvector2)

    # plt.savefig("flow_eigenvector_colormap.png")
    plt.show()


if __name__ == "__main__":
    # visualise_trajectories()
    visualize_eigenvector()