import dgl
from dgl.data import DGLDataset
import torch
from scipy.spatial import Delaunay
from scipy.linalg import null_space
from scipy.linalg import eig
import scipy.sparse as sp
# from scipy import sparse as sp
from numpy.linalg import eigh

import random
import numpy as np
import networkx as nx
import itertools


def is_inside_rectangle(x, rect):
    return rect[0, 0] <= x[0] <= rect[1, 0] and rect[0, 1] <= x[1] <= rect[1, 1]

def sample_point_from_rect(points, rect):
    samples = []
    for i in range(len(points)):
        if is_inside_rectangle(points[i], rect):
            samples.append(i)

    return random.choice(samples)

def create_hole(points, triangles, hole):
    kept_triangles = []
    removed_vertices = set()

    # Find the points and triangles to remove
    for i in range(len(triangles)):
        simplex = triangles[i]
        assert len(simplex) == 3
        xs = points[simplex]

        remove_triangle = False
        for j in range(3):
            vertex = simplex[j]
            if is_inside_rectangle(xs[j], hole):
                remove_triangle = True
                removed_vertices.add(vertex)

        if not remove_triangle:
            kept_triangles.append(i)

    # Remove the triangles and points inside the holes
    triangles = triangles[np.array(kept_triangles)]

    # Remove the points that are not part of any triangles anymore.
    # This can happen in some very rare cases
    for i in range(len(points)):
        if np.sum(triangles == i) == 0:
            removed_vertices.add(i)

    points = np.delete(points, list(removed_vertices), axis=0)

    # Renumber the indices of the triangles' vertices
    for vertex in sorted(removed_vertices, reverse=True):
        triangles[triangles >= vertex] -= 1

    return points, triangles

def create_graph_from_triangulation(points, triangles):
    # Create a graph from from this containing only the non-removed triangles
    G = nx.Graph()
    edge_idx = 0
    edge_to_tuple = {}
    tuple_to_edge = {}

    for i in range(len(triangles)):
        vertices = triangles[i]
        for j in range(3):
            if vertices[j] not in G:
                G.add_node(vertices[j], point=points[vertices[j]])

            for v1, v2 in itertools.combinations(vertices, 2):
                if not G.has_edge(v1, v2):
                    G.add_edge(v1, v2, index=edge_idx)
                    edge_to_tuple[edge_idx] = (min(v1, v2), max(v1, v2))
                    tuple_to_edge[(min(v1, v2), max(v1, v2))] = edge_idx
                    edge_idx += 1
                assert G.has_edge(v2, v1)
                
    G.graph['edge_to_tuple'] = edge_to_tuple
    G.graph['tuple_to_edge'] = tuple_to_edge
    G.graph['points'] = points
    G.graph['triangles'] = triangles
    return G

def generate_trajectory(start_rect, end_rect, ckpt_rect, G: nx.Graph):
    points = G.graph['points']
    tuple_to_edge = G.graph['tuple_to_edge']

    start_vertex = sample_point_from_rect(points, start_rect)
    end_vertex = sample_point_from_rect(points, end_rect)
    ckpt_vertex = sample_point_from_rect(points, ckpt_rect)

    x = np.zeros((len(tuple_to_edge), 1))

    vertex = start_vertex
    end_point = points[end_vertex]
    ckpt_point = points[ckpt_vertex]

    path = [vertex]
    explored = set()

    ckpt_reached = False

    while vertex != end_vertex:
        explored.add(vertex)
        if vertex == ckpt_vertex:
            ckpt_reached = True

        nv = np.array([nghb for nghb in G.neighbors(vertex)
                       if nghb not in explored])
        if len(nv) == 0:
            # If we get stuck because everything around was explored
            # Then just try to generate another trajectory.
            return generate_trajectory(start_rect, end_rect, ckpt_rect, G)
        npoints = points[nv]

        if ckpt_reached:
            dist = np.sum((npoints - end_point[None, :]) ** 2, axis=-1)
        else:
            dist = np.sum((npoints - ckpt_point[None, :]) ** 2, axis=-1)

        # prob = softmax(-dist**2)
        # vertex = nv[np.random.choice(len(prob), p=prob)]
        coin_toss = np.random.uniform()

        if coin_toss < 0.1:
            vertex = nv[np.random.choice(len(dist))]
        else:
            vertex = nv[np.argmin(dist)]

        path.append(vertex)

        # Set the flow value according to the orientation
        if path[-2] < path[-1]:
            x[tuple_to_edge[(path[-2], path[-1])], 0] = 1
        else:
            x[tuple_to_edge[(path[-1], path[-2])], 0] = -1

    return x, path


def extract_boundary_matrices(G: nx.Graph):
    """Compute the boundary and co-boundary matrices for the edges of the complex. """
    edge_to_tuple = G.graph['edge_to_tuple']
    tuple_to_edge = G.graph['tuple_to_edge']
    triangles = G.graph['triangles']

    B1 = np.zeros((G.number_of_nodes(), G.number_of_edges()), dtype=float)
    for edge_id in range(G.number_of_edges()):
        nodes = edge_to_tuple[edge_id]
        min_node = min(nodes)
        max_node = max(nodes)
        B1[min_node, edge_id] = -1
        B1[max_node, edge_id] = 1

    assert np.all(np.sum(np.abs(B1), axis=-1) > 0)
    assert np.all(np.sum(np.abs(B1), axis=0) == 2)
    assert np.all(np.sum(B1, axis=0) == 0)

    def extract_edge_and_orientation(triangle, i):
        assert i <= 2
        n1 = triangle[i]
        if i < 2:
            n2 = triangle[i + 1]
        else:
            n2 = triangle[0]

        if n1 < n2:
            orientation = 1
        else:
            orientation = -1

        return tuple_to_edge[(min(n1, n2), max(n1, n2))], orientation

    B2 = np.zeros((G.number_of_edges(), len(triangles)), dtype=float)
    for i in range(len(triangles)):
        edge1, orientation1 = extract_edge_and_orientation(triangles[i], 0)
        edge2, orientation2 = extract_edge_and_orientation(triangles[i], 1)
        edge3, orientation3 = extract_edge_and_orientation(triangles[i], 2)
        assert edge1 != edge2 and edge1 != edge3 and edge2 != edge3

        B2[edge1, i] = orientation1
        B2[edge2, i] = orientation2
        B2[edge3, i] = orientation3

    assert np.all(np.sum(np.abs(B2), axis=0) == 3)
    assert np.all(np.sum(np.abs(B2), axis=-1) > 0)
    return B1, B2

def get_hodge_laplacian(B1, B2):
    return B1.T @ B1 + B2 @ B2.T

def get_smallest_k_eigenvectors(L1, k):
    eigenvalues, eigenvectors = eigh(L1)
    sorted_eigenvalues = np.sort(eigenvalues)
    
    k_smallest_eigen = []
    for i in range(k):
        maxcol = list(eigenvalues).index(sorted_eigenvalues[i])
        v = eigenvectors[:, maxcol]
        k_smallest_eigen.append(np.abs(v))
    
    return k_smallest_eigen

def process_edge_features(k_smallest_eigen):
    k = len(k_smallest_eigen)

    eigenvectors = []
    for i in range(len(k_smallest_eigen[0])):
        eigenvectors.append([k_smallest_eigen[j][i] for j in range(k)])

    return eigenvectors

def generate_trajectory(start_rect, end_rect, ckpt_rect, G: nx.Graph):
    points = G.graph['points']
    tuple_to_edge = G.graph['tuple_to_edge']

    start_vertex = sample_point_from_rect(points, start_rect)
    end_vertex = sample_point_from_rect(points, end_rect)
    ckpt_vertex = sample_point_from_rect(points, ckpt_rect)

    vertex = start_vertex
    end_point = points[end_vertex]
    ckpt_point = points[ckpt_vertex]

    path = [vertex]
    explored = set()

    ckpt_reached = False

    upper_rect = np.array([[0.0, 0.5], [1.0, 1.0]])
    if is_inside_rectangle(ckpt_point, upper_rect):
        treshold_rect = np.array([[0.0, 0.5], [1.0, 1.0]])
    else:
        treshold_rect = np.array([[0.0, 0.0], [1.0, 0.5]])

    while vertex != end_vertex:
        explored.add(vertex)
        if vertex == ckpt_vertex:
            ckpt_reached = True

        nv = np.array([nghb for nghb in G.neighbors(vertex)
                       if nghb not in explored and is_inside_rectangle(points[nghb], treshold_rect)])
        if len(nv) == 0:
            # If we get stuck because everything around was explored
            # Then just try to generate another trajectory.
            return generate_trajectory(start_rect, end_rect, ckpt_rect, G)
        npoints = points[nv]

        if ckpt_reached:
            dist = np.sum((npoints - end_point[None, :]) ** 2, axis=-1)
        else:
            dist = np.sum((npoints - ckpt_point[None, :]) ** 2, axis=-1)

        coin_toss = np.random.uniform()

        if coin_toss < 0.1:
            vertex = nv[np.random.choice(len(dist))]
        else:
            vertex = nv[np.argmin(dist)]

        path.append(vertex)

    return path

def generate_random_trajectories(G: nx.Graph, num_train = 1000, num_test = 1000):
    start_rect = np.array([[0.0, 0.4], [0.1, 0.6]])
    end_rect = np.array([[0.8, 0.4], [1.0, 0.6]])
    bot_rect = np.array([[0.4, 0.0], [0.6, 0.2]])
    top_rect = np.array([[0.4, 0.8], [0.6, 1.0]])
    mid_rect = [bot_rect, top_rect]
    
    trajectories, labels = [], []
    
    for i in range(num_test + num_train):
        rand = random.randint(0, 1)
        
        trajectories.append(generate_trajectory(start_rect, end_rect, mid_rect[rand], G))
        labels.append(rand)
            
    return trajectories, labels

def get_positional_encoding(g, pos_enc_dim):
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)

    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1), dtype=float)
    L = N * sp.eye(g.number_of_nodes()) - A

    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]

    return torch.from_numpy(np.real(EigVec[:, :pos_enc_dim])).float()

def load_flow_dataset(num_points=1000, num_train=1000, num_test=1000, k=3):
    points = np.random.uniform(low=-0.05, high=1.05, size=(num_points, 2))
    triangulation = Delaunay(points)

    for i in range(len(points)):
        assert np.sum(triangulation.simplices == i) > 0

    hole = np.array([[0.4, 0.4], [0.6, 0.6]])
    points, triangles = create_hole(points, triangulation.simplices, hole)

    # Make sure that each point appears in some triangle.
    for i in range(len(points)):
        assert np.sum(triangles == i) > 0

    assert np.min(triangles) == 0
    assert np.max(triangles) == len(points) - 1
    
    G = create_graph_from_triangulation(points, triangles)
    assert G.number_of_nodes() == len(points)
    
    B1, B2 = extract_boundary_matrices(G)
    L1 = get_hodge_laplacian(B1, B2)
    
    trajectories, labels = generate_random_trajectories(G)
    k_smallest_eigen = get_smallest_k_eigenvectors(L1, 1)
    eigenvectors = process_edge_features(k_smallest_eigen)

    return G, eigenvectors, trajectories, labels

class Flow_Dataset(DGLDataset):
    def __init__(self):
        super().__init__(name='flow')

    def process(self):
        self.G, self.zero_eigenvector, self.trajectories, self.labels = load_flow_dataset()

        self.graphs = []
        self.graph_info = []

        for trajectory in self.trajectories:
            num_nodes = len(self.G.graph["points"])
            src = np.array([nodes[0] for nodes in self.G.graph["tuple_to_edge"]])
            dst = np.array([nodes[1] for nodes in self.G.graph["tuple_to_edge"]])

            node_features = np.array([[1] if i in trajectory else [0] for i in range(num_nodes)]).astype(float)
            node_features = torch.from_numpy(node_features)
            edge_features = np.array(self.zero_eigenvector)
            edge_features = torch.from_numpy(edge_features)

            g = dgl.graph((src, dst), num_nodes=num_nodes)
            # positional_encoding = get_positional_encoding(g, 3)
            
            g.ndata["node_features"] = node_features
            # g.ndata["eig"] = positional_encoding
            g.edata["edge_features"] = edge_features

            g_info = {}
            g_info["trajectory"] = trajectory

            g = dgl.add_self_loop(g)

            self.graphs.append(g)
            self.graph_info.append(g_info)

        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i], self.graph_info[i]

    def __len__(self):
        return len(self.graphs)

if __name__ == "__main__":
    dataset = Flow_Dataset()