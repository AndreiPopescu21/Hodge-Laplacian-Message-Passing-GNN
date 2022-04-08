import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from pathlib import Path
from scipy import sparse as sp

import dgl, torch
from dgl.data import DGLDataset

DEBUGGING_MODE = False

def load_data():
    with open(str(Path(__file__).parent.absolute()) + '/dataset/train.pickle', 'rb') as f:
        train = pickle.load(f)
    with open(str(Path(__file__).parent.absolute()) + '/dataset/test.pickle', 'rb') as f:
        test = pickle.load(f)
    with open(str(Path(__file__).parent.absolute()) + '/dataset/val.pickle', 'rb') as f:
        val = pickle.load(f)

    if DEBUGGING_MODE:
        train = train[:10]
        test = test[:10]
        val = val[:10]

    mask = {}
    mask['train'] = len(train)
    mask['test'] = len(train) + len(test)
    mask['val'] = len(train) + len(test) + len(val)
    
    data = train
    data.extend(test)
    data.extend(val)

    return data, mask

def get_color_scale(fraction):
    R =  (20-79) * fraction + 79
    G =  (255-79) * fraction + 79
    B =  (0-79) * fraction + 79
    return (int(R) / 255, int(G) / 255, int(B) / 255)

def draw(G:nx.graph, eigenvectors):
    edge_colors = []
    eigen1, eigen2 = [], []

    for i in range(len(eigenvectors)):
        eigen1.append(eigenvectors[i][0])
        eigen2.append(eigenvectors[i][1])

    min_value = min(eigen1)
    max_value = max(eigen1)

    for value in eigen1:
        mapped_value = (value - min_value) / (max_value - min_value)
        color = get_color_scale(mapped_value)
        edge_colors.append(color)

    nx.draw(G, node_size=5, edge_color = edge_colors)
    plt.show()

    edge_colors = []

    min_value = min(eigen2)
    max_value = max(eigen2)

    for value in eigen2:
        mapped_value = (value - min_value) / (max_value - min_value)
        color = get_color_scale(mapped_value)
        edge_colors.append(color)

    nx.draw(G, node_size=5, edge_color = edge_colors)
    plt.show()



def construct_graph_from_adjacency(A):
    G = nx.from_numpy_matrix(np.array(A))
    edge_to_tuple = {}
    tuple_to_edge = {}

    for idx, edge in enumerate(G.edges()):
        edge_to_tuple[idx] = edge
        tuple_to_edge[edge] = idx

    G.graph['edge_to_tuple'] = edge_to_tuple
    G.graph['tuple_to_edge'] = tuple_to_edge
    G.graph['triangles'] = [tri for tri in list(nx.find_cliques(G)) if len(tri) == 3]
    
    return G

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
    # assert np.all(np.sum(np.abs(B2), axis=-1) > 0)
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

def process_eigen(k_smallest_eigen):
    k = len(k_smallest_eigen)

    eigenvectors = []
    for i in range(len(k_smallest_eigen[0])):
        eigenvectors.append([k_smallest_eigen[j][i] for j in range(k)])

    return eigenvectors

def positional_encoding(g, pos_enc_dim):
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()

    return g

class ZINC_Dataset(DGLDataset):
    def __init__(self):
        super().__init__(name='ZINC')

    def process(self):
        dataset, self.mask = load_data()
        self.graphs = []
        self.logP_SA_cycle_normalized = []
        self.G = []

        for data in dataset:
            A = data['bond_type']
            G = construct_graph_from_adjacency(A)

            B1, B2 = extract_boundary_matrices(G)
            L1 = get_hodge_laplacian(B1, B2)

            k_smallest_eigen = get_smallest_k_eigenvectors(L1, 5)

            eigenvectors = np.array(process_eigen(k_smallest_eigen))
            eigenvectors = torch.from_numpy(eigenvectors)

            atom_type = torch.from_numpy(np.array(data['atom_type']))
            bond_types = torch.from_numpy(np.array([[edge[2]['weight']] for edge in G.edges(data=True)]))
            logP_SA_cycle_normalized = data['logP_SA_cycle_normalized']

            src = np.array([nodes[0] for nodes in G.graph['tuple_to_edge']])
            dst = np.array([nodes[1] for nodes in G.graph['tuple_to_edge']])
            num_nodes = G.number_of_nodes()

            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g.ndata['atom_type'] = atom_type
            g.edata['bond_type'] = bond_types.float()
            g.edata['hodge_eig'] = eigenvectors

            g = positional_encoding(g, 3)
            g.ndata['eig'] = g.ndata['pos_enc']

            self.graphs.append(g)
            self.logP_SA_cycle_normalized.append(logP_SA_cycle_normalized)
            self.G.append(G)

    def __getitem__(self, i):
        return self.graphs[i], self.logP_SA_cycle_normalized[i], self.G[i], self.mask

    def __len__(self):
        return len(self.graphs)

if __name__ == '__main__':
    dataset = ZINC_Dataset()
