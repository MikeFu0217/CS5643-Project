import numpy as np

def ge_square_edges_ijij(n, type='structural'):
    edges = []
    
    if type == 'structural':
        for i in range(n - 1):
            for j in range(n - 1):
                # (i, j) -> (i, j+1)
                edges.append([i, j, i, j+1])
                # (i, j+1) -> (i+1, j+1)
                edges.append([i, j+1, i+1, j+1])
                # (i+1, j) -> (i+1, j+1)
                edges.append([i+1, j, i+1, j+1])
                # (i, j) -> (i+1, j)
                edges.append([i, j, i+1, j])
    
    elif type == 'shear':
        for i in range(n - 1):
            for j in range(n - 1):
                # (i, j) -> (i+1, j+1)
                edges.append([i, j, i+1, j+1])
                # (i, j+1) -> (i+1, j)
                edges.append([i, j+1, i+1, j])
    
    elif type == 'flextion':
        for i in range(n - 2):
            for j in range(n - 2):
                # Horizontal (i, j) -> (i+2, j)
                edges.append([i, j, i+2, j])
                # Vertical (i, j) -> (i, j+2)
                edges.append([i, j, i, j+2])
    
    else:
        raise ValueError("Unknown type")
    
    # Convert the list of edges to a numpy array of shape (num_edges, 4)
    return np.array(edges, dtype=np.int32)