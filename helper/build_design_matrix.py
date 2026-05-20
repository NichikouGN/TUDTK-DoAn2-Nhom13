import numpy as np

def build_design_matrix(x):
    n_samples = len(x)
    design_matrix = []

    for i in range(n_samples):
        row = [1] + list(x[i]) # Add intercept term (1) and the original feature
        design_matrix.append(row)

    design_matrix = np.array(design_matrix)
    return design_matrix