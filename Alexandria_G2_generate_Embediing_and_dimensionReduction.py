# =============================================================================
# Author: Akeel A Shah
#
# Code based on the paper:
# Rapid high-fidelity quantum simulations using multi-step nonlinear autoregression 
# and graph embeddings", by A.A. Shah, P.K. Leung, W.W. Xing, 
# to appear in npj Computational Materials
#
# There are two parts: generate the embeddings of the molcules using diffusion maps 
# and then run the autoregressive model (with optional additional descriptors)
#
# In this code the first part is implemented for the Alexandria data set using the
# G2 values as high fidelity. There is data for various other methods that can be used in 
# lieu of the G2 data.
#
# For your own data, simply supply the list of SMILEs strings for the molecules (in this example it is done by
# coverting InChi strings since SMILEs were not available for this example - there is probably a way to
# implemenrt the embedding directly using InChi)
# 
# The embeddings along with reduced dimensional equivalents of optional descriptors are
# extracted and saved. They are then loaded into the code for implementing the second part
#
# You may freely use/modify the code
# =============================================================================

from rdkit import Chem
import numpy as np

quantities_1 = ['polar','dipole']
quantities_2 = ['s0','cv','enthalpy']

'''
enthalpy of formation \Delta_f H^0: kJ/mol, index 1
standard entropy S^0 J/(mol K), index 2
specific heat capacity at constant volume C_V J/(mol K), index 3

The embedding depends on the quantity (different molecules were used for quantities_1 vs quantities_2) 
'''
quantity_predict = 'enthalpy'

# Number of eigenvectors to retain in the first embedding (of each molecule separately)
n_desired = 10
# Number of steps in the Markoc chain 
t = 10
# Choose the number of eigenvectors to retain
graph_num_eigenvectors_to_keep = 5

'''
*********************************************************************
'''

# Get the relevant InChi and convert to SMILEs in order to generate the embeddings. 
# There is probabaly a way to do it with InChi directly rather than SMILEs
# I Will look into this when I have time becasue this way can potentially discard some data points due to 
# RDKit coversion errors
#
if quantity_predict in quantities_1:
    with open('/Users/akeelshah/Dropbox/Matlab Codes/GPAutoregForMutiFidelity/AlexandriaLib 2/tables/InChi.txt', 'r') as file:
        inchi_strings = file.readlines()
elif quantity_predict in quantities_2:
    with open('/Users/akeelshah/Dropbox/Matlab Codes/GPAutoregForMutiFidelity/AlexandriaLib 2/tables/InChi_CBSQB3.txt', 'r') as file:
        inchi_strings = file.readlines()
        
smiles_list = []

inn = 0
for Inch in inchi_strings:

    mol = Chem.MolFromInchi(Inch)
    
    if mol:
        # Convert Mol to SMILES
        smiles = Chem.MolToSmiles(mol)
        print("SMILES:", smiles)
        smiles_list.append(smiles)
    else:
        print("Invalid StdInChI")
    inn = inn + 1

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

inn=-1
n_bits = 1024  # The number of bits used in the fingerprint representation
sob_size = n_bits  # Size of the Sum over Bonds fingerprint
estate_size = n_bits  # Size of the E-state fingerprint
# ..... implement other fingerprints as desired

fingerprint_matrix_sob = np.empty((0, n_bits))
fingerprint_matrix_maccs = np.empty((0, 167))
fingerprint_matrix_estate = np.empty((0, n_bits))
fingerprint_matrix_morgan = np.empty((0, n_bits))

for smiles in smiles_list:
    inn=inn+1
    
    # Convert SMILES to RDKit molecule object
    molecule = Chem.MolFromSmiles(smiles)
    
    "Method 1: Calculate MACCS keys"
    maccs_keys = MACCSkeys.GenMACCSKeys(molecule)
    
    # Convert the MACCS keys to a list of integers
    maccs_keys_list = list(maccs_keys.ToBitString())
    
    maccs_keys_array = np.array(maccs_keys_list)
    
    fingerprint_matrix_maccs = np.vstack((fingerprint_matrix_maccs, maccs_keys_array)) 
    
    "Method 1: E-state fingerprint"
    # Calculate the E-state fingerprint
    estate_fingerprint = AllChem.GetHashedAtomPairFingerprint(molecule,nBits=estate_size)
    
    # Convert the E-state fingerprint to a list of integers
    Estate_array = np.zeros((1,), dtype=np.int32)
    AllChem.DataStructs.ConvertToNumpyArray(estate_fingerprint, Estate_array)
    
    fingerprint_matrix_estate = np.vstack((fingerprint_matrix_estate, Estate_array)) 
 
    print("\nMolecule number:", inn)

# =============================================================================
""" Now extract low-dimensional representations of the additional descriptors """
# 
# =============================================================================
from sklearn.decomposition import PCA,  KernelPCA
from sklearn.decomposition import SparsePCA 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

n_components=100 # number of components to save in PCA (change as desired)
# Do the Estate fingerprint first
sparse_fingerprint_matrix2 = fingerprint_matrix_estate
# Apply PCA
pca = PCA(n_components=n_components)  # Specify the desired number of components
reduced_fingerprint_matrix2 = pca.fit_transform(sparse_fingerprint_matrix2)
explained_variance_ratio = sum(pca.explained_variance_ratio_)
# Print the explained variance ratio
print("Explained Variance Ratio Estate:", explained_variance_ratio)

scaler = StandardScaler()
data_standardized2 = scaler.fit_transform(sparse_fingerprint_matrix2)
sparse_pca2 = SparsePCA(n_components=50)  # Specify the desired number of components
reduced_fingerprint_matrix2s = sparse_pca2.fit_transform(data_standardized2)

# Create a kPCA instance
kpca = KernelPCA(n_components=50, kernel='rbf') # Specify the desired number of components
# Fit and transform the data
data_transformed2 = kpca.fit_transform(sparse_fingerprint_matrix2)

# Save the low-dimensional descriptors
csv_filename_rEstate = "Estate_pca_100.csv"
np.savetxt(csv_filename_rEstate, reduced_fingerprint_matrix2, delimiter=',')
csv_filename_rEstates = "Estates_spca_50.csv"
np.savetxt(csv_filename_rEstates, reduced_fingerprint_matrix2s, delimiter=',')
csv_filename_rEstate_kpca = "Estate_kpca_50.csv"
np.savetxt(csv_filename_rEstate_kpca, data_transformed2, delimiter=',')

# Now do the MACCs key
n_components=100 
sparse_fingerprint_matrix3 = fingerprint_matrix_maccs
# Apply PCA
pca = PCA(n_components=n_components)  # Specify the desired number of components
reduced_fingerprint_matrix3 = pca.fit_transform(sparse_fingerprint_matrix3)
explained_variance_ratio = sum(pca.explained_variance_ratio_)
# Print the explained variance ratio
print("Explained Variance Ratio Maccs:", explained_variance_ratio)

scaler = StandardScaler()
data_standardized3 = scaler.fit_transform(sparse_fingerprint_matrix3)
sparse_pca3 = SparsePCA(n_components=50)  # Specify the desired number of components
reduced_fingerprint_matrix3s = sparse_pca3.fit_transform(data_standardized3)
label_encoder = LabelEncoder()

# Apply the LabelEncoder to each column (feature) of the matrix
numerical_matrix = []
for row in sparse_fingerprint_matrix3:
    numerical_row = label_encoder.fit_transform(row)
    numerical_matrix.append(numerical_row)

# Convert the numerical_matrix to a NumPy array
numerical_matrix = np.array(numerical_matrix)

# Create a kPCA instance; fit and transform the data
kpca = KernelPCA(n_components=50, kernel='rbf')
data_transformed3 = kpca.fit_transform(numerical_matrix)

# Save the low-dimensional representations
csv_filename_rMACCs = "/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/CBS/MACCs_pca_100.csv"
np.savetxt(csv_filename_rMACCs, reduced_fingerprint_matrix3, delimiter=',')
csv_filename_rMACCss = "/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/CBS/MACCs_spca_50.csv"
np.savetxt(csv_filename_rMACCss, reduced_fingerprint_matrix3s, delimiter=',')
csv_filename_rMACCs_kpca = "/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/CBS/MACCs_kpca_50.csv"
np.savetxt(csv_filename_rMACCs_kpca, data_transformed3, delimiter=',')


# =============================================================================
""" Now extract the molecular embeddings. This i9s a two step process (see paper) """
# 
# =============================================================================
        
# Initialize a list to store all eigenvector matrices
all_eigenvectors_list = []

# Determine the maximum number of atoms in the molecules
max_atoms = 0
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    if num_atoms > max_atoms:
        max_atoms = num_atoms

# Use bond matrix, connectivity matrix or the Laplacian from the bond matrix to construct an adjacency matrix
#
adjacency_choice = 'bond' 

inn = -1
# Loop through the smiles and calculate eigenvectors
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    inn = inn + 1
    
    if adjacency_choice == 'bond':
        
        # Initialize the bond type matrix with zeros
        bond_type_matrix = [[0 for _ in range(num_atoms)] for _ in range(num_atoms)]       
        # Fill the bond type matrix with bond type information
        bonds = mol.GetBonds()
        for bond in bonds:
            atom_i = bond.GetBeginAtomIdx()
            atom_j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()  # Get the bond type as a number
            bond_type_matrix[atom_i][atom_j] = bond_type
            bond_type_matrix[atom_j][atom_i] = bond_type  # Since the matrix is symmetric
            
        # Create a weighted adjacency matrix
        weighted_adjacency_matrix = np.array(bond_type_matrix)   
        # Create the degree matrix
        degree_matrix = np.diag(weighted_adjacency_matrix.sum(axis=1)) 
        # For numerical stability
        small_number = 0.00001

        identity_matrix = np.identity(degree_matrix.shape[0])
        degree_matrix = degree_matrix + small_number * identity_matrix
        
        # Compute the diffusion matrix
        diffusion_matrix = np.linalg.inv(degree_matrix) @ weighted_adjacency_matrix
    
    elif adjacency_choice == 'connect':
        
        # Generate molecular graph using RDKit
        mol_graph = Chem.rdmolops.GetAdjacencyMatrix(mol)
    
        # Calculate degree matrix D
        degree_vector = np.sum(mol_graph, axis=1)
        D = np.diag(degree_vector)
    
        # Calculate transition probability matrix P
        P = np.dot(np.linalg.inv(D), mol_graph)
    
        # Normalize P to get diffusion matrix
        diffusion_matrix = P / np.sum(P, axis=1, keepdims=True)

    elif adjacency_choice == 'laplacian':
        
        # Initialize the bond type matrix with zeros
        bond_type_matrix = [[0 for _ in range(num_atoms)] for _ in range(num_atoms)]       
        # Fill the bond type matrix with bond type information
        bonds = mol.GetBonds()
        for bond in bonds:
            atom_i = bond.GetBeginAtomIdx()
            atom_j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()  # Get the bond type as a number
            bond_type_matrix[atom_i][atom_j] = bond_type
            bond_type_matrix[atom_j][atom_i] = bond_type  # Since the matrix is symmetric
            
        # Create a weighted adjacency matrix
        weighted_adjacency_matrix = np.array(bond_type_matrix)   
        # Create the degree matrix
        degree_matrix = np.diag(weighted_adjacency_matrix.sum(axis=1))   
        # Compute the Laplacian matrix
        laplacian_matrix = degree_matrix - weighted_adjacency_matrix
        diffusion_matrix = laplacian_matrix # use the laplacian rather than the dission matrix
    
    else:
        print("invaid adjacency matrix")
        
    # Compute eigenvalues and eigenvectors
    eigenvalues, right_eigenvectors = np.linalg.eig(diffusion_matrix)

    # Sort eigenvalues in ascending order
    # sorted_indices = np.argsort(eigenvalues)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    right_eigenvectors = right_eigenvectors[:, sorted_indices]

    # Choose the number of eigenvectors to retain
    num_eigenvectors_to_keep = n_desired
    if num_atoms <  n_desired:
        #print("less than")
        num_eigenvectors_to_keep = diffusion_matrix.shape[1]  
    
    # Select the first n eigenvalues and eigenvectors
    selected_eigenvalues = eigenvalues[:num_eigenvectors_to_keep]
    selected_eigenvalues = [xi ** t for xi in selected_eigenvalues]
    selected_right_eigenvectors = right_eigenvectors[:, :num_eigenvectors_to_keep]
    
    # Calculate left eigenvectors using pseudoinverse of right eigenvectors
    selected_left_eigenvectors = np.linalg.pinv(selected_right_eigenvectors).T  # Transpose here
    
    # Pad eigenvectors to match max_atoms
    padded_right_eigenvectors = np.pad(selected_right_eigenvectors, ((0, max_atoms - selected_right_eigenvectors.shape[0]), (0, 0)))
    padded_left_eigenvectors = np.pad(selected_left_eigenvectors, ((0, max_atoms - selected_left_eigenvectors.shape[0]), (0, 0)))
    
    ndiff = n_desired - num_atoms
    if num_atoms < n_desired:
            padded_right_eigenvectors = np.hstack((padded_right_eigenvectors, np.zeros((max_atoms,ndiff))))
            padded_left_eigenvectors = np.hstack((padded_left_eigenvectors, np.zeros((max_atoms,ndiff))))
            selected_eigenvalues = np.concatenate((selected_eigenvalues, np.zeros(ndiff)))
    
    #eigenvectors_vector = padded_right_eigenvectors.flatten()
    # Concatenate right eigenvectors and left eigenvectors
    eigenvectors_vector = np.concatenate((padded_right_eigenvectors.flatten(), padded_left_eigenvectors.flatten()))
    
    # Concatenate eigenvector vector with selected eigenvalues
    input_for_molecule = np.concatenate((eigenvectors_vector, selected_eigenvalues))

    # Append input for current molecule to the list
    all_eigenvectors_list.append(input_for_molecule)

# Stack all inputs as rows in a matrix
all_eigenvectors_list_real = [arr.real for arr in all_eigenvectors_list]
input_matrix1 = np.vstack(all_eigenvectors_list_real)

""" second part of embedding"""
# Now embed each molecular graph as a node in a single graph

n = input_matrix1.shape[0]  # Number of vectors
sum_squared_distances = 0.0

for i in range(n):
    for j in range(n):
        sum_squared_distances += np.linalg.norm(input_matrix1[i] - input_matrix1[j])**2

squared_distance_mean = sum_squared_distances / (n**2)
scale_parameter = np.sqrt(squared_distance_mean)

K = np.zeros((n, n))  # Initialize the matrix K

for i in range(n):
    for j in range(n):
        distance_squared = np.linalg.norm(input_matrix1[i] - input_matrix1[j])**2
        K[i, j] = np.exp(-distance_squared / (scale_parameter**2)) # Uses a square exponential (others possible)

graph_adjacency_matrix = K   
# Create the degree matrix
graph_degree_matrix = np.diag(K.sum(axis=1))   
# Compute the diffusion matrix
graph_diffusion_matrix = np.linalg.inv(graph_degree_matrix) @ graph_adjacency_matrix
# Compute eigenvalues and eigenvectors
graph_eigenvalues, graph_right_eigenvectors = np.linalg.eig(graph_diffusion_matrix)
graph_eigenvalues = graph_eigenvalues.real

# Sort eigenvalues in ascending order
#sorted_indices = np.argsort(graph_eigenvalues)
sorted_indices = np.argsort(graph_eigenvalues)[::-1]
graph_eigenvalues = graph_eigenvalues[sorted_indices]
graph_eigenvalues = [xi ** t for xi in graph_eigenvalues]
graph_right_eigenvectors = graph_right_eigenvectors[:, sorted_indices]

# The diffusion map embedding is
#
# ψ^t_r(vj) = (γ_1^t r_{j1}, . . . , γ_r^t r_{jr}) 
#
# where r_{ji} is the j-th coordinate of r_i
#

psi=np.zeros((n,graph_num_eigenvectors_to_keep))
for j in range(n):
    for i in range(graph_num_eigenvectors_to_keep):
        psi[j,i]= graph_eigenvalues[i] * graph_right_eigenvectors[i,j]
        
input_matrix_graph = psi

# Save the embeddinbgs
csv_filename = "/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/CBS/input_matrix_graph.csv"
np.savetxt(csv_filename, input_matrix_graph, delimiter=',')
