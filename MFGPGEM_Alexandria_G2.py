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
# In this code the second part is implemented for the Alexandria data set using the
# G2 values as high fidelity. There is data for various other methods that can be used in 
# lieu of the G2 data.
# 
# The embeddings are loaded from a file, along with reduced dimensional equivalents of
# optional descriptors
#
# See the separate code for generating the embeddings
#
# You may freely use/modify the code
# =============================================================================

import numpy as np
np.bool = bool # dirty fix for a bug in my environment - you probably don't need this

'''
The polarisability and dipole moment calculations used different methods from
the other quantities so we distinguish
'''
quantities_1 = ['polar','dipole']
quantities_2 = ['s0','cv','enthalpy']

'''
choose qunatity to predict from enthalpy of formation \Delta_f H^0: kJ/mol, (index 1 in data file)
standard entropy S^0 J/(mol K), index 2
specific heat capacity at constant volume C_V J/(mol K), index 3
'''
quantity_predict = 'cv'

if quantity_predict in quantities_1:
    low_fidelity = 'HF' # Hartree Fock
    high_fidelity = 'DFT' # Density functional theory
    if quantity_predict == 'polar':
        indx = 2
    else:
        indx = 1
else:
    low_fidelity = 'CBS-QB3' # another composite method
    high_fidelity = 'G2' # there is also G4 and experimental data (can get this from the full data set and replace the G2 data)
    if quantity_predict == 'enthalpy':
        indx = 1
    elif quantity_predict == 's0':
        indx = 2
    else:
        indx = 3

cov_choice = 4 # 1 = matern 5, 2 = rbf, 3 = rbf+linear, 4 = matern 5 + linear

input_reduce = 'kpca' # choose dimension reduction method from pca, kpca or sparse pca (spca)
descriptor = 'maccs' # optionally add additional descriptor (maccs or estate)

add_descriptor = ['maccs','estate']

'''
Load the high and low fidelity data according to the selected quantity
'''

if quantity_predict in quantities_1:
    data_high_all = np.loadtxt('/Users/akeelshah/Dropbox/Matlab Codes/GPAutoregForMutiFidelity/AlexandriaLib 2/tables/Dipole_polar_B3.txt')
    data_low_all = np.loadtxt('/Users/akeelshah/Dropbox/Matlab Codes/GPAutoregForMutiFidelity/AlexandriaLib 2/tables/Dipole_polar_HF.txt')
    high_tmp = data_high_all[:,indx]
    low_tmp = data_low_all[:,indx]
    with open('/Users/akeelshah/Dropbox/Matlab Codes/GPAutoregForMutiFidelity/AlexandriaLib 2/tables/InChi.txt', 'r') as file:
        inchi_strings = file.readlines()
elif quantity_predict in quantities_2:
    data_high_all = np.loadtxt('/Users/akeelshah/Dropbox/Matlab Codes/GPAutoregForMutiFidelity/AlexandriaLib 2/tables/Properties_G2.txt')
    data_low_all = np.loadtxt('/Users/akeelshah/Dropbox/Matlab Codes/GPAutoregForMutiFidelity/AlexandriaLib 2/tables/Properties_CBSQB3.txt')
    high_tmp = data_high_all[:,indx]
    low_tmp = data_low_all[:,indx]
    with open('/Users/akeelshah/Dropbox/Matlab Codes/GPAutoregForMutiFidelity/AlexandriaLib 2/tables/InChi_CBSQB3.txt', 'r') as file:
        inchi_strings = file.readlines()
        
from rdkit import Chem
smiles_list = []

inn = 0
data_low = []
data_high = []

# The graph embeddings were performed using a SMILEs list. First Inchi were converted to SMILEs
# using RDKit (the Alexandria data doesn't contain SMILEs, only Inchi.) 
# For those molecules (just one or two) that cannot be converted, remove the corresponding data
# otherwise there is no corresponing graph embedding (input). In hindsight, there is 
# probably a way to create the graph embeddings directly with Inchi

for Inch in inchi_strings:

    mol = Chem.MolFromInchi(Inch)
    
    if mol:
        # Convert Mol to SMILES
        smiles = Chem.MolToSmiles(mol)
        print("SMILES:", smiles)
        smiles_list.append(smiles)
        data_low.append(low_tmp[inn])
        data_high.append(high_tmp[inn])
    else:
        print("Invalid StdInChI")
    inn = inn + 1
    
data_low  = np.array(data_low) 
data_high  = np.array(data_high) 

# Load the data according the quantity selected
if quantity_predict in quantities_1:
    # The diffusion map based descriptor (see separate code for generating this)
    input_matrix_graph = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/input_matrix_graph.csv",delimiter=",", skip_header=False)
    # The redcuced dimensional versions of the additional descriptor
    if input_reduce == 'pca':
        estateI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/Estate_pca_50.csv",delimiter=",", skip_header=False)    
        maccsI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/MACCs_pca_50.csv",delimiter=",", skip_header=False)
    elif input_reduce == 'spca':
        estateI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/Estate_spca_25.csv",delimiter=",", skip_header=False)    
        maccsI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/MACCs_spca_25.csv",delimiter=",", skip_header=False)
    else:
        estateI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/Estate_kpca_25.csv",delimiter=",", skip_header=False)    
        maccsI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/MACCs_kpca_25.csv",delimiter=",", skip_header=False)
elif quantity_predict in quantities_2:
    input_matrix_graph = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/CBS/input_matrix_graph.csv",delimiter=",", skip_header=False)
    if input_reduce == 'pca':
        estateI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/CBS/Estate_pca_100.csv",delimiter=",", skip_header=False)    
        maccsI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/CBS/MACCs_pca_100.csv",delimiter=",", skip_header=False)
    elif input_reduce == 'spca':
        estateI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/CBS/Estate_spca_50.csv",delimiter=",", skip_header=False)    
        maccsI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/CBS/MACCs_spca_50.csv",delimiter=",", skip_header=False)
    else:
        estateI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/CBS/Estate_kpca_50.csv",delimiter=",", skip_header=False)    
        maccsI = np.genfromtxt("/Users/akeelshah/Dropbox/Python Codes Main Folder/qeCalculations main/Machine learning inputs/Alex/CBS/MACCs_kpca_50.csv",delimiter=",", skip_header=False)

if descriptor == 'maccs':
    X_tmp = maccsI
elif descriptor == 'estate':
    X_tmp = estateI

input_matrix = input_matrix_graph

targets = data_high

if descriptor in add_descriptor:   
    [N_samps,dim_tmp]=X_tmp.shape
    [N_samps,dim] = input_matrix.shape
    input_matrix_gpnar = np.zeros((N_samps, dim + dim_tmp + 1))     
    input_matrix_gpnar[:, :dim] = input_matrix
    input_matrix_gpnar[:, dim:dim+dim_tmp] = X_tmp
    # Add the data from Y1_all as the 26th column
    input_matrix_gpnar[:, dim + dim_tmp] = data_low.squeeze()  # Squeeze Y1_all to make it a 1D array before
    dim = dim + dim_tmp + 1
else:   
    [N_samps,dim] = input_matrix.shape
    input_matrix_gpnar = np.zeros((N_samps, dim + 1))     
    input_matrix_gpnar[:, :dim] = input_matrix
    # Add the data from Y1_all as the 26th column
    input_matrix_gpnar[:, dim] = data_low.squeeze()  # Squeeze Y1_all to make it a 1D array before
    dim = dim + 1

import numpy as np
import GPy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

input_matrix_gpnar = input_matrix_gpnar / np.linalg.norm(input_matrix_gpnar)
normY = np.linalg.norm(targets)
targets = targets / normY

in_t = 0
train_numbers = [16,32,64,128,256] # change as required
error_values = np.zeros([len(train_numbers),3]) # save error values in arrays for different training #
np.float = float 

# Generate results for different training numbers
for train_number in train_numbers:
    # Split the data into training and test sets using train_test_split in python
    train_input, test_input, train_targets, test_targets = train_test_split(input_matrix_gpnar, targets, train_size=train_number, random_state=42)
    
    # Choose the kernel for the GP 
    if cov_choice == 1:
        kernel = GPy.kern.Matern52(train_input.shape[1],ARD=True) + GPy.kern.White(train_input.shape[1])
    elif cov_choice == 2:
        kernel = GPy.kern.RBF(train_input.shape[1],ARD=True) + GPy.kern.White(train_input.shape[1])
    elif cov_choice == 3:
        kernel = GPy.kern.RBF(train_input.shape[1],ARD=True) + GPy.kern.Linear(dim, ARD=True) + GPy.kern.White(dim) + GPy.kern.Bias(dim)
    elif cov_choice == 4:
        kernel = GPy.kern.Matern52(dim,ARD=True) + GPy.kern.Linear(dim, ARD=True) +  GPy.kern.White(dim) + GPy.kern.Bias(dim)
    
    # Create the GP model
    model = GPy.models.GPRegression(train_input, train_targets[:, None], kernel)
    
    # Optimize the model
    model.optimize_restarts(num_restarts=10, verbose=True)
    
    # Make predictions on the test data
    mean, var = model.predict(test_input)
    
    mean = mean.ravel()*normY
    var = var.flatten()*normY*normY
    test_targets = test_targets*normY
    
    absolute_errors = np.abs(test_targets - mean)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_targets, mean))
    mae = mean_absolute_error(test_targets, mean)
    r_squared = r2_score(test_targets, mean)
    
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R-squared:", r_squared)
    
    # create arrays fro the errors at different training point number
    error_values[in_t,0] = rmse
    error_values[in_t,1] = mae
    error_values[in_t,2] = r_squared
    
    # plotting - scatterplots
    import matplotlib.pyplot as plt
    import os 
    font_size = 19
    plt.figure(figsize=(6, 5.15))
    
    if quantity_predict == 'polar':
        plt.scatter(test_targets, mean, color='blue', marker='o', label='True vs. Predicted')
        label = r'MFGP-GEM, $\alpha$, $\mathrm{%s}/\mathrm{%s}$' % (high_fidelity, low_fidelity)
        label2 = r'True ($\mathrm{\AA}^3$), train # = ${%s}$' % (train_number)
        plt.title(label, fontsize=font_size)
        plt.xlabel(label2, fontsize=font_size)
        plt.ylabel(r'Predicted ($\mathrm{\AA}^3$)', fontsize=font_size)
        custom_y_ticks = [0,10,20,30,40] 
        plt.yticks(custom_y_ticks,fontsize=font_size)
        plt.xticks(custom_y_ticks,fontsize=font_size)
        plt.text(19,5,f'$R^2$ = {r_squared:.3f}', fontsize=font_size)
        plt.text(17,1.5,f'MAE = {mae:.3f} Å³', fontsize=font_size)
        plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], linestyle='--', color='red', linewidth=2, label='Perfect Prediction')
    elif quantity_predict == 'enthalpy':
        plt.scatter(test_targets*1.e-3, mean*1.e-3, color='blue', marker='o', label='True vs. Predicted')
        label = r'MFGP-GEM, $\Delta H_f^0$, $\mathrm{%s}/\mathrm{%s}$' % (high_fidelity, low_fidelity)
        plt.title(label, fontsize=font_size)
        plt.xlabel(f'True (MJ/mol), train # = {train_number}', fontsize=font_size)
        plt.ylabel(r'Predicted (MJ/mol)', fontsize=font_size)
        plt.text(2,-1,f'$R^2$ = {r_squared:.3f}', fontsize=font_size)
        custom_y_ticks = [-2,0,2,4,6] 
        plt.yticks(custom_y_ticks,fontsize=font_size)
        plt.xticks(custom_y_ticks,fontsize=font_size)
        plt.plot([min(test_targets*1.e-3), max(test_targets*1.e-3)], [min(test_targets*1.e-3), max(test_targets*1.e-3)], linestyle='--', color='red', linewidth=2, label='Perfect Prediction')
    elif quantity_predict == 's0':
        plt.scatter(test_targets*1.e-3, mean*1.e-3, color='blue', marker='o', label='True vs. Predicted')
        label = r'MFGP-GEM, $S^0$, $\mathrm{%s}/\mathrm{%s}$' % (high_fidelity, low_fidelity)
        plt.title(label, fontsize=font_size)
        plt.xlabel(f'True (kJ/mol/K), train # = {train_number}', fontsize=font_size)
        plt.ylabel(r'Predicted (kJ/mol/K)', fontsize=font_size)
        plt.text(0.3,0.2,f'$R^2$ = {r_squared:.3f}', fontsize=font_size)
        plt.text(0.25,0.15,f'MAE = {mae:.3f} J/mol/K', fontsize=font_size)
        custom_y_ticks = [0.1,0.3,0.5,0.7] 
        plt.yticks(custom_y_ticks,fontsize=font_size)
        plt.xticks(custom_y_ticks,fontsize=font_size)
        plt.plot([min(test_targets*1.e-3), max(test_targets*1.e-3)], [min(test_targets*1.e-3), max(test_targets*1.e-3)], linestyle='--', color='red', linewidth=2, label='Perfect Prediction')
    elif quantity_predict == 'cv':
        plt.scatter(test_targets, mean, color='blue', marker='o', label='True vs. Predicted')
        label = r'MFGP-GEM, $C_V$, $\mathrm{%s}/\mathrm{%s}$' % (high_fidelity, low_fidelity)
        plt.title(label, fontsize=font_size)
        plt.xlabel(f'True (J/mol/K), train # = {train_number}', fontsize=font_size)
        plt.ylabel(r'Predicted (J/mol/K)', fontsize=font_size)
        plt.text(100,50,f'$R^2$ = {r_squared:.3f}', fontsize=font_size)
        plt.text(100,25,f'MAE = {mae:.3f} J/mol/K', fontsize=font_size)
        custom_y_ticks = [0,60,120,180,240,300] 
        plt.yticks(custom_y_ticks,fontsize=font_size)
        plt.xticks(custom_y_ticks,fontsize=font_size)
        plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], linestyle='--', color='red', linewidth=2, label='Perfect Prediction')
    elif quantity_predict == 'dipole':        
        plt.scatter(test_targets, mean, color='blue', marker='o', label='True vs. Predicted')
        label = r'MFGP-GEM, $\mu$, $\mathrm{%s}/\mathrm{%s}$' % (high_fidelity, low_fidelity)
        label2 = f'True (D), train # = {train_number}'
        plt.title(label, fontsize=font_size)
        plt.xlabel(label2, fontsize=font_size)
        plt.ylabel('Predicted (D)', fontsize=font_size)
        # plt.text(1,10,f'$R^2$ = {r_squared:.3f}', fontsize=font_size)
        plt.text(7,2,f'$R^2$ = {r_squared:.3f}', fontsize=font_size)
        plt.text(7,0.5,f'MAE = {mae:.3f}', fontsize=font_size)
        custom_y_ticks = [0,3,6,9,12,15] 
        plt.yticks(custom_y_ticks,fontsize=font_size)
        plt.xticks(custom_y_ticks,fontsize=font_size)
        plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], linestyle='--', color='red', linewidth=2, label='Perfect Prediction')
    # Add the custom legend entry to the legend
    plt.grid(True)
    
    # Add a legend
    plt.legend(fontsize=font_size)
    
    # save_directory = '/Plots/predictions/'
    # filename = os.path.join(save_directory,f'Alex_{quantity_predict}_{high_fidelity}_{low_fidelity}_{train_number}_predictions_G2_MFGPGEM.eps')
    # plt.savefig(filename, format='eps', dpi=300,bbox_inches='tight')
    
    plt.show()
    in_t = in_t + 1