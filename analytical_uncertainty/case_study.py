#####################################################################################
# Case study using 1D topology optimization
# x: load orientation, location is fixed
# y: optimal topology
# y_hat: predicted topology
# z: optimal compliance
# z_l: lower bound on compliance prediction
# z_u: upper bound on compliance prediction
#####################################################################################

import numpy as np
import matlab.engine
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from neural_network import NeuralNetwork
from pathlib import Path


def main():
    # ground truth
    num_plot_data = 100
    X_all = np.linspace(-np.PI/2., np.PI/2., num_plot_data)
    Z_all, Y_all = get_ground_truth(X)

    # sample the space
    num_data = 10
    sample = range(0, num_plot_data, step=num_plot_data/num_data)
    X = X_all[sample,:]

    # get optimal compliance
    Y = Y_all[sample,:]
    Z = Z_all[sample,:]

    # create Gaussian Process model
    GP = gaussian_process(X,Z)

    # plot GP prediction and uncertainty

    # compute GP uncertainty
    z_prediction, uncertainty_GP = compute_GP_uncertainty(GP, plot_x)
    plot_uncertainty(plot_x, z_prediction, uncertainty_GP)

    # learn map from x to y
    nn = learn_neural_network(X,Y)

    # compute compliance prediction
    Y_hat = nn.predict(plot_x)
    Z_hat = [eng.calculate_compliance(y, nargout=0) for y in Y_hat]
    # compute deviation from optimality
    h = [eng.calculate_optimality(y, nargout=0) for y in Y_hat]

    # compute analytical uncertainty
    # uncertainty_analytical = compute_analytical_uncertainty(nn, plot_x)
    # plot_uncertainty(plot_x, NN, uncertainty_analytical)

    # plot true and predicted compliance, and deviation from optimality
    plt.plot(x_range, Z)
    plt.plot(x_range, Z_hat)
    plt.plot(x_range, h)

    plt.show()


def infill_topology_optimization(x):
    return eng.infill_topology_optimization(x, nargout=0)


def calculate_compliance(x, y):
    compliance, local_violation, global_violation = eng.calculate_compliance(x, y)
    return compliance


def gaussian_process(x, z):
    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(x, z)
    return gp


def compute_GP_uncertainty(gp, x):
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    z_pred, sigma = gp.predict(x, return_std=True)
    return z_pred, sigma


def learn_neural_network(x, y):
    input_dim = 1
    output_dim = mesh_size
    nn = NeuralNetwork(input_dim, output_dim)
    nn.fit(x, y)
    return nn


def compute_analytical_uncertainty(nn, x_range):
    Y_hat = nn.predict(x_range)
    Z_l = [eng.calculate_compliance(y, nargout=0) for y in Y_hat]


def plot_uncertainty(x, mean, std):
    plt.plot(x, mean, 'k-')
    plt.fill_between(x, mean - std, mean + std)
    plt.show()


def get_ground_truth(X):
    directory_data = '../CAD code/experiment_data'
    try:
        data = sio.loadmat('{}/1d_case.mat'.format(directory_data))
        Z = data['compliance']
        Y = data['topology']
    except FileNotFoundError:
        print('generating ground truth data...\n')
        Y = []
        Z = []
        for i, x in enumerate(X):
            z, y, _ = infill_topology_optimization(x) #TODO: generalize topology optimization inputs
            Y.append(y)
            Z.append(z)
            print('{}/{}\n'.format(i, len(X)))
        print('saving data...\n')
        sio.savemat('{}/1d_case.mat'.format(directory_data),{'topology':Y,'compliance':Z})
        print('done.\n')
    return Z, Y

np.random.seed(1)

# use matlab engine
eng = matlab.engine.start_matlab()

# problem specifications
nelx, nely = 12 * 10, 4 * 10
mesh_size = nelx * nely

main()