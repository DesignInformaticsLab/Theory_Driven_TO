import tensorflow as tf
import numpy as np
import os


# define a fully connected neural network in tensorflow
class NeuralNetwork():
    def __init__(self, input_dim, output_dim):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        self.directory_data = 'experiment_data/'
        self.directory_model = 'model_save/'
        self.directory_result = 'experiment_result/'

        self.batch_size = 50
        self.initial_num = 1000

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)


    def P(self, z):
        h1 = (tf.nn.relu(tf.matmul(z, P_W1) + P_b1))
        h2 = (tf.nn.relu(tf.matmul(h1, P_W2) + P_b2))
        h3 = (tf.nn.relu(tf.matmul(h2, P_W3) + P_b3))
        h4 = (tf.nn.relu(tf.matmul(h3, P_W4) + P_b4))
        h5 = (tf.nn.relu(tf.matmul(h4, P_W5) + P_b5))
        h6 = tf.matmul(h5, P_W6) + P_b6
        prob = tf.nn.sigmoid(h6)
        return prob




    # network parameter
    z_dim = 41 * 41 * 2
    width = nely
    height = nelx
    h_dim = width / 8 * height / 8

    F_input = tf.placeholder(tf.float32, shape=([batch_size, z_dim]))

    P_W1 = tf.Variable(xavier_init([z_dim, 1000]), name="P_W1")
    P_b1 = tf.Variable(tf.zeros(shape=[1000]), name="P_b1")

    P_W2 = tf.Variable(xavier_init([1000, 500]), name="P_W2")
    P_b2 = tf.Variable(tf.zeros(shape=[500]), name="P_b2")

    P_W3 = tf.Variable(xavier_init([500, 100]), name="P_W3")
    P_b3 = tf.Variable(tf.zeros(shape=[100]), name="P_b3")

    P_W4 = tf.Variable(xavier_init([100, 500]), name="P_W4")
    P_b4 = tf.Variable(tf.zeros(shape=[500]), name="P_b4")

    P_W5 = tf.Variable(xavier_init([500, 1000]), name="P_W5")
    P_b5 = tf.Variable(tf.zeros(shape=[1000]), name="P_b5")

    P_W6 = tf.Variable(xavier_init([1000, nn]), name="P_W6")
    P_b6 = tf.Variable(tf.zeros(shape=[nn]), name="P_b6")

    P_output = P(F_input)

    rho_true = tf.transpose(tf.reshape(P_output, [batch_size, nn]))