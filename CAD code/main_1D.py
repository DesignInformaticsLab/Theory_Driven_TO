import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import scipy.io as sio
import matlab.engine

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def P(z):
    h1 = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    h2_1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(tf.reshape(h1,[batch_size, width/8, height/8, 1]),
                                                  deconv2_1_weight, strides=[1, 1, 1, 1], padding='SAME',
                                       output_shape=[batch_size, width/8, height/8, deconv2_1_features]),deconv2_1_bias))

    h2_2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h2_1,deconv2_2_weight, strides=[1, 2, 2, 1], padding='SAME',
                                       output_shape=[batch_size, width/4, height/4, deconv2_2_features]),deconv2_2_bias))

    h3_1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h2_2, deconv3_1_weight, strides=[1, 1, 1, 1], padding='SAME',
                                       output_shape=[batch_size, width/4, height/4, deconv3_1_features]),deconv3_1_bias))

    h3_2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h3_1, deconv3_2_weight, strides=[1, 2, 2, 1], padding='SAME',
                                       output_shape=[batch_size, width/2, height/2, deconv3_2_features]),deconv3_2_bias))

    h4_1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h3_2, deconv4_1_weight, strides=[1, 1, 1, 1], padding='SAME',
                                       output_shape=[batch_size, width/2, height/2, deconv4_1_features]),deconv4_1_bias))

    h4_2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(h4_1, deconv4_2_weight, strides=[1, 2, 2, 1], padding='SAME',
                                       output_shape=[batch_size, width/1, height/1, deconv4_2_features]),deconv4_2_bias))

    h5 = (tf.add(tf.nn.conv2d_transpose(h4_2, deconv5_weight, strides=[1, 1, 1, 1], padding='SAME',
                                        output_shape=[batch_size, width/1, height/1, 1]),deconv5_bias))

    prob = tf.nn.sigmoid(h5)
#     prob = 1 / (1 + tf.exp(-h5))

    return prob

# Input parameter
nelx, nely = 12*10, 4*10
nn = nelx*nely
batch_size=5
initial_num=5
Prepared_training_sample = True # True if samples are pre-solved offline

# network parameter
z_dim = 1
width = nely
height = nelx
h_dim = width/8*height/8


deconv2_1_features=32*3
deconv2_2_features=32*3
deconv3_1_features=32*2
deconv3_2_features=32*2
deconv4_1_features=32
deconv4_2_features=32

F_input = tf.placeholder(tf.float32, shape=([batch_size, z_dim]))

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]),name="P_W1")
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]),name="P_b1")

deconv2_1_weight = tf.Variable(tf.truncated_normal([4, 4, deconv2_1_features, 1],
                                               stddev=0.1, dtype=tf.float32))
deconv2_1_bias = tf.Variable(tf.zeros([deconv2_1_features], dtype=tf.float32))

deconv2_2_weight = tf.Variable(tf.truncated_normal([4, 4, deconv2_2_features,deconv2_1_features],
                                               stddev=0.1, dtype=tf.float32))
deconv2_2_bias = tf.Variable(tf.zeros([deconv2_2_features], dtype=tf.float32))

deconv3_1_weight = tf.Variable(tf.truncated_normal([4, 4, deconv3_1_features, deconv2_2_features],
                                               stddev=0.1, dtype=tf.float32))
deconv3_1_bias = tf.Variable(tf.zeros([deconv3_1_features], dtype=tf.float32))

deconv3_2_weight = tf.Variable(tf.truncated_normal([4, 4, deconv3_2_features, deconv3_1_features],
                                               stddev=0.1, dtype=tf.float32))
deconv3_2_bias = tf.Variable(tf.zeros([deconv3_2_features], dtype=tf.float32))

deconv4_1_weight = tf.Variable(tf.truncated_normal([4, 4, deconv4_1_features, deconv3_2_features],
                                               stddev=0.1, dtype=tf.float32))
deconv4_1_bias = tf.Variable(tf.zeros([deconv4_1_features], dtype=tf.float32))

deconv4_2_weight = tf.Variable(tf.truncated_normal([8, 8, deconv4_2_features, deconv4_1_features],
                                               stddev=0.1, dtype=tf.float32))
deconv4_2_bias = tf.Variable(tf.zeros([deconv4_2_features], dtype=tf.float32))

deconv5_weight = tf.Variable(tf.truncated_normal([8, 8, 1, deconv4_2_features],
                                               stddev=0.1, dtype=tf.float32))
deconv5_bias = tf.Variable(tf.zeros([1], dtype=tf.float32))

P_output = P(F_input)

phi_true = tf.transpose(tf.reshape(P_output,[batch_size,nn]))


global_step=tf.Variable(0,trainable=False)
starter_learning_rate=0.0005
learning_rate=tf.train.exponential_decay(starter_learning_rate,global_step,500,1.0, staircase=True)
y_output=tf.placeholder(tf.float32, shape=([nn, batch_size]))
recon_loss = tf.reduce_sum((phi_true-y_output)**2)
solver = tf.train.AdamOptimizer(learning_rate).minimize(recon_loss, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# generating initial points
directory_data='experiment_data/'
directory_model='model_save/'
directory_result='experiment_result/'

# random sampling initial
index_ind=random.sample(range(0, 100), 5)

if Prepared_training_sample==True:
    pass
else:
    if not os.path.exists(directory_result):
        os.makedirs(directory_result)
    sio.savemat('{}/index_ind_1D.mat'.format(directory_result),{'index_ind':index_ind})
    eng = matlab.engine.start_matlab()
    eng.infill_1D(1,nargout=0)

Y_test=sio.loadmat('{}/phi_true_1D.mat'.format(directory_data))['phi_true']


theta= np.linspace(np.pi * 0., np.pi, 100)
force=-1
F_batch= np.zeros([100, z_dim])

for i in range(100):
    # Fx = force * np.sin(theta[i])
    # Fy = force * np.cos(theta[i])
    #
    # F_batch[i, 0] = Fx
    # F_batch[i, 1] = Fy
    F_batch[i,:]=theta[i]

budget=0
final_error=float('inf')
terminate_criteria=1 # can be adjusted
testing_num = 100

while final_error>terminate_criteria:
    print("requirement doesn't match, current final_error={}, keep sampling".format(final_error))
    try:
        add_point_index=sio.loadmat('{}/add_point_index_1D.mat'.format(directory_result))['add_point_index'][0]
        index_ind=list(add_point_index)+index_ind
    except:
        pass

    Y_train = sio.loadmat('{}/phi_true_1D.mat'.format(directory_data))['phi_true']


    for it in range(100000):
        random_ind=np.random.choice(index_ind,batch_size,replace=False)
        # Y_test=sio.loadmat('phi/phi_true_ratio10.mat')['phi_true'][random_ind].T

        _,error=sess.run([solver, recon_loss],feed_dict={y_output:Y_train[random_ind].T,F_input:F_batch[random_ind]})
        if it%5 == 0:
            print('iteration:{}, recon_loss:{}'.format(it,error))

        if error <= 0.05:
            if not os.path.exists(directory_model):
                os.makedirs(directory_model)
            saver=tf.train.Saver()
            saver.save(sess, '{}/model_1D_sample_{}'.format(directory_model,len(index_ind)))
            print('converges, saving the model.....')
            break

    ratio=testing_num/batch_size
    final_error=0
    for it in range(ratio):
        final_error_temp=sess.run(recon_loss,feed_dict={y_output:Y_test[it%ratio*batch_size:it%ratio*batch_size+batch_size].T,
                                                                    F_input:F_batch[it%ratio*batch_size:it%ratio*batch_size+batch_size]})
        final_error=final_error + final_error_temp
    final_error=final_error/testing_num
    print('average testing error is: {}'.format(final_error))

    if final_error<=terminate_criteria:
        break

    F_batch_test= np.zeros([testing_num, z_dim])

    for i in range(testing_num):
        # Fx = force * np.sin(theta[i])
        # Fy = force * np.cos(theta[i])
        #
        # # up-right corner
        # F_batch_test[i, 0] = Fx
        # F_batch_test[i, 1] = Fy
        F_batch_test[i,:]=theta[i]


    # evaluate all points (total 100)
    ratio=testing_num/batch_size
    phi_store=[]
    for it in range(ratio):
        phi_update=sess.run(phi_true,feed_dict={F_input:F_batch_test[it%ratio*batch_size:it%ratio*batch_size+batch_size]})
        phi_store.append(phi_update)


    if not os.path.exists(directory_result):
        os.makedirs(directory_result)
    phi_gen=np.concatenate(phi_store,axis=1).T
    sio.savemat('{}/phi_gen_1D.mat'.format(directory_result),{'phi_gen':phi_gen})
    sio.savemat('{}/random_candidate_1D.mat'.format(directory_result),{'random_candidate':index_ind})

    eng = matlab.engine.start_matlab()
    eng.cal_c_1D(nargout=0)






