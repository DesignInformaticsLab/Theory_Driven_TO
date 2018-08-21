import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import random
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
batch_size=10
initial_num=100
Prepared_training_sample = True # True if samples are pre-solved offline

# network parameter
z_dim = 41*41*2
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
learning_rate=tf.train.exponential_decay(starter_learning_rate,global_step,10000,1.0,staircase=True)
y_output=tf.placeholder(tf.float32, shape=([nn, batch_size]))
recon_loss = tf.reduce_sum((phi_true-y_output)**2)/batch_size
solver = tf.train.AdamOptimizer(learning_rate).minimize(recon_loss, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# generating initial points
directory_data='experiment_data/'
directory_model='model_save/'
directory_result='experiment_result/'


LHS = sio.loadmat('{}/LHS_train.mat'.format(directory_data))['LHS_train'] # pre-sampling the loading condition offline

LHS[:,0] = LHS[:,0]-81
LHS[:,1] = LHS[:,1]-1

LHS_x=np.int32(LHS[:,0])
LHS_y=np.int32(LHS[:,1])
LHS_z=LHS[:,2]

force=-1
F_batch = np.zeros([len(LHS), z_dim])
error_store=[]
for i in range(len(LHS)):
    Fx = force * np.sin(LHS_z[i])
    Fy = force * np.cos(LHS_z[i])
    F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-1]=Fy
    F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-2]=Fx


index_ind = random.sample(range(0,len(LHS)),initial_num) # initial start with 100, can be modified

if Prepared_training_sample==True:
    pass
else:
    if not os.path.exists(directory_result):
        os.makedirs(directory_result)
    sio.savemat('{}/index_ind.mat'.format(directory_result),{'index_ind':index_ind})
    eng = matlab.engine.start_matlab()
    eng.infill_high_dim(1,nargout=0)

Y_test = sio.loadmat('{}/phi_true_test2.mat'.format(directory_data))['phi_true_test'] # prepared off-line
LHS_test=sio.loadmat('{}/LHS_test2.mat'.format(directory_data))['LHS_test']

LHS_test[:,0] = LHS_test[:,0]-81
LHS_test[:,1] = LHS_test[:,1]-1

LHS_x_test=np.int32(LHS_test[:,0])
LHS_y_test=np.int32(LHS_test[:,1])
LHS_z_test=LHS_test[:,2]

force=-1
F_batch_test = np.zeros([len(LHS_test), z_dim])
for i in range(len(LHS_test)):
    Fx_test = force * np.sin(LHS_z_test[i])
    Fy_test = force * np.cos(LHS_z_test[i])
    F_batch_test[i,2*((nely+1)*LHS_x_test[i]+LHS_y_test[i]+1)-1]=Fy_test
    F_batch_test[i,2*((nely+1)*LHS_x_test[i]+LHS_y_test[i]+1)-2]=Fx_test


budget=0
final_error=float('inf')
terminate_criteria=1

# one-shot algorithm
while final_error>terminate_criteria:
    print("requirement doesn't match, current final_error={}, keep sampling".format(final_error))
    try:
        add_point_index=sio.loadmat('{}/add_point_index.mat'.format(directory_result))['add_point_index'][0]
        index_ind=list(add_point_index)+index_ind
    except:
        pass

    Y_train = sio.loadmat('{}/phi_true_train.mat'.format(directory_data))['phi_true_train']

    F_batch = np.zeros([len(LHS), z_dim])
    for i in range(len(LHS)):
        F_batch[i,0]=LHS_x[i]
        F_batch[i,1]=LHS_y[i]
        F_batch[i,2]=LHS_z[i]

    force=-1
    F_batch = np.zeros([len(LHS), z_dim])
    for i in range(len(LHS)):
        Fx = force * np.sin(LHS_z[i])
        Fy = force * np.cos(LHS_z[i])
        F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-1]=Fy
        F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-2]=Fx

    for it in range(1000000):
        random_ind=np.random.choice(index_ind,batch_size,replace=False)

        _,error=sess.run([solver, recon_loss],feed_dict={y_output:Y_train[random_ind].T,F_input:F_batch[random_ind]})
        if it%100 == 0:
            print('iteration:{}, recon_loss:{}, num:{}'.format(it,error,len(index_ind)))
        if error <= 1:
            if not os.path.exists(directory_model):
                os.makedirs(directory_model)
            saver=tf.train.Saver()
            saver.save(sess, '{}/model_sample_{}'.format(directory_model,len(index_ind)))
            print('converges, saving the model.....')
            break
    # print('converges at')
    testing_num=len(LHS_test)
    ratio=testing_num/batch_size
    final_error=0
    for it in range(ratio):
        final_error_temp=sess.run(recon_loss,feed_dict={y_output:Y_test[it%ratio*batch_size:it%ratio*batch_size+batch_size].T,
                                                                    F_input:F_batch_test[it%ratio*batch_size:it%ratio*batch_size+batch_size]})
        final_error=final_error + final_error_temp

    final_error=final_error/testing_num
    print('current final predicting error: {}/{}'.format(final_error,terminate_criteria))
    if final_error<=terminate_criteria:
        break

    # random generation
    candidate_pool=list(set(list(np.int32(np.linspace(0,len(Y_train)-1,len(Y_train)))))-set(index_ind))
    random_candidate=np.random.choice(candidate_pool,100,replace=False)

    LHS_candidate = sio.loadmat('{}/LHS_train.mat'.format(directory_data))['LHS_train'][random_candidate]
    LHS_candidate[:,0] = LHS_candidate[:,0]-81
    LHS_candidate[:,1] = LHS_candidate[:,1]-1

    LHS_x_candidate=np.int32(LHS_candidate[:,0])
    LHS_y_candidate=np.int32(LHS_candidate[:,1])
    LHS_z_candidate=LHS_candidate[:,2]

    valid_num = len(LHS_candidate)
    # F_batch_candidate = np.zeros([valid_num , z_dim])

    # error_store=[]
    # for i in range(len(LHS_candidate)):
    #     F_batch_candidate[i,0]=LHS_x_candidate[i]
    #     F_batch_candidate[i,1]=LHS_y_candidate[i]
    #     F_batch_candidate[i,2]=LHS_z_candidate[i]


    force=-1
    F_batch_candidate= np.zeros([len(LHS_candidate), z_dim])
    for i in range(len(LHS_candidate)):
        Fx = force * np.sin(LHS_z[i])
        Fy = force * np.cos(LHS_z[i])
        F_batch_candidate[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-1]=Fy
        F_batch_candidate[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-2]=Fx

    rho_gen=[]
    phi_store=[]
    ratio=valid_num /batch_size
    for it in range(ratio):
        phi_update=sess.run(phi_true,feed_dict={F_input:F_batch_candidate[it%ratio*batch_size:it%ratio*batch_size+batch_size]})
        phi_store.append(phi_update)

    if not os.path.exists(directory_result):
        os.makedirs(directory_result)
    phi_gen=np.concatenate(phi_store,axis=1).T
    sio.savemat('{}/phi_gen.mat'.format(directory_result),{'phi_gen':phi_gen})
    sio.savemat('{}/random_candidate.mat'.format(directory_result),{'random_candidate':random_candidate})

    # evaluate the random samples and pick the worst one
    eng = matlab.engine.start_matlab()
    eng.cal_c_high_dim(nargout=0)

    if Prepared_training_sample==False:
        budget=np.sum(sio.loadmat('{}/budget_store.mat')['budget_store'].reshape([-1]))+budget+100

    # solve the worst one
    if Prepared_training_sample == False:
        eng = matlab.engine.start_matlab()
        eng.infill_high_dim(0,nargout=0)

