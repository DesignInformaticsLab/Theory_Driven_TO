import matlab.engine
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import scipy.io as sio
import os
import random
import timeit

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def P(z):
    h1 = (tf.nn.relu(tf.matmul(z, P_W1) + P_b1))
    h2 = (tf.nn.relu(tf.matmul(h1, P_W2) + P_b2))
    h3 = (tf.nn.relu(tf.matmul(h2, P_W3) + P_b3))
    h4 = (tf.nn.relu(tf.matmul(h3, P_W4) + P_b4))
    h5 = (tf.nn.relu(tf.matmul(h4, P_W5) + P_b5))
    h6 = tf.matmul(h5, P_W6) + P_b6
    prob = tf.nn.sigmoid(h6)

    return prob

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
directory_data = 'experiment_data/'
directory_model = 'model_save/'
directory_result = 'experiment_result/'

# Input parameter
nelx, nely = 12*10, 4*10
nn = nelx*nely
batch_size=50
initial_num=1000
Prepared_training_sample = True # True if samples are pre-solved offline

# network parameter
z_dim = 41*41*2
width = nely
height = nelx
h_dim = width/8*height/8


F_input = tf.placeholder(tf.float32, shape=([batch_size, z_dim]))


P_W1 = tf.Variable(xavier_init([z_dim, 1000]),name="P_W1")
P_b1 = tf.Variable(tf.zeros(shape=[1000]),name="P_b1")

P_W2 = tf.Variable(xavier_init([1000, 500]),name="P_W2")
P_b2 = tf.Variable(tf.zeros(shape=[500]),name="P_b2")

P_W3 = tf.Variable(xavier_init([500, 100]),name="P_W3")
P_b3 = tf.Variable(tf.zeros(shape=[100]),name="P_b3")

P_W4 = tf.Variable(xavier_init([100, 500]),name="P_W4")
P_b4 = tf.Variable(tf.zeros(shape=[500]),name="P_b4")

P_W5 = tf.Variable(xavier_init([500, 1000]),name="P_W5")
P_b5 = tf.Variable(tf.zeros(shape=[1000]),name="P_b5")

P_W6 = tf.Variable(xavier_init([1000, nn]),name="P_W6")
P_b6 = tf.Variable(tf.zeros(shape=[nn]),name="P_b6")

P_output = P(F_input)

rho_true = tf.transpose(tf.reshape(P_output,[batch_size,nn]))

for iteration_total in range(0,5):

    global_step=tf.Variable(0,trainable=False)
    starter_learning_rate=0.0005
    learning_rate=tf.train.exponential_decay(starter_learning_rate,global_step,1000,0.96,staircase=True)
    y_output=tf.placeholder(tf.float32, shape=([nn, batch_size]))

    gradient_weight = tf.placeholder(tf.float32, shape=([nn, batch_size]))

    recon_loss = tf.reduce_sum((rho_true - y_output) ** 2 * gradient_weight) / batch_size

    solver = tf.train.AdamOptimizer(learning_rate).minimize(recon_loss, global_step=global_step)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    LHS = sio.loadmat('{}/LHS_train5.mat'.format(directory_data))['LHS_train'] # pre-sampling the loading condition offline

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

    # index_ind = random.sample(range(0,len(LHS)),initial_num) # initial start with 100, can be modified
    # sio.savemat('{}/random_index_used.mat'.format(directory_result), {'index_ind': index_ind})

    #################### sampling starting 1000 data ########################
    count_sample=0
    random_sample_index=random.sample(range(0,len(LHS)),1)
    index_ind=[]
    index_ind.append(random_sample_index[0])
    x_store=[LHS[random_sample_index,0][0]]
    y_store=[LHS[random_sample_index,1][0]]
    z_store=[LHS[random_sample_index,2][0]]

    for index_i in range(len(LHS)):
        i = (index_i + random_sample_index[0])%len(LHS)
        check=float(LHS[i,0] in x_store) + float(LHS[i,1] in y_store) + float(LHS[i,2] in z_store)
        if check<1:
            count_sample=count_sample+1
            x_store.append(LHS[i,0])
            y_store.append(LHS[i,1])
            z_store.append(LHS[i,2])
            index_ind.append(i)
    #
    left_sample = initial_num - len(index_ind)

    while left_sample>0:
        random_sample_index=random.sample(list(set(range(0,len(LHS)))-set(index_ind)),1)
        index_ind.append(random_sample_index[0])
        x_store = [LHS[random_sample_index, 0]]
        y_store = [LHS[random_sample_index, 1]]
        z_store = [LHS[random_sample_index, 2]]

        LHS_update=LHS[list(set(range(0,len(LHS)))-set(index_ind))]

        # for index_i in list(set(range(0,len(LHS)))-set(index_ind)):
        for index_i in range(len(LHS)):
            if index_i in index_ind:
                continue
            i = (index_i + random_sample_index[0]) % len(LHS)
            check = float(LHS[i, 0] in x_store) + float(LHS[i, 1] in y_store) + float(LHS[i, 2] in z_store)
            if check < 1:
                count_sample = count_sample + 1
                x_store.append(LHS[i, 0])
                y_store.append(LHS[i, 1])
                z_store.append(LHS[i, 2])
                index_ind.append(i)

        left_sample = initial_num - len(index_ind)
    index_ind=index_ind[0:initial_num]
    #########################################################################

    if Prepared_training_sample==True: #highly recommend otherwise too slow
        pass
    else:
        if not os.path.exists(directory_result):
            os.makedirs(directory_result)
        sio.savemat('{}/index_ind.mat'.format(directory_result),{'index_ind':index_ind})
        eng = matlab.engine.start_matlab()
        eng.infill_high_dim(1,nargout=0)

    Y_test = sio.loadmat('{}/xPhys_true_test5.mat'.format(directory_data))['xPhys_test'] # prepared off-line
    LHS_test=sio.loadmat('{}/LHS_test5.mat'.format(directory_data))['LHS_test']

    gradient_weight_store = sio.loadmat('{}/dc_drho_store.mat'.format(directory_data))['dc_drho_store'].T
    gradient_weight_store = 1./(gradient_weight_store+1e-2)

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
    candidate_chosen_num=10
    final_error_pre=float('inf')
    budget_store=[]
    # one-shot algorithm
    initial_round=0
    while final_error>terminate_criteria:

        initial_round=initial_round+1

        print("requirement doesn't match, current final_error={}, keep sampling".format(final_error))
        try:
            add_point_index=sio.loadmat('{}/add_point_index.mat'.format(directory_result))['add_point_index'][0]
            index_ind=np.int32(list(add_point_index)+list(index_ind))
            print('go try: training with {} number of samples'.format(len(index_ind)))
        except:
            print('go except: initial training')
            pass

        Y_train = sio.loadmat('{}/xPhys_true_train5.mat'.format(directory_data))['xPhys_true']

        force=-1
        F_batch = np.zeros([len(LHS), z_dim])
        for i in range(len(LHS)):
            Fx = force * np.sin(LHS_z[i])
            Fy = force * np.cos(LHS_z[i])
            F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-1]=Fy
            F_batch[i,2*((nely+1)*LHS_x[i]+LHS_y[i]+1)-2]=Fx

        import time
        start_time=time.time()
        count_overfit=0
        final_error_store=[]
        final_error_pre = float('inf')
        final_error_mean_pre = float('inf')
        for it in range(1,200001):
            # first time random batch, then the new added points will be added every time training
            try:
                random_ind=np.random.choice(index_ind,batch_size-candidate_chosen_num,replace=False)
                random_ind=np.int32(list(random_ind)+list(add_point_index))
            except:
                random_ind = np.random.choice(index_ind, batch_size, replace=False)
                # print()

            _,error=sess.run([solver, recon_loss],feed_dict={y_output:Y_train[random_ind,:].T,
                                                             F_input:F_batch[random_ind,:],
                                                             gradient_weight: gradient_weight_store[:,random_ind]})

            if it%100 == 0:
                print('iteration:{}, recon_loss:{}, num:{}'.format(it,error,len(index_ind)))
            # validation
            if it%2==0:
                if not os.path.exists(directory_model):
                    os.makedirs(directory_model)
                #################################################################################################
                testing_num = len(LHS_test)
                ratio = testing_num / batch_size
                final_error = 0
                phi_test_store=[]
                for iter in range(ratio):
                    final_error_temp,phi_test = sess.run([recon_loss,rho_true], feed_dict={
                        y_output: Y_test[iter % ratio * batch_size:iter % ratio * batch_size + batch_size].T,
                        F_input: F_batch_test[iter % ratio * batch_size:iter % ratio * batch_size + batch_size],
                        gradient_weight: gradient_weight_store[:, random_ind]}) #validation doesn't need penalty
                    final_error = final_error + final_error_temp
                    phi_test_store.append(phi_test)
                final_error = final_error / ratio
                print()
                print('iteration:{}, test_error:{}, count_overfit:{}, num of samples:{}'.format(it, final_error, count_overfit,len(index_ind)))

                final_error_store.append(final_error)
                if len(final_error_store) > 5:
                    del final_error_store[0]

                final_error_mean = np.mean(final_error_store)

                if final_error_mean > final_error_mean_pre and len(final_error_store) == 5:
                    count_overfit = count_overfit + 1

                if final_error_mean > final_error_mean_pre and count_overfit>=2:
                    if not os.path.exists(directory_result):
                        os.makedirs(directory_result)
                    if len(index_ind)%100==0:
                        rho_gen_test = np.concatenate(phi_test_store, axis=1).T
                        sio.savemat('{}/phi_gen_propose_sample{}_case{}_penalty_v1_xphys.mat'.format(directory_result,len(index_ind),iteration_total),
                                    {'rho_gen': rho_gen_test})

                    print()
                    print('iteration:{}, converges test_error:{}'.format(it, final_error))
                    # print()
                    break
                #################################################################################################

                # saver=tf.train.Saver()
                # saver.save(sess, '{}/model_sample_{}_it{}'.format(directory_model,len(index_ind), it))
                # print('converges, saving the model.....')
                # break
            #################################### validation ########################################
                final_error_mean_pre = final_error_mean + 1e-5

        end_time=time.time()
        print('converges at recon error {}, time cost {}'.format(error, end_time-start_time))

        # random generation
        candidate_pool=list(set(list(np.int32(np.linspace(0,len(Y_train)-1,len(Y_train)))))-set(index_ind))
        random_candidate=np.random.choice(candidate_pool,100,replace=False)

        LHS_candidate = sio.loadmat('{}/LHS_train5.mat'.format(directory_data))['LHS_train'][random_candidate,:]
        LHS_candidate[:,0] = LHS_candidate[:,0]-81
        LHS_candidate[:,1] = LHS_candidate[:,1]-1

        LHS_x_candidate=np.int32(LHS_candidate[:,0])
        LHS_y_candidate=np.int32(LHS_candidate[:,1])
        LHS_z_candidate=LHS_candidate[:,2]

        valid_num = len(LHS_candidate)

        force=-1
        F_batch_candidate = np.zeros([len(LHS_candidate), z_dim])
        for i in range(len(LHS_candidate)):
            Fx = force * np.sin(LHS_z_candidate[i])
            Fy = force * np.cos(LHS_z_candidate[i])
            F_batch_candidate[i,2*((nely+1)*LHS_x_candidate[i]+LHS_y_candidate[i]+1)-1]=Fy
            F_batch_candidate[i,2*((nely+1)*LHS_x_candidate[i]+LHS_y_candidate[i]+1)-2]=Fx


        rho_store=[]
        ratio=valid_num /batch_size
        for it in range(ratio):
            rho_update=sess.run(rho_true,feed_dict={F_input:F_batch_candidate[it%ratio*batch_size:it%ratio*batch_size+batch_size],
                                                    })
            rho_store.append(rho_update)

        if not os.path.exists(directory_result):
            os.makedirs(directory_result)
        rho_gen=np.concatenate(rho_store,axis=1).T
        sio.savemat('{}/rho_gen.mat'.format(directory_result),{'rho_gen':rho_gen})
        sio.savemat('{}/random_candidate.mat'.format(directory_result),{'random_candidate':random_candidate})

        ###################################################################

        # evaluate the random samples and pick the worst one
        eng = matlab.engine.start_matlab()
        eng.cal_c_high_dim(nargout=0)

        ####################################################################

        if Prepared_training_sample==False:
            budget=np.sum(sio.loadmat('{}/budget_store.mat')['budget'].reshape([-1]))+budget+100
        else:
            budget=sio.loadmat('{}/budget_store.mat'.format(directory_data))['budget'][index_ind]+100
        budget_store.append(budget)

        # solve the worst one
        if Prepared_training_sample == False:
            eng = matlab.engine.start_matlab()
            eng.infill_high_dim(0,nargout=0)
        sio.savemat('{}/random_index_used.mat'.format(directory_result), {'index_ind': index_ind})

        if len(index_ind)>7000:
            sio.savemat('{}/budget_store_case{}.mat'.format(directory_result,iteration_total), {'budget_store': budget_store})
            break
