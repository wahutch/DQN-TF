# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle
import matplotlib as mp
mp.use('Agg')
from matplotlib import pyplot as plt

class DQN_AGENT:
    
    def __init__(self, flags):
        self.flags = flags
        self.sess = tf.Session()
        
        #set seed in numpy/TensorFlow               
        tf.set_random_seed(flags.seed)
        np.random.seed(flags.seed)
        
        self.current_state_pl = tf.placeholder(tf.float32, shape=(None, flags.frame_dim, flags.frame_dim, flags.num_frame))
        self.newstate_pl = tf.placeholder(tf.float32, shape=(None, flags.frame_dim, flags.frame_dim, flags.num_frame))
        self.rewards_pl = tf.placeholder(tf.float32, shape=(None))
        self.action_mask_pl = tf.placeholder(tf.float32, shape=(None, flags.num_action))
        self.finished_pl = tf.placeholder(tf.float32, shape=(None))
        
        self.action_value_network = self.__init_network(self.current_state_pl, 'Qvars')
        self.target_network = tf.stop_gradient(self.__init_network(self.newstate_pl, 'target'))
        
        self.__init_training()
        self.__init_memory()
        self.__init_state()
        
        self.updateTargetNetwork()
        self.input_states = [np.zeros((flags.frame_dim, flags.frame_dim), dtype = 'float32')]*flags.num_frame
        
    
    def __init_network(self, input_frame, collection):
        
        flags = self.flags
        #network layer shapes
        CONV1_SHAPE = (8, 8, flags.num_frame, 32) 
        CONV2_SHAPE = (4, 4, 32, 64)
        CONV3_SHAPE = (3, 3, 64, 64)
        FC1_SHAPE = (512)
        
        with tf.variable_scope(collection):
            with tf.variable_scope('conv1'):
                kernel = tf.get_variable('weights', shape=CONV1_SHAPE,
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
                conv = tf.nn.conv2d(input_frame, kernel, [1,4,4,1], padding = 'SAME')
                biases = tf.get_variable('biases', shape=CONV1_SHAPE[-1],
                                         initializer=tf.random_uniform_initializer(minval=-flags.bias, maxval=flags.bias, dtype=tf.float32))
                hidden1 = tf.nn.relu(tf.nn.bias_add(conv,biases))
    
            with tf.variable_scope('conv2'):
                kernel=tf.get_variable('weights', shape=CONV2_SHAPE,
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform = True))
                conv = tf.nn.conv2d(hidden1, kernel, [1,2,2,1], padding = 'SAME')
                biases = tf.get_variable('biases', shape=CONV2_SHAPE[-1],
                                         initializer=tf.random_uniform_initializer(minval=-flags.bias, maxval=flags.bias, dtype=tf.float32))
                hidden2 = tf.nn.relu(tf.nn.bias_add(conv,biases))
                
            with tf.variable_scope('conv3'):
                kernel=tf.get_variable('weights', shape=CONV3_SHAPE,
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform = True))
                conv = tf.nn.conv2d(hidden2, kernel, [1,1,1,1], padding = 'SAME')
                biases = tf.get_variable('biases', shape=CONV3_SHAPE[-1],
                                         initializer=tf.random_uniform_initializer(minval=-flags.bias, maxval=flags.bias, dtype=tf.float32))
                hidden3 = tf.nn.relu(tf.nn.bias_add(conv,biases))
    
            with tf.variable_scope('fc1'):
                dims = [hidden3.get_shape()[i].value for i in range(1,len(hidden3.get_shape()))]
                reshape_hidden3 = tf.reshape(hidden3, [-1, np.prod(dims)])
                weights = tf.get_variable('weights', shape=[reshape_hidden3.get_shape()[1], FC1_SHAPE],
                                          initializer=tf.contrib.layers.xavier_initializer(uniform = True))
                biases = tf.get_variable('biases', shape=FC1_SHAPE,
                                         initializer=tf.random_uniform_initializer(minval=-flags.bias, maxval=flags.bias, dtype=tf.float32))
                hidden4 = tf.nn.relu(tf.matmul(reshape_hidden3, weights) + biases)
    
            with tf.variable_scope('action_value'):
                weights = tf.get_variable('weights', shape=[hidden4.get_shape()[1], flags.num_action],
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                biases=tf.get_variable('biases', shape=[flags.num_action],
                                       initializer=tf.constant_initializer(value=flags.bias, dtype=tf.float32))
                action_value=tf.matmul(hidden4,weights) + biases

            return action_value
            
    def __init_training(self):
        flags = self.flags
        
        self.greedy_action = tf.argmax(self.action_value_network, dimension=1)
        
        Qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Qvars')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        self.target_network_update = [tvar.assign(qvar) for qvar,tvar in zip(Qvars, target_vars)]
        
        max_one_step = self.rewards_pl + flags.gamma*tf.reduce_max(self.target_network, reduction_indices=[1,])*self.finished_pl
        masked_action_qvals = tf.reduce_sum(self.action_value_network*self.action_mask_pl, reduction_indices=[1,])
        Q_loss = tf.reduce_mean(tf.square(max_one_step - masked_action_qvals))
        
        optimizer = tf.train.RMSPropOptimizer(flags.lr, decay=flags.decay, momentum=flags.momentum, epsilon=flags.rms_denom)
        Qgrads = optimizer.compute_gradients(Q_loss, var_list=Qvars)
        Qgrads = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in Qgrads]
        self.apply_Qgrads = optimizer.apply_gradients(Qgrads)
        
        
    def __init_memory(self):
        flags = self.flags
        
        self.state_buffer = np.empty((flags.buffer_size, flags.frame_dim, flags.frame_dim), dtype='float16')
        self.action_buffer = np.empty(flags.buffer_size, dtype='uint8')
        self.reward_buffer = np.empty(flags.buffer_size, dtype='int16')
        self.finished_buffer = np.empty(flags.buffer_size, dtype='uint8')

        self.buffer_count = 0
        self.buffer_index = 0
        self.current_states = np.empty((flags.batch_size, flags.num_frame, flags.frame_dim, flags.frame_dim), dtype='float32')
        self.new_states = np.empty((flags.batch_size, flags.num_frame, flags.frame_dim, flags.frame_dim), dtype='float32')
        self.sample_inds = np.empty(flags.batch_size, dtype='int32')
        

    def preprocess(self, screen):
        flags = self.flags
        
        y = .2126*screen[:,:,0] + .7152*screen[:,:,1] + .0722*screen[:,:,2]
        y.astype(np.float)
        y = np.imresize(y, flags.frame_dim, flags.frame_dim)
        
  
    def getState(self, index, buffer_count, buffer_index, state_buffer):
        flags = self.flags
        
        index = index % buffer_count
        if index >= flags.num_frame - 1:
          # use faster slicing
          return state_buffer[(index - (flags.num_frame - 1)):(index + 1),:,:]
        else:
          # otherwise normalize indexes and use slower list based access
          indexes = [(index - i) % buffer_count for i in reversed(range(flags.num_frame))]
          return state_buffer[indexes,:,:]

    def __init_state(self):
        flags = self.flags
        
        self.saver = tf.train.Saver()
        if flags.resume:
            self.saver.restore(self.sess, "Qmodel.ckpt")
            fh = open('Qmodel.pkl', 'rb')
            self.epsilon, self.update_num, self.action_num, self.reward_list, \
                        self.running_reward, self.reward_sum, \
                        self.episode_num = pickle.load(fh)
            fh.close()
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.epsilon = flags.eps_init
            self.update_num = 1
            self.action_num = 0
            self.reward_list = []
            self.running_reward = -21
            self.reward_sum = 0
            self.episode_num = 0
            self.sess.run(init)
            
    def updateTargetNetwork(self):
        self.sess.run(self.target_network_update)
        
    def observeState(self, observation):
        
        self.state = self.preprocess(observation)
        self.input_states.append(self.state.astype('float32'))
        del self.input_states[0]

        self.formatted_input = np.stack(self.input_states, axis = 2)
        self.formatted_input = np.reshape(self.formatted_input, (1,) + self.formatted_input.shape)
        
    def chooseAction(self):
        flags = self.flags
        
        self.action_num+=1
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, flags.num_action)
        else:
            action = self.sess.run(self.greedy_action, feed_dict = {self.current_state_pl:self.formatted_input})[0]
                                   
        return action                         
            
    def annealExplore(self):
        flags = self.flags
        if self.epsilon > flags.eps_final and self.action_num > flags.start_train: 
            self.epsilon -= (flags.eps_init - flags.eps_final)/flags.anneal

    def storeReplay(self, action, reward, done):
        flags = self.flags
        
        self.state_buffer[self.buffer_index,:,:] = self.state
        self.action_buffer[self.buffer_index] = action
        self.reward_buffer[self.buffer_index] = np.clip(reward, -1, 1)
        self.finished_buffer[self.buffer_index] = done
        self.buffer_count = max(self.buffer_count, self.buffer_index + 1)
        self.buffer_index = (self.buffer_index + 1) % flags.buffer_size
        self.reward_sum += reward
        
    def train(self):
        flags = self.flags
        
        if self.action_num % flags.train_int == 0 and self.buffer_count >= flags.start_train:
            self.update_num += 1
            for i in range(flags.batch_size):
                while True:
                    ind = np.random.randint(flags.num_frame, self.buffer_count)
                    if self.finished_buffer[(ind-flags.num_frame):(ind-1)].any():
                        continue
                    if ind >= self.buffer_index and (ind - flags.num_frame) < self.buffer_index:
                        continue
                    break
                self.current_states[i,:,:,:] = self.getState(ind-1, self.buffer_count, self.buffer_index, self.state_buffer)
                self.new_states[i,:,:,:] = self.getState(ind, self.buffer_count, self.buffer_index, self.state_buffer)
                self.sample_inds[i] = ind-1
            
            rewards = self.reward_buffer[self.sample_inds]
            actions = self.action_buffer[self.sample_inds]
            finished = self.finished_buffer[self.sample_inds]
            action_mask = np.zeros((flags.batch_size, flags.num_action))
            action_mask[[range(flags.batch_size),actions]] = 1
    
            feed_dict = {self.current_state_pl:np.transpose(self.current_states,(0,2,3,1)), 
                         self.newstate_pl:np.transpose(self.new_states,(0,2,3,1)), self.rewards_pl:rewards,
                         self.action_mask_pl:action_mask, self.finished_pl:np.logical_not(finished)}
            self.sess.run(self.apply_Qgrads, feed_dict=feed_dict)
    
            if self.update_num % flags.tn_update_freq == 0:
                  self.updateTargetNetwork()
                  
    def recordProgress(self):
        flags = self.flags
        
        self.episode_num += 1
        self.reward_list.append(self.reward_sum)
        self.running_reward = self.running_reward * 0.99 + self.reward_sum * 0.01
        print('episode_num: %d, action_num: %d, epsilon: %2.2f, resetting env. episode reward total was %f. running mean: %f' \
              % (self.episode_num, self.action_num, self.epsilon, self.reward_sum, self.running_reward))
        self.reward_sum = 0
        self.input_states = [np.zeros((flags.frame_dim, flags.frame_dim), dtype = 'float32')]*flags.num_frame #reset frame queue
        plt.figure(0)
        plt.clf()
        plt.plot(self.reward_list)
        plt.xlabel('episodes')
        plt.ylabel('episode reward')
        plt.savefig('Q_learning_performance_%slr_%sbias_large.png' % (flags.lr, flags.bias))

        if self.episode_num % 100 == 0:
            self.saver.save(self.sess, "Qmodel.ckpt")
            fh = open('Qmodel.pkl', 'wb')
            pickle.dump((self.epsilon, self.update_num, self.action_num, self.reward_list, \
                self.running_reward, self.reward_sum, self.episode_num), fh)
            fh.close()        
