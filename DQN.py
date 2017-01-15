# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
import numpy as np
import pickle
import matplotlib as mp
mp.use('Agg')
from matplotlib import pyplot as plt
#import cv2
from skimage.transform import resize
from skimage.color import rgb2gray

class DQN_AGENT:
    
    def __init__(self, flags):
        self.flags = flags
        self._sess = tf.Session()
        
        #set seed in numpy/TensorFlow               
        tf.set_random_seed(flags.seed)
        np.random.seed(flags.seed)
        
        self._current_state_pl = tf.placeholder(tf.float32, shape=(None, flags.frame_dim, flags.frame_dim, flags.num_frame))
        self._newstate_pl = tf.placeholder(tf.float32, shape=(None, flags.frame_dim, flags.frame_dim, flags.num_frame))
        self._rewards_pl = tf.placeholder(tf.float32, shape=(None))
        self._action_mask_pl = tf.placeholder(tf.float32, shape=(None, flags.num_action))
        self._finished_pl = tf.placeholder(tf.float32, shape=(None))
        
        self._action_value_network = self._init_network(self._current_state_pl, 'Qvars')
        self._target_network = tf.stop_gradient(self._init_network(self._newstate_pl, 'target'))
        
        self._init_training()
        self._init_memory()
        self._init_state()
        
        self._updateTargetNetwork()
        self._input_states = [np.zeros((flags.frame_dim, flags.frame_dim), dtype = 'float32')]*flags.num_frame
                             
        self._evalEnv = gym.make(flags.env)
        self._evalEnv.ale.setBool('color_averaging', True)
        
    
    def _init_network(self, input_frame, collection):
        
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
                                         initializer=tf.random_uniform_initializer(minval=0, 
                                                                                   maxval=flags.bias, dtype=tf.float32))
                hidden1 = tf.nn.relu(tf.nn.bias_add(conv,biases))
    
            with tf.variable_scope('conv2'):
                kernel=tf.get_variable('weights', shape=CONV2_SHAPE,
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform = True))
                conv = tf.nn.conv2d(hidden1, kernel, [1,2,2,1], padding = 'SAME')
                biases = tf.get_variable('biases', shape=CONV2_SHAPE[-1],
                                         initializer=tf.random_uniform_initializer(minval=0, 
                                                                                   maxval=flags.bias, dtype=tf.float32))
                hidden2 = tf.nn.relu(tf.nn.bias_add(conv,biases))
                
            with tf.variable_scope('conv3'):
                kernel=tf.get_variable('weights', shape=CONV3_SHAPE,
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform = True))
                conv = tf.nn.conv2d(hidden2, kernel, [1,1,1,1], padding = 'SAME')
                biases = tf.get_variable('biases', shape=CONV3_SHAPE[-1],
                                         initializer=tf.random_uniform_initializer(minval=0, 
                                                                                   maxval=flags.bias, dtype=tf.float32))
                hidden3 = tf.nn.relu(tf.nn.bias_add(conv,biases))
    
            with tf.variable_scope('fc1'):
                dims = [hidden3.get_shape()[i].value for i in range(1,len(hidden3.get_shape()))]
                reshape_hidden3 = tf.reshape(hidden3, [-1, np.prod(dims)])
                weights = tf.get_variable('weights', shape=[reshape_hidden3.get_shape()[1], FC1_SHAPE],
                                          initializer=tf.contrib.layers.xavier_initializer(uniform = True))
                biases = tf.get_variable('biases', shape=FC1_SHAPE,
                                         initializer=tf.random_uniform_initializer(minval=0, 
                                                                                   maxval=flags.bias, dtype=tf.float32))
                hidden4 = tf.nn.relu(tf.matmul(reshape_hidden3, weights) + biases)
    
            with tf.variable_scope('action_value'):
                weights = tf.get_variable('weights', shape=[hidden4.get_shape()[1], flags.num_action],
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))
                biases=tf.get_variable('biases', shape=[flags.num_action],
                                       initializer=tf.constant_initializer(value=flags.bias, dtype=tf.float32))
                action_value=tf.matmul(hidden4,weights) + biases

            return action_value
            
    def _init_training(self):
        flags = self.flags
        
        self._greedy_action = tf.argmax(self._action_value_network, dimension=1)
        
        Qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Qvars')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
        self._target_network_update = [tvar.assign(qvar) for qvar,tvar in zip(Qvars, target_vars)]
        
        max_one_step = self._rewards_pl + flags.gamma*tf.reduce_max(self._target_network, reduction_indices=[1,])*self._finished_pl
        masked_action_qvals = tf.reduce_sum(self._action_value_network*self._action_mask_pl, reduction_indices=[1,])
        Q_loss = tf.reduce_mean(tf.square(max_one_step - masked_action_qvals))
        
        if flags.opt_type is 'RMSprop':
            optimizer = tf.train.RMSPropOptimizer(flags.lr, decay=flags.decay, momentum=flags.momentum, epsilon=flags.opt_eps)
        elif flags.opt_type is 'Adam':
            optimizer = tf.train.AdamOptimizer(flags.lr, beta1=flags.beta1, beta2=flags.beta2, epsilon=flags.opt_eps)
        Qgrads = optimizer.compute_gradients(Q_loss, var_list=Qvars)
        Qgrads = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in Qgrads]
        self._apply_Qgrads = optimizer.apply_gradients(Qgrads)
        
        
    def _init_memory(self):
        flags = self.flags
        
        self._state_buffer = np.empty((flags.buffer_size, flags.frame_dim, flags.frame_dim), dtype='float16')
        self._action_buffer = np.empty(flags.buffer_size, dtype='uint8')
        self._reward_buffer = np.empty(flags.buffer_size, dtype='int16')
        self._finished_buffer = np.empty(flags.buffer_size, dtype='uint8')

        self._buffer_count = 0
        self._buffer_index = 0
        self._current_states = np.empty((flags.batch_size, flags.num_frame, flags.frame_dim, flags.frame_dim), dtype='float32')
        self._new_states = np.empty((flags.batch_size, flags.num_frame, flags.frame_dim, flags.frame_dim), dtype='float32')
        self._sample_inds = np.empty(flags.batch_size, dtype='int32')
        
    def _init_state(self):
        flags = self.flags
        
        self._saver = tf.train.Saver(max_to_keep=1)
        if flags.resume:
            self._saver.restore(self._sess, "/tmp/DQNmodel.ckpt")
            fh = open('/tmp/DQNstate.pkl', 'rb')
            self._epsilon, self._update_num, self._action_num, \
                        self._epoch, self._epochReward, self._maxReward = pickle.load(fh)
            fh.close()
        else:
            init = tf.global_variables_initializer()
            self._sess.run(init)
            self._epsilon = flags.eps_init
            self._update_num = 1
            self._action_num = 0
            self._epoch = 0
            self._epochReward = []
            self._maxReward = 0
            self._sess.run(init)
            
    def initGame(self, env):
        flags = self.flags
        self._eval = False
        observation = env.reset()
        self._input_states = [self._preprocess(observation).astype('float32')]*flags.num_frame
        return observation
             
    def _preprocess(self, screen):
        flags = self.flags
        out_screen = resize(rgb2gray(screen), (110, flags.frame_dim))[16:-10,:]
        return out_screen
        
  
    def _getState(self, index):
        flags = self.flags    #buffer indexing as in tambetm
        # normalize index to expected range, allows negative indexes
        index = index % self._buffer_count
        if index >= flags.num_frame - 1:
          # use faster slicing
          return self._state_buffer[(index - (flags.num_frame - 1)):(index + 1),:,:]
        else:
          # otherwise normalize indexes and use slower list based access
          indexes = [(index - i) % self._buffer_count for i in reversed(range(flags.num_frame))]
          return self._state_buffer[indexes,:,:]

            
    def _updateTargetNetwork(self):
        self._sess.run(self._target_network_update)
        
    def observeState(self, observation):
        
        self._state = self._preprocess(observation)
        self._input_states.append(self._state.astype('float32'))
        del self._input_states[0]

        self._formatted_input = np.stack(self._input_states, axis = 2)
        self._formatted_input = np.reshape(self._formatted_input, (1,) + self._formatted_input.shape)
        
    def chooseAction(self, epsilon=None):
        flags = self.flags
        epsilon = epsilon or self._epsilon
        
        self._action_num+=1
        if np.random.rand() < epsilon:
            action = np.random.randint(0, flags.num_action)
        else:
            action = self._sess.run(self._greedy_action, feed_dict = {self._current_state_pl:self._formatted_input})[0]
                                   
        return action  

    def takeAction(self, env, action):   
        self._lives = env.ale.lives()
        observation, reward, done, info = env.step(action) 
        if self._lives > env.ale.lives():
            reward -= 1  
            done = True
        return observation, reward, done 

    def _takeAction(self, env, action):   
        observation, reward, done, info = env.step(action) 
        return observation, reward, done                     
            
    def annealExplore(self):
        flags = self.flags
        if self._epsilon > flags.eps_final and self._buffer_count > flags.start_train: 
            self._epsilon -= (flags.eps_init - flags.eps_final)/flags.anneal

    def storeReplay(self, action, reward, done):
        flags = self.flags
        
        self._state_buffer[self._buffer_index,:,:] = self._state
        self._action_buffer[self._buffer_index] = action
        self._reward_buffer[self._buffer_index] = np.clip(reward, -1, 1)
        self._finished_buffer[self._buffer_index] = done
        self._buffer_count = max(self._buffer_count, self._buffer_index + 1)
        self._buffer_index = (self._buffer_index + 1) % flags.buffer_size
        
    def train(self):
        flags = self.flags
        
        if self._action_num % flags.train_int == 0 and self._buffer_count >= flags.start_train:
            self._update_num += 1
            for i in range(flags.batch_size):
                while True:
                    ind = np.random.randint(flags.num_frame, self._buffer_count)
                    if self._finished_buffer[(ind-flags.num_frame):(ind-1)].any():
                        continue
                    if ind >= self._buffer_index and (ind - flags.num_frame) < self._buffer_index:
                        continue
                    break
                self._current_states[i,:,:,:] = self._getState(ind-1)
                self._new_states[i,:,:,:] = self._getState(ind)
                self._sample_inds[i] = ind-1
            
            rewards = self._reward_buffer[self._sample_inds]
            actions = self._action_buffer[self._sample_inds]
            finished = self._finished_buffer[self._sample_inds]
            action_mask = np.zeros((flags.batch_size, flags.num_action))
            action_mask[[range(flags.batch_size),actions]] = 1
    
            feed_dict = {self._current_state_pl:np.transpose(self._current_states,(0,2,3,1)), 
                         self._newstate_pl:np.transpose(self._new_states,(0,2,3,1)), self._rewards_pl:rewards,
                         self._action_mask_pl:action_mask, self._finished_pl:np.logical_not(finished)}
            self._sess.run(self._apply_Qgrads, feed_dict=feed_dict)
    
            if self._update_num % flags.tn_update_freq == 0:
                  self._updateTargetNetwork()
    
            if self._update_num % flags.epoch_length == 0:
                self._epoch += 1
                self._eval = True
                  
    def episodeFinished(self):
        flags = self.flags
        
        if self._eval:
            print("epoch {} finished, evaluating agent performance...".format(self._epoch))
            self._evalAgent()
            print('average episode reward: {}, epsilon: {}, max reward achieved thus far: {}'.format(
                  self._epochReward[self._epoch-1], self._epsilon, self._maxReward))
            
            plt.figure(0)
            plt.clf()
            plt.plot(range(self._epoch), self._epochReward, label='average episode reward')
            plt.xlabel('epoch')
            plt.ylabel('reward')
            plt.legend(loc=2)
            plt.savefig('DQN_%slr.png' % (flags.lr))
    
            self._saver.save(self._sess, "/tmp/DQNmodel.ckpt")
            fh = open('/tmp/DQNstate.pkl', 'wb')
            pickle.dump((self._epsilon, self._update_num, self._action_num, 
                         self._epoch, self._epochReward), fh)
            fh.close()   
            
    def _evalAgent(self):
        flags = self.flags
        
        observation = self.initGame(self._evalEnv)
        rewardSum = 0
        episodeReward = []
        episodeNum = 0

        while episodeNum < flags.num_eval:
            self.observeState(observation)
            action = self.chooseAction(.05)
            observation, reward, done = self._takeAction(self._evalEnv, action)   
            rewardSum += reward
            
            if done:
                episodeReward.append(rewardSum)
                episodeNum += 1
                rewardSum = 0
                observation = self.initGame(self._evalEnv)
        self._maxReward = np.max([self._maxReward, np.max(episodeReward)])        
        self._epochReward.append(np.mean(episodeReward))
                
        
                
            
            
