from __future__ import division
import gym
import argparse
from DQN import DQN_AGENT as AGENT

parser = argparse.ArgumentParser(description='Trains a DQN in Tensorflow.')
parser.add_argument('--env', metavar='environment', 
                    help='select an atari environment',
#                    default='Pong-v0',
                    default='Breakout-v0',
                    choices=['Pong-v0', 'Breakout-v0'])  #can add more, after testing
parser.add_argument('--seed', metavar='random seed',
                    help='set the random seed for numpy/tensorflow',
                    type=int,
                    default=150)
parser.add_argument('--gamma',
                    help='discount parameter for Q-learning',
                    metavar='gamma',
                    default=.99)
parser.add_argument('--lr', metavar='learning rate',
                    help='set learning rate for optimizer',
                    type=float,
                    default=2.5e-4)
parser.add_argument('--beta1', metavar='beta1',
                    help='beta1 param for Adam optimizer',
                    type=float,
                    default=.9)
parser.add_argument('--beta2', metavar='beta2',
                    help='beta2 param for Adam optimizer',
                    type=float,
                    default=.999)
parser.add_argument('--decay', metavar='decay',
                    help='decay param for RMSprop optimizer',
                    type=float,
                    default=.9)
parser.add_argument('--momentum', metavar='momentum',
                    help='momentum param for RMSprop optimizer',
                    type=float,
                    default=.95)
parser.add_argument('--bias', metavar='bias init',
                    help='max inital value for bias weights',
                    type=float,
                    default=1e-1)
parser.add_argument('--anneal', metavar='annealing length',
                    help='number of actions over which exploration probability\
                    is annealed',
                    type=int,
                    default=10**6)
parser.add_argument('--resume',
                    help='resume training from saved state',
                    action='store_true')
parser.add_argument('--render',
                    help='render the game screen',
                    action='store_true')
parser.add_argument('--opt_type', metavar='opt_type',
                    help='type of optimizer',
                    default='Adam',
                    choices=['RMSprop', 'Adam'])
parser.add_argument('--num_eval', metavar='num_eval',
                    help='number of episodes to eval at each epoch',
                    type=int,
                    default=10)
parser.add_argument('--epoch_length', metavar='epoch_length',
                    help='length of training epoch (in parameter updates)',
                    type=int,
                    default=5*10**1)

flags = parser.parse_args()

flags.opt_eps = 1e-2          #avoids division by zero in optimizer update
flags.num_frame = 4           #number of recent frames used as input to Q-network
flags.frame_dim = 84          #dimension of processed frame (84x84)
flags.batch_size = 32         #size of each training mini-batch 32
flags.eps_init = 1            #initial value for exploration probability
flags.eps_final = .1          #final value (after annealing) for explore prob
flags.start_train = 5*10**2   #number of actions after which training begins
flags.buffer_size = 10**6     #size of replay memory buffer
flags.train_int = 4           #number of actions selected between training iters
flags.tn_update_freq = 10**4  #frequency of target network update

env = gym.make(flags.env)  #set environment
flags.num_action = env.action_space.n     #Number of possible actions for chosen env
env.ale.setBool('color_averaging', True)

agent = AGENT(flags)

observation = agent.initGame(env)
while True:
    if flags.render: env.render()
    
    agent.observeState(observation)
    action = agent.chooseAction()

    observation, reward, done = agent.takeAction(env, action)         
            
    agent.storeReplay(action, reward, done)
    agent.train()
    agent.annealExplore()
    
    if done:
        agent.episodeFinished()
        observation = agent.initGame(env)
