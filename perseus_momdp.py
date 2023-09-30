#!/usr/bin/env python
from __future__ import print_function
from momdpy import MOMDP_Agent
from momdpy.solvers import PerseusPBVI
from momdpy.solvers import PerseusOLS
from examples.rdm3 import Rdm3_MOMDP_Model
from examples.rdm import Rdm_MOMDP_Model
from examples.iot2 import Iot2_MOMDP_Model
from examples.iot import Iot_MOMDP_Model
from examples.tagavoid import Tagavoid_MOMDP_Model
from examples.rocksample1x3 import Rocksample_MOMDP_Model
from pomdpy.log import init_logger
import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set the run parameters.')
    parser.add_argument('--env', type=str, help='Specify the env to solve {Tiger}')
    parser.add_argument('--solver', type=str,
                        help='Specify the solver to use {ValueIteration|LinearAlphaNet|VI-Baseline}')
    parser.add_argument('--seed', default=123, type=int, help='Specify the random seed for numpy.random')
    parser.add_argument('--use_tf', dest='use_tf', action='store_true', help='Set if using TensorFlow')
    parser.add_argument('--discount', default=0.95, type=float, help='Specify the discount factor (default=0.75)')
    parser.add_argument('--n_epochs', default=1, type=int, help='Num of epochs of the experiment to conduct')
    parser.add_argument('--max_steps', default=10, type=int, help='Max num of steps per trial/episode/trajectory/epoch')
    parser.add_argument('--save', dest='save', action='store_true', help='Pickle the weights/alpha vectors')

    # Args for Deep Alpha Nets
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--learning_rate_minimum', default=0.0025, type=float)
    parser.add_argument('--learning_rate_decay', default=0.996, type=float)
    parser.add_argument('--learning_rate_decay_step', default=50, type=int)
    parser.add_argument('--beta', default=0.001, type=float, help='L2 regularization parameter')

    parser.add_argument('--test', default=10, type=int, help='Evaluate the agent every `test` epochs')
    parser.add_argument('--epsilon_start', default=0.02, type=float)
    parser.add_argument('--epsilon_minimum', default=0.05, type=float)
    parser.add_argument('--epsilon_decay', default=0.96, type=float)
    parser.add_argument('--epsilon_decay_step', default=75, type=int)
    parser.add_argument('--planning_horizon', default=5, type=int, help='Number of lookahead steps for value iteration')
    parser.set_defaults(use_tf=False)
    parser.set_defaults(save=False)

    # For Perseus PBVI
    parser.add_argument('--sampling_belief_size', default=1000, type=int, help='Number belief sampling points')
    parser.add_argument('--convergence_tolerance', default=0.000001, type=float, help='Convergence tolerance')
    parser.add_argument('--time_limit', default=600000, type=float, help='Time limit to converge (in miliseconds)')

    # For Persues OLS
    parser.add_argument('--timeout', default=3600000, type=float, help='Time limit to converge for OLS (in seconds)')

    args = vars(parser.parse_args())

    init_logger()

    np.random.seed(int(args['seed']))

    if args['solver'] == 'PerseusPBVI' :
        #todo
        solver = PerseusPBVI
        pass
    elif args['solver'] == 'PerseusOLS' :
        #todo
        solver = PerseusOLS
        pass
    else :
        raise ValueError('solver not supported')

    if args['env'] == 'rdm3momdp':
        assert solver == PerseusOLS
        env = Rdm3_MOMDP_Model(args)
        agent = MOMDP_Agent(env, solver)
        agent.discounted_return()
    if args['env'] == 'iot2momdp':
        assert solver == PerseusOLS
        env = Iot2_MOMDP_Model(args)
        agent = MOMDP_Agent(env, solver)
        agent.discounted_return()
    elif args['env'] == 'rdmmomdp':
        assert solver == PerseusPBVI
        env = Rdm_MOMDP_Model(args)
        agent = MOMDP_Agent(env, solver)
        agent.discounted_return()
    elif args['env'] == 'iotmomdp':
        assert solver == PerseusPBVI
        env = Iot_MOMDP_Model(args)
        agent = MOMDP_Agent(env, solver)
        agent.discounted_return()
    elif args['env'] == 'tagavoidmomdp':
        assert solver == PerseusPBVI
        env = Tagavoid_MOMDP_Model(args)
        env.print_info()
        #agent = MOMDP_Agent(env, solver)
        #agent.discounted_return()
    elif args['env'] == 'rocksample1x3momdp':
        assert solver == PerseusPBVI
        env = Rocksample_MOMDP_Model(args)
        #env.print_info()
        agent = MOMDP_Agent(env, solver)
        agent.discounted_return()
    else :
        print('Unknown env {}'.format(args['env']))