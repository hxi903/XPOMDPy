#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the Pomdpx Parser class.
Modified from source: https://github.com/iciac/POMDP/blob/master/src/parser_main.py which extends a python pomdpx parser https://github.com/larics/python-pomdp to handle special characters ('*','-') and special terms ('identity', 'uniform')
consistent with the PomdpX File Format as documented at https://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.PomdpXDocumentation.
"""

from momdpy.parser.pomdpx_parser_utilities import *
from momdpy.momdp.alpha_vector import AlphaVector
from momdpy.momdp.alpha_matrix import AlphaMatrix

class POMDPX_Parser() :
    def __init__(self, model_filename) :
        root = ET.parse(model_filename).getroot()
        self.description = get_description(root)
        self.discount = get_discount(root)
        self.actions = get_actions(root)
        self.states = get_states(root)
        self.observations = get_observations(root)
        self.initial_belief = get_initial_belief(root)
        self.reward_table = get_reward_function(root)
        self.obs_table = get_obs_function(root)
        self.state_transition, self.state_variable_dict = get_state_transition(root)

class MOMDP_PolicyX_Vector_Parser() :
    def __init__(self, policy_filename) :
        root = ET.parse(policy_filename).getroot()
        self.Theta = self.make_alpha_set(root)

    def make_alpha_set(self, root):
        vectors, action, obsvalue = import_policy_vector(root)
        unique_states = np.unique(np.array(obsvalue))
        Theta = [[] for _ in range(len(unique_states))]
        for i in range(len(vectors)):
            av = AlphaVector(action[i], vectors[i])
            Theta[obsvalue[i]].append(av)
        return Theta

class MOMDP_PolicyX_Matrix_Parser() :
    def __init__(self, policy_filename) :
        root = ET.parse(policy_filename).getroot()
        self.Theta = self.make_alpha_set(root)

    def make_alpha_set(self, root):
        vectors, action, obsvalue = import_policy_matrix(root)
        unique_states = np.unique(np.array(obsvalue))
        Theta = [[] for _ in range(len(unique_states))]
        for i in range(len(vectors)):
            av = AlphaMatrix(action[i], vectors[i])
            Theta[obsvalue[i]].append(av)
        return Theta

# the codes below are only for testing

def main_model() :
    model_filename = '../../examples/rdm3/RDM3_v11_MOMDP.pomdpx'

    Prs = POMDPX_Parser(model_filename)
    print('description:', Prs.description)
    print('discount:', Prs.discount)
    print('actions:', list(Prs.actions.values())[0])
    print('observations:', list(Prs.observations.values())[0])
    print('x_states:', Prs.states[0]['x_0'][0])
    print('initial_belief:', Prs.initial_belief['x_0'])
    print('transition_table:', Prs.state_transition)
    print('obs_table:', Prs.obs_table.values())
    print('reward_table:', Prs.reward_table)
    reward=np.array(Prs.reward_table)
    print('reward_table_shape:', reward.shape)
def main_policy() :
    policy_filename = '../../rdm3momdp_matrix.policy'

    Prs = PolicyX_Matrix_Parser(policy_filename)
    print('Testing POLICY')
    print('alpha vectors:')
    i=0
    for subset in Prs.Theta:
        for av in subset:
            print(f"subset={i} av.v={av.vs} av.a={av.action}")
        i+=1

if __name__ == "__main__" :
    main_model()
    #main_policy()