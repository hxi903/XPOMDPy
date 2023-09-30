from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from past.utils import old_div
import numpy as np
from pomdpy.pomdp import model
from pomdpy.discrete_pomdp import DiscreteActionPool
from pomdpy.discrete_pomdp import DiscreteObservationPool
from pomdpy.parser import POMDPEnvironment


class RdmModel(model.Model):
    def __init__(self, problem_name="rdm"):
        super(RdmModel, self).__init__(problem_name)
        self.pomdpenv = POMDPEnvironment('examples/rdm/rdm_v2.pomdp')
        self.num_states = len(self.pomdpenv.states)
        self.num_actions = len(self.pomdpenv.actions)
        self.num_observations = len(self.pomdpenv.observations)
        self.T = self.transform_T(self)
        self.O = self.transform_Z(self)
        self.R = self.transform_R(self)
        self.action_set = self.transform_actions()

    def is_terminal(self, state):
        pass
    def start_scenario(self):
        pass

    ''' --------- Abstract Methods --------- '''

    def sample_an_init_state(self):
        pass

    def create_observation_pool(self, solver):
        pass

    def sample_state_uninformed(self):
        pass


    def sample_state_informed(self, belief):
        pass

    def get_all_states(self):
        pass

    def get_all_actions(self):
        pass

    def get_all_observations(self):
        """
        Either the roar of the tiger is heard coming from door 0 or door 1
        :return:
        """
        return self.pomdpenv.observations

    def get_legal_actions(self, _):
        pass

    def is_valid(self, _):
        return True

    def reset_for_simulation(self):
        self.start_scenario()

    # Reset every "episode"
    def reset_for_epoch(self):
        self.start_scenario()

    def update(self, sim_data):
        pass

    def get_max_undiscounted_return(self):
        return 10

    @staticmethod
    def transform_T(self):
        """
        |A| x |S| x |S'| matrix
        :return:
        """
        p = self.pomdpenv
        newT = np.zeros((len(p.actions), len(p.states), len(p.states)))
        for k, v in p.T.items() :
            newT[k[0]][k[1]][k[2]] = v

        return newT

    def get_transition_matrix(self) :
        return self.T

    @staticmethod
    def transform_Z(self):
        """
        |A| x |S| x |O| matrix
        :return:
        """
        p = self.pomdpenv
        newZ = np.zeros((len(p.actions), len(p.states), len(p.observations)))
        for k, v in p.Z.items() :
            newZ[k[0]][k[1]][k[2]] = v

        return newZ
    def get_observation_matrix(self):
        return self.O
    @staticmethod
    def transform_R(self) :
        """
        |A| x |S| matrix
        :return:
        """
        p = self.pomdpenv
        newR = np.zeros((len(p.actions), len(p.states)))
        for k, v in p.R.items() :
            newR[k[0]][k[1]] = v
        return newR
    def get_reward_matrix(self):
        return self.R
    def getReward(self, s, a):
        R = self.get_reward_matrix()
        return R[a][s]

    def transform_actions(self):
        p = self.num_actions
        i_set = []
        for i in range(self.num_actions):
            i_set.append(i)
        return i_set
    def get_action_set(self):
        # LISTEN = 0, OPEN LEFT DOOR = 1, OPEN RIGHT DOOR = 2
        return self.action_set

    @staticmethod
    def get_initial_belief_state():
        return np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0])
        #return np.array([0.5, 0.5])

    ''' Factory methods '''

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def create_root_historical_data(self, agent):
        return IotData(self)

    ''' --------- BLACK BOX GENERATION --------- '''

    def generate_step(self, action, state=None):
        pass

    @staticmethod
    def make_next_state(action):
        pass

    def make_reward(self, action, is_terminal):
        pass

    def make_observation(self, action):
        pass

    def belief_update(self, old_belief, action, observation):
        pass