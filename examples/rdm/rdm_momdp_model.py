from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from past.utils import old_div
import numpy as np
from pomdpy.pomdp import model
from pomdpy.discrete_pomdp import DiscreteActionPool
from pomdpy.discrete_pomdp import DiscreteObservationPool
from momdpy.parser import MOMDPEnvironment
from momdpy.momdp import BeliefPointXY
class Rdm_MOMDP_Model(model.Model):
    def __init__(self, problem_name="rdmmomdp"):
        super(Rdm_MOMDP_Model, self).__init__(problem_name)
        self.momdpenv = MOMDPEnvironment('examples/rdm/rdm_v2.momdp')
        # we have to transform the information given in momdpenv format to serve our needs
        self.Xstates = self.momdpenv.Xstates
        self.Ystates = self.momdpenv.Ystates
        self.actions = self.momdpenv.actions
        self.num_Xstates = len(self.momdpenv.Xstates)
        self.num_Ystates = len(self.momdpenv.Ystates)
        self.num_actions = len(self.momdpenv.actions)
        self.num_observations = len(self.momdpenv.observations)
        self.num_objectives = 1 # number of objectives
        self.epsilon = 0.01
        self.convergence_tolerance = 0.01
        self.Tx = self.transform_Tx(self)
        self.Ty = self.transform_Ty(self)
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
        return self.momdpenv.observations

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
    def transform_Tx(self):
        """
        |A| x |X| x |Y| x |X'|  matrix
        :return:
        """
        p = self.momdpenv
        newTx = np.zeros((len(p.actions), len(p.Xstates), len(p.Ystates), len(p.Xstates)))
        for k, v in p.Tx.items() :
            newTx[k[0]][k[1]][k[2]][k[3]]= v

        return newTx

    @staticmethod
    def transform_Ty(self) :
        """
        |A| x |X| x |Y| x |Y'|  matrix
        :return:
        """
        p = self.momdpenv
        newTy = np.zeros((len(p.actions), len(p.Xstates), len(p.Ystates), len(p.Ystates)))
        for k, v in p.Ty.items() :
            newTy[k[0]][k[1]][k[2]][k[3]] = v

        return newTy

    def get_transition_matrix_Tx(self) :
        return self.Tx

    def get_transition_matrix_Ty(self) :
        return self.Ty

    @staticmethod
    def transform_Z(self) :
        """
        |A| x |X'| x |Y'| x |O| matrix
        :return:
        """
        p = self.momdpenv
        newZ = np.zeros((len(p.actions),  len(p.Xstates), len(p.Ystates), len(p.observations)))
        for k, v in p.Z.items() :
            newZ[k[0]][k[1]][k[2]][k[3]] = v

        return newZ
    def get_observation_matrix(self) :
        return self.O

    @staticmethod
    def transform_R(self) :
        """
        |A| x |X| x |Y| matrix
        :return:
        """
        p = self.momdpenv
        newR = np.zeros((len(p.actions), len(p.Xstates), len(p.Ystates), self.num_objectives))
        for k, v in p.R.items() :
            newR[k[0]][k[1]][k[2]] = v
        return newR


    def get_reward_matrix(self) :
        return self.R


    def getReward(self, x, y, a) :
        R = self.get_reward_matrix()
        return R[a][x][y]


    def transform_actions(self) :
        p = self.num_actions
        i_set = []
        for i in range(self.num_actions) :
            i_set.append(i)
        return i_set


    def get_action_set(self) :
        # LISTEN = 0, OPEN LEFT DOOR = 1, OPEN RIGHT DOOR = 2
        return self.action_set



    def get_initial_belief_state_xy(self) :
        x = 0
        by = [0.5, 0.5] # belief Y
        #todo BeliefPoinXY

        return BeliefPointXY(x, by)


    ''' Factory methods '''


    def create_action_pool(self) :
        return DiscreteActionPool(self)


    def create_root_historical_data(self, agent) :
        return IotData(self)


    ''' --------- BLACK BOX GENERATION --------- '''


    def generate_step(self, action, state=None) :
        pass


    @staticmethod
    def make_next_state(action) :
        pass


    def make_reward(self, action, is_terminal) :
        pass


    def make_observation(self, action) :
        pass


    def belief_update(self, old_belief, action, observation) :
        pass
