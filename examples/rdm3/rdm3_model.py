from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from past.utils import old_div
import numpy as np
from pomdpy.pomdp import model
from pomdpy.discrete_pomdp import DiscreteActionPool
from pomdpy.discrete_pomdp import DiscreteObservationPool
from pomdpy.parser import POMDPEnvironment
from pomdpy.solvers.alpha_matrix import getBestMatrixIndex
from pomdpy.pomdp.belief_point import BeliefPoint

class Rdm3Model(model.Model):
    def __init__(self, problem_name="rdm3"):
        super(Rdm3Model, self).__init__(problem_name)
        self.pomdpenv = POMDPEnvironment('D:/PhD UoB 2019-2022/Publikasi/Transaction2023/POMDPy/examples/rdm3/RDM3_v12_POMDP.POMDP')
        self.num_states = len(self.pomdpenv.states)
        self.num_actions = len(self.pomdpenv.actions)
        self.num_observations = len(self.pomdpenv.observations)
        self.num_objectives = 3 # number of objectives
        self.epsilon = 0.1
        self.convergence_tolerance = 0.01
        self.T = self.transform_T(self)
        self.O = self.transform_Z(self)
        self.R = self.transform_R(self)
        self.action_set = self.transform_actions()
        self.minimum_value = [900, 900, 900]
        # 1819.4377, 1581.2696, 1096.1862
        self.policy_filename = "policies/12/rdm3pomdpv12_matrix_0.policy" # dont forget to create the folder first
        self.current_belief = False
        self.current_action = False
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
        newR = np.zeros((len(p.actions), len(p.states), self.num_objectives))
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

    # version 12
    def make_observation(self, attack, active_links_obs, bandwidth_consumption, threshold_bandwidth_consumption, write_time, threshold_write_time) :
        if not self.current_belief :
            self.current_belief = BeliefPoint(self.get_initial_belief_state())
        else:
            observation=np.zeros(self.num_states)
            if bandwidth_consumption <= threshold_bandwidth_consumption:
                mc_t=1
                mc_f=0
            else:
                mc_t=0
                mc_f=1

            if threshold_write_time <= threshold_write_time:
                mp_t=1
                mp_f=0
            else:
                mp_t=0
                mp_f=1

            if active_links_obs == 2 :
                if attack == True :
                    if self.current_action == 0 :
                        anl_t = 0.7 * 0.15
                        anl_f = 0.3 * 0.85
                    elif self.current_action == 1 :
                        anl_t = 0.85 * 0.15
                        anl_f = 0.15 * 0.85
                if attack == False :
                    if self.current_action == 0 :
                        anl_t = 0.7 * 0.85
                        anl_f = 0.3 * 0.15
                    elif self.current_action == 1 :
                        anl_t = 0.85 * 0.85
                        anl_f = 0.15 * 0.15
            else :
                if attack == True :
                    if self.current_action == 0 :
                        anl_t = 0.3 * 0.15
                        anl_f = 0.7 * 0.85
                    elif self.current_action == 1 :
                        anl_t = 0.15 * 0.15
                        anl_f = 0.85 * 0.85
                if attack == False :
                    if self.current_action == 0 :
                        anl_t = 0.3 * 0.85
                        anl_f = 0.7 * 0.15
                    elif self.current_action == 1 :
                        anl_t = 0.15 * 0.85
                        anl_f = 0.85 * 0.15


            observation[0] = mc_t * mp_t * anl_t
            observation[1] = mc_t * mp_t * anl_f
            observation[2] = mc_t * mp_f * anl_t
            observation[3] = mc_t * mp_f * anl_f
            observation[4] = mc_f * mp_t * anl_t
            observation[5] = mc_f * mp_t * anl_f
            observation[6] = mc_f * mp_f * anl_t
            observation[7] = mc_f * mp_f * anl_f
            print(f"current_belief_state = {self.current_belief.belief} current_action = {self.current_action} observ = {observation}")
            # update belief
            self.current_belief=self.belief_update(self.current_belief, self.current_action, observation)
            print(f"updated belief state = {self.current_belief.belief}")

    # version 11
    def make_observationv11(self, attack, active_links_obs, bandwidth_consumption, threshold_bandwidth_consumption,
                            write_time, threshold_write_time) :
        if not self.current_belief :
            self.current_belief = BeliefPoint(self.get_initial_belief_state())
        else :
            observation = np.zeros(self.num_states)
            if bandwidth_consumption <= threshold_bandwidth_consumption :
                mc_t = 1
                mc_f = 0
            else :
                mc_t = 0
                mc_f = 1

            if threshold_write_time <= threshold_write_time :
                mp_t = 1
                mp_f = 0
            else :
                mp_t = 0
                mp_f = 1

            if active_links_obs == 2 :
                if attack == True :
                    if self.current_action == 0 :
                        anl_t = 0.7 * 0.4
                        anl_f = 0.3 * 0.6
                    elif self.current_action == 1 :
                        anl_t = 0.85 * 0.4
                        anl_f = 0.15 * 0.6
                if attack == False :
                    if self.current_action == 0 :
                        anl_t = 0.7 * 0.6
                        anl_f = 0.3 * 0.4
                    elif self.current_action == 1 :
                        anl_t = 0.85 * 0.6
                        anl_f = 0.15 * 0.4
            else :
                if attack == True :
                    if self.current_action == 0 :
                        anl_t = 0.3 * 0.4
                        anl_f = 0.7 * 0.6
                    elif self.current_action == 1 :
                        anl_t = 0.15 * 0.4
                        anl_f = 0.85 * 0.6
                if attack == False :
                    if self.current_action == 0 :
                        anl_t = 0.3 * 0.6
                        anl_f = 0.7 * 0.4
                    elif self.current_action == 1 :
                        anl_t = 0.15 * 0.6
                        anl_f = 0.85 * 0.4

            observation[0] = mc_t * mp_t * anl_t
            observation[1] = mc_t * mp_t * anl_f
            observation[2] = mc_t * mp_f * anl_t
            observation[3] = mc_t * mp_f * anl_f
            observation[4] = mc_f * mp_t * anl_t
            observation[5] = mc_f * mp_t * anl_f
            observation[6] = mc_f * mp_f * anl_t
            observation[7] = mc_f * mp_f * anl_f
            print(
                f"current_belief_state = {self.current_belief.belief} current_action = {self.current_action} observ = {observation}")
            # update belief
            self.current_belief = self.belief_update(self.current_belief, self.current_action, observation)
            print(f"updated belief state = {self.current_belief.belief}")
    def belief_update(self, old_belief: BeliefPoint, action, observation):
        new_belief = np.zeros(self.num_states)
        sum_tb=0
        sum_new_belief=0
        for s_prime in range(self.num_states):
            for s in range(self.num_states):
                sum_tb += self.T[action][s][s_prime] * old_belief.belief[s]
            new_belief[s_prime] = observation[s_prime] * sum_tb
            sum_new_belief += new_belief[s_prime]

        #normalise new_belief
        for s in range(self.num_states):
            new_belief[s] = new_belief[s] / sum_new_belief

        return BeliefPoint(new_belief)
    def best_action(self, policy, weights):
        #obtain best alpha-matrix
        idx = getBestMatrixIndex(self.current_belief.getBelief(), policy,weights)
        best_av = policy[idx]

        #define best action
        action = best_av.action
        self.current_action = action

        # calculate expected reward by doing an action on particular belief
        value = np.dot(self.current_belief.getBelief(), self.R[action])
        print(f"reward value momdp = {value}")
        return value, action