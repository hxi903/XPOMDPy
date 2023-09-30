from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from past.utils import old_div
import numpy as np
from pomdpy.pomdp import model
from pomdpy.discrete_pomdp import DiscreteActionPool
from pomdpy.discrete_pomdp import DiscreteObservationPool
from momdpy.parser import POMDPX_Parser
from momdpy.momdp import BeliefPointXY
from momdpy.momdp.alpha_matrix import getVectorValue, getBestMatrixIndex
class Rdm3_MOMDP_Model(model.Model):
    def __init__(self, problem_name="rdm3momdp"):
        super(Rdm3_MOMDP_Model, self).__init__(problem_name)
        self.momdpenv = POMDPX_Parser("D:/PhD UoB 2019-2022/Publikasi/Transaction2023/POMDPy/examples/rdm3/RDM3_v12_MOMDP.pomdpx")
        # we have to transform the information given in momdpenv format to serve our needs
        self.Xstates = self.momdpenv.states[0]['x_0'][0]
        self.Ystates = self.momdpenv.states[0]['y_0'][0]
        self.actions = list(self.momdpenv.actions.values())[0]
        self.observations = list(self.momdpenv.observations.values())[0]
        self.num_Xstates = len(self.Xstates)
        self.num_Ystates = len(self.Ystates)
        self.num_actions = len(self.actions)
        self.num_observations = len(self.observations)
        self.num_objectives = 3 # number of objectives
        self.epsilon = 0.1
        self.convergence_tolerance = 0.01
        self.Tx = self.momdpenv.state_transition['x_0']
        self.Ty = self.momdpenv.state_transition['y_0']
        self.O = list(self.momdpenv.obs_table.values())[0]
        self.R = self.momdpenv.reward_table
        self.initial_belief = self.get_initial_belief_state_xy()
        self.action_set = self.transform_actions()
        self.minimum_value = [1700, 1700, 1100]
        self.policy_filename = "policies/12/rdm3momdpv12_matrix_0.policy" # dont forget to create the folder first
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

    # version 12
    def make_observation(self, attack, active_links_obs, bandwidth_consumption, threshold_bandwidth_consumption, write_time, threshold_write_time) :
        if not self.current_belief :
            self.current_belief = self.get_initial_belief_state_xy()
        else:
            # define x_prime
            if bandwidth_consumption <= threshold_bandwidth_consumption:
                mc = True
            else: mc = False

            if write_time <= threshold_write_time :
                mp = True
            else: mp = False

            if mc :
                if mp :
                    x_prime = 0 #x1
                else:
                    x_prime = 1 #x2
            else:
                if mp :
                    x_prime = 2 #x3
                else:
                    x_prime = 3 #x4

            # probe o[y1_prime, y2_prime]
            #| A | x | X'| x |Y' | x | O |
            observation=[0.0,0.0]

            if active_links_obs == 2:
                if attack == True :
                    if self.current_action == 0:
                        anl_t=0.7 * 0.15
                        anl_f=0.3 * 0.85
                    elif self.current_action == 1:
                        anl_t=0.85 * 0.15
                        anl_f=0.15 * 0.85
                if attack == False:
                    if self.current_action == 0 :
                        anl_t=0.7 * 0.85
                        anl_f=0.3 * 0.15
                    elif self.current_action == 1 :
                        anl_t=0.85 * 0.85
                        anl_f=0.15 * 0.15
            else:
                if attack == True :
                    if self.current_action == 0 :
                        anl_t=0.3 * 0.15
                        anl_f=0.7 * 0.85
                    elif self.current_action == 1 :
                        anl_t=0.15 * 0.15
                        anl_f=0.85 * 0.85
                if attack == False:
                    if self.current_action == 0 :
                        anl_t=0.3 * 0.85
                        anl_f=0.7 * 0.15
                    elif self.current_action == 1 :
                        anl_t=0.15 * 0.85
                        anl_f=0.85 * 0.15

            observation[0] = anl_t
            observation[1] = anl_f

            print(f"current_belief_xstate = {self.current_belief.x_state} current_belief_y= {self.current_belief.belief_y} current_action = {self.current_action} observ = {observation}")
            # update belief
            self.current_belief=self.belief_update(self.current_belief, self.current_action, x_prime, observation)
            print(f"updated beleif xstate = {self.current_belief.x_state} by = {self.current_belief.belief_y}")

    # version 11
    def make_observationv11(self, attack, active_links_obs, bandwidth_consumption, threshold_bandwidth_consumption,
                            write_time, threshold_write_time) :
        if not self.current_belief :
            self.current_belief = self.get_initial_belief_state_xy()
        else :
            # define x_prime
            if bandwidth_consumption <= threshold_bandwidth_consumption :
                mc = True
            else :
                mc = False

            if write_time <= threshold_write_time :
                mp = True
            else :
                print(f"beyond threshold! {write_time}")
                mp = False

            if mc :
                if mp :
                    x_prime = 0  # x1
                else :
                    x_prime = 1  # x2
            else :
                if mp :
                    x_prime = 2  # x3
                else :
                    x_prime = 3  # x4

            # probe o[y1_prime, y2_prime]
            # | A | x | X'| x |Y' | x | O |
            observation = [0.0, 0.0]

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

            observation[0] = anl_t
            observation[1] = anl_f

            print(
                f"current_belief_xstate = {self.current_belief.x_state} current_belief_y= {self.current_belief.belief_y} current_action = {self.current_action} observ = {observation}")
            # update belief
            self.current_belief = self.belief_update(self.current_belief, self.current_action, x_prime, observation)
            print(f"updated beleif xstate = {self.current_belief.x_state} by = {self.current_belief.belief_y}")
    def belief_update(self, old_belief, action, x_prime, observation) :
        #| A | x | X | x | Y | x | X'|  matrix
        b0_yprime = observation[0] * ((self.Tx[action][old_belief.x_state][0][x_prime] * self.Ty[action][old_belief.x_state][0][0] * old_belief.belief_y[0]) + (self.Tx[action][old_belief.x_state][1][x_prime] * self.Ty[action][old_belief.x_state][1][1] * old_belief.belief_y[1]))
        b1_yprime = observation[1] * ((self.Tx[action][old_belief.x_state][0][x_prime] * self.Ty[action][old_belief.x_state][0][0] * old_belief.belief_y[0]) + (self.Tx[action][old_belief.x_state][1][x_prime] * self.Ty[action][old_belief.x_state][1][1] * old_belief.belief_y[1]))
        sum_b_yprime = b0_yprime + b1_yprime
        print(f"sum_b_yprime = {sum_b_yprime} b0_yprime = {b0_yprime} b1_yprime = {b1_yprime}")
        new_y_belief = [b0_yprime/sum_b_yprime, b1_yprime/sum_b_yprime]
        # update self.current_belief
        new_belief = BeliefPointXY(x_prime, new_y_belief)
        return new_belief

    def best_action(self, policy, weights):
        #obtain best alpha-matrix
        for av in policy[self.current_belief.x_state]:
            print(f"action {av.action}")
            print(f"value = {av.vs}")
        idx = getBestMatrixIndex(self.current_belief.getBeliefY(), policy[self.current_belief.x_state],weights)
        print(f"weights = {weights}")
        print(f"idx = {idx}")
        best_av = policy[self.current_belief.x_state][idx]

        #define best action
        action = best_av.action
        print(f"action momdp = {action}")
        self.current_action = action
        # calculate expected reward by doing an action on particular belief
        value = np.dot(self.current_belief.getBeliefY(), self.R[action][self.current_belief.x_state])
        print(f"reward value momdp = {value}")


        return value, action