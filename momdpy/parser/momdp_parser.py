"""
Loading .MOMDP environment and policy file into a Python object. 
Author: Hargyo Ignatius (hxi903), based on pomdp_parser by mbforbes (https://github.com/mbforbes/py-pomdp/blob/master/LICENSE.txt)
TODO:
Add capability to support multiple X-states set and Y-states
Check model after construction to provide sanity check for specified pomdp environment (e.g. observation and transition probabilities sum to 1.0)
Add capability to support <ui_martrix> UNIFORM | IDENTITY | <prob_matrix> for the transition probability, observation probability, and reward function if applicable
"""
# builtins
import xml.etree.ElementTree as ET

# 3rd party
import numpy as np
class MOMDPEnvironment:
    def __init__(self, filename):
        """
        Parses .momdp file and loads the information into this object's attributes.
        Attributes:
            discount
            values
            Xstates
            Ystates
            actions
            observations
            Tx
            Ty
            Z
            R
        """
        f = open(filename, 'r')
        self.contents = [
            x.strip() for x in f.readlines()
            if (not (x.startswith("#") or x.isspace()))
        ]

        # set up transition function Tx and Ty, observation function Z, and
        # reward R
        self.Tx = {}
        self.Ty = {}
        self.Z = {}
        self.R = {}

        # go through line by line
        i = 0
        while i < len(self.contents):
            line = self.contents[i]
            if line.startswith('discount'):
                i = self.__get_discount(i)
            elif line.startswith('values'):
                i = self.__get_value(i)
            elif line.startswith('Xstates'):
                i = self.__get_Xstates(i)
            elif line.startswith('Ystates'):
                i = self.__get_Ystates(i)
            elif line.startswith('actions'):
                i = self.__get_actions(i)
            elif line.startswith('observations'):
                i = self.__get_observations(i)
            elif line.startswith('Tx'):
                i = self.__get_transitionX(i)
            elif line.startswith('Ty'):
                i = self.__get_transitionY(i)
            elif line.startswith('O'):
                i = self.__get_observation(i)
            elif line.startswith('R'):
                i = self.__get_reward(i)
            else:
                raise Exception("Unrecognized line: " + line)

        # cleanup
        f.close()

    def __get_discount(self, i) :
        line = self.contents[i]
        self.discount = float(line.split()[1])
        return i + 1

    def __get_value(self, i) :
        # Currently just supports "values: reward". I.e. currently
        # meaningless.
        line = self.contents[i]
        self.values = line.split()[1]
        return i + 1

    def __get_Xstates(self, i) :
        line = self.contents[i]
        self.Xstates = line.split()[1 :]
        if is_numeric(self.Xstates) :
            no_Xstates = int(self.Xstates[0])
            self.Xstates = [str(x) for x in range(no_Xstates)]
        return i + 1

    def __get_Ystates(self, i) :
        line = self.contents[i]
        self.Ystates = line.split()[1 :]
        if is_numeric(self.Ystates) :
            no_Ystates = int(self.Ystates[0])
            self.Ystates = [str(x) for x in range(no_Ystates)]
        return i + 1
    def __get_actions(self, i):
        line = self.contents[i]
        self.actions = line.split()[1:]
        if is_numeric(self.actions):
            no_actions = int(self.actions[0])
            self.actions = [str(x) for x in range(no_actions)]
        return i + 1
    def __get_observations(self, i):
        line = self.contents[i]
        self.observations = line.split()[1:]
        if is_numeric(self.observations):
            no_observations = int(self.observations[0])
            self.observations = [str(x) for x in range(no_observations)]
        return i + 1
    def __get_transitionX(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = self.actions.index(pieces[0])

        if len(pieces) == 5:
            # case 1: Tx: <action> : <X start-state> : <Y start-state>: <X next-state>  %f
            Xstart_state = self.Xstates.index(pieces[1])
            Ystart_state = self.Ystates.index(pieces[2])
            Xnext_state = self.Xstates.index(pieces[3])
            prob = float(pieces[4])
            self.Tx[(action, Xstart_state, Ystart_state, Xnext_state)] = prob
            return i + 1
        elif len(pieces) == 4:
            # case 1: Tx: <action> : <X start-state> : <Y start-state>: <X next-state>
            # %f
            Xstart_state = self.Xstates.index(pieces[1])
            Ystart_state = self.Ystates.index(pieces[2])
            Xnext_state = self.Xstates.index(pieces[3])
            next_line = self.contents[i+1]
            prob = float(next_line)
            self.Tx[(action, Xstart_state, Ystart_state, Xnext_state)] = prob
            return i + 2
        elif len(pieces) == 3:
            # case 3: Tx: <action> : <X start-state> : <Y start-state>
            # %f %f ... %f
            Xstart_state = self.Xstates.index(pieces[1])
            Ystart_state = self.Ystates.index(pieces[2])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.Xstates)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.Tx[(action, Xstart_state, Ystart_state, j)] = prob
            return i + 2
        else:
            raise Exception("Cannot parse line " + line)

    def __get_transitionY(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = self.actions.index(pieces[0])

        if len(pieces) == 5:
            # case 1: Ty: <action> : <X start-state> : <Y start-state> : <Y next-state> %f
            Xstart_state = self.Xstates.index(pieces[1])
            Ystart_state = self.Ystates.index(pieces[2])
            Ynext_state = self.Ystates.index(pieces[3])
            prob = float(pieces[4])
            self.Ty[(action, Xstart_state, Ystart_state, Ynext_state)] = prob
            return i + 1
        elif len(pieces) == 4:
            # case 1: Tx: <action> : <X start-state> : <Y start-state>: <Y next-state>
            # %f
            Xstart_state = self.Xstates.index(pieces[1])
            Ystart_state = self.Ystates.index(pieces[2])
            Ynext_state = self.Ystates.index(pieces[3])
            next_line = self.contents[i+1]
            prob = float(next_line)
            self.Ty[(action, Xstart_state, Ystart_state, Xnext_state, Ynext_state)] = prob
            return i + 2
        else:
            raise Exception("Cannot parse line " + line)
    def __get_observation(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        if pieces[0] == "*":
            # Case when action does not affect observation
            action = None
        else:
            action = self.actions.index(pieces[0])

        if len(pieces) == 5:
            # case 1: O: <action> : <X next-state> <Y next-state>: <obs> %f
            Xnext_state = self.Xstates.index(pieces[1])
            Ynext_state = self.Ystates.index(pieces[2])
            obs = self.observations.index(pieces[3])
            prob = float(pieces[4])
            self.Z[(action, Xnext_state, Ynext_state, obs)] = prob
            return i + 1
        elif len(pieces) == 4:
            # case 2: O: <action> : <X next-state> <Y next-state>: <obs>
            # %f
            Xnext_state = self.Xstates.index(pieces[1])
            Ynext_state = self.Ystates.index(pieces[2])
            obs = self.observations.index(pieces[3])
            next_line = self.contents[i+1]
            prob = float(next_line)
            self.Z[(action, Xnext_state, Ynext_state, obs)] = prob
            return i + 2
        elif len(pieces) == 3:
            # case 3: O: <action> : <X next-state> <Y next-state>
            # %f %f ... %f
            Xnext_state = self.Xstates.index(pieces[1])
            Ynext_state = self.Ystates.index(pieces[2])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.Z[(action, Xnext_state, Ynext_state, j)] = prob
            return i + 2
        else:
            raise Exception("Cannot parse line: " + line)

    def __get_reward(self, i) :
        """
        Wild card * are allowed when specifying a single reward
        probability. They are not allowed when specifying a vector or
        matrix of probabilities.
        """
        line = self.contents[i]
        # print(f"line {line}")
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        if pieces[0] == "*" :
            action = None
        else :
            action = self.actions.index(pieces[0])

        if len(pieces) >= 7 or len(pieces) == 6 :
            # case 1:
            # R: <action> : <X start-state> : <Y start-state> : <X next-state> :  <Y next-state> : <obs> %f
            # any of <start-state>, <next-state>, and <obs> can be *
            # %f can be on the next line (case where len(pieces) == 4)
            X_start_state_raw = pieces[1]
            Y_start_state_raw = pieces[2]
            X_next_state_raw = pieces[3]
            Y_next_state_raw = pieces[4]
            obs_raw = pieces[5]
            if len(pieces) >= 7 :
                reward = pieces[6:]
                lst_reward = list(reward)
                print(f"len lst_reward = {len(lst_reward)}")
                if len(lst_reward) == 1 :
                    prob = float(lst_reward[0])
                else :
                    prob = [float(i) for i in lst_reward]
            else :
                prob = float(self.contents[i + 1])

            # prob = float(pieces[4]) if len(pieces) == 5 \
            #    else float(self.contents[i+1])
            # print(f"action = {action} start_state_raw={start_state_raw} next_state_raw={next_state_raw} obs_raw={obs_raw} prob={prob}")
            self.__reward_ss(
                action, X_start_state_raw, Y_start_state_raw, X_next_state_raw, Y_next_state_raw, obs_raw, prob)
            return i + 1 if len(pieces) >= 5 else i + 2
        elif len(pieces == 5) :
            # case 2: R: <action> : <X start-state> : <Y start-state> : <X next-state> : <Y next-state>
            # %f %f ... %f
            X_start_state = self.states.index(pieces[1])
            Y_start_state = self.states.index(pieces[2])
            X_next_state = self.states.index(pieces[3])
            Y_next_state = self.states.index(pieces[4])
            next_line = self.contents[i + 1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)) :
                prob = float(probs[j])
                self.R[(action, X_start_state, Y_start_state, X_next_state, Y_next_state, j)] = prob
            return i + 2
        else :
            raise Exception("Cannot parse line: " + line)

    def __reward_ss(self, a, X_start_state_raw, Y_start_state_raw, X_next_state_raw, Y_next_state_raw, obs_raw, prob) :
        """
        reward_ss means we're at the start state of the unrolling of the
        reward expression. start_state_raw could be * or the name of the
        real start state.
        """
        if X_start_state_raw == '*' and Y_start_state_raw == '*' :
            for i in range(len(self.Xstates)) :
                for j in range(len(self.Ystates)) :
                    self.__reward_ns(a, i, j, X_next_state_raw, Y_next_state_raw, obs_raw, prob)
        else :
            X_start_state = self.Xstates.index(X_start_state_raw)
            Y_start_state = self.Ystates.index(Y_start_state_raw)
            self.__reward_ns(a, X_start_state, Y_start_state, X_next_state_raw, Y_next_state_raw, obs_raw, prob)

    def __reward_ns(self, a, X_start_state, Y_start_state, X_next_state_raw, Y_next_state_raw, obs_raw, prob) :
        """
        reward_ns means we're at the next state of the unrolling of the
        reward expression. start_state is the number of the real start
        state, and next_state_raw could be * or the name of the real
        next state.
        """
        if X_next_state_raw == '*' and Y_next_state_raw == '*' :
            for i in range(len(self.Xstates)) :
                for j in range(len(self.Ystates)) :
                    self.__reward_ob(a, X_start_state, Y_start_state, i, j, obs_raw, prob)
        else :
            X_next_state = self.Xstates.index(X_next_state_raw)
            Y_next_state = self.Ystates.index(Y_next_state_raw)
            self.__reward_ob(a, X_start_state, Y_start_state, X_next_state, Y_next_state, obs_raw, prob)

    def __reward_ob(self, a, X_start_state, Y_start_state, X_next_state, Y_next_state, obs_raw, prob) :
        """
        reward_ob means we're at the observation of the unrolling of the
        reward expression. start_state is the number of the real start
        state, next_state is the number of the real next state, and
        obs_raw could be * or the name of the real observation.
        """
        if obs_raw == '*' :
            for i in range(len(self.observations)) :
                self.R[(a, X_start_state, Y_start_state, X_next_state, Y_next_state, i)] = prob
        else :
            obs = self.observations.index(obs_raw)
            self.R[(a, X_start_state, Y_start_state, X_next_state, Y_next_state, obs)] = prob

    def print_summary(self):
        print("discount:", self.discount)
        print("values:", self.values)
        print("Xstates:", self.Xstates)
        print("Ystates:", self.Ystates)
        print("actions:", self.actions)
        print("observations:", self.observations)
        print("")
        print("Tx:", self.Tx)
        print("")
        print("Ty:", self.Ty)
        print("")
        print("Z:", self.Z)
        print("")
        print("R:", self.R)


def is_numeric(lst):
    if len(lst) == 1:
        try:
            int(lst[0])
            return True
        except Exception:
            return False
    else:
        return False
