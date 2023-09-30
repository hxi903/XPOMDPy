from __future__ import absolute_import

import time

from .solver import Solver
from pomdpy.pomdp import BeliefPoint
from .alpha_vector import AlphaVector, sumVectors, getBestVectorIndex, getValue
#from scipy.optimize import linprog
import numpy as np
#from itertools import product
class PerseusPBVI(Solver):
    def __init__(self, agent):
        """
        Initialize the POMDP Perseus PBVI approximate solver
        :param agent:
        :return:
        """
        super(PerseusPBVI, self).__init__(agent)
        self.history = agent.histories.create_sequence()
        self.totalSolveTime = 0
        self.expectedValue = 0.0
        self.gamma=set()

    @staticmethod
    def reset(agent):
        return PerseusPBVI(agent)

    def getTotalSolveTime(self):
        return self.totalSolveTime * 0.001

    def perseus_pbvi(self, t, o, r, B: list[BeliefPoint]):

        #This implement PerseusUpdate (see Alg.4)

        actions = self.model.num_actions  # |A| actions
        states = self.model.num_states  # |S| states

        # create initial vector set and vectors defining immediate rewards
        start_time = time.time()
        V = []
        immediateRewards = []
        for a in range(actions):
            entries = np.zeros(states)
            for s in range(states):
                entries[s] = self.model.getReward(s, a)
            av = AlphaVector(a, entries)
            V.append(av)
            immediateRewards.append(av)
        stage = 1
        print(f"Stage 1: {len(V)} vectors")
        #print(f"The vectors are:")
        #for av in V :
        #    print(f"a={av.action} v={av.v}")
        #print("--------end of stage--------")
        # run the backup stage
        while (True):
            stage += 1
            Vnext = self.backupStage(immediateRewards, V, B)
            # until V has converged
            valueDifference = self.getValueDifference(B, V, Vnext)
            print(f"Stage {stage}: {len(Vnext)} vectors, diff: {valueDifference}")
            #print(f"The vectors are:")
            #for av in Vnext:
            #    print(f"a={av.action} v={av.v}")
            #print("--------end of stage--------")

            V = Vnext
            elapsedTime = round((time.time() - start_time) * 1000)
            if (valueDifference < self.model.convergence_tolerance) or (elapsedTime > self.model.time_limit):
                break
        # assigned converged V to gamma
        self.totalSolveTime = round((time.time() - start_time) * 1000)
        self.gamma = V
        # print alpha-vectors to a policy file

        self.expectedValue = getValue(self.model.initial_belief, V)
        print(f"total solve time = {self.getTotalSolveTime()} ms")
        print(f"expected value = {self.expectedValue}")

    def getValueDifference(self, B: list[BeliefPoint], V: list[AlphaVector], Vnext: list[AlphaVector]):
        maxDiff = float('-inf')

        for b in B:
            #print(f"]]]]]]]]]]]]]]]]calculating v_next[[[[[[[[[[[[[[[")
            Vnext_Val=getValue(b.getBelief(), Vnext)
            #print(f"[[[[[[[[[[[[[[[calculating v_val ]]]]]]]]]]]]]]]")
            V_val = getValue(b.getBelief(), V)
            diff = Vnext_Val - V_val
            #print(f"Vnext_val = {Vnext_Val} V_val= {V_val} diff={diff}")
            if diff > maxDiff:
                maxDiff = diff
        return maxDiff

    def backupStage(self, immediateRewards: list[AlphaVector], V: list[AlphaVector], B: list[BeliefPoint]):
        nActions=self.model.num_actions
        nObservations=self.model.num_observations
        nStates=self.model.num_states
        O = self.model.get_observation_matrix()
        T = self.model.get_transition_matrix()

        Vnext = []
        Btilde = B


        # initialise gkao vectors
        gkao = np.empty((len(V), nActions, nObservations), dtype=object)
        for k in range(len(V)):
            for a in range(nActions):
                for o in range(nObservations):
                    entries = np.zeros(nStates)
                    for s in range(nStates):
                        val = 0.0
                        for sPrime in range(nStates):
                            val += O[a][sPrime][o] * T[a][s][sPrime] * V[k].v[sPrime]
                        entries[s] = val
                    av = AlphaVector(a, entries)
                    gkao[k][a][o] = av
        assert len(gkao) == len(V)

        # run the backup stage
        while len(Btilde) > 0:
            # sample a belief point uniformly at random
            beliefIndex = np.random.randint(len(Btilde))
            b = Btilde[beliefIndex]

            # compute backup(b,V) to get max alpha vector
            alpha = self.backup(immediateRewards, gkao, b)

            # check if we need to add alpha
            oldValue = getValue(b.getBelief(), V)
            newValue = np.dot(b.getBelief(), alpha.v)

            if (newValue >= oldValue):
                assert alpha.action >= 0 and alpha.action < self.model.num_actions , f"invalid action {alpha.action}"
                Vnext.append(alpha)
            else:
                bestVectorIndex = getBestVectorIndex(b.getBelief(), V)
                Vnext.append(V[bestVectorIndex])

            # compute new Btilde containing non-improved belief points
            newBtilde = []
            for bp in B:
                oV = getValue(bp.getBelief(), V)
                nV = getValue(bp.getBelief(), Vnext)

                if nV < oV:
                    newBtilde.append(bp)
            Btilde = newBtilde
        return Vnext

    def backup(self, immediateRewards: list[AlphaVector], gkao: list[AlphaVector], b: BeliefPoint):
        nActions=self.model.num_actions
        nObservations=self.model.num_observations
        nStates=self.model.num_states
        O = self.model.get_observation_matrix()
        T = self.model.get_transition_matrix()

        ga = []
        for a in range(nActions):
            oVectors = []
            for o in range(nObservations):
                maxVal = float('-inf')
                maxVector = None

                K = len(gkao)
                for k in range(K):
                    product = np.dot(b.getBelief(), gkao[k][a][o].v)
                    if (product > maxVal):
                        maxVal = product
                        maxVector = gkao[k][a][o]

                assert maxVector != None
                oVectors.append(maxVector)

            assert len(oVectors) > 0

            # take sum of the vectors
            sumVector = oVectors[0]
            for j in range(1,len(oVectors)):
                sumVector = sumVectors(sumVector, oVectors[j])

            # multiply by discount factor
            sumVectorEntries = sumVector.v
            for s in range(nStates):
                sumVectorEntries[s] = self.model.discount * sumVectorEntries[s]

            sumVector.v = sumVectorEntries

            av  = sumVectors(immediateRewards[a], sumVector)
            ga.append(av)

        assert len(ga) == nActions

        # find the maximising vector
        maxVal = float('-inf')
        vFinal = None
        for av in ga:
            product = np.dot(b.getBelief(), av.v)
            if product > maxVal:
                maxVal = product
                vFinal = av

        assert vFinal != None
        return vFinal

    @staticmethod
    def select_action(belief, vector_set):
        """
        Compute optimal action given a belief distribution
        :param belief: dim(belief) == dim(AlphaVector)
        :param vector_set
        :return:
        """
        max_v = -np.inf
        best = None
        for av in vector_set:
            v = np.dot(av.v, belief)

            if v > max_v:
                max_v = v
                best = av

        if best is None:
            raise ValueError('Vector set should not be empty')

        return best.action, best