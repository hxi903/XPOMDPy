from __future__ import absolute_import

import time

from .solver import Solver
from momdpy.momdp import BeliefPointXY
from momdpy.momdp.alpha_vector import AlphaVector, sumVectors, getBestVectorIndex, getValue
#from scipy.optimize import linprog
import numpy as np
#from itertools import product
from functools import reduce
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

    def perseus_pbvi(self, Tx, Ty, o, r, B):

        #This implement PerseusUpdate (see Alg.4)
        actions = self.model.num_actions  # |A| actions
        Xstates = self.model.num_Xstates  # |X| states
        Ystates = self.model.num_Ystates  # |Y| states


        # create initial alpha vector sets and vectors defining immediate rewards
        # in MOMDP each X_state has its own vector set, thus V will contain |X| vector sets.
        start_time = time.time()
        V = [[] for x in range(Xstates)]
        immediateRewards = [[] for x in range(Xstates)]

        for a in range(actions):
            for xstate in range(Xstates):
                entries = np.zeros(Ystates)
                for ystate in range(Ystates):
                    entries[ystate] = self.model.getReward(xstate, ystate, a)
                av = AlphaVector(a, entries)
                V[xstate].append(av)
                immediateRewards[xstate].append(av)

        stage = 1
        lenV = reduce(lambda count, l : count + len(l), V, 0)
        print(f"Stage 1: {lenV} vectors")
        #for xstate in range(Xstates):
        #    print(f"Visible state (XState) : {xstate}")
        #    print(f"The vectors are:")
        #    for av in V[xstate] :
        #        print(f"a={av.action} v={av.v}")
        #print("--------end of stage--------")

        # run the backup stage
        while (True) :
            stage += 1
            Vnext = self.backupStage(immediateRewards, V, B)
            # until V has converged

            valueDifference = self.getValueDifference(B, V, Vnext)
            lenVnext = reduce(lambda count, l : count + len(l), Vnext, 0)
            print(f"Stage {stage}: {lenVnext} vectors, diff: {valueDifference}")
            #for xstate in range(Xstates) :
            #    print(f"Visible state (XState) : {xstate}")
            #    print(f"The vectors are:")
            #    for av in V[xstate] :
            #        print(f"a={av.action} v={av.v}")
            #print("--------end of stage--------")

            V = Vnext
            elapsedTime = round((time.time() - start_time) * 1000)
            if (valueDifference < self.model.convergence_tolerance) or (elapsedTime > self.model.time_limit) :
                break
        # assigned converged V to gamma
        self.totalSolveTime = round((time.time() - start_time) * 1000)
        self.gamma = V
        b0 = self.model.initial_belief
        self.expectedValue = getValue(b0.getBeliefY(), V[b0.x_state])
        print(f"total solve time = {self.getTotalSolveTime()} ms")
        print(f"expected value = {self.expectedValue}")
        for v in V:
            print("alpha vector", v)

    def getValueDifference(self, B: list[BeliefPointXY], V: list[AlphaVector], Vnext: list[AlphaVector]):
        maxDiff = float('-inf')

        for b in B:
            #print(f"]]]]]]]]]]]]]]]]calculating v_next[[[[[[[[[[[[[[[")
            Vnext_Val=getValue(b.getBeliefY(), Vnext[b.x_state])
            #print(f"[[[[[[[[[[[[[[[calculating v_val ]]]]]]]]]]]]]]]")
            V_val = getValue(b.getBeliefY(), V[b.x_state])
            diff = Vnext_Val - V_val
            #print(f"Vnext_val = {Vnext_Val} V_val= {V_val} diff={diff}")
            if diff > maxDiff:
                maxDiff = diff
        return maxDiff
    def backupStage(self, immediateRewards: list[AlphaVector], V: list[AlphaVector], B: list[BeliefPointXY]):
        nActions=self.model.num_actions
        nObservations=self.model.num_observations
        nXstates=self.model.num_Xstates
        nYstates = self.model.num_Ystates
        O = self.model.get_observation_matrix()
        Tx = self.model.get_transition_matrix_Tx()
        Ty = self.model.get_transition_matrix_Ty()

        Vnext = [[] for x in range(nXstates)]
        Btilde = B

        # initialise gxkao vectors
        gxkao = [[] for x in range(nXstates)]
        for x in range(nXstates):
            gkao = np.empty((len(V[x]), nActions, nObservations), dtype=object)
            for k in range(len(V[x])):
                for a in range(nActions):
                    for o in range(nObservations):
                        entries = np.zeros(nYstates)
                        for y in range(nYstates):
                            val = 0.0
                            for xPrime in range(nXstates):
                                for yPrime in range(nYstates):
                                    val += O[a][xPrime][yPrime][o] * Tx[a][x][y][xPrime] * Ty[a][x][y][yPrime] * V[x][k].v[yPrime]
                            entries[y] = val
                        av = AlphaVector(a, entries)
                        gkao[k][a][o] = av
            gxkao[x] = gkao
            assert len(gkao) == len(V[x])

        # run the backup stage
        while len(Btilde) > 0 :
            # sample a belief point uniformly at random
            beliefIndex = np.random.randint(len(Btilde))
            b = Btilde[beliefIndex]


            # compute backup(b,V) to get max alpha vector
            alpha = self.backup(immediateRewards, gxkao, b)

            # check if we need to add alpha
            oldValue = getValue(b.getBeliefY(), V[b.x_state])
            newValue = np.dot(b.getBeliefY(), alpha.v)

            if (newValue >= oldValue) :
                assert alpha.action >= 0 and alpha.action < self.model.num_actions, f"invalid action {alpha.action}"
                Vnext[b.x_state].append(alpha)
            else :
                bestVectorIndex = getBestVectorIndex(b.getBeliefY(), V[b.x_state])
                Vnext[b.x_state].append(V[b.x_state][bestVectorIndex])

            # compute new Btilde containing non-improved belief points
            newBtilde = []
            for bp in B :
                oV = getValue(bp.getBeliefY(), V[bp.x_state])
                nV = getValue(bp.getBeliefY(), Vnext[bp.x_state])

                if nV < oV :
                    newBtilde.append(bp)
            Btilde = newBtilde

        return Vnext

    def backup(self, immediateRewards: list[AlphaVector], gxkao: list[AlphaVector], b: BeliefPointXY):
        nActions=self.model.num_actions
        nObservations=self.model.num_observations
        nXstates=self.model.num_Xstates
        nYstates = self.model.num_Ystates


        ga = []

        for a in range(nActions):
            oVectors = []
            for o in range(nObservations):
                maxVal = float('-inf')
                maxVector = None

                K = len(gxkao[b.x_state])
                for k in range(K):
                    product = np.dot(b.getBeliefY(), gxkao[b.x_state][k][a][o].v)
                    if (product > maxVal):
                        maxVal = product
                        maxVector = gxkao[b.x_state][k][a][o]

                assert maxVector != None
                oVectors.append(maxVector)

            assert len(oVectors) > 0

            # take sum of the vectors
            sumVector = oVectors[0]
            for j in range(1,len(oVectors)):
                sumVector = sumVectors(sumVector, oVectors[j])

            # multiply by discount factor
            sumVectorEntries = sumVector.v
            for y in range(nYstates):
                sumVectorEntries[y] = self.model.discount * sumVectorEntries[y]

            sumVector.v = sumVectorEntries

            av  = sumVectors(immediateRewards[b.x_state][a], sumVector)
            ga.append(av)

        assert len(ga) == nActions

        # find the maximising vector
        maxVal = float('-inf')
        vFinal = None
        for av in ga:
            product = np.dot(b.getBeliefY(), av.v)
            if product > maxVal:
                maxVal = product
                vFinal = av

        assert vFinal != None
        return vFinal