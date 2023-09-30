from __future__ import absolute_import

import time

from .solver import Solver
from pomdpy.pomdp import BeliefPoint
from .alpha_matrix import AlphaMatrix, sumMetrices, getVectorValue, getBestAlphaMatrices, getBestMatrixIndex, getValue
from .ols import OLS
from pomdpy.solvers.rl_utils import random_weights, hypervolume, policy_evaluation_mo
from pomdpy.parser.pomdp_policy_writer import alphamatrices_to_policy
#from scipy.optimize import linprog
from scipy.spatial import distance
import numpy as np
#from itertools import product
class PerseusOLS(Solver):
    def __init__(self, agent):
        """
        Initialize the OLS compliant Perseus PBVI approximate solver for Multi-Objective POMDP
        :param agent:
        :return:
        """
        super(PerseusOLS, self).__init__(agent)
        self.history = agent.histories.create_sequence()
        self.totalSolveTime = 0
        self.expectedValue = 0.0
        self.gamma=set()

    @staticmethod
    def reset(agent):
        return PerseusOLS(agent)

    def getTotalSolveTime(self):
        return self.totalSolveTime * 0.001

    def perseus_ols(self, t, o, r, B: list[BeliefPoint]):
        #This implement PerseusUpdate (see Alg.4)

        discount = self.model.discount
        actions = self.model.num_actions  # |A| actions
        states = self.model.num_states  # |S| states
        observations = len(self.model.get_all_observations())  # |Z| observations
        first = True

        # initialize gamma with a 0 alpha-vector
        dummy = AlphaMatrix(a=-1, vs=np.zeros((states, self.model.num_objectives)))
        self.gamma.add(dummy)


        # create initial set of matrix and matrices defining immediate rewards
        A_all = []
        immediateRewards = []
        for a in range(actions):
            entries = np.zeros((states, self.model.num_objectives))
            for s in range(states):
                entries[s] = self.model.getReward(s, a)
            av = AlphaMatrix(a, entries)
            A_all.append(av)
            immediateRewards.append(av)

        Q = OLS(m=self.model.num_objectives, epsilon=self.model.epsilon)
        start_time_perseus = time.time()
        elapsedTime = 0
        time_backupMO = []
        while not Q.ended() and not (elapsedTime > self.model.timeout):
            print(f"Queue = {Q.queue}")
            w = Q.next_w()
            print("w:", w)
            print(f"A_all = {A_all}")
            Ar = getBestAlphaMatrices(A_all, B, w) #select best AlphaMatrix for each belief in B
            #print(f"Ar {Ar}")
            backupMO_start = time.time()
            Aw =  self.solveScalarizedPOMDP(immediateRewards, Ar, B, w)
            backupMO_finish = time.time()
            backupMO_elapsed = backupMO_finish - backupMO_start
            time_backupMO.append(backupMO_elapsed)
            Q.WA[str(w)] = Aw  # store alphamatrices set A of weight w
            print(f"OLS weights: {w} "
                  f"The Aw vectors are:")
            for av in Aw:
                print(f"a={av.action} v={av.vs}")

            Vb0 = getVectorValue(self.model.get_initial_belief_state(), Aw, w) #get vector value from initial belief
            A_all += Aw
            Q.add_solution(Vb0, w)
            print(f"hv: {hypervolume(np.zeros(Q.m), Q.ccs)} css: {Q.ccs} css_weights: {Q.ccs_weights}")
            elapsedTime = round((time.time() - start_time_perseus) * 1000)
            print(f"elapsed time= {elapsedTime} ms")

        # assigned converged V to gamma
        self.totalSolveTime = round((time.time() - start_time_perseus) * 1000)
        if self.totalSolveTime > self.model.timeout:
            print(f"!!!!! OLS convergence timeout reached !!!!")
        self.gamma = Q.WA
        self.expectedValue = Q.ccs

        # find knee point of css
        knee_index= self.get_knee_point_index(Q.ccs)

        '''
        if self.model.num_objectives == 3 :
            Q.plot_ccs_xyz(Q.ccs, Q.ccs_weights)
        else :
            Q.plot_ccs_basic_xy(Q.ccs, Q.ccs_weights)
        '''
        print(f"expected CCS value = {Q.ccs}")
        #print(f"maxi policy weights = {self.gamma.keys()}")
        print(f"css_weights = {Q.ccs_weights}")
        print(f"----------------------------------")
        print(f"CCS size = {len(Q.ccs)}")
        #print(f"knee css_weights = {Q.ccs_weights[knee_index]}")
        #print(f"knee CCS value = {Q.ccs[knee_index]}")
        print(f"total solve time = {self.getTotalSolveTime()}")
        average_backupMO = sum(time_backupMO) / len(time_backupMO)
        #print(f"SumbackupMO = {sum(time_backupMO)}")
        #print(f"SumbackupMO[0] = {time_backupMO[0]}")
        #print("Average time for backupMO: ", round(average_backupMO, 3))
        #writing up policy file
        #Theta = Q.WA[str(Q.ccs_weights[knee_index])]
        #alphamatrices_to_policy(Theta, self.model.num_states, self.model.policy_filename)

        #i=0
        #for w in Q.ccs_weights:
        #    alphamatrices_to_policy(w, Q.WA[str(w)], self.model.num_states, self.model.policy_filename + '_'+ str(i))
        #    i += 1

    def get_volume(self, item):
        vol = 1
        for obj in range(len(item)) :
            vol = vol * item[obj]
        return vol
    def get_knee_point_index(self, CSS):
        max_item = np.array(self.model.minimum_value)
        max_volume = self.get_volume(max_item)
        idx=0
        for item in CSS:
            threshold=max_item
            feasible = item > threshold

            if all(feasible):
                print(f"feasible item {item} - {max_item}")
                vol = self.get_volume(item)
                if vol > max_volume:
                    max_volume = vol
                    #max_item = item
                    max_idx = idx
            idx+=1

        return(max_idx)

    def solveScalarizedPOMDP(self, immediateRewards, V: list[AlphaMatrix], B: list[BeliefPoint], w: np.ndarray):
        x_start_time = time.time()
        stage = 1
        print(f"Stage 1: {len(V)} metrices")
        #print(f"The metrices are:")
        #for av in V :
        #    print(f"a={av.action} vs={av.vs}")
        #print("--------end of stage--------")
        # run the backup stage
        while (True):
            stage += 1
            Vnext = self.backupStage(immediateRewards, V, B, w)
            # until V has converged
            elapsedTime = round((time.time() - x_start_time) * 1000)
            valueDifference = self.getValueDifference(B, V, Vnext, w)
            print(f"Stage {stage}: {len(Vnext)} vectors, diff: {valueDifference}")
            #print(f"The vectors are:")
            #for av in Vnext:
            #    print(f"a={av.action} v={av.vs}")
            #print("--------end of stage--------")

            V = Vnext
            if (valueDifference < self.model.convergence_tolerance) or (elapsedTime > self.model.time_limit):
                break
        return V

    def getValueDifference(self, B: list[BeliefPoint], V: list[AlphaMatrix], Vnext: list[AlphaMatrix], w: np.ndarray):
        maxDiff = float('-inf')

        for b in B:
            #print(f"]]]]]]]]]]]]]]]]calculating v_next[[[[[[[[[[[[[[[")
            Vnext_Val=getValue(b.getBelief(), Vnext, w)
            #print(f"[[[[[[[[[[[[[[[calculating v_val ]]]]]]]]]]]]]]]")
            V_val = getValue(b.getBelief(), V, w)
            diff = Vnext_Val - V_val
            #print(f"Vnext_val = {Vnext_Val} V_val= {V_val} diff={diff}")
            if diff > maxDiff:
                maxDiff = diff
        return maxDiff

    def backupStage(self, immediateRewards: list[AlphaMatrix], V: list[AlphaMatrix], B: list[BeliefPoint], w: np.ndarray):
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
                    entries = np.zeros((nStates, self.model.num_objectives))
                    for s in range(nStates):
                        val = np.zeros(self.model.num_objectives)
                        for sPrime in range(nStates):
                            val += np.multiply(O[a][sPrime][o], np.multiply(T[a][s][sPrime], V[k].vs[sPrime]))
                        entries[s] = val
                    av = AlphaMatrix(a, entries)
                    gkao[k][a][o] = av
        assert len(gkao) == len(V)

        # run the backup stage
        while len(Btilde) > 0:
            # sample a belief point uniformly at random
            beliefIndex = np.random.randint(len(Btilde))
            b = Btilde[beliefIndex]

            # compute backup(b,V) to get max alpha vector
            alpha = self.backupMO(immediateRewards, gkao, b, w)

            # check if we need to add alpha
            oldValue = getValue(b.getBelief(), V, w)
            newValue = getValue(b.getBelief(), [alpha], w)

            if (newValue >= oldValue):
                assert alpha.action >= 0 and alpha.action < self.model.num_actions , f"invalid action {alpha.action}"
                Vnext.append(alpha)
            else:
                bestMatrixIndex = getBestMatrixIndex(b.getBelief(), V, w)
                Vnext.append(V[bestMatrixIndex])

            # compute new Btilde containing non-improved belief points
            newBtilde = []
            for bp in B:
                oV = getValue(bp.getBelief(), V, w)
                nV = getValue(bp.getBelief(), Vnext, w)

                if nV < oV:
                    newBtilde.append(bp)
            Btilde = newBtilde
        return Vnext

    def backupMO(self, immediateRewards: list[AlphaMatrix], gkao: list[AlphaMatrix], b: BeliefPoint, w: np.ndarray):
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
                    vms = np.multiply(w, gkao[k][a][o].vs)
                    sum_vms = np.sum(vms, axis=1)
                    product = float(np.dot(b.getBelief(), sum_vms))
                    if (product > maxVal):
                        maxVal = product
                        maxVector = gkao[k][a][o]

                assert maxVector != None
                oVectors.append(maxVector)

            assert len(oVectors) > 0

            # take sum of the vectors
            sumVector = oVectors[0]
            for j in range(1,len(oVectors)):
                sumVector = sumMetrices(sumVector, oVectors[j])

            # multiply by discount factor
            sumVectorEntries = sumVector.vs
            for s in range(nStates):
                sumVectorEntries[s] = np.multiply(self.model.discount, sumVectorEntries[s])

            sumVector.vs = sumVectorEntries

            av  = sumMetrices(immediateRewards[a], sumVector)
            ga.append(av)

        assert len(ga) == nActions

        # find the maximising vector
        maxVal = float('-inf')
        vFinal = None
        for av in ga:
            vms = np.multiply(w, av.vs)
            sum_vms = np.sum(vms, axis=1)
            product = float(np.dot(b.getBelief(), sum_vms))
            if product > maxVal:
                maxVal = product
                vFinal = av

        assert vFinal != None
        return vFinal

    @staticmethod
    def select_action(belief, vector_set):
        """
        Compute optimal action given a belief distribution
        :param belief: dim(belief) == dim(AlphaMatrix)
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