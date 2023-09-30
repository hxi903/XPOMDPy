from __future__ import absolute_import

import time

from .solver import Solver
from momdpy.momdp import BeliefPointXY
from momdpy.momdp.alpha_matrix import  AlphaMatrix, sumMetrices, getVectorValue, getBestAlphaMatrices, getBestMatrixIndex, getValue
from pomdpy.solvers.ols import OLS
from pomdpy.solvers.rl_utils import random_weights, hypervolume, policy_evaluation_mo
from momdpy.parser.momdp_policy_writer import alphamatrices_to_policy
#from scipy.optimize import linprog
import numpy as np
#from itertools import product

from functools import reduce
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

    def perseus_ols(self, Tx, Ty, o, r, B):
        #This implement PerseusUpdate (see Alg.4)
        actions = self.model.num_actions  # |A| actions
        Xstates = self.model.num_Xstates  # |X| states
        Ystates = self.model.num_Ystates  # |Y| states

        # initialize gamma with a 0 alpha-matrix (a vector reward version of scalar alpha vector of POMDP)
        #dummy = AlphaMatrix(a=-1, vs=np.zeros((states, self.model.num_objectives)))
        #self.gamma.add(dummy)

        # create initial set of alpha-matrix sets and matrices set defining immediate rewards
        # in MR-MOMDP, each X state maintains different alpha matrix sets and immediate rewards sets
        A_all = [[] for x in range(Xstates)]
        immediateRewards = [[] for x in range(Xstates)]
        for a in range(actions):
            for xstate in range(Xstates) :
                entries = np.zeros((Ystates, self.model.num_objectives))
                for ystate in range(Ystates):
                    entries[ystate] = self.model.getReward(xstate, ystate, a)
                av = AlphaMatrix(a, entries)
                A_all[xstate].append(av)
                immediateRewards[xstate].append(av)

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
            print(f"Ar = {Ar}")
            backupMO_start = time.time()
            Aw =  self.solveScalarizedPOMDP(immediateRewards, Ar, B, w)
            backupMO_finish = time.time()
            backupMO_elapsed = backupMO_finish - backupMO_start
            time_backupMO.append(backupMO_elapsed)
            Q.WA[str(w)]=Aw  #store alphamatrices set A of weight w
            print(f"OLS weights: {w} "
                  f"The Aw vectors are:")
            for x in range(Xstates):
                print(f"Visible state X = {x}")
                for av in Aw[x]:
                    print(f"a={av.action} v={av.vs}")
            b0 = self.model.get_initial_belief_state_xy()
            Vb0 = getVectorValue(b0, Aw, w) #get scalarized value from initial belief
            for x in range(Xstates):
                A_all[x] += Aw[x]
            Q.add_solution(Vb0, w)
            print(f"hv: {hypervolume(np.zeros(Q.m), Q.ccs)} css: {Q.ccs} css_weights: {Q.ccs_weights}")
            elapsedTime = round((time.time() - start_time_perseus) * 1000)

        # assigned converged V to gamma
        self.totalSolveTime = round((time.time() - start_time_perseus) * 1000)
        if self.totalSolveTime > self.model.timeout:
            print(f"!!!!! OLS convergence timeout reached !!!!")
        self.gamma = Q.WA
        self.expectedValue = Q.ccs
        # find knee point of css
        knee_index= self.get_knee_point_index(Q.ccs)

        print(f"expected CCS value = {Q.ccs}")
        #print(f"maxi policy weights = {self.gamma.keys()}")
        print(f"CCS_weights = {Q.ccs_weights}")
        print(f"----------------------------------")
        print(f"CCS size = {len(Q.ccs)}")
        #print(f"knee css_weights = {Q.ccs_weights[knee_index]}")
        #print(f"knee CCS value = {Q.ccs[knee_index]}")
        print(f"total solve time = {self.getTotalSolveTime()}")
        average_backupMO = sum(time_backupMO) / len(time_backupMO)
        #print(f"SumbackupMO = {sum(time_backupMO)}")
        #print(f"SumbackupMO[0] = {time_backupMO[0]}")
        #print("Average time for backupMO: ", round(average_backupMO, 3))

        #writing all best alpha metrices to policy
        #Theta = Q.WA[str(Q.ccs_weights[knee_index])]
        #i=0
        #for w in Q.ccs_weights:
        #    alphamatrices_to_policy(w, Q.WA[str(w)], self.model.num_Ystates, self.model.policy_filename + '_'+ str(i))
        #    i += 1
        '''    
        if self.model.num_objectives == 3:
            Q.plot_ccs_xyz(Q.ccs, Q.ccs_weights)
        else:
            Q.plot_ccs_basic_xy(Q.ccs, Q.ccs_weights)
        '''
        #for w in Q.ccs_weights:
        #    idx=Q.get_set_max_policy_index(w)
        #    print(f" w = {w} index {idx} v{Q.ccs[idx]}")
        #print(f"sample WA[w] = {self.gamma[str(Q.ccs_weights[1])]}")

    def get_volume(self, item):
        vol = 1
        for obj in range(len(item)) :
            vol = vol * item[obj]
        return vol
    def get_knee_point_index(self, CSS):
        min_item = np.array(self.model.minimum_value)
        max_volume = self.get_volume(min_item)
        idx=0
        for item in CSS:
            feasible = item > min_item

            if all(feasible):
                print(f"feasible item {item} - {min_item}")
                vol = self.get_volume(item)
                if vol > max_volume:
                    max_volume = vol
                    #max_item = item
                    max_idx = idx
            idx+=1

        return(max_idx)
    def solveScalarizedPOMDP(self, immediateRewards, V: list[AlphaMatrix], B: list[BeliefPointXY], w: np.ndarray):
        Xstates = self.model.num_Xstates  # |X| states
        x_start_time = time.time()
        stage = 1
        lenV = reduce(lambda count, l : count + len(l), V, 0)
        print(f"Stage 1: {lenV} vectors")
        #for xstate in range(Xstates):
        #    print(f"Visible state (XState) : {xstate}")
        #    print(f"The vectors are:")
        #    for av in V[xstate] :
        #        print(f"a={av.action} v={av.vs}")
        #print("--------end of stage--------")
        # run the backup stage
        while (True):
            stage += 1
            Vnext = self.backupStage(immediateRewards, V, B, w)
            # until V has converged
            elapsedTime = round((time.time() - x_start_time) * 1000)
            valueDifference = self.getValueDifference(B, V, Vnext, w)
            lenVnext = reduce(lambda count, l : count + len(l), Vnext, 0)
            print(f"Stage {stage}: {lenVnext} vectors, diff: {valueDifference}")
            #for xstate in range(Xstates) :
            #    print(f"Visible state (XState) : {xstate}")
            #    print(f"The vectors are:")
            #    for av in V[xstate] :
            #        print(f"a={av.action} v={av.vs}")
            #print("--------end of stage--------")

            V = Vnext
            if (valueDifference < self.model.convergence_tolerance) or (elapsedTime > self.model.time_limit):
                break
        return V

    def getValueDifference(self, B: list[BeliefPointXY], V: list[AlphaMatrix], Vnext: list[AlphaMatrix], w: np.ndarray):
        maxDiff = float('-inf')

        for b in B:
            #print(f"]]]]]]]]]]]]]]]]calculating v_next[[[[[[[[[[[[[[[")
            Vnext_Val=getValue(b.getBeliefY(), Vnext[b.x_state], w)
            #print(f"[[[[[[[[[[[[[[[calculating v_val ]]]]]]]]]]]]]]]")
            V_val = getValue(b.getBeliefY(), V[b.x_state], w)
            diff = Vnext_Val - V_val
            #print(f"Vnext_val = {Vnext_Val} V_val= {V_val} diff={diff}")
            if diff > maxDiff:
                maxDiff = diff
        return maxDiff

    def backupStage(self, immediateRewards: list[AlphaMatrix], V: list[AlphaMatrix], B: list[BeliefPointXY], w: np.ndarray):
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
                        entries = np.zeros((nYstates, self.model.num_objectives))
                        for y in range(nYstates):
                            val = np.zeros(self.model.num_objectives)
                            for xPrime in range(nXstates):
                                for yPrime in range(nYstates):
                                    val += O[a][xPrime][yPrime][o] * Tx[a][x][y][xPrime] * Ty[a][x][y][yPrime] * V[x][k].vs[yPrime]
                            entries[y] = val
                        av = AlphaMatrix(a, entries)
                        gkao[k][a][o] = av
            gxkao[x] = gkao
            assert len(gkao) == len(V[x])

        # run the backup stage
        while len(Btilde) > 0 :
            # sample a belief point uniformly at random
            beliefIndex = np.random.randint(len(Btilde))
            b = Btilde[beliefIndex]

            # compute backup(b,V) to get max alpha vector
            alpha = self.backupMO(immediateRewards, gxkao, b, w)

            # check if we need to add alpha
            oldValue = getValue(b.getBeliefY(), V[b.x_state], w)
            newValue = getValue(b.getBeliefY(), [alpha], w)

            if (newValue >= oldValue) :
                assert alpha.action >= 0 and alpha.action < self.model.num_actions, f"invalid action {alpha.action}"
                Vnext[b.x_state].append(alpha)
            else :
                bestMatrixIndex = getBestMatrixIndex(b.getBeliefY(), V[b.x_state], w)
                Vnext[b.x_state].append(V[b.x_state][bestMatrixIndex])

            # compute new Btilde containing non-improved belief points
            newBtilde = []
            for bp in B :
                oV = getValue(bp.getBeliefY(), V[bp.x_state], w)
                nV = getValue(bp.getBeliefY(), Vnext[bp.x_state], w)

                if nV < oV :
                    newBtilde.append(bp)
            Btilde = newBtilde

        return Vnext

    def backupMO(self, immediateRewards: list[AlphaMatrix], gxkao: list[AlphaMatrix], b: BeliefPointXY, w: np.ndarray):
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
                    vms = np.multiply(w, gxkao[b.x_state][k][a][o].vs)
                    sum_vms = np.sum(vms, axis=1)
                    product = float(np.dot(b.getBeliefY(), sum_vms))
                    if (product > maxVal):
                        maxVal = product
                        maxVector = gxkao[b.x_state][k][a][o]

                assert maxVector != None
                oVectors.append(maxVector)

            assert len(oVectors) > 0

            # take sum of the vectors
            sumVector = oVectors[0]
            for j in range(1,len(oVectors)):
                sumVector = sumMetrices(sumVector, oVectors[j])

            # multiply by discount factor
            sumVectorEntries = sumVector.vs
            for y in range(nYstates):
                sumVectorEntries[y] = np.multiply(self.model.discount, sumVectorEntries[y])

            sumVector.vs = sumVectorEntries

            av  = sumMetrices(immediateRewards[b.x_state][a], sumVector)
            ga.append(av)

        assert len(ga) == nActions

        # find the maximising vector
        maxVal = float('-inf')
        vFinal = None
        for av in ga:
            vms = np.multiply(w, av.vs)
            sum_vms = np.sum(vms, axis=1)
            product = float(np.dot(b.getBeliefY(), sum_vms))
            if product > maxVal:
                maxVal = product
                vFinal = av

        assert vFinal != None
        return vFinal