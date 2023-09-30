from __future__ import print_function, division
import numpy as np
import time
import logging
import os
from pomdpy.pomdp import Statistic
from pomdpy.pomdp.history import Histories, HistoryEntry
from pomdpy.util import console, print_divider
from experiments.scripts.pickle_wrapper import save_pkl
from momdpy.momdp import BeliefPointXY
module = "agent"


class MOMDP_Agent:
    """
    Train and store experimental results for different types of runs
    """

    def __init__(self, model, solver):
        """
        Initialize the POMDPY agent for MOMDP
        :param model:
        :param solver:
        :return:
        """
        self.logger = logging.getLogger('POMDPy_MOMDP.Solver')
        self.model = model
        self.results = Results()
        self.experiment_results = Results()
        self.histories = Histories()
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        self.solver_factory = solver.reset  # Factory method for generating instances of the solver

    def discounted_return(self):

        if self.model.solver == 'PerseusOLS':
            solver = self.solver_factory(self)

            self.run_perseus_ols(solver,1)
        if self.model.solver == 'PerseusPBVI':
            solver = self.solver_factory(self)

            self.run_perseus_pbvi(solver,1)
    def prepareBelief(self, b: BeliefPointXY) :
        assert b != None
        if b.hasActionXstateObservationProbabilities():
            return
        nActions = self.model.num_actions
        nObservations = self.model.num_observations
        nXstates = self.model.num_Xstates
        nYstates = self.model.num_Ystates
        Tx = self.model.get_transition_matrix_Tx()
        Ty = self.model.get_transition_matrix_Ty()

        O = self.model.get_observation_matrix()


        axProbs = np.zeros((nActions, nXstates))
        axoProbs = np.zeros((nActions, nXstates, nObservations))

        x = b.x_state

        for a in range(nActions) :
            for xNext in range(nXstates) :
                probX = 0
                pX = 0
                for y in range(nYstates) :
                    #print(f"printtt {a} {x} {y} {xNext}")
                    #print(Tx[a][x][y][xNext])
                    pX += Tx[a][x][y][xNext] * b.getBeliefFromState(y)
                probX += pX
                axProbs[a][xNext] = probX
        b.setActionXstateProbabilities(axProbs)

        for a in range(nActions) :
            for xNext in range(nXstates) :
                for o in range(nObservations) :
                    probO = 0
                    for yNext in range(nYstates) :
                        pO = 0
                        for y in range(nYstates) :
                            pO += Tx[a][x][y][xNext] * Ty[a][x][y][yNext] * b.getBeliefFromState(y)
                        probO += O[a][xNext][yNext][o] * pO
                    axoProbs[a][xNext][o] = probO
        b.setActionXstateObservationProbabilities(axoProbs)
        '''
        for a in range(nActions) :
            for o in range(nObservations) :
                probO = 0
                for xNext in range(nXstates) :
                    for yNext in range(nYstates) :
                        pO = 0
                        for y in range(nYstates) :
                            pO += Tx[a][x][y][xNext] * Ty[a][x][y][yNext] * b.getBeliefFromState(y)
                        probO += O[a][xNext][yNext][o] * pO
                aoProbs[a][o] = probO
        b.setActionObservationProbabilities(aoProbs)
        '''



    def updateBelief(self, b: BeliefPointXY, a, xNext, o):
        assert a < self.model.num_actions and o < self.model.num_observations
        Tx = self.model.get_transition_matrix_Tx()
        Ty = self.model.get_transition_matrix_Ty()
        O = self.model.get_observation_matrix()

        # check if belief point has been prepared
        if not b.hasActionXstateObservationProbabilities():
            self.prepareBelief(b)

        newBeliefY = np.empty((self.model.num_Ystates))

        # compute normalising constant
        nc = b.getActionXstateObservationProbability(a, xNext, o)
        assert nc > 0.0, "o can not be observed when executing a in belief b"

        # compute the new belief vector
        x = b.x_state
        for yNext in range(self.model.num_Ystates) :
            beliefEntry = 0.0
            for y in range(self.model.num_Ystates) :
                beliefEntry += Tx[a][x][y][xNext] * Ty[a][x][y][yNext] * b.getBeliefFromState(y)
            newBeliefY[yNext] = beliefEntry * O[a][xNext][yNext][o] / nc

        return BeliefPointXY(xNext, newBeliefY)

    def random_explore(self, n: int, b0: BeliefPointXY) :
        B = []
        Bset = set()
        b = b0
        B.append(b)
        Bset.add(b)

        while len(B) < (n - (self.model.num_Xstates * self.model.num_Ystates)) :
            self.prepareBelief(b)
            # select action and observation
            action = np.random.choice(self.model.get_action_set())

            # select nextX following Pr(x'|x,by,a)
            Xprime = np.random.choice(self.model.Xstates, p=b.getXstateProbability(action))
            newX = self.model.Xstates.index(Xprime)

            o_item = []
            o_ps = []
            for o in range(self.model.num_observations) :
                prob = b.getActionXstateObservationProbability(action, newX, o)
                if prob > 1.0 : prob = 1.0
                o_item.append(o)
                o_ps.append(prob)
            #print(f"o_item {o_item} o_ps {o_ps}")
            o_ps = np.array(o_ps)
            o_ps /= o_ps.sum()  # normalise

            # select observation folowing Pr(o|x, by,a) distribution
            observation = np.random.choice(o_item, p=o_ps)

            # find new belief point
            bao = self.updateBelief(b, action, newX, observation)
            bao.setHistory(b.getHistoryCopy())
            bao.addToHistory(action)
            bao.addToHistory(observation)

            # add belief point and prepare for next step
            if bao not in Bset :
                B.append(bao)
                Bset.add(bao)

            b = bao

        # add corner belief
        print(f"len B before corner {len(B)}")
        for x in range(self.model.num_Xstates):
            for y in range(self.model.num_Ystates) :
                beliefEntries = np.zeros(self.model.num_Ystates)
                beliefEntries[y] = 1.0
                B.append(BeliefPointXY(x,beliefEntries))

        return B

    def run_perseus_ols(self, solver, epoch):
        run_start_time = time.time()

        reward = 0
        discounted_reward = 0
        discount = 1.0

        print("=== RUN Perseus OLS MR-[M]OMDP SOLVER ===")
        print("Belief sampling started ...")
        b0 = self.model.get_initial_belief_state_xy()
        print(f"initial belief (x, by) x_state={b0.x_state} belief_y={b0.belief_y}")

        # RandomExplore(n) (see Alg.4)
        assert self.model.sampling_belief_size >= (self.model.num_Xstates * self.model.num_Ystates), "error: sampling belief size at least = Xstate * Ystate"
        B = self.random_explore(self.model.sampling_belief_size, b0)
        for b in B:
            print(f"b x={b.x_state} by={b.belief_y}")
        print(f"Number of belief: {len(B)}")

        # We implement OLSAR() (outer loop) and backupMO() (inner loop in solver.persues_ols()
        solver.perseus_ols(self.model.get_transition_matrix_Tx(), self.model.get_transition_matrix_Ty,
                            self.model.get_observation_matrix(),
                            self.model.get_reward_matrix(), B)

    def run_perseus_pbvi(self, solver, epoch) :
        run_start_time = time.time()

        reward = 0
        discounted_reward = 0
        discount = 1.0

        print("=== RUN Perseus PBVI MR-[M]OMDP SOLVER ===")
        print("Belief sampling started ...")
        b0 = self.model.initial_belief
        #print(f"initial belief (x, by) x_state={b0.x_state} belief_y={b0.belief_y}")

        # RandomExplore(n) (see Alg.4)
        assert self.model.sampling_belief_size >= (self.model.num_Xstates * self.model.num_Ystates), "error: sampling belief size at least = Xstate * Ystate"
        B = self.random_explore(self.model.sampling_belief_size, b0)
        for b in B:
            print(f"b x={b.x_state} by={b.belief_y}")
        print(f"Number of belief: {len(B)}")

        # PerseusUpdate(B,V) (see Alg.4)

        solver.perseus_pbvi(self.model.get_transition_matrix_Tx(), self.model.get_transition_matrix_Ty(),
                              self.model.get_observation_matrix(), self.model.get_reward_matrix(), B)
        self.results.time.add(time.time() - run_start_time)
        #self.results.update_reward_results(reward, discounted_reward)

        # Pretty Print results
        #self.results.show(epoch)

class Results(object):
    """
    Maintain the statistics for each run
    """
    def __init__(self):
        self.time = Statistic('Time')
        self.discounted_return = Statistic('discounted return')
        self.undiscounted_return = Statistic('undiscounted return')

    def update_reward_results(self, r, dr):
        self.undiscounted_return.add(r)
        self.discounted_return.add(dr)

    def reset_running_totals(self):
        self.time.running_total = 0.0
        self.discounted_return.running_total = 0.0
        self.undiscounted_return.running_total = 0.0

    def show(self, epoch):
        print_divider('large')
        print('\tEpoch #' + str(epoch) + ' RESULTS')
        print_divider('large')
        console(2, module, 'discounted return statistics')
        print_divider('medium')
        self.discounted_return.show()
        print_divider('medium')
        console(2, module, 'undiscounted return statistics')
        print_divider('medium')
        self.undiscounted_return.show()
        print_divider('medium')
        console(2, module, 'Time')
        print_divider('medium')
        self.time.show()
        print_divider('medium')