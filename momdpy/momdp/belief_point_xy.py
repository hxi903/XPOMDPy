#!/usr/bin/env python
#
#  SolvePOMDPy (c) 2023 Hargyo Ignatius
#  This file is converted from SolvePOMDP written in Java @2017 by Erwin Walraven
#
# package: momdp

class BeliefPointXY(object):
    """
    Represents a single belief point from a belief sample set. This is suitable if you are not using a belief-tree structure
    """
    def __init__(self, x_state, y_belief):
        self.x_state = x_state
        self.belief_y = y_belief
        self.actionXstateObservationProbInitialized = False
        #  axoProbs[a][o] represents P(o|x,by,a)
        self.axoProbs = []
        self.axProbs = []
        self.history = []

    #
    # 	 Get array containing belief of Y states
    # 	 @return belief
    #

    def getBeliefY(self):
        return self.belief_y

    #   Return the X-state of current belief
    def getStateFromBelief(self):
        return self.x_state

    #
    # 	 * Get the belief for a specific state
    # 	 * @param s state ID
    # 	 * @return belief
    #
    def getBeliefFromState(self, y):
        assert y >=0 and y < len(self.belief_y)
        return self.belief_y[y]

    #
    # 	 * Add element to the history included in this belief point
    # 	 * @param i history element
    #
    def addToHistory(self, i):
        self.history.append(i)

    #
    # 	 * Get a list representing the full action-observation history
    # 	 * @return history
    #
    def getHistory(self):
        return self.history

    #
    # 	 * Set the full action-observation history
    # 	 * @param history list containing history
    #
    def setHistory(self, history):
        self.history = history

    #
    # 	 * Get a copy of the history included in this beliefpoint
    # 	 * @return history
    #
    def getHistoryCopy(self):
        newHistory = []
        newHistory = self.history
        return newHistory

    #
    # 	 * Get the hashcode of this beliefpoint, based on its history
    #
    def hashCode(self):
        return hash(self.history)

    #
    # 	 * Checks whether two belief points are identical based on their history list
    #
    def equals(self, o):
        if isinstance(o, BeliefPointXY):
            if len(self.history) != len(otherHistory):
                return False
            else:
                while i < len(self.history) and isEqual:
                    isEqual = isEqual and (self.history[i] == otherHistory[i])
                    i += 1
                return isEqual
        else:
            return False

    def toString(self):
        ret = "<BP("
        i = 0
        while len(self.belief_y):
            ret += self.belief_y[i] + ","
            i += 1
        return ret + ")>"

    #
    # 	 * Returns true if action observation probabilities have been initialized
    # 	 * @return
    #
    def hasActionXstateObservationProbabilities(self):
        return self.actionXstateObservationProbInitialized

    #
    # 	 * Sets the action observation probabilities for this belief
    # 	 * @param aoProbs action observation probabilities
    #
    def setActionXstateObservationProbabilities(self, axoProbs):
        assert self.axoProbs == [], "aoProbs has already assigned previously"
        self.axoProbs = axoProbs
        self.actionXstateObservationProbInitialized = True

    def setActionXstateProbabilities(self, axProbs):
        assert self.axProbs == [], "axProbs has already assigned previously"
        self.axProbs = axProbs
        self.actionXstateObservationProbInitialized = True
    #
    # 	 * Get action observation probability
    # 	 * @param a action
    # 	 * @param o observation
    # 	 * @return action observation probability for action a and observation o
    #
    def getActionXstateObservationProbability(self, a, x, o):
        assert self.axoProbs != []
        return self.axoProbs[a][x][o]

    def getActionXstateProbability(self, a, x):
        assert self.axProbs != []
        return self.axProbs[a][x]

    def getXstateProbability(self, a):
        assert self.axProbs != []
        return self.axProbs[a]