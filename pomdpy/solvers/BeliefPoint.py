#!/usr/bin/env python
# 
#  SolvePOMDPy (c) 2023 Hargyo Ignatius
#  Converted from SolvePOMDP written in Java @2017 by Erwin Walraven
#  
# package: solver

class BeliefPoint(object):


    def __init__(self, belief):
        self.belief = belief
        self.actionObservationProbInitialized = False
        #  aoProbs[a][o] represents P(o|b,a)
        self.aoProbs = []
        self.history = []

    # 
    # 	 Get array containing belief
    # 	 @return belief
    # 	 

    def getBelief(self):
        return self.belief

    # 
    # 	 * Get the belief for a specific state
    # 	 * @param s state ID
    # 	 * @return belief
    # 	 
    def getBeliefFromState(self, s):
        assert s >=0 and s < len(self.belief)
        return self.belief[s]

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
        if isinstance(o, BeliefPoint):
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
        while len(belief):
            ret += self.belief[i] + ","
            i += 1
        return ret + ")>"

    # 
    # 	 * Returns true if action observation probabilities have been initialized
    # 	 * @return
    # 	 
    def hasActionObservationProbabilities(self):
        return self.actionObservationProbInitialized

    # 
    # 	 * Sets the action observation probabilities for this belief
    # 	 * @param aoProbs action observation probabilities
    # 	 
    def setActionObservationProbabilities(self, aoProbs):
        assert self.aoProbs == [], "aoProbs has already assigned previously"
        self.aoProbs = aoProbs
        self.actionObservationProbInitialized = True

    # 
    # 	 * Get action observation probability
    # 	 * @param a action
    # 	 * @param o observation
    # 	 * @return action observation probability for action a and observation o
    # 	 
    def getActionObservationProbability(self, a, o):
        assert self.aoProbs != []
        return self.aoProbs[a][o]
