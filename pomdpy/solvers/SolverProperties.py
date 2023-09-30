#!/usr/bin/env python
""" generated source for module SolverProperties """
# 
#  * SolvePOMDP
#  * Copyright (C) 2017 Erwin Walraven
#  *
#  * This program is free software: you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation, either version 3 of the License, or
#  * (at your option) any later version.
#  *
#  * This program is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  
# package: program
class SolverProperties(object):
    """ generated source for class SolverProperties """
    fixedStages = -1

    #  fixed number of stages, and -1 otherwise
    epsilon = float()

    #  vectors are included if d>epsilon
    valueFunctionTolerance = float()

    #  allowed bellman error
    acceleratedLPThreshold = int()

    #  accelerated LP is only used if |U| > threshold
    acceleratedLPTolerance = float()

    #  will be used to determine when the accelerated LP routine terminates
    coefficientThreshold = float()

    #  if absolute value of an LP coefficient is lower than threshold, it will be set to zero
    dumpPolicyGraph = bool()

    #  if true, then the solver writes a policy graph to a file
    dumpActionLabels = bool()

    #  if true, then the solver writes action labels rather than IDs
    workingDir = str()

    #  path of the working directory (empty if executed from IDE)
    outputDirName = str()

    #  name of the output directory, which should be a directory in workingDir
    timeLimit = float()

    #  time limit in seconds
    beliefSamplingRuns = int()

    #  belief sampling runs
    beliefSamplingSteps = int()

    #  belief sampling steps
    def getFixedStages(self):
        """ generated source for method getFixedStages """
        return self.fixedStages

    def setFixedStages(self, fixedStages):
        """ generated source for method setFixedStages """
        self.fixedStages = fixedStages

    def getEpsilon(self):
        """ generated source for method getEpsilon """
        return self.epsilon

    def setEpsilon(self, epsilon):
        """ generated source for method setEpsilon """
        self.epsilon = epsilon

    def getValueFunctionTolerance(self):
        """ generated source for method getValueFunctionTolerance """
        return self.valueFunctionTolerance

    def setValueFunctionTolerance(self, valueFunctionTolerance):
        """ generated source for method setValueFunctionTolerance """
        self.valueFunctionTolerance = valueFunctionTolerance

    def getAcceleratedLPThreshold(self):
        """ generated source for method getAcceleratedLPThreshold """
        return self.acceleratedLPThreshold

    def setAcceleratedLPThreshold(self, bendersThreshold):
        """ generated source for method setAcceleratedLPThreshold """
        self.acceleratedLPThreshold = bendersThreshold

    def getAcceleratedLPTolerance(self):
        """ generated source for method getAcceleratedLPTolerance """
        return self.acceleratedLPTolerance

    def setAcceleratedLPTolerance(self, bendersTolerance):
        """ generated source for method setAcceleratedLPTolerance """
        self.acceleratedLPTolerance = bendersTolerance

    def getCoefficientThreshold(self):
        """ generated source for method getCoefficientThreshold """
        return self.coefficientThreshold

    def setCoefficientThreshold(self, coefficientThreshold):
        """ generated source for method setCoefficientThreshold """
        self.coefficientThreshold = coefficientThreshold

    def dumpPolicyGraph(self):
        """ generated source for method dumpPolicyGraph """
        return self.dumpPolicyGraph

    def setDumpPolicyGraph(self, dumpPolicyGraph):
        """ generated source for method setDumpPolicyGraph """
        self.dumpPolicyGraph = dumpPolicyGraph

    def dumpActionLabels(self):
        """ generated source for method dumpActionLabels """
        return self.dumpActionLabels

    def setDumpActionLabels(self, dumpActionLabels):
        """ generated source for method setDumpActionLabels """
        self.dumpActionLabels = dumpActionLabels

    def getOutputDirName(self):
        """ generated source for method getOutputDirName """
        return self.outputDirName

    def setOutputDirName(self, outputDirName):
        """ generated source for method setOutputDirName """
        self.outputDirName = outputDirName

    def getOutputDir(self):
        """ generated source for method getOutputDir """
        if 0 == len(workingDir):
            return self.outputDirName
        else:
            return self.workingDir + "/" + self.outputDirName

    def getWorkingDir(self):
        """ generated source for method getWorkingDir """
        return self.workingDir

    def setWorkingDir(self, workingDir):
        """ generated source for method setWorkingDir """
        self.workingDir = workingDir

    def getTimeLimit(self):
        """ generated source for method getTimeLimit """
        return self.timeLimit

    def setTimeLimit(self, timeLimit):
        """ generated source for method setTimeLimit """
        self.timeLimit = timeLimit

    def getBeliefSamplingRuns(self):
        """ generated source for method getBeliefSamplingRuns """
        return self.beliefSamplingRuns

    def setBeliefSamplingRuns(self, beliefSamplingRuns):
        """ generated source for method setBeliefSamplingRuns """
        self.beliefSamplingRuns = beliefSamplingRuns

    def getBeliefSamplingSteps(self):
        """ generated source for method getBeliefSamplingSteps """
        return self.beliefSamplingSteps

    def setBeliefSamplingSteps(self, beliefSamplingSteps):
        """ generated source for method setBeliefSamplingSteps """
        self.beliefSamplingSteps = beliefSamplingSteps

