import numpy as np
from momdpy.momdp import BeliefPointXY
class AlphaMatrix:
    """
    Simple wrapper for an alpha matrix, used for representing the value function for a Multi-Reward POMDP as a piecewise-linear,
    convex function
    """
    def __init__(self, a, vs):
        assert type(vs) == list or type(vs) == np.ndarray
        self.action = a
        self.vs = vs

    def copy(self):
        return alphamatrix(self.action, self.vs)

    def __key(self):
        str_vs = ''.join(str(e) for e in self.vs)
        return (self.action, str_vs)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, AlphaMatrix):
            return self.__key() == other.__key()
        return NotImplemented

def getVectorValue_old(b: BeliefPointXY , V: list[AlphaMatrix] , w: np.ndarray ):
    max = float('-inf')
    max_vms = []
    for i in range(len(V[b.x_state])):
        u = V[b.x_state][i]
        vms = np.multiply(w, u.vs)
        sum_vms = np.sum(vms, axis=1)
        #print(f"GetValue belief={belief} V[{i}].v={u.vs} vms = {vms}")
        assert  len(b.getBeliefY())==len(sum_vms)
        product = float(np.dot(b.getBeliefY(), sum_vms))
        if product > max:
            max = product
            max_vms = np.dot(b.getBeliefY(), vms)
    return max_vms

def getVectorValue(b: BeliefPointXY , V: list[AlphaMatrix] , w: np.ndarray ):
    max = float('-inf')
    max_vms = []
    print(f"len of Aw[b0] = {len(V[b.x_state])}")
    for i in range(len(V[b.x_state])):
        u = V[b.x_state][i]
        vms = np.multiply(w, u.vs)
        sum_vms = np.sum(vms, axis=1)
        #print(f"GetValue belief={belief} V[{i}].v={u.vs} vms = {vms}")
        assert  len(b.getBeliefY())==len(sum_vms)
        product = float(np.dot(b.getBeliefY(), sum_vms))
        if product > max:
            max = product
            max_vms = np.dot(b.getBeliefY(), u.vs)
    return max_vms

def getValue(belief: list , V: list[AlphaMatrix] , w: np.ndarray ):
    max = float('-inf')
    max_vms = []
    for i in range(len(V)):
        u = V[i]
        vms = np.multiply(w, u.vs)
        sum_vms = np.sum(vms, axis=1)
        #print(f"GetValue belief={belief} V[{i}].v={u.vs} vms = {vms}")
        assert  len(belief)==len(sum_vms)
        product = float(np.dot(belief, sum_vms))
        if product > max:
            max = product
    return max

def getBestMatrixIndex(belief: list, U: list[AlphaMatrix], w: np.ndarray):
    wIndex = -1
    list_product = []
    for i in range(len(U)):
        u = U[i]
        #print(u)
        vms = np.multiply(w, u.vs)
        sum_vms = np.sum(vms, axis=1)
        product = np.dot(belief, sum_vms)
        #print(product)
        list_product.append(product)
    return np.argmax(list_product)

def sumMetrices(v1: AlphaMatrix, v2: AlphaMatrix):
    assert  len(v1.vs) == len(v2.vs)
    assert len(v1.vs[0]) == len(v2.vs[0])
    newEntries = np.empty((len(v1.vs), len(v1.vs[0])))

    for s in range(len(newEntries)):
        for x in range(len(newEntries[s])):
            newEntries[s][x] = v1.vs[s][x] + v2.vs[s][x]

    assert v1.action == v2.action, "error: sum of vectors with differenct action!"
    action = v1.action

    newMatrix = AlphaMatrix(action, newEntries)
    return newMatrix

def getBestAlphaMatrices(A_all: list[AlphaMatrix], B: list[BeliefPointXY], w: np.ndarray):
    Ar = [[]] * len(A_all)
    seen = [[]] * len(A_all)
    for b in B:
      m_idx=getBestMatrixIndex(b.getBeliefY(), A_all[b.x_state], w)
      # avoid duplicate AlphaMatrices in Ar
      if A_all[b.x_state][m_idx].__hash__() not in seen[b.x_state]:
          Ar[b.x_state].append(A_all[b.x_state][m_idx])
          seen[b.x_state].append(A_all[b.x_state][m_idx].__hash__())
    return Ar