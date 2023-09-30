import numpy as np


class AlphaVector:
    """
    Simple wrapper for an alpha vector, used for representing the value function for a POMDP as a piecewise-linear,
    convex function
    """

    def __init__(self, a, v):
        self.action = a
        self.v = v

    def copy(self):
        return alphavector(self.action, self.v)

def getValue(belief: list, V: list[AlphaVector]):
    max = float('-inf')

    for i in range(len(V)):
        u = V[i]
        #print(f"belief={belief} V[{i}].v={u.v}")
        product = np.dot(belief, u.v)
        if product > max:
            max = product
    return max

def getBestVectorIndex(belief: list, U: list[AlphaVector]):
    wIndex = -1
    list_product = []
    for i in range(len(U)):
        u = U[i]
        product = np.dot(belief, u.v)
        list_product.append(product)
    return np.argmax(list_product)

def sumVectors(v1: AlphaVector, v2: AlphaVector):
    assert  len(v1.v) == len(v2.v)
    newEntries = np.empty(len(v1.v))

    for s in range(len(newEntries)):
        newEntries[s] = v1.v[s] + v2.v[s]

    assert v1.action == v2.action, "error: sum of vectors with differenct action!"
    action = v1.action

    newVector = AlphaVector(action, newEntries)
    return newVector


