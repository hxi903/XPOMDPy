"""
Write a set of alpha-vectors to into a policy (XML) file
(c) 2023, Hargyo Ignatius
TODO:
-alphamatrices to policy for MR-POMDP
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from pomdpy.solvers.alpha_vector import AlphaVector
from pomdpy.solvers.alpha_matrix import AlphaMatrix
def alphavectors_to_policy(AlphaVectorSet: list, no_entries: int, filename: str):
    # This is the parent (root) tag
    # onto which other tags would be
    # created
    data = ET.Element('Policy')
    data.set('version', '0.1')
    data.set('type', 'value')
    data.set('model', 'ModelName')
    data.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    data.set('xsi:noNamespaceSchemaLocation', 'policyx.xsd')

    # Adding a subtag named `AlphaVector`
    # inside our root (i.e. policy) tag
    element1 = ET.SubElement(data, 'AlphaVector')
    element1.set('vectorLength', str(no_entries))
    element1.set('numObsValue', '1') #POMDP
    element1.set('numVectors', str(len(AlphaVectorSet)))


    for av in AlphaVectorSet:
        # Adding subtags under the `AlphaVector`
        # subtag
        s_elem1 = ET.SubElement(element1, 'Vector')

        # Adding attributes to the tags under
        # `items`
        s_elem1.set('action', str(av.action))
        s_elem1.set('obsValue', '0')  # set 0 if POMDP

        # Adding text between <Vector></Vector>
        # subtag
        entry = str(av.v).replace('[','')
        entry = entry.replace(']','')
        s_elem1.text = str(entry)

    # creating an XML file
    tree = ET.ElementTree(data)

    xmlstr = minidom.parseString(ET.tostring(data)).toprettyxml(indent="   ")
    with open(filename, "w") as f :
        f.write(xmlstr)

def alphamatrices_to_scalarized_policy(AlphaMatrixSet: list, no_entries: int, weights: list, filename: str):
    # This is the parent (root) tag
    # onto which other tags would be
    # created
    data = ET.Element('Policy')
    data.set('version', '0.1')
    data.set('type', 'value')
    data.set('model', 'ModelName')
    data.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    data.set('xsi:noNamespaceSchemaLocation', 'policyx.xsd')

    # Adding a subtag named `AlphaVector`
    # inside our root (i.e. policy) tag
    element1 = ET.SubElement(data, 'AlphaVector')
    element1.set('vectorLength', str(no_entries))
    element1.set('numObsValue', '1') #POMDP
    element1.set('numVectors', str(len(AlphaMatrixSet)))

    for av in AlphaMatrixSet:
        # Adding subtags under the `AlphaVector`
        # subtag
        s_elem1 = ET.SubElement(element1, 'Vector')

        # Adding attributes to the tags under
        # `items`
        s_elem1.set('action', str(av.action))
        s_elem1.set('obsValue', '0')  # set 0 if POMDP

        # apply weights vector to the value vector
        new_val = ''
        for values in av.vs :
            scalar_value = np.dot(weights, values)
            entry = str(scalar_value).replace('[', '')
            entry = entry.replace(']', '')
            new_val = new_val + ' ' + entry
        # Adding text between <Vector></Vector>
        # subtag
        s_elem1.text = new_val.strip()

    # creating an XML file
    tree = ET.ElementTree(data)

    xmlstr = minidom.parseString(ET.tostring(data)).toprettyxml(indent="   ")
    with open(filename, "w") as f :
        f.write(xmlstr)

def alphamatrices_to_policy(weight_vector, AlphaMatrixSet: list, vector_entries: int, filename: str):
    # This is the parent (root) tag
    # onto which other tags would be
    # created
    data = ET.Element('Policy')
    data.set('version', '0.1')
    data.set('type', 'value')
    data.set('model', 'ModelName')
    data.set('weight', str(weight_vector))
    data.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    data.set('xsi:noNamespaceSchemaLocation', 'policyx.xsd')

    # Adding a subtag named `AlphaVector`
    # inside our root (i.e. policy) tag
    element1 = ET.SubElement(data, 'AlphaMatrix')
    element1.set('vectorLength', str(vector_entries))
    element1.set('numObsValue', '1') #POMDP
    element1.set('numVectors', str(len(AlphaMatrixSet)))

    for av in AlphaMatrixSet:
        # Adding subtags under the `AlphaVector`
        # subtag
        s_elem1 = ET.SubElement(element1, 'Matrix')
        # Adding attributes to the tags under
        # `items`
        s_elem1.set('action', str(av.action))
        s_elem1.set('obsValue', '0')  # set 0 if POMDP

        new_val = ''
        for values in av.vs :
            entry = str(values).replace('[', '')
            entry = entry.replace(']', '')
            entry = entry.replace('  ', ' ')
            new_val = new_val + ';' + entry
        # Adding text between <Matrix></Matrix>
        # subtag
        s_elem1.text=new_val.strip(';')

    # creating an XML file
    tree = ET.ElementTree(data)

    xmlstr = minidom.parseString(ET.tostring(data)).toprettyxml(indent="   ")
    with open(filename, "w") as f :
        f.write(xmlstr)
    f.close()

if __name__ == "__main__":
    # testing alphavectors_to_policy()
    a1 = AlphaVector(1, np.array([0.5, 0.7]))
    a2 = AlphaVector(1, np.array([0.125, 0.317]))

    A=[a1, a2]
    alphavectors_to_policy(A, 2, "test_pomdp.xml")

    # testing multi_rewards_alphavectros_to_policy()
    weights = [0.8, 0.2]
    a1 = AlphaMatrix(1, np.array([[0.5, 0.7], [1, 7], [10, 10]]))
    a2 = AlphaMatrix(1, np.array([[0.125, 0.317], [125, 317], [10, 10]]))
    a3 = AlphaMatrix(2, np.array([[0.675, 0.754], [675, 754], [10, 10]]))
    a4 = AlphaMatrix(1, np.array([[0.125, 0.977], [125, 977], [10, 10]]))
    a5 = AlphaMatrix(2, np.array([[0.1235, 0.1547], [1235, 1547], [10, 10]]))

    A=[a1, a2, a3, a4, a5]
    alphamatrices_to_policy(A, 3, "test_mrpomdp.xml")
    alphamatrices_to_scalarized_policy(A,3,weights,'test_scalarised_mrpomdp.xml')


