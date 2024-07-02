import numpy as np
from math import ceil
from qiskit.circuit.library import RYGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile


def prepare(k,l, classical_register=False):
    """
    Args:
        k (int): number of index states in the superposition 
        l (int): number of new index states to be expected to be added in the superposition, added probability weigth to the |0> state
        classical_register (bool): if True, a classical register is added to the quantum circuit
    Returns:
        qc (QuantumCircuit): qiskit quantum circuit representing extended quantum state
    """

    #compute number of qubits the circuit is applied to 
    t = ceil(np.log2(k))  
    print("number of qubits t =", t)

    #binary representation of k-1 (Big-endian format)
    s = np.binary_repr((k-1), width=t) 
    p = (2**(t-1)+l)/(k+l)

    #initialize a quantum circuit with t qubits
    qr = QuantumRegister(t)
    #classical register if one may like to measure the state 
    if classical_register:
        cr = ClassicalRegister(t)
        qc = QuantumCircuit(qr, cr)
    else:
        qc = QuantumCircuit(qr)

    #note that qiskit uses little-endian representation of the qubits
    qubit_list = list(np.arange(0, t))

    #compute angle 
    theta = 2*np.arccos(np.sqrt(p))
    print("[W] RY gate, theta =", theta, "on qubit", 0)
    qc.ry(theta, qr[qubit_list[0]])

    #loop over range from 1 to (t-1) 
    for i in np.arange(1,t): 
        #define range from (t-2) to 0
        j = (t-1) - i 
        theta12 = 2*np.arccos(np.sqrt(1/2.))
        #apply layer of RY gates that form a balanced superposition and act Hadamard-like on the |0> state
        print("[Z] RY gate, theta =", theta12, "on qubit", i)
        qc.ry(theta12, i)


        if s[i] == '0':
            #define inverse RY rotation 
            ry_gate = RYGate(-theta12)

            #define custom multi-controlled RY gate
            num_controls=len(np.arange(0,i))
            multi_control_ry = ry_gate.control(num_controls, ctrl_state=s[:(i)][::-1])
            #apply custom multi-controlled RY gate
            qc.append(multi_control_ry, [qr[qubit_list[i]] for i in np.arange(0,i+1)])
            print("[A] controlled RY gate, p =", 1, "control states", s[:(i)])

        else:
            #define custom multi-controlled RY gate
            ry_gate = RYGate(-theta12)
            num_controls=len(np.arange(0,i))
            multi_control_ry = ry_gate.control(num_controls, ctrl_state=s[:(i)][::-1])
            #apply custom multi-controlled RY gate
            qc.append(multi_control_ry, [qr[qubit_list[i]] for i in (np.arange(0,i))] + [qr[qubit_list[i]]])

            #compute angle
            p = (2**j)/(int(s[i:], 2)+1.)
            theta1 = 2*np.arccos(np.sqrt(p))
            #define custom multi-controlled RY gate controlled by string s_{t-1}, ..., s_{j+1} (notation as in A.1)
            ry_gate = RYGate(theta1)
            num_controls=len(np.arange(0,i))
            multi_control_ry = ry_gate.control(num_controls, ctrl_state=s[:(i)][::-1])
            #apply custom multi-controlled RY gate
            qc.append(multi_control_ry, [qr[qubit_list[i]] for i in (np.arange(0,i))] + [qr[qubit_list[i]]])
            print("[B] controlled RY gate, p =", p, "control states", s[:(i)])

        #define custom multi-controlled RY gate controlled by all zero string 0_{t-1}, ..., 0_{j+1} (notation as in A.1)
        ry_gate = RYGate(-theta12)
        num_controls=len(np.arange(0,i))
        multi_control_ry = ry_gate.control(len(string0(i)), ctrl_state=string0(i))
        qc.append(multi_control_ry, [qr[qubit_list[i]] for i in np.arange(0,i+1)])

        p = (2**j + l)/(2**(j+1) + l*1.)
        theta1 = 2*np.arccos(np.sqrt(p))

        ry_gate = RYGate(theta1)
        num_controls=len(np.arange(0,i))
        multi_control_ry = ry_gate.control(len(string0(i)), ctrl_state=string0(i))
        qc.append(multi_control_ry, [qr[qubit_list[i]] for i in np.arange(0,i+1)])
        print("[C] controlled RY gate, p =", p, "control states", string0(i))
 
    return qc

def string0(n):
    """ Return string of n zeros.
    Args:
        n (int): number of zeros in the string
    Returns:
        (str): string of n zeros 
    """
    #return string of n zeros
    return '0' * n

def revert_endian_encoding(k, statevector):
    """Revert the endian encoding of the statevector.
    Args:
        k (int): number of qubits
        statevector (np.array): statevector
        
    Returns:    
        new_statevector (np.array): statevector with endian encoding reverted
    """
    # get order of the string
    t = ceil(np.log2(k)) 
    all_str = []
    for i in range(2**t):
        si = np.binary_repr((i), width=t) 
        si = si[::-1] # revert string
        # binary to decimal
        int_si = int(si, 2)
        all_str.append(int_si)

    # order the statevector according to the numbers assigned in all_str
    new_statevector = np.zeros(len(statevector), dtype=complex)
    assert len(all_str) == len(statevector), "Length of statevector and all_str must be equal"
    for i, si in enumerate(all_str):
        new_statevector[si] = statevector[i]

    return new_statevector



def run_qc(qc, num_indices, revert_end=True):
    """Runs the simulator including transpilation to get the statevector of the quantum circuit.

    Args:
        qc (QuantumCircuit): qiskit quantum circuit representing extended quantum state
        num_indices (int): number of index states in the superposition 
        revert_end (bool): if True, endian encoding is reverted otw. numpy array is returned as is

    Returns:
        psi (statevector): qiskit statevector of the quantum circuit
        psi_vector (np.array): np.array statevector with endian encoding reverted
    """
    simulator = Aer.get_backend('statevector_simulator')
    result = simulator.run(transpile(qc, simulator)).result()
    psi = result.get_statevector(qc, decimals=4)
    if revert_end:
        psi_array = revert_endian_encoding(num_indices, psi)
    else: 
        psi_array = np.array(psi)
    return psi, psi_array


if __name__ == "__main__":

    #execute example cases for prepare function

    #define parameters k and l
    k = 11
    l = 0

    assert k > 1, "k must be strictly greater than 1"
    assert l >= 0, "l must be postive or zero"


    #create example circuit
    qc = prepare(k,l)
    print(qc.draw('text'))
    
    #example run for specific k, l
    psi, psi_array = run_qc(qc, num_indices=k, revert_end=True)

    #visualize statevector and save as pdf
    psi2 = psi.copy()
    print("statevector: ", psi_array)

    print("Number of non-zero elements in the statevector: ", np.count_nonzero(psi_array), ", (k, l) =  (", k, ",", l, ")")
    assert np.count_nonzero(psi_array) == k, "Number of non-zero elements in the statevector must be equal to k"
    assert (psi_array[0] - 1/np.sqrt(k))< 1e-4, "norm is incorrect"
    filename = './visualize_statevector.pdf'
    psi2.draw('city', filename=filename)