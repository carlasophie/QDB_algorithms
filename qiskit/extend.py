import numpy as np
from math import ceil
from qiskit.circuit.library import RYGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile

from prepare import prepare, string0


def extend(qc, l):
    """Extend the quantum database state with k indices to a state with k+l indices by adding one qubit.

    Args:
        qc (QuantumCircuit): quantum circuit representing the imbalanced quantum database state with k indices
        l (int): to be added number of new index states to the superposition, limited by added number of ancillas (here z=1)

    Returns:
        qc (QuantumCircuit): quantum circuit representing the quantum database state with (k + l) indices
    """
    #add one qubit to the quantum circuit
    #dim_index = ceil(np.log2(k))
    qr = QuantumRegister(1)
    t = len(qc.qubits)
    print("num qubits before extension = ", t)
    dim_index = len(qc.qubits) -1
    qc.add_register(qr)
    t_old = t

    #compute number of qubits the circuit is applied to based on qc
    t = len(qc.qubits)
    print("num qubits after extension = ", t)

    #get all the qubits in register qc
    qr = qc.qubits

    #compute number of qubits 
    dim_index = len(qr) - 1

    #compute angle 
    theta = 2*np.arccos(np.sqrt(1/(l+1.)))

    #define custom multi-controlled RY gate controlled by all zero string 0_{t-1}, ..., 0_{j+1} (notation as in A.1)
    ry_gate = RYGate(theta)
    num_controls = len(string0(dim_index))
    #print("num_controls = ", num_controls)
    multi_control_ry = ry_gate.control(num_controls, ctrl_state=string0(dim_index))
    qc.append(multi_control_ry, [qr[i] for i in np.arange(0,dim_index+1)[::1]])

    #apply gate only if l > 1 as we already added one state 
    if l > 1:
        # get P_(l) operation
        qc_pl = prepare(l, 0)
        dim_pl = len(qc_pl.qubits)
        pl_gate = qc_pl.to_gate()

        diff = t_old - dim_pl

        #append P_(l) gate to the circuit 
        multi_control_pl = pl_gate.control(1, ctrl_state="1")
        qc.append(multi_control_pl, [qr[dim_pl+diff]] + [qr[i] for i in np.arange(0+diff,dim_pl+diff) ]) 
        
    return qc




def run_qc2(qc, k, l, revert_end=False):
    """Runs the simulator including transpilation to get the statevector of the quantum circuit.

    Args:
        qc (QuantumCircuit): qiskit quantum circuit representing extended quantum state
        k (int): number of index states in the superposition state before extension
        l (int): number of newly added index states in the superposition state after extension
        revert_end (bool): if True, endian encoding is reverted otw. numpy array is returned as is

    Returns:
        psi (statevector): qiskit statevector of the quantum circuit
        psi_array (np.array): np.array statevector with endian encoding reverted
    """
    simulator = Aer.get_backend('statevector_simulator')
    result = simulator.run(transpile(qc, simulator)).result()
    psi = result.get_statevector(qc, decimals=4)
    
    psi_array = np.array(psi)
    if revert_end == True:
        psi_array = revert_endian(psi_array, k, l)
    return psi, psi_array

def revert_endian(statevector, k, l=0):
    """Revert endian encoding used in qiskit for statevector.

    Args:
        statevector (statevector): qiskit statevector for the qc result
        k (int): number of index states in the superposition state before extension
        l (int): number of newly added index states in the superposition state after extension, set to zero before extension 
        added_ancilla (bool): if True, the statevector has one newly added ancilla

    Returns:
        new_order_statevector (np.array): np.array statevector with reverted endian encoding 
    """
    #get number of qubits for encoding 
    t = ceil(np.log2(k)) 
    if l >= 1:
        t = t + 1
    all_str = []
    #loop over all possible strings (comp. basis states)
    for i in range(2**t):
        si1 = np.binary_repr((i), width=t) 
        all_str.append(si1)  
    
    if l >= 1:
        new_order_statevector = np.zeros(len(statevector))
        #check = np.zeros(len(statevector))
        for (i, si1) in enumerate(all_str):

            if si1[0]=="0":
                #remove first bit
                si = si1[1:]
                
                #reverse order
                si = si[::-1]
                int_si = int(si, 2)
                new_order_statevector[int_si] = (statevector[i]).real
                #check[i] = int_si
            
            if si1[0]=="1":
                #remove first bit
                si = si1[1:]
                #reverse order
                si = si[::-1]
                int_si = int(si, 2)
                new_order_statevector[int_si+2**(t-1)] = (statevector[i]).real
                #check[i] = int_si

    else:
        for i, e in enumerate(all_str):
            all_str[i] = int(e[::-1], 2)

        # order the statevector according to the numbers assigned in all_str
        new_order_statevector = np.zeros(len(statevector), dtype=complex)
        assert len(all_str) == len(statevector), "Length of statevector and all_str must be equal"
        for i, si in enumerate(all_str):
            new_order_statevector[si] = statevector[i]
            
    return new_order_statevector


if __name__ == "__main__":

    # define parameters k and l
    k = 22
    l = 7

    # check if k and l are valid inputs
    assert k > 1, "k must be strictly greater than 1"
    assert l >= 0, "l must be postive or zero"

    # here we are adding one qubit to the quantum circuit
    # thus l must be smaller than 2**ceil(np.log2(k))
    assert l <= 2**ceil(np.log2(k)), "l must be smaller than 2**ceil(np.log2(k))"

    # prepare and print the statevector before extension
    qc = prepare(k, l)
    psi_before, psi_array_before = run_qc2(qc, k, l=0, revert_end=True)
    print("statevector before extension:", psi_array_before)

    # extend the quantum circuit
    # print the complete quantum circuit diagram 
    qc = extend(qc, l)
    qc_cop = qc.copy()
    print(qc_cop.draw('text'))
    
    # print the statevector after extension
    psi, psi_array = run_qc2(qc, k, l, revert_end=True)
    print("statevector after extension:", psi_array )

    #check resulting state  
    print("Number of non-zero elements in the statevector: ", np.count_nonzero(psi_array), ", (k + l) =  (", (k+l), ")")
    #print("locations of nonzero elements", np.argwhere(psi_array))
    assert np.count_nonzero(psi_array) == (k+l), "Number of non-zero elements in the statevector must be equal to (k+l)"
    assert (psi_array[0] - 1/np.sqrt(k+l)) < 1e-4, "norm is incorrect"
    filename = './visualize_statevector_extension.pdf'
    psi.draw('city', filename=filename)
