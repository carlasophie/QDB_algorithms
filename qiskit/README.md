# Quantum Databases (QDB) with Qiskit

Simulation of the `prepare` and `extend` operation for a QDB with quantum indices and classical data as given in [arXiv:2405.14947](https://arxiv.org/abs/2405.14947). The implementation is focused on transforming the index register and can be simply extended to be applied to the data register also. Furthermore, the extension operation implements the specific case of adding one ancilla qubit ($z=1$) as described in the exemplary case of Algorithm 5 in [arXiv:2405.14947](https://arxiv.org/abs/2405.14947). 


## Installation

To install the dependencies, run:
```bash
pip3 install -r requirements.txt
```


## Simulation of the QDB operations prepare and extend 

```bash
python prepare.py
python extend.py
```

## To test the QDB operations prepare and extend use the Jupyter notebook
The jupyter notebook is given by [visualize_algs.ipynb](https://github.com/carlasophie/Quantum_DB/blob/main/qiskit/visualize_algs.ipynb).
