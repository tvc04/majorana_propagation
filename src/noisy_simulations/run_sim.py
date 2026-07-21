from typing import Optional
import torch
import sys
import json
import time
import random

import ffsim
import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})
import numpy as np

import qiskit
from qiskit import qasm2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector

import cirq
from cirq.contrib import qasm_import
import quimb as qu
import quimb.tensor as qtn


from pyscf import ao2mo, tools, cc

from simulate import simulate
from fes import fes_circuit
from h_chain import h_circuit

if __name__ == "__main__":

    test_num = int(sys.argv[1])     # 1 = FeS, 2 = H_chain

    noise = 0.01
    if len(sys.argv) == 3:
        noise = float(sys.argv[2])

    output_data = None
    mps = None

    backend_hw = "cpu"
    if torch.cuda.is_available() == True:
        backend_hw = "gpu"

    print(f"SIMULATING Fe4S4 using {backend_hw}")

    prefix = "fes" if test_num == 1 else "hc"
    filename = f"{prefix}_{noise}"

    if test_num == 1:
        qc = fes_circuit()  # LUCJ
    if test_num == 2:
        qc = h_circuit(16)

    mps, output_data, latencies, nqubits, errors = simulate(qc, verbose=True, backend=backend_hw, noise_rate=noise)

    output = {
        "n_qubits": nqubits,
        "n_layers": 1,
        "noise": noise,
        "data": output_data,
        "latencies": latencies,
        "errors": errors
    }

    with open(f"data/{filename}.json", "w") as f:
        json.dump(output, f, indent=4)

    qu.utils.save_to_disk(mps, f"product_states/{filename}.qu")
