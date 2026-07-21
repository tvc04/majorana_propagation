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



def fes_circuit(local: bool = True):
    fcidump_filename = "fcidump_Fe4S4_MO.txt"

    mf_as = tools.fcidump.to_scf(fcidump_filename)
    mf_as.max_cycle = 100
    mf_as.conv_tol = 1e-9
    mf_as = mf_as.newton()
    mf_as.kernel()
    assert mf_as.converged, "SCF did not converge"
    h1e = mf_as.get_hcore()

    num_orb = h1e.shape[0]
    _nelec = tools.fcidump.read(fcidump_filename)["NELEC"]
    num_elec_a = _nelec // 2
    num_elec_b = _nelec - num_elec_a
    print(f"Number of orbitals: {num_orb}")
    print(f"Number of electrons: {num_elec_a}α / {num_elec_b}β")

    ccsd = cc.CCSD(mf_as).run()
    ccsd_energy = ccsd.e_tot
    print(f"CCSD energy: {ccsd_energy:.10e}")
    
    if local:
        alpha_alpha_indices = [(p, p + 1) for p in range(num_orb - 1)]
        alpha_beta_indices  = [(p, p) for p in range(0, num_orb, 4) if p <= 16]
    else:
        alpha_alpha_indices = None
        alpha_beta_indices  = None

    print(f"\naa pairs: {alpha_alpha_indices}")
    print(f"ab pairs: {alpha_beta_indices}\n")

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=ccsd.t2, t1=ccsd.t1, n_reps=1,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
    )

    #ucj_op = ffsim.UCJOpSpinBalanced(
    #    diag_coulomb_mats=ucj_op_2layer.diag_coulomb_mats[:1],
    #    orbital_rotations=ucj_op_2layer.orbital_rotations[:1],
    #    final_orbital_rotation=ucj_op_2layer.orbital_rotations[1].T.conj(),
    #)

    nelec = (num_elec_a, num_elec_b)
    qubits = qiskit.QuantumRegister(2 * num_orb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    coupling_map = CouplingMap.from_full(num_qubits=circuit.num_qubits)
    backend = GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    compiled = qiskit.transpile(circuit, backend=backend, optimization_level=0)

    print(f"Number of qubits: {compiled.num_qubits}")
    print(f"Gate counts: {compiled.count_ops()}")
    
    return compiled
