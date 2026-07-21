from typing import Optional

import sys
import torch
import json
import numpy as np
import ffsim
import pyscf
import qiskit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector

import quimb as qu
import quimb.tensor as qtn

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.mappers import JordanWignerMapper

import matplotlib.pyplot as plt; plt.rcParams.update({"font.family": "serif", "font.size": 12})

import warnings

warnings.filterwarnings("ignore")


def generate_linear_geometry(atom: str, natoms: int, atomic_distance: float = 1.0) -> str:
    return "; ".join([f"{atom} 0 0 {i * atomic_distance}" for i in range(natoms)])


def h_circuit(natoms: int = 16, nlayers: int = 1):
    mol = pyscf.gto.Mole()
    mol.build(
        atom=generate_linear_geometry('H', natoms),
        basis="sto-6g",
    )

    n_frozen = 0
    active_space = range(n_frozen, mol.nao_nr())

    scf = pyscf.scf.RHF(mol).run()

    norb = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)

    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    t1 = ccsd.t1
    t2 = ccsd.t2
    e_ccsd = ccsd.e_tot

    cas = pyscf.mcscf.CASCI(scf, norb, nelec)
    mo = cas.sort_mo(active_space, base=0)

    hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

    h2e_phys = np.einsum("prqs->pqrs", eri)  # chemist -> physicist notation
    elec_ints = ElectronicIntegrals.from_raw_integrals(hcore, h2e_phys)
    elec_hamiltonian = ElectronicEnergy(elec_ints)
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(elec_hamiltonian.second_q_op())
    hamiltonian = (hamiltonian + SparsePauliOp("I" * (2 * norb), coeffs=[nuclear_repulsion_energy])).simplify()

    sorted_indices = np.argsort(-np.abs(hamiltonian.coeffs))
    hamiltonian = hamiltonian[sorted_indices]

    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(0, norb, 1)]

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2, t1=t1, n_reps=nlayers,
        interaction_pairs=(pairs_aa, pairs_ab),
    )

    qubits = qiskit.QuantumRegister(2 * norb, name="q")
    circuit = qiskit.QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    coupling_map = CouplingMap.from_full(num_qubits=circuit.num_qubits)
    backend = GenericBackendV2(
        coupling_map.size(),
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    compiled = qiskit.transpile(circuit, backend=backend, optimization_level=0)
    compiled.count_ops()

    return compiled
