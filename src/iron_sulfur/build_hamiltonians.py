import numpy as np
import ffsim
from pyscf import tools, cc, ao2mo
import qiskit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import SparsePauliOp
from qiskit import qpy
from qiskit import qasm2

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.mappers import JordanWignerMapper

import cirq


ALL_CONNECTIVITIES = ["square", "heavy-hex", "all"]
ALL_LOCALITIES = [True, False]
fcidump_filename = "fcidump_Fe4S4_MO.txt"

def make_compiled_ham(local):
    locality = "LUCJ" if local else "UCJ"
    
    mf_as = tools.fcidump.to_scf(fcidump_filename)
    mf_as.kernel()
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
        alpha_beta_indices  = [(p, p) for p in range(0, num_orb)]
    else:
        alpha_alpha_indices = None
        alpha_beta_indices  = None

    ucj_op_2layer = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=ccsd.t2, t1=ccsd.t1, n_reps=2,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
    )

    ucj_op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=ucj_op_2layer.diag_coulomb_mats[:1],
        orbital_rotations=ucj_op_2layer.orbital_rotations[:1],
        final_orbital_rotation=ucj_op_2layer.orbital_rotations[1].T.conj(),
    )

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

    circuit_path = f"hamiltonians/{locality}_circuit.qpy"
    with open(circuit_path, "wb") as f:
        qpy.dump(compiled, f)
    print(f"Saved circuit: {circuit_path}")

    cache = np.load("hamiltonians/hamiltonian_cache.npz", allow_pickle=False)
    hamiltonian = SparsePauliOp.from_list(
        list(zip(cache["paulis"].astype(str), cache["coeffs"]))
    )

    hamiltonian_physical = hamiltonian.apply_layout(compiled.layout)

    np.savez(
        f"hamiltonians/{locality}_hamiltonian.npz",
        paulis=np.array(hamiltonian_physical.paulis.to_labels()),
        coeffs=np.array(hamiltonian_physical.coeffs),
        ccsd_energy=np.float64(ccsd_energy),
        n_qubits=np.int64(compiled.num_qubits),
    )
    print(f"{locality}_compiled_hamiltonian.npz")


remake_cache = False
if remake_cache:
    hamiltonian_cache = "hamiltonians/hamiltonian_cache.npz"

    mf_as = tools.fcidump.to_scf(fcidump_filename)
    h1e = mf_as.get_hcore()

    num_orb = h1e.shape[0]
    n_qubits = 2 * num_orb
    print(f"Number of spatial orbitals: {num_orb}, Number of qubits: {n_qubits}")

    h2e = ao2mo.restore(1, mf_as._eri, num_orb)
    h2e_phys = np.einsum("prqs->pqrs", h2e)

    constant = tools.fcidump.read(fcidump_filename).get("ECORE", 0.0)
    print("Constant term (ECORE):", constant)

    elec_ints = ElectronicIntegrals.from_raw_integrals(h1e, h2e_phys)
    elec_hamiltonian = ElectronicEnergy(elec_ints)

    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(elec_hamiltonian.second_q_op())
    hamiltonian = (hamiltonian + SparsePauliOp("I" * n_qubits, coeffs=[constant])).simplify()
    print(f"Hamiltonian has {len(hamiltonian)} Pauli terms before cutoff.")

    hamiltonian = hamiltonian.simplify()
    #hamiltonian = hamiltonian.chop(1e-10)
    sorted_indices = np.argsort(-np.abs(hamiltonian.coeffs))
    hamiltonian = hamiltonian[sorted_indices]

    np.savez(hamiltonian_cache, paulis=hamiltonian.paulis.to_labels(), coeffs=hamiltonian.coeffs)
    print("Number of terms after cutoff:", len(hamiltonian.coeffs))
    print(f"Hamiltonian cached to {hamiltonian_cache}")

for locality in ALL_LOCALITIES:
    make_compiled_ham(locality)