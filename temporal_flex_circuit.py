# temporal_flex_circuit.py
# Requires: qiskit
# A small utility to build "temporal flexible" quantum circuits with
# relative corrective blocks (classical-conditional corrections
# addressed relative to a measured qubit index).

from qiskit import QuantumCircuit, ClassicalRegister, Aer, execute
from qiskit.circuit.library import XGate, ZGate, HGate, SGate, TGate

class TemporalFlexCircuit:
    # temporal_flex_circuit.py
    # Requires: qiskit
    # A small utility to build "temporal flexible" quantum circuits with
    # relative corrective blocks (classical-conditional corrections
    # addressed relative to a measured qubit index).

    from typing import List, Tuple, Dict, Any

    from qiskit import QuantumCircuit, ClassicalRegister, execute
    from qiskit.providers.aer import AerSimulator
    from qiskit.circuit.library import XGate, ZGate, HGate, SGate, TGate


    class TemporalFlexCircuit:
        """Helper to build circuits where corrective (classical-conditional)
        gates are applied relative to a previously measured qubit index.

        The class keeps a list of single-bit classical registers created when
        measuring qubits so each measurement result is stored in its own
        ClassicalRegister(1). This makes it easy to set c_if conditions on
        later gates.
        """

        def __init__(self, n_qubits: int):
            self.n: int = int(n_qubits)
            self.qc: QuantumCircuit = QuantumCircuit(n_qubits)
            self.cregs: List[ClassicalRegister] = []  # list of ClassicalRegister(1)
            self.measure_map: Dict[int, int] = {}     # measured_qubit -> creg index

        def add_layer(self, gates: List[Tuple]) -> None:
            """Append a small layer described by tuples.

            Supported formats:
            - ('h', q)
            - ('x', q)
            - ('z', q)
            - ('s', q)
            - ('t', q)
            - ('cx', control, target)
            - ('swap', q1, q2)
            """
            for g in gates:
                if not isinstance(g, (list, tuple)) or len(g) < 2:
                    raise ValueError(f"Unsupported gate format: {g}")
                name = str(g[0]).lower()
                # single-qubit gates
                if name in ("h", "x", "z", "s", "t") and len(g) == 2:
                    target = int(g[1])
                    self._validate_qubit_index(target)
                    getattr(self.qc, name)(target)
                elif name == 'cx' and len(g) == 3:
                    c = int(g[1]); t = int(g[2])
                    self._validate_qubit_index(c); self._validate_qubit_index(t)
                    self.qc.cx(c, t)
                elif name == 'swap' and len(g) == 3:
                    q1 = int(g[1]); q2 = int(g[2])
                    self._validate_qubit_index(q1); self._validate_qubit_index(q2)
                    self.qc.swap(q1, q2)
                else:
                    raise ValueError(f"Unsupported gate format: {g}")

        def _validate_qubit_index(self, idx: int) -> None:
            if not (0 <= idx < self.n):
                raise IndexError(f"Qubit index {idx} out of range (0..{self.n-1})")

        def measure(self, q_index: int) -> int:
            """Measure the given qubit into a fresh 1-bit ClassicalRegister.

            Returns the index of the new classical register in self.cregs.
            """
            self._validate_qubit_index(q_index)
            creg = ClassicalRegister(1, f'c{len(self.cregs)}')
            self.qc.add_register(creg)
            self.qc.measure(q_index, creg[0])
            self.cregs.append(creg)
            creg_idx = len(self.cregs) - 1
            self.measure_map[q_index] = creg_idx
            return creg_idx

        def _apply_conditional_single(self, gate_name: str, target: int, creg: ClassicalRegister, value: int) -> None:
            gate_map = {'x': XGate, 'z': ZGate, 'h': HGate, 's': SGate, 't': TGate}
            gate_name = gate_name.lower()
            if gate_name not in gate_map:
                raise ValueError("Unsupported corrective gate: " + gate_name)
            self._validate_qubit_index(target)
            instr = gate_map[gate_name]()    # create instruction
            # set classical condition on the single-bit register
            instr.c_if(creg, int(value))
            # append the instruction for the single target qubit
            self.qc.append(instr, [self.qc.qubits[target]], [])

        def relative_corrective_block(self, measured_qubit: int, correction_map: Dict[Any, List[Tuple[str, int]]]) -> None:
            """
            Apply corrective blocks relative to a measured qubit.
            - measured_qubit: index of qubit that was measured (must have been measured with measure()).
            - correction_map: dict mapping classical outcome (int) -> list of (gate_name, target_offset)
              where target = (measured_qubit + target_offset) % n
            Example:
              # if measured qubit m gave 1, apply X to (m+1) and Z to (m+2)
              {1: [('x', 1), ('z', 2)]}
            """
            if measured_qubit not in self.measure_map:
                raise ValueError(f"Qubit {measured_qubit} hasn't been measured (call measure() first)")
            creg = self.cregs[self.measure_map[measured_qubit]]
            for outcome, ops in correction_map.items():
                for gate_name, offset in ops:
                    target = (measured_qubit + int(offset)) % self.n
                    # apply the conditional corrective gate
                    self._apply_conditional_single(str(gate_name), target, creg, int(outcome))

        def run_qasm(self, shots: int = 1024) -> Dict[str, int]:
            """
            Execute the built circuit on the Aer qasm simulator and return counts.
            Use qasm (counts) because statevector after mid-circuit measurement + classical
            conditional gates is not meaningful.
            """
            # Use AerSimulator for modern Qiskit Aer provider
            backend = AerSimulator()
            job = execute(self.qc, backend, shots=shots)
            result = job.result()
            # return counts dict for convenience
            return result.get_counts(self.qc)


    def _teleportation_example() -> Dict[str, int]:
        """Build and run a deterministic teleportation test (|1> teleportation).

        Returns the counts observed on the qasm simulator as a dictionary.
        """
        tfc = TemporalFlexCircuit(3)
        # Prepare |1> on q0
        tfc.add_layer([('x', 0)])
        # Create Bell pair between q1 and q2
        tfc.add_layer([('h', 1), ('cx', 1, 2)])
        # Bell measurement of q0 & q1
        tfc.add_layer([('cx', 0, 1), ('h', 0)])
        tfc.measure(0)   # c0
        tfc.measure(1)   # c1
        # Relative corrective blocks: conditional on those measurements apply
        tfc.relative_corrective_block(1, {1: [('x', 1)]})
        tfc.relative_corrective_block(0, {1: [('z', 2)]})
        # Final measurement of q2 to verify teleportation
        tfc.measure(2)
        counts = tfc.run_qasm(shots=1024)
        return counts


    if __name__ == "__main__":
        counts = _teleportation_example()
        print("Teleportation test counts:")
        print(counts)
