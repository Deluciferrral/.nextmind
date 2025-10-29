# temporal_flex_circuit.py
# Requires: qiskit
# A small utility to build "temporal flexible" quantum circuits with
# relative corrective blocks (classical-conditional corrections
# addressed relative to a measured qubit index).

from qiskit import QuantumCircuit, ClassicalRegister, Aer, execute
from qiskit.circuit.library import XGate, ZGate, HGate, SGate, TGate

class TemporalFlexCircuit:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.qc = QuantumCircuit(n_qubits)
        self.cregs = []            # list of ClassicalRegister(1) objects
        self.measure_map = {}      # measured_qubit -> creg index

    def add_layer(self, gates):
        # gates: list of tuples like ('h', q), ('cx', c, t), ('x', q)
        for g in gates:
            name = g[0].lower()
            if name == 'h' and len(g) == 2:
                self.qc.h(g[1])
            elif name == 'x' and len(g) == 2:
                self.qc.x(g[1])
            elif name == 'z' and len(g) == 2:
                self.qc.z(g[1])
            elif name == 's' and len(g) == 2:
                self.qc.s(g[1])
            elif name == 't' and len(g) == 2:
                self.qc.t(g[1])
            elif name == 'cx' and len(g) == 3:
                self.qc.cx(g[1], g[2])
            elif name == 'swap' and len(g) == 3:
                self.qc.swap(g[1], g[2])
            else:
                raise ValueError("Unsupported gate format: {}".format(g))

    def measure(self, q_index):
        # creates 1-bit classical register and measures q_index into it
        creg = ClassicalRegister(1, f'c{len(self.cregs)}')
        self.qc.add_register(creg)
        # measure the qubit into the newly created classical bit
        self.qc.measure(q_index, creg[0])
        self.cregs.append(creg)
        self.measure_map[q_index] = len(self.cregs) - 1
        return len(self.cregs) - 1

    def _apply_conditional_single(self, gate_name, target, creg, value):
        gate_map = {'x': XGate, 'z': ZGate, 'h': HGate, 's': SGate, 't': TGate}
        if gate_name not in gate_map:
            raise ValueError("Unsupported corrective gate: " + gate_name)
        instr = gate_map[gate_name]()    # create instruction
        # set classical condition on the single-bit register
        instr.c_if(creg, value)
        # append the instruction for the single target qubit
        self.qc.append(instr, [self.qc.qubits[target]], [])

    def relative_corrective_block(self, measured_qubit, correction_map):
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
            raise ValueError("Qubit {} hasn't been measured (call measure() first)".format(measured_qubit))
        creg = self.cregs[self.measure_map[measured_qubit]]
        for outcome, ops in correction_map.items():
            for gate_name, offset in ops:
                target = (measured_qubit + offset) % self.n
                # ensure gate_name lowercase for mapping
                self._apply_conditional_single(gate_name.lower(), target, creg, int(outcome))

    def run_qasm(self, shots=1024):
        """
        Execute the built circuit on the Aer qasm simulator and return result.
        Use qasm (counts) because statevector after mid-circuit measurement + classical
        conditional gates is not meaningful.
        """
        backend = Aer.get_backend('aer_simulator')
        job = execute(self.qc, backend, shots=shots)
        return job.result()


if __name__ == "__main__":
    # Deterministic teleportation test:
    # Prepare |1> on q0, teleport to q2, verify final measurement of q2 is 1.
    tfc = TemporalFlexCircuit(3)

    # Prepare |1> on q0 (so we can deterministically check teleportation)
    tfc.add_layer([('x', 0)])

    # Create Bell pair between q1 and q2
    tfc.add_layer([('h', 1), ('cx', 1, 2)])

    # Bell measurement of q0 & q1 (teleportation)
    tfc.add_layer([('cx', 0, 1), ('h', 0)])
    tfc.measure(0)   # c0
    tfc.measure(1)   # c1

    # Relative corrective blocks:
    # if measurement of q1 (m=1) == 1 -> apply X to target (m+1) -> q2
    tfc.relative_corrective_block(1, {1: [('x', 1)]})
    # if measurement of q0 (m=0) == 1 -> apply Z to target (m+2) -> q2
    tfc.relative_corrective_block(0, {1: [('z', 2)]})

    # Final measurement of q2 to verify teleportation (measure into new classical bit)
    tfc.measure(2)  # this will be c2

    # Run on qasm simulator to verify teleportation deterministically
    result = tfc.run_qasm(shots=1024)
    counts = result.get_counts(tfc.qc)

    print("Circuit:\n", tfc.qc)
    print("\nCounts (format: classical registers string):")
    print(counts)
