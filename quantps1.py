import math, random

def matrix_vector_multiply(A, v):
    """Matrix * vector multiplication."""
    result = [0j for _ in range(len(A))]
    for i in range(len(A)):
        result[i] = sum(A[i][k] * v[k] for k in range(len(v)))
    return result

def kron(A, B):
    """Kronecker product of two matrices."""
    res = []
    for rowA in A:
        for rowB in B:
            res.append([a * b for a in rowA for b in rowB])
    return res

# Standard single-qubit gates
I = [
    [1,0],
    [0,1]
]

X = [
    [0,1],
    [1,0]
]
Y = [
    [0,-1j],
    [1j,0]
]
S = [
    [1,0],
    [0,1j]
]
H = [[1/math.sqrt(2), 1/math.sqrt(2)],
     [1/math.sqrt(2), -1/math.sqrt(2)]]
P0 = [
    [1,0],
    [0,0]
]
P1 = [
    [0,0],
    [0,1]
]

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        # statevector starts as |00..0>
        self.state = [0j]*(2**num_qubits)
        self.state[0] = 1.0+0j
    
    def operator_for_single_gate(self, gate, qubit):
        oper = None
        for i in range(self.num_qubits):
            factor = gate if i == qubit else I
            oper = factor if oper is None else kron(oper, factor)
        return oper
       

    def single_gate(self, gate, qubit):
        op = self.operator_for_single_gate(gate, qubit)
        self.state = matrix_vector_multiply(op,self.state)

    def operator_for_controlled(self, control, target, gate):
        op0, op1 = None, None
        for i in range(self.num_qubits):
            if i == control:
                f0, f1 = P0, P1
            elif i == target:
                f0, f1 = I, gate
            else:
                f0 = f1 = I
            op0 = f0 if op0 is None else kron(op0, f0)
            op1 = f1 if op1 is None else kron(op1, f1)
        return [[op0[i][j] + op1[i][j] for j in range(len(op0))] for i in range(len(op0))]


    def controlled_gate(self, control, target, gate):
        if control == target:
            raise ValueError("Control and target must be different")
        op = self.operator_for_controlled(control, target, gate)
        self.state =matrix_vector_multiply(op, self.state)

    # Shorthand methods
    def x(self, qubit): self.single_gate(X, qubit)
    def y(self, qubit): self.single_gate(Y, qubit)
    def h(self, qubit): self.single_gate(H, qubit)
    def i(self, qubit): self.single_gate(I, qubit)
    def s(self, qubit): self.single_gate(S, qubit)
    def cx(self, control, target): self.controlled_gate(control, target, X)
    def cy(self, control, target): self.controlled_gate(control, target, Y)
    def ch(self, control, target): self.controlled_gate(control, target, H)

    def statevector(self):
        return self.state

    def measure(self):
        probs = [abs(a)**2 for a in self.state]
        # normalize (fix floating errors)
        total = sum(probs)
        probs = [p/total for p in probs]
        # weighted random pick
        r = random.random()
        cum = 0
        outcome = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                outcome = i
                break
        # collapse state
        self.state = [0j]*len(self.state)
        self.state[outcome] = 1.0+0j
        return format(outcome, f'0{self.num_qubits}b')

qc = QuantumCircuit(2)
qc.h(0)       # put qubit 0 in superposition
qc.cx(0,1)    # entangle with qubit 1 (Bell state)

print("Statevector:", qc.statevector())
print("Measurement:", qc.measure())
