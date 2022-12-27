from __future__ import annotations
import qutip
from qutip import operators, entropy_vn
from typing import List, Tuple, Optional, Type, Dict
import numpy as np


class AbstractOperator:
    def __init__(
        self,
        index: List[Tuple],
        direction: List[List],
        coupling: List,
        size: int,
        exc_numb: Optional[int] = None,
    ) -> None:

        self.index = index
        self.direction = direction
        self.coupling = coupling
        self.size = size
        self.op: dict = {}
        self.exc_numb = exc_numb

        # if indices> size the operator is ill defined
        assert max([max(idx) for idx in self.index]) <= (
            self.size - 1
        ), f"operator defined in a larger size system: idx > l={self.size}"
        self.__get_operator()
        self.__abstract2qutip()

    def __get_operator(self):
        for i, idx in enumerate(self.index):
            self.op[idx] = {
                "coupling": self.coupling[i],
                "direction": self.direction[i],
            }

    def append(self, ao: AbstractOperator) -> Optional[AbstractOperator]:
        assert self.size == ao.size, "size!=operator size"
        self.direction += ao.direction
        self.index += ao.index
        self.coupling += ao.coupling
        self.__get_operator()
        self.__abstract2qutip()

        return self

    def printout(self):
        print(self.op)

    def __abstract2qutip(self) -> Tuple[qutip.Qobj, List[qutip.Qobj]]:

        # operation that convert the abstract string to the qutip.Qobj
        # pauli dictionary
        local_obs_dict = {
            "id": qutip.identity(2),
            "x": qutip.sigmax(),
            "y": qutip.sigmay(),
            "z": qutip.sigmaz(),
            "+": qutip.sigmap(),
            "-": qutip.sigmam(),
        }

        # operation that convert the abstract string to the qutip.Qobj
        # Fock dictionary

        if self.exc_numb != None:
            local_obs_dict.update(
                {
                    "id_fock": qutip.identity(self.exc_numb),
                    "a_dag": qutip.create(self.exc_numb),
                    "a": qutip.sigmay(self.exc_numb),
                }
            )

        qutip_op: qutip.Qobj = 0
        qutip_op_density: Dict[qutip.Qobj] = {}
        # create the op representation
        for index in self.op.keys():
            # starting point -> identity operator
            idx_mb: List[str] = ["id" for i in range(self.size)]
            for j, idx in enumerate(index):
                idx_mb[idx] = self.op[index]["direction"][j]
            # convert into qutip.Qobj
            for i in range(self.size):
                if i == 0:
                    op = local_obs_dict[idx_mb[i]]
                else:
                    op = qutip.tensor(op, local_obs_dict[idx_mb[i]])
            # sum each direction
            qutip_op = qutip_op + op * self.op[index]["coupling"]
            qutip_op_density[index] = op * self.op[index]["coupling"]

        # initialize the attributes in the class
        # once for all
        self.qutip_op = qutip_op
        self.qutip_op_density = qutip_op_density
        return qutip_op, qutip_op_density

    def expect_value(self, psi: qutip.Qobj) -> float:
        return qutip.expect(self.qutip_op, psi)

    def expect_value_density(self, psi: qutip.Qobj) -> Dict[qutip.QObj]:
        values: dict = {}
        for index in self.qutip_op_density.keys():
            values[index] = qutip.expect(self.qutip_op_density[index], psi)
        return values


# we still can implement new attributes
# such as eigsh and gs_state
class IsingHamiltonian(AbstractOperator):
    def __init__(
        self,
        direction_coupling: Tuple[str],
        field_direction: str,
        pbc: Optional[bool] = False,
        size: Optional[int] = None,
        j: Optional[float] = None,
        h: Optional[float] = None,
        j_coupling: Optional[Dict] = None,
        ext_field: Optional[Dict] = None,
    ) -> None:

        # Fast Clean Transverse Ising Chain with nearest neighbourhoods
        if j is not (None):
            j_coupling = {}
            for i in range(size):
                if pbc:
                    j_coupling[(i, (i + 1) % size)] = j
                else:
                    if i + 1 < size:
                        j_coupling[(i, (i + 1))] = j

        if h is not (None):
            ext_field = {}
            for i in range(size):
                ext_field[(i,)] = h

        self.len_couplings = len(list(j_coupling.keys()))
        sum_coupling = j_coupling | ext_field
        index = list(sum_coupling.keys())
        directions = [
            [direction_coupling[0], direction_coupling[1]] for k in j_coupling.keys()
        ] + [[field_direction] for k in ext_field.keys()]
        size = len(ext_field)
        interaction_values = list(sum_coupling.values())

        super().__init__(index, directions, interaction_values, len(ext_field))

    def printout(self):

        print("Coupling Term: \n")
        print(list(self.op.items())[: self.len_couplings], "\n")
        print("External field: \n")
        print(list(self.op.items())[self.len_couplings :], "\n")


class SteadyStateSolver:
    def __init__(
        self, hamiltonian: AbstractOperator, dissipative_ops: List[AbstractOperator]
    ) -> None:

        # parameters
        self.hamiltonian = hamiltonian
        self.dissipative_ops = dissipative_ops
        self.size = hamiltonian.size

        # attributes
        self.steady_state: qutip.Qobj = None
        self.limbladian: qutip.Qobj = None

    def __get_the_limbladian(self) -> None:
        # define the hamiltonian

        hamiltonian_qutip = self.hamiltonian.qutip_op
        dissipative_qutip = [d.qutip_op for d in self.dissipative_ops]
        self.limbladian = qutip.liouvillian(
            H=hamiltonian_qutip, c_ops=dissipative_qutip
        )

    def get_steady_state(self) -> qutip.Qobj:
        self.__get_the_limbladian()
        self.steady_state = qutip.steadystate(qutip.to_super(self.limbladian))

    def print_liouvillian(self) -> None:
        print("Unitary part=\n")
        self.hamiltonian.printout()
        print("\n")
        print("Dissipative part=\n")
        for d in self.dissipative_ops:
            d.printout()
        print("\n")

    def steady_state_expect(self, op: AbstractOperator) -> float:
        # define the operator in
        return op.exp_value(self.steady_state)

    def steady_state_expect_density(self, op: AbstractOperator) -> float:
        return op.exp_value_density()

    def entanglement_entropy(self, size_a: int) -> float:

        rho_b = self.steady_state.copy()
        for i in range(size_a):
            rho_b = rho_b.ptrace(0)
        ent = entropy_vn(rho_b, base=2)

        return ent
