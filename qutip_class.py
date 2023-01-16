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
        direction_couplings: List[Tuple[str]],
        field_directions: List[str],
        pbc: Optional[bool] = False,
        size: Optional[int] = None,
        js: Optional[List[float]] = None,
        hs: Optional[List[float]] = None,
        j_couplings: Optional[List[Dict]] = None,
        ext_fields: Optional[List[Dict]] = None,
    ) -> None:

        # size attribute
        self.size = size
        # Fast Clean Transverse Ising Chain with nearest neighbourhoods
        self.h_ao = []
        if hs is not (None):
            for m, h in enumerate(hs):
                index = [(i,) for i in range(self.size)]
                coupling = [h for i in range(self.size)]
                dir = [field_directions[m] for i in range(self.size)]
                self.h_ao.append(
                    AbstractOperator(
                        index=index, direction=dir, coupling=coupling, size=self.size
                    )
                )
        else:
            for m, h in enumerate(ext_fields):
                coupling = list(h.values().item())
                index = list(h.keys().item())
                dir = [field_directions[m] for i in range(self.size)]
                self.h_ao.append(
                    AbstractOperator(
                        index=index, direction=dir, coupling=coupling, size=self.size
                    )
                )

        # if js is a list of coupling constants
        # initialize the coupling hamiltonian
        self.j_ao = []
        if js is not (None):
            # initialize the coupling
            # dictionary for the abstract
            # operator

            # a loop over the different
            # couplings (e.g.: j_1xx +j_2yy  )
            if pbc:
                index = [(i, (i + 1) % size) for i in range(self.size)]
            else:
                index = [(i, (i + 1)) for i in range(self.size - 1)]
            for m, j in enumerate(js):
                dir = [
                    [direction_couplings[m][0], direction_couplings[m][1]]
                    for s in index
                ]
                coupling = [j for s in index]
                self.j_ao.append(
                    AbstractOperator(
                        index=index, direction=dir, coupling=coupling, size=self.size
                    )
                )
        else:
            for m, j in enumerate(j_couplings):
                dir = [
                    [direction_couplings[m][0], direction_couplings[m][1]]
                    for i in j.keys()
                ]
                coupling = list(j.values().item())
                self.j_ao.append(
                    AbstractOperator(
                        index=list(j.keys().item()),
                        direction=dir,
                        coupling=coupling,
                        size=self.size,
                    )
                )

        self.qutip_op = 0
        self.qutip_op_density = {}
        for m, ham_j in enumerate(self.j_ao):
            self.qutip_op = self.qutip_op + ham_j.qutip_op
            self.qutip_op_density[direction_couplings[m]] = ham_j.qutip_op_density
        for m, ham_h in enumerate(self.h_ao):
            self.qutip_op = self.qutip_op + ham_h.qutip_op
            self.qutip_op_density[field_directions[m]] = ham_h.qutip_op_density

    def printout(self):
        """Printout of the total Hamiltonian, with the description of the Coupling term and the External field"""

        print("Coupling Term: \n")
        for ham_j in self.j_ao:
            ham_j.printout()
        print("External field: \n")
        for ham_h in self.h_ao:
            ham_h.printout()

    def expect_value_density(
        self, psi: qutip.Qobj, key: Tuple[str]
    ) -> Dict[qutip.QObj]:
        values: dict = {}
        for index in self.qutip_op_density.keys():
            values[index] = qutip.expect(self.qutip_op_density[key][index], psi)
        return values


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

    def negativity(self, indices: List[int]):
        """Compute the negativity of a set of sites in the steady state. Code by Simon Kothe.

        Args:
            indices (List[int]): list of spins in which the partial transposition acts.
        """

        # make sure that the partial
        # transpose does not affect
        # the steadystate outcome
        x = self.steady_state.copy()

        # define the mask
        mask = np.zeros(self.size)
        for idx in indices:
            mask[idx] = 1
        x = qutip.partial_transpose(x, mask=mask)
        return (x.norm() - 1)/2
