from __future__ import annotations
import qutip
from qutip import operators, entropy_vn
from typing import List, Tuple, Optional, Type, Dict
import numpy as np

# stackoverflow https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


class ManyBodyQutipOperator:
    def __init__(
        self,
        local_op: Optional[List[qutip.Qobj]] = None,
        description: Optional[str] = None,
    ) -> None:

        self.size = None
        self.qutip_op = None
        if local_op is not (None):
            self.__get_qutip_op(local_op)
            self.size = len(local_op)
        self.description = description

    def __get_qutip_op(self, local_op: List[qutip.Qobj]):

        for i, op in enumerate(local_op):
            if i == 0:
                self.qutip_op = op
            else:
                self.qutip_op = qutip.tensor(self.qutip_op, op)

    def expect_value(self, psi: qutip.Qobj) -> float:
        return qutip.expect(self.qutip_op, psi)

    def printout(self):
        print(self.description, "\n")
        print(self.qutip_op, "\n")


class SpinOperator(ManyBodyQutipOperator):
    def __init__(
        self,
        index: List[Tuple],
        coupling: List,
        size: int,
        exc_numb: Optional[int] = None,
    ) -> None:

        super().__init__()

        self.size = size
        self.index = index
        self.coupling = coupling
        self.exc_numb = exc_numb

        # if indices> size the operator is ill defined
        assert max([max(idx) for idx in self.index]) <= (
            self.size - 1
        ), f"operator defined in a larger size system: idx > l={self.size}"
        self.__operator_description()
        self.__abstract2qutip()

    def __operator_description(self):
        op: Dict = {}
        for i, idx in enumerate(self.index):
            op[idx] = {
                "coupling": self.coupling[i],
                "operator": idx,
            }
        self.description = op

    def __abstract2qutip(self) -> Tuple[qutip.Qobj, List[qutip.Qobj]]:

        # operation that convert the abstract string to the qutip.Qobj
        # pauli dictionary
        local_obs_dict = {
            "x": qutip.sigmax(),
            "y": qutip.sigmay(),
            "z": qutip.sigmaz(),
            "+": qutip.sigmap(),
            "-": qutip.sigmam(),
        }

        self.qutip_op_density: Dict[qutip.Qobj] = {}
        # create the op representation
        # considering each Tuple of
        # indices and directions
        for k, tuple_indices in enumerate(self.index):

            coupling = self.coupling[k]
            # initialize the SpinOperator
            op_list: List[qutip.Qobj] = [qutip.identity(2) for r in range(self.size)]
            for direction, idx in pairwise(tuple_indices):
                # define the given
                # local label operator
                op_list[idx] = local_obs_dict[direction]
            # starting point -> identity operator
            op = ManyBodyQutipOperator(size=self.size, local_op=op_list)

            # sum each direction
            if k == 0:
                self.qutip_op = op.qutip_op * coupling
            else:
                self.qutip_op = self.qutip_op + op.qutip_op * coupling

            self.qutip_op_density[tuple_indices] = op.qutip_op * coupling

        return self.qutip_op.copy(), self.qutip_op_density.copy()

    def expect_value_density(self, psi: qutip.Qobj) -> Dict[qutip.Qobj]:
        values: dict = {}
        for index in self.qutip_op_density.keys():
            values[index] = qutip.expect(self.qutip_op_density[index], psi)
        return values


class FockOperator(ManyBodyQutipOperator):
    def __init__(
        self,
        index: List[Tuple],
        coupling: List,
        size: int,
        exc_numb: Optional[int] = None,
    ) -> None:

        super().__init__()

        self.size = size
        self.index = index
        self.coupling = coupling
        self.op: dict = {}
        self.exc_numb = exc_numb

        # if indices> size the operator is ill defined
        assert max([max(idx) for idx in self.index]) <= (
            self.size - 1
        ), f"operator defined in a larger size system: idx > l={self.size}"
        self.__operator_description()
        self.__abstract2qutip()

    def __operator_description(self):
        op: Dict = {}
        for i, idx in enumerate(self.index):
            op[idx] = {
                "coupling": self.coupling[i],
                "direction": self.direction[i],
            }
        self.description = op

    def __abstract2qutip(self) -> Tuple[qutip.Qobj, List[qutip.Qobj]]:

        # operation that convert the abstract string to the qutip.Qobj
        # Fock dictionary
        local_obs_dict = {
            "id_fock": qutip.identity(self.exc_numb),
            "a_dag": qutip.create(self.exc_numb),
            "a": qutip.sigmay(self.exc_numb),
        }

        self.qutip_op_density: Dict[qutip.Qobj] = {}
        # create the op representation
        # considering each Tuple of
        # indices and directions
        for k, tuple_indices in enumerate(self.index):

            coupling = self.coupling[k]
            # initialize the FockOperator
            op_list: List[qutip.Qobj] = [
                qutip.identity(self.exc_numb) for r in range(self.size)
            ]
            for direction, idx in pairwise(tuple_indices):
                # define the given
                # local label operator
                op_list[idx] = local_obs_dict[direction]
            # starting point -> identity operator
            op = ManyBodyQutipOperator(size=self.size, local_op=op_list)

            # sum each direction
            if k == 0:
                self.qutip_op = op.qutip_op * coupling
            else:
                self.qutip_op = self.qutip_op + op.qutip_op * coupling

            self.qutip_op_density[tuple_indices] = op.qutip_op * coupling

        return self.qutip_op.copy(), self.qutip_op_density.copy()

    def expect_value_density(self, psi: qutip.Qobj) -> Dict[qutip.Qobj]:
        values: dict = {}
        for index in self.qutip_op_density.keys():
            values[index] = qutip.expect(self.qutip_op_density[index], psi)
        return values


class Hamiltonian(ManyBodyQutipOperator):
    def __init__(
        self,
        size: int,
        couplings: Optional[List[ManyBodyQutipOperator]] = None,
        ext_fields: Optional[List[ManyBodyQutipOperator]] = None,
        extra_terms: Optional[List[ManyBodyQutipOperator]] = None,
    ) -> None:

        super().__init__()
        # size attribute
        self.size = size
        # Fast Clean Transverse Ising Chain with nearest neighbourhoods
        self.h_ao: List[ManyBodyQutipOperator] = []
        for m, h in enumerate(ext_fields):
            self.h_ao.append(h)

        # if js is a list of coupling constants
        # initialize the coupling hamiltonian
        self.j_ao: List[ManyBodyQutipOperator] = []
        for m, j in enumerate(couplings):
            self.j_ao.append(j)

        self.others_ao: List[ManyBodyQutipOperator] = []
        for m, j in enumerate(extra_terms):
            self.others_ao.append(j)

        self.qutip_op = 0
        self.qutip_op_density = {}
        for ham_j in self.j_ao:
            self.qutip_op = self.qutip_op + ham_j.qutip_op
            if ham_j.description is not (None):
                self.qutip_op_density[ham_j.description] = ham_j.qutip_op_density
        for ham_h in self.h_ao:
            self.qutip_op = self.qutip_op + ham_h.qutip_op
            if ham_h.description is not (None):
                self.qutip_op_density[ham_h.description] = ham_h.qutip_op_density
        for ham_o in self.others_ao:
            self.qutip_op = self.qutip_op + ham_h.qutip_op
            if ham_o.description is not (None):
                self.qutip_op_density[ham_o.description] = ham_o.qutip_op_density

    def printout(self):
        """Printout of the total Hamiltonian, with the description of the Coupling term and the External field"""

        print("Coupling Term: \n")
        for ham_j in self.j_ao:
            ham_j.printout()
        print("External field: \n")
        for ham_h in self.h_ao:
            ham_h.printout()
        print("External field: \n")
        for ham_o in self.others_ao:
            ham_o.printout()

    def expect_value_density(
        self, psi: qutip.Qobj, key: Tuple[str]
    ) -> Dict[qutip.Qobj]:
        values: dict = {}
        for index in self.qutip_op_density.keys():
            values[index] = qutip.expect(self.qutip_op_density[key][index], psi)
        return values


# we still can implement new attributes
# such as eigsh and gs_state
class IsingHamiltonian(Hamiltonian):
    def __init__(
        self,
        direction_couplings: List[Tuple[str]],
        field_directions: List[str],
        pbc: Optional[bool] = False,
        size: Optional[int] = None,
        js: Optional[List[float]] = None,
        hs: Optional[List[float]] = None,
        j_couplings: Optional[List[ManyBodyQutipOperator]] = None,
        ext_fields: Optional[List[ManyBodyQutipOperator]] = None,
    ) -> None:

        super().__init__(size=size)

        # Fast Clean Transverse Ising Chain with nearest neighbourhoods
        self.h_ao: List[ManyBodyQutipOperator] = []
        if hs is not (None):
            for m, h in enumerate(hs):
                index = [(field_directions[m], i) for i in range(self.size)]
                coupling = [h for i in range(self.size)]
                self.h_ao.append(
                    SpinOperator(index=index, coupling=coupling, size=self.size)
                )
        else:
            for m, h in enumerate(ext_fields):
                self.h_ao.append(h)

        # if js is a list of coupling constants
        # initialize the coupling hamiltonian
        self.j_ao: List[ManyBodyQutipOperator] = []
        if js is not (None):
            # initialize the coupling
            # dictionary for the abstract
            # operator

            # a loop over the different
            # couplings (e.g.: j_1xx +j_2yy  )

            for m, j in enumerate(js):
                if pbc:
                    index = [
                        (
                            direction_couplings[m][0],
                            i,
                            direction_couplings[m][1],
                            (i + 1) % size,
                        )
                        for i in range(self.size)
                    ]
                else:
                    index = [
                        (
                            direction_couplings[m][0],
                            i,
                            direction_couplings[m][1],
                            (i + 1),
                        )
                        for i in range(self.size - 1)
                    ]

                coupling = [j for s in index]
                self.j_ao.append(
                    SpinOperator(index=index, coupling=coupling, size=self.size)
                )
        else:
            for m, j in enumerate(j_couplings):
                self.j_ao.append(j)

        self.qutip_op = 0
        self.qutip_op_density = {}
        for m, ham_j in enumerate(self.j_ao):
            self.qutip_op = self.qutip_op + ham_j.qutip_op
            self.qutip_op_density[index[m]] = ham_j.qutip_op_density
        for m, ham_h in enumerate(self.h_ao):
            self.qutip_op = self.qutip_op + ham_h.qutip_op
            self.qutip_op_density[index[m]] = ham_h.qutip_op_density

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
    ) -> Dict[qutip.Qobj]:
        values: dict = {}
        for index in self.qutip_op_density.keys():
            values[index] = qutip.expect(self.qutip_op_density[key][index], psi)
        return values


class SteadyStateSolver:
    def __init__(
        self,
        hamiltonian: ManyBodyQutipOperator,
        dissipative_ops: List[ManyBodyQutipOperator],
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

    def get_steady_state(self, method: str) -> qutip.Qobj:
        self.__get_the_limbladian()
        try:
            self.steady_state = qutip.steadystate(
                qutip.to_super(self.limbladian),
            )
        except Exception:
            print("Zero pivot, numerical factorization or iterative refinement problem")

        else:
            self.steady_state = qutip.steadystate(
                qutip.to_super(self.limbladian),
                method=method,
            )

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
        return (x.norm() - 1) / 2

    def purity(self):
        x = self.steady_state.copy()
        return 1 - (x * x).norm()
