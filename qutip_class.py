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


class ManyBodyQutipOperatorOld:
    def __init__(
        self,
        local_op: Optional[List[qutip.Qobj]] = None,
        description: Optional[str] = None,
    ) -> None:
        """_summary_
        Args:
            local_op (Optional[List[qutip.Qobj]], optional): _description_. Defaults to None.
            description (Optional[str], optional): _description_. Defaults to None.
        """

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


class SpinOperatorOld(ManyBodyQutipOperatorOld):
    def __init__(
        self,
        index: List[Tuple],
        coupling: List,
        size: int,
        density: Optional[bool] = None,
    ) -> None:

        super().__init__()

        self.size = size
        self.index = index
        self.coupling = coupling

        self.__operator_description()
        self.__abstract2qutip(density)

    def __operator_description(self):
        op: Dict = {}
        for i, idx in enumerate(self.index):
            op[idx] = {
                "coupling": self.coupling[i],
                "operator": idx,
            }
        self.description = op

    def __abstract2qutip(
        self, density: Optional[bool]
    ) -> Tuple[qutip.Qobj, List[qutip.Qobj]]:

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
            op = ManyBodyQutipOperatorOld(local_op=op_list)

            # sum each direction
            if k == 0:
                self.qutip_op = op.qutip_op * coupling
            else:
                self.qutip_op = self.qutip_op + op.qutip_op * coupling

            if density:
                self.qutip_op_density[tuple_indices] = op.qutip_op * coupling

        return self.qutip_op.copy(), self.qutip_op_density.copy()

    def expect_value_density(self, psi: qutip.Qobj) -> Dict[qutip.Qobj]:
        values: dict = {}
        for index in self.qutip_op_density.keys():
            values[index] = qutip.expect(self.qutip_op_density[index], psi)
        return values


class ManyBodyQutipOperator:
    def __init__(
        self,
    ) -> None:
        """_summary_

        Args:
            local_op (Optional[List[qutip.Qobj]], optional): _description_. Defaults to None.
            description (Optional[str], optional): _description_. Defaults to None.
        """

        self._qutip_op = None
        self._description = None

    @property
    def qutip_op(self):
        return self._qutip_op

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, comment: str):
        self._description = comment

    @qutip_op.setter
    def qutip_op(self, local_op: List[qutip.Qobj]):

        for i, op in enumerate(local_op):
            if type(op) != qutip.qobj.Qobj:
                raise TypeError(
                    f"Element {i} is not a Qutip Object Qobj ({type(op)} instead)"
                )
            if i == 0:
                self._qutip_op = op
            else:
                self._qutip_op = qutip.tensor(self.qutip_op, op)

    def expect_value(self, psi: qutip.Qobj) -> float:
        return qutip.expect(self.qutip_op, psi)

    def __str__(self) -> str:
        return f"{self.description} \n {self.qutip_op} \n"


class SpinOperator(ManyBodyQutipOperator):
    def __init__(
        self,
        index: List[Tuple],
        coupling: List,
        size: int,
    ) -> None:

        super().__init__()

        # dictionary for the conversion string to local operator
        self._local_obs_dict = {
            "x": qutip.sigmax(),
            "y": qutip.sigmay(),
            "z": qutip.sigmaz(),
            "+": qutip.sigmap(),
            "-": qutip.sigmam(),
            "id": qutip.identity(2),
        }

        # size of the system
        self.size = size
        # list of tuples (indices, local operator)
        self.index = index
        # coupling constants
        self.coupling = coupling

        # getter of the qutip op
        # in this subclass
        self.__get_qutip_op()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: List[Tuple]):
        for k, tuple_indices in enumerate(index):
            for direction, idx in pairwise(tuple_indices):
                # check the direction
                if not (direction in self._local_obs_dict.keys()):
                    raise f"local operator string not defined -> {direction} index -> {idx}"

                if idx > self.size - 1:
                    raise f"error, index larger than the number of sites"

        self._index = index

    @property
    def coupling(self):
        return self._coupling

    @coupling.setter
    def coupling(self, coupling: List):
        self._coupling = coupling

    @property
    def qutip_op(self):
        return super().qutip_op

    def __get_qutip_op(
        self,
    ):
        self._description: str = "(couplings, operator) -> \n"
        for k, tuple_indices in enumerate(self.index):

            # initialize the description
            self._description = (
                self._description + f" ( {self.coupling[k]} ,  {tuple_indices} ) \n"
            )

            coupling = self.coupling[k]
            # initialize the SpinOperator
            op_dict: Dict = {}
            indices: List = []
            for direction, idx in pairwise(tuple_indices):

                # define the given
                # local label operator

                if idx in op_dict.keys():
                    op_dict[idx] = op_dict[idx] * self._local_obs_dict[direction]
                else:
                    op_dict[idx] = self._local_obs_dict[direction]
                    indices.append(idx)

                # we fix the first part of the chain with
                # an identity operator
                if not (0 in op_dict.keys()):
                    indices.append(0)
                    op_dict[0] = self._local_obs_dict["id"]

                # and the last part with another identity
                # operator
                if not (self.size - 1 in op_dict.keys()):
                    indices.append(self.size - 1)
                    op_dict[self.size - 1] = self._local_obs_dict["id"]

            # order the indices
            indices.sort()
            # method for qutip.tensor optimization by Ewen Lawrence https://github.com/ewenlawrence
            # dumb index
            jdx: int = 0

            for i, idx in enumerate(indices):
                # if the sites are not nearest neighbours
                # create a identity with dim 2**(i-j)

                if idx - jdx > 1:
                    identity = qutip.identity(2 ** (idx - jdx - 1))
                    chain_oper = qutip.tensor(identity, op_dict[idx])
                # otherwise define the operator
                else:
                    chain_oper = op_dict[idx]

                if i == 0:
                    many_body_op = chain_oper
                else:
                    many_body_op = qutip.tensor(many_body_op, chain_oper)

                jdx = idx
                # if k == 1:
                #     print("partial many body op=", many_body_op)

            # if k == 1:
            #     print("mboperator=", many_body_op)

            # sum each direction
            if k == 0:
                self._qutip_op = many_body_op * coupling
            else:

                self._qutip_op = self._qutip_op + many_body_op * coupling


class FockOperatorOld(ManyBodyQutipOperator):
    def __init__(
        self,
        index: List[Tuple],
        coupling: List,
        size: int,
        exc_numb: Optional[int] = None,
        density: Optional[bool] = None,
    ) -> None:

        super().__init__()

        self.size = size
        self.index = index
        self.coupling = coupling
        self.op: dict = {}
        self.exc_numb = exc_numb

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


class FockOperator(ManyBodyQutipOperator):
    def __init__(
        self,
        index: List[Tuple],
        coupling: List,
        size: int,
        exc_numb: Optional[int] = None,
    ) -> None:

        super().__init__()

        # operation that convert the abstract string to the qutip.Qobj
        # Fock dictionary
        self.local_obs_dict = {
            "id_fock": qutip.identity(self.exc_numb),
            "a_dag": qutip.create(self.exc_numb),
            "a": qutip.destroy(self.exc_numb),
        }

        self.size = size
        self.index = index
        self.coupling = coupling
        self.exc_numb = exc_numb

        self.__get_qutip_op()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: List[Tuple]):
        for k, tuple_indices in enumerate(index):
            for direction, idx in pairwise(tuple_indices):
                # check the direction
                if not (direction in self._local_obs_dict.keys()):
                    raise f"local operator string not defined -> {direction} index -> {idx}"

                if idx > self.size - 1:
                    raise f"error, index larger than the number of sites"

        self._index = index

    @property
    def coupling(self):
        return self._coupling

    @coupling.setter
    def coupling(self, coupling: List):
        self._coupling = coupling

    @property
    def qutip_op(self):
        return super().qutip_op

    def __get_qutip_op(
        self,
    ):
        self._description: str = "(couplings, operator) -> \n"
        for k, tuple_indices in enumerate(self.index):

            # initialize the description
            self._description = (
                self._description + f" ( {self.coupling[k]} ,  {tuple_indices} ) \n"
            )

            coupling = self.coupling[k]
            # initialize the SpinOperator
            op_dict: Dict = {}
            indices: List = []
            for direction, idx in pairwise(tuple_indices):

                # define the given
                # local label operator

                if idx in op_dict.keys():
                    op_dict[idx] = op_dict[idx] * self._local_obs_dict[direction]
                else:
                    op_dict[idx] = self._local_obs_dict[direction]
                    indices.append(idx)

                # we fix the first part of the chain with
                # an identity operator
                if not (0 in op_dict.keys()):
                    indices.append(0)
                    op_dict[0] = self._local_obs_dict["id_fock"]

                # and the last part with another identity
                # operator
                if not (self.size - 1 in op_dict.keys()):
                    indices.append(self.size - 1)
                    op_dict[self.size - 1] = self._local_obs_dict["id_fock"]

            # order the indices
            indices.sort()
            # method for qutip.tensor optimization by Ewen Lawrence https://github.com/ewenlawrence
            # dumb index
            jdx: int = 0

            for i, idx in enumerate(indices):
                # if the sites are not nearest neighbours
                # create a identity with dim 2**(i-j)

                if idx - jdx > 1:
                    identity = qutip.identity(self.exc_numb ** (idx - jdx - 1))
                    chain_oper = qutip.tensor(identity, op_dict[idx])
                # otherwise define the operator
                else:
                    chain_oper = op_dict[idx]

                if i == 0:
                    many_body_op = chain_oper
                else:
                    many_body_op = qutip.tensor(many_body_op, chain_oper)

                jdx = idx
                # if k == 1:
                #     print("partial many body op=", many_body_op)

            # if k == 1:
            #     print("mboperator=", many_body_op)

            # sum each direction
            if k == 0:
                self._qutip_op = many_body_op * coupling
            else:

                self._qutip_op = self._qutip_op + many_body_op * coupling


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

        self.h_ao: List[ManyBodyQutipOperator] = ext_fields
        # if js is a list of coupling constants
        # initialize the coupling hamiltonian
        self.j_ao: List[ManyBodyQutipOperator] = couplings

        # other terms
        self.others_ao: List[ManyBodyQutipOperator] = extra_terms

        self.__get_qutip_op()

    @property
    def h_ao(self):
        return self._h_ao

    @h_ao.setter
    def h_ao(self, ext_fields: List[qutip.Qobj]):
        self._h_ao: List = []
        if ext_fields != None:
            for m, h in enumerate(ext_fields):

                if type(h) != ManyBodyQutipOperator:
                    raise TypeError(
                        f"element {m} of the external field list is not a qutip.Qobj ({type(h)})"
                    )

                self._h_ao.append(h)

    @property
    def j_ao(self):
        return self._j_ao

    @j_ao.setter
    def j_ao(self, couplings: List[qutip.Qobj]):
        self._j_ao: List = []
        if couplings != None:
            for m, j in enumerate(couplings):

                if type(j) != ManyBodyQutipOperator:
                    raise TypeError(
                        f"element {m} of the coupling terms list is not a qutip.Qobj ({type(j)})"
                    )

                self._j_ao.append(j)

    @property
    def others_ao(self):
        return self._others_ao

    @h_ao.setter
    def others_ao(self, other_terms: List[qutip.Qobj]):
        self._others_ao: List = []
        if other_terms != None:
            for m, o in enumerate(other_terms):

                if type(o) != ManyBodyQutipOperator:
                    raise TypeError(
                        f"element {m} of the coupling terms list is not a qutip.Qobj ({type(o)})"
                    )

                self._others_ao.append(o)

    def __str__(self) -> str:

        description = "Coupling Term: \n"
        for ham_j in self.j_ao:
            description = description + f"{ham_j}"
        description = description + "External field: \n"
        for ham_h in self.h_ao:
            description = description + f"{ham_h}"
        description = description + "External field: \n"
        for ham_o in self.others_ao:
            description = description + f"{ham_o}"
        description = description + "\n"

        return description

    def __get_qutip_op(self):

        for ham_j in self.j_ao:
            self.qutip_op = self.qutip_op + ham_j.qutip_op
        for ham_h in self.h_ao:
            self.qutip_op = self.qutip_op + ham_h.qutip_op
        for ham_o in self.others_ao:
            self.qutip_op = self.qutip_op + ham_h.qutip_op

        if self.qutip_op != None:
            if self.qutip_op.check_herm():
                print("Hermitian Check positive! well done! \n")
            else:
                raise ValueError("Non Hermitian Hamiltonian \n")


# we still can implement new attributes
# such as eigsh and gs_state
class SpinHamiltonian(Hamiltonian):
    def __init__(
        self,
        direction_couplings: List[Tuple[str]],
        field_directions: List[str],
        pbc: Optional[bool] = False,
        size: Optional[int] = None,
        coupling_values: Optional[List[float]] = None,
        field_values: Optional[List[float]] = None,
        j_couplings: Optional[List[ManyBodyQutipOperator]] = None,
        ext_fields: Optional[List[ManyBodyQutipOperator]] = None,
    ) -> None:

        super().__init__(size=size)

        # Fast Clean Transverse Ising Chain with nearest neighbourhoods

        self.__get_external_field(
            field_directions=field_directions,
            field_values=field_values,
            ext_fields=ext_fields,
        )
        self.__get_coupling_term(
            direction_couplings=direction_couplings,
            coupling_values=coupling_values,
            pbc=pbc,
            j_couplings=j_couplings,
        )

        self.__get_qutip_op()

    def __get_external_field(
        self,
        field_directions: Optional[List[str]],
        field_values: Optional[List[float]],
        ext_fields: List[ManyBodyQutipOperator],
    ):

        h_ao: List[ManyBodyQutipOperator] = []
        if field_values is not (None):
            for m, h in enumerate(field_values):
                print(h)
                index = [(field_directions[m], i) for i in range(self.size)]
                coupling = [h for i in range(self.size)]
                print(index)
                print(self.size)
                print(coupling)
                h_ao.append(
                    SpinOperator(index=index, coupling=coupling, size=self.size)
                )

        elif ext_fields is not (None):
            for m, h in enumerate(ext_fields):
                h_ao.append(h)
        self.h_ao: List[ManyBodyQutipOperator] = h_ao

    def __get_coupling_term(
        self,
        direction_couplings: Optional[List[Tuple]],
        coupling_values: Optional[List[float]],
        pbc: float,
        j_couplings: List[ManyBodyQutipOperator],
    ):
        # if js is a list of coupling constants
        # initialize the coupling hamiltonian
        self.j_ao: List[ManyBodyQutipOperator] = []
        if coupling_values is not (None):
            # initialize the coupling
            # dictionary for the abstract
            # operator

            # a loop over the different
            # couplings (e.g.: j_1xx +j_2yy  )
            for m, j in enumerate(coupling_values):
                if pbc:
                    index = [
                        (
                            direction_couplings[m][0],
                            i,
                            direction_couplings[m][1],
                            (i + 1) % self.size,
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
        elif j_couplings is not (None):
            for m, j in enumerate(j_couplings):
                self.j_ao.append(j)

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
        # get the limbladian
        self.__get_the_limbladian()

    @property
    def hamiltonian(self):
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, ham: ManyBodyQutipOperator):
        self._hamiltonian = ham

    @property
    def dissipative_ops(self):
        return self._dissipative_ops

    @dissipative_ops.setter
    def dissipative_ops(self, diss_op: List[ManyBodyQutipOperator]):
        self._dissipative_ops = diss_op

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

    def __str__(self) -> str:
        description = (
            "Unitary part=\n" + f"{self.hamiltonian}" + "\n Dissipative part=\n"
        )
        for d in self.dissipative_ops:
            description = description + f"{d}"
        description = description + "\n"
        return description

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
