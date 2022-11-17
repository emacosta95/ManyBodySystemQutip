from typing import List


class Operator:
    def __init__(self, s, site, value) -> None:
        self.s = s
        self.site = site
        self.value = value


def elementary_commutator(s1: Operator, s2: Operator):

    if s1.site != s2.site:
        return None
    if s1.site == s2.site:
        if s1 == s2:
            return None
        if s1 == "X" and s2 == "Y":
            return Operator(s="Z", site=s1.site, value=(s1.value * s2.value * 1j))
        if s1 == "X" and s2 == "Z":
            return Operator(s="Y", site=s1.site, value=(s1.value * s2.value * 1j))
        if s1 == "Y" and s2 == "X":
            return Operator(s="Z", site=s1.site, value=(-1 * s1.value * s2.value * 1j))
        if s1 == "Y" and s2 == "Z":
            return Operator(s="X", site=s1.site, value=(s1.value * s2.value * 1j))
        if s1 == "Z" and s2 == "X":
            return Operator(s="Y", site=s1.site, value=(-1 * s1.value * s2.value * 1j))
        if s1 == "Z" and s2 == "Y":
            return Operator(s="X", site=s1.site, value=(-1 * s1.value * s2.value * 1j))


def commutator(p1: List[Operator], p2: List[Operator]):

    s = []
    for i, a in enumerate(p2):
        p_before = p2[:i]
        p_after = p2[i:]
        for j, b in enumerate(p1):
            new_list = p1.copy()
            new_list[j] = elementary_commutator(b, a)
            s.append(p_before + new_list + p_after)
