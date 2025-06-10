from firedrake import *

from src.math.norms.space import l2_space

def kinetic_distance(function1, function2) -> float:
    integratedFunc1 = l2_space(function1)
    integratedFunc2 = l2_space(function2)
    return integratedFunc1 - integratedFunc2

def SIN_distance(function1, function2) -> float:
    integratedFunc1 = assemble(sin(sqrt(inner(function1,function1)))*dx)
    integratedFunc2 = assemble(sin(sqrt(inner(function2,function2)))*dx)
    return integratedFunc1 - integratedFunc2