from firedrake import *
from typing import TypeAlias, Callable, Any, Tuple

from src.discretisation.space import SpaceDiscretisation
from src.discretisation.projections import Stokes_projection, HL_projection, HL_projection_withBC


def knownVelocity(mesh: MeshGeometry, velocity_space: FunctionSpace) -> Function:
    """Returns the analytic velocity projected onto the finitie element space."""
    x, y = SpatialCoordinate(mesh)
    expr = as_vector([
        x**2*(1-x)**2*(2*y-6*y**2+4*y**3),
        -y**2*(1-y)**2*(2*x-6*x**2+4*x**3)
        ])
    return project(expr, velocity_space)

def knownPressure(time: float, gamma: float, mesh: MeshGeometry, pressure_space: FunctionSpace) -> Function:
    """Returns the analytic pressure projected onto the finitie element space at time "time". The parameter gamma quantifies the time regularity."""
    x, y = SpatialCoordinate(mesh)
    expr = (x**2 + y**2 - 2/3.0)*time**gamma
    return Function(pressure_space).interpolate(expr)

def knownForcing(time: float,
                         gamma: float,
                         viscosity: float,
                         mesh: MeshGeometry,
                         velocity_space: FunctionSpace) -> Function:
    """Returns the forcing evaluated projected onto the finite element space at time "time". The forcing depends on
    -   gamma: the time regularity parameter of pressure
    -   viscosity: the inverse Reynolds number of the system"""
    x, y = SpatialCoordinate(mesh)
    expr = as_vector([
        -4*viscosity*(x**2*y*(2*y**2 - 3*y + 1) + x**2*(x - 1)**2*(6*y - 3) + 4*x*y*(x - 1)*(2*y**2 - 3*y + 1) + y*(x - 1)**2*(2*y**2 - 3*y + 1)) + 2*time**gamma*x,
        4*viscosity*(x*y**2*(2*x**2 - 3*x + 1) + 4*x*y*(y - 1)*(2*x**2 - 3*x + 1) + x*(y - 1)**2*(2*x**2 - 3*x + 1) + y**2*(6*x - 3)*(y - 1)**2) + 2*time**gamma*y
        ])
    return project(expr, velocity_space)


        
