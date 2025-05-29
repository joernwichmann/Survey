from firedrake import *
from src.discretisation.time import increments_to_trajectory

def _poly(x,y):
    return 2*x*x*(1-x)*(1-x)*y*(y-1)*(2*y-1)

def _force(x,y):
    return 4*x*x*y*(2*y*y-3*y+1) + 16*x*y*(x-1)*(2*y*y-3*y+1) + 4*y*(x-1)*(x-1)*(2*y*y-3*y+1) + 4*x*x*(6*y-3)*(x-1)*(x-1)

def noiseCoefficient(mesh: MeshGeometry, velocity_space: FunctionSpace) -> Function:
    """Returns the noise coefficient based on the BENCHMARK: Survey -- Example 4.2
    Parameters: 
        -   mesh:       triangulation of the domain
        -   velocity_space: finite element space used for the approximation of velocity
    """
    x, y = SpatialCoordinate(mesh)
    expr = as_vector([
        _poly(x,y),
        -1*_poly(y,x)
        ])
    return project(expr, velocity_space)

def initialCondition(mesh: MeshGeometry, velocity_space: FunctionSpace) -> Function:
    """Returns the initial condition based on the BENCHMARK: Survey -- Example 4.2
    Parameters: 
        -   mesh:       triangulation of the domain
        -   velocity_space: finite element space used for the approximation of velocity
    """
    x, y = SpatialCoordinate(mesh)
    expr = as_vector([
        _poly(x,y),
        -1*_poly(y,x)
        ])
    return project(expr, velocity_space)

def bodyforce1(mesh: MeshGeometry, velocity_space: FunctionSpace) -> Function:
    """Returns the body force based on the BENCHMARK: Survey -- Example 4.2
    Parameters: 
        -   mesh:       triangulation of the domain
        -   velocity_space: finite element space used for the approximation of velocity
    """
    x, y = SpatialCoordinate(mesh)
    expr = as_vector([
        _force(x,y),
        -1*_force(y,x)
        ])
    return project(expr, velocity_space)

def bodyforce2(mesh: MeshGeometry, velocity_space: FunctionSpace) -> Function:
    """Returns the body force based on the BENCHMARK: Survey -- Example 4.2
    Parameters: 
        -   mesh:       triangulation of the domain
        -   velocity_space: finite element space used for the approximation of velocity
    """
    x, y = SpatialCoordinate(mesh)
    expr = as_vector([
        x,
        y
        ])
    return project(expr, velocity_space)

def time_to_exact_velocity(mesh, velocity_space, time_grid, noiseIncrements) -> dict[float,Function]:
    x, y = SpatialCoordinate(mesh)
    expr = as_vector([
        _poly(x,y),
        -1*_poly(y,x)
        ])
    W= increments_to_trajectory(0,noiseIncrements)
    exprFunction = project(expr, velocity_space)
    return {time: exprFunction*(1+W[index]) for index, time in enumerate(time_grid)}

def time_to_exact_pressure(mesh, pressure_space, time_grid, noiseIncrements) -> dict[float,Function]:
    x, y = SpatialCoordinate(mesh)
    expr = x*x + y*y - 2.0/3.0
    exprFunction = project(expr, pressure_space)
    return {time: exprFunction*time for index, time in enumerate(time_grid)}

def _hill_wave(mesh, velocity_space) -> Function:
    x, y = SpatialCoordinate(mesh)
    expr = as_vector([
        sin(pi*x)*sin(pi*y),
        sin(2*pi*x)*sin(2*pi*y)
        ])
    return project(expr, velocity_space)

