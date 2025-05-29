from firedrake import *

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
        _poly(x,y) + _poly(y,x),
        -1*_poly(y,x) + _poly(x,y)
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

