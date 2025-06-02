from firedrake import *

def _poly(x,y):
    return 2*x*x*(1-x)*(1-x)*y*(y-1)*(2*y-1)

def _force(x,y):
    return 4*x*x*y*(2*y*y-3*y+1) + 16*x*y*(x-1)*(2*y*y-3*y+1) + 4*y*(x-1)*(x-1)*(2*y*y-3*y+1) + 4*x*x*(6*y-3)*(x-1)*(x-1)

def noiseCoefficient(x,y):
    """Returns the noise coefficient based on the BENCHMARK: Survey -- Example 4.2
    """
    expr = as_vector([1,1])
    return expr

def initialCondition(x,y):
    """Returns the initial condition based on the BENCHMARK: Survey -- Example 4.2
    """
    expr = as_vector([
        _poly(x,y),
        -1*_poly(y,x)
        ])
    return expr

def bodyforce1(x,y):
    """Returns the body force based on the BENCHMARK: Survey -- Example 4.2
    """
    expr = as_vector([
        _force(x,y),
        -1*_force(y,x)
        ])
    return expr

def bodyforce2(x,y):
    """Returns the body force based on the BENCHMARK: Survey -- Example 4.2
    """
    expr = as_vector([x,y])
    return expr

def exact_velocity(x, y):
    expr = as_vector([
        _poly(x,y),
        -1*_poly(y,x)
        ])
    return expr

def exact_pressure(x, y):
    expr = x*x + y*y
    return expr

def transformation(x,y,expW):
    return x*expW, y*expW

