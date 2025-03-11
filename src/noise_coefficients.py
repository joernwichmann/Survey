from firedrake import *

def NON_SOLENOIDAL_CarelliHausenblasProhl(end_level: int, mesh: MeshGeometry, velocity_space: FunctionSpace) -> list[Function]:
    """Returns a list of noise coefficients based on the construction for NON-SOLENOIDAL, additive noise in Carelli, Hausenblas and Prohl.
    Parameters: 
        -   end_level:  truncation index of the sum
        -   mesh:       triangulation of the domain
        -   velocity_space: finite element space used for the approximation of velocity
    """
    x, y = SpatialCoordinate(mesh)
    noiseCoefficients = []
    for j in range(1,end_level+1):
        for k in range(1,end_level+1):
            expr = 1/((j+k)*(j+k))*as_vector([
                sin(j*pi*x)*sin(k*pi*y),
                sin(j*pi*x)*sin(k*pi*y)
                ])
            noiseCoefficients.append(project(expr, velocity_space))
    return noiseCoefficients

def SOLENOIDAL_CarelliHausenblasProhl(end_level: int, mesh: MeshGeometry, velocity_space: FunctionSpace) -> list[Function]:
    """Returns a list of noise coefficients based on the construction for SOLENOIDAL, additive noise in Carelli, Hausenblas and Prohl.
    Parameters: 
        -   end_level:  truncation index of the sum
        -   mesh:       triangulation of the domain
        -   velocity_space: finite element space used for the approximation of velocity
    """
    x, y = SpatialCoordinate(mesh)
    noiseCoefficients = []
    for j in range(1,end_level+1):
        expr = 1/(j*j)*as_vector([
            -1.0*cos(j*pi*x - pi/2.0)*sin(j*pi*y - pi/2.0),
            sin(j*pi*x - pi/2.0)*cos(j*pi*y - pi/2.0)
            ])
        noiseCoefficients.append(project(expr, velocity_space))
    return noiseCoefficients

def NON_SOLENOIDAL_NEW(end_level: int, mesh: MeshGeometry, velocity_space: FunctionSpace) -> list[Function]:
    """Returns a list of noise coefficients based on the construction for NON-SOLENOIDAL, additive noise in Carelli, Hausenblas and Prohl.
    Parameters: 
        -   end_level:  truncation index of the sum
        -   mesh:       triangulation of the domain
        -   velocity_space: finite element space used for the approximation of velocity
    """
    x, y = SpatialCoordinate(mesh)
    noiseCoefficients = []
    for j in range(1,end_level+1):
        for k in range(1,end_level+1):
            expr = 1/((j+k)*(j+k))*as_vector([
                sin(j*pi*x)*sin(k*pi*y),
                sin(k*pi*x)*sin(j*pi*y)
                ])
            noiseCoefficients.append(project(expr, velocity_space))
    return noiseCoefficients