from firedrake import *

def SIN_BASIS(end_level: int, mesh: MeshGeometry, velocity_space: FunctionSpace, r: float) -> list[Function]:
    """Returns a list of noise coefficients based on the BENCHMARK: Survey -- Example 6.8
    Parameters: 
        -   end_level:  truncation index of the sum
        -   mesh:       triangulation of the domain
        -   velocity_space: finite element space used for the approximation of velocity
        -   r: rate of decay of high frequencies, i.e., determines the spatial regularity of noise
    """
    x, y = SpatialCoordinate(mesh)
    noiseCoefficients = []
    for j in range(1,end_level+1):
        for k in range(1,end_level+1):
            expr = (1.0/((j*j + k*k)**(r/2.0)))*as_vector([
                sin(j*pi*x)*sin(k*pi*y),
                sin(k*pi*x)*sin(j*pi*y)
                ])
            noiseCoefficients.append(project(expr, velocity_space))
    return noiseCoefficients

def COS_BASIS(end_level: int, mesh: MeshGeometry, velocity_space: FunctionSpace, r: float) -> list[Function]:
    """Returns a list of noise coefficients based on the BENCHMARK: Survey -- Example 6.8
    Parameters: 
        -   end_level:  truncation index of the sum
        -   mesh:       triangulation of the domain
        -   velocity_space: finite element space used for the approximation of velocity
        -   r: rate of decay of high frequencies, i.e., determines the spatial regularity of noise
    """
    x, y = SpatialCoordinate(mesh)
    noiseCoefficients = []
    for j in range(1,end_level+1):
        for k in range(1,end_level+1):
            expr = (1.0/((j*j + k*k)**(r/2.0)))*as_vector([
                cos(j*pi*x)*cos(k*pi*y),
                cos(k*pi*x)*cos(j*pi*y)
                ])
            noiseCoefficients.append(project(expr, velocity_space))
    return noiseCoefficients
