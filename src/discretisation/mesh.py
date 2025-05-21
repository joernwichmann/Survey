from firedrake import UnitSquareMesh, COMM_WORLD, Mesh

from src.string_formatting import format_header

### converter that maps a mesh name and some resolution parameter to an implemenation
def get_mesh(name_mesh: str, space_points: int, comm = COMM_WORLD):
    """Return mesh based on name and resolution."""
    match name_mesh:
        case "unit square":
            return UnitSquareMesh(space_points,space_points,name = name_mesh,comm=comm)
        case "unit_square_non_singular":
            raise NotImplementedError
        case "unit L-shape":
            raise NotImplementedError
        case other:
            raise NotImplementedError

class MeshObject:
    """Store mesh parameter."""
    def __init__(self, name_mesh: str, space_points: int, comm = COMM_WORLD):
        self.name = name_mesh
        self.space_points = space_points
        self.mesh = get_mesh(name_mesh, space_points, comm)

    def __str__(self):
        out = format_header("MESH")
        out += f"\nName: \t \t {self.name}"
        out += f"\nSpace points: \t {self.space_points}"
        return out
