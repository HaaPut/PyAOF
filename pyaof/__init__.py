import warp as wp
import logging

wp.config.quiet = True
wp.init()

# Define the library logger with a NullHandler (prevents hijacking user's logging)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from .meshing import compute_sdf, voxelize_vtk, normalize_mesh_to_unit
from .aof import compute_aof, quick_aof, sample_sphere_points, nonmax_supp
from .st import structure_tensor_3d, eig_special_3d

__all__ = [
    "compute_sdf",
    "normalize_mesh_to_unit",
    "voxelize_vtk",
    "compute_aof",
    "quick_aof" 
    "sample_sphere_points",
    "structure_tensor_3d", 
    "eig_special_3d"
]