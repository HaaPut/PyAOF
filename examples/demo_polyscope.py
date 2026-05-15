import os
import sys
import vtk
import logging
import numpy as np
import polyscope as ps
from types import SimpleNamespace
from vtk.util.numpy_support import vtk_to_numpy

from pyaof import compute_aof, compute_sdf

logger = logging.getLogger('pyaof_demo')
logger.setLevel(level=logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)



def extract_vertices_and_faces(vtk_mesh):
    """Helper to extract numpy arrays of vertices and faces from a vtkPolyData object."""
    # Extract Vertices
    points = vtk_mesh.GetPoints()
    vertices = vtk_to_numpy(points.GetData())
    
    polys = vtk_mesh.GetPolys()
    poly_array = vtk_to_numpy(polys.GetData())
    
    try:
        faces = poly_array.reshape(-1, 4)[:, 1:4]
    except ValueError:
        logger.warning("Mesh contains non-triangle faces. Attempting triangulation...")
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputData(vtk_mesh)
        triangle_filter.Update()
        tri_mesh = triangle_filter.GetOutput()
        poly_array = vtk_to_numpy(tri_mesh.GetPolys().GetData())
        faces = poly_array.reshape(-1, 4)[:, 1:4]
        
    return vertices, faces

def main():
    params = SimpleNamespace()
    params.obj = 'cuboid'
    params.resolution = 1000
    params.normalize = True
    params.threshold = 0.15

    data_folder = '../data'
    obj_path = os.path.join(data_folder, f"{params.obj}.obj")
    
    if not os.path.exists(obj_path):
        logger.critical(f"Object {obj_path} not found")
        sys.exit(-1)

    if obj_path.endswith('obj'):
        reader = vtk.vtkOBJReader()
    elif obj_path.endswith('off'):
        reader = vtk.vtkOFFReader()
    else:
        logger.critical("Unsupported file format.")
        sys.exit(-1)
        
    reader.SetFileName(obj_path)
    reader.Update()
    original_mesh = reader.GetOutput()

    logger.debug("Computing SDF...")
    sdf, normalized_mesh, mesh_vertices, spacing = compute_sdf(
        original_mesh,
        resolution=params.resolution,
        normalize=params.normalize
    )

    logger.debug("Computing AOF...")
    aof_vol = compute_aof(
        sdf,
        hx=spacing[0],
        hy=spacing[1],
        hz=spacing[2],
        tile_size=1024
    )

    logger.debug(f"Thresholding AOF at {params.threshold}...")
    binary_mask = aof_vol > params.threshold
    indices = np.argwhere(binary_mask)
    
    xmin = -1
    ymin = -1
    zmin = -1

    pts_x = xmin + indices[:, 0] * spacing[0]
    pts_y = ymin + indices[:, 1] * spacing[1]
    pts_z = zmin + indices[:, 2] * spacing[2]
    
    aof_point_cloud = np.column_stack((pts_z, pts_y, pts_x))
    logger.debug(f"Generated {len(aof_point_cloud)} points from the thresholded AOF.")

    # ................ Visualize with Polyscope ..............
    logger.debug("Launching Polyscope...")
    ps.init()

    mesh_vertices_np, mesh_faces_np = extract_vertices_and_faces(normalized_mesh)
    ps_mesh = ps.register_surface_mesh("Normalized Mesh", mesh_vertices_np, mesh_faces_np)
    
    ps_mesh.set_transparency(0.4)

    if len(aof_point_cloud) > 0:
        ps_cloud = ps.register_point_cloud("AOF Thresholded skeleton", aof_point_cloud)
        ps_cloud.set_radius(0.005, relative=False) 
        ps_cloud.set_color((1.0, 0.2, 0.2))
    else:
        logger.warning("No points found above the threshold!")

    ps.show()

if __name__ == "__main__":
    main()