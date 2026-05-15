import vtk
import itk
import numpy as np
import scipy.ndimage as ndi
from typing import Tuple
from vtk.util.numpy_support import vtk_to_numpy

import logging

logger = logging.getLogger(__name__)

def normalize_mesh_to_unit(
    mesh: vtk.vtkPolyData,
    padding: float = 0.01
) -> vtk.vtkPolyData:
    """Normalize a VTK mesh into a unit cube centered at the origin.
    A small padding is added to avoid boundary clipping.

    Args:
        mesh (vtk.vtkPolyData):
            Input triangular mesh to normalize.

        padding (float, optional):
            Extra margin added to the mesh bounding radius before scaling.
            Prevents tight clipping at the unit cube boundary.
            Defaults to 0.01.

    Returns:
        vtk.vtkPolyData:
            Normalized mesh centered at the origin and scaled to fit
            inside a unit cube.

    Example:
        >>> normalized = normalize_mesh_to_unit(mesh)
    """
    bounds = mesh.GetBounds()
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)

    dx = max(abs(xmin - cx), abs(xmax - cx))
    dy = max(abs(ymin - cy), abs(ymax - cy))
    dz = max(abs(zmin - cz), abs(zmax - cz))

    scale_val = max(dx, dy, dz) + padding

    scale = 1.0 / scale_val

    transform = vtk.vtkTransform()
    transform.PostMultiply()

    transform.Translate(-cx, -cy, -cz)
    transform.Scale(scale, scale, scale)

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(mesh)
    tf.SetTransform(transform)
    tf.Update()

    return tf.GetOutput()

def voxelize_vtk(
    mesh: vtk.vtkPolyData,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (-0.5, -0.5, -0.5),
    resolution: int = 256,
    padding: float = 0.1,
    inside_value: int = 1,
    outside_value: int = 0,
    normalize: bool = False
) -> tuple[
    np.ndarray,
    tuple[float, float, float],
    vtk.vtkPolyData,
    np.ndarray
]:
    """Voxelize a VTK mesh into a 3D binary volume.

    This function converts a triangular VTK mesh into a structured 3D
    voxel grid using VTK's image stencil pipeline. It performs several
    preprocessing steps (cleaning, triangulation, hole filling, and
    normal correction) before rasterizing the mesh into a volumetric grid.

    Args:
        mesh (vtk.vtkPolyData):
            Input triangular surface mesh.

        spacing (tuple[float, float, float], optional):
            Voxel spacing in (x, y, z). Defaults to (1.0, 1.0, 1.0).

        origin (tuple[float, float, float], optional):
            Origin of the voxel grid. Defaults to (-0.5, -0.5, -0.5).

        resolution (int, optional):
            Number of voxels along each axis (NxNxN grid).
            Defaults to 256.

        padding (float, optional):
            Extra padding applied during normalization to avoid clipping.
            Defaults to 0.1.

        inside_value (int, optional):
            Value assigned to voxels inside the mesh. Defaults to 1.

        outside_value (int, optional):
            Value assigned to voxels outside the mesh. Defaults to 0.

        normalize (bool, optional):
            If True, scales mesh into a unit cube before voxelization
            and overrides spacing/origin accordingly. Defaults to False.

    Returns:
        tuple:
            A tuple containing:

            - **np.ndarray**:
                3D voxel grid of shape (Nz, Ny, Nx) containing
                binarized object.

            - **tuple[float, float, float]**:
                Effective voxel spacing used for the grid.

            - **vtk.vtkPolyData**:
                Processed (cleaned + normalized) mesh used for voxelization.

            - **np.ndarray**:
                Vertex coordinates of the processed mesh in reordered
                (z, y, x) format.

    Example:
        >>> vol, spacing, mesh_n, coords = voxelize_vtk(
        ...     mesh,
        ...     resolution=1024,
        ...     normalize=True
        ... )
    """
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(mesh)
    clean.Update()

    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputConnection(clean.GetOutputPort())
    tri_filter.Update()

    # Fill holes 
    fill = vtk.vtkFillHolesFilter()
    fill.SetInputConnection(tri_filter.GetOutputPort())
    fill.SetHoleSize(1000) 
    fill.Update()

    # Fix normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(fill.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    mesh = normals.GetOutput()
    mesh = normalize_mesh_to_unit(mesh, padding=padding)


    if normalize:
        output_spacing = (2.0 / resolution,) * 3
        origin = (-1.0 + output_spacing[0] / 2.0,) * 3

    else:
        output_spacing = spacing
        origin = (0.0,0.0,0.0)
    
    image = vtk.vtkImageData()
    image.SetSpacing(output_spacing)
    image.SetOrigin(origin)
    image.SetDimensions(resolution, resolution, resolution)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Fill with inside value
    scalars = vtk_to_numpy(image.GetPointData().GetScalars())
    scalars[:] = inside_value
    
    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(mesh)
    poly_to_stencil.SetOutputSpacing(output_spacing)
    poly_to_stencil.SetOutputOrigin(origin)
    poly_to_stencil.SetOutputWholeExtent(image.GetExtent())
    poly_to_stencil.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(image)
    stencil.SetStencilConnection(poly_to_stencil.GetOutputPort())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(outside_value)
    stencil.Update()

    output = stencil.GetOutput()

    # Convert to NumPy
    vtk_array = output.GetPointData().GetScalars()
    np_array = vtk_to_numpy(vtk_array)

    dims = output.GetDimensions()
    np_array = np_array.reshape(dims[2], dims[1], dims[0])

    coords = vtk_to_numpy(mesh.GetPoints().GetData())
     # swap x and y to match voxel grid orientation
    coords = np.stack([coords[:,2],coords[:,1],coords[:,0]], axis=-1) 
    return np_array, output_spacing, mesh, coords



def _sdf(inside_mask, spacing, *, smoothing=False):
    # Convert numpy array to ITK image
    itk_image = itk.image_view_from_array(inside_mask.astype(np.float32))
    itk_image.SetSpacing(spacing) 
    distance_filter = itk.SignedMaurerDistanceMapImageFilter[type(itk_image), type(itk_image)].New()
    distance_filter.SetInput(itk_image)
    
    distance_filter.SetInsideIsPositive(False) 
    distance_filter.SetSquaredDistance(False)
    distance_filter.SetUseImageSpacing(True)
    
    distance_filter.Update()
    
    sdf = itk.array_from_image(distance_filter.GetOutput())
    if smoothing:
        sdf = ndi.gaussian_filter(sdf, sigma=1.0)

    return sdf

def compute_sdf(mesh: vtk.vtkPolyData,
                resolution: int,
                padding=0.1,
                normalize=False,
                smoothing=True,
    ) -> tuple[
    np.ndarray,
    vtk.vtkPolyData,
    np.ndarray,
    tuple[float, float, float]
]:
    """Compute a Signed Distance Field (SDF) from a VTK mesh.

    This function voxelizes the input mesh, builds an occupancy grid,
    and computes a signed distance field (SDF) using ITK signed distance transform.

    Args:
        mesh (vtk.vtkPolyData):
            Input VTK mesh.

        resolution (int):
            Resolution of the voxel grid along the longest axis.

        padding (float, optional):
            Extra padding around mesh bounds to avoid clipping artifacts.
            Defaults to 0.1.

        normalize (bool, optional):
            If True, scales mesh into a unit cube before voxelization.
            Defaults to False.
        smoothing (bool, optional):
            If True, smoothes the sdf to avoid staircasing problems at the boundary.
            Defaults to True.


    Returns:
        tuple[np.ndarray, vtk.vtkPolyData, np.ndarray, tuple[float, float, float]]:
            A tuple containing:

            - **sdf (np.ndarray)**:
                Signed distance field volume of shape (N, N, N).

            - **normalized_mesh (vtk.vtkPolyData)**:
                Mesh used for voxelization (normalized if enabled).

            - **coords (np.ndarray)**:
                3D grid coordinates corresponding to voxel centers.

            - **spacing (tuple[float, float, float])**:
                Grid spacing in (x, y, z) directions.

    Notes:
        - SDF sign convention: negative inside, positive outside.
        - Requires VTK for mesh processing.
        - Distance computation uses Euclidean distance transform.

    Example:
        >>> sdf, mesh_n, coords, spacing = compute_sdf(
        ...     mesh,
        ...     resolution=1024,
        ...     normalize=True
        ... )
    """
    logger.debug("Compute sdf @ resolution = %d" % (resolution))
    voxels, spacing, normalized_mesh, coords = voxelize_vtk(mesh,
                                                            resolution=resolution,
                                                            padding=padding,
                                                            normalize=normalize)
    inside = voxels > 0

    sdf = _sdf(inside, spacing, smoothing=smoothing)
    return sdf, normalized_mesh, coords, spacing
