import itk
import numpy as np
import polyscope as ps
from scipy import stats
from pyaof import quick_aof
import matplotlib.pyplot as plt

def generate_spherical_arc(
    grid_size: int = 512, 
    radius: float = 600.0, 
    thickness: float = 21.0, 
    arc_length: float = 200.0, 
    arc_width: float = 150.0
) -> np.ndarray:
    """
    Generates a thin spherical arc in a 3D voxel grid.
    
    Args:
        grid_size: The size of the grid along each dimension (N x N x N).
        radius: The radius of the large sphere.
        thickness: The shell thickness in voxels.
        arc_length: The extent of the arc along the X-axis.
        arc_width: The extent of the arc along the Y-axis.
        
    Returns:
        A 3D numpy array (uint8) containing the binary mask of the arc.
    """
    print(f"Generating Spherical shell in {grid_size}^3 voxel grid...")
    
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    z = np.arange(grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Center of the sphere
    cx = grid_size // 2
    cy = grid_size // 2
    cz = (grid_size // 2) - radius + (grid_size // 4)

    sphere_edt = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)

    shell = np.abs(sphere_edt - radius) <= (thickness / 2.0)

    # Restrict span of the shell
    in_x = np.abs(X - cx) <= (arc_length / 2.0)
    in_y = np.abs(Y - cy) <= (arc_width / 2.0)

    arc_volume = shell & in_x & in_y
    
    return arc_volume.astype(np.uint8)

def draw_grid_wireframe(grid_shape: tuple[int, int, int]):
    """
    Registers a 3D bounding box wireframe for any arbitrary grid shape (Nx, Ny, Nz).
    
    Args:
        grid_shape: Tuple representing grid dimensions along (X, Y, Z).
    """
    nx, ny, nz = grid_shape
    
    # Define the 8 corner vertices 
    mx, my, mz = nx - 1, ny - 1, nz - 1
    
    nodes = np.array([
        [0,  0,  0 ],  # 0
        [mx, 0,  0 ],  # 1
        [mx, my, 0 ],  # 2
        [0,  my, 0 ],  # 3
        [0,  0,  mz],  # 4
        [mx, 0,  mz],  # 5
        [mx, my, mz],  # 6
        [0,  my, mz]   # 7
    ], dtype=np.float32)

    # Define the 12 connecting edges of the rectangular prism
    edges = np.array([
        # Bottom face (z = 0)
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face (z = mz)
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical pillars (z direction)
        [0, 4], [1, 5], [2, 6], [3, 7]
    ])

    wireframe = ps.register_curve_network("Grid Bounding Box", nodes, edges)
    wireframe.set_color((0.8, 0.8, 0.8))
    wireframe.set_radius(0.0015, relative=True)


def visualize_surface(volume: np.ndarray, isovalue: float = 0.5):
    """
    Extracts the surface using scikit-image and visualizes it using Polyscope.
    
    Args:
        volume: 3D numpy array representing the volume.
        isovalue: The contour extraction threshold.
    """
    from skimage.measure import marching_cubes

    print("Visualize shell...")
    
    # Extract vertices and faces from the binary volume
    verts, faces, _, _ = marching_cubes(volume, level=isovalue)
        
    ps.init()
    
    mesh = ps.register_surface_mesh("Thin Spherical Arc", verts, faces)
    
    # properties
    mesh.set_color((0.1, 0.7, 0.7))
    mesh.set_smooth_shade(True)

    # Add wireframe box
    draw_grid_wireframe(grid_shape=volume.shape)
    
    ps.show()

def compute_sdf_itk(binary_volume: np.ndarray) -> np.ndarray:
    """
    Computes the Signed Distance Function (SDF) of a binary volume using ITK.
    """
    print("Computing Signed Danielsson Distance Map using ITK...")
    
    # Convert numpy array to ITK Image format
    itk_image = itk.image_from_array(binary_volume)

    distance_filter = itk.SignedDanielssonDistanceMapImageFilter.New(itk_image)
    distance_filter.SetUseImageSpacing(True)
    distance_filter.Update()
    sdf_itk = distance_filter.GetOutput()

    sdf_array = itk.array_from_image(sdf_itk)
    
    return sdf_array


if __name__ == "__main__":
    VISUALIZE=False
    # Generate Spherical Shell 
    GRID_SIZE = 256
    ORIGINAL_THICKNESS = 21
    volume = generate_spherical_arc(
        grid_size=GRID_SIZE,
        radius=600.0,
        thickness=ORIGINAL_THICKNESS,
        arc_length=150.0,
        arc_width=80.0
    )


    if VISUALIZE:
        visualize_surface(volume)

    # Compute Distance Function
    sdf_volume = compute_sdf_itk(volume)
    print(f"SDF ({sdf_volume.shape}) Value range: {sdf_volume.min():.2f} -> {sdf_volume.max():.2f}")
    
    aof_vol = quick_aof(sdf_volume)

    # Extract Medial Surface points
    MEDIAL_THRESHOLD = 0.3
    """
    Thresholding Considerations for Medial Surface Extraction:

    1. High Threshold (Sparser Surface):
    Filters out boundary noise, producing a cleaner skeleton. However, it selectively
    retains thicker, parallel regions, which can positively bias the calculated radius profile.

    2. Low Threshold (Denser / Branching Surface):
    Captures subtle geometric features, but is sensitive to surface roughness and discretization
    artifacts. This creates spurious skeleton branches, negatively biasing the calculated 
    radius profile toward smaller values.
    """
   
    medial_surface = np.argwhere(aof_vol > MEDIAL_THRESHOLD)

    thickness_values = -1*sdf_volume[medial_surface[:, 0],
                                     medial_surface[:, 1],
                                     medial_surface[:, 2]]
    
    estimated_radius,_ = stats.mode(thickness_values)
    print(f"Estimated thickness : {estimated_radius*2} with variance of {np.var(thickness_values)}")
    print(f"Original thickness: {ORIGINAL_THICKNESS}")
    if VISUALIZE:
        plt.hist(thickness_values,
                color='#2b5c8f', 
                edgecolor='black', 
                linewidth=0.5,
                alpha=0.85)
        plt.xlabel("Signed Distance to Arc Surface (voxels)", fontsize=11)
        plt.ylabel("Voxel Count", fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.show()
        #plt.savefig("/tmp/thickness_hist.png")
