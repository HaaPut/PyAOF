import numpy as np
import warp as wp
from .st import structure_tensor_3d, eig_special_3d


def sample_sphere_points(n_points:int, *, dimension=3, iterations=50)->np.ndarray:
    """Generate approximately uniform points on a unit sphere.
    This function generates points on the surface of a unit sphere using
    an iterative electrostatic repulsion method. Points are initialized
    randomly and repeatedly pushed apart using inverse-square repulsive
    forces to improve uniformity.

    The first point is fixed at ``[1, 0, 0, ...]`` to stabilize the
    optimization and remove rotational ambiguity.

    Args:
        n_points (int):
            Number of points to generate on the sphere surface.

        dimension (int, optional):
            Dimensionality of the embedding space. Defaults to ``3``.

            - ``dimension=3`` generates points on a standard 3D sphere.
            - Higher values generate points on hyperspheres.

        iterations (int, optional):
            Number of relaxation iterations used to improve point
            distribution. Defaults to ``50``.

    Returns:
        np.ndarray:
            Array of shape ``(n_points, dimension)`` containing normalized
            direction vectors on the unit sphere surface.

    Example:
        >>> dirs = sample_sphere_points(
        ...     n_points=128,
        ...     dimension=3,
        ...     iterations=100
        ... )
        >>> dirs.shape
        (128, 3)

    """
    dirs = np.random.rand(n_points, dimension)
    # First point is fixed at [1, 0, 0...], others are random
    dirs[0, :] = 0
    dirs[0, 0] = 1.0
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    for _ in range(iterations):
        # 2. Calculate pairwise difference vectors: (N, N, Dim)
        # points[:, np.newaxis, :] is (N, 1, Dim)
        # points[np.newaxis, :, :] is (1, N, Dim)
        diff = dirs[:, np.newaxis, :] - dirs[np.newaxis, :, :]
        
        r2 = np.sum(diff**2, axis=-1)
        np.fill_diagonal(r2, np.inf)
        forces_ij = diff / r2[:, :, np.newaxis]
        total_forces = np.sum(forces_ij, axis=1)
        
        # Points[0] is fixed, so we only update points[1:] and Re-normalize
        dirs[1:] += total_forces[1:]
        dirs[1:] /= np.linalg.norm(dirs[1:], axis=-1, keepdims=True)

    return dirs


@wp.func
def interp(sdf: wp.array3d(dtype=float),
           x: float, y: float, z: float)->float:

    i = int(wp.floor(x))
    j = int(wp.floor(y))
    k = int(wp.floor(z))

    #return sdf[i,j,k]    
    tx = (x - float(i))
    ty = (y - float(j))
    tz = (z - float(k))

    c000 = sdf[i,     j,     k    ]
    c100 = sdf[i + 1, j,     k    ]
    c010 = sdf[i,     j + 1, k    ]
    c110 = sdf[i + 1, j + 1, k    ]
    c001 = sdf[i,     j,     k + 1]
    c101 = sdf[i + 1, j,     k + 1]
    c011 = sdf[i,     j + 1, k + 1]
    c111 = sdf[i + 1, j + 1, k + 1]

    return wp.lerp(
        wp.lerp(wp.lerp(c000, c100, tx), wp.lerp(c010, c110, tx), ty),
        wp.lerp(wp.lerp(c001, c101, tx), wp.lerp(c011, c111, tx), ty),
        tz
    )

@wp.func
def aof_stencil_2d(
    sdf: wp.array2d(dtype=float), 
    i: int, j: int,
    hx: float, hy: float,
) -> float:
    
    inv_hx = 1.0/hx
    inv_hy = 1.0/hy
    inv_hxy = 1.0 / wp.sqrt(hx*hx + hy*hy)
    # Load the core 9-point stencil into registers to prevent repeated lookup
    c   = sdf[i, j]
    
    # Orthogonal
    xp = sdf[i+1, j]; xm = sdf[i-1, j]
    yp = sdf[i, j+1]; ym = sdf[i, j-1]
    # Edge/Plane Diagonals
    xypp = sdf[i+1, j+1]; xypm = sdf[i+1, j-1]
    xymp = sdf[i-1, j+1]; xymm = sdf[i-1, j-1]
    
    # Compute flux 
    sum_flux = (xp + xm) * inv_hx + \
               (yp + ym) * inv_hy + \
               (xypp + xypm + xymp + xymm) * inv_hxy 

 
    # Centeral weight:
    # (inverse) Distances to (8) neighbors
    # 4 orthogonal nbrs +
    # 4 corners + 

    w_c = 2.0*(inv_hx + inv_hy) + \
          4.0*inv_hxy
    #(1/8 = 0.124)
    return (sum_flux - w_c * c) * 0.125

@wp.func
def spherical_aof_stencil(
    sdf: wp.array3d(dtype=float),
    offsets: wp.array(dtype=wp.vec3),
    i: float, j: float, k: float, 
    hx:float, hy:float, hz:float,
) -> float:
    eps = 1e-9
    N = offsets.shape[0]
    radius = 1.0 # wp.sqrt(1.0)
    # radius = wp.sqrt(.0)
    dx, dy, dz = hx*radius, hy*radius, hz*radius

    sum_flux = float(0.0)
    for n in range(N):
        offset = offsets[n]
        nx = radius*float(offset[0])
        ny = radius*float(offset[1])
        nz = radius*float(offset[2])
        x, y, z = i+nx, j+ny, k+nz

        gx = (interp(sdf, x+dx, y, z) - interp(sdf, x-dx, y, z))/(2.0*dx)
        gy = (interp(sdf, x, y+dy, z) - interp(sdf, x, y-dy, z))/(2.0*dy)
        gz = (interp(sdf, x, y, z+dz) - interp(sdf, x, y, z-dz))/(2.0*dz)

        # Normalize
        grad_mag = wp.max( wp.sqrt(gx*gx + gy*gy + gz*gz), eps )
        n_mag = wp.sqrt(nx*nx + ny*ny + nz*nz) #+ eps
        
        # Project the gradient onto the direction..
        # don't neet to change sign for -ve directions 
        # since it is already swapped in grad component computation above.
        proj_grad = 0.0
        proj_grad += (gx/grad_mag) * (nx/n_mag)
        proj_grad += (gy/grad_mag) * (ny/n_mag)
        proj_grad += (gz/grad_mag) * (nz/n_mag)

        sum_flux += proj_grad
        
    return sum_flux / float(N)




@wp.kernel
def aof_kernel_3d(
    sdf: wp.array3d(dtype=float),
    aof: wp.array3d(dtype=float),
    offsets: wp.array(dtype=wp.vec3),
    hx:float, hy:float, hz:float,
):
    # Warp automatically unpacks 3D launch dimensions
    i, j, k = wp.tid()

    nx = sdf.shape[0]
    ny = sdf.shape[1]
    nz = sdf.shape[2]
    
    # Skip borders to prevent out-of-bounds access
    if i <= 0 or j <= 0 or k <= 0 or i >= nx-1 or j >= ny-1 or k >= nz-1:
        return
    aof[i, j, k] = spherical_aof_stencil(sdf,
                                         offsets,
                                         float(i), float(j), float(k),
                                         hx, hy, hz,
                                        )


@wp.kernel
def aof_kernel_2d(
    sdf: wp.array2d(dtype=float),
    aof: wp.array2d(dtype=float),
    hx:float, hy:float,
):
    # Warp automatically unpacks 3D launch dimensions
    i, j = wp.tid()
    #nx,ny,nz = sdf.shape
    nx = sdf.shape[0]
    ny = sdf.shape[1]
    
    # Skip borders to prevent out-of-bounds access
    if i <= 0 or j <= 0 or i >= nx-1 or j >= ny-1:
        return
    aof[i, j] = aof_stencil_2d(sdf,
                              i, j,
                              hx,hy,
                            )


def get_tangents(aof, sigma=2, rho=1.5, aof_threshold = 0.3, return_sparse=True, *,
                 tile_size=None, batch_size=None):
    """
    Computes the Eigen-Decomposition of 3D Structure Tensor of aof volume 
    at points above aof_threshold.

    This function supports both dense (full volume) and sparse (feature-only) 
    computations. It returns  sorted eigenvalues and their corresponding eigenvectors.

    Args:
        aof (ndarray): 
            Input 3D array (Z, Y, X) representing intensity data.
        sigma (float): 
            Standard deviation for the initial Gaussian gradient smoothing. 
            Controls the scale of the features to be detected.
        rho (float): 
            Standard deviation for the post-tensor smoothing (integration scale). 
            Controls the neighborhood size for local orientation estimation.
        aof_threshold (float, optional): 
            - If float: Only  points above this value are computed (Sparse Mode).
            - If None: The entire volume is computed (Dense Mode).
        tile_size (int, optional):
            Tile size for computation of batched AOF
        batch_size (int, optional):
            Batch size for eigen vector decomposition of structure tensors.
    Returns:
        e_vals (ndarray): 
            Eigenvalues sorted in ascending order (l1 < l2 < l3).
            Shape: (3, Z, Y, X) for Dense; (3, N) for Sparse.
        e_vecs (ndarray): 
            Eigen vectors stored as a 3x3 matrix where columns 
            correspond to l1, l2, and l3 respectively.
            Shape: (3, 3, Z, Y, X) for Dense; (3, 3, N) for Sparse.
        fa (ndarray): 
            The fractional anisotropy measure
            Shape: (Z, Y, X) for Dense; (N,) for Sparse.

    """
    if tile_size is None:
        tile_size = max(aof.shape)
    st, pts = structure_tensor_3d(aof, sigma, rho,
                                        aof_threshold=aof_threshold,
                                        tile_size=tile_size)

    e_vals, e_vecs, fa = eig_special_3d(st, points=pts, shape=aof.shape,
                                        return_sparse=return_sparse,
                                        batch_size=batch_size)
    return e_vals, e_vecs, fa, pts


def _block_aof(sdf_np, offsets, hx=1.0, hy=1.0, hz=1.0):
    """
    Computes the Average Outward Flux (AOF) map for a block of 3D Signed Distance Field.
    Args:
        sdf_np (ndarray): 
            A 3D/2D NumPy array (Z, Y, X) representing the Signed Distance Field 
            or intensity volume.
    Returns:
        aof (ndarray): 
            A 3D/2D NumPy array of the same shape as 'sdf_np'.
    """
    wp.init()

    sdf_gpu = wp.from_numpy(sdf_np, dtype=float)
    aof_gpu = wp.zeros_like(sdf_gpu)

    if len(sdf_np.shape) == 3:
        wp.launch(
            kernel=aof_kernel_3d,
            dim=sdf_gpu.shape,
            inputs=[sdf_gpu, aof_gpu, offsets,
                    hx,hy,hz,
                    ]
        )
    elif len(sdf_np.shape) == 2:
        wp.launch(
            kernel=aof_kernel_2d,
            dim=sdf_gpu.shape,
            inputs=[sdf_gpu, aof_gpu, offsets,
                    hx,hy,]
        )
    else:
        print(f"Unsported dimension {len(sdf_np.shape)}. Only 2,3 is supported")

    wp.synchronize()


    return aof_gpu.numpy()   



def quick_aof(sdf_np:np.ndarray, hx = 1.0, hy=1.0, hz=1.0, tile_size=128, halo=2)->np.ndarray:
    """
    Computes the Average Outward Flux (AOF) map for 3D Signed Distance volume.  

    Args:
        sdf_np (ndarray): 
            A 3D/2D NumPy array (Z, Y, X) representing the Signed Distance Field 
            or intensity volume.
    Returns:
        aof (ndarray): 
            A 3D/2D NumPy array of the same shape as 'sdf_np'.
    """
    
    depth, height, width = sdf_np.shape
    aof_full = np.zeros_like(sdf_np)

    OFFSET_DATA = sample_sphere_points(128)
    offsets = wp.array(OFFSET_DATA, dtype=wp.vec3)

    # Iterate through the volume in chunks
    for z in range(0, depth, tile_size):
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                
                # Actual Core bounds we want to fill
                z_end, y_end, x_end = min(z + tile_size, depth), min(y + tile_size, height), min(x + tile_size, width)

                # Pad with halo to be discarded later if possible...
                z0_h, z1_h = max(z - halo, 0), min(z_end + halo, depth)
                y0_h, y1_h = max(y - halo, 0), min(y_end + halo, height)
                x0_h, x1_h = max(x - halo, 0), min(x_end + halo, width)

                tile_sdf = sdf_np[z0_h:z1_h, y0_h:y1_h, x0_h:x1_h]

                tile_aof_processed = _block_aof(tile_sdf, offsets, hx=hx,hy=hy,hz=hz)

                dz0, dy0, dx0 = z - z0_h, y - y0_h, x - x0_h
                dz1, dy1, dx1 = dz0 + (z_end - z), dy0 + (y_end - y), dx0 + (x_end - x)

                # Paste "Core" region into the final result
                aof_full[z:z_end, y:y_end, x:x_end] = tile_aof_processed[dz0:dz1, dy0:dy1, dx0:dx1]

    return aof_full



@wp.kernel
def threshold_local_max(
    input: wp.array(dtype=float, ndim=3),
    output: wp.array(dtype=float, ndim=3),
    N: int,
    fraction: float,
):
    i, j, k = wp.tid()

    nx = input.shape[0]
    ny = input.shape[1]
    nz = input.shape[2]

    half_n = N // 2

    # Compute local max in k x k x k neighborhood
    local_max = float(-1e5)

    for di in range(-half_n, half_n + 1):
        for dj in range(-half_n, half_n + 1):
            for dk in range(-half_n, half_n + 1):

                ni = i + di
                nj = j + dj
                nl = k + dk

                # boundary check
                if (ni >= 0 and ni < nx and
                    nj >= 0 and nj < ny and
                    nl >= 0 and nl < nz):

                    val = input[ni, nj, nl]
                    if val > local_max:
                        local_max = val

    val = input[i, j, k]

    # Apply threshold
    if val < fraction * local_max:
        output[i, j, k] = 0.0
    else:
        output[i, j, k] = val


def nonmax_supp(vol: np.ndarray,*,window_size=3, fraction=0.7)->np.ndarray:
    """Perform local non-maximum suppression on a 3D volume.

    This function applies local maximum thresholding to a volumetric
    array. Each voxel is compared against the maximum value within a
    local neighborhood, and values below a specified fraction of 
    the local maximum are suppressed.
    BUG: Ideally one would want to do this only in the direction of
    local gradients

    Args:
        vol (np.ndarray):
            Input 3D volume array.

        window_size (int, optional):
            Size of the cubic local neighborhood used to compute local
            maxima. Defaults to ``3``.

        fraction (float, optional):
            Relative threshold factor in the range ``[0, 1]``.
            Defaults to ``0.7``.

            Voxels satisfying:
            ``value < fraction * local_max``
            are suppressed to zero.

    Returns:
        np.ndarray:
            Thresholded 3D volume with suppressed low-response voxels.

    Example:
        >>> filtered = nonmax_supp(
        ...     volume,
        ...     window_size=5,
        ...     fraction=0.8
        ... )

    """
    vol_gpu = wp.from_numpy(vol, dtype=float)
    out_gpu = wp.zeros_like(vol_gpu)

    wp.launch(
        threshold_local_max,
        dim=vol_gpu.shape,
        inputs=[vol_gpu, out_gpu, window_size, fraction]
    )
    
    wp.synchronize()

    return out_gpu.numpy()


def compute_aof(sdf_np: np.ndarray, hx = 1.0, hy=1.0, hz=1.0, tile_size=128, halo=2) -> np.ndarray:
    """Compute Average Outward Flux (AOF) from a Signed Distance Field.

    This function computes the Average Outward Flux (AOF) volume from a
    Signed Distance Field (SDF) and applies local non-maximum suppression
    to enhance ridge-like structures and suppress weak responses.

    Args:
        sdf_np (np.ndarray):
            Input Signed Distance Field (SDF) volume as a 3D NumPy array.

        hx (float, optional):
            Grid spacing along the x-axis. Defaults to ``1.0``.

        hy (float, optional):
            Grid spacing along the y-axis. Defaults to ``1.0``.

        hz (float, optional):
            Grid spacing along the z-axis. Defaults to ``1.0``.

        tile_size (int, optional):
            Tile size used during tiled AOF computation. Defaults to ``128``.

            Larger tile sizes may improve performance on larger GPUs but
            increase memory usage.

        halo (int, optional):
            Halo padding size used for neighborhood access during tiled
            computation. Defaults to ``2``.

    Returns:
        np.ndarray:
            A thresholded 3D AOF volume after non-maximum suppression.

    Example:
        >>> aof_vol = compute_aof(sdf)

    Typical Use Cases:
        - Medial axis approximation
        - Skeleton extraction
    """
    
    _aof = quick_aof(sdf_np, hx = 1.0, hy=1.0, hz=1.0, tile_size=128, halo=2)
    return nonmax_supp(_aof)
