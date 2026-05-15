import warp as wp
import numpy as np
import matplotlib.pyplot as plt


INV3 = 1.0 / 3.0
TWO_PI_OVER3 = 2.0 * 3.14159265358979 / 3.0
SMALL = 1e-12

@wp.kernel
def outer_product_kernel(
    volume: wp.array3d(dtype=float),
    output_component: wp.array3d(dtype=float),
    comp_index: int, # 0:xx, 1:yy, 2:zz, 3:xy, 4:xz, 5:yz
    hx: float, hy: float, hz: float,
):
    i, j, k = wp.tid()
    nx, ny, nz = volume.shape[0], volume.shape[1], volume.shape[2]
    # Boundary check for central difference
    if i < 1 or i >= nx-1 or j < 1 or j >= ny-1 or k < 1 or k >= nz-1:
        return

    # Compute gradients locally
    vx = (volume[i+1, j, k] - volume[i-1, j, k]) / (2.0 * hx)
    vy = (volume[i, j+1, k] - volume[i, j-1, k]) / (2.0 * hy)
    vz = (volume[i, j, k+1] - volume[i, j, k-1]) / (2.0 * hz)

    # Compute Outer product component
    val = 0.0
    if comp_index == 0: val = vx * vx
    elif comp_index == 1: val = vy * vy
    elif comp_index == 2: val = vz * vz
    elif comp_index == 3: val = vx * vy
    elif comp_index == 4: val = vx * vz
    elif comp_index == 5: val = vy * vz
    
    output_component[i, j, k] = val


def get_gaussian_weights(sigma, truncate=4.0):
    radius = int(truncate * sigma + 0.5)
    x = np.linspace(-radius, radius, 2 * radius + 1)
    phi_x = np.exp(-0.5 * (x / sigma)**2)
    phi_x /= phi_x.sum() # Normalize so sum of weights is 1.0
    return wp.from_numpy(phi_x.astype(np.float32)), radius


@wp.kernel
def blur_x_weighted_kernel(src: wp.array3d(dtype=float),
                           dst: wp.array3d(dtype=float),
                           weights: wp.array1d(dtype=float),
                           radius: int):
    i, j, k = wp.tid()
    nx, ny, nz = src.shape[0], src.shape[1], src.shape[2]
    
    res = float(0.0)
    for x in range(-radius, radius + 1):
        #'entending' value normal to boundary
        ix = wp.clamp(i + x, 0, nx - 1) 
        res += src[ix, j, k] * weights[x + radius]
    dst[i, j, k] = res

@wp.kernel
def blur_y_weighted_kernel(src: wp.array3d(dtype=float),
                           dst: wp.array3d(dtype=float),
                           weights: wp.array1d(dtype=float),
                           radius: int):
    i, j, k = wp.tid()
    nx, ny, nz = src.shape[0], src.shape[1], src.shape[2]
    
    res = float(0.0)
    for y in range(-radius, radius + 1):
        iy = wp.clamp(j + y, 0, ny - 1)
        res += src[i, iy, k] * weights[y + radius]
    dst[i, j, k] = res

@wp.kernel
def blur_z_weighted_kernel(src: wp.array3d(dtype=float),
                           dst: wp.array3d(dtype=float),
                           weights: wp.array1d(dtype=float),
                           radius: int):
    i, j, k = wp.tid()
    nx, ny, nz = src.shape[0], src.shape[1], src.shape[2]
    
    res = float(0.0)
    for z in range(-radius, radius + 1):
        iz = wp.clamp(k + z, 0, nz - 1)
        res += src[i, j, iz] * weights[z + radius]
    dst[i, j, k] = res


def apply_gaussian_3d(
    src_buffer: wp.array3d, 
    tmp_buffer: wp.array3d, 
    weights: wp.array1d, 
    radius: int
):
    """
    Applies a 3D Gaussian blur using three 1D separable passes.
    CAUTION: The src_buffer is over-written!
    Note: The final result will be stored in tmp_buffer.
    """
    shape = src_buffer.shape

    # X-Blur (src_buffer -> tmp_buffer)
    wp.launch(
        kernel=blur_x_weighted_kernel,
        dim=shape, 
        inputs=[src_buffer, tmp_buffer, weights, radius]
    )
    
    # Y-Blur (tmp_buffer -> src_buffer)
    wp.launch(
        kernel=blur_y_weighted_kernel, 
        dim=shape, 
        inputs=[tmp_buffer, src_buffer, weights, radius]
    )
    
    # Z-Blur (src_buffer -> tmp_buffer)
    wp.launch(
        kernel=blur_z_weighted_kernel, 
        dim=shape, 
        inputs=[src_buffer, tmp_buffer, weights, radius]
    )



@wp.func
def eigenvalues_sym33(A: wp.mat33):

    m = (A[0,0] + A[1,1] + A[2,2]) * INV3

    a00 = A[0,0] - m
    a11 = A[1,1] - m
    a22 = A[2,2] - m

    a01 = A[0,1]
    a02 = A[0,2]
    a12 = A[1,2]

    p = (a00*a00 + a11*a11 + a22*a22 +
         2.0*(a01*a01 + a02*a02 + a12*a12)) * INV3

    p = wp.sqrt(p)

    if p < SMALL:
        return wp.vec3(m, m, m)

    invp = 1.0 / p

    B00 = a00 * invp
    B11 = a11 * invp
    B22 = a22 * invp
    B01 = a01 * invp
    B02 = a02 * invp
    B12 = a12 * invp

    detB = (
        B00*B11*B22
        + 2.0*B01*B02*B12
        - B00*B12*B12
        - B11*B02*B02
        - B22*B01*B01
    )

    r = detB * 0.5
    r = wp.clamp(r, -1.0, 1.0)
    # r \in [-1,1] 
    phi = wp.acos(r) * INV3
    # phi \in [0, pi/3] => cos(phi) \in [0.5, 1]
    # => phi + 2pi/3 \in => cos(phi + TWO_PI_OVER3) \in [-1, -0.5]
    # p is positive.. ~ sqrt of sum of squares of entries

    eig_max = m + 2.0*p*wp.cos(phi)
    eig_min = m + 2.0*p*wp.cos(phi + TWO_PI_OVER3)
    eig_mid = 3.0*m - eig_min - eig_max

    return wp.vec3(eig_min, eig_mid, eig_max)



@wp.func
def eigenvector_from_value(A: wp.mat33, lam: float):

    M = A - wp.mat33(
        lam,0.0,0.0,
        0.0,lam,0.0,
        0.0,0.0,lam
    )

    r0 = wp.vec3(M[0,0], M[0,1], M[0,2])
    r1 = wp.vec3(M[1,0], M[1,1], M[1,2])
    r2 = wp.vec3(M[2,0], M[2,1], M[2,2])

    c0 = wp.cross(r0, r1)
    c1 = wp.cross(r0, r2)
    c2 = wp.cross(r1, r2)

    d0 = wp.dot(c0,c0)
    d1 = wp.dot(c1,c1)
    d2 = wp.dot(c2,c2)

    v = c0
    if d1 > d0:
        v = c1
        d0 = d1
    if d2 > d0:
        v = c2

    if d0 < SMALL:
        return wp.vec3(0.0,0.0,0.0)

    return wp.normalize(v)


@wp.func
def eigen_sym33(A: wp.mat33):

    d = eigenvalues_sym33(A)

    # add filter for fa here...?

    v0 = eigenvector_from_value(A, d[0])
    v1 = eigenvector_from_value(A, d[1])
    v2 = eigenvector_from_value(A, d[2])

    if wp.length(v0) < SMALL:
        return 0, d, wp.mat33()

    if wp.length(v1) < SMALL:
        return 0, d, wp.mat33()

    if wp.length(v2) < SMALL:
        return 0, d, wp.mat33()
    
    v2 = wp.normalize(v2)
    v0 = wp.normalize(v0 - v2 * wp.dot(v0, v2))
    v1 = wp.cross(v2, v0)

    V = wp.mat33(
        v0[0], v1[0], v2[0],
        v0[1], v1[1], v2[1],
        v0[2], v1[2], v2[2]
    )

    return 1, d, V

@wp.func
def fractional_anisotropy(eigs: wp.vec3) -> float:
    """
    Compute fractional anisotropy (FA) from a vector of 3 eigenvalues
    """
    lam1, lam2, lam3 = eigs[0], eigs[1], eigs[2]
    mean = (lam1 + lam2 + lam3) / 3.0

    num = wp.sqrt(
        (lam1 - mean)*(lam1 - mean) +
        (lam2 - mean)*(lam2 - mean) +
        (lam3 - mean)*(lam3 - mean)
    )

    denom = wp.length(eigs) + SMALL

    FA = wp.sqrt(1.5) * num / denom
    return FA


@wp.func
def jacobi_solve(A: wp.mat33):
    # Initialize eigenvalues as diagonal elements
    e_val = wp.vec3(A[0,0], A[1,1], A[2,2])
    
    # Off-diagonal elements
    m01 = A[0,1]
    m02 = A[0,2]
    m12 = A[1,2]

    # Initialize eigenvectors as Identity matrix
    v0 = wp.vec3(1.0, 0.0, 0.0)
    v1 = wp.vec3(0.0, 1.0, 0.0)
    v2 = wp.vec3(0.0, 0.0, 1.0)

    # 8 iterations is usually enough for float32 convergence
    for i in range(10):
        # 1. Find the largest off-diagonal element (p, q)
        # Check the 3 unique off-diagonal slots in a 3x3 symmetric matrix
        abs01 = wp.abs(m01)
        abs02 = wp.abs(m02)
        abs12 = wp.abs(m12)
        
        # Convergence check
        if abs01 < 1e-9 and abs02 < 1e-9 and abs12 < 1e-9:
            break

        # Assign p and q based on the maximum element
        p = 0
        q = 1
        if abs02 > abs01 and abs02 > abs12:
            p = 0
            q = 2
        elif abs12 > abs01:
            p = 1
            q = 2

        # Calculate the rotation angle (Schur decomposition)
        # We need M[p,p], M[q,q], and M[p,q]
        m_pp = 0.0
        m_qq = 0.0
        m_pq = 0.0
        
        if p == 0 and q == 1:
            m_pp, m_qq, m_pq = e_val[0], e_val[1], m01
        elif p == 0 and q == 2:
            m_pp, m_qq, m_pq = e_val[0], e_val[2], m02
        else: # p=1, q=2
            m_pp, m_qq, m_pq = e_val[1], e_val[2], m12

        phi = 0.5 * wp.atan2(2.0 * m_pq, m_qq - m_pp)
        c = wp.cos(phi)
        s = wp.sin(phi)

        #  Update Eigenvalues (Diagonal elements)
        new_pp = c*c*m_pp - 2.0*s*c*m_pq + s*s*m_qq
        new_qq = s*s*m_pp + 2.0*s*c*m_pq + c*c*m_qq
        
        if p == 0 and q == 1:
            e_val = wp.vec3(new_pp, new_qq, e_val[2])
            # Update remaining off-diagonals
            new_m02 = c*m02 - s*m12
            new_m12 = s*m02 + c*m12
            m01, m02, m12 = 0.0, new_m02, new_m12
        elif p == 0 and q == 2:
            e_val = wp.vec3(new_pp, e_val[1], new_qq)
            new_m01 = c*m01 - s*m12 # note: index logic changes
            new_m12 = s*m01 + c*m12
            m02, m01, m12 = 0.0, new_m01, new_m12
        else: # p=1, q=2
            e_val = wp.vec3(e_val[0], new_pp, new_qq)
            new_m01 = c*m01 - s*m02
            new_m02 = s*m01 + c*m02
            m12, m01, m02 = 0.0, new_m01, new_m02

        # We rotate the vectors v_p and v_q
        old_vp = wp.vec3(0.0)
        old_vq = wp.vec3(0.0)
        
        if p == 0: old_vp = v0
        elif p == 1: old_vp = v1
        
        if q == 1: old_vq = v1
        else: old_vq = v2

        new_vp = c*old_vp - s*old_vq
        new_vq = s*old_vp + c*old_vq

        if p == 0: v0 = new_vp
        elif p == 1: v1 = new_vp
        
        if q == 1: v1 = new_vq
        else: v2 = new_vq

    if e_val[0] > e_val[1]:
        # Swap values
        tmp_e = e_val[0]; e_val = wp.vec3(e_val[1], tmp_e, e_val[2])
        # Swap vectors
        tmp_v = v0; v0 = v1; v1 = tmp_v
        
    if e_val[1] > e_val[2]:
        # Swap values
        tmp_e = e_val[1]; e_val = wp.vec3(e_val[0], e_val[2], tmp_e)
        # Swap vectors
        tmp_v = v1; v1 = v2; v2 = tmp_v

    # Pass 2 (Final check for the first two)
    if e_val[0] > e_val[1]:
        # Swap values
        tmp_e = e_val[0]; e_val = wp.vec3(e_val[1], tmp_e, e_val[2])
        # Swap vectors
        tmp_v = v0; v0 = v1; v1 = tmp_v

    V = wp.mat33(v0[0], v1[0], v2[0],
                 v0[1], v1[1], v2[1],
                 v0[2], v1[2], v2[2])
                 
    return e_val, V

@wp.kernel
def eigen_decomposition(
    A: wp.array(dtype=wp.mat33),
    eigvals: wp.array(dtype=wp.vec3),
    eigvecs: wp.array(dtype=wp.mat33),
    fa:wp.array(dtype=float),
):

    i = wp.tid()

    success, d, V = eigen_sym33(A[i])

    if success == 0:
        d, V = jacobi_solve(A[i])

    eigvals[i] = d
    eigvecs[i] = V
    fa[i] = fractional_anisotropy(d)


def _prepare_st_matrix(sparse_st):
    """
    Converts (6, N) sparse tensor components into (N, 3, 3) symmetric matrices.
    Input sparse_st order assumed: [xx, yy, zz, xy, xz, yz]
    """
    N = sparse_st.shape[1]
    
    # Initialize an (N, 3, 3) float32 array
    ST_matrix = np.empty((N, 3, 3), dtype=np.float32)
    
    # Fill Diagonal Components
    ST_matrix[:, 0, 0] = sparse_st[0] # xx
    ST_matrix[:, 1, 1] = sparse_st[1] # yy
    ST_matrix[:, 2, 2] = sparse_st[2] # zz
    
    # Fill Off-Diagonal Components (Symmetric)
    # xy
    ST_matrix[:, 0, 1] = sparse_st[3]
    ST_matrix[:, 1, 0] = sparse_st[3]
    
    # xz
    ST_matrix[:, 0, 2] = sparse_st[4]
    ST_matrix[:, 2, 0] = sparse_st[4]
    
    # yz
    ST_matrix[:, 1, 2] = sparse_st[5]
    ST_matrix[:, 2, 1] = sparse_st[5]
    
    return ST_matrix


def block_structure_tensor_3d(volume_np, sigma=1.0, rho=1.0, aof_threshold=None):
    """
    Computes a GPU-accelerated 3D Structure Tensor for a volumetric scalar field.

    Args:
        volume_np (ndarray): 
            The input 3D scalar volume of shape (X, Y, Z).
        sigma (float): 
            The differentiation scale. Controls the Gaussian smoothing applied 
            to the volume before calculating gradients. High values suppress noise.
        rho (float): 
            The integration scale. Controls the Gaussian smoothing applied 
            to the outer product of the gradients. High values provide a 
            more stable orientation estimate by averaging over a larger neighborhood.
        aof_threshold (float, optional): 
            - If None: Computes the tensor for every voxel (Dense Mode).
            - If float: Identifies voxels where intensity > aof_threshold and 
              performs computations only at those 'sparse' locations to save memory.

    Returns:
        S (ndarray): 
            The 6 unique components of the symmetric 3D structure tensor:
            [Sxx, Syy, Szz, Sxy, Sxz, Syz].
            - Shape (6, Z, Y, X) in Dense Mode.
            - Shape (6, N) in Sparse Mode, where N is the number of valid points.
        points (tuple of ndarrays or None): 
            The (z, y, x) coordinate arrays for the computed points. 
            Only returned if aof_threshold is not None.

    """
    #wp.init()
    device = wp.get_device("cuda")
    
    # 1. Transfer input (Only 1 input array)
    vol_gpu = wp.from_numpy(volume_np, dtype=wp.float32, device=device)
    shape = vol_gpu.shape
    
    rho_weights, rho_radius = get_gaussian_weights(rho)
    sigma_weights, sigma_radius = get_gaussian_weights(sigma)

    # 2. Allocate Output Of Shape (6, Z, Y, X) or (6, N)
    if aof_threshold is None:
        dense_S = np.empty((6,) + shape, dtype=np.float32)
    else:
        points = np.nonzero(volume_np > aof_threshold)
        sparse_S = np.empty((6, len(points[0])))
        temp_S = np.empty_like(shape)

    # Temporary buffer for smoothing
    temp_buffer = wp.zeros_like(vol_gpu)
    comp_buffer = wp.zeros_like(vol_gpu)
    
    apply_gaussian_3d(vol_gpu,
                      comp_buffer,
                      sigma_weights,
                      sigma_radius)
    
    wp.copy(dest=vol_gpu, src=comp_buffer)
    for c in range(6):
 
        # Compute product and store in comp_buffer
        wp.launch(
            kernel=outer_product_kernel,
            dim=shape,
            inputs=[vol_gpu, comp_buffer, c, 1.0, 1.0, 1.0],
            device=device
        )
        
        # Apply Gaussian smoothing (Rho)
        # Need vol_gpu to stay fixed for derivatives during this loop
        # so don't pass the buffer to apply_gaussian_3d
        apply_gaussian_3d(comp_buffer,
                          temp_buffer,
                          rho_weights,
                          rho_radius)

        # Copy back to CPU and reuse buffers for next component
        if aof_threshold is None:
            dense_S[c] = temp_buffer.numpy()
        else:
            temp_S = temp_buffer.numpy()
            # Only copy relevant points in sparse case
            sparse_S[c] = temp_S[points]

    wp.synchronize()   
    if aof_threshold is None:
        return dense_S, None
    return sparse_S, points


def structure_tensor_3d(volume_full, sigma=1.0, rho=1.0, aof_threshold=None,*, tile_size=128):
    """
    Tiled structure tensor computation..
    Notes:
    ------
    * Adjust tile size according to GPU memory available
    * See structure_tensor_3d of documentation
    """
    # 1. Calculate required halo
    # Gaussian filters need 'radius' voxels. Since we do sigma then rho, 
    # we need enough padding for both.
    _, sigma_radius = get_gaussian_weights(sigma)
    _, rho_radius = get_gaussian_weights(rho)
    halo = sigma_radius + rho_radius
    
    shape = volume_full.shape
    
    # Containers for results
    all_sparse_S = []
    all_points = [[], [], []] # Z, Y, X
    
    # Pre-allocate dense output if needed (Warning: this can be huge)
    dense_S = None
    if aof_threshold is None:
        dense_S = np.zeros((6,) + shape, dtype=np.float32)

    # 2. Iterate through Tiles
    for z in range(0, shape[0], tile_size):
        for y in range(0, shape[1], tile_size):
            for x in range(0, shape[2], tile_size):
                
                # Core bounds
                z_end, y_end, x_end = min(z+tile_size, shape[0]), min(y+tile_size, shape[1]), min(x+tile_size, shape[2])
                
                # Halo bounds (clamped to volume limits)
                z0_h, z1_h = max(z - halo, 0), min(z_end + halo, shape[0])
                y0_h, y1_h = max(y - halo, 0), min(y_end + halo, shape[1])
                x0_h, x1_h = max(x - halo, 0), min(x_end + halo, shape[2])
                
                # Slice tile
                tile_np = volume_full[z0_h:z1_h, y0_h:y1_h, x0_h:x1_h]
                
                # pass aof_threshold=None here to get the full tile processed,
                # then handle the thresholding/cropping here for sparse mode.
                S_tile, _ = block_structure_tensor_3d(tile_np, sigma, rho, aof_threshold=None)
                
                dz0, dz1 = z - z0_h, (z - z0_h) + (z_end - z)
                dy0, dy1 = y - y0_h, (y - y0_h) + (y_end - y)
                dx0, dx1 = x - x0_h, (x - x0_h) + (x_end - x)
                
                S_core = S_tile[:, dz0:dz1, dy0:dy1, dx0:dx1]
                
                if aof_threshold is None:
                    # Dense mode: Stitch into global array
                    dense_S[:, z:z_end, y:y_end, x:x_end] = S_core
                else:
                    # Sparse mode: Check threshold on the original core volume
                    core_vol = volume_full[z:z_end, y:y_end, x:x_end]
                    local_pts = np.nonzero(core_vol > aof_threshold)
                    
                    if len(local_pts[0]) > 0:
                        # Store tensor components for these points
                        # S_core is (6, dZ, dY, dX), local_pts indexes the last 3 dims
                        all_sparse_S.append(S_core[:, local_pts[0], local_pts[1], local_pts[2]])
                        
                        # Store Global Coordinates
                        all_points[0].append(local_pts[0] + z)
                        all_points[1].append(local_pts[1] + y)
                        all_points[2].append(local_pts[2] + x)

    if aof_threshold is None:
        return dense_S, None
    else:
        if not all_sparse_S:
            return np.array([]), (np.array([]), np.array([]), np.array([]))
            
        final_S = np.concatenate(all_sparse_S, axis=1)
        final_points = (np.concatenate(all_points[0]), 
                        np.concatenate(all_points[1]), 
                        np.concatenate(all_points[2]))
        return final_S, final_points



def sparse_eigen_decomposition(sparse_st):
    """
    Performs GPU-accelerated eigen-decomposition on a sparse set of symmetric 
    structure tensors.

    This function converts a compressed (6, N) representation of symmetric tensors 
    into full 3x3 matrices, uploads them to the GPU, and computes sorted 
    eigenvalues, eigenvectors, and anisotropy measures in parallel.

    Args:
        sparse_st (ndarray): 
            A (6, N) NumPy array where each column contains the unique components 
            of a symmetric tensor in the order: [Sxx, Syy, Szz, Sxy, Sxz, Syz].
            N represents the number of sparse points/voxels.

    Returns:
        evals (ndarray): 
            The computed eigenvalues sorted in ascending order (l1 < l2 < l3).
            Shape: (N, 3). 
        evecs (ndarray): 
            The corresponding eigenvectors stored as 3x3 matrices.
            Shape: (N, 3, 3). 
        fa (ndarray): 
            The Fractional Anisotropy for each point.
            Shape: (N,). 

    """
    #wp.init()
    ST_matrix = _prepare_st_matrix(sparse_st)
    N = len(ST_matrix)
    #print(f"Computing Eigen decompostion of {N} STs")

    device = wp.get_device("cuda")
    wp_st = wp.from_numpy(ST_matrix, dtype=wp.mat33, device=device)
    wp_evals = wp.zeros(N, dtype=wp.vec3, device=device)
    wp_evecs = wp.zeros(N, dtype=wp.mat33, device=device)
    wp_fa = wp.zeros(N, dtype=float, device=device)
    
    # 3. Compute
    wp.launch(
        kernel=eigen_decomposition,
        dim=N,
        inputs=[wp_st, wp_evals, wp_evecs,wp_fa]
    )
    
    evals, evecs, fa = wp_evals.numpy(), wp_evecs.numpy(), wp_fa.numpy()
    wp.synchronize()
    return evals, evecs, fa



def tiled_sparse_eigen_decomposition(sparse_st, batch_size=1e6):
    """
    Batched  of eigen-decomposition 
    Args:
        sparse_st (ndarray): (6, N) array of symmetric tensor components.
        batch_size (int): Number of points to process per GPU upload.
    """
    num_points = sparse_st.shape[1]
    
    # Initialize output lists
    all_evals = []
    all_evecs = []
    all_fa = []

    # Process in batches
    for i in range(0, num_points, batch_size):
        end_idx = min(i + batch_size, num_points)
        batch_st = sparse_st[:, i:end_idx]
        
        batch_evals, batch_evecs, batch_fa = sparse_eigen_decomposition(batch_st)
        
        all_evals.append(batch_evals)
        all_evecs.append(batch_evecs)
        all_fa.append(batch_fa)
        
    final_evals = np.concatenate(all_evals, axis=0)
    final_evecs = np.concatenate(all_evecs, axis=0)
    final_fa = np.concatenate(all_fa, axis=0)

    return final_evals, final_evecs, final_fa


def eig_special_3d(st, points=None, shape=None, return_sparse=False, *, batch_size=None):
    """
    Wrapper for Eigen-Decomposition, handling both sparse and dense structure tensor inputs.
    X, Y, Z
    Args:
        st (ndarray): 
            Input structure tensor data.
            - If shape is (6, N): Treated as Sparse (requires 'points' and 'shape').
            - If shape is (6, X, Y, Z): Treated as Dense.
        points (tuple of ndarrays, optional): 
            The (x,y,z) indices of the sparse points. Required if 'st' is (6, N).
        shape (tuple, optional): 
            The original (X, Y, Z) spatial dimensions of the volume. 
            Required for mapping sparse results back to a dense grid.
        batch_size (int, optional):
            batch size for computing eigen decompostion
    Returns:
        evals (ndarray): 
            Eigenvalues (l1, l2, l3) sorted ascending.
            Shape: (3, X, Y, Z). Uncomputed voxels are filled with NaN.
        evecs (ndarray): 
            Eigenvector matrices where each voxel contains a 3x3 Matrix.
            Shape: (3, 3, X, Y, Z).
        fa (ndarray): 
            Fractional Anisotropy.
            Shape: (X, Y, Z).

    """
    if len(st.shape) == 2:
        #sparse points case
        sparse = True
        if points is None or shape is None:
            raise ValueError("Sparse 'st' requires 'points' indices and 'shape' of the volume.")
        if batch_size is None:
            batch_size = st.shape[1]
    else:
        sparse = False
        if shape is None:
            shape = st.shape[1:]  # (X, Y, Z)
        if batch_size is None:
            batch_size = np.prod(shape)
    if sparse:
        evals, evecs, fa = tiled_sparse_eigen_decomposition(st,batch_size=batch_size)
        if return_sparse:
            return evals, evecs, fa
        dense_vals = np.full((3,) + shape, np.nan)
        dense_vecs = np.full((3,3) + shape, np.nan) 
        dense_fa = np.full(shape, np.nan)
       
        dense_vals[:, points[0], points[1], points[2]] = evals.T
        #dense_fa[points[0], points[1], points[2]] = fa
        dense_fa[points] = fa

        # evecs_flat is (N, 3, 3) -> transpose to (3, 3, N)
        dense_vecs[:, :, points[0], points[1], points[2]] = np.moveaxis(evecs,(0,1,2),(2,0,1))
        #breakpoint()
        return dense_vals, dense_vecs, dense_fa
    else:
        #st of shape 6, X, Y, Z 
        evals, evecs, fa = tiled_sparse_eigen_decomposition(st.reshape(6,-1),batch_size=batch_size)
        #evals transform (N, 3) -> (X, Y, Z, 3) -> transpose to (3, X, Y, Z)
        evals = evals.reshape(shape + (3,)).transpose(3, 0, 1, 2)
        #evecs transform (N, 3, 3) -> (X, Y, Z, 3, 3) -> transpose to (3, 3, X, Y, Z)
        evecs = evecs.reshape(shape + (3, 3)).transpose(3, 4, 0, 1, 2)
        fa = fa.reshape(shape)
        return evals, evecs, fa
