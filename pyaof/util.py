import os
import math
import numpy as np
ps = None
try:
    import polyscope as _ps
    ps = _ps
except ImportError:
    ps = None


def render_360_rotation(
    target=np.array([0.0, 0.0, 0.0]),
    radius=5.0,
    elevation=2.0,
    num_frames=60,
    output_folder="renders",
    up_axis="Z"
):
    """
    Renders a 360-degree camera rotation around a target point in Polyscope.
    Safely exits if Polyscope is not installed or initialized.
    
    Args:
        target (np.ndarray, optional): The 3D coordinate [X, Y, Z] the camera 
            will focus on. Defaults to np.array([0.0, 0.0, 0.0]).
        radius (float, optional): The horizontal distance from the camera to 
            the target point. Defaults to 5.0.
        elevation (float, optional): The vertical height offset of the camera 
            relative to the target's height. Defaults to 2.0.
        num_frames (int, optional): The total number of subdivisions along 
            the circular camera path. Defaults to 60.
        up_axis (str, optional): The global vertical vector convention of the 
            dataset. Must be either 'X','Y' or 'Z' (case-insensitive). Defaults to "Z".

    Raises:
        ValueError: If `up_axis` is passed as anything other than 'Y' or 'Z'.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> import polyscope as ps
        >>> ps.init()
        >>> # ... register your point clouds or meshes here ...
        >>> render_360_rotation(target=np.array([0, 1, 0]), radius=3.5, up_axis="Y")
    """
    # 1. Fallback guard: check if ps is None or has not been initialized yet
    if ps is None or not ps.is_initialized():
        print("[Warning] Polyscope is not available or initialized. Skipping render function.")
        return

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    print(f"Starting 360 render ({num_frames} frames)...")
    
    for i in range(num_frames):
        angle = (i / num_frames) * 2 * math.pi
        coord_1 = radius * math.cos(angle)
        coord_2 = radius * math.sin(angle)
        
        if up_axis.upper() == "Z":
            up_vector = np.array([0.0, 0.0, 1.0])
            eye_x = target[0] + coord_1
            eye_y = target[1] + coord_2
            eye_z = target[2] + elevation
        elif up_axis.upper() == "Y":
            up_vector = np.array([0.0, 1.0, 0.0])
            eye_x = target[0] + coord_1
            eye_y = target[1] + elevation
            eye_z = target[2] + coord_2
        elif up_axis.upper() == "X":
            up_vector = np.array([1.0, 0.0, 0.0])
            eye_x = target[0] + elevation
            eye_y = target[1] + coord_1
            eye_z = target[2] + coord_2
        else:
            raise ValueError("up_axis must be either 'X', 'Y' or 'Z'")
            
        camera_pos = np.array([eye_x, eye_y, eye_z])
        
        # Execute the camera shift and update UI environment using the safe global ps
        ps.look_at_dir(camera_pos, target, up_vector)
        ps.frame_tick()
        
        filename = os.path.join(output_folder, f"frame_{i:04d}.png")
        ps.screenshot(filename)
        
    print(f"Render complete. Frames saved to: {output_folder}/")