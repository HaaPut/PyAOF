import numpy as np
import pytest
from pyaof import compute_aof

def test_aof_output_shape():
    # Create a dummy 3D SDF (10x10x10)
    sdf = np.random.rand(10, 10, 10).astype(np.float32)
    
    # Run the computation
    aof = compute_aof(sdf, hx=1.0, hy=1.0, hz=1.0)
    
    # Assert the output matches the input volume shape
    assert aof.shape == (10, 10, 10)
    assert aof.dtype == np.float32
