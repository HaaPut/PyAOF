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

def test_aof_flat():
    sdf = np.zeros((5, 5, 5), dtype=np.float32)
    aof = compute_aof(sdf, 1.0, 1.0, 1.0)
    
    # Example assertion: A flat field should result in zero flux
    assert np.allclose(aof, 0.0, atol=1e-4)

def test_aof_range():

    volume = np.random.rand(32, 32, 32)
    
    aof_map = compute_aof(volume)
    
    # Check range
    assert np.max(aof_map) <= 1.0, f"Max AOF {np.max(aof_map)} exceeds 1.0"
    assert np.min(aof_map) >= -1.0, f"Min AOF {np.min(aof_map)} is below -1.0"