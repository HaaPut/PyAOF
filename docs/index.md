# Home
<p align="center">
  <video 
    width="800" 
    autoplay 
    loop 
    muted 
    playsinline 
    poster="assets/aof-spokes.png">
    <source src="assets/aof-spokes.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

[![PyPI](https://img.shields.io/pypi/v/pyaof.svg)](https://pypi.org/project/pyaof/)
[![Python](https://img.shields.io/pypi/pyversions/pyaof.svg)](https://pypi.org/project/pyaof/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


PyAOF is a high-performance Python library for computing the **Average Outward Flux (AOF)**  on volumetric meshes. 

By leveraging **NVIDIA Warp**, PyAOF provides GPU-accelerated kernels that make skeletal feature extraction and medial axis computations significantly faster than traditional CPU-based implementations.

It provides efficient utilities for:

- Compute **Average Outward Flux (AOF)** volumes
- Efficient tiled computation for large volumes
- Optional 3D visualization with **Polyscope** and **Matplotlib**
