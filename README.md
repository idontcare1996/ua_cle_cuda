# Weighted Ranked Filters

Large Scale Computation Assignment using CUDA to ranked filter an image.

Compile with:  ```nvcc qdbmp.c -o tex tex.cu -Wno-deprecated-gpu-targets```

Place ```image.bmp``` in the root directory, and run ```./tex``` .

The file ```image.bmp``` should appear modified in the root directory with the 5x5 filter applied.

Note: this project is beeing tested on a _NVIDIA GeForce GTX 480_.
