# Weighted Ranked Filters

Large Scale Computation Assignment using CUDA to ranked filter an image.

Compile with:  ```nvcc qdbmp.c -o tex tex.cu -Wno-deprecated-gpu-targets```

Run with ```./tex [filter_size] [input_image.bmp] [output_image.bmp]``` where _filter_size_ is an integer (5,9,21 or 25).

Example: ```./tex 25 image.bmp image.bmp```

Note: this project is beeing tested on a _NVIDIA GeForce GTX 480_.
