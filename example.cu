#include <stdio.h>

//
// Nearly minimal CUDA example.
// Compile with:
//
// nvcc -o example example.cu
//
extern "C"
{
    #include "qdbmp.h"
}
#define N 1000

//
// A function marked __global__
// runs on the GPU but can be called from
// the CPU.
//
// This function multiplies the elements of an array
// of ints by 2.
//
// The entire computation can be thought of as running
// with one thread per array element with blockIdx.x
// identifying the thread.
//
// The comparison i<N is because often it isn't convenient
// to have an exact 1-1 correspondence between threads
// and array elements. Not strictly necessary here.
//
// Note how we're mixing GPU and CPU code in the same source
// file. An alternative way to use CUDA is to keep
// C/C++ code separate from CUDA code and dynamically
// compile and load the CUDA code at runtime, a little
// like how you compile and load OpenGL shaders from
// C/C++ code.
//
__global__
void add(int *a, int *b) {
    int i = blockIdx.x;
    if (i<N) {
        b[i] = 2*a[i];
    }
}

int main() {
    //
    // Create int arrays on the CPU.
    // ('h' stands for "host".)
    //
    int ha[N], hb[N];

    //
    // Create corresponding int arrays on the GPU.
    // ('d' stands for "device".)
    //
    int *da, *db;
    cudaMalloc((void **)&da, N*sizeof(int));
    cudaMalloc((void **)&db, N*sizeof(int));

    //
    // Initialise the input data on the CPU.
    //
    for (int i = 0; i<N; ++i) {
        ha[i] = i;
    }

    //
    // Copy input data to array on GPU.
    //
    cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice);

    //
    // Launch GPU code with N threads, one per
    // array element.
    //
    add<<<N, 1>>>(da, db);

    //
    // Copy output array from GPU back to CPU.
    //
    cudaMemcpy(hb, db, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i<N; ++i) {
        printf("%d\n", hb[i]);
    }

    //
    // Free up the arrays on the GPU.
    //
    cudaFree(da);
    cudaFree(db);

    // Start stuff

        BMP* bmp;
        UCHAR r, g, b;
        UINT width, height;
        UINT x, y;

    /* Read an image file */
    bmp = BMP_ReadFile( argv[ 1 ] );
    BMP_CHECK_ERROR( stderr, -1 ); 
    /* If an error has occurred, notify and exit */
    /* Get image's dimensions */
    width = BMP_GetWidth( bmp );
    height = BMP_GetHeight( bmp );
    /* Iterate through all the image's pixels */
    for ( x = 0 ; x < width ; ++x )
    {
        for ( y = 0 ; y < height ; ++y )
        {
            /* Get pixel's RGB values */
            BMP_GetPixelRGB( bmp, x, y, &r, &g, &b );
            /* Invert RGB values */
            BMP_SetPixelRGB( bmp, x, y, 255 - r, 255 - g, 255 - b );
        }
    }
    /* Save result */
    BMP_WriteFile( bmp, argv[ 2 ] );
    BMP_CHECK_ERROR( stderr, -2 );
    /* Free all memory allocated for the image */
    BMP_Free( bmp );    

    return 0;
}