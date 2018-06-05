#include <stdio.h>
#include <unistd.h>
#include <math.h>
extern "C"
{
    #include "qdbmp.h"
    
}
#include <device_launch_parameters.h>
//#include <conio.h>

#define BLOCKSIZE_x 8
#define BLOCKSIZE_y 8

#define BETWEEN(value, min, max) (value < max && value > min)



/*****************/
/* CUDA MEMCHECK */
/*****************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
        if (abort) { getchar(); exit(code); }
    }
}

/*******************/
/* iDivUp FUNCTION */
/*******************/
UINT iDivUp(UINT hostPtr, UINT b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }

/******************/
/* TEST KERNEL 2D */
/******************/
__global__ void test_kernel_2D(UINT *devPtr, size_t pitch, int Ncols, int Nrows) 
{
    int    tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int    tidy = blockIdx.y*blockDim.y + threadIdx.y;
    
    int value;
    
    
    if ((tidx < Ncols) && (tidy < Nrows))
    {
        UINT *row_a = (UINT *)((char*)devPtr + tidy * pitch);
        
        
        row_a[tidx] = 1 + row_a[tidx];

        
    }
    
}

int main() {   

    
    BMP* bmp;
    UCHAR r, g, b;
    UINT width, height;
    UINT x, y;
    
    /* Read an image file */
    
    printf(" Reading the image file... \n");
   
    bmp = BMP_ReadFile( "image_input.bmp" );
    
    BMP_CHECK_ERROR( stderr, -1 ); 
    /* If an error has occurred, notify and exit */
    /* Get image's dimensions */
    
    width = BMP_GetWidth( bmp );
    height = BMP_GetHeight( bmp );

    UINT Nrows = width;
    UINT Ncols = height;

    // Create the array where the rgb values will be stored:
    //int host_image_array[width][height];
    UINT hostPtr[Nrows][Ncols];

    /* Iterate through all the image's pixels */
    for ( x = 0 ; x < width ; ++x )
    {
        for ( y = 0 ; y < height ; ++y )
        {
            /* Get pixel's RGB values */
            BMP_GetPixelRGB( bmp, x, y, &r, &g, &b );
            // Put the rgb values inside a single value, to send to the array
            UINT rgb = (1000000*(int)r)+(1000*(int)g)+((int)b);
            //printf(" [%i:%i] [ %i ] [ %i ]\n", x,y,rgb,(1000000*(int)r)+(1000*(int)g)+((int)b));
            hostPtr[x][y] = rgb;
            // /* Invert RGB values */
            // BMP_SetPixelRGB( bmp, x, y, 255 - r, 255 - g, 255 - b );
        }
    }

    // Copy the array to the device

    
    
    UINT *devPtr;
    size_t pitch;
   
    gpuErrchk(cudaMallocPitch(&devPtr, &pitch, Ncols * sizeof(UINT), Nrows));
    gpuErrchk(cudaMemcpy2D(devPtr, pitch, hostPtr, Ncols*sizeof(UINT), Ncols*sizeof(UINT), Nrows, cudaMemcpyHostToDevice));

    dim3 gridSize(iDivUp(Ncols, BLOCKSIZE_x), iDivUp(Nrows, BLOCKSIZE_y));
    dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);

    test_kernel_2D << <gridSize, blockSize >> >(devPtr, pitch,Ncols,Nrows);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy2D(hostPtr, Ncols * sizeof(UINT), devPtr, pitch, Ncols * sizeof(UINT), Nrows, cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nrows; i++) 
    {
        for (int j = 0; j < Ncols; j++)
        {
            printf("row %i column %i value %i \n", i, j, hostPtr[i][j]);
            UINT rgb_output = hostPtr[i][j];
            float rgb_output_float = rgb_output;
            r = trunc( rgb_output_float/1000000 );
            g = trunc ( (rgb_output_float/1000) - (r*1000) );
            b = rgb_output_float - (r*1000000) - (g*1000);
            BMP_SetPixelRGB( bmp, j, i,(int)r, (int)g, (int)b);
        }
    }
    printf(" %u %u ", width, height);
    printf(" RGB Value at 20:20 %i \n", hostPtr[0][16]);
    /* Save result */
    BMP_WriteFile( bmp, "image_output.bmp" );
    printf(" Image output is ready \n");
    BMP_CHECK_ERROR( stderr, -2 );
    /* Free all memory allocated for the image */
    BMP_Free( bmp );    
    
    return 0;
}