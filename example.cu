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
UINT iDivUp(UINT hostPtr, UINT b)
{
     return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); 
}

/********************************************************************/
/* Transform RGB (24Bit) (0-255255255) to Grey value (0-1) FUNCTION */
/********************************************************************/

__host__ __device__ float greymaker (UINT rgb)
{  
    int red;
    int green;
    int blue;
    //int greyscaled;

    float red_float;
    float green_float;
    float blue_float;
    float grey;

    red = rgb/1000000;
    green = (rgb/1000) - (red*1000);
    blue = rgb - (red*1000000) - (green*1000);

    red_float = (float) (red)/256;
    green_float = (float) (green)/256;
    blue_float = (float) (blue)/256;

    grey = (0.299 * red_float)+ (0.587 * green_float) + (0.114 * blue_float);

    return grey;
}

/******************/
/* TEST KERNEL 2D */
/******************/
__global__ void test_kernel_2D(UINT *devPtr, size_t pitch, int Ncols, int Nrows) 
{
    int    tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int    tidy = blockIdx.y*blockDim.y + threadIdx.y;
    
    int greyscaled;
    UINT rgb;
    int i;
    int j;
    int greys[9];
    
    if ((tidx < Ncols) && (tidy < Nrows))
    {
        for (i=1;i<tidy-1;i++);
        {

            printf(" i=%i ", i);

            UINT *row_a = (UINT *)((char*)devPtr + (i-1) * pitch);
            UINT *row_b = (UINT *)((char*)devPtr + (i) * pitch);
            UINT *row_c = (UINT *)((char*)devPtr + (i+1) * pitch);

            /*
            rgb = row_a[1];
            greyscaled = (int)((greymaker(rgb))*256);

            rgb = (greyscaled*1000000) + (greyscaled * 1000) + greyscaled;

            */

            for(j=1;j<tidx-1;j++)
            {
            row_a[j-1] = 255255000;
            row_b[j] = 255000000;
            row_c[j+1] = 000255000;
            }

           
           
        }
        
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
           // printf("row %i column %i value %i \n", i, j, hostPtr[i][j]);
            UINT rgb_output = hostPtr[i][j];            
            r = rgb_output/1000000;
            g =  (rgb_output/1000) - (r*1000) ;
            b = rgb_output - (r*1000000) - (g*1000);
            BMP_SetPixelRGB( bmp, i, j,(int)r, (int)g, (int)b);
        }
    }
    printf(" Width: %u Height: %u \n", width, height);
    printf(" RGB Value at 20:20 %i \n", hostPtr[0][16]);
    /* Save result */
    BMP_WriteFile( bmp, "image_output.bmp" );
    printf(" Image output is ready \n");
    BMP_CHECK_ERROR( stderr, -2 );
    /* Free all memory allocated for the image */
    BMP_Free( bmp );    
    
    return 0;
}