//-------------------------------------------// 
/*           Weighted Rank Filter
            CLE - First Assignment

                by

        Carlos Oliveira (88702) (carlosmbo@ua.pt)
                and
        João Caires (89094) (?@ua.pt)


*/
//-------------------------------------------// 



#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <unistd.h>
#include <math.h>
#include <device_launch_parameters.h>

// Load the BMP Library
extern "C"
{
    #include "qdbmp.h"    
}

// Define CUDA Kernel's launch Block Size
#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32

// Some colour codes for the terminal

#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KCYN  "\x1B[36m"
#define RESET "\x1B[0m"

using namespace std;

/*******************/
/* iDivUp FUNCTION */
/*******************/
UINT iDivUp(UINT hostPtr, UINT b)
{
     return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); 
}

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



/****************************/
/* Convert RGB to Greyscale */
/****************************/

__host__ __device__ float greymaker (float rgb)
{  
 
    int red = floor(rgb / 256 / 256);
    int green = floor((rgb - (red*256*256))/256);
    int blue = floor(rgb- (red*256*256)-(green*256));

    float red_float = (float) (red)/255;
    float green_float = (float) (green)/255;
    float blue_float = (float) (blue)/255;

    float grey = (0.299 * red_float)+ (0.587 * green_float) + (0.114 * blue_float);

    return grey;
}

//Texture reference Declaration
texture<float,2> texref;

__global__ void wrf_textures(float* devMPPtr, float * devMPtr, int pitch, int width, int height, int filter_grid_size)
{

   
    // Thread indexes
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;

    // Declaring some variables and arrays for later use
    float greyarrey[25];
    float greyarrey_new[25];
    
    
    
    
    if((idx < width)  && (idy < height)){
        // Texutre Coordinates

        float u_offset = (1/(float)(width))/2;
        float v_offset = (1/(float)(height))/2;
        float u=(idx)/(float)(width)+ u_offset;
        float v=(idy)/(float)(height)+ v_offset;  

        devMPtr[idy*width+idx]=devMPPtr[idy*pitch/sizeof(float)+idx];
        // Write Texture Contents to malloc array +1
        //printf(" %f ", tex2D(texref,u,v));

        
        
    //3x3 sem cantos
        greyarrey[0] = greymaker(tex2D(texref,     u,                       v                  ));
        greyarrey[1] = greymaker(tex2D(texref,     u,                       v - ( 2 * v_offset)));
        greyarrey[2] = greymaker(tex2D(texref,     u + ( 2 * u_offset),     v                  ));
        greyarrey[3] = greymaker(tex2D(texref,     u,                       v + ( 2 * v_offset)));
        greyarrey[4] = greymaker(tex2D(texref,     u + ( 2 * u_offset),     v                  ));
    //3x3 com cantos
        greyarrey[5] = greymaker(tex2D(texref,     u - ( 2 * u_offset),     v - ( 2 * v_offset)));
        greyarrey[6] = greymaker(tex2D(texref,     u + ( 2 * u_offset),     v - ( 2 * v_offset)));
        greyarrey[7] = greymaker(tex2D(texref,     u + ( 2 * u_offset),     v + ( 2 * v_offset)));
        greyarrey[8] = greymaker(tex2D(texref,     u - ( 2 * u_offset),     v + ( 2 * v_offset)));
    //5x5 sem cantos
        greyarrey[9] = greymaker(tex2D(texref,     u,                       v - ( 4 * v_offset)));
        greyarrey[10] = greymaker(tex2D(texref,    u + ( 4 * u_offset),     v                  ));
        greyarrey[11] = greymaker(tex2D(texref,    u,                       v + ( 4 * v_offset)));
        greyarrey[12] = greymaker(tex2D(texref,    u - ( 4 * u_offset),     v                  ));
        greyarrey[13] = greymaker(tex2D(texref,    u + ( 2 * u_offset),     v - ( 4 * v_offset)));
        greyarrey[14] = greymaker(tex2D(texref,    u + ( 4 * u_offset),     v - ( 2 * v_offset)));
        greyarrey[15] = greymaker(tex2D(texref,    u + ( 4 * u_offset),     v + ( 2 * v_offset)));
        greyarrey[16] = greymaker(tex2D(texref,    u + ( 2 * u_offset),     v + ( 4 * v_offset)));
        greyarrey[17] = greymaker(tex2D(texref,    u - ( 2 * u_offset),     v + ( 4 * v_offset)));
        greyarrey[18] = greymaker(tex2D(texref,    u - ( 4 * u_offset),     v + ( 2 * v_offset)));
        greyarrey[19] = greymaker(tex2D(texref,    u - ( 4 * u_offset),     v - ( 2 * v_offset)));
        greyarrey[20] = greymaker(tex2D(texref,    u - ( 2 * u_offset),     v - ( 4 * v_offset)));
    //5x5 com cantos
        greyarrey[21] = greymaker(tex2D(texref,     u + ( 4 * u_offset),     v - ( 4 * v_offset)));
        greyarrey[22] = greymaker(tex2D(texref,     u + ( 4 * u_offset),     v + ( 4 * v_offset)));
        greyarrey[23] = greymaker(tex2D(texref,     u - ( 4 * u_offset),     v + ( 4 * v_offset)));
        greyarrey[24] = greymaker(tex2D(texref,     u - ( 4 * u_offset),     v - ( 4 * v_offset)));


        __syncthreads();

        for(int copy_iterator = 0;copy_iterator<filter_grid_size;copy_iterator++)
            {
                greyarrey_new[copy_iterator] = greyarrey [copy_iterator];
            }

        int c,d;
        float t;
        for (c = 1 ; c < filter_grid_size; c++) {
            d = c;
         
            while ( d > 0 && greyarrey_new[d-1] > greyarrey_new[d]) {
              t = greyarrey_new[d];
              greyarrey_new[d] = greyarrey_new[d-1];
              greyarrey_new[d-1] = t;         
              d--;
            }
            
        }
        
        __syncthreads();

        for (int i=0;i<filter_grid_size;i++)
        {
            if(greyarrey_new[(filter_grid_size+1)/2]==greyarrey[i])
            switch(i)
            {
            //3x3 sem cantos
                case 0:
                    devMPtr [idy*width+idx] = tex2D(texref,     u,                       v                  );
                    break;
                case 1:
                    devMPtr [idy*width+idx] = tex2D(texref,     u,                       v - ( 2 * v_offset));
                    break;
                case 2: 
                    devMPtr [idy*width+idx] = tex2D(texref,     u + ( 2 * u_offset),     v                  );
                    break;
                case 3:
                    devMPtr [idy*width+idx] = tex2D(texref,     u,                       v + ( 2 * v_offset));
                    break; 
                case  4:
                    devMPtr [idy*width+idx] = tex2D(texref,     u + ( 2 * u_offset),     v                  );
                    break;
            //3x3 com cantos
                case  5:
                    devMPtr [idy*width+idx] = tex2D(texref,     u - ( 2 * u_offset),     v - ( 2 * v_offset));
                    break; 
                case  6:
                    devMPtr [idy*width+idx] = tex2D(texref,     u + ( 2 * u_offset),     v - ( 2 * v_offset));
                    break; 
                case  7:
                    devMPtr [idy*width+idx] = tex2D(texref,     u + ( 2 * u_offset),     v + ( 2 * v_offset));
                    break; 
                case  8:
                    devMPtr [idy*width+idx] = tex2D(texref,     u - ( 2 * u_offset),     v + ( 2 * v_offset));
            //5x5 sem cantos
                    break; 
                case  9:
                    devMPtr [idy*width+idx] = tex2D(texref,     u,                       v - ( 4 * v_offset));
                    break; 
                case  10:
                    devMPtr [idy*width+idx] = tex2D(texref,    u + ( 4 * u_offset),     v                  );
                    break; 
                case  11:
                    devMPtr [idy*width+idx] = tex2D(texref,    u,                       v + ( 4 * v_offset));
                    break; 
                case  12:
                    devMPtr [idy*width+idx] = tex2D(texref,    u - ( 4 * u_offset),     v                  );
                    break; 
                case  13:
                    devMPtr [idy*width+idx] = tex2D(texref,    u + ( 2 * u_offset),     v - ( 4 * v_offset));
                    break; 
                case  14:
                    devMPtr [idy*width+idx] = tex2D(texref,    u + ( 4 * u_offset),     v - ( 2 * v_offset));
                    break; 
                case  15:
                    devMPtr [idy*width+idx] = tex2D(texref,    u + ( 4 * u_offset),     v + ( 2 * v_offset));
                    break; 
                case  16:
                    devMPtr [idy*width+idx] = tex2D(texref,    u + ( 2 * u_offset),     v + ( 4 * v_offset));
                    break; 
                case  17:
                    devMPtr [idy*width+idx] = tex2D(texref,    u - ( 2 * u_offset),     v + ( 4 * v_offset));
                    break; 
                case  18:
                    devMPtr [idy*width+idx] = tex2D(texref,    u - ( 4 * u_offset),     v + ( 2 * v_offset));
                    break; 
                case  19:
                    devMPtr [idy*width+idx] = tex2D(texref,    u - ( 4 * u_offset),     v - ( 2 * v_offset));
                    break; 
                case  20:
                    devMPtr [idy*width+idx] = tex2D(texref,    u - ( 2 * u_offset),     v - ( 4 * v_offset));
            //5x5 com cantos
                    break; 
                case  21:
                    devMPtr [idy*width+idx] = tex2D(texref,     u + ( 4 * u_offset),     v - ( 4 * v_offset));
                    break; 
                case  22:
                    devMPtr [idy*width+idx] = tex2D(texref,     u + ( 4 * u_offset),     v + ( 4 * v_offset));
                    break; 
                case  23:
                    devMPtr [idy*width+idx] = tex2D(texref,     u - ( 4 * u_offset),     v + ( 4 * v_offset));
                    break; 
                case  24:
                    devMPtr [idy*width+idx] = tex2D(texref,     u - ( 4 * u_offset),     v - ( 4 * v_offset));
                    break;    
                default:
                    printf(KRED "  \n\n ERROR IN SWITCH CASE: KERNEL! \n\n" RESET);
                    
            }
            
        }
        __syncthreads();
      
    }
}

int main( int argc,char *argv[] )
{
    // Some variables
    int width,height;           // Declaring for later assignment
    int filter_grid_size = 25;  //Size of the filter (5,9,21,25);

    int ir,ig,ib;               // Placeholder for RGB decomposition/composition in int
    float fr,fg,fb;             // Placeholder for RGB decomposition/composition in float   

    // BMP-related variables
    BMP* bmp;
    UCHAR r, g, b;  
    UINT Nrows,Ncols;

    /* Read an image file */
    
    printf(" [INFO] Reading the image file...\n");    
    bmp = BMP_ReadFile( "image.bmp" );
    
    
    BMP_CHECK_ERROR( stderr, -1 ); 
    /* If an error has occurred, notify and exit */
    printf(KGRN" [GOOD] File successfully read!\n"RESET);  

    /* Get image's dimensions */    
    Ncols = BMP_GetWidth( bmp );
    Nrows = BMP_GetHeight( bmp );
     

    // Assign dimension values to the int's:
    width = (int) Ncols;
    height = (int) Nrows;

    printf(" [INFO] Image: Width: "KGRN"%i"RESET"   Height: "KGRN"%i"RESET"\n",width,height); 
    
    /* Define memory needed for image */
    // memory size
    size_t memsize=width*height;
    size_t offset;
    size_t pitch;

    // Arrays to store the data (their pointers)
    float   *data, // input from host (Image goes here)
            *h_out, // host space for output (Ready to send to kernel)
            *devMPPtr, // malloc Pitch ptr
            *devMPtr; // malloc ptr

    // Allocate space on the host
    printf(" [INFO] Allocating space on the host...\n");

    data=(float *)malloc(sizeof(float)*memsize);
    h_out=(float *)malloc(sizeof(float)*memsize);

    printf(KGRN" [GOOD] Space successfully alocated!\n"RESET);  

    

    // Convert RGB values to a floating point (255/255/255 to 0-16777216)
    printf(" [INFO] Reading image into array, converting from RGB to FP...\n");
    for (int i = 0; i <  height; i++)
    {
        for (int j=0; j < width; j++)
        {
            // Get the Red, Green and Blue values from the pixel in (j,i)
            BMP_GetPixelRGB( bmp, j, i, &r, &g, &b );

            // Send them through some variables to convert from UINT to int
            ir = (int)r;    ig = (int)g;    ib = (int)b;

            // Send them through some variables to convert from int to float
            fr = (float)((float)ir);    fg = (float)((float)ig);  fb = (float)((float)ib);
            
            //Put the rgb values inside a single value ( 0.00 - 16,777,216.00 ), to send to the array  
            data[i*width+j] = fr*256*256 + fg * 256 + fb; // Red_Value x 256^2 + Green_Value x 256 + Blue_Value
        }        
    }
    printf(KGRN" [GOOD] Image successfully read and converted!\n"RESET);  


    printf(" [INFO] Defining Grid, allocating memory, binding textures, setting their properties... \n");
    // Define the grid
    dim3 gridSize((int)(width/BLOCKSIZE_X)+1,(int)(height/BLOCKSIZE_Y)+1);
    dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y);

    // Allocate Malloc Pitch
    cudaMallocPitch((void**)&devMPPtr,&pitch, width * sizeof(float), height);
    
    // Texture Channel Description    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);

    // Bind texture to pitch memory:
    cudaBindTexture2D(&offset,&texref,devMPPtr,&channelDesc,width,height,pitch);

    // Set mutable properties:
    texref.normalized=true;
    texref.addressMode[0]=cudaAddressModeMirror;
    texref.addressMode[1]=cudaAddressModeMirror;
    texref.filterMode= cudaFilterModePoint;

    // Allocate cudaMalloc memory
    cudaMalloc((void**)&devMPtr,memsize*sizeof(float));

    printf(KGRN" [GOOD] Done! \n"RESET);  


    printf(" [INFO] Copying data to device and reading it back... \n");

    // Read data from host to device
    cudaMemcpy2D((void*)devMPPtr,pitch,(void*)data,sizeof(float)*width,sizeof(float)*width,height,cudaMemcpyHostToDevice);

    //Read back and check this memory
    cudaMemcpy2D((void*)h_out,width*sizeof(float),(void*)devMPPtr,pitch,sizeof(float)*width,height,cudaMemcpyDeviceToHost);

    // Print the memory after allocating
        /*
        for (int i=0; i<height; i++)
        {
            for (int j=0; j<width; j++)
            {
                int red,green,blue;
                red = floor(h_out[i*width+j] / 256.0 / 256.0);
                green = floor((h_out[i*width+j] - (red*256*256))/256);
                blue = floor( h_out[i*width+j] - (red*256*256)-(green*256));
                printf("data[%i,%i] = %i %i %i \t",i,j,red,green,blue);
                
                printf("%f ",h_out[i*width+j]);
            }
            printf("\n");
        }
        */
    printf(KGRN" [GOOD] Done! \n"RESET);



    /*********************/
    // Launch the Kernel //
    /*********************/
     
    printf(" [INFO] Copying data to device and reading it back... \n");
    wrf_textures<<<gridSize,blockSize>>>(devMPPtr, devMPtr, pitch,width,height,filter_grid_size);
    
    // Check for errors
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    printf(KGRN" [GOOD] Done! \n"RESET);

    // Copy back data to host
    printf(" [INFO] Copying data from device and reading it back... \n");
    cudaMemcpy((void*)h_out,(void*)devMPtr,width*height*sizeof(float),cudaMemcpyDeviceToHost);


    // Save the result on the BMP file
    
    for (int i=0; i<height; i++)
    {
        for (int j=0; j<width; j++)
        {
            int red,green,blue;
            red =   floor( h_out[i*width+j] / 256.0 / 256.0);
            green = floor((h_out[i*width+j] - (red*256*256))/256);
            blue =  floor( h_out[i*width+j] - (red*256*256)-(green*256));

            // printf("data[%i,%i] = %i %i %i \t",i,j,red,green,blue);

            BMP_SetPixelRGB( bmp, j, i,(int)red, (int)green, (int)blue);

            // printf("%f ",h_out[i*width+j]);
        }
        //printf("\n");
    }
    printf(KGRN" [GOOD] Done! \n"RESET);

    // Write the resulting image to a file:
    printf(" [INFO] Writing results to file... \n");
    BMP_WriteFile( bmp, "image.bmp" );
    
    // Check for errors
    BMP_CHECK_ERROR( stderr, -2 );
    printf(KGRN" [GOOD] Done! \n"RESET);

    // Free all memory allocated for the image
    BMP_Free( bmp );   

    printf(KGRN" [GOOD] Program Terminated Successfully! \n"RESET);

    return(0);
}