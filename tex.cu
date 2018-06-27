#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<iostream>
#include <unistd.h>
#include <math.h>
extern "C"
{
    #include "qdbmp.h"    
}
#include <device_launch_parameters.h>


#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define RESET "\x1B[0m"

using namespace std;

/*******************/
/* iDivUp FUNCTION */
/*******************/
UINT iDivUp(UINT hostPtr, UINT b)
{
     return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); 
}

// Device Kernels

//Texture reference Declaration
texture<float,2> texRefEx;

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


__global__ void kernel_w_textures(float* devMPPtr, float * devMPtr, int pitch, int width, int height)
{

   
    // Thread indexes
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
    float greyarrey[25];
    float greyarrey_new[25];
    int grid_size = 9;
    float average;
    
    
    if((idx < width)  && (idy < height)){
        // Texutre Coordinates

        float u_offset = (1/(float)(width))/2;
        float v_offset = (1/(float)(height))/2;
        float u=(idx)/(float)(width)+ u_offset;
        float v=(idy)/(float)(height)+ v_offset;  

        devMPtr[idy*width+idx]=devMPPtr[idy*pitch/sizeof(float)+idx];
        // Write Texture Contents to malloc array +1
        //printf(" %f ", tex2D(texRefEx,u,v));

        -(2*u_offset)
        
        //3x3
        greyarrey[0] = greymaker(tex2D(texRefEx,u,                  v));
        greyarrey[1] = greymaker(tex2D(texRefEx,u,                  v-(2*v_offset)));
        greyarrey[2] = greymaker(tex2D(texRefEx,u+(2*u_offset),     v));
        greyarrey[3] = greymaker(tex2D(texRefEx,u,                  v+(2*v_offset)));
        greyarrey[4] = greymaker(tex2D(texRefEx,u+(2*u_offset),     v));
        greyarrey[5] = greymaker(tex2D(texRefEx,u-(2*u_offset),     v-(2*v_offset)));
        greyarrey[6] = greymaker(tex2D(texRefEx,u+(2*u_offset),     v-(2*v_offset)));
        greyarrey[7] = greymaker(tex2D(texRefEx,u+(2*u_offset),     v+(2*v_offset)));
        greyarrey[8] = greymaker(tex2D(texRefEx,u-(2*u_offset),     v+(2*v_offset)));
        greyarrey[9] = greymaker(tex2D(texRefEx,u,                  v+(4*v_offset)));
        greyarrey[10] = greymaker(tex2D(texRefEx,u+(4*u_offset),    v));
        greyarrey[11] = greymaker(tex2D(texRefEx,u,                 v+(4*v_offset)));
        greyarrey[12] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[13] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[14] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[15] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[16] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[17] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[18] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[19] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[20] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[21] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[22] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[23] = greymaker(tex2D(texRefEx,u,v));
        greyarrey[24] = greymaker(tex2D(texRefEx,u,v));



        

        __syncthreads();

        for(int copy_iterator = 0;copy_iterator<grid_size;copy_iterator++)
            {
                greyarrey_new[copy_iterator] = greyarrey [copy_iterator];
            }

        
        
        int c,d,t;
        for (c = 1 ; c < 9; c++) {
            d = c;
         
            while ( d > 0 && greyarrey_new[d-1] > greyarrey_new[d]) {
              t = greyarrey_new[d];
              greyarrey_new[d] = greyarrey_new[d-1];
              greyarrey_new[d-1] = t;         
              d--;
            }
            
        }
        
        __syncthreads();

        for (i=0;i<grid_size;i++)
        {
            if(greyarrey_new[4]==greyarrey[i])
            switch(i)
            {
                case 0:
                    devMPtr[idy*width+idx] = tex2D(texRefEx,u,v);
                    break;                
                case 1:
                    devMPtr[idy*width+idx] = tex2D(texRefEx,u-(2*u_offset),v);
                    break;
                case 2:
                    devMPtr[idy*width+idx] = tex2D(texRefEx,u-(2*u_offset),v+(2*v_offset));
                    break;
                case 3:
                    devMPtr[idy*width+idx] = tex2D(texRefEx,u,v-(2*v_offset));
                    break;
                case 4:
                    devMPtr[idy*width+idx] = tex2D(texRefEx,u-(2*u_offset),v-(2*v_offset));
                    break;              
                case 5:
                    devMPtr[idy*width+idx] = tex2D(texRefEx,u,v+(2*v_offset));
                    break;
                case 6:
                    devMPtr[idy*width+idx] = tex2D(texRefEx,u+(2*u_offset),v-(2*v_offset));
                    break;
                case 7:
                    devMPtr[idy*width+idx] = tex2D(texRefEx,u+(2*u_offset),v);
                    break;
                case 8:
                    devMPtr[idy*width+idx] = tex2D(texRefEx,u+(2*u_offset),v+(2*v_offset));
                    break;
                default:
                    printf(KRED "  \n\n ERROR IN SWITCH CASE: KERNEL! \n\n" RESET);
                    
            }
            
        }

        //printf(" [%i,%i] %f \n", idx,idy,greyarrey[4] );

        __syncthreads();
      
        
        /*
        int red,green,blue,inv_red,inv_green,inv_blue;
        float value;
        value = greyarrey[4];
        red = floor(value / 256.0 / 256.0);
        green = floor((value - (red*256*256))/256);
        blue = floor( value - (red*256*256)-(green*256));

        inv_red = 255-red;
        inv_green = 255-green;
        inv_blue = 255-blue;
        /*
        float new_value = (float)
        printf("[%i,%i][%f,%f] = [%f] %i(%i) %i(%i) %i(%i) \n",idx,idy,u,v,value,red,inv_red,green,inv_green,blue,inv_blue);
        */
        //printf("Arrey: %f | Tex2D: %f \n ",greymaker(tex2D(texRefEx,u,v)),tex2D(texRefEx,u,v));
        
        //devMPtr[idy*width+idx]= tex2D(texRefEx,u,v) ;//+1.0f;
    }
}

int main()
{
    int width;
    int height;
    BMP* bmp;
    UCHAR r, g, b;
    int ir,ig,ib;
    float fr,fg,fb;
    UINT Nrows,Ncols;
    

     /* Read an image file */
    
     printf(" Reading the image file... \n");
   
     bmp = BMP_ReadFile( "image.bmp" );
     
     BMP_CHECK_ERROR( stderr, -1 ); 
     /* If an error has occurred, notify and exit */
     /* Get image's dimensions */
     
     Ncols = BMP_GetWidth( bmp );
     Nrows = BMP_GetHeight( bmp );
   
     width = (int) Ncols;
     height = (int) Nrows;   
    

 // memory size
 size_t memsize=width*height;
 size_t offset;
float * data,  // input from host
 *h_out, // host space for output
 *devMPPtr, // malloc Pitch ptr
  *devMPtr; // malloc ptr

 size_t pitch;

 // Allocate space on the host
 data=(float *)malloc(sizeof(float)*memsize);
 h_out=(float *)malloc(sizeof(float)*memsize);


// Define data
for (int i = 0; i <  height; i++)
{
    for (int j=0; j < width; j++)
    {
    BMP_GetPixelRGB( bmp, j, i, &r, &g, &b );    
    ir = (int)r;    ig = (int)g;    ib = (int)b;
    //printf("I %i, %i, %i \t",ir,ig,ib);
    fr = (float)((float)ir);    fg = (float)((float)ig);  fb = (float)((float)ib);
    //printf("F %f, %f, %f \t",fr,fg,fb);
    //Put the rgb values inside a single value, to send to the array
    
    
    data[i*width+j]=fr*256*256 + fg * 256 + fb;
    }
    //printf("\n");
}

// Define the grid
dim3 gridSize((int)(width/BLOCKSIZE_X)+1,(int)(height/BLOCKSIZE_Y)+1);
dim3 blockSize(BLOCKSIZE_X, BLOCKSIZE_Y);

// allocate Malloc Pitch
cudaMallocPitch((void**)&devMPPtr,&pitch, width * sizeof(float), height);

// Print the pitch
//printf("The pitch is %d \n",pitch/sizeof(float));

// Texture Channel Description
//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);

// Bind texture to pitch mem:
cudaBindTexture2D(&offset,&texRefEx,devMPPtr,&channelDesc,width,height,pitch);
/*
printf("My Description x is %i\n",channelDesc.x);
printf("My Description y is %i\n",channelDesc.y);
printf("My Description z is %i\n",channelDesc.z);
printf("My Description w is %i\n",channelDesc.w);;
printf("My Description kind is %i\n",channelDesc.x);
printf("Offset is %i \n",offset);
*/ 

// Set mutable properties:
texRefEx.normalized=true;
texRefEx.addressMode[0]=cudaAddressModeMirror;
texRefEx.addressMode[1]=cudaAddressModeMirror;
texRefEx.filterMode= cudaFilterModePoint;

// Allocate cudaMalloc memory
cudaMalloc((void**)&devMPtr,memsize*sizeof(float));

// Read data from host to device
cudaMemcpy2D((void*)devMPPtr,pitch,(void*)data,sizeof(float)*width,
  sizeof(float)*width,height,cudaMemcpyHostToDevice);

//Read back and check this memory
cudaMemcpy2D((void*)h_out,width*sizeof(float),(void*)devMPPtr,pitch,
  sizeof(float)*width,height,cudaMemcpyDeviceToHost);

// Print the memory
 for (int i=0; i<height; i++){
  for (int j=0; j<width; j++){
    int red,green,blue;
    red = floor(h_out[i*width+j] / 256.0 / 256.0);
    green = floor((h_out[i*width+j] - (red*256*256))/256);
    blue = floor( h_out[i*width+j] - (red*256*256)-(green*256));
    //printf("data[%i,%i] = %i %i %i \t",i,j,red,green,blue);
    
   //printf("%f ",h_out[i*width+j]);
  }
 //printf("\n");
 }

 printf("\n DONE \n");
// Memory is fine... 

kernel_w_textures<<<gridSize,blockSize>>>(devMPPtr, devMPtr, pitch,width,height);
//gpuErrchk(cudaPeekAtLastError());
//gpuErrchk(cudaDeviceSynchronize());

// Copy back data to host
cudaMemcpy((void*)h_out,(void*)devMPtr,width*height*sizeof(float),cudaMemcpyDeviceToHost);


// Print the Result
for (int i=0; i<height; i++){
    for (int j=0; j<width; j++){
      int red,green,blue;
      red = floor(h_out[i*width+j] / 256.0 / 256.0);
      green = floor((h_out[i*width+j] - (red*256*256))/256);
      blue = floor( h_out[i*width+j] - (red*256*256)-(green*256));
      //printf("data[%i,%i] = %i %i %i \t",i,j,red,green,blue);
      BMP_SetPixelRGB( bmp, j, i,(int)red, (int)green, (int)blue);
     //printf("%f ",h_out[i*width+j]);
    }
   //printf("\n");
   }
   BMP_WriteFile( bmp, "image.bmp" );
   printf("  \x1B[32m Image output is ready! \n \x1B[0m");
   BMP_CHECK_ERROR( stderr, -2 );
   /* Free all memory allocated for the image */
   BMP_Free( bmp );    
   printf("\n DONE \n");

return(0);
}