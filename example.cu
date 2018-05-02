#include <stdio.h>

extern "C"
{
    #include "qdbmp.h"
}

__global__
void do_the_math ()
{

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


    // Create the array where the rgb values will be stored:
    float host_image_array[width][height];

    /* Iterate through all the image's pixels */
    for ( x = 0 ; x < width ; ++x )
    {
        for ( y = 0 ; y < height ; ++y )
        {
            /* Get pixel's RGB values */
            BMP_GetPixelRGB( bmp, x, y, &r, &g, &b );
            // Put the rgb values inside a single value, to send to the array
            float rgb = (1000000*(int)r)+(1000*(int)g)+((int)b);
            printf(" %f \n", rgb);
            
            // /* Invert RGB values */
            // BMP_SetPixelRGB( bmp, x, y, 255 - r, 255 - g, 255 - b );
        }
    }
    /* Save result */
    BMP_WriteFile( bmp, "image_output.bmp" );
    printf(" Image output is ready \n");
    BMP_CHECK_ERROR( stderr, -2 );
    /* Free all memory allocated for the image */
    BMP_Free( bmp );    
    
    return 0;
}