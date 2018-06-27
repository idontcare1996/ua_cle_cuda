rm -rf tex
clear
echo " Cleaned screen, started compiling..."
nvcc qdbmp.c -o tex tex.cu -Wno-deprecated-gpu-targets 
echo " Compiler was run, if the program was successfully compiled, it should be executed now."
./tex 25 image.bmp image.bmp