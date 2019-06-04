#include <stdio.h>

#define TILE_SIZE 32
#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

__constant__ float c_M[KERNEL_LENGTH][KERNEL_LENGTH];
const int Width = 3072;
const int Height = 3072;
const int nIter = 300;

float * h_Kernel,*h_Input,*h_Output;
float * d_Input, *d_Output;

//this optimization is not good.
__global__ void convolution_2D_tiled_kernel(float* P, float* N, int height, int width, int pitch, int Mask_Width){
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * TILE_SIZE + ty;
    int col_o = blockIdx.x * TILE_SIZE + tx;

    int row_i = row_o - Mask_Width/2;
    int col_i = col_o - Mask_Width/2;

    __shared__ float N_ds[TILE_SIZE + KERNEL_LENGTH - 1][TILE_SIZE + KERNEL_LENGTH - 1];
    if((row_i >= 0) && (row_i < height) && 
       (col_i >= 0) && (col_i < width)){
           N_ds[ty][tx] = N[row_i * pitch + col_i];
       }else{
           N_ds[ty][tx] = 0.0f;
       }
    
    __syncthreads();

    float output = 0.0f;
    if(ty < TILE_SIZE && tx < TILE_SIZE){
        for(int i = 0;i<Mask_Width;i++){
            for(int j = 0;j<Mask_Width;j++){
                output += c_M[i][j] * N_ds[i + ty][j + tx];
            }
        }
        if(row_o < height && col_o < width){
            P[row_o * width + col_o] = output;
        }
    }
}


void gpuRunTiledKernel(){

    dim3 blockDim(TILE_SIZE,TILE_SIZE);
    dim3 gridDim((Width + TILE_SIZE - 1)/TILE_SIZE,(Height + TILE_SIZE - 1)/TILE_SIZE);

    convolution_2D_tiled_kernel<<<gridDim,blockDim>>>(d_Output,d_Input,Height,Width,Width,KERNEL_LENGTH);
    
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(start,NULL);

    for(int i = 0;i<nIter;i++){
        convolution_2D_tiled_kernel<<<gridDim,blockDim>>>(d_Output,d_Input,Height,Width,Width,KERNEL_LENGTH);
    }

    cudaEventRecord(stop, NULL);
    cudaDeviceSynchronize();
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal,start,stop);

    float gpuTime = (msecTotal / nIter) * 0.001;
    printf("Tiled Kernel Throughput = %0.4f MPixels/sec, Time= %.5f sec, Size= %u Pixels\n",
            (1.0e-6 * (Width*Height)/gpuTime),
            gpuTime,
            Width);

    cudaMemcpy(h_Output,d_Output,Width*Height* sizeof(float),cudaMemcpyDeviceToHost);

    bool correct = true;
    double eps = 1.e-6;
    for(int i = 0;i<Width*Height;i++){
        double abs_err = fabs(h_Output[i] - KERNEL_LENGTH * KERNEL_LENGTH * 1.0);

        if (abs_err > eps)
        {
            //printf("Error! Index = %d,h_Output = %f,true value = %d\n", 
             //              i, h_Output[i], KERNEL_LENGTH*KERNEL_LENGTH);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
}

int main(){
    h_Kernel = (float*)malloc(KERNEL_LENGTH * KERNEL_LENGTH * sizeof(float));
    h_Input  = (float*)malloc(Width * Height * sizeof(float));
    h_Output = (float*)malloc(Width * Height * sizeof(float));

    for(unsigned int i = 0;i<KERNEL_LENGTH*KERNEL_LENGTH;i++){
        h_Kernel[i] = 1.0f;
    }

    for(unsigned int i = 0;i<Width*Height;i++){
        h_Input[i] = 1.0f;
    }

    cudaMalloc((void**)&d_Input, Width * Height * sizeof(float));
    cudaMalloc((void**)&d_Output,Width * Height * sizeof(float));

    cudaMemcpy(d_Input,h_Input,Width * Height * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_M,h_Kernel,KERNEL_LENGTH * KERNEL_LENGTH * sizeof(float));

    printf("Running GPU convlution 2D ....\n");

    gpuRunTiledKernel();

    free(h_Input);
    free(h_Kernel);
    free(h_Output);
    cudaFree(d_Input);
    cudaFree(d_Output);
}