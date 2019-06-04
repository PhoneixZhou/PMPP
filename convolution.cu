
#include <stdio.h>
#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)
#define TILE_SIZE 1024

__constant__ float c_M[KERNEL_LENGTH];
const int Width = 10240000;
const int nIter = 300;

float * h_Kernel,*h_Input,*h_Output;
float * d_Kernel, *d_Input, * d_Output;

__global__ void convolution_1D_basic_kernel(float* N, float* g_M, float* P,int Mask_Width, int Width){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (Mask_Width/2);
    for(int j = 0;j<Mask_Width;j++){
        if(N_start_point + j>=0 && N_start_point + j <Width){
            Pvalue += N[N_start_point + j] * g_M[j];
        }else if(N_start_point + j<0){
            Pvalue += N[0] * g_M[j];
        }else{
            Pvalue += N[Width - 1] * g_M[j];
        }
    }
    P[i] = Pvalue;
}

__global__ void convolution_1D_constant_kernel(float * N,float* P,int Mask_Width,int Width){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (Mask_Width/2);
    for(int j = 0;j<Mask_Width;j++){
        if(N_start_point + j>=0 && N_start_point + j <Width){
            Pvalue += N[N_start_point + j] * c_M[j];
        }else if(N_start_point + j < 0){
            Pvalue += N[0] * c_M[j];
        }else{
            Pvalue += N[Width - 1] * c_M[j];
        }
    }
    P[i] = Pvalue;
}

__global__ void convolution_1D_tiled_kernel(float* N, float* P,int Mask_Width, int Width){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float N_ds[TILE_SIZE + KERNEL_LENGTH-1];

    int n = Mask_Width/2;
    int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if(threadIdx.x >= blockDim.x - n){
        N_ds[threadIdx.x - (blockDim.x - n)] = (halo_index_left<0) ? N[0]:N[halo_index_left];
    }

    N_ds[n+threadIdx.x] = N[blockIdx.x * blockDim.x + threadIdx.x];
    
    int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if(threadIdx.x < n){
        N_ds[n + blockDim.x + threadIdx.x] = (halo_index_right >= Width) ? N[Width - 1]:N[halo_index_right];
    }

    __syncthreads();

    float Pvalue = 0;
    for(int j = 0;j<Mask_Width;j++){
        Pvalue += N_ds[threadIdx.x + j] * c_M[j];
    }
    P[i] = Pvalue;
}

__global__ void convolution_1D_tiled_caching_kernel(float* N, float* P, int Mask_Width, int Width){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float N_ds[TILE_SIZE];
    N_ds[threadIdx.x] = N[i];

    __syncthreads();

    int This_tile_start_point = blockIdx.x * blockDim.x;
    int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;

    int N_start_point = i - (Mask_Width/2);
    float Pvalue = 0;
    for(int j = 0;j<Mask_Width;j++){
        int N_index = N_start_point + j;

        if(N_index >=0 && N_index<Width){
            if((N_index>=This_tile_start_point) && (N_index< Next_tile_start_point)){
                Pvalue += N_ds[threadIdx.x+ j - (Mask_Width/2)] * c_M[j];
            }else{
                Pvalue += N[N_index];
            }
        }
    }
    P[i] = Pvalue;
}

void gpuRunBasicKernel(){

    convolution_1D_basic_kernel<<<(Width+TILE_SIZE - 1)/TILE_SIZE,TILE_SIZE>>>(d_Input,d_Kernel,d_Output,KERNEL_LENGTH,Width);
    
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(start,NULL);

    for(int i = 0;i<nIter;i++){
        convolution_1D_basic_kernel<<<(Width+TILE_SIZE - 1)/TILE_SIZE,TILE_SIZE>>>(d_Input,d_Kernel,d_Output,KERNEL_LENGTH,Width);
    }

    cudaEventRecord(stop, NULL);
    cudaDeviceSynchronize();
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal,start,stop);

    float gpuTime = (msecTotal / nIter) * 0.001;
    printf("Basic Kernel Throughput = %0.4f MPixels/sec, Time= %.5f sec, Size= %u Pixels\n",
            (1.0e-6 * (Width)/gpuTime),
            gpuTime,
            Width);

    cudaMemcpy(h_Output,d_Output,Width * sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    bool correct = true;
    double eps = 1.e-6;
    for(int i = 0;i<Width;i++){

        double abs_err = fabs(h_Output[i] - KERNEL_LENGTH * 1.0);

        if (abs_err > eps)
        {
            printf("Error! Index = %d,h_Output = %f,true value = %d\n", 
                           i, h_Output[i], KERNEL_LENGTH);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
}

void gpuRunConstantKernel(){

    convolution_1D_constant_kernel<<<(Width+TILE_SIZE - 1)/TILE_SIZE,TILE_SIZE>>>(d_Input,d_Output,KERNEL_LENGTH,Width);
    
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(start,NULL);

    for(int i = 0;i<nIter;i++){
        convolution_1D_constant_kernel<<<(Width+TILE_SIZE - 1)/TILE_SIZE,TILE_SIZE>>>(d_Input,d_Output,KERNEL_LENGTH,Width);
    }

    cudaEventRecord(stop, NULL);
    cudaDeviceSynchronize();
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal,start,stop);

    float gpuTime = (msecTotal / nIter) * 0.001;
    printf("Constant Kernel Throughput = %0.4f MPixels/sec, Time= %.5f sec, Size= %u Pixels\n",
            (1.0e-6 * (Width)/gpuTime),
            gpuTime,
            Width);

    cudaMemcpy(h_Output,d_Output,Width * sizeof(float),cudaMemcpyDeviceToHost);

    bool correct = true;
    double eps = 1.e-6;
    for(int i = 0;i<Width;i++){
        double abs_err = fabs(h_Output[i] - KERNEL_LENGTH * 1.0);

        if (abs_err > eps)
        {
            printf("Error! Index = %d,h_Output = %f,true value = %d\n", 
                           i, h_Output[i], KERNEL_LENGTH);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
}

void gpuRunTiledKernel(){

    convolution_1D_tiled_kernel<<<(Width+TILE_SIZE - 1)/TILE_SIZE,TILE_SIZE>>>(d_Input,d_Output,KERNEL_LENGTH,Width);
    
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(start,NULL);

    for(int i = 0;i<nIter;i++){
        convolution_1D_tiled_kernel<<<(Width+TILE_SIZE - 1)/TILE_SIZE,TILE_SIZE>>>(d_Input,d_Output,KERNEL_LENGTH,Width);
    }

    cudaEventRecord(stop, NULL);
    cudaDeviceSynchronize();
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal,start,stop);

    float gpuTime = (msecTotal / nIter) * 0.001;
    printf("Tiled Kernel Throughput = %0.4f MPixels/sec, Time= %.5f sec, Size= %u Pixels\n",
            (1.0e-6 * (Width)/gpuTime),
            gpuTime,
            Width);

    cudaMemcpy(h_Output,d_Output,Width * sizeof(float),cudaMemcpyDeviceToHost);

    bool correct = true;
    double eps = 1.e-6;
    for(int i = 0;i<Width;i++){
        double abs_err = fabs(h_Output[i] - KERNEL_LENGTH * 1.0);

        if (abs_err > eps)
        {
            printf("Error! Index = %d,h_Output = %f,true value = %d\n", 
                           i, h_Output[i], KERNEL_LENGTH);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
}

void gpuRunTiledCacheKernel(){

    convolution_1D_tiled_caching_kernel<<<(Width+TILE_SIZE - 1)/TILE_SIZE,TILE_SIZE>>>(d_Input,d_Output,KERNEL_LENGTH,Width);
    
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(start,NULL);

    for(int i = 0;i<nIter;i++){
        convolution_1D_tiled_caching_kernel<<<(Width+TILE_SIZE - 1)/TILE_SIZE,TILE_SIZE>>>(d_Input,d_Output,KERNEL_LENGTH,Width);
    }

    cudaEventRecord(stop, NULL);
    cudaDeviceSynchronize();
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal,start,stop);

    float gpuTime = (msecTotal / nIter) * 0.001;
    printf("Tiled Cache Kernel Throughput = %0.4f MPixels/sec, Time= %.5f sec, Size= %u Pixels\n",
            (1.0e-6 * (Width)/gpuTime),
            gpuTime,
            Width);

    cudaMemcpy(h_Output,d_Output,Width * sizeof(float),cudaMemcpyDeviceToHost);

    bool correct = true;
    double eps = 1.e-6;
    for(int i = 0;i<Width;i++){
        double abs_err = fabs(h_Output[i] - KERNEL_LENGTH * 1.0);

        if (abs_err > eps)
        {
            printf("Error! Index = %d,h_Output = %f,true value = %d\n", 
                           i, h_Output[i], KERNEL_LENGTH);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
}

int main(){
    h_Kernel = (float*)malloc(KERNEL_LENGTH * sizeof(float));
    h_Input  = (float*)malloc(Width * sizeof(float));
    h_Output = (float*)malloc(Width * sizeof(float));

    for(unsigned int i = 0;i<KERNEL_LENGTH;i++){
        h_Kernel[i] = 1.0f;
    }

    for(unsigned int i = 0;i<Width;i++){
        h_Input[i] = 1.0f;
    }

    cudaMalloc((void**)&d_Input,Width * sizeof(float));
    cudaMalloc((void**)&d_Output,Width * sizeof(float));
    cudaMalloc((void**)&d_Kernel,KERNEL_LENGTH * sizeof(float));

    cudaMemcpy(d_Input,h_Input,Width * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kernel,h_Kernel,KERNEL_LENGTH * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_M,h_Kernel,KERNEL_LENGTH * sizeof(float));

    printf("Running GPU convlution 1D ....\n");

    gpuRunBasicKernel();
    gpuRunConstantKernel();
    gpuRunTiledKernel();
    gpuRunTiledCacheKernel();

    free(h_Input);
    free(h_Kernel);
    free(h_Output);
    cudaFree(d_Input);
    cudaFree(d_Kernel);
    cudaFree(d_Output);
}

