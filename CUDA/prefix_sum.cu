
#include <cuda.h>

const uint THREADBLOCK_SIZE = 256;
const uint MIN_SHORT_ARRAY_SIZE = 4;
const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;

void sequential_scan(float *x, float* y, int Max_i){
    int accumulator = x[0];
    y[0] = accumulator;
    for(int i = 1;i<Max_i;i++){
        accumulator += x[i];
        y[i] = accumulator;
    }
}

//Kogge-Stone Kernel for inclusive scan
__global__ void Kogge_Stone_scan_kernel(float * X, float* Y, int InputSize){
    __shared__ float XY[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < InputSize){
        XY[threadIdx.x] = X[i];
    }

    //the code below performs iterative scan on XY
    for(unsigned int stride = 1;stride < blockDim.x; stride *=2){
        __syncthreads();

        if(threadIdx.x >= stride)XY[threadIdx.x] += XY[threadIdx.x - stride];
    }
    Y[i] = XY[threadIdx.x];
}
//Brent-Kung kernel for inclusive scan
__global__ void Brent_Kung_scan_kernel(float* X, float* Y, int InputSize){
    __shared__ float XY[SECTION_SIZE_2];

    int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if(i < InputSize)XY[threadIdx.x] = X[i];
    if(i + blockDim.x < InputSize) XY[threadIdx.x + blockDim.x] = X[i + blockDim.x];

    for(unsigned int stride = 1;stride <= blockDim.x;stride *=2){
        __syncthreads();
        int index = (threadIdx.x+1) * 2 * stride - 1;
        if(index < SECTION_SIZE_2){
            XY[index] += XY[index-stride];
        }
    }

    for(int stride = SECTION_SIZE_2/4;stride > 0 ;stride /=2){
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride -1;
        if(index + stride < SECTION_SIZE_2){
            XY[index+stride] += XY[index];
        }
    }

    __syncthreads();

    if(i<InputSize)Y[i] = XY[threadIdx.x];
    if(i + blockDim.x < InputSize) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

//3 phases kernel

int main(){
    uint *d_Input,*d_Ouput;
    uint *h_Input,*h_Output,*h_OutputGPU;
    const uint N = 13 * 1048576/2;

    h_Input     = (uint*)malloc(N*sizeof(uint));
    h_OutputCPU = (uint*)malloc(N*sizeof(uint));
    h_OutputGPU = (uint*)malloc(N*sizeof(uint));
    srand(2019);

    for(uint i = 0;i<N;i++){
        h_Input[i] = rand();
    }
    cudaMalloc((void**)&d_Input,N*sizeof(uint));
    cudaMalloc((void**)&d_Output,N*sizeof(uint));
    cudaMemcpy(d_Input,h_Input,N*sizeof(uint),cudaMemcpyHostToDevice);

    const int nIter = 100;
    printf("Running GPU scan for short arrays (%d identical iterations)....\n",nIter);

    for(uint arrayLength = MIN_SHORT_ARRAY_SIZE;arrayLength <= MAX_SHORT_ARRAY_SIZE;arrayLength<<1){
        printf("running scan for %u elements (%u arrays)...\n",arrayLength, N / arrayLength);

        cudaDeviceSynchronize();

        cudaEvent_t start;
        cudaEventCreate(&start);
        cudaEvent_t stop;
        cudaEventCreate(&stop);
        cudaEventRecord(start,NULL);

        for(int i = 0;i<nIter;i++){

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



    }



}
