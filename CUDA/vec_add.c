#include <cuda.h>
void checkCudaError(cudaError_t err,int32_t __file__,int32_t __line__){
    if(err != cudaSuccess){
        printf(“%s in %s at line %d\n”,cudaGetErrorString(err),__file__,__line__);
        exit(EXIT_FAILURE);
    }
}

__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<n)C[i] = A[i] + B[i];
}

void vecAdd(float* h_A,float* h_B,float* h_C, int n){
    int size = n * sizeof(float);
    float* d_A,*d_B,*d_C;
    cudaError_t err;
    err = cudaMalloc((void**)&d_A,size);
    checkCudaError(err,__FILE__,__LINE__);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B,size);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_C,size);

    vecAddKernel<<<ceil(n/256.0),256>>>(d_A,d_B,d_C,n);

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}