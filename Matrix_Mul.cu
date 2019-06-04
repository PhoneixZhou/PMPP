
#include <stdio.h>
#include <cuda.h>
__global__ void MatrixMulKernel(float* M, float* N,float * P,int width,int height,int one_stripe){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if((Row<height) && (Col < width)){
        float Pvalue = 0;
        for(int k = 0;k<one_stripe;k++){
            Pvalue += M[Row*one_stripe + k] * N[k * width + Col];
        }
        P[Row* width + Col] = Pvalue; 
    }
}
//opt_1 using shared memory
//d_M:m * k ,d_N:k*n,d_P = m * n;
//d_P = d_M * d_N
template <int TILE_WIDTH>
__global__ void MatrixMulKernel_Shared(float* d_M,float* d_N,float* d_P,int n,int m,int k){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x;int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for(int ph = 0;ph < ceil(k/(float)TILE_WIDTH);ph++){
        if((Row < m) && ((ph * TILE_WIDTH + tx)<k))
           Mds[ty][tx] = d_M[Row * k + ph * TILE_WIDTH + tx];
        if(((ph * TILE_WIDTH + ty)< k)&& Col < n)
           Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty) * n + Col];
      
      __syncthreads();

      for(int k = 0;k <TILE_WIDTH;++k){
        Pvalue += Mds[ty][k] * Nds[k][tx];
      }
      __syncthreads();
    }
    if((Row<m) && (Col < n))
    d_P[Row * n + Col] = Pvalue;
} 

//opt_2 thread granularity : one can eliminate this redundancy by merging the two thread blocks into one. 
//Each thread in the new thread block now calculates two P elements. 


void constantInit(float* data,int size, float val){
    for(int i = 0;i<size;++i){
        data[i] = val;
    }
}

int main(){
    int m = 1024;
    int n = 1024;
    int k = 1024;
    unsigned int sizeA = m * k;
    unsigned int mem_size_A = sizeof(float) * sizeA;
    float* h_A = (float*)malloc(mem_size_A);
    unsigned int sizeB = k * n;
    unsigned int mem_size_B = sizeof(float) * sizeB;
    float* h_B = (float*)malloc(mem_size_B);
    unsigned int mem_size_C = m * n * sizeof(float);
    float* h_C = (float*)malloc(mem_size_C);

    const float valB = 0.01f;
    const float valA = 1.0f;

    constantInit(h_A,sizeA,valA);
    constantInit(h_B,sizeB,valB);

    float* d_A,*d_B,*d_C;
    cudaMalloc((void**)&d_A,mem_size_A);
    cudaMalloc((void**)&d_B,mem_size_B);
    cudaMalloc((void**)&d_C,mem_size_C);

    cudaMemcpy(d_A,h_A,mem_size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,mem_size_B,cudaMemcpyHostToDevice);

    int block_size = 32;
    dim3 threads(block_size,block_size);
    dim3 grid((n + block_size-1)/block_size,(m + block_size-1)/block_size);

    //warp up
    //MatrixMulKernel<<<grid,threads>>>(d_A,d_B,d_C,n,m,k);
    MatrixMulKernel_Shared<32><<<grid,threads>>>(d_A,d_B,d_C,n,m,k);

    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEventCreate(&start);

    cudaEvent_t stop;
    cudaEventCreate(&stop);

    cudaEventRecord(start,NULL);

    int nIter = 300;
    for(int j = 0;j<nIter;j++){
        //MatrixMulKernel<<<grid,threads>>>(d_A,d_B,d_C,n,m,k);
        MatrixMulKernel_Shared<32><<<grid,threads>>>(d_A,d_B,d_C,n,m,k);
    }

    cudaEventRecord(stop, NULL);
    cudaDeviceSynchronize();

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal,start,stop);

    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * n * m * k;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f)/(msecPerMatrixMul/1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul,
            block_size * block_size);

    cudaMemcpy(h_C,d_C,mem_size_C,cudaMemcpyDeviceToHost);

    bool correct = true;
    double eps = 1.e-6;
    for(int i = 0;i<(m * n);i++){
        double abs_err = fabs(h_C[i] - (k * valB));
        double dot_length = k;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/dot_length ;

        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], k*valB, eps);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}