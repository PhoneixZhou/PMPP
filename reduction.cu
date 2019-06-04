__global__ void reduction_1(float* X,float* Sum){
    __shared__ float partialSum[SIZE];
    partialSum[threadIdx.x] = X[blockIdx.x * blockDim.x + threadIdx.x];
    unsigned int t = threadIdx.x;

    for(unsigned int stride = 1;stride < blockDim.x;stride *=2){
        __syncthreads();

        if(t%(2*stride) == 0){
            partialSum[t] += partialSum[t+stride];
        }
    }
    Sum[blockIdx.x] = partialSum[0];
}

__global__ void reduction_2(float* X,float* Sum){
    __shared__ float partialSum[SIZE];
    partialSum[threadIdx.x] = X[blockIdx.x * blockDim.x + threadIdx.x];

    unsigned int t = threadIdx.x;
    for(unsigned int stride = blockDim.x/2;stride >= 1;stride = stride >>1){
        __syncthreads();
        if(t<stride){
            partialSum[t] += partialSum[t + stride];
        }
    }
    Sum[blockIdx.x] = partialSum[0];
} 

__global__ void reduction_3(float* X,float *Sum){
    __shared__ float partialSum[SIZE*2];
    partialSum[threadIdx.x] = X[blockIdx.x * blockDim.x + threadIdx.x];
    partialSum[blockDim.x + threadIdx.x] = X[(blockIdx.x+1)*blockDim.x + threadIdx.x];

    unsigned int t = threadIdx.x;
    for(unsigned int stride = blockDim.x;stride >=1;stride = stride >>1){
        __syncthreads();
        if(t < stride){
            partialSum[t] += partialSum[t + stride];
        }
    }
    Sum[blockIdx.x] = partialSum[0];
}