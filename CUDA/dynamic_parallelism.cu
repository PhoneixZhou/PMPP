__global__ void kernel(unsigned int* start, unsigned int* end, float* someData,float* moreData){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    doSomeWork(someData[i]);

    for(unsigned int j = start[i];j<end[i];++j){
        doMoreWork(moreData[j]);
    }
}

__global__ void kernel_parent(unsigned int* start, unsigned int* end,float* someData, float* moreData){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    doSomeWork(someData[i]);

    kernel_child<<<ceil((end[i] - start[i])/256.0,256)>>>(start[i],end[i],moreData);
}

__global__ void kernel_child(unsigned int start, unsigned int end, float* moreData){
    unsigned int j = start + blockIdx.x * blockDim.x + threadIdx.x;

    if(j<end){
        doMoreWork(moreData[j]);
    }
}

