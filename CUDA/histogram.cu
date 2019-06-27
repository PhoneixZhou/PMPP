#include <stdint.h>
#include <stdio.h>
#define HISTOGRAM256_BIN_COUNT 256
#define HISTOGRAM256_THREADBLOCK_SIZE 1024
static const uint PARTIAL_HISTOGRAM256_COUNT = 240;
void sequential_Histogram(uint8_t * data,int length, uint32_t * histo){
    for(int i = 0;i<length;i++){
        int alphabet_position = data[i];
        histo[alphabet_position]++;
    }
}

__global__ void histo_kernel(unsigned char* input,long size, unsigned int* histo){
    int i  = threadIdx.x + blockIdx.x * blockDim.x;

    int threadNum = blockDim.x * gridDim.x;
    int section_size = (size+threadNum - 1)/threadNum;
    int start = i * section_size;

    //All threads handle blockDim.x * gridDim.x consecutive elements
    for(int k = 0;k < section_size;k++){
        if(start+k < size){
            int alphabet_position = input[start + k];
            atomicAdd(&(histo[alphabet_position]),1);
        }
    }
}

__global__ void histo_kernel_interleave(unsigned char* input,long size, unsigned int* histo){
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for(unsigned int i = tid;i<size;i+=blockDim.x * gridDim.x){
        int alphabet_position = input[i];
        atomicAdd(&(histo[alphabet_position]),1);
    }
}

__global__ void histogram_privatized_kernel(unsigned char* input, unsigned int* bins, unsigned int num_elements,unsigned int num_bins){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ unsigned int histo_s[];
    for(unsigned int binIdx = threadIdx.x;binIdx<num_bins;binIdx += blockDim.x){
        histo_s[binIdx] = 0u;
    }
    __syncthreads();

    for(unsigned int i = tid;i < num_elements;i+= blockDim.x * gridDim.x){
        int alphabet_position = input[i];
        atomicAdd(&(histo_s[alphabet_position]),1);
    }

    __syncthreads();

    for(unsigned int binIdx = threadIdx.x ;binIdx < num_bins;binIdx += blockDim.x){
        atomicAdd(&(bins[binIdx]),histo_s[binIdx]);
    }
}

__global__ void histogram_privatized_aggregation_kernel(unsigned char* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ unsigned int histo_s[];

    for(unsigned int binIdx = threadIdx.x;binIdx < num_bins;binIdx += blockDim.x){
        histo_s[binIdx] = 0u;
    }
    __syncthreads();

    unsigned int prev_index  = 0;
    unsigned int accumulator = 0;

    for(unsigned int i = tid;i<num_elements;i+= blockDim.x * gridDim.x){
        int alphabet_position = input[i];
        unsigned int cur_index = alphabet_position;
        if(cur_index != prev_index){
                if(accumulator >= 0)atomicAdd(&(histo_s[alphabet_position]),accumulator);
                accumulator = 1;
                prev_index = cur_index;
        }else{
            accumulator++;
        }
    }

    __syncthreads();
    for(unsigned int binIdx = threadIdx.x ;binIdx < num_bins;binIdx += blockDim.x){
        atomicAdd(&(bins[binIdx]),histo_s[binIdx]);
    }
}

int main(){
    int PassFailFlag = 1;
    uint8_t  *h_Data;
    uint32_t *h_HistogramCPU,*h_HistogramGPU;
    uint8_t  *d_Data;
    uint32_t *d_Historgram;

    uint byteCount = 128 * 1048576;
    int nIter = 1;

    h_Data = (uint8_t*)malloc(byteCount);
    h_HistogramCPU = (uint32_t *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint32_t));
    h_HistogramGPU = (uint32_t *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint32_t));

    printf("...generating input data\n");
    srand(2019);

    for(uint32_t i = 0;i<byteCount;i++){
        h_Data[i] = rand() % 256;
    }
    memset(h_HistogramCPU,0,sizeof(uint32_t)*HISTOGRAM256_BIN_COUNT);

    printf("...allocating GPU memory and copying input data\n");
    cudaMalloc((void**)&d_Data,byteCount);
    cudaMalloc((void**)&d_Historgram,HISTOGRAM256_BIN_COUNT* sizeof(uint));

    cudaMemcpy(d_Data,h_Data,byteCount,cudaMemcpyHostToDevice);

    //histo_kernel<<<PARTIAL_HISTOGRAM256_COUNT,HISTOGRAM256_THREADBLOCK_SIZE>>>(d_Data,byteCount,d_Historgram);

    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(start,NULL);

    {    
        //for(int iter = 0;iter < nIter;iter++){
        //    cudaMemset(d_Historgram,0,byteCount);
        //histo_kernel<<<PARTIAL_HISTOGRAM256_COUNT,HISTOGRAM256_THREADBLOCK_SIZE>>>(d_Data,byteCount,d_Historgram);
        //histo_kernel_interleave<<<PARTIAL_HISTOGRAM256_COUNT,HISTOGRAM256_THREADBLOCK_SIZE>>>(d_Data,byteCount,d_Historgram);
        histogram_privatized_kernel<<<PARTIAL_HISTOGRAM256_COUNT,HISTOGRAM256_THREADBLOCK_SIZE,256 * sizeof(int)>>>(d_Data,d_Historgram,byteCount,256);
        //histogram_privatized_aggregation_kernel<<<PARTIAL_HISTOGRAM256_COUNT,HISTOGRAM256_THREADBLOCK_SIZE,256 * sizeof(int)>>>(d_Data,d_Historgram,byteCount,256);
        //}
    }

    cudaEventRecord(stop, NULL);
    cudaDeviceSynchronize();
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal,start,stop);

    float gpuTime = (msecTotal / nIter) * 0.001;
    printf("histogram basic time = %.5f sec, %.4f MB/sec \n",gpuTime,(byteCount*1e-6)/gpuTime);

    cudaMemcpy(h_HistogramGPU,d_Historgram,HISTOGRAM256_BIN_COUNT * sizeof(int),cudaMemcpyDeviceToHost);

    sequential_Histogram(h_Data,byteCount,h_HistogramCPU);
    for(uint i = 0;i<HISTOGRAM256_BIN_COUNT;i++){
        if(h_HistogramCPU[i] != h_HistogramGPU[i]){
            PassFailFlag = 0;

            printf("index i = %d, CPU = %d, GPU = %d\n",i,h_HistogramCPU[i],h_HistogramGPU[i]);
        }
    }

    printf(PassFailFlag ? " ...histograms match\n\n" : " ***histograms do not match!!!***\n\n");

    cudaFree(d_Historgram);
    cudaFree(d_Data);
    free(h_HistogramCPU);
    free(h_HistogramGPU);
    free(h_Data);
}



