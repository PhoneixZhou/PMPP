#define MAX_TESS_POINTS 32 
struct BezierLine{
    float2 CP[3];//control points for the line
    float2 *vertexPos;//Vertex position array to tessellate into 
                                      //Number of tessellated vertices
    int nVertices;
};

__forceinline__ __device__ float2 operator+(float2 a,float2 b){
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__forceinline__ __device__ float2 operator-(float2 a, float2 b){
    float2 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

__forceinline__ __device__ float2 operator*(float a, float2 b){
    float2 c;
    c.x = a * b.x;
    c.y = a * b.y;
    return c;
} 

__forceinline__ __device__ float length(float2 a){
    return sqrtf(a.x * a.x + a.y * a.y);
}

__global__ void computeBezierLine_child(int lidx, BezierLine* bLines, int nTessPoints){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < nTessPoints){
        float u = (float)idx/(float)(nTessPoints - 1);
        float omu = 1.0f - u;
        float B3u[3];
        B3u[0] = omu * omu;
        B3u[1] = 2.0f * u * omu;
        B3u[2] = u * u;
        float2 position = {0,0};
        for(int i = 0;i<3;i++){
            position = position + B3u[i] * bLines[lidx].CP[i];
        }
        bLines[lidx].vertexPos[idx] = position;
    }
}

__device__ float computeCurvature(BezierLine* bLines){
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;

    float curvature = length(bLines[lidx].CP[1] - 0.5f * (bLines[lidx].CP[0] + bLines[lidx].CP[2]))/length(bLines[lidx].CP[2] - bLines[lidx].CP[0]);
    return curvature;
}

__global__ void computeBezierLines_parent(BezierLine * bLines, int nLines){
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;
    if(lidx < nLines){
        float curvature = computeCurvature(bLines);

        bLines[lidx].nVertices = min(max((int)(curvature * 16.0f),4),MAX_TESS_POINTS);
        cudaMalloc((void**)&bLines[lidx].vertexPos,bLines[lidx].nVertices*sizeof(float2));

        //computeBezierLine_child<<<ceil((float)bLines[lidx].nVertices/32.0f),32>>>(lidx,bLines,bLines[lidx].nVertices);
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
        computeBezierLine_child<<<ceil((float)bLines[lidx].nVertices/32.0f),32,0,stream>>>(lidx,bLines,bLines[lidx].nVertices);

        cudaStreamDestroy(stream);
    }
}



__global__ void freeVertexMem(BezierLine* bLines, int nLines){
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;
    if(lidx < nLines)
    cudaFree(bLines[lidx].vertexPos);
}

#define N_LINES 4096
#define BLOCK_DIM 32

void initializeBLines(BezierLine * bLines_h){
    float2 last = {0,0};
    for(int i = 0;i<N_LINES;i++){
        bLines_h[i].CP[0] = last;
        for(int j = 1;j<3;j++){
            bLines_h[i].CP[j].x = (float)rand()/(float)RAND_MAX;
            bLines_h[i].CP[j].y = (float)rand()/(float)RAND_MAX;
        }
        last = bLines_h[i].CP[2];
        bLines_h[i].nVertices = 0;
    }
}

int main(){
    BezierLine * bLines_h = new BezierLine[N_LINES];
    initializeBLines(bLines_h);

    BezierLine * bLines_d;
    cudaMalloc((void**)&bLines_d,N_LINES * sizeof(BezierLine));
    cudaMemcpy(bLines_d,bLines_h,N_LINES * sizeof(BezierLine),cudaMemcpyHostToDevice);

    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount,N_LINES);
    computeBezierLines_parent<<<ceil((float)N_LINES/(float)BLOCK_DIM),BLOCK_DIM>>>(bLines_d,N_LINES);

    freeVertexMem<<<ceil((float)N_LINES/(float)BLOCK_DIM),BLOCK_DIM>>>(bLines_d,N_LINES);
    cudaFree(bLines_d);

    delete[] bLines_h;
}