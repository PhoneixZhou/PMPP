#include <stdio.h>
#include <cuda.h>

#define MAX_TESS_POINTS 32 

struct BezierLine{
    float2 CP[3];//control points for the line
    float2 vertexPos[MAX_TESS_POINTS];//Vertex position array to tessellate into 
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

__device__ float computeCurvature(BezierLine* bLines){
    int bidx = blockIdx.x;

    float curvature = length(bLines[bidx].CP[1] - 0.5f * (bLines[bidx].CP[0] + bLines[bidx].CP[2]))/length(bLines[bidx].CP[2] - bLines[bidx].CP[0]);
    return curvature;
}

__global__ void computeBezierLines(BezierLine* bLines, int nLines){
    int bidx = blockIdx.x;

    if(bidx < nLines){
        //compute the curvature of the line
        float curvature = computeCurvature(bLines);

        //From the curvature, compute the number of tessellation points
        int nTessPoints = min(max((int)(curvature * 16.0f),4),32);
        bLines[bidx].nVertices = nTessPoints;

        //Loop through vertices to be tessellated, incrementing by blockDim.x
        for(int inc = 0;inc < nTessPoints;inc += blockDim.x){
            int idx = inc + threadIdx.x;//Compute a unique index for this point
            if(idx < nTessPoints){
                float u = (float)idx /(float)(nTessPoints - 1);//compute u from idx
                float omu = 1.0f - u;
                float B3u[3];
                B3u[0] = omu * omu;
                B3u[1] = 2.0f * u * omu;
                B3u[2] = u * u;
                float2 position = {0,0};
                for(int i = 0;i<3;i++){
                    position = position + B3u[i] * bLines[bidx].CP[i];
                }
                bLines[bidx].vertexPos[idx] = position;
            }
        }
    }
}

#define N_LINES 256
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
    cudaMalloc((void**)&bLines_d,N_LINES*sizeof(BezierLine));
    cudaMemcpy(bLines_d,bLines_h,N_LINES*sizeof(BezierLine),cudaMemcpyHostToDevice);

    computeBezierLines<<<N_LINES,BLOCK_DIM>>>(bLines_d,N_LINES);

    cudaFree(bLines_d);
    delete[] bLines_h;
}

