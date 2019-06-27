float alpha = 1.0f;
float beta = 0.0f;
int lda = k;
int ldb = n / numGPU;
int ldc = n;

for(int id = 0;id < numGPU;id++){
    cudaSetDevice(id);
    cublasSetStream(handle[id],streams[id][0]);
    cublasSgemm(handle[id],CUBLAS_OP_N,CUBLAS_OP_N,n / numGPU,m/numGPU,k,&alpha,d_b[id][0],ldb,d_a[id],lda,&beta,c[id] + id*(n/numGPU),ldc);
    for(int offset = 1;offset < numGPU;offset++){
        int srcDevice = (id + offset) % numGPU;
        cudaMemcpyPeerAsync(d_b[id][offset],id,d_b[srcDevice][0],srcDevice,n/numGPU * k * sizeof(float),streams[id][offset]);
    }

    for(int offset = 1;offset < numGPU;offset++){
        int srcDevice = (id + offset)%numGPU;
        cublasSetStream(handle[id],streams[id][offset]);
        cublasSgemm(handle[id],CUBLAS_OP_N,CUBLAS_OP_N,n/numGPU,m/numGPU,k,&alpha,b_d[id][offset],ldb,d_a[id],lda,&beta,d_c[id]+srcDevice * (n/numGPU),ldc);
    }
}

for(int id = 0;id < numGPU;id++){
    cudaSetDevice(id);
    cudaDeviceSynchronize();
}