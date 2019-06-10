void squential_spmv_csr(){
    for(int row = 0;row < num_rows;row++){
        float dot = 0;
        int row_start = row_ptr[row];
        int row_end   = row_ptr[row+1];
    
        for(int elem = row_start;elem < row_end;elem++){
            dot += data[elem] * x[col_index[elem]];
        }
        y[row] += dot;
    }
}

__global__ void SpMV_CSR(int num_rows, float *data,int *col_index, int *row_ptr, float *x, float *y){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < num_rows){
        float dot = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for(int elem = row_start;elem< row_end;elem++){
            dot += data[elem] * x[col_index[elem]];
        }
        y[row] += dot;
    }
}


__global__ void SpMV_ELL(int num_rows, float* data, int* col_index, int num_elem, float* x, float* y){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < num_rows){
        float dot = 0;
        for(int i = 0;i<num_elem;i++){
            dot += data[row + i * num_rows] * x[col_index[row + i * num_rows]];
        }
        y[row] += dot;
    }
}