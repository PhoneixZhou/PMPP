void merge_sequential(int* A, int m, int* B, int n, int *C){
    int i = 0;
    int j = 0;
    int k = 0;

    while((i<m) && (j<n)){
        if(A[i] <= B[i]){
            C[k++] = A[i++];
        }else{
            C[k++] = B[j++];
        }
    }

    if(i == m){
        for(;j<n;j++){
            C[k++] = B[j];
        }
    }else{
        for(;i<m;i++){
            C[k++] = A[i];
        }
    }
}

int co_rank(int k, int* A, int m,int * B, int n){
    int i = k < m ? k:m;
    int j = k - i;
    int i_low = 0 > (k-n) ? 0:k - n;
    int j_low = 0 > (k-m) ? 0:k - m;

    int delta;
    bool active = true;

    while(active){
        if(i > 0 && j < n && A[i-1] > B[j]){
            delta = ((i - i_low + i)>>1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }else if(j > 0 && i < m && B[j-1] >= A[i]){
            delta = ((j-j_low + 1)>>1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }else{
            active = false;
        }
    }
    return i;
}


__global__ void merge_basic_kernel(int *A, int m, int* B,int n, int* C){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int k_curr = tid *ceil((m+n)/(blockDim.x * gridDim.x));
    int k_next = min((tid + 1) * ceil((m+n)/(blockDim.x * gridDim.x)),m+n);
    int i_curr = co_rank(k_curr,A,m,B,n);
    int i_next = co_rank(k_next,A,m,B,n);

    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr],i_next - i_curr,&B[j_curr],j_next - j_curr,&C[k_curr]);
}

__global__ void merge_tiled_kernel(int* A, int m, int * B, int n, int* C, int tile_size){
    extern __shared__ int shareAB[];

    int* A_S = &shareAB[0];
    int* B_S = &shareAB[tile_size];

    int C_curr = blockIdx.x * ceil((m+n)/gridDim.x);
    int C_next = min((blockIdx.x + 1) * ceil((m+n)/gridDim.x),m+n);

    if(threadIdx.x == 0){
        A_S[0] = co_rank(C_curr,A,m,B,n);
        A_S[1] = co_rank(C_next,A,m,B,n);
    }

    _syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    _syncthreads();


    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil((C_length)/tile_size));

    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    while(counter < total_iteration){
        for(int i = 0;i<tile_size;i+= blockDim.x){
            if(i + threadIdx.x < A_length - A_consumed){
                A_S[i+threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }

        for(int i = 0;i<tile_size;i+= blockDim.x){
            if(i+threadIdx.x < B_length - B_consumed){
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size/blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size/blockDim.x);

        c_curr = (c_curr <= C_length - C_completed) ? c_curr:C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next:C_length - C_completed;

        int a_curr = co_rank(c_curr,A_S,min(tile_size,A_length - A_consumed),B_S,min(tile_size,B_length - B_consumed));
        int b_curr = c_curr - a_curr;
        int a_next = co_rank(c_next,A_S,min(tile_size,A_length - A_consumed),B_S,min(tile_size,B_length - B_consumed));
        int b_next = c_next - a_next;

        merge_sequential(A_S + a_curr,a_next - a_curr,B_S + b_curr,b_next - b_curr,C+C_curr + C_completed+c_curr);
        counter ++ ;
        C_completed += tile_size;
        A_consumed += co_rank(tile_size,A_S,tile_size,B_S,tile_size);
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
}

int co_rank_circular(int k,int * A,int m, int * B,int n, int A_S_start,int B_S_start,int tile_size){
    int i = k < m ? k:m;
    int j = k - i;
    int i_low = 0 > (k-n) ? 0 :k - n;
    int j_low = 0 > (k-m) ? 0 : k - m;
    int delta;
    bool active = true;

    while(active){
        int i_cir = (A_S_start + i>= tile_size) ? A_S_start + i - tile_size : A_S_start + i;
        int i_m_1_cir = (A_S_start + i - 1>= tile_size) ? A_S_start + i - tile_size : A_S_start + i - 1;
        int j_cir = (B_S_start + j >= tile_size) ? B_S_start + j - tile_size : B_S_start + j;
        int j_m_1_cir = (B_S_start + i - 1 >= tile_size) ? B_S_start + j - 1 - tile_size :B_S_start + j - 1;

        if(i > 0 && j < n && A[i_m_1_cir] > B[j_cir]){
            delta = ((i-i_low + 1)>>1);
            j_low = j;
            i = i -delta;
            j = j + delta;
        }else if(j > 0 && i< m && B[j_m_1_cir] >= A[i_cir]){
            delta = ((j-j_low + 1)>>1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }else{
            active =false;
        }
    }
    return i;
}

void merge_sequential_circular(int * A,int m ,int * B,int n,int* C, int A_S_start,int B_S_start, int tile_size){
    int i = 0;
    int j = 0;
    int k = 0;

    while((i<m) && (j<n)){
        int i_cir = (A_S_start + i >= tile_size) ? A_S_start + i - tile_size : A_S_start + i;
        int j_cir = (B_S_start + j >= tile_size) ? B_S_start + j - tile_size : B_S_start + j;

        if(A[i_cir] <= B[j_cir]){
            C[k++] = A[i_cir];i++;
        }else{
            C[k++] = B[j_cir];j++;
        }
    }

    if(i == m){
        for(;j<n;j++){
            int j_cir = (B_S_start + j>= tile_size) ? B_S_start + j - tile_size : B_S_start + j;
            C[k++] = B[j_cir];
        }
    }else{
        for(;i<m;i++){
            int i_cir = (A_S_start + i >= tile_size) ? A_S_start + i - tile_size : A_S_start + i;
            C[k++] = A[i_cir];
        }
    }
}

__global__ void merge_circular_buffer_kernel(int* A, int m, int * B, int n, int* C, int tile_size){
    extern __shared__ int shareAB[];

    int* A_S = &shareAB[0];
    int* B_S = &shareAB[tile_size];

    int C_curr = blockIdx.x * ceil((m+n)/gridDim.x);
    int C_next = min((blockIdx.x + 1) * ceil((m+n)/gridDim.x),m+n);

    if(threadIdx.x == 0){
        A_S[0] = co_rank(C_curr,A,m,B,n);
        A_S[1] = co_rank(C_next,A,m,B,n);
    }

    _syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    _syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil((C_length)/tile_size));

    int A_consumed = 0;
    int B_consumed = 0;

    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = tile_size;
    int B_S_consumed = tile_size;

    while(counter < total_iteration){
        //loading A_S_consumed elements into A_S
        for(int i = 0;i<A_S_consumed;i+= blockDim.x){
            if(i + threadIdx.x < A_length - A_consumed && i + threadIdx.x < A_S_consumed){
                A_S[(A_S_start + i + threadIdx.x) % tile_size] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }

        //loading B_S_consumed elements into B_S
        for(int i = 0;i<B_S_consumed;i+= blockDim.x){
            if(i + threadIdx.x < B_length - B_consumed && i + threadIdx.x < B_S_consumed){
                B_S[(B_S_start + i + threadIdx.x) % tile_size] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }

        int c_curr = threadIdx.x * (tile_size/blockDim.x);
        int c_next = (threadIdx.x + 1) *(tile_size/blockDim.x);

        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

        //find co_rank for c_curr and c_next
        int a_curr = co_rank_circular(c_curr,
                                      A_S,min(tile_size,A_length - A_consumed),
                                      B_S,min(tile_size,B_length - B_consumed),
                                      A_S_start,B_S_start,tile_size);

        int b_curr = c_curr - a_curr;
        int a_next = co_rank_circular(c_next,
                                      A_S,min(tile_size,A_length - A_consumed),
                                      B_S,min(tile_size,B_length - B_consumed),
                                      A_S_start,B_S_start,tile_size);
        int b_next = c_next - a_next;

        //All threads call the circular-buffer version of the sequential merge function
        merge_sequential_circular(A_S,a_next - a_curr,
                                  B_S,b_next - b_curr,C+C_curr + C_completed+C_curr, 
                                  A_S_start + a_curr,B_S_start + b_curr,tile_size);
        //Figure out the work has been done
        counter++;
        A_S_consumed = co_rank_circular(min(tile_size,C_length - C_completed),
                                        A_S,min(tile_size,A_length - A_consumed),
                                        B_S,min(tile_size,B_length - B_consumed),
                                        A_S_start,B_S_start,tile_size);
        B_S_consumed = min(tile_size,C_length - C_completed) - A_S_consumed;
        A_consumed += A_S_consumed;
        C_completed += min(tile_size,C_length - C_completed);
        B_consumed = C_completed - A_consumed;

        A_S_start = A_S_start + A_S_consumed;
        if(A_S_start >= tile_size) A_S_start = A_S_start - tile_size;
        B_S_start = B_S_start + B_S_consumed;
        if(B_S_start >= tile_size) B_S_start = B_S_start - tile_size;

        __syncthreads();
    }


}



int mian(){
    uint* h_SrcVal,*h_DstVal;
    uint* d_SrcVal,*d_DstVal;

    const uint N = 4 * 1048576;
    const uint numValues = 65536;
    
    h_SrcVal = (uint*)malloc(N * sizeof(uint));
    h_DstVal = (uint*)malloc(N * sizeof(uint));

    srand(2019);

    for(uint i = 0;i<N;i++){
        h_SrcVal[i] = rand() % numValues;
    }

    cudaMalloc((void**)&d_SrcVal,N * sizeof(uint));
    cudaMalloc((void**)&d_DstVal,N * sizeof(uint));

    cudaMemcpy(d_SrcVal,h_SrcVal,N * sizeof(uint),cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    mergeSort(d_DstVal,d_SrcVal,N);




}