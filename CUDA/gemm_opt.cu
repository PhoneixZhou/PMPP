//Device funtion to compute a thread block's accumulated matrix product
__device__ void block_matrix_product(int K_dim){
    //Fragmemnts used to store data fetched from SMEM
    value_t frag_a[ThreadItemsY];
    value_t frag_b[ThreadItemsX];

    //Accumulator storage
    accum_t accumulator[ThreadItemsX][ThreadItemsY];

    //GEMM mailoop - iterates over the entire K dimension - not unrolled
    for(int kblock = 0;kblock < K_dim;kblock += BlockItemsK){
        //Load A and B tiles from global memory and store to SMEM
        //
        __syncthreads();

        //Warp tile structure - iterates over the thread Block tile
        #pragma unroll 
        for(int warp_k = 0; warp_k < BlockItemsK; warp_k += WarpItemsK){
            //Fetch frag_a and frag_b from SMEM corresponding to k-index
            
            //Thread tile structure -- accumulate an outer product
            #pragma unroll 
            for(int thread_x = 0; thread_x < ThreadItemsX;++thread_x){
                #pragma unroll
                for(int thread_y = 0;thread_y < ThreadItemsY;++thread_y){
                    accumulator[thread_x][thread_y] += frag_a[y] * frag_b[x];
                }
            }
        }

        __syncthreads();
    }
}

template <
   int BlockItemsY, //Height in rows of a tile in matrix C
   int BlockItemsX, //Width in columns of a tile in matrix C
   int ThreadItemsY,//Height in rows of a thread-tile C
   int ThreadItemsX,//Width in columns of a thread-tile C
   int BlockItemsK, //Number of K-split subgroups in a block
   bool UseDounbleScratchTiles,///Whether to double buffer shared memory
   grid_raster_strategy::kind_t RasterStrategy //Grid rasterization strategy
> struct block_task_policy;

template <
   ///Parameterization of block_task_policy
   typename block_task_policy_t,
   //Multiplicand value type(matrices A and B)
   typename value_t,
   //Accumulator value type(matrix C and scalars)
   typename accum_t,

   //layout enumerant for matrix A
   matrix_transform_t::kind_t TransformA,

   //Alignment (in bytes) for A operand
   int LdgAlignA,

   //Layout enumerant for matrix B
   matrix_transform_t::kind_t TransformB,

   //Alignment (in bytes) for B operand
   int LdgAlignB,

   //Epilogue functor applied to matrix product
   typename epilogue_op_t,

   //Alignment (in bytes) for C operand
   int LdgAlignC,

   //Whether GEMM supports matrix sizes other than mult of BlockItems{XY}
   bool Ragged
>struct block_task;



//CUTLASS SGEMM example
__global__ void gemm_kernel(float* C, float const * A, float const *B, int M, int N, int K){
    //Define the GEMM tile_sizes

    typedef block_task_policy<
    128,//BlockItemsY: Height in rows of a tile
    32, //BlockItemsX - Width in columns of a tile
    8,  //ThreadItemsY - Height in rows of a thread-tile
    4,  //ThreadItemsX - Width in columns of a thread-tile
    8,  //BlockItemsK - Depth of a tile
    true, //UseDoubleScratchTiles -whether to double-buffer SMEM
    block_raster_enum::Default //Block rasterization strategy
    >block_task_policy_t;

    //Define the epilogue functor
    typedef gemm::blas_scaled_epilogue<float,float,float> epilogue_op_t;

    //Define the block task type
    typedef block_type<
        block_task_policy_t,
        float,
        float,
        matrix_transform_t::NonTranspose,
        4,
        matrix_transform_t::NonTranspose,
        4,
        epilogue_op_t,
        4,
        true,
    > block_type_t;

    __shared__ block_task_t::scratch_storage_t smem;

    //Construct and run the task
    block_task_t(
        reinterpret_cast(&smem),
        &smem,
        A,
        B,
        C,
        epilogue_op_t(1,0),
        M,
        N,
        K
    ).run();
}