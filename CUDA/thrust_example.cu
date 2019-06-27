#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

#include <iostream>
// int main(){
//     thrust::host_vector<int> H(4);

//     H[0] = 14;
//     H[1] = 20;
//     H[2] = 38;
//     H[3] = 46;

//     std::cout<<"H has size "<< H.size()<<std::endl;

//     for(int i = 0;i<H.size();i++)
//         std::cout<<"H[" << i << "] = " << H[i] <<std::endl;
    
//     H.resize(2);

//     std::cout<< "H now has size "<< H.size() << std::endl;

//     thrust::device_vector<int> D = H;
//     D[0] = 99;
//     D[1] = 88;

//     for(int i = 0;i<D.size();i++)
//     std::cout<<"D[" << i << "] = " << D[i]<<std::endl;

//     return 0;
// }

// int main(void){
//     thrust::host_vector<int> h_vec(1<<24);
//     thrust::generate(h_vec.begin(),h_vec.end(),rand);

//     thrust::device_vector<int> d_vec = h_vec;

//     thrust::sort(d_vec.begin(),d_vec.end());

//     thrust::copy(d_vec.begin(),d_vec.end(),h_vec.begin());

//     return 0;
// }

//Interfacing thrust to CUDA
// int main(){
//     size_t N = 1024;

//     device_vector<int> d_vec(N);

//     int raw_ptr = raw_pointer_cast(&d_vec[0]);

//     cudaMemset(raw_ptr,0,N,sizeof(int));

//     my_kernel<<<N/128,128>>>(N,raw_ptr);

//     //memory is automatically freed.
// }

//Interfacing CUDA to thrust
int main(){
    size_t N = 1024;

    //raw pointer to device memory
    int raw_ptr;
    cudaMalloc(&raw_ptr,N * sizeof(float));

    //warp raw pointer with a device ptr
    device_ptr<int> dev_ptr = device_pointer_cast(raw_ptr);

    //Use device ptr in Thrust alogithms 
    sort(dev_ptr,dev_ptr + N);

    //access device memory through device ptr
    dev_ptr[0] = 1;
    cudaFree(raw_ptr);
}
