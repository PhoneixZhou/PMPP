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

int main(void){
    thrust::host_vector<int> h_vec(1<<24);
    thrust::generate(h_vec.begin(),h_vec.end(),rand);

    thrust::device_vector<int> d_vec = h_vec;

    thrust::sort(d_vec.begin(),d_vec.end());

    thrust::copy(d_vec.begin(),d_vec.end(),h_vec.begin());

    return 0;
}
