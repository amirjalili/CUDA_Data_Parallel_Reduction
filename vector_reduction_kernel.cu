#ifndef _SCAN_NAIVE_KERNEL_H_
#define _SCAN_NAIVE_KERNEL_H_

// **===--------------------- Modify this function -----------------------===**
//! @param g_data  input data in global memory
//                  result is expected in index 0 of g_data
//! @param n        input number of elements to reduce from input data
// **===------------------------------------------------------------------===**
__global__ void reduction(unsigned int *g_data, int n)
{
    // Allocate 2*blockDim in shared memory
    __shared__ float partialSum[2*BLOCK_SIZE];

    int t = threadIdx.x;
    int start = 2*blockDim.x*blockIdx.x;

    // Each thread loads 2 elements into the shared memory
    if (start + t < n) 
        partialSum[t] = g_data[start+t];
    else 
        partialSum[t] = 0;

    if (start + blockDim.x + t < n)    
        partialSum[blockDim.x + t] = g_data[start + blockDim.x + t];
    else 
        partialSum[blockDim.x + t] = 0;

    for(int stride = blockDim.x; stride >= 1; stride >>= 1){
        __syncthreads();
        if(t<stride)
            partialSum[t] += partialSum[t+stride];
    }
    if(t==0) 
        g_data[blockIdx.x] = partialSum[0];
	
}

#endif // #ifndef _SCAN_NAIVE_KERNEL_H_
