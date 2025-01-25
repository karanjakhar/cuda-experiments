#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}


void print_array(float * arr, int num_elements){
    

    for (int i = 0; i < num_elements; i++){
        printf("%f\t",arr[i]);
    }
    printf("\n");
}

__global__
void reluKernelShared(float *d_in, float *d_out, int size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ float s_data[];
    
    if(idx < size){
        s_data[threadIdx.x] = d_in[idx];
    }
    __syncthreads();

    if(idx < size){
        s_data[threadIdx.x] = (s_data[threadIdx.x] > 0) ? s_data[threadIdx.x]: 0; 
    }
    __syncthreads();

    if(idx < size){
        d_out[idx] = s_data[threadIdx.x];
    }
}

void check_relu(float *arr, int size){
    int count_zeros = 0;
    int count_non_zeros = 0;
    for(int i = 0; i < size; i++){
        
        if(arr[i] == 0){
            count_zeros++;
        }
        else if(arr[i] > 0){
            count_non_zeros++;
        }
        if(arr[i] < 0){
            printf("ReLU check failed at %d, as value is %f\n", i, arr[i]);
            return;
        }

    }
    printf("ReLU check passed!!, zeros: %d, more than zero: %d\n", count_zeros, count_non_zeros);

}




/*

The function uses the Box-Muller transform to convert two 
uniformly distributed random numbers (u1 and u2) into a 
normally distributed number (num) with mean 0 and std 1.


*/
float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}


int main(){
    
    


    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
   
    int rows = 1024, cols = 1024;
    int matrixSize = rows * cols;
    size_t totalBytes = matrixSize * sizeof(float);
    float * inputMatrix = (float*)malloc(totalBytes);
    float * resultMatrix = (float*)malloc(totalBytes);

    if (inputMatrix == NULL){
        printf("Memory allocation failed\n");
        return;
    }


    printf("Num elements:%d\n", matrixSize);
    
    for(int i = 0; i < matrixSize; i++){
        inputMatrix[i] = random_normal_clamped(-10, -5);
    }
    
    

    float * inputMatrix_d, * resultMatrix_d;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&inputMatrix_d, totalBytes));
    CUDA_CHECK(cudaMalloc(&resultMatrix_d, totalBytes));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU allocation time: %f ms\n", ms);

    cudaEventRecord(start);
    cudaMemcpy(inputMatrix_d, inputMatrix, totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(resultMatrix_d, inputMatrix, totalBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Host to device transfer time: %f ms\n", ms);

    cudaEventRecord(start);
    int threadsPerBlock = maxThreadsPerBlock;
    int blocksPerGrid = 1024;
    printf("threadsPerBlock: %d\n", threadsPerBlock);
    size_t shared_memory_size = threadsPerBlock * sizeof(float);
    reluKernelShared<<<blocksPerGrid,threadsPerBlock,shared_memory_size>>>(inputMatrix_d,resultMatrix_d, matrixSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f ms\n", ms);
    
    
    cudaEventRecord(start);
    cudaMemcpy(resultMatrix, resultMatrix_d, totalBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Device to host transfer time: %f ms\n", ms);





    check_relu(resultMatrix, matrixSize);

    free(inputMatrix);
    free(resultMatrix);
    cudaFree(inputMatrix_d);
    cudaFree(resultMatrix_d);

    return 0;
}
