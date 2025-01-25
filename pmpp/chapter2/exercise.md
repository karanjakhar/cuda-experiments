# Exercises

### 1. Mapping Thread/Block Indices to Data Index for Vector Addition
If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?

- (A) `i = threadIdx.x + threadIdx.y;`
- (B) `i = blockIdx.x + threadIdx.x;`
- (C) `i = blockIdx.x * blockDim.x + threadIdx.x;`
- (D) `i = blockIdx.x * threadIdx.x;`

<details>
<summary>Click to reveal the answer</summary>
The answer is (C).
</details>

---

### 2. Mapping Thread/Block Indices for Adjacent Elements
Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?

- (A) `i = blockIdx.x * blockDim.x + threadIdx.x + 2;`
- (B) `i = blockIdx.x * threadIdx.x * 2;`
- (C) `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`
- (D) `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`

<details>
<summary>Click to reveal the answer</summary>
The answer is (C).
</details>

---

### 3. Mapping Indices for Two Sections of Vector Addition
We want to use each thread to calculate two elements of a vector addition. Each thread block processes `2 * blockDim.x` consecutive elements that form two sections. All threads in each block will process one section first and then move to the next section. Assume variable `i` should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to the data index of the first element?

- (A) `i = blockIdx.x * blockDim.x + threadIdx.x + 2;`
- (B) `i = blockIdx.x * threadIdx.x * 2;`
- (C) `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`
- (D) `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`

<details>
<summary>Click to reveal the answer</summary>
The answer is (D).
</details>

---

### 4. Calculating Threads in the Grid for Vector Addition
For a vector addition, assume the vector length is 8000. Each thread calculates one output element, and the thread block size is 1024 threads. The kernel call configures a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?

- (A) 8000  
- (B) 8196  
- (C) 8192  
- (D) 8200  

<details>
<summary>Click to reveal the answer</summary>
The answer is (C).
</details>

---

### 5. Allocating Integer Array in CUDA Global Memory
If we want to allocate an array of `v` integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the `cudaMalloc` call?

- (A) `n`
- (B) `v`
- (C) `n * sizeof(int)`
- (D) `v * sizeof(int)`

<details>
<summary>Click to reveal the answer</summary>
The answer is (D).
</details>


---

### 6. Allocating Floating-Point Array in CUDA Global Memory
If we want to allocate an array of `n` floating-point elements and have a floating-point pointer variable `A_d` point to the allocated memory, what would be an appropriate expression for the first argument of the `cudaMalloc()` call?

- (A) `n`
- (B) `(void *) A_d`
- (C) `* A_d`
- (D) `(void **) &A_d`

<details>
<summary>Click to reveal the answer</summary>
The answer is (D).
</details>

---

### 7. Copying Data Between Host and Device in CUDA
If we want to copy 3000 bytes of data from host array `A_h` (pointer to the source array) to device array `A_d` (pointer to the destination array), what would be an appropriate API call for this data copy in CUDA?

- (A) `cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);`
- (B) `cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);`
- (C) `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`
- (D) `cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);`

<details>
<summary>Click to reveal the answer</summary>
The answer is (C).
</details>

---

### 8. Declaring a CUDA Error Variable
How would one declare a variable `err` that can appropriately receive the returned value of a CUDA API call?

- (A) `int err;`
- (B) `cudaError err;`
- (C) `cudaError_t err;`
- (D) `cudaSuccess_t err;`

<details>
<summary>Click to reveal the answer</summary>
The answer is (C).
</details>

---

### 9. Analyzing a CUDA Kernel and Host Function
#### CUDA Kernel:
```c
__global__ void foo_kernel(float* a, float* b, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        b[i] = 2.7f * a[i] - 4.3f;
    }
}
void foo(float* a_d, float* b_d) {
    unsigned int N = 200000;
    foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
}
```

Answer the following questions:

- What is the number of threads per block? -> 128
- What is the number of threads in the grid? -> 200064
- What is the number of blocks in the grid? -> 1563
- What is the number of threads that execute the code on line 02? -> 200064
- What is the number of threads that execute the code on line 04? -> 200000

--- 

### 10. Intern’s Complaint About CUDA
A new summer intern is frustrated with CUDA, complaining that it is tedious to declare many functions to execute on both the host and the device. He suggests that functions need to be declared twice: once as a host function and once as a device function. What is your response?


`**Answer:** CUDA provides the __host__ and __device__ qualifiers, which can be combined as __host__ __device__. Declaring a function with both qualifiers allows it to execute on both the host and the device. This eliminates the need to declare the function twice.

```C
__host__ __device__ float add(float a, float b) {
    return a + b;
}
```
