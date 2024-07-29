#include<iostream>
#include<chrono>
#include"helper.h"

#define matSize 1024

__global__
void mmNaive(float *A, float *B, float *C, int M, int N, int K){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i < M && j < N){
        float tmp = 0.;
        for(int k = 0; k < K; k++){
            tmp += A[i*K + k] * B[j + k*N];
        }
        C[i * N + j] = tmp;
    }
}

__global__
void mmCoalesced(float *A, float *B, float *C, int M, int N, int K){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i < N  && j < M){
        float tmp = 0.;
        for(int k = 0; k < K; k++){
            tmp += A[j*K + k] * B[i + k*N];
        }
        C[j * N + i] = tmp;
    }
}

// shared memory optimization
template <const int BLOCKSIZE>
__global__
void mmShared(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int M, int N, int K){
    __shared__ float sA[BLOCKSIZE * BLOCKSIZE];
    __shared__ float sB[BLOCKSIZE * BLOCKSIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    A += K * BLOCKSIZE * blockIdx.y;
    B += blockIdx.x * BLOCKSIZE;

    // grid block configured based on output matrix dimensions
    // each thread takes care of respective output (via tmp)
    float tmp = 0.0;
    for(int k = 0; k < K; k += BLOCKSIZE) {             // sliding window

        // copy data to shared memory
        sA[ty * BLOCKSIZE + tx] = A[ty * K + tx];
        sB[ty * BLOCKSIZE + tx] = B[ty * N + tx];
        __syncthreads();

        A += BLOCKSIZE;         // update starting element of A's submatrix
        B += BLOCKSIZE * N;     // update starting element of B's submatrix

        // loop over submatrix and add to respective thread's tmp variable
        for(int ps = 0; ps < BLOCKSIZE; ps++) {
            tmp += sA[ty * BLOCKSIZE + ps] * sB[ps * BLOCKSIZE + tx];
        }
        __syncthreads();
    }
    C[y * N + x] = tmp;
}


// 1D register tiling
template <const int BLOCKSIZE, const int REGTILESIZE>
__global__
void mm1DRegisterTiling(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int M, int N, int P){
    __shared__ float sA[BLOCKSIZE * BLOCKSIZE];
    __shared__ float sB[BLOCKSIZE * BLOCKSIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * REGTILESIZE;

    int tx = threadIdx.x;
    int ty = threadIdx.y * REGTILESIZE;

    // each thread takes care of REGTILESIZE outputs along the columnwise direction
    float tmp[REGTILESIZE] = {0.0};
    for(int p = 0; p < P; p += BLOCKSIZE) {             // sliding window

        int startA = P * BLOCKSIZE * blockIdx.y + p;    // starting element of A's submatrix
        int startB = p * N + blockIdx.x * BLOCKSIZE;    // starting element of B's submatrix

        // copy data to shared memory
        #pragma unroll
        for(int ts = 0; ts < REGTILESIZE; ts++){
            sA[(ty + ts) * BLOCKSIZE + tx] = A[startA + (ty + ts) * P + tx];
            sB[(ty + ts) * BLOCKSIZE + tx] = B[startB + (ty + ts) * N + tx];
        }
        __syncthreads();

        // loop over submatrix and add to respective output's tmp variable
        for(int ps = 0; ps < BLOCKSIZE; ps++) {
            float Btmp = sB[ps * BLOCKSIZE + tx];   // B matrix value remains same, hence save it to register
            #pragma unroll
            for(int ts = 0; ts < REGTILESIZE; ts++) {
                tmp[ts] += sA[(ty + ts) * BLOCKSIZE + ps] * Btmp;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int ts = 0; ts < REGTILESIZE; ts++){
        C[(y+ts) * N + x] = tmp[ts];
    }
}




// 2D register tiling (without sA transposed)
template <const int BLOCKSIZE, const int REGTILESIZE>
__global__
void mm2DRegisterTilingNoSAtranspose(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int M, int N, int P){
    __shared__ float sA[BLOCKSIZE * BLOCKSIZE];
    __shared__ float sB[BLOCKSIZE * BLOCKSIZE];

    int x = (blockIdx.x * blockDim.x + threadIdx.x) * REGTILESIZE;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * REGTILESIZE;

    int tx = threadIdx.x * REGTILESIZE;
    int ty = threadIdx.y * REGTILESIZE;

    // each thread takes care of REGTILESIZE*REGTILESIZE outputs
    float tmp[REGTILESIZE * REGTILESIZE] = {0.0f};
    float Atmp[REGTILESIZE] = {0.0f};
    float Btmp[REGTILESIZE] = {0.0f};

    for(int p = 0; p < P; p += BLOCKSIZE) {             // sliding window

        int startA = P * BLOCKSIZE * blockIdx.y + p;    // starting element of A's submatrix
        int startB = p * N + blockIdx.x * BLOCKSIZE;    // starting element of B's submatrix

        // copy data to shared memory
        #pragma unroll
        for(int ts = 0; ts < REGTILESIZE; ts++){
            #pragma unroll
            for(int tt = 0; tt < REGTILESIZE; tt++){
                sA[(ty + ts) * BLOCKSIZE + tx + tt] = A[startA + (ty + ts) * P + tx + tt];
                sB[(ty + ts) * BLOCKSIZE + tx + tt] = B[startB + (ty + ts) * N + tx + tt];
            }
        }
        __syncthreads();

        // loop over row/col and add to respective output's tmp variable
        for(int pt = 0; pt < BLOCKSIZE; pt += REGTILESIZE) {
            for(int k = 0; k < REGTILESIZE; k++){
                #pragma unroll
                for(int ts = 0; ts < REGTILESIZE; ts++){
                    Atmp[ts] = sA[(ty + ts) * BLOCKSIZE + pt + k];
                    Btmp[ts] = sB[(pt + k) * BLOCKSIZE + ts + tx];
                }

                #pragma unroll
                for(int ts = 0; ts < REGTILESIZE; ts++){
                    #pragma unroll
                    for(int tt = 0; tt < REGTILESIZE; tt++){
                        tmp[ts*REGTILESIZE + tt] += Atmp[ts] * Btmp[tt];
                    }
                }
            }
        }
        __syncthreads();
    }

    for(int ts = 0; ts < REGTILESIZE; ts++){
        #pragma unroll
        for(int tt = 0; tt < REGTILESIZE; tt++){
            C[(y+ts) * N + x + tt] = tmp[ts * REGTILESIZE + tt];
        }
    }
}


// 2D register tiling (with sA transposed)
template <const int BLOCKSIZE, const int REGTILESIZE>
__global__
void mm2DRegisterTiling(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int M, int N, int P){
    __shared__ float sA[BLOCKSIZE * BLOCKSIZE];
    __shared__ float sB[BLOCKSIZE * BLOCKSIZE];

    int x = (blockIdx.x * blockDim.x + threadIdx.x) * REGTILESIZE;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * REGTILESIZE;

    int tx = threadIdx.x * REGTILESIZE;
    int ty = threadIdx.y * REGTILESIZE;

    // each thread takes care of REGTILESIZE*REGTILESIZE outputs
    float tmp[REGTILESIZE * REGTILESIZE] = {0.0f};
    float Atmp[REGTILESIZE] = {0.0f};
    float Btmp[REGTILESIZE] = {0.0f};

    for(int p = 0; p < P; p += BLOCKSIZE) {             // sliding window

        int startA = P * BLOCKSIZE * blockIdx.y + p;    // starting element of A's submatrix
        int startB = p * N + blockIdx.x * BLOCKSIZE;    // starting element of B's submatrix

        // copy data to shared memory
        #pragma unroll
        for(int ts = 0; ts < REGTILESIZE; ts++){
            #pragma unroll
            for(int tt = 0; tt < REGTILESIZE; tt++){
                sA[(tx + tt) * BLOCKSIZE + ty + ts] = A[startA + (ty + ts) * P + tx + tt];
                sB[(ty + ts) * BLOCKSIZE + tx + tt] = B[startB + (ty + ts) * N + tx + tt];
            }
        }
        __syncthreads();

        // loop over row/col and add to respective output's tmp variable
        for(int pt = 0; pt < BLOCKSIZE; pt += REGTILESIZE) {
            #pragma unroll
            for(int k = 0; k < REGTILESIZE; k++){
                int AtmpOffset = (pt + k) * BLOCKSIZE + ty;
                int BtmpOffset = (pt + k) * BLOCKSIZE + tx;
                #pragma unroll
                for(int ts = 0; ts < REGTILESIZE; ts++){
                    Atmp[ts] = sA[AtmpOffset + ts];
                    Btmp[ts] = sB[BtmpOffset + ts];
                }

                #pragma unroll
                for(int ts = 0; ts < REGTILESIZE; ts++){
                    #pragma unroll
                    for(int tt = 0; tt < REGTILESIZE; tt++){
                        tmp[ts*REGTILESIZE + tt] += Atmp[ts] * Btmp[tt];
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int ts = 0; ts < REGTILESIZE; ts++){
        #pragma unroll
        for(int tt = 0; tt < REGTILESIZE; tt++){
            C[(y+ts) * N + x + tt] = tmp[ts * REGTILESIZE + tt];
        }
    }
}


// vectorize register loads
template <const int BLOCKSIZE, const int REGTILESIZE>
__global__
void vectorizeRegisterLoads(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int M, int N, int P){
    __shared__ float sA[BLOCKSIZE * BLOCKSIZE];
    __shared__ float sB[BLOCKSIZE * BLOCKSIZE];

    int x = (blockIdx.x * blockDim.x + threadIdx.x) * REGTILESIZE;
    int y = (blockIdx.y * blockDim.y + threadIdx.y) * REGTILESIZE;

    int tx = threadIdx.x * REGTILESIZE;
    int ty = threadIdx.y * REGTILESIZE;

    // each thread takes care of REGTILESIZE*REGTILESIZE outputs
    float tmp[REGTILESIZE * REGTILESIZE] = {0.0f};
    float Atmp[REGTILESIZE] = {0.0f};
    float Btmp[REGTILESIZE] = {0.0f};

    float4 ldVec;

    for(int p = 0; p < P; p += BLOCKSIZE) {             // sliding window

        int startA = P * BLOCKSIZE * blockIdx.y + p;    // starting element of A's submatrix
        int startB = p * N + blockIdx.x * BLOCKSIZE;    // starting element of B's submatrix

        // copy data to shared memory
        for(int ts = 0; ts < REGTILESIZE; ts++) {
            for(int tt = 0; tt < REGTILESIZE; tt+=4){
                ldVec = 
                reinterpret_cast<float4*>(&A[startA + (ty + ts) * P + tx + tt])[0];
                sA[(tx + tt) * BLOCKSIZE + ty + ts] = ldVec.w;
                sA[(tx + tt + 1) * BLOCKSIZE + ty + ts] = ldVec.x;
                sA[(tx + tt + 2) * BLOCKSIZE + ty + ts] = ldVec.y;
                sA[(tx + tt + 3) * BLOCKSIZE + ty + ts] = ldVec.z;
            }
        }
        
        for(int ts = 0; ts < REGTILESIZE; ts++) {
            for(int tt = 0; tt < REGTILESIZE; tt+=4){
                reinterpret_cast<float4*> (&sB[(ty + ts) * BLOCKSIZE + tx + tt])[0] = 
                reinterpret_cast<float4*> (&B[startB + (ty + ts) * N + tx + tt])[0];
            }
        }
        __syncthreads();

        // loop over row/col and add to respective output's tmp variable
        for(int pt = 0; pt < BLOCKSIZE; pt += REGTILESIZE) {

            for(int k = 0; k < REGTILESIZE; k++){
                int AtmpOffset = (pt + k) * BLOCKSIZE + ty;
                int BtmpOffset = (pt + k) * BLOCKSIZE + tx;

                for(int ts = 0; ts < REGTILESIZE; ts+=4){
                    reinterpret_cast<float4*>(&Atmp[ts])[0] = 
                        reinterpret_cast<float4*>(&sA[AtmpOffset + ts])[0];
                    reinterpret_cast<float4*>(&Btmp[ts])[0] = 
                        reinterpret_cast<float4*> (&sB[BtmpOffset + ts])[0];
                }

                for(int ts = 0; ts < REGTILESIZE; ts++){
                    for(int tt = 0; tt < REGTILESIZE; tt++){
                        tmp[ts*REGTILESIZE + tt] += Atmp[ts] * Btmp[tt];
                    }
                }
            }
        }
        
        __syncthreads();


    }

    for(int ts = 0; ts < REGTILESIZE; ts++){
        for(int tt = 0; tt < REGTILESIZE; tt+=4){
            reinterpret_cast<float4*>(&C[(y+ts) * N + x + tt])[0] = 
            reinterpret_cast<float4*>(&tmp[ts * REGTILESIZE + tt])[0];
        }
    }

    // if(blockIdx.x == 1 && blockIdx.y == 0 && tx == 4 && ty == 0){
    //     printf("\n");
    //     for(int ts = 0; ts < REGTILESIZE; ts++){
    //         for(int tt = 0; tt < REGTILESIZE; tt++){                
    //             printf("%f ", tmp[ts * REGTILESIZE + tt]);
    //         }
    //         printf("\n");                
    //     }
    // }

    // if(x == 0 && y == 0){
    //     printf("GPU output\n");
    //     printArrayDevice(C, M, N);
    // }
}


int main(){
    // A(M x P) * B(P x N) = C(M x N)
    int M, N, P;
    M = N = P = matSize;

    // matrix sizes
    int sizeA = M * P;
    int sizeB = P * N;
    int sizeC = M * N;

    // host memory allocation
    float *hA = (float *) malloc(sizeA * sizeof(float));
    float *hB = (float *) malloc(sizeB * sizeof(float));
    float *hC = (float *) malloc(sizeC * sizeof(float));
    float *hCfromGPU    = (float *) malloc(sizeC * sizeof(float));
    float *hCfromCUBLAS = (float *) malloc(sizeC * sizeof(float));


    // device memory allocation
    float *A = (float *)fixed_cudaMalloc(sizeA * sizeof(float));
    float *B = (float *)fixed_cudaMalloc(sizeB * sizeof(float));
    float *C = (float *)fixed_cudaMalloc(sizeC * sizeof(float));
    float *C_cuBLAS = (float *)fixed_cudaMalloc(sizeC * sizeof(float));
    
    // host memory initialization
    srand (static_cast <unsigned> (time(0)));  // seed for random initialization
    intializeMatrix(hA, sizeA);//, 1.0);
    intializeMatrix(hB, sizeB);

    // copy matrices from host to device
    gpuErrchk(cudaMemcpy(A, hA, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(B, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice));

    //*********** GPU compute **********************//
    // // Naive MM
    // int numThreads = 32;
    // dim3 blockSize(numThreads, numThreads);
    // dim3 gridSize(ceil(M / numThreads), ceil(N / numThreads));
    // std::cout << "BlockSize : " << blockSize.x << " " << blockSize.y <<"\n";
    // std::cout << "GridSize : " << gridSize.x << " " << gridSize.y <<"\n";        
    // mmNaive<<<gridSize, blockSize>>>(A, B, C, M, N, P);

    // // Coalesced access
    // mmCoalesced<<<gridSize, blockSize>>>(A, B, C, M, N, P);

{
    // Shared memory
    const int numThreads = 32;
    dim3 blockSize(numThreads, numThreads);
    dim3 gridSize(CEIL_DIV(N, numThreads), CEIL_DIV(M, numThreads));
    mmShared<numThreads><<<gridSize, blockSize>>>(A, B, C, M, N, P);
}

    // // 1D tiling
    // const int numThreads = 32;
    // const int REGTILESIZE = 8;
    // dim3 blockSize(numThreads, numThreads / REGTILESIZE);
    // dim3 gridSize(CEIL_DIV(N, numThreads), CEIL_DIV(M, numThreads));
    // mm1DRegisterTiling<numThreads, REGTILESIZE><<<gridSize, blockSize>>>(A, B, C, M, N, P);

    // // 2D tiling (no sA transpose)
    // const int BLOCKSIZE = 64;
    // const int REGTILESIZE = 4;
    // dim3 blockSize(BLOCKSIZE/REGTILESIZE, BLOCKSIZE/REGTILESIZE); // numThreads in x, y direction
    // dim3 gridSize(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
    // mm2DRegisterTilingNoSAtranspose<BLOCKSIZE, REGTILESIZE><<<gridSize, blockSize>>>(A, B, C, M, N, P);

    // // 2D tiling (with sA transposed)
    // {
    // const int BLOCKSIZE = 64;
    // const int REGTILESIZE = 4;
    // dim3 blockSize(BLOCKSIZE/REGTILESIZE, BLOCKSIZE/REGTILESIZE); // numThreads in x, y direction
    // dim3 gridSize(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
    // mm2DRegisterTiling<BLOCKSIZE, REGTILESIZE><<<gridSize, blockSize>>>(A, B, C, M, N, P);
    // }

/*{
    // vectorize register loads
    const int BLOCKSIZE = 64;
    const int REGTILESIZE = 4;
    dim3 blockSize(BLOCKSIZE/REGTILESIZE, BLOCKSIZE/REGTILESIZE); // numThreads in x, y direction
    dim3 gridSize(CEIL_DIV(N, BLOCKSIZE), CEIL_DIV(M, BLOCKSIZE));
    vectorizeRegisterLoads<BLOCKSIZE, REGTILESIZE><<<gridSize, blockSize>>>(A, B, C, M, N, P);
}*/
    
    auto start = std::chrono::steady_clock::now();
    cuBLAScomputeMM(A, B, C_cuBLAS, M, N, P);
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << std::chrono::duration<double, std::milli>(diff).count() << " milli-seconds" << std::endl;    
    gpuErrchk(cudaMemcpy(hCfromCUBLAS, C_cuBLAS, sizeC * sizeof(float), cudaMemcpyDeviceToHost));    

    // copy back results from device to host
    gpuErrchk(cudaMemcpy(hCfromGPU, C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));    


    // CPU compute of MM
    computeMM(hA, hB, hC, M, N, P);

    // cout << "Host output\n";
    // printArrayDevice(hC, M, N);

    // compare GPU and CPU results
    // compareResults(hC, hCfromGPU, M, N);
    compareResults(hCfromGPU, hCfromCUBLAS, M, N);


    // Free the memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C_cuBLAS);

    free(hA);
    free(hB);
    free(hC);
    free(hCfromGPU);
    free(hCfromCUBLAS);

    return 0;
}
