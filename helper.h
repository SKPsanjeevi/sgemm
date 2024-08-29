#include <cstdlib>  // header for rand()
#include <ctime>    // header for time-seed for rand()
#include <limits>   // get the smallest increment for a datatype
#include <cublas_v2.h>

using namespace std;

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void intializeMatrix(float *A, int size){
    for(int i = 0; i < size; i++){
        // A[i] = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
        A[i] = (float)i;
        // A[i] = 1.0f;
    }
}

void intializeMatrix(float *A, int size, float val){
    for(int i = 0; i < size; i++){
        A[i] = val;
    }
}

void *fixed_cudaMalloc(size_t len)
{
    void *p;
    if (cudaMalloc(&p, len) == cudaSuccess) return p;
    return 0;
}

template<class T>
bool approximatelyEqual(T a, T b, T epsilon)
{
    return fabs(a - b) <= ( (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

template<class T>
void compareResults(T *C1, T *C2, int M, int N){
    int factor = 100;
    T epsilon = std::numeric_limits<T>::epsilon() * factor;
    std::cout<<"Epsilon : " << epsilon << std::endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            if( !approximatelyEqual(C1[i*N + j], C2[i*N + j], epsilon) ){
                printf("Outside tolerance at indices at i : %d, j : %d, C1 : %f, C2 : %f\n",
                                                i, j, C1[i*N + j], C2[i*N + j]);
                exit(0);
            }
        }
    }
}

__host__  __device__
void printArrayDevice(float *A, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%1.0f ", A[i*cols + j]);
        }
        printf("\n");
    }
}

void computeMM(float *hA, float *hB, float *hC, int M, int N, int K){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            double tmp = 0.;
            for(int k = 0; k < K; k++){
                tmp += (double)hA[i*K + k] * (double)hB[j + k*N];
            }
            hC[i * N + j] = (float)tmp;
        }
    }
}

void cuBLAScomputeMM(float *A, float *B, float *C_cuBLAS, int M, int N, int K, float alpha, float beta){
    cublasHandle_t handle;
    cublasCreate(&handle);

    // CUBLAS MM uses matrices in column-major order
    // Therefore, instead of computing C' = (A * B)', we compute B' * A'

    cublasSgemm(handle
                , CUBLAS_OP_N   // no transpose for B (as it is row-major)
                , CUBLAS_OP_N   // no transpose for A (as it is row-major)
                , N
                , M
                , K
                , &alpha
                , B             // note first input is B
                , N
                , A             // note second input is A
                , K
                , &beta
                , C_cuBLAS
                , N
                );

    cublasDestroy(handle);
}
