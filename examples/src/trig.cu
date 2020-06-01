#include <math.h>
#include <stdio.h>

#include <cuda_runtime.h>

__global__ void kernel_add_sq(float* c, const float* a, const float* b, int N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < N) {
    c[i] = a[i] * a[i] + b[i] * b[i];
  }
}

inline cudaError_t CHECK(cudaError_t err)
{
  if (err != cudaSuccess) {
    printf("Error: %d %s\n", err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return err;
}

int main(int argc, char** argv)
{
  const int N = 1024 * 1024;
  const int block_size = 256;
  int i;
  int size = N * sizeof(float);

  float* h_a = (float*)malloc(size);
  float* h_b = (float*)malloc(size);
  float* h_c = (float*)malloc(size);

  for (i = 0; i < N; i++) {
    h_a[i] = sin(i);
    h_b[i] = cos(i);
  }

  float *d_a, *d_b, *d_c;
  CHECK(cudaMalloc((void**)&d_a, size));
  CHECK(cudaMalloc((void**)&d_b, size));
  CHECK(cudaMalloc((void**)&d_c, size));

  CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

  // assumes block_size devices N
  kernel_add_sq<<<N / block_size, block_size>>>(d_c, d_a, d_b, N);

  CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

  printf("=== sin(i) + cos(i)\n");
  for (i = 0; i < N; i += N / 32) {
    printf("%0.2f = %0.2f + %0.2f\n", h_c[i], h_a[i], h_b[i]);
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);
}
