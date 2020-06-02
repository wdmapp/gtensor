#include <math.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include <gtensor/gtensor.h>

__global__ void kernel_add(float* c, const float* a, const float* b, int N)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < N) {
    c[i] = a[i] + b[i];
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
  kernel_add<<<N / block_size, block_size>>>(d_c, d_a, d_b, N);

  CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

  printf("=== sin(i) + cos(i)\n");
  for (i = 0; i < N; i += N / 32) {
    printf("%0.2f = %0.2f + %0.2f\n", h_c[i], h_a[i], h_b[i]);
  }

  // new code to calculate sin^2 + cos^2 using existing data arrays
  auto shape = gt::shape(N);

  auto gh_a = gt::adapt(h_a, shape);
  auto gh_b = gt::adapt(h_b, shape);
  auto gh_c = gt::adapt(h_c, shape);

  auto gd_a = gt::adapt_device(d_a, shape);
  auto gd_b = gt::adapt_device(d_b, shape);
  auto gd_c = gt::adapt_device(d_c, shape);

  gd_c = gd_a * gd_a + gd_b * gd_b;
  copy(gd_c, gh_c);

  printf("=== sin(i)^2 + cos(i)^2\n");
  for (i = 0; i < N; i += N / 32) {
    printf("%0.2f = %0.2f + %0.2f\n", gh_c(i), gh_a(i), gh_b(i));
  }
  // end new code

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);
}
