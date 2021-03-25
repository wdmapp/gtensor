#ifndef GTENSOR_FFT_HIP_H
#define GTENSOR_FFT_HIP_H

#include <hipfft.h>

#define CUFFT_SUCCESS HIPFFT_SUCCESS

#define CUFFT_Z2Z HIPFFT_Z2Z
#define CUFFT_C2C HIPFFT_C2C
#define CUFFT_D2Z HIPFFT_D2Z
#define CUFFT_Z2D HIPFFT_Z2D
#define CUFFT_R2C HIPFFT_R2C
#define CUFFT_C2R HIPFFT_C2R

#define CUFFT_FORWARD HIPFFT_FORWARD
#define CUFFT_INVERSE HIPFFT_BACKWARD
#define CUFFT_BACKWARD HIPFFT_BACKWARD

using cufftHandle = hipfftHandle;
using cufftType = hipfftType;
using cudaStream_t = hipStream_t;

using cufftDoubleComplex = hipfftDoubleComplex;
using cufftComplex = hipfftComplex;
using cufftDoubleReal = hipfftDoubleReal;
using cufftReal = hipfftReal;

#define cufftPlanMany hipfftPlanMany
#define cufftDestroy hipfftDestroy

#define cufftExecZ2Z hipfftExecZ2Z
#define cufftExecC2C hipfftExecC2C
#define cufftExecD2Z hipfftExecD2Z
#define cufftExecZ2D hipfftExecZ2D
#define cufftExecR2C hipfftExecR2C
#define cufftExecC2R hipfftExecC2R

#include "gtensor/fft/cuda.h"

#endif // GTENSOR_FFT_HIP_H
