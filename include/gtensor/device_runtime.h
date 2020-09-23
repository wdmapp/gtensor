// ======================================================================
// device_runtime.h
//
// Include any files required by the GPU device runtime. For example, the HIP
// compiler does not recognize __host__ and __device__ annotatios until it's
// runtime header is included.
//
// Copyright (C) 2020 Kai Germaschewski, Bryce Allen

#ifndef GTENSOR_DEVICE_RUNTIME_H
#define GTENSOR_DEVICE_RUNTIME_H

#ifdef GTENSOR_DEVICE_CUDA
#include "cuda_runtime_api.h"
#elif defined(GTENSOR_DEVICE_HIP)
#include "hip/hip_runtime.h"
#elif defined(GTENSOR_DEVICE_SYCL)
#include <CL/sycl.hpp>
#endif

#endif
