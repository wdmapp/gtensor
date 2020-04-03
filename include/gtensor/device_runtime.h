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

#if GTENSOR_HAVE_DEVICE

#ifdef __HCC__
#include "hip/hip_runtime.h"
#endif

#endif

#endif
