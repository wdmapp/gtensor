# CONFIGURATION
#   GTENSOR_DEVICE - cuda, hip, or host (default cuda)
#   GTENSOR_CXX - override default compiler (default is device dependent)
#   ROCM_PATH - for hip device, override default rocm path (default /opt/rocm)

# gtensor configuration
# TODO: add example recipe to auto-fetch gtensor
GTENSOR_DIR = external/gtensor
GTENSOR_DEVICE ?= cuda

GTENSOR_HEADERS = $(wildcard $(GTENSOR_DIR)/include/*.h)

GTENSOR_DEVICE_DEFINE := GTENSOR_DEVICE_$(shell echo $(GTENSOR_DEVICE) | tr a-z A-Z)
GTENSOR_DEFINES = -D$(GTENSOR_DEVICE_DEFINE)
GTENSOR_INCLUDES = -I$(GTENSOR_DIR)/include
GTENSOR_LIBS =
GTENSOR_OPTIONS = -std=c++14 -O3
ifeq ($(GTENSOR_DEVICE),cuda)
  GTENSOR_CXX ?= nvcc
  GTENSOR_OPTIONS += -x cu --expt-extended-lambda --expt-relaxed-constexpr
  GTENSOR_DEFINES += -DGTENSOR_HAVE_DEVICE
else ifeq ($(GTENSOR_DEVICE),hip)
  ROCM_PATH ?= /opt/rocm
  AMDGPU_TARGET = gfx803
  GTENSOR_CXX ?= $(ROCM_PATH)/bin/hipcc
  GTENSOR_OPTIONS += -hc -amdgpu-target=$(AMDGPU_TARGET)
  # Note: HIP doesn't support assert in device code
  GTENSOR_DEFINES += -DGTENSOR_HAVE_DEVICE -DNDEBUG
  GTENSOR_INCLUDES += -isystem $(ROCM_PATH)/include -isystem $(ROCM_PATH)/hip/include -isystem $(ROCM_PATH)/rocprim/include -isystem $(ROCM_PATH)/rocthrust/include
else
  GTENSOR_CXX ?= clang++
endif

GTENSOR_FLAGS = $(GTENSOR_DEFINES) $(GTENSOR_INCLUDES) $(GTENSOR_LIBS) $(GTENSOR_OPTIONS)

# targets using gtensor
GTENSOR_TARGETS = daxpy stencil1d mult_table
ifeq ($(GTENSOR_DEVICE),cuda)
  GTENSOR_TARGETS += trig_adapted
endif

.PHONY: all
all: $(GTENSOR_TARGETS)

$(GTENSOR_TARGETS): % : src/%.cxx $(GTENSOR_HEADERS)
	$(GTENSOR_CXX) $(GTENSOR_FLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -rf $(GTENSOR_TARGETS)