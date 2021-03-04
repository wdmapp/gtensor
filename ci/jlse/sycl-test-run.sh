#!/bin/bash
#COBALT -n 1
#COBALT -t 20
#COBALT --jobname GTENSOR_SYCL_TEST_$jobid
#COBALT -O GTENSOR_SYCL_TEST.$jobid

set -e

REPO=${REPO:-https://github.com/wdmapp/gtensor.git}
BRANCH=${BRANCH:-master}
ROOT_DIR="$(pwd)"
SOURCE_DIR="$ROOT_DIR"/gtensor
BIN_DIR="$ROOT_DIR"/build-sycl-gpu

export http_proxy="http://proxy:3128"
export https_proxy="http://proxy:3128"
export ftp_proxy="http://proxy:3128"

rm -rf "$SOURCE_DIR"

echo REPO=$REPO BRANCH=$BRANCH
git clone -b $BRANCH "$REPO"

pushd gtensor
git status
git log -n2
popd

module load --no-pager oneapi
module load --no-pager cmake

CXX=$(which dpcpp) cmake -S "$SOURCE_DIR" -B "$BIN_DIR" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGTENSOR_DEVICE=sycl -DGTENSOR_DEVICE_SYCL_SELECTOR=gpu \
  -DBUILD_TESTING=ON -DGTENSOR_BUILD_EXAMPLES=ON -DGTENSOR_TEST_DEBUG=ON \
  -DONEAPI_PATH=$A21_SDK_ROOT -DMKL_PATH=$MKLROOT \
  -DGTENSOR_ENABLE_CLIB=ON -DGTENSOR_ENABLE_BLAS=ON -DGTENSOR_ENABLE_FFT=ON

cmake --build "$BIN_DIR" -j8

set +e

#export SYCL_DEVICE_FILTER=opencl
export SYCL_DEVICE_FILTER=level_zero

sycl-ls

tail_lines=15

exit_code=0

for f in "$BIN_DIR"/tests/test_*; do
  if [ -x "$f" ]; then
    test_dir=$(dirname "$f")
    test_name=$(basename "$f")
    test_log=$test_dir/$test_name.log
    $f >$test_log 2>&1
    result=$?
    echo $test_name $result
    if [ $result -ne 0 ]; then
      tail -n$tail_lines $test_log | sed 's/^/  /'
      exit_code=$result
    fi
  fi
done

exit $exit_code
