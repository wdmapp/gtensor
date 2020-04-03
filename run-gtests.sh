#!/bin/bash

build_dir=${1:-.}

for f in "$build_dir"/tests/test_*; do
    if [ -x "$f" ]; then
        test_dir=$(dirname "$f")
        test_name=$(basename "$f")
        $f >$test_dir/$test_name.log 2>&1
        result=$?
        echo $test_name $result
    fi
done
