#!/bin/bash

build_dir=${1:-.}

exit_code=0

for f in "$build_dir"/tests/test_*; do
    if [ -x "$f" ]; then
        test_dir=$(dirname "$f")
        test_name=$(basename "$f")
        test_log=$test_dir/$test_name.log
        $f >$test_log 2>&1
        result=$?
        echo $test_name $result
        if [ $result -ne 0 ]; then
          cat $test_log
          exit_code=$result
        fi
    fi
done

exit $exit_code
