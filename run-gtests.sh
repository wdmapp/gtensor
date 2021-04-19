#!/bin/bash

build_dir=${1:-.}

exit_code=0

exclude=${GTENSOR_TEST_EXCLUDE:-}
echo "EXCLUDE: $exclude"

for f in "$build_dir"/tests/test_*; do
    if [ -x "$f" ]; then
        test_dir=$(dirname "$f")
        test_name=$(basename "$f")
        echo $exclude | grep -wq $test_name
        if [ $? -eq 0 ]; then
          echo "$test_name excluded"
          continue
        fi
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
