#!/bin/bash

./build/bin/knn-bench 4096 64 3 32

# use -v at the end in order to disable custom validation for large searches
#./build/bin/knn-bench 4096 64 3 32 -v
