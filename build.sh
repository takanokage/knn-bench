#!/bin/bash

. clean.sh

echo generating...

mkdir -p build
cd build

# ~/cmake/v3.9.0/bin/cmake .. -DCMAKE_BUILD_TYPE=Debug
~/cmake/v3.13.4/bin/cmake .. -DCMAKE_BUILD_TYPE=Debug

echo
echo building...

make -j $(nproc)

echo
