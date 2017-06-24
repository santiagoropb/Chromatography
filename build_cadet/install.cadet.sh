#!/bin/sh

export SUNDIALS_ROOT=$PWD/cadet-install/Libs/sundials
mkdir $PWD/cadet-install/build
cd $PWD/cadet-install/buid
cmake -DCMAKE_INSTALL_PREFIX=$PWD/cadet-install/cadet -DMATLAB_FOUND=OFF $PWD/cadet-install/code/
make
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/cadet-install/cadet/lib
cd ../../
