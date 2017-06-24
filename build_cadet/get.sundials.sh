#!/bin/sh

set -e

wgetcmd="wget"
wgetcount=`which wget 2>/dev/null | wc -l`
if test ! $wgetcount = 1; then
  echo "Utility wget not found in your PATH."
  if test `uname` = Darwin; then
    wgetcmd="curl -L -k -O"
    echo "Using ftp command instead."
  elif test `uname` = FreeBSD; then
    wgetcmd=fetch
    echo "Using fetch command instead."
  else
    exit -1
  fi
fi

sundials=sundials-2.7.0

echo " "
echo "Running script for downloading the source code for the sundials"
echo " "

rm -f $sundials.tar.gz

echo "Downloading the source code from Github..."
if $wgetcmd https://computation.llnl.gov/projects/sundials/download/$sundials.tar.gz ;
then
  echo "Download finished."
else
  echo
  echo "Download failed...exiting"
fi

echo "Unpacking the source code..."
gunzip -f $sundials.tar.gz
tar xf $sundials.tar
rm -rf $sundials.tar.gz
rm -rf $sundials.tar

# building sundials
mkdir build_sundials
cd build_sundials
cmake -DCMAKE_INSTALL_PREFIX=$PWD/cadet-install/Libs/sundials -DEXAMPLES_ENABLE=OFF -DOPENMP_ENABLE=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS=-fPIC -DOPENMP_FOUND=0 ../$sundials
make
make install
cd ../..
