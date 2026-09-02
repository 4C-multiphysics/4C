#!/usr/bin/env bash

# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install necessary packages and linkages for compilation on macos.
# This script is meant to use in the github workflow but can also
# be used locally by removing the lines with GITHUB_ENV.

# Exit the script at the first failure
set -e

echo "System information:"
g++ -v
uname
sysctl -n hw.physicalcpu
sudo mkdir -p /usr/local/lib
brew update
sudo ln -sf $(brew --prefix gcc@14)/bin/gfortran-14 /opt/homebrew/bin/gfortran
brew install jpeg-turbo
ls -l /opt/homebrew/lib/libjpeg.dylib
brew install openmpi hdf5 llvm boost cln metis netcdf lld scalapack fftw vtk gmsh
brew install suite-sparse
echo "DYLD_LIBRARY_PATH=/opt/homebrew/lib:${DYLD_LIBRARY_PATH}" >> $GITHUB_ENV
echo "HOME=$HOME" >> $GITHUB_ENV

# There is a bug in macos-26 that dependencies of gmsh library are
# installed to incorrect location. The lines below relinks it.
mkdir -p $HOME/opt
sudo chown -R $(whoami) $HOME/opt
sudo mkdir -p /opt/homebrew/Cellar/fltk/1.4.4/lib
sudo mkdir -p /opt/homebrew/Cellar/libpng/1.6.54_1/lib
sudo ln -sf /opt/homebrew/opt/fltk/lib/libfltk_images.1.4.dylib /opt/homebrew/Cellar/fltk/1.4.4/lib/libfltk_images.dylib
sudo ln -sf /opt/homebrew/opt/libpng/lib/libpng16.16.dylib /opt/homebrew/Cellar/libpng/1.6.54_1/lib/libpng16.dylib
sudo ln -sf /opt/homebrew/opt/libpng/lib/libpng.dylib /opt/homebrew/Cellar/libpng/1.6.54_1/lib/libpng.dylib
sudo ln -sf /opt/homebrew/opt/jpeg-turbo/lib/libjpeg.8.dylib /opt/homebrew/Cellar/libpng/1.6.54_1/lib/libjpeg.dylib
sudo ln -sf /opt/homebrew/opt/fltk/lib/libfltk.1.4.dylib /opt/homebrew/Cellar/fltk/1.4.4/lib/libfltk.dylib
sudo ln -sf /opt/homebrew/opt/fltk/lib/libfltk_gl.1.4.dylib /opt/homebrew/Cellar/fltk/1.4.4/lib/libfltk_gl.dylib
