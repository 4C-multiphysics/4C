#!/bin/zsh
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install required tools on Darwin and compile the dependencies automatically
# It is important to run this script will download 4C source code down the road,
# therefore it is not suggested to execute under the home folder.
# By default, the compiled dependencies are installed in $HOME/opt. If it is
# system folder, access grant might be needed.
# Tested with MacOs 27.3
export DEP_DIR=$HOME/opt

# install required tools
brew install openmpi git wget gcc@14 hdf5 ninja llvm boost cln

brew tap botantony/cmake3
brew install cmake3
echo 'export PATH="/opt/homebrew/opt/cmake3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# clone 4C repository
git clone https://github.com/4C-multiphysics/4C.git

# compiled the dependencies
cd 4C/dependencies/current/backtrace
sh install.sh $DEP_DIR/backtrace
#
cd ..
cd suitesparse
sh install.sh $DEP_DIR/suitesparse
#
cd ..
cd qhull
sh install.sh $DEP_DIR/qhull "-DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14"
#
cd ../../darwin
cd parmetis
sh install.sh $DEP_DIR/parmetis
#
cd ..
cd superlu_dist
sh install.sh $DEP_DIR/superlu_dist
#
cd ..
cd trilinos
sh install.sh $DEP_DIR/trilinos $DEP_DIR
