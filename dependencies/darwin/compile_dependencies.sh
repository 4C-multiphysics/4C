#!/bin/zsh
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

export DEP_DIR=$HOME/opt

# compiled the dependencies
cd dependencies/current/backtrace
sh install.sh $DEP_DIR/libbacktrace
#
cd ..
cd suitesparse
sh install.sh $DEP_DIR/suitesparse
#
cd ../../darwin
cd parmetis
sh install.sh $DEP_DIR/parmetis
#
cd ..
cd qhull
sh install.sh $DEP_DIR/qhull "-DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14"
#
cd ..
cd superlu_dist
sh install.sh $DEP_DIR/superlu_dist
#
cd ..
cd mumps
sh install.sh $DEP_DIR/mumps
#
cd ..
cd trilinos
sh install.sh $DEP_DIR/trilinos $DEP_DIR
