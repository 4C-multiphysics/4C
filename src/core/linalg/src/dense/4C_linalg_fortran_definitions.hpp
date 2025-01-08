// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_FORTRAN_DEFINITIONS_HPP
#define FOUR_C_LINALG_FORTRAN_DEFINITIONS_HPP

#include "4C_config.hpp"

// Do not lint the file for identifier names since we do not own these identifiers.

// NOLINTBEGIN(readability-identifier-naming)

// append underscores, if necessary. Important for linking to fortran routines
#undef CCA_APPEND_U
#define CCA_APPEND_U (1)

#ifdef CCA_APPEND_U
// required to use lapack functions
#define dsytrf dsytrf_
#define dsytri dsytri_
#define dhgeqz dhgeqz_
#define dgghrd dgghrd_
#define dgeqp3 dgeqp3_
#define dggbal dggbal_

#endif

// fortran routines from the lapack package
#ifdef __cplusplus
extern "C"
{
#endif

  void dsytrf(
      char* uplo, int* n, double* a, int* lda, int* ipiv, double* work, int* lwork, int* info);
  void dsytri(char* uplo, int* n, double* a, int* lda, int* ipiv, double* work, int* info);
  void dhgeqz(char* job, char* compq, char* compz, int* n, int* ilo, int* ihi, double* h, int* ldh,
      double* t, int* ldt, double* alphar, double* alphai, double* beta, double* q, int* ldq,
      double* z, int* ldz, double* work, int* lwork, int* info);
  void dgghrd(char* compq, char* compz, int* n, int* ilo, int* ihi, double* a, int* lda, double* b,
      int* ldb, double* q, int* ldq, double* z, int* lzd, int* info);
  void dgeqp3(int* m, int* n, double* a, int* lda, int* jpvt, double* tau, double* work, int* lwork,
      int* info);
  void dggbal(const char* job, const int* n, double* A, const int* lda, double* B, const int* ldb,
      int* ilo, int* ihi, double* lscale, double* rscale, double* work, int* info);

#ifdef __cplusplus
}

#endif

// NOLINTEND(readability-identifier-naming)

#endif
