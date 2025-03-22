// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_w1.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 |  setup of constant EAS data (private)                       mgit 01/08|
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_eassetup(Core::LinAlg::SerialDenseMatrix& boplin0,
    Core::LinAlg::SerialDenseVector& F0,           // deformation gradient at origin
    Core::LinAlg::SerialDenseMatrix& xjm0,         // jacobian matrix at origin
    double& detJ0,                                 // det of Jacobian at origin
    const Core::LinAlg::SerialDenseMatrix& xrefe,  // material element coords
    const Core::LinAlg::SerialDenseMatrix& xcure,  // current element coords
    const Core::FE::CellType& distype)

{
  // derivatives at origin
  Core::LinAlg::SerialDenseMatrix deriv0;
  deriv0.shape(2, num_node());

  Core::FE::shape_function_2d_deriv1(deriv0, 0.0, 0.0, distype);

  // compute jacobian matrix at origin
  xjm0.putScalar(0.0);
  for (int k = 0; k < num_node(); k++)
  {
    xjm0(0, 0) += deriv0(0, k) * xrefe(0, k);  // X,r += (X,r)^k
    xjm0(0, 1) += deriv0(0, k) * xrefe(1, k);  // Y,r += (Y,r)^k
    xjm0(1, 0) += deriv0(1, k) * xrefe(0, k);  // X,s += (X,s)^k
    xjm0(1, 1) += deriv0(1, k) * xrefe(1, k);  // Y,s += (Y,s)^k
  }

  /*------------------------------------------ determinant of jacobian ---*/
  detJ0 = xjm0[0][0] * xjm0[1][1] - xjm0[1][0] * xjm0[0][1];

  if (detJ0 < 0.0) FOUR_C_THROW("NEGATIVE JACOBIAN DETERMINANT");


  // compute boplin at origin (boplin0)

  int inode0;
  int dnode0;
  double xji0[2][2];
  /*---------------------------------------------- inverse of jacobian ---*/
  xji0[0][0] = xjm0(1, 1) / detJ0;
  xji0[0][1] = -xjm0(0, 1) / detJ0;
  xji0[1][0] = -xjm0(1, 0) / detJ0;
  xji0[1][1] = xjm0(0, 0) / detJ0;
  /*----------------------------- get operator boplin of global derivatives -*/
  /*-------------- some comments, so that even fluid people are able to
   understand this quickly :-)
   the Boplin looks like
       | Nk,X    0   |
       |   0    Nk,Y |
       | Nk,Y    0   |
       |  0     Nk,X |
  */
  for (inode0 = 0; inode0 < num_node(); inode0++)
  {
    dnode0 = inode0 * 2;

    boplin0(0, dnode0 + 0) = deriv0(0, inode0) * xji0[0][0] + deriv0(1, inode0) * xji0[0][1];
    boplin0(1, dnode0 + 1) = deriv0(0, inode0) * xji0[1][0] + deriv0(1, inode0) * xji0[1][1];
    boplin0(2, dnode0 + 0) = boplin0(1, dnode0 + 1);
    boplin0(3, dnode0 + 1) = boplin0(0, dnode0 + 0);
  }


  // compute displacement-based deformation gradient at origin (F0)

  /*------------------calculate defgrad --------- (Summenschleife->+=) ---*
  defgrad looks like:

        |  1 + Ux,X  |
        |  1 + Uy,Y  |
        |      Ux,Y  |
        |      Uy,X  |
  */
  F0[0] = 1;
  F0[1] = 1;
  F0[2] = 0;
  F0[3] = 0;
  for (int inode = 0; inode < num_node(); inode++)
  {
    F0[0] += boplin0(0, 2 * inode) * (xcure(0, inode) - xrefe(0, inode));
    F0[1] += boplin0(1, 2 * inode + 1) * (xcure(1, inode) - xrefe(1, inode));
    F0[2] += boplin0(2, 2 * inode) * (xcure(0, inode) - xrefe(0, inode));
    F0[3] += boplin0(3, 2 * inode + 1) * (xcure(1, inode) - xrefe(1, inode));
  } /* end of loop over nodes */


  return;
}  // end of w1_eassetup


/*----------------------------------------------------------------------*
 | get the enhanced deformation gradient and                            |
 | also the operators G, W0 and Z                   (private) mgit 01/08|
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_call_defgrad_enh(Core::LinAlg::SerialDenseMatrix& F_enh,
    const Core::LinAlg::SerialDenseMatrix xjm0, const Core::LinAlg::SerialDenseMatrix xjm,
    const double detJ0, const double det, const Core::LinAlg::SerialDenseVector F0,
    const Core::LinAlg::SerialDenseMatrix alpha, const double e1, const double e2,
    Core::LinAlg::SerialDenseMatrix& G, Core::LinAlg::SerialDenseMatrix& W0,
    const Core::LinAlg::SerialDenseMatrix boplin0, Core::LinAlg::SerialDenseMatrix& Z)
{
  // EAS

  // The interpolation functions for the 4 EAS modes are based on
  //
  // Q1E4:
  //     M1 = r 0    M2 = 0 s    M3 = 0 0    M4 = 0 0
  //          0 0         0 0         r 0         0 s
  //
  // Q1ET4:
  //     M1 = r 0    M2 = 0 r    M3 = 0 0    M4 = 0 0
  //          0 0         0 0         s 0         0 s
  //
  //     M = M1*alpha1 + M2*alpha2 + M3*alpha3 + M4*alpha4
  //

  Core::LinAlg::SerialDenseMatrix M;
  M.shape(2, 2);

  Core::LinAlg::SerialDenseMatrix M_temp;
  M_temp.shape(2, 2);

  // fill up 4 EAS matrices at each GP
  if (eastype_ == eas_q1e4)
  {
    M(0, 0) = e1 * alpha(0, 0);  // r
    M(0, 1) = e2 * alpha(1, 0);  // s
    M(1, 0) = e1 * alpha(2, 0);  // r
    M(1, 1) = e2 * alpha(3, 0);  // s
  }
  else if (eastype_ == eas_q1et4)
  {
    M(0, 0) = e1 * alpha(0, 0);  // r
    M(0, 1) = e1 * alpha(1, 0);  // r
    M(1, 0) = e2 * alpha(2, 0);  // s
    M(1, 1) = e2 * alpha(3, 0);  // s
  }
  else
  {
    FOUR_C_THROW("Cannot handle EAS type={}", eastype_);
  }

  // inverse of jacobian matrix at element origin
  Core::LinAlg::SerialDenseMatrix xjm_inv0;
  xjm_inv0.shape(2, 2);

  xjm_inv0(0, 0) = xjm0(1, 1) / detJ0;
  xjm_inv0(0, 1) = -xjm0(0, 1) / detJ0;
  xjm_inv0(1, 0) = -xjm0(1, 0) / detJ0;
  xjm_inv0(1, 1) = xjm0(0, 0) / detJ0;

  // build A-operator
  //
  // Q1E4
  //    A = det(J_o)/det(J) * J_o^T . sum_i^neas ( M_i alpha_i ) . J_o^{-T}
  // Q1ET4
  //    A = det(J_o)/det(J) * J_o . sum_i^neas ( M_i alpha_i ) . J_o^{-T}
  Core::LinAlg::SerialDenseMatrix A(2, 2);  // A operator
  Core::LinAlg::multiply_nt(M_temp, M, xjm_inv0);
  if (eastype_ == eas_q1e4)
    Core::LinAlg::multiply_tn(0.0, A, detJ0 / det, xjm0, M_temp);
  else if (eastype_ == eas_q1et4)
    Core::LinAlg::multiply(0.0, A, detJ0 / det, xjm0, M_temp);
  else
    FOUR_C_THROW("Cannot handle EAS type={}", eastype_);

  // enhanced deformation gradient at origin (four rows, one column)

  F_enh(0, 0) = A(0, 0) * F0(0) + A(1, 0) * F0(2);
  F_enh(1, 0) = A(1, 1) * F0(1) + A(0, 1) * F0(3);
  F_enh(2, 0) = A(0, 1) * F0(0) + A(1, 1) * F0(2);
  F_enh(3, 0) = A(1, 0) * F0(1) + A(0, 0) * F0(3);

  // get operator W0

  // write matrix A in a different way (matrix 4x4)

  Core::LinAlg::SerialDenseMatrix A_big;
  A_big.shape(4, 4);  // entries are zero

  A_big(0, 0) = A(0, 0);
  A_big(0, 2) = A(1, 0);
  A_big(1, 1) = A(1, 1);
  A_big(1, 3) = A(0, 1);
  A_big(2, 0) = A(0, 1);
  A_big(2, 2) = A(1, 1);
  A_big(3, 1) = A(1, 0);
  A_big(3, 3) = A(0, 0);

  // multiplication A_big x boplin0

  Core::LinAlg::multiply(W0, A_big, boplin0);

  // calculation operators G and Z, therefore matrices A are needed
  // without alphas

  // vector M_ges, includes the matrices M1 to M4
  std::vector<Core::LinAlg::SerialDenseMatrix> M_ges(Wall1::neas_);

  // vector A_ges, includes the matrices A1 to A4
  std::vector<Core::LinAlg::SerialDenseMatrix> A_ges(4);

  for (int ieas = 0; ieas < Wall1::neas_; ieas++)
  {
    (M_ges[ieas]).shape(2, 2);
    (A_ges[ieas]).shape(2, 2);
  }

  // fill M-Matrixes, not including eas-parameters alpha

  if (eastype_ == eas_q1e4)
  {
    (M_ges[0])(0, 0) = e1;
    (M_ges[1])(0, 1) = e2;
    (M_ges[2])(1, 0) = e1;
    (M_ges[3])(1, 1) = e2;
  }
  else if (eastype_ == eas_q1et4)
  {
    (M_ges[0])(0, 0) = e1;
    (M_ges[1])(0, 1) = e1;
    (M_ges[2])(1, 0) = e2;
    (M_ges[3])(1, 1) = e2;
  }
  else
  {
    FOUR_C_THROW("Cannot handle EAS type={}", eastype_);
  }

  // declaration of matrix (4x4) without eas-parameter alpha

  Core::LinAlg::SerialDenseMatrix Awa_big;
  Awa_big.shape(4, 4);

  // declaration of matrix WO without eas-parameter alpha

  Core::LinAlg::SerialDenseMatrix W0wa;
  W0wa.shape(4, 2 * num_node());


  for (int i = 0; i < Wall1::neas_; i++)  // loop over eas-parameter
  {
    Core::LinAlg::multiply_nt(M_temp, M_ges[i], xjm_inv0);
    if (eastype_ == eas_q1e4)
      Core::LinAlg::multiply_tn(0.0, A_ges[i], detJ0 / det, xjm0, M_temp);
    else if (eastype_ == eas_q1et4)
      Core::LinAlg::multiply(0.0, A_ges[i], detJ0 / det, xjm0, M_temp);
    else
      FOUR_C_THROW("Cannot handle EAS type={}", eastype_);

    // fill G-operator

    G(0, i) = A_ges[i](0, 0) * F0(0) + A_ges[i](1, 0) * F0(2);
    G(1, i) = A_ges[i](1, 1) * F0(1) + A_ges[i](0, 1) * F0(3);
    G(2, i) = A_ges[i](0, 1) * F0(0) + A_ges[i](1, 1) * F0(2);
    G(3, i) = A_ges[i](1, 0) * F0(1) + A_ges[i](0, 0) * F0(3);


    Awa_big.putScalar(0.0);

    Awa_big(0, 0) = A_ges[i](0, 0);
    Awa_big(0, 2) = A_ges[i](1, 0);
    Awa_big(1, 1) = A_ges[i](1, 1);
    Awa_big(1, 3) = A_ges[i](0, 1);
    Awa_big(2, 0) = A_ges[i](0, 1);
    Awa_big(2, 2) = A_ges[i](1, 1);
    Awa_big(3, 1) = A_ges[i](1, 0);
    Awa_big(3, 3) = A_ges[i](0, 0);

    // calculate operator W0wa without eas-parameters alpha

    Core::LinAlg::multiply(W0wa, Awa_big, boplin0);

    // fill Z-operator

    for (int indof = 0; indof < num_node(); indof = indof + 2)
    {
      Z(indof, i) = W0wa(0, indof);
      Z(indof + 1, i) = W0wa(1, indof + 1);
    }


  }  // end loop over eas parameter

  return;
}  // end of w1_call_fenh


/*----------------------------------------------------------------------*
 |total deformation gradient and green lagrange strain (private)mgit 01/08|
 *----------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_call_defgrad_tot(const Core::LinAlg::SerialDenseMatrix& F_enh,
    Core::LinAlg::SerialDenseMatrix& F_tot, const Core::LinAlg::SerialDenseVector& F,
    Core::LinAlg::SerialDenseVector& strain)
{
  // total deformation gradient in matrix notation
  F_tot(0, 0) = F(0) + F_enh(0, 0);
  F_tot(0, 2) = F(2) + F_enh(2, 0);
  F_tot(1, 1) = F(1) + F_enh(1, 0);
  F_tot(1, 2) = F(3) + F_enh(3, 0);
  F_tot(2, 1) = F(2) + F_enh(2, 0);
  F_tot(2, 2) = F(0) + F_enh(0, 0);
  F_tot(3, 0) = F(3) + F_enh(3, 0);
  F_tot(3, 2) = F(1) + F_enh(1, 0);


  /*-----------------------calculate Green-Lagrange strain ------------*/
  strain[0] = 0.5 * (F_tot(0, 0) * F_tot(0, 0) + F_tot(3, 0) * F_tot(3, 0) - 1.0);
  strain[1] = 0.5 * (F_tot(0, 2) * F_tot(0, 2) + F_tot(3, 2) * F_tot(3, 2) - 1.0);
  strain[2] = 0.5 * (F_tot(0, 0) * F_tot(0, 2) + F_tot(3, 0) * F_tot(3, 2));
  strain[3] = strain[2];

  return;
}  // end of w1_call_defgrad_tot

/*-----------------------------------------------------------------------------*
 |first piola-kirchhoff stress vector                       (private)mgit 02/08|
 *----------------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_stress_eas(const Core::LinAlg::SerialDenseMatrix& stress,
    const Core::LinAlg::SerialDenseMatrix& F_tot, Core::LinAlg::SerialDenseMatrix& p_stress)
{
  /*-------------reduce stress matrix-----------------------------------------*/

  Core::LinAlg::SerialDenseMatrix stress_red(3, 1, false);  // 2. piola-krichhoff (vector (3-dim))

  stress_red(0, 0) = stress(0, 0);  // S_11
  stress_red(1, 0) = stress(1, 1);  // S_22
  stress_red(2, 0) = stress(0, 2);  // S_12 (=S_21)

  /*-first piola-kirchhoff stress vector--------------------------------------*/
  Core::LinAlg::multiply(p_stress, F_tot, stress_red);

  return;
}  // end of w1_p_stress


/*-----------------------------------------------------------------------------*
| calculate stiffness matrix kdd                                     mgit 03/08|
*-----------------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_kdd(const Core::LinAlg::SerialDenseMatrix& boplin,
    const Core::LinAlg::SerialDenseMatrix& W0, const Core::LinAlg::SerialDenseMatrix& F_tot,
    const Core::LinAlg::SerialDenseMatrix& C, const Core::LinAlg::SerialDenseMatrix& stress,
    Core::LinAlg::SerialDenseMatrix& FCF, Core::LinAlg::SerialDenseMatrix& estif, const double fac)
{
  // contitutive matrix (3x3)
  Core::LinAlg::SerialDenseMatrix C_red(3, 3, false);
  C_red(0, 0) = C(0, 0);
  C_red(0, 1) = C(0, 1);
  C_red(0, 2) = C(0, 2);
  C_red(1, 0) = C(1, 0);
  C_red(1, 1) = C(1, 1);
  C_red(1, 2) = C(1, 2);
  C_red(2, 0) = C(2, 0);
  C_red(2, 1) = C(2, 1);
  C_red(2, 2) = C(2, 2);

  // BplusW = B+W0
  Core::LinAlg::SerialDenseMatrix BplusW(4, 2 * num_node(), false);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2 * num_node(); j++) BplusW(i, j) = boplin(i, j) + W0(i, j);

  // Temp (4x3) = F*C
  Core::LinAlg::SerialDenseMatrix Temp(4, 3, true);
  Core::LinAlg::multiply(Temp, F_tot, C_red);

  // FCF^T (4x4) = Temp*F^T
  FCF.putScalar(0.0);
  Core::LinAlg::multiply_nt(FCF, Temp, F_tot);

  // Temp1 (4x8) = FCF^T * (B+W0)
  Core::LinAlg::SerialDenseMatrix Temp1(4, 2 * num_node(), true);
  Core::LinAlg::multiply(Temp1, FCF, BplusW);

  // Temp3 (4x8) = S*(B+W0)
  Core::LinAlg::SerialDenseMatrix Temp3(4, 2 * num_node(), true);
  Core::LinAlg::multiply(Temp3, stress, BplusW);

  // Kdd = (B+W0)^T*FCF^T*(B+W0) + (B+W0)^T*S*(B+W0)
  Core::LinAlg::multiply_tn(1.0, estif, fac, BplusW, Temp1);
  Core::LinAlg::multiply_tn(1.0, estif, fac, BplusW, Temp3);

  return;
}  // Discret::Elements::Wall1::w1_kdd


/*-----------------------------------------------------------------------------*
| calculate matrix kda                                               mgit 03/08|
*-----------------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_kda(const Core::LinAlg::SerialDenseMatrix& FCF,
    const Core::LinAlg::SerialDenseMatrix& W0, const Core::LinAlg::SerialDenseMatrix& boplin,
    const Core::LinAlg::SerialDenseMatrix& stress, const Core::LinAlg::SerialDenseMatrix& G,
    const Core::LinAlg::SerialDenseMatrix& Z, Core::LinAlg::SerialDenseMatrix& Kda,
    const Core::LinAlg::SerialDenseMatrix& p_stress, const double fac)
{
  // BplusW = B+W0
  Core::LinAlg::SerialDenseMatrix BplusW(4, 2 * num_node(), false);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2 * num_node(); j++) BplusW(i, j) = boplin(i, j) + W0(i, j);

  // Temp1 = FCF^T*G
  Core::LinAlg::SerialDenseMatrix Temp1(4, Wall1::neas_, true);
  Core::LinAlg::multiply(Temp1, FCF, G);

  // Temp3 = S*(G)
  Core::LinAlg::SerialDenseMatrix Temp3(4, Wall1::neas_, true);
  Core::LinAlg::multiply(Temp3, stress, G);

  // Kda (8x4) = (B+W0)^T*FCF^T*G) + (B+W0)^T*S*G + PZ
  Core::LinAlg::multiply_tn(1.0, Kda, fac, BplusW, Temp1);
  Core::LinAlg::multiply_tn(1.0, Kda, fac, BplusW, Temp3);
  // Temp5 = fac * P*Z
  for (int i = 0; i < num_node(); i++)
  {
    for (int ieas = 0; ieas < Wall1::neas_; ieas++)
    {
      Kda(i * 2, ieas) +=
          (p_stress(0, 0) * Z(i * 2, ieas) + p_stress(2, 0) * Z(i * 2 + 1, ieas)) * fac;
      Kda(i * 2 + 1, ieas) +=
          (p_stress(3, 0) * Z(i * 2, ieas) + p_stress(1, 0) * Z(i * 2 + 1, ieas)) * fac;
    }
  }

  return;
}  // Discret::Elements::Wall1::w1_kda


/*-----------------------------------------------------------------------------*
| calculate matrix kaa                                               mgit 03/08|
*-----------------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_kaa(const Core::LinAlg::SerialDenseMatrix& FCF,
    const Core::LinAlg::SerialDenseMatrix& stress, const Core::LinAlg::SerialDenseMatrix& G,
    Core::LinAlg::SerialDenseMatrix& Kaa, const double fac)
{
  // Temp1 = FCF*G
  Core::LinAlg::SerialDenseMatrix Temp1(4, Wall1::neas_, true);
  Core::LinAlg::multiply(Temp1, FCF, G);

  // Temp3 = S*G
  Core::LinAlg::SerialDenseMatrix Temp3(4, Wall1::neas_, true);
  Core::LinAlg::multiply(Temp3, stress, G);

  // Kaa = G^T*FCF^T*G + G^T*S*G
  Core::LinAlg::multiply_tn(1.0, Kaa, fac, G, Temp1);
  Core::LinAlg::multiply_tn(1.0, Kaa, fac, G, Temp3);

  return;
}  // Discret::Elements::Wall1::w1_kaa


/*-----------------------------------------------------------------------------*
| calculate internal forces fint(displacements u) and feas           mgit 03/08|
*-----------------------------------------------------------------------------*/
void Discret::Elements::Wall1::w1_fint_eas(const Core::LinAlg::SerialDenseMatrix& W0,
    const Core::LinAlg::SerialDenseMatrix& boplin, const Core::LinAlg::SerialDenseMatrix& G,
    const Core::LinAlg::SerialDenseMatrix& p_stress, Core::LinAlg::SerialDenseVector& intforce,
    Core::LinAlg::SerialDenseVector& feas, const double fac)

{
  // BplusW = B+W0
  Core::LinAlg::SerialDenseMatrix BplusW(4, 2 * num_node(), false);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 2 * num_node(); j++) BplusW(i, j) = boplin(i, j) + W0(i, j);

  // Temp1 (8x1) = (BL+W0)^T*p_stress
  Core::LinAlg::SerialDenseMatrix Temp1(2 * num_node(), 1, true);
  Core::LinAlg::multiply_tn(Temp1, BplusW, p_stress);

  for (int i = 0; i < 2 * num_node(); i++) intforce(i) += fac * Temp1(i, 0);

  // Temp2 = G^T*p_stress
  Core::LinAlg::SerialDenseMatrix Temp2(Wall1::neas_, 1, true);
  Core::LinAlg::multiply_tn(Temp2, G, p_stress);

  for (int i = 0; i < Wall1::neas_; i++) feas(i) += fac * Temp2(i, 0);

  return;
}  // Discret::Elements::Wall1::w1_fint_eas


/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE
