// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beam3_triad_interpolation_local_rotation_vectors.hpp"

#include "4C_beam3_triad_interpolation.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_fad.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
LargeRotations::TriadInterpolationLocalRotationVectors<numnodes,
    T>::TriadInterpolationLocalRotationVectors()
    : node_i_(0), node_j_(0)
{
  set_node_iand_j();

  distype_ = get_dis_type();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::set_node_iand_j()
{
  // first the nodes for the reference triad \Lambda_r of the element are chosen
  // according to eq. (6.2), Crisfield 1999;
  const unsigned int nodeI = (unsigned int)std::floor(0.5 * (double)(numnodes + 1));
  const unsigned int nodeJ = (unsigned int)std::floor(0.5 * (double)(numnodes + 2));

  // The node numbering applied in Crisfield 1999 differs from the order in which nodal quantities
  // are stored in 4C.
  // Therefore we have to apply the following transformation:
  node_i_ = Core::LargeRotations::numbering_trafo(nodeI, numnodes);
  node_j_ = Core::LargeRotations::numbering_trafo(nodeJ, numnodes);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
Core::FE::CellType
LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::get_dis_type() const
{
  switch (numnodes)
  {
    case 2:
    {
      return Core::FE::CellType::line2;
    }
    case 3:
    {
      return Core::FE::CellType::line3;
    }
    case 4:
    {
      return Core::FE::CellType::line4;
    }
    case 5:
    {
      return Core::FE::CellType::line5;
    }
    default:
    {
      FOUR_C_THROW("only 2...5 nodes allowed here! got {}", numnodes);
      return Core::FE::CellType::max_distype;
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::reset(
    std::vector<Core::LinAlg::Matrix<4, 1, T>> const& nodal_quaternions)
{
  if (nodal_quaternions.size() != numnodes)
    FOUR_C_THROW("size mismatch: expected {} nodal quaternions but got {}!", numnodes,
        nodal_quaternions.size());

  // set new nodal triads
  qnode_ = nodal_quaternions;

  // compute reference triad Lambda_r according to (3.9), Jelenic 1999
  calc_ref_quaternion(nodal_quaternions[node_i_], nodal_quaternions[node_j_], q_r_);


  // rotation angles between nodal triads and reference triad according to (3.8), Jelenic 1999
  psi_li_.resize(numnodes);

  // compute nodal local rotation vectors according to (3.8), Jelenic 1999
  for (unsigned int inode = 0; inode < numnodes; ++inode)
  {
    calc_psi_li(nodal_quaternions[inode], q_r_, psi_li_[inode]);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::reset(
    std::vector<Core::LinAlg::Matrix<3, 3, T>> const& nodal_triads)
{
  if (nodal_triads.size() != numnodes)
    FOUR_C_THROW(
        "size mismatch: expected {} nodal triads but got {}!", numnodes, nodal_triads.size());

  std::vector<Core::LinAlg::Matrix<4, 1, T>> nodal_quaternions(numnodes);

  for (unsigned int inode = 0; inode < numnodes; ++inode)
  {
    Core::LargeRotations::triadtoquaternion(nodal_triads[inode], nodal_quaternions[inode]);
  }

  reset(nodal_quaternions);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes,
    T>::get_interpolated_triad_at_xi(Core::LinAlg::Matrix<3, 3, T>& triad, const double xi) const
{
  Core::LinAlg::Matrix<4, 1, T> quaternion;
  get_interpolated_quaternion_at_xi(quaternion, xi);

  Core::LargeRotations::quaterniontotriad(quaternion, triad);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::get_interpolated_triad(
    Core::LinAlg::Matrix<3, 3, T>& triad, const Core::LinAlg::Matrix<3, 1, T>& Psi_l) const
{
  Core::LinAlg::Matrix<4, 1, T> quaternion;
  get_interpolated_quaternion(quaternion, Psi_l);

  Core::LargeRotations::quaterniontotriad(quaternion, triad);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes,
    T>::get_interpolated_quaternion_at_xi(Core::LinAlg::Matrix<4, 1, T>& quaternion,
    const double xi) const
{
  // local rotation vector at xi
  Core::LinAlg::Matrix<3, 1, T> Psi_l(Core::LinAlg::Initialization::zero);
  get_interpolated_local_rotation_vector_at_xi(Psi_l, xi);

  calc_qgauss(Psi_l, q_r_, quaternion);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes,
    T>::get_interpolated_quaternion(Core::LinAlg::Matrix<4, 1, T>& quaternion,
    const Core::LinAlg::Matrix<3, 1, T>& Psi_l) const
{
  calc_qgauss(Psi_l, q_r_, quaternion);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes,
    T>::get_interpolated_local_rotation_vector_at_xi(Core::LinAlg::Matrix<3, 1, T>& Psi_l,
    const double xi) const
{
  // values of individual shape functions at xi
  Core::LinAlg::Matrix<1, numnodes> I_i(Core::LinAlg::Initialization::zero);

  Core::FE::shape_function_1d(I_i, xi, distype_);

  calc_psi_l(psi_li_, I_i, Psi_l);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes,
    T>::get_interpolated_local_rotation_vector(Core::LinAlg::Matrix<3, 1, T>& Psi_l,
    const Core::LinAlg::Matrix<1, numnodes, double>& I_i) const
{
  calc_psi_l(psi_li_, I_i, Psi_l);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes,
    T>::get_interpolated_local_rotation_vector_derivative_at_xi(Core::LinAlg::Matrix<3, 1, T>&
                                                                    Psi_l_s,
    const double jacobifac, const double xi) const
{
  // values of individual shape functions derivatives at xi
  Core::LinAlg::Matrix<1, numnodes> I_i_xi(Core::LinAlg::Initialization::zero);

  Core::FE::shape_function_1d_deriv1(I_i_xi, xi, distype_);

  calc_psi_l_s(psi_li_, I_i_xi, jacobifac, Psi_l_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes,
    T>::get_interpolated_local_rotation_vector_derivative(Core::LinAlg::Matrix<3, 1, T>& Psi_l_s,
    const Core::LinAlg::Matrix<1, numnodes, double>& I_i_xi, const double jacobifac) const
{
  calc_psi_l_s(psi_li_, I_i_xi, jacobifac, Psi_l_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::
    get_nodal_generalized_rotation_interpolation_matrices_at_xi(
        std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itilde, const double xi) const
{
  // transform stored reference quaternion to triad
  Core::LinAlg::Matrix<3, 3, T> Lambda_r(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::quaterniontotriad(q_r_, Lambda_r);

  // compute angle of relative rotation between node I and J
  Core::LinAlg::Matrix<3, 1, T> Phi_IJ(Core::LinAlg::Initialization::zero);
  calc_phi_ij(qnode_[node_i_], qnode_[node_j_], Phi_IJ);

  // values of individual shape functions at xi
  Core::LinAlg::Matrix<1, numnodes> I_i(Core::LinAlg::Initialization::zero);
  Core::FE::shape_function_1d(I_i, xi, distype_);

  // compute interpolated local relative rotation vector \Psi^l
  Core::LinAlg::Matrix<3, 1, T> Psi_l(Core::LinAlg::Initialization::zero);
  calc_psi_l(psi_li_, I_i, Psi_l);


  compute_itilde(Psi_l, Itilde, Phi_IJ, Lambda_r, psi_li_, I_i);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::
    get_nodal_generalized_rotation_interpolation_matrices(
        std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itilde,
        const Core::LinAlg::Matrix<3, 1, T>& Psi_l,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i) const
{
  // transform stored reference quaternion to triad
  Core::LinAlg::Matrix<3, 3, T> Lambda_r(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::quaterniontotriad(q_r_, Lambda_r);

  // compute angle of relative rotation between node I and J
  Core::LinAlg::Matrix<3, 1, T> Phi_IJ(Core::LinAlg::Initialization::zero);
  calc_phi_ij(qnode_[node_i_], qnode_[node_j_], Phi_IJ);


  compute_itilde(Psi_l, Itilde, Phi_IJ, Lambda_r, psi_li_, I_i);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::
    get_nodal_generalized_rotation_interpolation_matrices_derivative(
        std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itilde_prime,
        const Core::LinAlg::Matrix<3, 1, T>& Psi_l, const Core::LinAlg::Matrix<3, 1, T>& Psi_l_s,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i_xi, const double jacobifac) const
{
  Core::LinAlg::Matrix<1, numnodes, double> I_i_s(I_i_xi);
  I_i_s.scale(std::pow(jacobifac, -1.0));

  get_nodal_generalized_rotation_interpolation_matrices_derivative(
      Itilde_prime, Psi_l, Psi_l_s, I_i, I_i_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::
    get_nodal_generalized_rotation_interpolation_matrices_derivative(
        std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itilde_prime,
        const Core::LinAlg::Matrix<3, 1, T>& Psi_l, const Core::LinAlg::Matrix<3, 1, T>& Psi_l_s,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i,
        const Core::LinAlg::Matrix<1, numnodes, double>& I_i_s) const
{
  // transform stored reference quaternion to triad
  Core::LinAlg::Matrix<3, 3, T> Lambda_r(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::quaterniontotriad(q_r_, Lambda_r);

  // compute angle of relative rotation between node I and J
  Core::LinAlg::Matrix<3, 1, T> Phi_IJ(Core::LinAlg::Initialization::zero);
  calc_phi_ij(qnode_[node_i_], qnode_[node_j_], Phi_IJ);


  compute_itildeprime(Psi_l, Psi_l_s, Itilde_prime, Phi_IJ, Lambda_r, psi_li_, I_i, I_i_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::calc_ref_quaternion(
    const Core::LinAlg::Matrix<4, 1, T>& Q_nodeI, const Core::LinAlg::Matrix<4, 1, T>& Q_nodeJ,
    Core::LinAlg::Matrix<4, 1, T>& Q_r) const
{
  Q_r.clear();
  Core::LinAlg::Matrix<3, 1, T> Phi_IJ(Core::LinAlg::Initialization::zero);

  // compute angle of relative rotation between node I and J
  calc_phi_ij(Q_nodeI, Q_nodeJ, Phi_IJ);

  Core::LinAlg::Matrix<3, 1, T> Phi_IJhalf(Phi_IJ);
  Phi_IJhalf.scale(0.5);

  // quaternion of half relative rotation between node I and J according to (3.9), Jelenic 1999
  Core::LinAlg::Matrix<4, 1, T> QIJhalf(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::angletoquaternion<T>(Phi_IJhalf, QIJhalf);

  Core::LargeRotations::quaternionproduct(QIJhalf, Q_nodeI, Q_r);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::calc_phi_ij(
    const Core::LinAlg::Matrix<4, 1, T>& Q_nodeI, const Core::LinAlg::Matrix<4, 1, T>& Q_nodeJ,
    Core::LinAlg::Matrix<3, 1, T>& Phi_IJ) const
{
  // angle and quaternion of relative rotation between node I and J
  Phi_IJ.clear();
  Core::LinAlg::Matrix<4, 1, T> QIJ(Core::LinAlg::Initialization::zero);

  // computation according to (3.10), Jelenic 1999
  Core::LargeRotations::quaternionproduct(
      Q_nodeJ, Core::LargeRotations::inversequaternion(Q_nodeI), QIJ);
  Core::LargeRotations::quaterniontoangle(QIJ, Phi_IJ);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::calc_psi_li(
    const Core::LinAlg::Matrix<4, 1, T>& Q_i, const Core::LinAlg::Matrix<4, 1, T>& Q_r,
    Core::LinAlg::Matrix<3, 1, T>& Psi_li) const
{
  // angle and quaternion of local rotation vectors at nodes i=0...numnodes
  Psi_li.clear();
  Core::LinAlg::Matrix<4, 1, T> Q_li(Core::LinAlg::Initialization::zero);

  Core::LargeRotations::quaternionproduct(
      Q_i, Core::LargeRotations::inversequaternion<T>(Q_r), Q_li);
  Core::LargeRotations::quaterniontoangle<T>(Q_li, Psi_li);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::calc_psi_l(
    const std::vector<Core::LinAlg::Matrix<3, 1, T>>& Psi_li,
    const Core::LinAlg::Matrix<1, numnodes, double>& func,
    Core::LinAlg::Matrix<3, 1, T>& Psi_l) const
{
  Psi_l.clear();

  for (unsigned int dof = 0; dof < 3; ++dof)
    for (unsigned int node = 0; node < numnodes; ++node)
      Psi_l(dof) += func(node) * (Psi_li[node])(dof);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::calc_psi_l_s(
    const std::vector<Core::LinAlg::Matrix<3, 1, T>>& Psi_li,
    const Core::LinAlg::Matrix<1, numnodes, double>& deriv_xi, const double& jacobi,
    Core::LinAlg::Matrix<3, 1, T>& Psi_l_s) const
{
  Psi_l_s.clear();

  for (unsigned int dof = 0; dof < 3; ++dof)
    for (unsigned int node = 0; node < numnodes; ++node)
      Psi_l_s(dof) += deriv_xi(node) * (Psi_li[node])(dof);

  /* at this point we have computed derivative with respect to the element parameter \xi \in [-1;1];
   * as we want a derivatives with respect to the reference arc-length parameter s,
   * we have to divide it by the Jacobi determinant at the respective point */
  Psi_l_s.scale(1.0 / jacobi);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::calc_lambda(
    const Core::LinAlg::Matrix<3, 1, T>& Psi_l, const Core::LinAlg::Matrix<4, 1, T>& Q_r,
    Core::LinAlg::Matrix<3, 3, T>& Lambda) const
{
  Lambda.clear();

  Core::LinAlg::Matrix<4, 1, T> Ql;
  Core::LinAlg::Matrix<4, 1, T> Qgauss;

  // c ompute relative rotation between triad at Gauss point and reference triad Qr
  Core::LargeRotations::angletoquaternion(Psi_l, Ql);

  // compute rotation at Gauss point, i.e. the quaternion equivalent to \Lambda(s) in
  // Crisfield 1999, eq. (4.7)
  Core::LargeRotations::quaternionproduct(Ql, Q_r, Qgauss);

  // compute rotation matrix at Gauss point, i.e. \Lambda(s) in Crisfield 1999, eq. (4.7)
  Core::LargeRotations::quaterniontotriad(Qgauss, Lambda);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::calc_qgauss(
    const Core::LinAlg::Matrix<3, 1, T>& Psi_l, const Core::LinAlg::Matrix<4, 1, T>& Q_r,
    Core::LinAlg::Matrix<4, 1, T>& Qgauss) const
{
  Qgauss.clear();

  Core::LinAlg::Matrix<4, 1, T> Ql;

  // compute relative rotation between triad at Gauss point and reference triad Qr
  Core::LargeRotations::angletoquaternion(Psi_l, Ql);

  // compute rotation at Gauss point, i.e. the quaternion equivalent to \Lambda(s) in
  // Crisfield 1999, eq. (4.7)
  Core::LargeRotations::quaternionproduct(Ql, Q_r, Qgauss);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::compute_itilde(
    const Core::LinAlg::Matrix<3, 1, T>& Psil, std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itilde,
    const Core::LinAlg::Matrix<3, 1, T>& phiIJ, const Core::LinAlg::Matrix<3, 3, T>& Lambdar,
    const std::vector<Core::LinAlg::Matrix<3, 1, T>>& Psili,
    const Core::LinAlg::Matrix<1, numnodes, double>& funct) const
{
  // auxiliary matrices for storing intermediate results
  Core::LinAlg::Matrix<3, 3, T> auxmatrix(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, T> auxmatrix2(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3, T> Tinv_Psil = Core::LargeRotations::tinvmatrix(Psil);

  // make sure that Itilde has proper dimensions
  Itilde.resize(numnodes);

  // compute squared brackets term in (3.18), Jelenic 1999
  Core::LinAlg::Matrix<3, 3, T> squaredbrackets(Core::LinAlg::Initialization::zero);

  for (unsigned int node = 0; node < numnodes; ++node)
  {
    auxmatrix.clear();

    auxmatrix = Core::LargeRotations::tmatrix(Psili[node]);
    auxmatrix.scale(funct(node));
    auxmatrix2.update(-1.0, auxmatrix, 1.0);
  }

  squaredbrackets.multiply(Tinv_Psil, auxmatrix2);

  for (unsigned int i = 0; i < 3; i++) squaredbrackets(i, i) += 1;

  Core::LinAlg::Matrix<3, 3, T> v_matrix(Core::LinAlg::Initialization::zero);

  // loop through all nodes i
  for (unsigned int node = 0; node < numnodes; ++node)
  {
    // compute rightmost term in curley brackets in (3.18), Jelenic 1999
    Itilde[node].clear();
    Itilde[node].multiply(Tinv_Psil, Core::LargeRotations::tmatrix(Psili[node]));
    Itilde[node].scale(funct(node));

    // if node i is node I then add squared brackets term times v_I
    if (node == node_i_)
    {
      calc_v_i(v_matrix, phiIJ);
      auxmatrix.multiply(squaredbrackets, v_matrix);
      Itilde[node] += auxmatrix;
    }

    // if node i is node J then add squared brackets term times v_J
    if (node == node_j_)
    {
      calc_v_j(v_matrix, phiIJ);
      auxmatrix.multiply(squaredbrackets, v_matrix);
      Itilde[node] += auxmatrix;
    }

    // now the term in the curly brackets has been computed and has to be rotated by \Lambda_r
    auxmatrix.multiply_nt(Itilde[node], Lambdar);
    Itilde[node].multiply_nn(Lambdar, auxmatrix);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::compute_itildeprime(
    const Core::LinAlg::Matrix<3, 1, T>& Psil, const Core::LinAlg::Matrix<3, 1, T>& Psilprime,
    std::vector<Core::LinAlg::Matrix<3, 3, T>>& Itildeprime,
    const Core::LinAlg::Matrix<3, 1, T>& phiIJ, const Core::LinAlg::Matrix<3, 3, T>& Lambdar,
    const std::vector<Core::LinAlg::Matrix<3, 1, T>>& Psili,
    const Core::LinAlg::Matrix<1, numnodes, double>& funct,
    const Core::LinAlg::Matrix<1, numnodes, double>& deriv_s) const
{
  // auxiliary matrices for storing intermediate results
  Core::LinAlg::Matrix<3, 3, T> auxmatrix(Core::LinAlg::Initialization::zero);

  // make sure that Itildeprime has proper dimensions
  Itildeprime.resize(numnodes);

  // matrix d(T^{-1})/dx
  Core::LinAlg::Matrix<3, 3, T> dTinvdx(Core::LinAlg::Initialization::zero);
  Core::LargeRotations::computed_tinvdx(Psil, Psilprime, dTinvdx);

  // compute T^{~} according to remark subsequent to (3.19), Jelenic 1999
  Core::LinAlg::Matrix<3, 3, T> Ttilde(Core::LinAlg::Initialization::zero);
  for (unsigned int node = 0; node < numnodes; ++node)
  {
    auxmatrix = Core::LargeRotations::tmatrix(Psili[node]);
    auxmatrix.scale(funct(node));
    Ttilde += auxmatrix;
  }

  // compute T^{~'} according to remark subsequent to (3.19), Jelenic 1999
  Core::LinAlg::Matrix<3, 3, T> Ttildeprime(Core::LinAlg::Initialization::zero);
  for (unsigned int node = 0; node < numnodes; ++node)
  {
    auxmatrix = Core::LargeRotations::tmatrix(Psili[node]);
    auxmatrix.scale(deriv_s(node));
    Ttildeprime += auxmatrix;
  }

  // compute first squared brackets term in (3.18), Jelenic 1999
  Core::LinAlg::Matrix<3, 3, T> squaredbrackets(Core::LinAlg::Initialization::zero);
  squaredbrackets.multiply(dTinvdx, Ttilde);
  auxmatrix.multiply(Core::LargeRotations::tinvmatrix(Psil), Ttildeprime);
  squaredbrackets += auxmatrix;

  Core::LinAlg::Matrix<3, 3, T> v_matrix(Core::LinAlg::Initialization::zero);

  // loop through all nodes i
  for (unsigned int node = 0; node < numnodes; ++node)
  {
    // compute first term in second squared brackets
    Itildeprime[node] = dTinvdx;
    Itildeprime[node].scale(funct(node));

    // compute second term in second squared brackets
    auxmatrix.clear();
    auxmatrix += Core::LargeRotations::tinvmatrix(Psil);
    auxmatrix.scale(deriv_s(node));

    // compute second squared brackets
    auxmatrix += Itildeprime[node];

    // compute second squared brackets time T(\Psi^l_j)
    Itildeprime[node].multiply(auxmatrix, Core::LargeRotations::tmatrix(Psili[node]));

    // if node i is node I then add first squared brackets term times v_I
    if (node == node_i_)
    {
      calc_v_i(v_matrix, phiIJ);
      auxmatrix.multiply(squaredbrackets, v_matrix);
      Itildeprime[node] -= auxmatrix;
    }

    // if node i is node J then add first squared brackets term times v_J
    if (node == node_j_)
    {
      calc_v_j(v_matrix, phiIJ);
      auxmatrix.multiply(squaredbrackets, v_matrix);
      Itildeprime[node] -= auxmatrix;
    }

    // now the term in the curly brackets has been computed and has to be rotated by \Lambda_r
    auxmatrix.multiply_nt(Itildeprime[node], Lambdar);
    Itildeprime[node].multiply_nn(Lambdar, auxmatrix);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::calc_v_i(
    Core::LinAlg::Matrix<3, 3, T>& vI, const Core::LinAlg::Matrix<3, 1, T>& phiIJ) const
{
  // matrix v_I
  vI.clear();

  Core::LargeRotations::computespin(vI, phiIJ);
  // Fixme @grill: think about introducing a tolerance here to avoid singularity
  if (Core::FADUtils::vector_norm(phiIJ) == 0.0)
    vI.scale(0.25);
  else  // Fixme @grill: why do we cast to double here?
    vI.scale(std::tan(Core::FADUtils::cast_to_double(Core::FADUtils::vector_norm(phiIJ)) / 4.0) /
             Core::FADUtils::vector_norm(phiIJ));

  for (unsigned int i = 0; i < 3; i++) vI(i, i) += 1.0;

  vI.scale(0.5);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LargeRotations::TriadInterpolationLocalRotationVectors<numnodes, T>::calc_v_j(
    Core::LinAlg::Matrix<3, 3, T>& vJ, const Core::LinAlg::Matrix<3, 1, T>& phiIJ) const
{
  // matrix v_J
  vJ.clear();

  Core::LargeRotations::computespin(vJ, phiIJ);
  // Fixme @grill: think about introducing a tolerance here to avoid singularity
  if (Core::FADUtils::vector_norm(phiIJ) == 0.0)
    vJ.scale(-0.25);
  else  // Fixme why do we cast to double here?
    vJ.scale(-1.0 *
             std::tan(Core::FADUtils::cast_to_double(Core::FADUtils::vector_norm(phiIJ)) / 4.0) /
             Core::FADUtils::vector_norm(phiIJ));

  for (unsigned int i = 0; i < 3; i++) vJ(i, i) += 1.0;

  vJ.scale(0.5);
}

// explicit template instantiations
template class LargeRotations::TriadInterpolationLocalRotationVectors<2, double>;
template class LargeRotations::TriadInterpolationLocalRotationVectors<3, double>;
template class LargeRotations::TriadInterpolationLocalRotationVectors<4, double>;
template class LargeRotations::TriadInterpolationLocalRotationVectors<5, double>;
template class LargeRotations::TriadInterpolationLocalRotationVectors<2, Sacado::Fad::DFad<double>>;
template class LargeRotations::TriadInterpolationLocalRotationVectors<3, Sacado::Fad::DFad<double>>;
template class LargeRotations::TriadInterpolationLocalRotationVectors<4, Sacado::Fad::DFad<double>>;
template class LargeRotations::TriadInterpolationLocalRotationVectors<5, Sacado::Fad::DFad<double>>;

FOUR_C_NAMESPACE_CLOSE
