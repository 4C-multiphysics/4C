/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief a triad interpolation scheme based on local rotation vectors


\level 3
*/
/*-----------------------------------------------------------------------------------------------*/

#include "beam3_triad_interpolation_local_rotation_vectors.H"
#include "beam3_triad_interpolation.H"

#include "lib_element.H"

#include "discretization_fem_general_largerotations.H"
#include "discretization_fem_general_utils_fem_shapefunctions.H"

#include "linalg_fixedsizematrix.H"

#include "linalg_FAD_utils.H"

#include <Sacado.hpp>

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes,
    T>::TriadInterpolationLocalRotationVectors()
    : nodeI_(0), nodeJ_(0)
{
  SetNodeIandJ();

  distype_ = GetDisType();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::SetNodeIandJ()
{
  // first the nodes for the reference triad \Lambda_r of the element are chosen
  // according to eq. (6.2), Crisfield 1999;
  const unsigned int nodeI = (unsigned int)std::floor(0.5 * (double)(numnodes + 1));
  const unsigned int nodeJ = (unsigned int)std::floor(0.5 * (double)(numnodes + 2));

  // The node numbering applied in Crisfield 1999 differs from the order in which nodal quantities
  // are stored in BACI.
  // Therefore we have to apply the following transformation:
  nodeI_ = LARGEROTATIONS::NumberingTrafo(nodeI, numnodes);
  nodeJ_ = LARGEROTATIONS::NumberingTrafo(nodeJ, numnodes);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
DRT::Element::DiscretizationType
LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::GetDisType() const
{
  switch (numnodes)
  {
    case 2:
    {
      return DRT::Element::line2;
    }
    case 3:
    {
      return DRT::Element::line3;
    }
    case 4:
    {
      return DRT::Element::line4;
    }
    case 5:
    {
      return DRT::Element::line5;
    }
    default:
    {
      dserror("only 2...5 nodes allowed here! got %d", numnodes);
      return DRT::Element::max_distype;
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::Reset(
    std::vector<LINALG::Matrix<4, 1, T>> const& nodal_quaternions)
{
  if (nodal_quaternions.size() != numnodes)
    dserror("size mismatch: expected %d nodal quaternions but got %d!", numnodes,
        nodal_quaternions.size());

  // set new nodal triads
  Qnode_ = nodal_quaternions;

  // compute reference triad Lambda_r according to (3.9), Jelenic 1999
  CalcRefQuaternion(nodal_quaternions[nodeI_], nodal_quaternions[nodeJ_], Q_r_);


  // rotation angles between nodal triads and reference triad according to (3.8), Jelenic 1999
  Psi_li_.resize(numnodes);

  // compute nodal local rotation vectors according to (3.8), Jelenic 1999
  for (unsigned int inode = 0; inode < numnodes; ++inode)
  {
    CalcPsi_li(nodal_quaternions[inode], Q_r_, Psi_li_[inode]);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::Reset(
    std::vector<LINALG::Matrix<3, 3, T>> const& nodal_triads)
{
  if (nodal_triads.size() != numnodes)
    dserror("size mismatch: expected %d nodal triads but got %d!", numnodes, nodal_triads.size());

  std::vector<LINALG::Matrix<4, 1, T>> nodal_quaternions(numnodes);

  for (unsigned int inode = 0; inode < numnodes; ++inode)
  {
    LARGEROTATIONS::triadtoquaternion(nodal_triads[inode], nodal_quaternions[inode]);
  }

  Reset(nodal_quaternions);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::GetInterpolatedTriadAtXi(
    LINALG::Matrix<3, 3, T>& triad, const double xi) const
{
  LINALG::Matrix<4, 1, T> quaternion;
  GetInterpolatedQuaternionAtXi(quaternion, xi);

  LARGEROTATIONS::quaterniontotriad(quaternion, triad);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::GetInterpolatedTriad(
    LINALG::Matrix<3, 3, T>& triad, const LINALG::Matrix<3, 1, T>& Psi_l) const
{
  LINALG::Matrix<4, 1, T> quaternion;
  GetInterpolatedQuaternion(quaternion, Psi_l);

  LARGEROTATIONS::quaterniontotriad(quaternion, triad);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes,
    T>::GetInterpolatedQuaternionAtXi(LINALG::Matrix<4, 1, T>& quaternion, const double xi) const
{
  // local rotation vector at xi
  LINALG::Matrix<3, 1, T> Psi_l(true);
  GetInterpolatedLocalRotationVectorAtXi(Psi_l, xi);

  Calc_Qgauss(Psi_l, Q_r_, quaternion);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::GetInterpolatedQuaternion(
    LINALG::Matrix<4, 1, T>& quaternion, const LINALG::Matrix<3, 1, T>& Psi_l) const
{
  Calc_Qgauss(Psi_l, Q_r_, quaternion);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes,
    T>::GetInterpolatedLocalRotationVectorAtXi(LINALG::Matrix<3, 1, T>& Psi_l,
    const double xi) const
{
  // values of individual shape functions at xi
  LINALG::Matrix<1, numnodes> I_i(true);

  DRT::UTILS::shape_function_1D(I_i, xi, distype_);

  Calc_Psi_l(Psi_li_, I_i, Psi_l);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes,
    T>::GetInterpolatedLocalRotationVector(LINALG::Matrix<3, 1, T>& Psi_l,
    const LINALG::Matrix<1, numnodes, double>& I_i) const
{
  Calc_Psi_l(Psi_li_, I_i, Psi_l);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes,
    T>::GetInterpolatedLocalRotationVectorDerivativeAtXi(LINALG::Matrix<3, 1, T>& Psi_l_s,
    const double jacobifac, const double xi) const
{
  // values of individual shape functions derivatives at xi
  LINALG::Matrix<1, numnodes> I_i_xi(true);

  DRT::UTILS::shape_function_1D_deriv1(I_i_xi, xi, distype_);

  Calc_Psi_l_s(Psi_li_, I_i_xi, jacobifac, Psi_l_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes,
    T>::GetInterpolatedLocalRotationVectorDerivative(LINALG::Matrix<3, 1, T>& Psi_l_s,
    const LINALG::Matrix<1, numnodes, double>& I_i_xi, const double jacobifac) const
{
  Calc_Psi_l_s(Psi_li_, I_i_xi, jacobifac, Psi_l_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes,
    T>::GetNodalGeneralizedRotationInterpolationMatricesAtXi(std::vector<LINALG::Matrix<3, 3, T>>&
                                                                 Itilde,
    const double xi) const
{
  // transform stored reference quaternion to triad
  LINALG::Matrix<3, 3, T> Lambda_r(true);
  LARGEROTATIONS::quaterniontotriad(Q_r_, Lambda_r);

  // compute angle of relative rotation between node I and J
  LINALG::Matrix<3, 1, T> Phi_IJ(true);
  CalcPhi_IJ(Qnode_[nodeI_], Qnode_[nodeJ_], Phi_IJ);

  // values of individual shape functions at xi
  LINALG::Matrix<1, numnodes> I_i(true);
  DRT::UTILS::shape_function_1D(I_i, xi, distype_);

  // compute interpolated local relative rotation vector \Psi^l
  LINALG::Matrix<3, 1, T> Psi_l(true);
  Calc_Psi_l(Psi_li_, I_i, Psi_l);


  computeItilde(Psi_l, Itilde, Phi_IJ, Lambda_r, Psi_li_, I_i);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes,
    T>::GetNodalGeneralizedRotationInterpolationMatrices(std::vector<LINALG::Matrix<3, 3, T>>&
                                                             Itilde,
    const LINALG::Matrix<3, 1, T>& Psi_l, const LINALG::Matrix<1, numnodes, double>& I_i) const
{
  // transform stored reference quaternion to triad
  LINALG::Matrix<3, 3, T> Lambda_r(true);
  LARGEROTATIONS::quaterniontotriad(Q_r_, Lambda_r);

  // compute angle of relative rotation between node I and J
  LINALG::Matrix<3, 1, T> Phi_IJ(true);
  CalcPhi_IJ(Qnode_[nodeI_], Qnode_[nodeJ_], Phi_IJ);


  computeItilde(Psi_l, Itilde, Phi_IJ, Lambda_r, Psi_li_, I_i);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::
    GetNodalGeneralizedRotationInterpolationMatricesDerivative(
        std::vector<LINALG::Matrix<3, 3, T>>& Itilde_prime, const LINALG::Matrix<3, 1, T>& Psi_l,
        const LINALG::Matrix<3, 1, T>& Psi_l_s, const LINALG::Matrix<1, numnodes, double>& I_i,
        const LINALG::Matrix<1, numnodes, double>& I_i_xi, const double jacobifac) const
{
  LINALG::Matrix<1, numnodes, double> I_i_s(I_i_xi);
  I_i_s.Scale(std::pow(jacobifac, -1.0));

  GetNodalGeneralizedRotationInterpolationMatricesDerivative(
      Itilde_prime, Psi_l, Psi_l_s, I_i, I_i_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::
    GetNodalGeneralizedRotationInterpolationMatricesDerivative(
        std::vector<LINALG::Matrix<3, 3, T>>& Itilde_prime, const LINALG::Matrix<3, 1, T>& Psi_l,
        const LINALG::Matrix<3, 1, T>& Psi_l_s, const LINALG::Matrix<1, numnodes, double>& I_i,
        const LINALG::Matrix<1, numnodes, double>& I_i_s) const
{
  // transform stored reference quaternion to triad
  LINALG::Matrix<3, 3, T> Lambda_r(true);
  LARGEROTATIONS::quaterniontotriad(Q_r_, Lambda_r);

  // compute angle of relative rotation between node I and J
  LINALG::Matrix<3, 1, T> Phi_IJ(true);
  CalcPhi_IJ(Qnode_[nodeI_], Qnode_[nodeJ_], Phi_IJ);


  computeItildeprime(Psi_l, Psi_l_s, Itilde_prime, Phi_IJ, Lambda_r, Psi_li_, I_i, I_i_s);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::CalcRefQuaternion(
    const LINALG::Matrix<4, 1, T>& Q_nodeI, const LINALG::Matrix<4, 1, T>& Q_nodeJ,
    LINALG::Matrix<4, 1, T>& Q_r) const
{
  Q_r.Clear();
  LINALG::Matrix<3, 1, T> Phi_IJ(true);

  // compute angle of relative rotation between node I and J
  CalcPhi_IJ(Q_nodeI, Q_nodeJ, Phi_IJ);

  LINALG::Matrix<3, 1, T> Phi_IJhalf(Phi_IJ);
  Phi_IJhalf.Scale(0.5);

  // quaternion of half relative rotation between node I and J according to (3.9), Jelenic 1999
  LINALG::Matrix<4, 1, T> QIJhalf(true);
  LARGEROTATIONS::angletoquaternion<T>(Phi_IJhalf, QIJhalf);

  LARGEROTATIONS::quaternionproduct(QIJhalf, Q_nodeI, Q_r);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::CalcPhi_IJ(
    const LINALG::Matrix<4, 1, T>& Q_nodeI, const LINALG::Matrix<4, 1, T>& Q_nodeJ,
    LINALG::Matrix<3, 1, T>& Phi_IJ) const
{
  // angle and quaternion of relative rotation between node I and J
  Phi_IJ.Clear();
  LINALG::Matrix<4, 1, T> QIJ(true);

  // computation according to (3.10), Jelenic 1999
  LARGEROTATIONS::quaternionproduct(Q_nodeJ, LARGEROTATIONS::inversequaternion(Q_nodeI), QIJ);
  LARGEROTATIONS::quaterniontoangle(QIJ, Phi_IJ);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::CalcPsi_li(
    const LINALG::Matrix<4, 1, T>& Q_i, const LINALG::Matrix<4, 1, T>& Q_r,
    LINALG::Matrix<3, 1, T>& Psi_li) const
{
  // angle and quaternion of local rotation vectors at nodes i=0...numnodes
  Psi_li.Clear();
  LINALG::Matrix<4, 1, T> Q_li(true);

  LARGEROTATIONS::quaternionproduct(Q_i, LARGEROTATIONS::inversequaternion<T>(Q_r), Q_li);
  LARGEROTATIONS::quaterniontoangle<T>(Q_li, Psi_li);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::Calc_Psi_l(
    const std::vector<LINALG::Matrix<3, 1, T>>& Psi_li,
    const LINALG::Matrix<1, numnodes, double>& func, LINALG::Matrix<3, 1, T>& Psi_l) const
{
  Psi_l.Clear();

  for (unsigned int dof = 0; dof < 3; ++dof)
    for (unsigned int node = 0; node < numnodes; ++node)
      Psi_l(dof) += func(node) * (Psi_li[node])(dof);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::Calc_Psi_l_s(
    const std::vector<LINALG::Matrix<3, 1, T>>& Psi_li,
    const LINALG::Matrix<1, numnodes, double>& deriv_xi, const double& jacobi,
    LINALG::Matrix<3, 1, T>& Psi_l_s) const
{
  Psi_l_s.Clear();

  for (unsigned int dof = 0; dof < 3; ++dof)
    for (unsigned int node = 0; node < numnodes; ++node)
      Psi_l_s(dof) += deriv_xi(node) * (Psi_li[node])(dof);

  /* at this point we have computed derivative with respect to the element parameter \xi \in [-1;1];
   * as we want a derivatives with respect to the reference arc-length parameter s,
   * we have to divide it by the Jacobi determinant at the respective point */
  Psi_l_s.Scale(1.0 / jacobi);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::Calc_Lambda(
    const LINALG::Matrix<3, 1, T>& Psi_l, const LINALG::Matrix<4, 1, T>& Q_r,
    LINALG::Matrix<3, 3, T>& Lambda) const
{
  Lambda.Clear();

  LINALG::Matrix<4, 1, T> Ql;
  LINALG::Matrix<4, 1, T> Qgauss;

  // c ompute relative rotation between triad at Gauss point and reference triad Qr
  LARGEROTATIONS::angletoquaternion(Psi_l, Ql);

  // compute rotation at Gauss point, i.e. the quaternion equivalent to \Lambda(s) in
  // Crisfield 1999, eq. (4.7)
  LARGEROTATIONS::quaternionproduct(Ql, Q_r, Qgauss);

  // compute rotation matrix at Gauss point, i.e. \Lambda(s) in Crisfield 1999, eq. (4.7)
  LARGEROTATIONS::quaterniontotriad(Qgauss, Lambda);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::Calc_Qgauss(
    const LINALG::Matrix<3, 1, T>& Psi_l, const LINALG::Matrix<4, 1, T>& Q_r,
    LINALG::Matrix<4, 1, T>& Qgauss) const
{
  Qgauss.Clear();

  LINALG::Matrix<4, 1, T> Ql;

  // compute relative rotation between triad at Gauss point and reference triad Qr
  LARGEROTATIONS::angletoquaternion(Psi_l, Ql);

  // compute rotation at Gauss point, i.e. the quaternion equivalent to \Lambda(s) in
  // Crisfield 1999, eq. (4.7)
  LARGEROTATIONS::quaternionproduct(Ql, Q_r, Qgauss);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::computeItilde(
    const LINALG::Matrix<3, 1, T>& Psil, std::vector<LINALG::Matrix<3, 3, T>>& Itilde,
    const LINALG::Matrix<3, 1, T>& phiIJ, const LINALG::Matrix<3, 3, T>& Lambdar,
    const std::vector<LINALG::Matrix<3, 1, T>>& Psili,
    const LINALG::Matrix<1, numnodes, double>& funct) const
{
  // auxiliary matrices for storing intermediate results
  LINALG::Matrix<3, 3, T> auxmatrix(true);
  LINALG::Matrix<3, 3, T> auxmatrix2(true);

  LINALG::Matrix<3, 3, T> Tinv_Psil = LARGEROTATIONS::Tinvmatrix(Psil);

  // make sure that Itilde has proper dimensions
  Itilde.resize(numnodes);

  // compute squared brackets term in (3.18), Jelenic 1999
  LINALG::Matrix<3, 3, T> squaredbrackets(true);

  for (unsigned int node = 0; node < numnodes; ++node)
  {
    auxmatrix.Clear();

    auxmatrix = LARGEROTATIONS::Tmatrix(Psili[node]);
    auxmatrix.Scale(funct(node));
    auxmatrix2.Update(-1.0, auxmatrix, 1.0);
  }

  squaredbrackets.Multiply(Tinv_Psil, auxmatrix2);

  for (unsigned int i = 0; i < 3; i++) squaredbrackets(i, i) += 1;

  LINALG::Matrix<3, 3, T> v_matrix(true);

  // loop through all nodes i
  for (unsigned int node = 0; node < numnodes; ++node)
  {
    // compute rightmost term in curley brackets in (3.18), Jelenic 1999
    Itilde[node].Clear();
    Itilde[node].Multiply(Tinv_Psil, LARGEROTATIONS::Tmatrix(Psili[node]));
    Itilde[node].Scale(funct(node));

    // if node i is node I then add squared brackets term times v_I
    if (node == nodeI_)
    {
      Calc_vI(v_matrix, phiIJ);
      auxmatrix.Multiply(squaredbrackets, v_matrix);
      Itilde[node] += auxmatrix;
    }

    // if node i is node J then add squared brackets term times v_J
    if (node == nodeJ_)
    {
      Calc_vJ(v_matrix, phiIJ);
      auxmatrix.Multiply(squaredbrackets, v_matrix);
      Itilde[node] += auxmatrix;
    }

    // now the term in the curly brackets has been computed and has to be rotated by \Lambda_r
    auxmatrix.MultiplyNT(Itilde[node], Lambdar);
    Itilde[node].MultiplyNN(Lambdar, auxmatrix);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::computeItildeprime(
    const LINALG::Matrix<3, 1, T>& Psil, const LINALG::Matrix<3, 1, T>& Psilprime,
    std::vector<LINALG::Matrix<3, 3, T>>& Itildeprime, const LINALG::Matrix<3, 1, T>& phiIJ,
    const LINALG::Matrix<3, 3, T>& Lambdar, const std::vector<LINALG::Matrix<3, 1, T>>& Psili,
    const LINALG::Matrix<1, numnodes, double>& funct,
    const LINALG::Matrix<1, numnodes, double>& deriv_s) const
{
  // auxiliary matrices for storing intermediate results
  LINALG::Matrix<3, 3, T> auxmatrix(true);

  // make sure that Itildeprime has proper dimensions
  Itildeprime.resize(numnodes);

  // matrix d(T^{-1})/dx
  LINALG::Matrix<3, 3, T> dTinvdx(true);
  LARGEROTATIONS::computedTinvdx(Psil, Psilprime, dTinvdx);

  // compute T^{~} according to remark subsequent to (3.19), Jelenic 1999
  LINALG::Matrix<3, 3, T> Ttilde(true);
  for (unsigned int node = 0; node < numnodes; ++node)
  {
    auxmatrix = LARGEROTATIONS::Tmatrix(Psili[node]);
    auxmatrix.Scale(funct(node));
    Ttilde += auxmatrix;
  }

  // compute T^{~'} according to remark subsequent to (3.19), Jelenic 1999
  LINALG::Matrix<3, 3, T> Ttildeprime(true);
  for (unsigned int node = 0; node < numnodes; ++node)
  {
    auxmatrix = LARGEROTATIONS::Tmatrix(Psili[node]);
    auxmatrix.Scale(deriv_s(node));
    Ttildeprime += auxmatrix;
  }

  // compute first squared brackets term in (3.18), Jelenic 1999
  LINALG::Matrix<3, 3, T> squaredbrackets(true);
  squaredbrackets.Multiply(dTinvdx, Ttilde);
  auxmatrix.Multiply(LARGEROTATIONS::Tinvmatrix(Psil), Ttildeprime);
  squaredbrackets += auxmatrix;

  LINALG::Matrix<3, 3, T> v_matrix(true);

  // loop through all nodes i
  for (unsigned int node = 0; node < numnodes; ++node)
  {
    // compute first term in second squared brackets
    Itildeprime[node] = dTinvdx;
    Itildeprime[node].Scale(funct(node));

    // compute second term in second squared brackets
    auxmatrix.Clear();
    auxmatrix += LARGEROTATIONS::Tinvmatrix(Psil);
    auxmatrix.Scale(deriv_s(node));

    // compute second squared brackets
    auxmatrix += Itildeprime[node];

    // compute second squared brackets time T(\Psi^l_j)
    Itildeprime[node].Multiply(auxmatrix, LARGEROTATIONS::Tmatrix(Psili[node]));

    // if node i is node I then add first squared brackets term times v_I
    if (node == nodeI_)
    {
      Calc_vI(v_matrix, phiIJ);
      auxmatrix.Multiply(squaredbrackets, v_matrix);
      Itildeprime[node] -= auxmatrix;
    }

    // if node i is node J then add first squared brackets term times v_J
    if (node == nodeJ_)
    {
      Calc_vJ(v_matrix, phiIJ);
      auxmatrix.Multiply(squaredbrackets, v_matrix);
      Itildeprime[node] -= auxmatrix;
    }

    // now the term in the curly brackets has been computed and has to be rotated by \Lambda_r
    auxmatrix.MultiplyNT(Itildeprime[node], Lambdar);
    Itildeprime[node].MultiplyNN(Lambdar, auxmatrix);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::Calc_vI(
    LINALG::Matrix<3, 3, T>& vI, const LINALG::Matrix<3, 1, T>& phiIJ) const
{
  // matrix v_I
  vI.Clear();

  LARGEROTATIONS::computespin(vI, phiIJ);
  // Fixme @grill: think about introducing a tolerance here to avoid singularity
  if (FADUTILS::VectorNorm(phiIJ) == 0.0)
    vI.Scale(0.25);
  else  // Fixme @grill: why do we cast to double here?
    vI.Scale(std::tan(FADUTILS::CastToDouble(FADUTILS::VectorNorm(phiIJ)) / 4.0) /
             FADUTILS::VectorNorm(phiIJ));

  for (unsigned int i = 0; i < 3; i++) vI(i, i) += 1.0;

  vI.Scale(0.5);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, typename T>
void LARGEROTATIONS::TriadInterpolationLocalRotationVectors<numnodes, T>::Calc_vJ(
    LINALG::Matrix<3, 3, T>& vJ, const LINALG::Matrix<3, 1, T>& phiIJ) const
{
  // matrix v_J
  vJ.Clear();

  LARGEROTATIONS::computespin(vJ, phiIJ);
  // Fixme @grill: think about introducing a tolerance here to avoid singularity
  if (FADUTILS::VectorNorm(phiIJ) == 0.0)
    vJ.Scale(-0.25);
  else  // Fixme why do we cast to double here?
    vJ.Scale(-1.0 * std::tan(FADUTILS::CastToDouble(FADUTILS::VectorNorm(phiIJ)) / 4.0) /
             FADUTILS::VectorNorm(phiIJ));

  for (unsigned int i = 0; i < 3; i++) vJ(i, i) += 1.0;

  vJ.Scale(0.5);
}

// explicit template instantiations
template class LARGEROTATIONS::TriadInterpolationLocalRotationVectors<2, double>;
template class LARGEROTATIONS::TriadInterpolationLocalRotationVectors<3, double>;
template class LARGEROTATIONS::TriadInterpolationLocalRotationVectors<4, double>;
template class LARGEROTATIONS::TriadInterpolationLocalRotationVectors<5, double>;
template class LARGEROTATIONS::TriadInterpolationLocalRotationVectors<2, Sacado::Fad::DFad<double>>;
template class LARGEROTATIONS::TriadInterpolationLocalRotationVectors<3, Sacado::Fad::DFad<double>>;
template class LARGEROTATIONS::TriadInterpolationLocalRotationVectors<4, Sacado::Fad::DFad<double>>;
template class LARGEROTATIONS::TriadInterpolationLocalRotationVectors<5, Sacado::Fad::DFad<double>>;
