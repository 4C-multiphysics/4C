// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_art_net_artery_ele_calc_pres_based.hpp"

#include "4C_art_net_art_terminal_bc.hpp"
#include "4C_art_net_artery_ele_calc.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_cnst_1d_art.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <fstream>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ArteryEleCalcPresBased<distype>::ArteryEleCalcPresBased(
    const int numdofpernode, const std::string& disname)
    : Discret::Elements::ArteryEleCalc<distype>(numdofpernode, disname)
{
}

/*----------------------------------------------------------------------*
 | singleton access method                                   vuong 08/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ArteryEleCalcPresBased<distype>*
Discret::Elements::ArteryEleCalcPresBased<distype>::instance(
    const int numdofpernode, const std::string& disname)
{
  using Key = std::pair<std::string, int>;
  static auto singleton_map = Core::Utils::make_singleton_map<Key>(
      [](const int numdofpernode, const std::string& disname)
      {
        return std::unique_ptr<ArteryEleCalcPresBased<distype>>(
            new ArteryEleCalcPresBased<distype>(numdofpernode, disname));
      });

  std::pair<std::string, int> key(disname, numdofpernode);

  return singleton_map[key].instance(Core::Utils::SingletonAction::create, numdofpernode, disname);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ArteryEleCalcPresBased<distype>::evaluate(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra, std::shared_ptr<Core::Mat::Material> mat)
{
  // the number of nodes
  const int numnode = my::iel_;

  // construct views
  Core::LinAlg::Matrix<numnode, numnode> elemat1(elemat1_epetra.values(), true);
  Core::LinAlg::Matrix<numnode, 1> elevec1(elevec1_epetra.values(), true);

  // ---------------------------------------------------------------------
  // call routine for calculating element matrix and right hand side
  // ---------------------------------------------------------------------
  sysmat(ele, discretization, la, elemat1, elevec1, mat);

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ArteryEleCalcPresBased<distype>::evaluate_service(Artery* ele,
    const Arteries::Action action, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra, std::shared_ptr<Core::Mat::Material> mat)
{
  switch (action)
  {
    case Arteries::calc_flow_pressurebased:
      evaluate_flow(ele, discretization, la, elevec1_epetra, mat);
      break;
    default:
      FOUR_C_THROW("Unknown type of action {} for Artery (PressureBased formulation)", action);
  }

  return 0;
}

template <Core::FE::CellType distype>
int Discret::Elements::ArteryEleCalcPresBased<distype>::scatra_evaluate(Artery* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra, std::shared_ptr<Core::Mat::Material> mat)
{
  FOUR_C_THROW(
      "not implemented by pressure-based formulation, should be done by cloned "
      "ScaTra-discretization");

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ArteryEleCalcPresBased<distype>::sysmat(Artery* ele,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::Matrix<my::iel_, my::iel_>& sysmat, Core::LinAlg::Matrix<my::iel_, 1>& rhs,
    std::shared_ptr<const Core::Mat::Material> material)
{
  // clear
  rhs.clear();
  sysmat.clear();

  // set element data
  const int numnode = my::iel_;

  // get pressure
  std::shared_ptr<const Core::LinAlg::Vector<double>> pressnp =
      discretization.get_state(0, "pressurenp");
  if (pressnp == nullptr) FOUR_C_THROW("could not get pressure inside artery element");

  // extract local values of pressure field from global state vector
  Core::LinAlg::Matrix<my::iel_, 1> mypress(Core::LinAlg::Initialization::zero);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<my::iel_, 1>>(*pressnp, mypress, la[0].lm_);

  // calculate the element length
  const double L = calculate_ele_length(ele, discretization, la);

  // check here, if we really have an artery !!
  if (material->material_type() != Core::Materials::m_cnst_art)
    FOUR_C_THROW("Wrong material type for artery");

  // cast the material to artery material material
  const Mat::Cnst1dArt* actmat = static_cast<const Mat::Cnst1dArt*>(material.get());

  if (actmat->is_collapsed()) return;

  // Read in diameter
  const double diam = actmat->diam();
  // Read in blood viscosity
  const double visc = actmat->viscosity();

  const double hag_pois = M_PI * pow(diam, 4) / 128.0 / visc;
  // gaussian points
  const Core::FE::IntegrationPoints1D intpoints(ele->gauss_rule());

  // get Jacobian matrix and determinant
  // actually compute its transpose....
  /*
                                        _____________________________________
      ds     L      dxi    2           /         2            2            2
      --- = ---   ; --- = ---   ; L = / ( x - x )  + ( y - y )  + ( z - z )
      dxi    2      ds     L         v     1   2        2    2       1   2

  */
  my::xji_ = 2.0 / L;

  const double prefac = hag_pois * my::xji_(0, 0);

  // integration loop
  for (int iquad = 0; iquad < intpoints.nquad; ++iquad)
  {
    // coordinates of the current integration point
    const double xi = intpoints.qxg[iquad][0];
    const double wgt = intpoints.qwgt[iquad];

    const double fac = prefac * wgt;

    // shape functions and their derivatives
    Core::FE::shape_function_1d_deriv1(my::deriv_, xi, distype);

    for (int inode = 0; inode < numnode; inode++)
      for (int jnode = 0; jnode < numnode; jnode++)
        sysmat(inode, jnode) += my::deriv_(0, inode) * fac * my::deriv_(0, jnode);

    // note: incremental form since rhs-coupling with poromultielastscatra-framework might be
    //       nonlinear
    Core::LinAlg::Matrix<1, 1> pressgrad;
    pressgrad.multiply(my::deriv_, mypress);
    for (int inode = 0; inode < numnode; inode++)
      rhs(inode) -= my::deriv_(0, inode) * fac * pressgrad(0, 0);
  }


  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ArteryEleCalcPresBased<distype>::evaluate_flow(Artery* ele,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseVector& flowVec, std::shared_ptr<const Core::Mat::Material> material)
{
  // get pressure
  std::shared_ptr<const Core::LinAlg::Vector<double>> pressnp =
      discretization.get_state(0, "pressurenp");
  if (pressnp == nullptr) FOUR_C_THROW("could not get pressure inside artery element");

  // extract local values of pressure field from global state vector
  Core::LinAlg::Matrix<my::iel_, 1> mypress(Core::LinAlg::Initialization::zero);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<my::iel_, 1>>(*pressnp, mypress, la[0].lm_);

  // calculate the element length
  const double L = calculate_ele_length(ele, discretization, la);

  // check here, if we really have an artery !!
  if (material->material_type() != Core::Materials::m_cnst_art)
    FOUR_C_THROW("Wrong material type for artery");

  // cast the material to artery material material
  const Mat::Cnst1dArt* actmat = static_cast<const Mat::Cnst1dArt*>(material.get());

  // Read in diameter
  const double diam = actmat->diam();
  // Read in blood viscosity
  const double visc = actmat->viscosity();

  const double hag_pois = M_PI * pow(diam, 4) / 128.0 / visc;

  // TODO: this works only for line 2 elements
  flowVec(0) = -hag_pois * (mypress(1) - mypress(0)) / L;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double Discret::Elements::ArteryEleCalcPresBased<distype>::calculate_ele_length(
    Artery* ele, Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  double length;
  // get current element length
  if (discretization.num_dof_sets() > 1 && discretization.has_state(1, "curr_seg_lengths"))
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> curr_seg_lengths =
        discretization.get_state(1, "curr_seg_lengths");
    std::vector<double> seglengths = Core::FE::extract_values(*curr_seg_lengths, la[1].lm_);

    length = std::accumulate(seglengths.begin(), seglengths.end(), 0.0);
  }
  else
    length = my::calculate_ele_length(ele);

  return length;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes

// 1D elements
template class Discret::Elements::ArteryEleCalcPresBased<Core::FE::CellType::line2>;

FOUR_C_NAMESPACE_CLOSE
