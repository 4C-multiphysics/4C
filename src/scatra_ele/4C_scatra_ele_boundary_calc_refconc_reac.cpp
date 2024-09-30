/*----------------------------------------------------------------------*/
/*! \file
\brief main file containing routines for calculation of scatra element formulated in reference
concentrations and with advanced reaction terms

\level 3

 *----------------------------------------------------------------------*/

#include "4C_scatra_ele_boundary_calc_refconc_reac.hpp"

#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  Singleton access method                                  thon 02/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<distype, probdim>*
Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::UTILS::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleBoundaryCalcRefConcReac<distype, probdim>>(
            new ScaTraEleBoundaryCalcRefConcReac<distype, probdim>(
                numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::UTILS::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 |  Private constructor                                      thon 02/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<distype,
    probdim>::ScaTraEleBoundaryCalcRefConcReac(const int numdofpernode, const int numscal,
    const std::string& disname)
    : Discret::ELEMENTS::ScaTraEleBoundaryCalc<distype, probdim>::ScaTraEleBoundaryCalc(
          numdofpernode, numscal, disname)
{
  return;
}


/*---------------------------------------------------------------------------*
 | Factor needed for the calculation of reference concentrations  thon 02/16 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<distype, probdim>::fac_for_ref_conc(
    const int iquad,                          ///< current boundary integration point
    const Core::Elements::FaceElement* bele,  ///< current boundary element
    Teuchos::ParameterList& params,           ///< parameter list
    Core::FE::Discretization& discretization  ///< discretization
)
{
  const Core::Elements::Element* pele = bele->parent_element();

  double J = 1.0;
  // only 3D cases:
  if (bele->shape() == Core::FE::CellType::tri3)
  {
    if (pele->shape() == Core::FE::CellType::tet4)
      J = calc_jat_int_point<Core::FE::CellType::tri3, Core::FE::CellType::tet4>(
          iquad, bele, pele, params, discretization);
    else if (pele->shape() == Core::FE::CellType::pyramid5)
      J = calc_jat_int_point<Core::FE::CellType::tri3, Core::FE::CellType::pyramid5>(
          iquad, bele, pele, params, discretization);
    else
      FOUR_C_THROW("Parent element not supported here!");
  }
  else if (bele->shape() == Core::FE::CellType::quad4)
  {
    if (pele->shape() == Core::FE::CellType::hex8)
      J = calc_jat_int_point<Core::FE::CellType::quad4, Core::FE::CellType::hex8>(
          iquad, bele, pele, params, discretization);
    else if (pele->shape() == Core::FE::CellType::pyramid5)
      J = calc_jat_int_point<Core::FE::CellType::quad4, Core::FE::CellType::pyramid5>(
          iquad, bele, pele, params, discretization);
    else
      FOUR_C_THROW("Parent element not supported here!");
  }
  else
    FOUR_C_THROW("Boundary element not supported here!");

  return 1.0 / J;
}


/*---------------------------------------------------------------------------*
 | Factor needed for the calculation of reference concentrations  thon 02/16 |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType bdistype, Core::FE::CellType pdistype>
double Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<distype, probdim>::calc_jat_int_point(
    const int iquad,                          ///< current boundary integration point
    const Core::Elements::FaceElement* bele,  ///< current boundary element
    const Core::Elements::Element* pele,      ///< current parent element
    Teuchos::ParameterList& params,           ///< parameter list
    Core::FE::Discretization& discretization  ///< discretization
)
{
  // NOTE: we want to evaluate J=det(F) on the current gauss point of the current boundary element.
  // Since this does depend on ALL values of the involved element this is quite a hassle :(

  // number of parent spatial dimensions
  const int pnsd = Core::FE::dim<pdistype>;
  // number of boundary spatial dimensions
  const int bnsd = Core::FE::dim<bdistype>;

  if (pnsd != nsd_) FOUR_C_THROW("dimension do not match!");
  if (bnsd != nsd_ele_) FOUR_C_THROW("dimension do not match!");

  // number of parent element nodes
  const int pnen = Core::FE::num_nodes<pdistype>;
  // number of (boundary) element nodes
  static const int bnen = Core::FE::num_nodes<bdistype>;

  if (bnen != nen_) FOUR_C_THROW("Number of element nodes do not match!");

  // get local node coordinates
  Core::LinAlg::Matrix<pnsd, pnen> pxyze(true);
  Core::LinAlg::Matrix<pnsd, pnen> pxyze0(true);
  Core::Geo::fill_initial_position_array<pdistype, pnsd, Core::LinAlg::Matrix<pnsd, pnen>>(
      pele, pxyze0);
  pxyze = pxyze0;

  if (my::scatraparams_->is_ale())
  {
    // get number of dof-set associated with displacement related dofs
    const int ndsdisp = my::scatraparams_->nds_disp();

    Teuchos::RCP<const Core::LinAlg::Vector> dispnp = discretization.get_state(ndsdisp, "dispnp");
    if (dispnp == Teuchos::null) FOUR_C_THROW("Cannot get state vector 'dispnp'");

    // parent element location array
    Core::Elements::LocationArray pla(discretization.num_dof_sets());
    pele->location_vector(discretization, pla, false);

    // determine number of velocity related dofs per node
    const int numdispdofpernode = pla[ndsdisp].lm_.size() / pnen;

    // construct location vector for velocity related dofs
    std::vector<int> plmdisp(pnsd * pnen, -1);
    for (int inode = 0; inode < pnen; ++inode)
      for (int idim = 0; idim < pnsd; ++idim)
        plmdisp[inode * pnsd + idim] = pla[ndsdisp].lm_[inode * numdispdofpernode + idim];

    // we deal with a nsd_-dimensional flow field
    Core::LinAlg::Matrix<pnsd, pnen> pedispnp(true);

    // extract local values of convective velocity field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<pnsd, pnen>>(*dispnp, pedispnp, plmdisp);

    // rotate the vector field in the case of rotationally symmetric boundary conditions
    // my::rotsymmpbc_->template rotate_my_values_if_necessary<pnsd,pnen>(pedispnp);

    pxyze += pedispnp;
  }

  // get Gaussian integration points
  const Core::FE::IntPointsAndWeights<pnsd> pintpoints(
      Discret::ELEMENTS::DisTypeToOptGaussRule<pdistype>::rule);

  // get Gaussian integration points
  const Core::FE::IntPointsAndWeights<bnsd> bintpoints(
      Discret::ELEMENTS::DisTypeToOptGaussRule<bdistype>::rule);

  Core::LinAlg::SerialDenseMatrix gps(bintpoints.ip().nquad, bnsd);
  for (int biquad = 0; biquad < bintpoints.ip().nquad; ++biquad)
  {
    const double* gpcoord = (bintpoints.ip().qxg)[biquad];
    for (int idim = 0; idim < bnsd; idim++)
    {
      gps(biquad, idim) = gpcoord[idim];
    }
  }

  // distinguish 2- and 3-D case
  Core::LinAlg::SerialDenseMatrix pqxg(pintpoints.ip().nquad, pnsd);
  if (pnsd == 2)
    Core::FE::boundary_gp_to_parent_gp2(pqxg, gps, pdistype, bdistype, bele->face_master_number());
  else if (pnsd == 3)
    Core::FE::boundary_gp_to_parent_gp3(pqxg, gps, pdistype, bdistype, bele->face_master_number());


  Core::LinAlg::Matrix<pnsd, 1> pxsi(true);
  Core::LinAlg::Matrix<pnsd, pnen> pderiv(true);

  // reference coordinates of integration point from parent element
  for (int idim = 0; idim < pnsd; idim++)
  {
    pxsi(idim) = pqxg(iquad, idim);
  }

  // parent element shape functions and local derivatives
  Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

  // Jacobian matrix and determinant of parent element (including check)
  Core::LinAlg::Matrix<pnsd, pnsd> dxds(true);
  dxds.multiply_nt(pderiv, pxyze);
  const double detdxds = dxds.determinant();

  // Jacobian matrix and determinant of parent element (including check)
  Core::LinAlg::Matrix<pnsd, pnsd> dXds(true);
  dXds.multiply_nt(pderiv, pxyze0);
  const double detdXds = dXds.determinant();

  // deformation gradtient dx/dX = dx/ds * ds/dX = dx/ds * (dX/ds)^(-1)
  const double J = detdxds / detdXds;

  return J;
}


// template classes
template class Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<Core::FE::CellType::quad4, 3>;
template class Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<Core::FE::CellType::quad8, 3>;
template class Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<Core::FE::CellType::quad9, 3>;
template class Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<Core::FE::CellType::tri3, 3>;
template class Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<Core::FE::CellType::tri6, 3>;
template class Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<Core::FE::CellType::line2, 2>;
template class Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<Core::FE::CellType::line2, 3>;
template class Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<Core::FE::CellType::line3, 2>;
template class Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<Core::FE::CellType::nurbs3, 2>;
template class Discret::ELEMENTS::ScaTraEleBoundaryCalcRefConcReac<Core::FE::CellType::nurbs9, 3>;

FOUR_C_NAMESPACE_CLOSE
