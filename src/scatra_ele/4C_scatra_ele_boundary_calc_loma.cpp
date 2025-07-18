// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_boundary_calc_loma.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_sutherland.hpp"
#include "4C_mat_thermostvenantkirchhoff.hpp"
#include "4C_material_base.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | singleton access method                                   fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleBoundaryCalcLoma<distype, probdim>*
Discret::Elements::ScaTraEleBoundaryCalcLoma<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleBoundaryCalcLoma<distype, probdim>>(
            new ScaTraEleBoundaryCalcLoma<distype, probdim>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 | private constructor for singletons                        fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleBoundaryCalcLoma<distype, probdim>::ScaTraEleBoundaryCalcLoma(
    const int numdofpernode, const int numscal, const std::string& disname)
    :  // constructor of base class
      my::ScaTraEleBoundaryCalc(numdofpernode, numscal, disname)
{
  return;
}


/*----------------------------------------------------------------------*
 | evaluate action                                           fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleBoundaryCalcLoma<distype, probdim>::evaluate_action(
    Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, ScaTra::BoundaryAction action,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3)
{
  // determine and evaluate action
  switch (action)
  {
    case ScaTra::BoundaryAction::calc_loma_therm_press:
    {
      calc_loma_therm_press(ele, params, discretization, la);

      break;
    }

    default:
    {
      my::evaluate_action(
          ele, params, discretization, action, la, elemat1, elemat2, elevec1, elevec2, elevec3);

      break;
    }
  }  // switch (action)

  return 0;
}


/*----------------------------------------------------------------------*
 | calculate loma therm pressure                              vg 03/09  |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcLoma<distype, probdim>::calc_loma_therm_press(
    Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  // get location vector associated with primary dofset
  std::vector<int>& lm = la[0].lm_;

  Core::Elements::Element* parentele = ele->parent_element();
  // we dont know the parent element's lm vector; so we have to build it here
  const int nenparent = parentele->num_node();
  std::vector<int> lmparent(nenparent);
  std::vector<int> lmparentowner;
  std::vector<int> lmparentstride;
  parentele->location_vector(discretization, lmparent, lmparentowner, lmparentstride);

  // get number of dofset associated with velocity related dofs
  const int ndsvel = my::scatraparams_->nds_vel();

  // get velocity values at nodes
  const std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
      discretization.get_state(ndsvel, "convective velocity field");

  // safety check
  if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

  // get values of velocity field from secondary dof-set
  const std::vector<int>& lmvel = la[ndsvel].lm_;
  std::vector<double> myconvel(lmvel.size());

  // extract local values of the global vectors
  myconvel = Core::FE::extract_values(*convel, lmvel);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  my::rotsymmpbc_->rotate_my_values_if_necessary(myconvel);

  // define vector for normal diffusive and velocity fluxes
  std::vector<double> mynormdiffflux(lm.size());
  std::vector<double> mynormvel(lm.size());

  // determine constant outer normal to this element
  my::normal_ = my::get_const_normal(my::xyze_);

  // extract temperature flux vector for each node of the parent element
  Core::LinAlg::SerialDenseMatrix eflux(3, nenparent);
  Core::Elements::Element* peleptr = (Core::Elements::Element*)parentele;
  int k = my::numscal_ - 1;  // temperature is always last degree of freedom!!
  std::string name = "flux_phi_" + std::to_string(k);
  // try to get the pointer to the entry (and check if type is
  // std::shared_ptr<Core::LinAlg::MultiVector<double>>)
  std::shared_ptr<Core::LinAlg::MultiVector<double>>* f =
      params.getPtr<std::shared_ptr<Core::LinAlg::MultiVector<double>>>(name);
  // check: field has been set and is not of type nullptr
  if (f != nullptr)
    Core::FE::extract_my_node_based_values(peleptr, eflux, **f, 3);
  else
    FOUR_C_THROW("MultiVector {} has not been found!", name);

  // calculate normal diffusive and velocity flux at each node of the
  // present boundary element
  for (int i = 0; i < nen_; ++i)
  {
    for (int j = 0; j < nenparent; ++j)
    {
      mynormdiffflux[i] = 0.0;
      mynormvel[i] = 0.0;
      for (int l = 0; l < nsd_; l++)
      {
        mynormdiffflux[i] += eflux(l, j) * my::normal_(l);
        mynormvel[i] += myconvel[i * nsd_ + l] * my::normal_(l);
      }
    }
  }

  // calculate integral of normal diffusive and velocity flux
  // NOTE: add integral value only for elements which are NOT ghosted!
  if (ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
  {
    norm_diff_flux_and_vel_integral(ele, params, mynormdiffflux, mynormvel);
  }
}


/*----------------------------------------------------------------------*
 | calculate Neumann inflow boundary conditions              fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcLoma<distype, probdim>::neumann_inflow(
    const Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& emat, Core::LinAlg::SerialDenseVector& erhs)
{
  // set thermodynamic pressure
  thermpress_ = params.get<double>("thermodynamic pressure");

  // call base class routine
  my::neumann_inflow(ele, params, discretization, la, emat, erhs);

  return;
}  // Discret::Elements::ScaTraEleBoundaryCalcLoma<distype, probdim>::neumann_inflow


/*----------------------------------------------------------------------*
 | calculate integral of normal diffusive flux and velocity     vg 09/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalcLoma<distype,
    probdim>::norm_diff_flux_and_vel_integral(const Core::Elements::Element* ele,
    Teuchos::ParameterList& params, const std::vector<double>& enormdiffflux,
    const std::vector<double>& enormvel)
{
  // get variables for integrals of normal diffusive flux and velocity
  double normdifffluxint = params.get<double>("normal diffusive flux integral");
  double normvelint = params.get<double>("normal velocity integral");

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    const double fac = my::eval_shape_func_and_int_fac(intpoints, gpid);

    // compute integral of normal flux
    for (int node = 0; node < nen_; ++node)
    {
      normdifffluxint += my::funct_(node) * enormdiffflux[node] * fac;
      normvelint += my::funct_(node) * enormvel[node] * fac;
    }
  }  // loop over integration points

  // add contributions to the global values
  params.set<double>("normal diffusive flux integral", normdifffluxint);
  params.set<double>("normal velocity integral", normvelint);

  return;
}  // Discret::Elements::ScaTraEleBoundaryCalcLoma<distype,
   // probdim>::norm_diff_flux_and_vel_integral


// template classes
template class Discret::Elements::ScaTraEleBoundaryCalcLoma<Core::FE::CellType::quad4, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcLoma<Core::FE::CellType::quad8, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcLoma<Core::FE::CellType::quad9, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcLoma<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcLoma<Core::FE::CellType::tri6, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcLoma<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcLoma<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcLoma<Core::FE::CellType::line3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcLoma<Core::FE::CellType::nurbs3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcLoma<Core::FE::CellType::nurbs9, 3>;

FOUR_C_NAMESPACE_CLOSE
