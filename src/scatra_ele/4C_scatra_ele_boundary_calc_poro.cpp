// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_boundary_calc_poro.hpp"

#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"
#include "4C_global_data.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  Singleton access method                               hemmler 07/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleBoundaryCalcPoro<distype, probdim>*
Discret::Elements::ScaTraEleBoundaryCalcPoro<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleBoundaryCalcPoro<distype, probdim>>(
            new ScaTraEleBoundaryCalcPoro<distype, probdim>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*----------------------------------------------------------------------*
 |  Private constructor                                   hemmler 07/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleBoundaryCalcPoro<distype, probdim>::ScaTraEleBoundaryCalcPoro(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::ScaTraEleBoundaryCalc(
          numdofpernode, numscal, disname),
      eporosity_(Core::LinAlg::Initialization::zero),
      eprenp_(Core::LinAlg::Initialization::zero),
      isnodalporosity_(false)
{
  return;
}

/*----------------------------------------------------------------------*
 | evaluate action                                           fang 02/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleBoundaryCalcPoro<distype, probdim>::evaluate_action(
    Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, ScaTra::BoundaryAction action,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // determine and evaluate action
  switch (action)
  {
    case ScaTra::BoundaryAction::calc_fps3i_surface_permeability:
    case ScaTra::BoundaryAction::calc_fs3i_surface_permeability:
    case ScaTra::BoundaryAction::calc_Neumann:
    case ScaTra::BoundaryAction::calc_Robin:
    case ScaTra::BoundaryAction::calc_normal_vectors:
    case ScaTra::BoundaryAction::integrate_shape_functions:
    {
      my::evaluate_action(ele, params, discretization, action, la, elemat1_epetra, elemat2_epetra,
          elevec1_epetra, elevec2_epetra, elevec3_epetra);
      break;
    }
    case ScaTra::BoundaryAction::add_convective_mass_flux:
    {
      // calculate integral of convective mass/heat flux
      // NOTE: since results are added to a global vector via normal assembly
      //       it would be wrong to suppress results for a ghosted boundary!

      // get actual values of transported scalars
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");

      // extract local values from the global vector
      std::vector<Core::LinAlg::Matrix<nen_, 1>> ephinp(
          my::numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(Core::LinAlg::Initialization::zero));
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp, la[0].lm_);

      // get number of dofset associated with velocity related dofs
      const int ndsvel = my::scatraparams_->nds_vel();

      // get convective (velocity - mesh displacement) velocity at nodes
      std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
          discretization.get_state(ndsvel, "convective velocity field");
      if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

      // determine number of velocity related dofs per node
      const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

      // construct location vector for velocity related dofs
      std::vector<int> lmvel(nsd_ele_ * nen_, -1);
      for (int inode = 0; inode < nen_; ++inode)
        for (int idim = 0; idim < nsd_ele_; ++idim)
          lmvel[inode * nsd_ele_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

      // we deal with a nsd_-dimensional flow field
      Core::LinAlg::Matrix<nsd_, nen_> econvel(Core::LinAlg::Initialization::zero);

      // extract local values of convective velocity field from global state vector
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*convel, econvel, lmvel);

      // rotate the vector field in the case of rotationally symmetric boundary conditions
      my::rotsymmpbc_->rotate_my_values_if_necessary(econvel);

      // construct location vector for pressure dofs
      std::vector<int> lmpre(nen_, -1);
      for (int inode = 0; inode < nen_; ++inode)
        lmpre[inode] = la[ndsvel].lm_[inode * numveldofpernode + nsd_ele_];

      // extract local values of pressure field from global state vector
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*convel, eprenp_, lmpre);

      // this is a hack. Check if the structure (assumed to be the dofset 1) has more DOFs than
      // dimension. If so, we assume that this is the porosity
      if (discretization.num_dof(1, ele->nodes()[0]) == nsd_ele_ + 2)
      {
        isnodalporosity_ = true;

        // get number of dofset associated with velocity related dofs
        const int ndsdisp = my::scatraparams_->nds_disp();

        std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
            discretization.get_state(ndsdisp, "dispnp");

        if (disp != nullptr)
        {
          std::vector<double> mydisp = Core::FE::extract_values(*disp, la[ndsdisp].lm_);

          for (int inode = 0; inode < nen_; ++inode)  // number of nodes
            eporosity_(inode, 0) = mydisp[nsd_ + (inode * (nsd_ele_ + 2))];
        }
        else
          FOUR_C_THROW("Cannot get state vector displacement");
      }
      else
        isnodalporosity_ = false;

      // for the moment we ignore the return values of this method
      calc_convective_flux(ele, ephinp, econvel, elevec1_epetra);
      // vector<double> locfluxintegral = calc_convective_flux(ele,ephinp,evel,elevec1_epetra);
      // std::cout<<"locfluxintegral[0] = "<<locfluxintegral[0]<<std::endl;

      break;
    }
    default:
    {
      FOUR_C_THROW("Invalid action parameter nr. {}!", action);
      break;
    }
  }  // switch(action)

  return 0;
}

/*----------------------------------------------------------------------*
 | calculate integral of convective flux across boundary    vuong 07/15 |
 | (overwrites method in ScaTraEleBoundaryCalc)                         |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
std::vector<double>
Discret::Elements::ScaTraEleBoundaryCalcPoro<distype, probdim>::calc_convective_flux(
    const Core::Elements::FaceElement* ele,
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& ephinp,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelnp, Core::LinAlg::SerialDenseVector& erhs)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  std::vector<double> integralflux(my::numscal_);

  // loop over all scalars
  for (int k = 0; k < my::numscal_; ++k)
  {
    integralflux[k] = 0.0;

    // loop over all integration points
    for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
    {
      const double fac = my::eval_shape_func_and_int_fac(intpoints, iquad, &(this->normal_));

      const double porosity = compute_porosity(ele);

      // get velocity at integration point
      my::velint_.multiply(evelnp, my::funct_);

      // normal velocity (note: normal_ is already a unit(!) normal)
      const double normvel = my::velint_.dot(my::normal_);

      // scalar at integration point
      const double phi = my::funct_.dot(ephinp[k]);

      const double val = porosity * phi * normvel * fac;
      integralflux[k] += val;
      // add contribution to provided vector (distribute over nodes using shape fct.)
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * my::numdofpernode_ + k;
        erhs[fvi] += val * my::funct_(vi);
      }
    }
  }

  return integralflux;

}  // ScaTraEleBoundaryCalcPoro<distype>::ConvectiveFlux


/*----------------------------------------------------------------------*
 |  get the material constants  (protected)                  vuong 10/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleBoundaryCalcPoro<distype, probdim>::compute_porosity(
    const Core::Elements::FaceElement* ele  //!< the element we are dealing with
)
{
  double porosity = 0.0;

  if (isnodalporosity_)
  {
    porosity = eporosity_.dot(my::funct_);
  }
  else
  {
    FOUR_C_THROW("porosity calculation not yet implemented for non-node-based porosity!");
  }

  return porosity;
}


// template classes
template class Discret::Elements::ScaTraEleBoundaryCalcPoro<Core::FE::CellType::quad4, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcPoro<Core::FE::CellType::quad8, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcPoro<Core::FE::CellType::quad9, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcPoro<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcPoro<Core::FE::CellType::tri6, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcPoro<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcPoro<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalcPoro<Core::FE::CellType::line3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcPoro<Core::FE::CellType::nurbs3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalcPoro<Core::FE::CellType::nurbs9, 3>;

FOUR_C_NAMESPACE_CLOSE
