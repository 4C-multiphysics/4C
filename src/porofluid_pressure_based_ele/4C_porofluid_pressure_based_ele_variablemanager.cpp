// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_porofluid_pressure_based_ele_variablemanager.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_mat_fluidporo_singlephase.hpp"
#include "4C_porofluid_pressure_based_ele_calc_utils.hpp"
#include "4C_porofluid_pressure_based_ele_parameter.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 | factory method                                           vuong 08/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
std::shared_ptr<Discret::Elements::PoroFluidManager::VariableManagerInterface<nsd, nen>>
Discret::Elements::PoroFluidManager::VariableManagerInterface<nsd, nen>::create_variable_manager(
    const Discret::Elements::PoroFluidMultiPhaseEleParameter& para,
    const PoroPressureBased::Action& action, std::shared_ptr<Core::Mat::Material> mat,
    int numdofpernode, int numfluidphases, int numvolfrac)
{
  std::shared_ptr<VariableManagerInterface<nsd, nen>> varmanager = nullptr;

  // get the number of volume fractions
  // the check for correct input definition is performed in
  // Mat::PAR::FluidPoroMultiPhase::initialize()

  // determine closing relation of volume fraction material

  const Mat::PAR::PoroFluidPressureBased::ClosingRelation volfrac_closing_relation = std::invoke(
      [&]()
      {
        if (numvolfrac)
        {
          if (numdofpernode - numfluidphases == numvolfrac)
          {
            return Mat::PAR::PoroFluidPressureBased::ClosingRelation::evolutionequation_blood_lung;
          }
          else if (numdofpernode - numfluidphases == numvolfrac * 2)
          {
            return Mat::PAR::PoroFluidPressureBased::ClosingRelation::
                evolutionequation_homogenized_vasculature_tumor;
          }
          else
          {
            FOUR_C_THROW("You have an invalid number of volfracs in the variable manager!");
          }
        }
        else
        {
          return Mat::PAR::PoroFluidPressureBased::ClosingRelation::undefined;
        }
      });
  // determine action
  switch (action)
  {
    // calculate true pressures and saturation
    case PoroPressureBased::calc_pres_and_sat:
    {
      // only phi values are needed
      varmanager = std::make_shared<VariableManagerPhi<nsd, nen>>(numdofpernode);

      break;
    }
    // calculate variables for output
    case PoroPressureBased::calc_solidpressure:
    case PoroPressureBased::calc_porosity:
    case PoroPressureBased::calc_determinant_of_deformationgradient:
    case PoroPressureBased::calc_volfrac_blood_lung:
    {
      // only phi values are needed
      varmanager = std::make_shared<VariableManagerPhi<nsd, nen>>(numdofpernode);

      // add manager for displacements and solid velocities in case of ALE
      if (para.is_ale())
        varmanager = std::make_shared<VariableManagerStruct<nsd, nen>>(
            para.nds_vel(), para.nds_disp(), varmanager);

      break;
    }
    // calculate the valid dofs
    case PoroPressureBased::calc_valid_dofs:
    {
      // only phi values are needed
      varmanager = std::make_shared<VariableManagerPhi<nsd, nen>>(numdofpernode);

      if (numvolfrac > 0)
        varmanager = std::make_shared<VariableManagerMaximumNodalVolFracValue<nsd, nen>>(
            numvolfrac, varmanager, mat, volfrac_closing_relation);

      break;
    }
    case PoroPressureBased::recon_flux_at_nodes:
    case PoroPressureBased::calc_phase_velocities:
    {
      // state vector and gradients are needed
      varmanager = std::make_shared<VariableManagerPhiGradPhi<nsd, nen>>(numdofpernode);

      // add manager for displacements and solid velocities in case of ALE
      if (para.is_ale())
        varmanager = std::make_shared<VariableManagerStruct<nsd, nen>>(
            para.nds_vel(), para.nds_disp(), varmanager);
      break;
    }
    // read data from scatra
    case PoroPressureBased::get_access_from_scatra:
    case PoroPressureBased::get_access_from_artcoupling:
    {
      // NOTE: we do not need the variable manager struct here, since the call
      // ExtractElementsAndNodeValues will
      //       otherwise also update the current configuration vector xyze_ with which it is called
      //       and scatra only needs phi and gradphi Also, the scatra discretization already has all
      //       necessary data for moving meshes such as displacements
      varmanager = std::make_shared<VariableManagerPhiGradPhi<nsd, nen>>(numdofpernode);

      if (numvolfrac > 0)
        varmanager = std::make_shared<VariableManagerMaximumNodalVolFracValue<nsd, nen>>(
            numvolfrac, varmanager, mat, volfrac_closing_relation);

      break;
    }
    default:
    {
      // default: potentially read everything
      varmanager = std::make_shared<VariableManagerPhiGradPhi<nsd, nen>>(numdofpernode);

      if (not para.is_stationary())
        varmanager = std::make_shared<VariableManagerInstat<nsd, nen>>(varmanager);

      if (para.is_ale())
        varmanager = std::make_shared<VariableManagerStruct<nsd, nen>>(
            para.nds_vel(), para.nds_disp(), varmanager);

      if (numvolfrac > 0)
        varmanager = std::make_shared<VariableManagerMaximumNodalVolFracValue<nsd, nen>>(
            numvolfrac, varmanager, mat, volfrac_closing_relation);

      break;
    }
  }  // switch(action)

  // if there are other scalar values (from ScaTra coupling) add another manager
  if (para.has_scalar())
    varmanager = std::make_shared<VariableManagerScalar<nsd, nen>>(para.nds_scalar(), varmanager);

  // done
  return varmanager;
};

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 | extract node values related to the state vector 'phinp'   vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidManager::VariableManagerPhi<nsd,
    nen>::extract_element_and_node_values(const Core::Elements::Element& ele,
    const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::Matrix<nsd, nen>& xyze, const int dofsetnum)
{
  // extract local values from the global vectors
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp =
      discretization.get_state(dofsetnum, "phinp_fluid");
  if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");

  // values of fluid are in 'dofsetnum' --> 0 if called with porofluid-element
  //                                    --> correct number has be passed if called with another
  //                                    element (e.g. scatra)
  const std::vector<int>& lm = la[dofsetnum].lm_;

  // extract element vector from global vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen, 1>>(*phinp, ephinp_, lm);

  // set flag
  this->isextracted_ = true;
  return;
};

/*----------------------------------------------------------------------*
 | evaluate state vector at gauss point                      vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidManager::VariableManagerPhi<nsd, nen>::evaluate_gp_variables(
    const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
    const Core::LinAlg::Matrix<nsd, nen>&
        derxy  //! array for shape function derivatives w.r.t x,y,z
)
{
  // check
  if (not this->isextracted_)
    FOUR_C_THROW("extract_element_and_node_values() has not been called!");

  // loop over DOFs
  for (int k = 0; k < this->numdofpernode_; ++k)
  {
    // calculate scalar at t_(n+1) or t_(n+alpha_F)
    phinp_[k] = funct.dot(ephinp_[k]);
  }

  // done
  this->isevaluated_ = true;

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 | evaluate state vector at gauss point                      vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidManager::VariableManagerPhiGradPhi<nsd,
    nen>::evaluate_gp_variables(const Core::LinAlg::Matrix<nen, 1>&
                                    funct,  //! array for shape functions
    const Core::LinAlg::Matrix<nsd, nen>&
        derxy  //! array for shape function derivatives w.r.t x,y,z
)
{
  // loop over DOFs
  for (int k = 0; k < this->numdofpernode_; ++k)
  {
    // spatial gradient of current scalar value
    gradphi_[k].multiply(derxy, this->ephinp_[k]);
  }

  // call base class
  VariableManagerPhi<nsd, nen>::evaluate_gp_variables(funct, derxy);

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 | extract node values related to time derivatives           vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidManager::VariableManagerInstat<nsd,
    nen>::extract_element_and_node_values(const Core::Elements::Element& ele,
    const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::Matrix<nsd, nen>& xyze, const int dofsetnum)
{
  // extract local values from the global vectors
  std::shared_ptr<const Core::LinAlg::Vector<double>> hist = discretization.get_state("hist");
  std::shared_ptr<const Core::LinAlg::Vector<double>> phidtnp = discretization.get_state("phidtnp");
  if (phidtnp == nullptr) FOUR_C_THROW("Cannot get state vector 'phidtnp'");


  // values of fluid are in 'dofsetnum' --> 0 if called with porofluid-element
  //                                    --> correct number has be passed if called with another
  //                                    element (e.g. scatra)
  const std::vector<int>& lm = la[dofsetnum].lm_;

  // extract values from global vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen, 1>>(*hist, ehist_, lm);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen, 1>>(*phidtnp, ephidtnp_, lm);

  // call wrapped class
  this->varmanager_->extract_element_and_node_values(ele, discretization, la, xyze, dofsetnum);

  return;
};

/*----------------------------------------------------------------------*
 | evaluate state vector at gauss point                      vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidManager::VariableManagerInstat<nsd, nen>::evaluate_gp_variables(
    const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
    const Core::LinAlg::Matrix<nsd, nen>&
        derxy  //! array for shape function derivatives w.r.t x,y,z
)
{
  // loop over DOFs
  for (int k = 0; k < this->varmanager_->num_dof_per_node(); ++k)
  {
    // history data (or acceleration)
    hist_[k] = funct.dot(ehist_[k]);
    // calculate time derivative of scalar at t_(n+1)
    phidtnp_[k] = funct.dot(ephidtnp_[k]);
  }

  // call wrapped class
  this->varmanager_->evaluate_gp_variables(funct, derxy);

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 | extract node values related to deforming domain           vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidManager::VariableManagerStruct<nsd,
    nen>::extract_element_and_node_values(const Core::Elements::Element& ele,
    const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::Matrix<nsd, nen>& xyze, const int dofsetnum)
{
  if (dofsetnum != 0)
    FOUR_C_THROW(
        "VariableManagerStruct has been called with dofsetnum = {}, this is not possible.\n"
        "The VariableManagerStruct always has to be called with a porofluid-element",
        dofsetnum);

  // call internal class
  this->varmanager_->extract_element_and_node_values(ele, discretization, la, xyze, dofsetnum);

  // determine number of velocity related dofs per node
  const int numveldofpernode = la[ndsvel_].lm_.size() / nen;

  // construct location vector for velocity related dofs
  std::vector<int> lmvel(nsd * nen, -1);
  for (int inode = 0; inode < nen; ++inode)
    for (int idim = 0; idim < nsd; ++idim)
      lmvel[inode * nsd + idim] = la[ndsvel_].lm_[inode * numveldofpernode + idim];

  // get velocity at nodes
  std::shared_ptr<const Core::LinAlg::Vector<double>> vel =
      discretization.get_state(ndsvel_, "velocity field");
  if (vel == nullptr) FOUR_C_THROW("Cannot get state vector velocity");

  // extract local values of velocity field from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd, nen>>(*vel, econvelnp_, lmvel);

  // safety check
  std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
      discretization.get_state(ndsdisp_, "dispnp");
  if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

  // determine number of displacement related dofs per node
  const int numdispdofpernode = la[ndsdisp_].lm_.size() / nen;

  // construct location vector for displacement related dofs
  std::vector<int> lmdisp(nsd * nen, -1);
  for (int inode = 0; inode < nen; ++inode)
    for (int idim = 0; idim < nsd; ++idim)
      lmdisp[inode * nsd + idim] = la[ndsdisp_].lm_[inode * numdispdofpernode + idim];

  // extract local values of displacement field from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd, nen>>(*dispnp, edispnp_, lmdisp);

  // add nodal displacements to point coordinates
  xyze += edispnp_;

  return;
};

/*----------------------------------------------------------------------*
 | evaluate state vector at gauss point                      vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidManager::VariableManagerStruct<nsd, nen>::evaluate_gp_variables(
    const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
    const Core::LinAlg::Matrix<nsd, nen>&
        derxy  //! array for shape function derivatives w.r.t x,y,z
)
{
  // velocity divergence required for conservative form
  Core::LinAlg::Matrix<nsd, nsd> vderxy;
  vderxy.multiply_nt(econvelnp_, derxy);
  divconvelint_ = 0.0;
  // compute vel x,x  + vel y,y +  vel z,z at integration point
  for (int j = 0; j < nsd; ++j) divconvelint_ += vderxy(j, j);

  // convective velocity
  convelint_.multiply(econvelnp_, funct);

  // gauss point displacements
  dispint_.multiply(edispnp_, funct);

  // call wrapped class
  this->varmanager_->evaluate_gp_variables(funct, derxy);

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 | extract node values related to ScaTra coupling           vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidManager::VariableManagerScalar<nsd,
    nen>::extract_element_and_node_values(const Core::Elements::Element& ele,
    const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::Matrix<nsd, nen>& xyze, const int dofsetnum)
{
  // call internal class
  this->varmanager_->extract_element_and_node_values(ele, discretization, la, xyze, dofsetnum);

  // get state vector from discretization
  std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp =
      discretization.get_state(ndsscalar_, "scalars");
  if (scalarnp == nullptr) FOUR_C_THROW("Cannot get state vector 'scalars'");

  // determine number of scalars related dofs per node
  const int numscalardofpernode = la[ndsscalar_].lm_.size() / nen;

  // rebuild scalar vector
  escalarnp_.clear();
  escalarnp_.resize(
      numscalardofpernode, Core::LinAlg::Matrix<nen, 1>(Core::LinAlg::Initialization::zero));
  // extract local values of displacement field from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen, 1>>(
      *scalarnp, escalarnp_, la[ndsscalar_].lm_);

  return;
};

/*----------------------------------------------------------------------*
 | evaluate state vector at gauss point                      vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidManager::VariableManagerScalar<nsd, nen>::evaluate_gp_variables(
    const Core::LinAlg::Matrix<nen, 1>& funct,  //! array for shape functions
    const Core::LinAlg::Matrix<nsd, nen>&
        derxy  //! array for shape function derivatives w.r.t x,y,z
)
{
  // call wrapped class
  this->varmanager_->evaluate_gp_variables(funct, derxy);

  // evaluate scalar values
  if (not escalarnp_.empty())
  {
    scalarnp_.resize(escalarnp_.size(), 0.0);
    gradscalarnp_.resize(escalarnp_.size());
    for (unsigned k = 0; k < escalarnp_.size(); ++k)
    {
      scalarnp_[k] = funct.dot(escalarnp_[k]);
      gradscalarnp_[k].multiply(derxy, escalarnp_[k]);
    }
  }

  return;
}

/*----------------------------------------------------------------------*
 * **********************************************************************
 *----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*
 | extract node values related to the state vector 'phinp'   vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidManager::VariableManagerMaximumNodalVolFracValue<nsd,
    nen>::extract_element_and_node_values(const Core::Elements::Element& ele,
    const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::Matrix<nsd, nen>& xyze, const int dofsetnum)
{
  // call internal class
  this->varmanager_->extract_element_and_node_values(ele, discretization, la, xyze, dofsetnum);

  std::shared_ptr<const Core::LinAlg::Vector<double>> phin =
      discretization.get_state(dofsetnum, "phin_fluid");
  if (phin == nullptr) FOUR_C_THROW("Cannot get state vector 'phin_fluid'");

  // values of fluid are in 'dofsetnum' --> 0 if called with porofluid-element
  //                                    --> correct number has be passed if called with another
  //                                    element (e.g. scatra)
  const std::vector<int>& lm = la[dofsetnum].lm_;

  // extract values from global vector
  std::vector<Core::LinAlg::Matrix<nen, 1>> ephin(this->num_dof_per_node());
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen, 1>>(*phin, ephin, lm);

  const int numfluidphases = std::invoke(
      [&]()
      {
        if (volfrac_closing_relation_ ==
            Mat::PAR::PoroFluidPressureBased::ClosingRelation::evolutionequation_blood_lung)
          return (int)(this->num_dof_per_node() - numvolfrac_);
        else if (volfrac_closing_relation_ == Mat::PAR::PoroFluidPressureBased::ClosingRelation::
                                                  evolutionequation_homogenized_vasculature_tumor)
          return (int)(this->num_dof_per_node() - (2 * numvolfrac_));
        else
        {
          FOUR_C_THROW("Internal error!");
        }
      });

  if (volfrac_closing_relation_ == Mat::PAR::PoroFluidPressureBased::ClosingRelation::
                                       evolutionequation_homogenized_vasculature_tumor)
  {
    // loop over DOFs
    for (int k = 0; k < numvolfrac_; ++k)
    {
      // get the volfrac pressure material
      const Mat::FluidPoroVolFracPressure& volfracpressmat =
          PoroPressureBased::ElementUtils::get_vol_frac_pressure_mat_from_material(
              *multiphasemat_, k + numvolfrac_ + numfluidphases);

      // this approach proved to be the most stable one
      // we evaluate the volume fraction pressure if at least one volfrac value is bigger than
      // minvolfrac
      ele_has_valid_volfrac_press_[k] =
          ephin[k + numfluidphases].max_value() > volfracpressmat.min_vol_frac();

      // and we evaluate the volume fraction species if all volfrac values are bigger than
      // minvolfrac
      ele_has_valid_volfrac_spec_[k] =
          ephin[k + numfluidphases].min_value() > volfracpressmat.min_vol_frac();
    }
  }
  else if (volfrac_closing_relation_ ==
           Mat::PAR::PoroFluidPressureBased::ClosingRelation::evolutionequation_blood_lung)
  {
    // sofar every dof is valid no minimal volfrac necessary
    ele_has_valid_volfrac_press_[0] = true;
    ele_has_valid_volfrac_spec_[0] = true;
  }
  else
  {
    FOUR_C_THROW("Internal error!");
  }
}

/*----------------------------------------------------------------------*
 | evaluate state vector at gauss point                      vuong 09/16 |
 *----------------------------------------------------------------------*/
template <int nsd, int nen>
void Discret::Elements::PoroFluidManager::VariableManagerMaximumNodalVolFracValue<nsd,
    nen>::evaluate_gp_variables(const Core::LinAlg::Matrix<nen, 1>&
                                    funct,  //! array for shape functions
    const Core::LinAlg::Matrix<nsd, nen>&
        derxy  //! array for shape function derivatives w.r.t x,y,z
)
{
  // call wrapped class
  this->varmanager_->evaluate_gp_variables(funct, derxy);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes

// 1D elements
// line 2
template class Discret::Elements::PoroFluidManager::VariableManagerInterface<1, 2>;

// line 3
template class Discret::Elements::PoroFluidManager::VariableManagerInterface<1, 3>;

// 2D elements
// tri3
template class Discret::Elements::PoroFluidManager::VariableManagerInterface<2, 3>;
// tri6
template class Discret::Elements::PoroFluidManager::VariableManagerInterface<2, 6>;
// quad4
template class Discret::Elements::PoroFluidManager::VariableManagerInterface<2, 4>;

// quad9 and nurbs9
template class Discret::Elements::PoroFluidManager::VariableManagerInterface<2, 9>;

// 3D elements
// hex8
template class Discret::Elements::PoroFluidManager::VariableManagerInterface<3, 8>;

// hex27
template class Discret::Elements::PoroFluidManager::VariableManagerInterface<3, 27>;
// tet4
template class Discret::Elements::PoroFluidManager::VariableManagerInterface<3, 4>;
// tet10
template class Discret::Elements::PoroFluidManager::VariableManagerInterface<3, 10>;
// pyramid5
template class Discret::Elements::PoroFluidManager::VariableManagerInterface<3, 5>;

FOUR_C_NAMESPACE_CLOSE
