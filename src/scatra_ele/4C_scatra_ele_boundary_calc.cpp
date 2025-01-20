// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_boundary_calc.hpp"

#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_fluid_rotsym_periodicbc.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_mat_fourieriso.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_mat_thermostvenantkirchhoff.hpp"
#include "4C_scatra_ele_parameter_boundary.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::ScaTraEleBoundaryCalc(
    const int numdofpernode, const int numscal, const std::string& disname)
    : scatraparamstimint_(Discret::Elements::ScaTraEleParameterTimInt::instance(disname)),
      scatraparams_(Discret::Elements::ScaTraEleParameterStd::instance(disname)),
      scatraparamsboundary_(Discret::Elements::ScaTraEleParameterBoundary::instance("scatra")),
      numdofpernode_(numdofpernode),
      numscal_(numscal),
      xyze_(true),  // initialize to zero
      weights_(true),
      myknots_(nsd_ele_),
      mypknots_(nsd_),
      normalfac_(1.0),
      ephinp_(numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(true)),
      edispnp_(true),
      diffus_(numscal_, 0),
      shcacp_(0.0),
      xsi_(true),
      funct_(true),
      deriv_(true),
      derxy_(true),
      normal_(true),
      velint_(true),
      metrictensor_(true),
      rotsymmpbc_(std::make_shared<FLD::RotationallySymmetricPeriodicBC<distype, nsd_ + 1,
              Discret::Elements::Fluid::none>>())
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::setup_calc(
    Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // Now do the nurbs specific stuff (for isogeometric elements)
  if (Core::FE::is_nurbs<distype>)
  {
    // for isogeometric elements --- get knotvectors for parent
    // element and boundary element, get weights
    bool zero_size = Core::FE::Nurbs::get_knot_vector_and_weights_for_nurbs_boundary(ele,
        ele->face_parent_number(), ele->parent_element()->id(), discretization, mypknots_, myknots_,
        weights_, normalfac_);

    // if we have a zero sized element due to a interpolated point -> exit here
    if (zero_size) return -1;
  }  // Nurbs specific stuff

  // rotationally symmetric periodic bc's: do setup for current element
  rotsymmpbc_->setup(ele);

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::evaluate(
    Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  //--------------------------------------------------------------------------------
  // preparations for element
  //--------------------------------------------------------------------------------
  if (setup_calc(ele, params, discretization) == -1) return 0;

  extract_displacement_values(ele, discretization, la);

  // check for the action parameter
  const auto action = Teuchos::getIntegralValue<ScaTra::BoundaryAction>(params, "action");
  // evaluate action
  evaluate_action(ele, params, discretization, action, la, elemat1_epetra, elemat2_epetra,
      elevec1_epetra, elevec2_epetra, elevec3_epetra);

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::extract_displacement_values(
    Core::Elements::FaceElement* ele, const Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  switch (ele->parent_element()->shape())
  {
    case Core::FE::CellType::hex8:
    {
      extract_displacement_values<Core::FE::CellType::hex8>(ele, discretization, la);
      break;
    }
    case Core::FE::CellType::hex27:
    {
      extract_displacement_values<Core::FE::CellType::hex27>(ele, discretization, la);
      break;
    }
    case Core::FE::CellType::tet4:
    {
      extract_displacement_values<Core::FE::CellType::tet4>(ele, discretization, la);
      break;
    }
    case Core::FE::CellType::quad4:
    {
      extract_displacement_values<Core::FE::CellType::quad4>(ele, discretization, la);
      break;
    }
    case Core::FE::CellType::tri6:
    {
      extract_displacement_values<Core::FE::CellType::tri6>(ele, discretization, la);
      break;
    }
    case Core::FE::CellType::tri3:
    {
      extract_displacement_values<Core::FE::CellType::tri3>(ele, discretization, la);
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      extract_displacement_values<Core::FE::CellType::nurbs9>(ele, discretization, la);
      break;
    }
    default:
      FOUR_C_THROW("Not implemented for discretization type: %i!", ele->parent_element()->shape());
      break;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType parentdistype>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::extract_displacement_values(
    Core::Elements::FaceElement* ele, const Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  // get additional state vector for ALE case: grid displacement
  if (scatraparams_->is_ale())
  {
    // get number of dof-set associated with displacement related dofs
    const int ndsdisp = scatraparams_->nds_disp();

    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
        discretization.get_state(ndsdisp, "dispnp");
    FOUR_C_ASSERT(dispnp != nullptr, "Cannot get state vector 'dispnp'");

    // determine number of displacement related dofs per node
    const int numdispdofpernode = la[ndsdisp].lm_.size() / nen_;

    // construct location vector for displacement related dofs
    std::vector<int> lmdisp(nsd_ * nen_, -1);
    for (int inode = 0; inode < nen_; ++inode)
      for (int idim = 0; idim < nsd_; ++idim)
        lmdisp[inode * nsd_ + idim] = la[ndsdisp].lm_[inode * numdispdofpernode + idim];

    // extract local values of displacement field from global state vector
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*dispnp, edispnp_, lmdisp);

    // add nodal displacements to point coordinates
    update_node_coordinates();

    // determine location array information of parent element
    Core::Elements::LocationArray parent_la(discretization.num_dof_sets());
    ele->parent_element()->location_vector(discretization, parent_la, false);

    const int num_node_parent_ele = Core::FE::num_nodes<parentdistype>;

    // determine number of the displacement related dofs per node
    const int parent_numdispdofpernode = parent_la[ndsdisp].lm_.size() / num_node_parent_ele;

    std::vector<int> parent_lmdisp(nsd_ * num_node_parent_ele, -1);
    for (int inode = 0; inode < num_node_parent_ele; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        parent_lmdisp[inode * nsd_ + idim] =
            parent_la[ndsdisp].lm_[inode * parent_numdispdofpernode + idim];
      }
    }

    // extract local values of displacement field from global state vector
    Core::LinAlg::Matrix<nsd_, num_node_parent_ele> parentdispnp;
    Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, num_node_parent_ele>>(
        *dispnp, parentdispnp, parent_lmdisp);

    eparentdispnp_.resize(num_node_parent_ele * nsd_);
    for (int i = 0; i < num_node_parent_ele; ++i)
      for (int idim = 0; idim < nsd_; ++idim)
        eparentdispnp_[i * nsd_ + idim] = parentdispnp(idim, i);
  }
  else
  {
    edispnp_.clear();
    eparentdispnp_.clear();
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::evaluate_action(
    Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, ScaTra::BoundaryAction action,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  std::vector<int>& lm = la[0].lm_;

  switch (action)
  {
    case ScaTra::BoundaryAction::calc_normal_vectors:
    {
      calc_normal_vectors(params, ele);
      break;
    }

    case ScaTra::BoundaryAction::integrate_shape_functions:
    {
      // NOTE: add area value only for elements which are NOT ghosted!
      const bool addarea =
          (ele->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()));
      integrate_shape_functions(ele, params, elevec1_epetra, addarea);

      break;
    }

    case ScaTra::BoundaryAction::calc_mass_matrix:
    {
      calc_mat_mass(ele, elemat1_epetra);

      break;
    }

    case ScaTra::BoundaryAction::calc_Neumann:
    {
      Core::Conditions::Condition* condition =
          params.get<Core::Conditions::Condition*>("condition");
      if (condition == nullptr) FOUR_C_THROW("Cannot access Neumann boundary condition!");

      evaluate_neumann(ele, params, discretization, *condition, la, elevec1_epetra, 1.);

      break;
    }

    case ScaTra::BoundaryAction::calc_Neumann_inflow:
    {
      neumann_inflow(ele, params, discretization, la, elemat1_epetra, elevec1_epetra);

      break;
    }

    case ScaTra::BoundaryAction::calc_convective_heat_transfer:
    {
      // get the parent element including its material
      Core::Elements::Element* parentele = ele->parent_element();
      std::shared_ptr<Core::Mat::Material> mat = parentele->material();

      // get values of scalar
      std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
      if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");

      // extract local values from global vector
      std::vector<Core::LinAlg::Matrix<nen_, 1>> ephinp(
          numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(true));
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp, lm);

      // get condition
      std::shared_ptr<Core::Conditions::Condition> cond =
          params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");
      if (cond == nullptr) FOUR_C_THROW("Cannot access condition 'TransportThermoConvections'!");

      // get heat transfer coefficient and surrounding temperature
      const auto heatranscoeff = cond->parameters().get<double>("coeff");
      const auto surtemp = cond->parameters().get<double>("surtemp");

      convective_heat_transfer(
          ele, mat, ephinp, elemat1_epetra, elevec1_epetra, heatranscoeff, surtemp);

      break;
    }

    case ScaTra::BoundaryAction::calc_weak_Dirichlet:
    {
      // get the parent element including its material
      Core::Elements::Element* parentele = ele->parent_element();
      std::shared_ptr<Core::Mat::Material> mat = parentele->material();

      if (numscal_ > 1) FOUR_C_THROW("not yet implemented for more than one scalar\n");

      switch (distype)
      {
        // 2D:
        case Core::FE::CellType::line2:
        {
          if (ele->parent_element()->shape() == Core::FE::CellType::quad4)
          {
            weak_dirichlet<Core::FE::CellType::line2, Core::FE::CellType::quad4>(
                ele, params, discretization, mat, elemat1_epetra, elevec1_epetra);
          }
          else
          {
            FOUR_C_THROW("expected combination quad4/hex8 or line2/quad4 for surface/parent pair");
          }
          break;
        }

        // 3D:
        case Core::FE::CellType::quad4:
        {
          if (ele->parent_element()->shape() == Core::FE::CellType::hex8)
          {
            weak_dirichlet<Core::FE::CellType::quad4, Core::FE::CellType::hex8>(
                ele, params, discretization, mat, elemat1_epetra, elevec1_epetra);
          }
          else
            FOUR_C_THROW("expected combination quad4/hex8 or line2/quad4 for surface/parent pair");

          break;
        }

        default:
        {
          FOUR_C_THROW("not implemented yet\n");
          break;
        }
      }

      break;
    }

    case ScaTra::BoundaryAction::calc_fs3i_surface_permeability:
    {
      evaluate_surface_permeability(
          ele, params, discretization, la, elemat1_epetra, elevec1_epetra);

      break;
    }

    case ScaTra::BoundaryAction::calc_fps3i_surface_permeability:
    {
      evaluate_kedem_katchalsky(ele, params, discretization, la, elemat1_epetra, elevec1_epetra);

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
          numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(true));
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp, lm);

      // get number of dofset associated with velocity related dofs
      const int ndsvel = scatraparams_->nds_vel();

      // get convective (velocity - mesh displacement) velocity at nodes
      std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
          discretization.get_state(ndsvel, "convective velocity field");
      if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

      // determine number of velocity related dofs per node
      const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

      // construct location vector for velocity related dofs
      std::vector<int> lmvel(nsd_ * nen_, -1);
      for (int inode = 0; inode < nen_; ++inode)
        for (int idim = 0; idim < nsd_; ++idim)
          lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

      // we deal with a nsd_-dimensional flow field
      Core::LinAlg::Matrix<nsd_, nen_> econvel(true);

      // extract local values of convective velocity field from global state vector
      Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*convel, econvel, lmvel);

      // rotate the vector field in the case of rotationally symmetric boundary conditions
      rotsymmpbc_->rotate_my_values_if_necessary(econvel);

      // for the moment we ignore the return values of this method
      calc_convective_flux(ele, ephinp, econvel, elevec1_epetra);

      break;
    }

    case ScaTra::BoundaryAction::calc_s2icoupling:
    {
      evaluate_s2_i_coupling(
          ele, params, discretization, la, elemat1_epetra, elemat2_epetra, elevec1_epetra);

      break;
    }

    case ScaTra::BoundaryAction::calc_s2icoupling_capacitance:
    {
      evaluate_s2_i_coupling_capacitance(
          discretization, la, elemat1_epetra, elemat2_epetra, elevec1_epetra, elevec2_epetra);

      break;
    }

    case ScaTra::BoundaryAction::calc_s2icoupling_od:
    {
      evaluate_s2_i_coupling_od(ele, params, discretization, la, elemat1_epetra);
      break;
    }

    case ScaTra::BoundaryAction::calc_s2icoupling_capacitance_od:
    {
      evaluate_s2_i_coupling_capacitance_od(
          params, discretization, la, elemat1_epetra, elemat2_epetra);
      break;
    }

    case ScaTra::BoundaryAction::calc_boundary_integral:
    {
      calc_boundary_integral(ele, elevec1_epetra);
      break;
    }
    case ScaTra::BoundaryAction::calc_nodal_size:
    {
      evaluate_nodal_size(ele, params, discretization, la, elevec1_epetra);
      break;
    }
    case ScaTra::BoundaryAction::calc_Robin:
    {
      calc_robin_boundary(ele, params, discretization, la, elemat1_epetra, elevec1_epetra, 1.);
      break;
    }
    case ScaTra::BoundaryAction::calc_s2icoupling_flux:
    {
      calc_s2_i_coupling_flux(ele, params, discretization, la, elevec1_epetra);
      break;
    }
    default:
    {
      FOUR_C_THROW("Not acting on this boundary action. Forgot implementation?");
      break;
    }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::evaluate_neumann(
    Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseVector& elevec1,
    const double scalar)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // find out whether we will use a time curve
  const double time = scatraparamstimint_->time();

  // get values, switches and spatial functions from the  condition
  // (assumed to be constant on element boundary)
  const int numdof = condition.parameters().get<int>("NUMDOF");
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition.parameters().get<std::vector<double>>("VAL");
  const auto func = condition.parameters().get<std::vector<Core::IO::Noneable<int>>>("FUNCT");

  if (numdofpernode_ != numdof)
  {
    FOUR_C_THROW(
        "The NUMDOF you have entered in your TRANSPORT NEUMANN CONDITION does not equal the number "
        "of scalars.");
  }

  // integration loop
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    double fac = eval_shape_func_and_int_fac(intpoints, iquad);

    // factor given by spatial function
    double functfac = 1.0;

    // determine global coordinates of current Gauss point
    Core::LinAlg::Matrix<nsd_, 1> coordgp;  // coordinate has always to be given in 3D!
    coordgp.multiply_nn(xyze_, funct_);

    const double* coordgpref = &coordgp(0);  // needed for function evaluation

    for (int dof = 0; dof < numdofpernode_; ++dof)
    {
      if (onoff[dof])  // is this dof activated?
      {
        // factor given by spatial function
        if (func[dof].has_value() && func[dof].value() > 0)
        {
          // evaluate function at current Gauss point (provide always 3D coordinates!)
          functfac = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfSpaceTime>(func[dof].value() - 1)
                         .evaluate(coordgpref, time, dof);
        }
        else
          functfac = 1.;

        const double val_fac_funct_fac = val[dof] * fac * functfac;

        for (int node = 0; node < nen_; ++node)
          // TODO: with or without eps_
          elevec1[node * numdofpernode_ + dof] += scalar * funct_(node) * val_fac_funct_fac;
      }  // if (onoff[dof])
    }  // loop over dofs
  }  // loop over integration points

  return 0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::calc_normal_vectors(
    Teuchos::ParameterList& params, Core::Elements::FaceElement* ele)
{
  // access the global vector
  const auto normals =
      params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("normal vectors", nullptr);
  if (normals == nullptr) FOUR_C_THROW("Could not access vector 'normal vectors'");

  // determine constant outer normal to this element
  if constexpr (nsd_ == 3 and nsd_ele_ == 1)
  {
    // get first 3 nodes in parent element
    auto* p_ele = ele->parent_element();
    FOUR_C_ASSERT(p_ele->num_node() >= 3, "Parent element must at least have 3 nodes.");
    Core::LinAlg::Matrix<nsd_, 3> xyz_parent_ele;

    for (int i_node = 0; i_node < 3; ++i_node)
    {
      const auto& coords = p_ele->nodes()[i_node]->x();
      for (int dim = 0; dim < nsd_; ++dim) xyz_parent_ele(dim, i_node) = coords[dim];
    }

    normal_ = get_const_normal(xyze_, xyz_parent_ele);
  }
  else if constexpr (nsd_ - nsd_ele_ == 1)
  {
    normal_ = get_const_normal(xyze_);
  }
  else
    FOUR_C_THROW("This combination of space dimension and element dimension makes no sense.");

  for (int j = 0; j < nen_; j++)
  {
    const int nodegid = (ele->nodes()[j])->id();
    if (normals->Map().MyGID(nodegid))
    {
      // scaling to a unit vector is performed on the global level after
      // assembly of nodal contributions since we have no reliable information
      // about the number of boundary elements adjacent to a node
      for (int dim = 0; dim < nsd_; dim++)
      {
        normals->SumIntoGlobalValue(nodegid, dim, normal_(dim));
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::neumann_inflow(
    const Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& emat, Core::LinAlg::SerialDenseVector& erhs)
{
  // get location vector associated with primary dofset
  std::vector<int>& lm = la[0].lm_;

  // get parent element
  Core::Elements::Element* parentele = ele->parent_element();

  // get material of parent element
  std::shared_ptr<Core::Mat::Material> material = parentele->material();

  // we don't know the parent element's lm vector; so we have to build it here
  const int nenparent = parentele->num_node();
  std::vector<int> lmparent(nenparent);
  std::vector<int> lmparentowner;
  std::vector<int> lmparentstride;
  parentele->location_vector(discretization, lmparent, lmparentowner, lmparentstride);

  // get values of scalar
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
  if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");

  // extract local values from global vector
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ephinp(
      numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(true));
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp, lm);

  // get number of dofset associated with velocity related dofs
  const int ndsvel = scatraparams_->nds_vel();

  // get convective (velocity - mesh displacement) velocity at nodes
  std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
      discretization.get_state(ndsvel, "convective velocity field");
  if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

  // determine number of velocity related dofs per node
  const int numveldofpernode = la[ndsvel].lm_.size() / nen_;

  // construct location vector for velocity related dofs
  std::vector<int> lmvel(nsd_ * nen_, -1);
  for (int inode = 0; inode < nen_; ++inode)
    for (int idim = 0; idim < nsd_; ++idim)
      lmvel[inode * nsd_ + idim] = la[ndsvel].lm_[inode * numveldofpernode + idim];

  // we deal with a nsd_-dimensional flow field
  Core::LinAlg::Matrix<nsd_, nen_> econvel(true);

  // extract local values of convective velocity field from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*convel, econvel, lmvel);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  rotsymmpbc_->rotate_my_values_if_necessary(econvel);

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over all scalars
  for (int k = 0; k < numdofpernode_; ++k)
  {
    // loop over all integration points
    for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
    {
      const double fac = eval_shape_func_and_int_fac(intpoints, iquad, &normal_);

      // get velocity at integration point
      velint_.multiply(econvel, funct_);

      // normal velocity
      const double normvel = velint_.dot(normal_);

      if (normvel < -0.0001)
      {
        // set density to 1.0
        double dens = get_density(material, ephinp, k);

        // integration factor for left-hand side
        const double lhsfac = dens * normvel * scatraparamstimint_->time_fac() * fac;

        // integration factor for right-hand side
        double rhsfac = 0.0;
        if (scatraparamstimint_->is_incremental() and scatraparamstimint_->is_gen_alpha())
          rhsfac = lhsfac / scatraparamstimint_->alpha_f();
        else if (not scatraparamstimint_->is_incremental() and scatraparamstimint_->is_gen_alpha())
          rhsfac = lhsfac * (1.0 - scatraparamstimint_->alpha_f()) / scatraparamstimint_->alpha_f();
        else if (scatraparamstimint_->is_incremental() and not scatraparamstimint_->is_gen_alpha())
          rhsfac = lhsfac;

        // matrix
        for (int vi = 0; vi < nen_; ++vi)
        {
          const double vlhs = lhsfac * funct_(vi);

          const int fvi = vi * numdofpernode_ + k;

          for (int ui = 0; ui < nen_; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            emat(fvi, fui) -= vlhs * funct_(ui);
          }
        }

        // scalar at integration point
        const double phi = funct_.dot(ephinp[k]);

        // rhs
        const double vrhs = rhsfac * phi;
        for (int vi = 0; vi < nen_; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          erhs[fvi] += vrhs * funct_(vi);
        }
      }
    }
  }
}  // Discret::Elements::ScaTraEleBoundaryCalc<distype>::neumann_inflow

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::get_density(
    std::shared_ptr<const Core::Mat::Material> material,
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& ephinp, const int k)
{
  // initialization
  double density(0.);

  // get density depending on material
  switch (material->material_type())
  {
    case Core::Materials::m_matlist:
    {
      const auto* actmat = static_cast<const Mat::MatList*>(material.get());

      const int matid = actmat->mat_id(0);

      if (actmat->material_by_id(matid)->material_type() == Core::Materials::m_scatra)
      {
        // set density to unity
        density = 1.;
      }
      else
        FOUR_C_THROW("type of material found in material list is not supported");

      break;
    }

    case Core::Materials::m_matlist_reactions:
    case Core::Materials::m_scatra:
    {
      // set density to unity
      density = 1.;

      break;
    }

    default:
    {
      FOUR_C_THROW("Invalid material type!");
      break;
    }
  }

  return density;
}  // Discret::Elements::ScaTraEleBoundaryCalc<distype>::GetDensity

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
std::vector<double>
Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::calc_convective_flux(
    const Core::Elements::FaceElement* ele,
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& ephinp,
    const Core::LinAlg::Matrix<nsd_, nen_>& evelnp, Core::LinAlg::SerialDenseVector& erhs)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  std::vector<double> integralflux(numscal_);

  // loop over all scalars
  for (int k = 0; k < numscal_; ++k)
  {
    integralflux[k] = 0.0;

    // loop over all integration points
    for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
    {
      const double fac = eval_shape_func_and_int_fac(intpoints, iquad, &normal_);

      // get velocity at integration point
      velint_.multiply(evelnp, funct_);

      // normal velocity (note: normal_ is already a unit(!) normal)
      const double normvel = velint_.dot(normal_);

      // scalar at integration point
      const double phi = funct_.dot(ephinp[k]);

      const double val = phi * normvel * fac;
      integralflux[k] += val;
      // add contribution to provided vector (distribute over nodes using shape fct.)
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * numdofpernode_ + k;
        erhs[fvi] += val * funct_(vi);
      }
    }
  }

  return integralflux;

}  // ScaTraEleBoundaryCalc<distype>::ConvectiveFlux

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::convective_heat_transfer(
    const Core::Elements::FaceElement* ele, std::shared_ptr<const Core::Mat::Material> material,
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& ephinp, Core::LinAlg::SerialDenseMatrix& emat,
    Core::LinAlg::SerialDenseVector& erhs, const double heatranscoeff, const double surtemp)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over all scalars
  for (int k = 0; k < numdofpernode_; ++k)
  {
    // loop over all integration points
    for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
    {
      const double fac = eval_shape_func_and_int_fac(intpoints, iquad, &normal_);

      // get specific heat capacity at constant volume
      double shc = 0.0;
      if (material->material_type() == Core::Materials::m_th_fourier_iso)
      {
        const auto* actmat = static_cast<const Mat::FourierIso*>(material.get());

        shc = actmat->capacity();
      }
      else if (material->material_type() == Core::Materials::m_thermostvenant)
      {
        const auto* actmat = static_cast<const Mat::ThermoStVenantKirchhoff*>(material.get());

        shc = actmat->capacity();
      }
      else
        FOUR_C_THROW("Material type is not supported for convective heat transfer!");

      // integration factor for left-hand side
      const double lhsfac = heatranscoeff * scatraparamstimint_->time_fac() * fac / shc;

      // integration factor for right-hand side
      double rhsfac = 0.0;
      if (scatraparamstimint_->is_incremental() and scatraparamstimint_->is_gen_alpha())
        rhsfac = lhsfac / scatraparamstimint_->alpha_f();
      else if (not scatraparamstimint_->is_incremental() and scatraparamstimint_->is_gen_alpha())
        rhsfac = lhsfac * (1.0 - scatraparamstimint_->alpha_f()) / scatraparamstimint_->alpha_f();
      else if (scatraparamstimint_->is_incremental() and not scatraparamstimint_->is_gen_alpha())
        rhsfac = lhsfac;

      // matrix
      for (int vi = 0; vi < nen_; ++vi)
      {
        const double vlhs = lhsfac * funct_(vi);

        const int fvi = vi * numdofpernode_ + k;

        for (int ui = 0; ui < nen_; ++ui)
        {
          const int fui = ui * numdofpernode_ + k;

          emat(fvi, fui) -= vlhs * funct_(ui);
        }
      }

      // scalar at integration point
      const double phi = funct_.dot(ephinp[k]);

      // rhs
      const double vrhs = rhsfac * (phi - surtemp);
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * numdofpernode_ + k;

        erhs[fvi] += vrhs * funct_(vi);
      }
    }
  }
}  // Discret::Elements::ScaTraEleBoundaryCalc<distype>::convective_heat_transfer

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::
    evaluate_spatial_derivative_of_area_integration_factor(
        const Core::FE::IntPointsAndWeights<nsd_ele_>& intpoints, const int iquad,
        Core::LinAlg::Matrix<nsd_, nen_>& dsqrtdetg_dd)
{
  // safety check
  if (nsd_ele_ != 2)
    FOUR_C_THROW("Computation of shape derivatives only implemented for 2D interfaces!");

  evaluate_shape_func_and_derivative_at_int_point(intpoints, iquad);

  // compute derivatives of spatial coordinates w.r.t. reference coordinates
  static Core::LinAlg::Matrix<nsd_ele_, nsd_> dxyz_drs;
  dxyz_drs.multiply_nt(deriv_, xyze_);

  // compute basic components of shape derivatives
  const double xr(dxyz_drs(0, 0)), xs(dxyz_drs(1, 0)), yr(dxyz_drs(0, 1)), ys(dxyz_drs(1, 1)),
      zr(dxyz_drs(0, 2)), zs(dxyz_drs(1, 2));
  const double denominator_inv =
      1.0 / std::sqrt(xr * xr * ys * ys + xr * xr * zs * zs - 2.0 * xr * xs * yr * ys -
                      2.0 * xr * xs * zr * zs + xs * xs * yr * yr + xs * xs * zr * zr +
                      yr * yr * zs * zs - 2.0 * yr * ys * zr * zs + ys * ys * zr * zr);
  const double numerator_xr = xr * ys * ys + xr * zs * zs - xs * yr * ys - xs * zr * zs;
  const double numerator_xs = -(xr * yr * ys + xr * zr * zs - xs * yr * yr - xs * zr * zr);
  const double numerator_yr = -(xr * xs * ys - xs * xs * yr - yr * zs * zs + ys * zr * zs);
  const double numerator_ys = xr * xr * ys - xr * xs * yr - yr * zr * zs + ys * zr * zr;
  const double numerator_zr = -(xr * xs * zs - xs * xs * zr + yr * ys * zs - ys * ys * zr);
  const double numerator_zs = xr * xr * zs - xr * xs * zr + yr * yr * zs - yr * ys * zr;

  // compute shape derivatives
  for (int ui = 0; ui < nen_; ++ui)
  {
    dsqrtdetg_dd(0, ui) =
        denominator_inv * (numerator_xr * deriv_(0, ui) + numerator_xs * deriv_(1, ui));
    dsqrtdetg_dd(1, ui) =
        denominator_inv * (numerator_yr * deriv_(0, ui) + numerator_ys * deriv_(1, ui));
    dsqrtdetg_dd(2, ui) =
        denominator_inv * (numerator_zr * deriv_(0, ui) + numerator_zs * deriv_(1, ui));
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::eval_shape_func_and_int_fac(
    const Core::FE::IntPointsAndWeights<nsd_ele_>& intpoints, const int iquad,
    Core::LinAlg::Matrix<nsd_, 1>* normalvec)
{
  evaluate_shape_func_and_derivative_at_int_point(intpoints, iquad);

  // the metric tensor and the area of an infinitesimal surface/line element
  // optional: get normal at integration point as well
  double drs(0.0);
  Core::FE::compute_metric_tensor_for_boundary_ele<distype, probdim>(
      xyze_, deriv_, metrictensor_, drs, true, normalvec);

  // for nurbs elements the normal vector must be scaled with a special orientation factor!!
  if (Core::FE::is_nurbs<distype>)
  {
    if (normalvec != nullptr) normal_.scale(normalfac_);
  }

  // return the integration factor
  return intpoints.ip().qwgt[iquad] * drs;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::
    evaluate_shape_func_and_derivative_at_int_point(
        const Core::FE::IntPointsAndWeights<nsd_ele_>& intpoints, const int iquad)
{
  // coordinates of the current integration point
  const double* gpcoord = (intpoints.ip().qxg)[iquad];
  for (int idim = 0; idim < nsd_ele_; idim++)
  {
    xsi_(idim) = gpcoord[idim];
  }

  if (not Core::FE::is_nurbs<distype>)
  {
    // shape functions and their first derivatives
    Core::FE::shape_function<distype>(xsi_, funct_);
    Core::FE::shape_function_deriv1<distype>(xsi_, deriv_);
  }
  else  // nurbs elements are always somewhat special...
  {
    Core::FE::Nurbs::nurbs_get_funct_deriv(funct_, deriv_, xsi_, myknots_, weights_, distype);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Core::LinAlg::Matrix<3, 1>
Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::get_const_normal(
    const Core::LinAlg::Matrix<3, nen_>& xyze)
{
  if (Core::FE::is_nurbs<distype>) FOUR_C_THROW("Element normal not implemented for NURBS");

  Core::LinAlg::Matrix<3, 1> normal(true), dist1(true), dist2(true);
  for (int i = 0; i < 3; i++)
  {
    dist1(i) = xyze(i, 1) - xyze(i, 0);
    dist2(i) = xyze(i, 2) - xyze(i, 0);
  }

  normal.cross_product(dist1, dist2);

  const double length = normal.norm2();
  if (length < 1.0e-16) FOUR_C_THROW("Zero length for element normal");

  normal.scale(1.0 / length);

  return normal;
}  // ScaTraEleBoundaryCalc<distype>::get_const_normal

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Core::LinAlg::Matrix<2, 1>
Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::get_const_normal(
    const Core::LinAlg::Matrix<2, nen_>& xyze)
{
  if (Core::FE::is_nurbs<distype>) FOUR_C_THROW("Element normal not implemented for NURBS");

  Core::LinAlg::Matrix<2, 1> normal(true);

  normal(0) = xyze(1, 1) - xyze(1, 0);
  normal(1) = (-1.0) * (xyze(0, 1) - xyze(0, 0));

  const double length = normal.norm2();
  if (length < 1.0e-16) FOUR_C_THROW("Zero length for element normal");

  normal.scale(1.0 / length);

  return normal;
}  // ScaTraEleBoundaryCalc<distype>::get_const_normal

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Core::LinAlg::Matrix<3, 1>
Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::get_const_normal(
    const Core::LinAlg::Matrix<3, nen_>& xyze, const Core::LinAlg::Matrix<3, 3>& nodes_parent_ele)
{
  if (Core::FE::is_nurbs<distype>) FOUR_C_THROW("Element normal not implemented for NURBS");

  Core::LinAlg::Matrix<3, 1> normal(true), normal_parent_ele(true), boundary_ele(true),
      parent_ele_v1(true), parent_ele_v2(true);

  for (int dim = 0; dim < 3; ++dim)
  {
    boundary_ele(dim, 0) = xyze(dim, 0) - xyze(dim, 1);
    parent_ele_v1(dim, 0) = nodes_parent_ele(dim, 0) - nodes_parent_ele(dim, 1);
    parent_ele_v2(dim, 0) = nodes_parent_ele(dim, 0) - nodes_parent_ele(dim, 2);
  }

  normal_parent_ele.cross_product(parent_ele_v1, parent_ele_v2);
  normal.cross_product(normal_parent_ele, boundary_ele);

  // compute inward vector and check if its scalar product with the normal vector is negative.
  // Otherwise, change the sign of the normal vector
  Core::LinAlg::Matrix<3, 1> distance(true), inward_vector(true);
  // find node on parent element, that has non-zero distance to all boundary nodes
  for (int i_parent_node = 0; i_parent_node < 3; ++i_parent_node)
  {
    bool is_boundary_node = false;
    for (int i_boundary_node = 0; i_boundary_node < nen_; ++i_boundary_node)
    {
      for (int dim = 0; dim < 3; ++dim)
        distance(dim, 0) = nodes_parent_ele(dim, i_parent_node) - xyze(dim, i_boundary_node);

      // if the distance of the parent element to one boundary node is zero, it cannot be a
      // non-boundary node
      if (distance.norm2() < 1.0e-10)
      {
        is_boundary_node = true;
        break;
      }
    }
    if (!is_boundary_node)
    {
      inward_vector.update(1.0, distance, 0.0);
      break;
    }
  }
  if (inward_vector.dot(normal) >= 0.0) normal.scale(-1.0);

  const double length = normal.norm2();
  if (length < 1.0e-16) FOUR_C_THROW("Zero length for element normal");

  normal.scale(1.0 / length);

  return normal;
}  // ScaTraEleBoundaryCalc<distype>::get_const_normal

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::evaluate_s2_i_coupling(
    const Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& eslavematrix, Core::LinAlg::SerialDenseMatrix& emastermatrix,
    Core::LinAlg::SerialDenseVector& eslaveresidual)
{
  // extract local nodal values on present and opposite sides of scatra-scatra interface
  extract_node_values(discretization, la);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterphinp(numscal_);
  extract_node_values(emasterphinp, discretization, la, "imasterphinp");

  // dummy element matrix and vector
  Core::LinAlg::SerialDenseMatrix dummymatrix;
  Core::LinAlg::SerialDenseVector dummyvector;

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  Core::LinAlg::Matrix<nsd_, 1> normal;

  // element slave mechanical stress tensor
  const bool is_pseudo_contact = scatraparamsboundary_->is_pseudo_contact();
  std::vector<Core::LinAlg::Matrix<nen_, 1>> eslavestress_vector(
      6, Core::LinAlg::Matrix<nen_, 1>(true));
  if (is_pseudo_contact)
    extract_node_values(eslavestress_vector, discretization, la, "mechanicalStressState",
        scatraparams_->nds_two_tensor_quantity());

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac =
        Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::eval_shape_func_and_int_fac(
            intpoints, gpid, &normal);

    // evaluate overall integration factors
    const double timefacfac = scatraparamstimint_->time_fac() * fac;
    const double timefacrhsfac = scatraparamstimint_->time_fac_rhs() * fac;
    if (timefacfac < 0. or timefacrhsfac < 0.) FOUR_C_THROW("Integration factor is negative!");

    const double pseudo_contact_fac =
        calculate_pseudo_contact_factor(is_pseudo_contact, eslavestress_vector, normal, funct_);

    evaluate_s2_i_coupling_at_integration_point<distype>(ephinp_, emasterphinp, pseudo_contact_fac,
        funct_, funct_, funct_, funct_, numscal_, scatraparamsboundary_, timefacfac, timefacrhsfac,
        eslavematrix, emastermatrix, dummymatrix, dummymatrix, eslaveresidual, dummyvector);
  }  // end of loop over integration points
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType distype_master>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::
    evaluate_s2_i_coupling_at_integration_point(
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>& eslavephinp,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>>&
            emasterphinp,
        const double pseudo_contact_fac, const Core::LinAlg::Matrix<nen_, 1>& funct_slave,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& funct_master,
        const Core::LinAlg::Matrix<nen_, 1>& test_slave,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<distype_master>, 1>& test_master,
        const int numscal,
        const Discret::Elements::ScaTraEleParameterBoundary* const scatra_parameter_boundary,
        const double timefacfac, const double timefacrhsfac, Core::LinAlg::SerialDenseMatrix& k_ss,
        Core::LinAlg::SerialDenseMatrix& k_sm, Core::LinAlg::SerialDenseMatrix& k_ms,
        Core::LinAlg::SerialDenseMatrix& k_mm, Core::LinAlg::SerialDenseVector& r_s,
        Core::LinAlg::SerialDenseVector& r_m)
{
  // get condition specific parameters
  const int kineticmodel = scatra_parameter_boundary->kinetic_model();
  const std::vector<double>* permeabilities = scatra_parameter_boundary->permeabilities();

  // number of nodes of master-side mortar element
  const int nen_master = Core::FE::num_nodes<distype_master>;

  // loop over scalars
  for (int k = 0; k < numscal; ++k)
  {
    // evaluate dof values at current integration point on slave and master sides of scatra-scatra
    // interface
    const double slavephiint = funct_slave.dot(eslavephinp[k]);
    const double masterphiint = funct_master.dot(emasterphinp[k]);

    // compute matrix and vector contributions according to kinetic model for current scatra-scatra
    // interface coupling condition
    switch (kineticmodel)
    {
      // constant permeability model
      case Inpar::S2I::kinetics_constperm:
      {
        if (permeabilities == nullptr)
          FOUR_C_THROW(
              "Cannot access vector of permeabilities for scatra-scatra interface coupling!");
        if (permeabilities->size() != (unsigned)numscal)
          FOUR_C_THROW("Number of permeabilities does not match number of scalars!");

        // core residual
        const double N_timefacrhsfac = pseudo_contact_fac * timefacrhsfac * (*permeabilities)[k] *
                                       (slavephiint - masterphiint);

        // core linearizations
        const double dN_dc_slave_timefacfac =
            pseudo_contact_fac * timefacfac * (*permeabilities)[k];
        const double dN_dc_master_timefacfac = -dN_dc_slave_timefacfac;

        if (k_ss.numRows() and k_sm.numRows() and r_s.length())
        {
          for (int vi = 0; vi < nen_; ++vi)
          {
            const int fvi = vi * numscal + k;

            for (int ui = 0; ui < nen_; ++ui)
              k_ss(fvi, ui * numscal + k) +=
                  test_slave(vi) * dN_dc_slave_timefacfac * funct_slave(ui);

            for (int ui = 0; ui < nen_master; ++ui)
              k_sm(fvi, ui * numscal + k) +=
                  test_slave(vi) * dN_dc_master_timefacfac * funct_master(ui);

            r_s[fvi] -= test_slave(vi) * N_timefacrhsfac;
          }
        }
        else if (k_ss.numRows() or k_sm.numRows() or r_s.length())
          FOUR_C_THROW(
              "Must provide both slave-side matrices and slave-side vector or none of them!");

        if (k_ms.numRows() and k_mm.numRows() and r_m.length())
        {
          for (int vi = 0; vi < nen_master; ++vi)
          {
            const int fvi = vi * numscal + k;

            for (int ui = 0; ui < nen_; ++ui)
              k_ms(fvi, ui * numscal + k) -=
                  test_master(vi) * dN_dc_slave_timefacfac * funct_slave(ui);

            for (int ui = 0; ui < nen_master; ++ui)
              k_mm(fvi, ui * numscal + k) -=
                  test_master(vi) * dN_dc_master_timefacfac * funct_master(ui);

            r_m[fvi] += test_master(vi) * N_timefacrhsfac;
          }
        }
        else if (k_ms.numRows() or k_mm.numRows() or r_m.length())
          FOUR_C_THROW(
              "Must provide both master-side matrices and master-side vector or none of them!");

        break;
      }

      default:
      {
        FOUR_C_THROW("Kinetic model for scatra-scatra interface coupling not yet implemented!");
        break;
      }
    }
  }  // end of loop over scalars
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::calculate_pseudo_contact_factor(
    const bool is_pseudo_contact,
    const std::vector<Core::LinAlg::Matrix<nen_, 1>>& eslavestress_vector,
    const Core::LinAlg::Matrix<nsd_, 1>& gp_normal,
    const Core::LinAlg::Matrix<nen_, 1>& funct_slave)
{
  if (is_pseudo_contact)
  {
    static Core::LinAlg::Matrix<1, 1> normal_stress_comp_gp;
    static Core::LinAlg::Matrix<nsd_, nsd_> current_gp_stresses;
    static Core::LinAlg::Matrix<nsd_, 1> tmp;
    current_gp_stresses(0, 0) = funct_slave.dot(eslavestress_vector[0]);
    current_gp_stresses(1, 1) = funct_slave.dot(eslavestress_vector[1]);
    current_gp_stresses(2, 2) = funct_slave.dot(eslavestress_vector[2]);
    current_gp_stresses(0, 1) = current_gp_stresses(1, 0) = funct_slave.dot(eslavestress_vector[3]);
    current_gp_stresses(1, 2) = current_gp_stresses(2, 1) = funct_slave.dot(eslavestress_vector[4]);
    current_gp_stresses(0, 2) = current_gp_stresses(2, 0) = funct_slave.dot(eslavestress_vector[5]);

    tmp.multiply_nn(1.0, current_gp_stresses, gp_normal, 0.0);
    normal_stress_comp_gp.multiply_tn(1.0, gp_normal, tmp, 0.0);

    // if tensile stress, i.e. normal stress component > 0 return 0.0, otherwise return 1.0
    return normal_stress_comp_gp(0) > 0.0 ? 0.0 : 1.0;
  }
  else
    return 1.0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double
Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::calculate_det_f_of_parent_element(
    const Core::Elements::FaceElement* faceele, const double* faceele_xsi)
{
  if (scatraparams_->is_ale())
  {
    switch (faceele->parent_element()->shape())
    {
      case Core::FE::CellType::hex8:
      {
        return calculate_det_f_of_parent_element<Core::FE::CellType::hex8>(faceele, faceele_xsi);
      }
      case Core::FE::CellType::tet4:
      {
        return calculate_det_f_of_parent_element<Core::FE::CellType::tet4>(faceele, faceele_xsi);
      }
      default:
      {
        FOUR_C_THROW(
            "Not implemented for discretization type: %i!", faceele->parent_element()->shape());
        break;
      }
    }
  }

  return 1.0;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType parentdistype>
double
Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::calculate_det_f_of_parent_element(
    const Core::Elements::FaceElement* faceele, const double* faceele_xi)
{
  const int parent_ele_dim = Core::FE::dim<parentdistype>;
  const int parent_ele_num_nodes = Core::FE::num_nodes<parentdistype>;

  auto parent_xi =
      Core::FE::calculate_parent_gp_from_face_element_data<parent_ele_dim>(faceele_xi, faceele);
  static Core::LinAlg::Matrix<probdim, probdim> defgrd;

  static Core::LinAlg::Matrix<parent_ele_num_nodes, probdim> xdisp, xrefe, xcurr;

  for (auto i = 0; i < parent_ele_num_nodes; ++i)
  {
    const auto& x = faceele->parent_element()->nodes()[i]->x();
    for (auto dim = 0; dim < probdim; ++dim)
    {
      xdisp(i, dim) = eparentdispnp_.at(i * probdim + dim);
      xrefe(i, dim) = x[dim];
    }
  }

  Core::LinAlg::Matrix<probdim, parent_ele_num_nodes> deriv_parent(true);
  Core::FE::shape_function_deriv1<parentdistype>(parent_xi, deriv_parent);

  static Core::LinAlg::Matrix<probdim, probdim> inv_detF;
  inv_detF.multiply(deriv_parent, xrefe);
  inv_detF.invert();

  static Core::LinAlg::Matrix<probdim, parent_ele_num_nodes> N_XYZ;
  xcurr.update(1.0, xrefe, 1.0, xdisp);
  N_XYZ.multiply(inv_detF, deriv_parent);
  defgrd.multiply_tt(xcurr, N_XYZ);

  return defgrd.determinant();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::evaluate_s2_i_coupling_od(
    const Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& eslavematrix)
{
  // extract local nodal values on present and opposite side of scatra-scatra interface
  extract_node_values(discretization, la);
  std::vector<Core::LinAlg::Matrix<nen_, 1>> emasterphinp(
      numscal_, Core::LinAlg::Matrix<nen_, 1>(true));
  extract_node_values(emasterphinp, discretization, la, "imasterphinp");

  Core::LinAlg::Matrix<nsd_, 1> normal;

  // element slave mechanical stress tensor
  const bool is_pseudo_contact = scatraparamsboundary_->is_pseudo_contact();
  std::vector<Core::LinAlg::Matrix<nen_, 1>> eslavestress_vector(
      6, Core::LinAlg::Matrix<nen_, 1>(true));
  if (is_pseudo_contact)
    extract_node_values(eslavestress_vector, discretization, la, "mechanicalStressState",
        scatraparams_->nds_two_tensor_quantity());

  // get current scatra-scatra interface coupling condition
  std::shared_ptr<Core::Conditions::Condition> s2icondition =
      params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");
  if (s2icondition == nullptr)
    FOUR_C_THROW("Cannot access scatra-scatra interface coupling condition!");

  // get primary variable to derive the linearization
  const auto differentiationtype =
      Teuchos::getIntegralValue<ScaTra::DifferentiationType>(params, "differentiationtype");

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions at current integration point
    eval_shape_func_and_int_fac(intpoints, gpid, &normal);

    const double pseudo_contact_fac =
        calculate_pseudo_contact_factor(is_pseudo_contact, eslavestress_vector, normal, funct_);

    // evaluate shape derivatives
    static Core::LinAlg::Matrix<nsd_, nen_> dsqrtdetg_dd;
    if (differentiationtype == ScaTra::DifferentiationType::disp)
      evaluate_spatial_derivative_of_area_integration_factor(intpoints, gpid, dsqrtdetg_dd);

    // evaluate overall integration factor
    const double timefacwgt = scatraparamstimint_->time_fac() * intpoints.ip().qwgt[gpid];
    if (timefacwgt < 0.) FOUR_C_THROW("Integration factor is negative!");

    // loop over scalars
    for (int k = 0; k < numscal_; ++k)
    {
      // evaluate dof values at current integration point on slave and master sides of scatra-scatra
      // interface
      const double slavephiint = funct_.dot(ephinp_[k]);
      const double masterphiint = funct_.dot(emasterphinp[k]);

      // compute matrix contributions according to kinetic model for current scatra-scatra interface
      // coupling condition
      switch (scatraparamsboundary_->kinetic_model())
      {
        // constant permeability model
        case Inpar::S2I::kinetics_constperm:
        {
          // dervivative of interface flux w.r.t. displacement
          switch (differentiationtype)
          {
            case ScaTra::DifferentiationType::disp:
            {
              // access real vector of constant permeabilities associated with current condition
              const std::vector<double>* permeabilities = scatraparamsboundary_->permeabilities();
              if (permeabilities == nullptr)
                FOUR_C_THROW(
                    "Cannot access vector of permeabilities for scatra-scatra interface coupling!");
              if (permeabilities->size() != static_cast<unsigned>(numscal_))
                FOUR_C_THROW("Number of permeabilities does not match number of scalars!");

              // core linearization
              const double dN_dsqrtdetg_timefacwgt = pseudo_contact_fac * timefacwgt *
                                                     (*permeabilities)[k] *
                                                     (slavephiint - masterphiint);

              // loop over matrix columns
              for (int ui = 0; ui < nen_; ++ui)
              {
                const int fui = ui * 3;

                // loop over matrix rows
                for (int vi = 0; vi < nen_; ++vi)
                {
                  const int fvi = vi * numscal_ + k;
                  const double vi_dN_dsqrtdetg = funct_(vi) * dN_dsqrtdetg_timefacwgt;

                  // loop over spatial dimensions
                  for (int dim = 0; dim < 3; ++dim)
                    // compute linearizations w.r.t. slave-side structural displacements
                    eslavematrix(fvi, fui + dim) += vi_dN_dsqrtdetg * dsqrtdetg_dd(dim, ui);
                }
              }
              break;
            }
            default:
            {
              FOUR_C_THROW("Unknown primary quantity to calculate derivative");
              break;
            }
          }

          break;
        }

        default:
        {
          FOUR_C_THROW("Kinetic model for scatra-scatra interface coupling not yet implemented!");
          break;
        }
      }  // selection of kinetic model
    }  // loop over scalars
  }  // loop over integration points
}  // Discret::Elements::ScaTraEleBoundaryCalc<distype>::evaluate_s2_i_coupling_od

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::extract_node_values(
    const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  // extract nodal state variables associated with time t_{n+1} or t_{n+alpha_f}
  extract_node_values(ephinp_, discretization, la);
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::extract_node_values(
    Core::LinAlg::Matrix<nen_, 1>& estate, const Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, const std::string& statename, const int& nds) const
{
  // initialize matrix vector
  std::vector<Core::LinAlg::Matrix<nen_, 1>> estate_temp(1, Core::LinAlg::Matrix<nen_, 1>(true));

  // call more general routine
  extract_node_values(estate_temp, discretization, la, statename, nds);

  // copy extracted state variables
  estate = estate_temp[0];
}

/*-----------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::extract_node_values(
    std::vector<Core::LinAlg::Matrix<nen_, 1>>& estate,
    const Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    const std::string& statename, const int& nds) const
{
  // extract global state vector from discretization
  const std::shared_ptr<const Core::LinAlg::Vector<double>> state =
      discretization.get_state(nds, statename);
  if (state == nullptr)
    FOUR_C_THROW("Cannot extract state vector \"" + statename + "\" from discretization!");

  // extract nodal state variables associated with boundary element
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*state, estate, la[nds].lm_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::calc_boundary_integral(
    const Core::Elements::FaceElement* ele, Core::LinAlg::SerialDenseVector& scalar)
{
  // initialize variable for boundary integral
  double boundaryintegral(0.);

  // get integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate values of shape functions and boundary integration factor at current integration
    // point
    const double fac =
        Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::eval_shape_func_and_int_fac(
            intpoints, iquad);

    // add contribution from current integration point to boundary integral
    boundaryintegral += fac;
  }  // loop over integration points

  // write result into result vector
  scalar(0) = boundaryintegral;
}  // Discret::Elements::ScaTraEleBoundaryCalc<distype>::calc_boundary_integral

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::calc_mat_mass(
    const Core::Elements::FaceElement* const element, Core::LinAlg::SerialDenseMatrix& massmatrix)
{
  // get integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate values of shape functions and boundary integration factor at current integration
    // point
    const double fac =
        Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::eval_shape_func_and_int_fac(
            intpoints, iquad);

    // add contribution from current integration point to element mass matrix
    for (int k = 0; k < numdofpernode_; ++k)
    {
      for (int vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * numdofpernode_ + k;

        for (int ui = 0; ui < nen_; ++ui)
          massmatrix(fvi, ui * numdofpernode_ + k) += funct_(vi) * funct_(ui) * fac;
      }
    }
  }  // loop over integration points
}  // Discret::Elements::ScaTraEleBoundaryCalc<distype>::calc_boundary_integral

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::calc_robin_boundary(
    Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra, const double scalar)
{
  //////////////////////////////////////////////////////////////////////
  //              get current condition and parameters
  //////////////////////////////////////////////////////////////////////

  // get current condition
  std::shared_ptr<Core::Conditions::Condition> cond =
      params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");
  if (cond == nullptr) FOUR_C_THROW("Cannot access condition 'TransportRobin'");

  // get on/off flags
  const auto onoff = cond->parameters().get<std::vector<int>>("ONOFF");

  // safety check
  if ((int)(onoff.size()) != numscal_)
  {
    FOUR_C_THROW(
        "Mismatch in size for Robin boundary conditions, onoff has length %i, but you have %i "
        "scalars",
        onoff.size(), numscal_);
  }

  // extract prefactor and reference value from condition
  const auto prefac = cond->parameters().get<double>("PREFACTOR");
  const auto refval = cond->parameters().get<double>("REFVALUE");

  //////////////////////////////////////////////////////////////////////
  //                  read nodal values
  //////////////////////////////////////////////////////////////////////

  std::vector<int>& lm = la[0].lm_;

  // ------------get values of scalar transport------------------
  // extract global state vector from discretization
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
  if (phinp == nullptr) FOUR_C_THROW("Cannot read state vector \"phinp\" from discretization!");

  // extract local nodal state variables from global state vector
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ephinp(
      numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(true));
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp, lm);

  //////////////////////////////////////////////////////////////////////
  //                  build RHS and StiffMat
  //////////////////////////////////////////////////////////////////////

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over all scalars
  for (int k = 0; k < numscal_; ++k)
  {
    // flag for dofs to be considered by robin conditions
    if (onoff[k] == 1)
    {
      for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
      {
        // evaluate values of shape functions and domain integration factor at current integration
        // point
        const double intfac =
            Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::eval_shape_func_and_int_fac(
                intpoints, gpid);

        // evaluate reference concentration factor
        const double refconcfac = fac_for_ref_conc(gpid, ele, params, discretization);

        // evaluate overall integration factors
        const double fac_3 = prefac * intfac * refconcfac;

        // evaluate current scalar at current integration point
        const double phinp_gp = funct_.dot(ephinp[k]);

        // build RHS and matrix
        {
          //////////////////////////////////////////////////////////////////////
          //                  rhs
          //////////////////////////////////////////////////////////////////////
          const double vrhs = scatraparamstimint_->time_fac_rhs() * (phinp_gp - refval) * fac_3;

          for (int vi = 0; vi < nen_; ++vi)
          {
            const int fvi = vi * numscal_ + k;

            elevec1_epetra[fvi] += vrhs * funct_(vi);
          }

          //////////////////////////////////////////////////////////////////////
          //                  matrix
          //////////////////////////////////////////////////////////////////////
          for (int vi = 0; vi < nen_; ++vi)
          {
            const double vlhs = scatraparamstimint_->time_fac() * fac_3 * funct_(vi);
            const int fvi = vi * numscal_ + k;

            for (int ui = 0; ui < nen_; ++ui)
            {
              const int fui = ui * numdofpernode_ + k;

              elemat1_epetra(fvi, fui) -= vlhs * funct_(ui);
            }
          }
        }
      }  // loop over integration points
    }  // if(onoff[k]==1)
    // else //in the case of "OFF", a no flux condition is automatically applied

  }  // loop over scalars
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::evaluate_surface_permeability(
    const Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseVector& elevec1)
{
  //////////////////////////////////////////////////////////////////////
  //                  read nodal values
  //////////////////////////////////////////////////////////////////////

  if (scatraparamstimint_->is_gen_alpha() or not scatraparamstimint_->is_incremental())
    FOUR_C_THROW("Not a valid time integration scheme!");

  std::vector<int>& lm = la[0].lm_;

  // ------------get values of scalar transport------------------
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
  if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
  // extract local values from global vector
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ephinp(
      numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(true));
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp, lm);

  //------------get membrane concentration at the interface (i.e. within the
  // membrane)------------------
  std::shared_ptr<const Core::LinAlg::Vector<double>> phibar =
      discretization.get_state("MembraneConcentration");
  if (phibar == nullptr) FOUR_C_THROW("Cannot get state vector 'MembraneConcentration'");
  // extract local values from global vector
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ephibar(
      numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(true));
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phibar, ephibar, lm);

  // ------------get values of wall shear stress-----------------------
  // get number of dofset associated with pressure related dofs
  const int ndswss = scatraparams_->nds_wss();
  if (ndswss == -1) FOUR_C_THROW("Cannot get number of dofset of wss vector");
  std::shared_ptr<const Core::LinAlg::Vector<double>> wss =
      discretization.get_state(ndswss, "WallShearStress");
  if (wss == nullptr) FOUR_C_THROW("Cannot get state vector 'WallShearStress'");

  // determine number of velocity (and pressure) related dofs per node
  const int numwssdofpernode = la[ndswss].lm_.size() / nen_;
  // construct location vector for wss related dofs
  std::vector<int> lmwss(nsd_ * nen_, -1);
  for (int inode = 0; inode < nen_; ++inode)
    for (int idim = 0; idim < nsd_; ++idim)
      lmwss[inode * nsd_ + idim] = la[ndswss].lm_[inode * numwssdofpernode + idim];

  Core::LinAlg::Matrix<nsd_, nen_> ewss(true);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*wss, ewss, lmwss);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  // rotsymmpbc_->rotate_my_values_if_necessary(ewss);

  //////////////////////////////////////////////////////////////////////
  //                  get current condition
  //////////////////////////////////////////////////////////////////////

  std::shared_ptr<Core::Conditions::Condition> cond =
      params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");
  if (cond == nullptr) FOUR_C_THROW("Cannot access condition 'SurfacePermeability'");

  const auto onoff = cond->parameters().get<std::vector<int>>("ONOFF");

  const auto perm = cond->parameters().get<double>("PERMCOEF");

  // get flag if concentration flux across membrane is affected by local wall shear stresses:
  // false->no, true->yes
  const bool wss_onoff = cond->parameters().get<bool>("WSSON");

  const auto* coeffs = &cond->parameters().get<std::vector<double>>("WSSCOEFFS");

  //////////////////////////////////////////////////////////////////////
  //                  build RHS and StiffMat
  //////////////////////////////////////////////////////////////////////
  {
    // integration points and weights
    const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
        ScaTra::DisTypeToOptGaussRule<distype>::rule);

    // define vector for wss concentration values at nodes
    //  Core::LinAlg::Matrix<nen_,1> fwssnod(true);

    // loop over all scalars
    for (int k = 0; k < numdofpernode_; ++k)
    {
      // flag for dofs to be considered by membrane equations of Kedem and Katchalsky
      if (onoff[k] == 1)
      {
        // loop over all integration points
        for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
        {
          const double fac = eval_shape_func_and_int_fac(intpoints, iquad, &normal_);
          const double refconcfac = fac_for_ref_conc(iquad, ele, params, discretization);
          // integration factor for right-hand side
          double facfac = 0.0;
          if (scatraparamstimint_->is_incremental() and not scatraparamstimint_->is_gen_alpha())
            facfac = scatraparamstimint_->time_fac() * fac * refconcfac;
          else
            FOUR_C_THROW("evaluate_surface_permeability: Requested scheme not yet implemented");

          // scalar at integration point
          const double phi = funct_.dot(ephinp[k]);

          // permeabilty scaling factor (depending on the norm of the wss) at integration point
          const double facWSS = ws_sinfluence(ewss, wss_onoff, coeffs);

          // matrix
          for (int vi = 0; vi < nen_; ++vi)
          {
            const double vlhs = facfac * facWSS * perm * funct_(vi);
            const int fvi = vi * numdofpernode_ + k;

            for (int ui = 0; ui < nen_; ++ui)
            {
              const int fui = ui * numdofpernode_ + k;

              elemat1(fvi, fui) += vlhs * funct_(ui);
            }
          }

          // rhs
          const double vrhs = facfac * facWSS * perm * phi;

          for (int vi = 0; vi < nen_; ++vi)
          {
            const int fvi = vi * numdofpernode_ + k;

            elevec1[fvi] -= vrhs * funct_(vi);
          }
        }
      }  // if(onoff[k]==1)
      // else //in the case of "OFF", a no flux condition is automatically applied
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::evaluate_kedem_katchalsky(
    const Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseVector& elevec1)
{
  // safety checks
  if (scatraparamstimint_->is_gen_alpha() or not scatraparamstimint_->is_incremental())
    FOUR_C_THROW("Not a valid time integration scheme!");

  std::vector<int>& lm = la[0].lm_;

  // ------------get values of scalar transport------------------
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
  if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
  // extract local values from global vector
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ephinp(
      numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(true));
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phinp, ephinp, lm);


  //------------get membrane concentration at the interface (i.e. within the
  // membrane)------------------
  std::shared_ptr<const Core::LinAlg::Vector<double>> phibar =
      discretization.get_state("MembraneConcentration");
  if (phibar == nullptr) FOUR_C_THROW("Cannot get state vector 'MembraneConcentration'");
  // extract local values from global vector
  std::vector<Core::LinAlg::Matrix<nen_, 1>> ephibar(
      numdofpernode_, Core::LinAlg::Matrix<nen_, 1>(true));
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*phibar, ephibar, lm);


  //--------get values of pressure at the interface ----------------------
  // get number of dofset associated with pressure related dofs
  const int ndspres = scatraparams_->nds_pres();
  if (ndspres == -1) FOUR_C_THROW("Cannot get number of dofset of pressure vector");
  std::shared_ptr<const Core::LinAlg::Vector<double>> pressure =
      discretization.get_state(ndspres, "Pressure");
  if (pressure == nullptr) FOUR_C_THROW("Cannot get state vector 'Pressure'");

  // determine number of velocity (and pressure) related dofs per node
  const int numveldofpernode = la[ndspres].lm_.size() / nen_;
  // construct location vector for pressure related dofs
  std::vector<int> lmpres(nen_, -1);
  for (int inode = 0; inode < nen_; ++inode)
    lmpres[inode] = la[ndspres].lm_[inode * numveldofpernode + nsd_];  // only pressure dofs

  Core::LinAlg::Matrix<nen_, 1> epressure(true);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nen_, 1>>(*pressure, epressure, lmpres);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  // rotsymmpbc_->rotate_my_values_if_necessary(epressure);


  // ------------get values of wall shear stress-----------------------
  // get number of dofset associated with pressure related dofs
  const int ndswss = scatraparams_->nds_wss();
  if (ndswss == -1) FOUR_C_THROW("Cannot get number of dofset of wss vector");
  std::shared_ptr<const Core::LinAlg::Vector<double>> wss =
      discretization.get_state(ndswss, "WallShearStress");
  if (wss == nullptr) FOUR_C_THROW("Cannot get state vector 'WallShearStress'");

  // determine number of velocity (and pressure) related dofs per node
  const int numwssdofpernode = la[ndswss].lm_.size() / nen_;
  // construct location vector for wss related dofs
  std::vector<int> lmwss(nsd_ * nen_, -1);
  for (int inode = 0; inode < nen_; ++inode)
    for (int idim = 0; idim < nsd_; ++idim)
      lmwss[inode * nsd_ + idim] = la[ndswss].lm_[inode * numwssdofpernode + idim];

  Core::LinAlg::Matrix<nsd_, nen_> ewss(true);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<nsd_, nen_>>(*wss, ewss, lmwss);

  // ------------get current condition----------------------------------
  std::shared_ptr<Core::Conditions::Condition> cond =
      params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");
  if (cond == nullptr)
    FOUR_C_THROW("Cannot access condition 'DESIGN SCATRA COUPLING SURF CONDITIONS'");

  const auto onoff = cond->parameters().get<std::vector<int>>("ONOFF");

  // get the standard permeability of the interface
  const auto perm = cond->parameters().get<double>("PERMCOEF");

  // get flag if concentration flux across membrane is affected by local wall shear stresses: 0->no
  // 1->yes
  const bool wss_onoff = cond->parameters().get<bool>("WSSON");
  const auto* coeffs = &cond->parameters().get<std::vector<double>>("WSSCOEFFS");

  // hydraulic conductivity at interface
  const auto conductivity = cond->parameters().get<double>("CONDUCT");

  // Staverman filtration coefficient at interface
  const auto sigma = cond->parameters().get<double>("FILTR");

  ///////////////////////////////////////////////////////////////////////////
  // ------------do the actual calculations----------------------------------
  ///////////////////////////////////////////////////////////////////////////

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over all scalars
  for (int k = 0; k < numdofpernode_; ++k)  // numdofpernode_//1
  {
    // flag for dofs to be considered by membrane equations of Kedem and Katchalsky
    if (onoff[k] == 1)
    {
      // loop over all integration points
      for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
      {
        const double fac = eval_shape_func_and_int_fac(intpoints, iquad, &normal_);

        // integration factor
        double facfac = 0.0;
        if (scatraparamstimint_->is_incremental() and not scatraparamstimint_->is_gen_alpha())
          facfac = scatraparamstimint_->time_fac() * fac;
        else
          FOUR_C_THROW("Kedem-Katchalsky: Requested time integration scheme not yet implemented");

        // scalar at integration point
        const double phi = funct_.dot(ephinp[k]);

        // pressure at integration point
        const double p = funct_.dot(epressure);

        // mean concentration at integration point
        const double phibar_gp = funct_.dot(ephibar[k]);

        // mean concentration at integration point
        const double facWSS = ws_sinfluence(ewss, wss_onoff, coeffs);


        // matrix
        for (int vi = 0; vi < nen_; ++vi)
        {
          const double vlhs = facfac * facWSS * perm * funct_(vi);

          const int fvi = vi * numdofpernode_ + k;

          for (int ui = 0; ui < nen_; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            elemat1(fvi, fui) += vlhs * funct_(ui);
          }
        }

        // rhs
        const double vrhs =
            facfac * facWSS * (perm * phi + (1 - sigma) * phibar_gp * conductivity * p);
        // J_s =f_WSS*[perm*(phi1-phi2)+(1-sigma)*phibar*conductivity*(p1-p2)]
        // J_s := solute flux through scalar scalar interface
        // perm:=membrane permeability
        // sigma:=Staverman filtration coefficient
        // phibar_gp:= mean concentration within the membrane (for now: simply linear interpolated,
        // but other interpolations also possible) conductivity:=local hydraulic conductivity of
        // membrane

        for (int vi = 0; vi < nen_; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          elevec1[fvi] -= vrhs * funct_(vi);
        }
      }
    }  // if(onoff[k]==1)
    // else //in the case of "OFF", a no flux condition is automatically applied
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
double Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::ws_sinfluence(
    const Core::LinAlg::Matrix<nsd_, nen_>& ewss, const bool wss_onoff,
    const std::vector<double>* coeffs)
{
  // permeabilty scaling factor at integration point
  double facWSS = 1.0;

  if (wss_onoff)
  {
    Core::LinAlg::Matrix<nsd_, 1> wss(true);
    for (int ii = 0; ii < nsd_; ii++)
      for (int jj = 0; jj < nen_; jj++) wss(ii) += ewss(ii, jj) * funct_(jj);

    // euklidian norm of act node wss
    const double wss_norm = sqrt(wss(0) * wss(0) + wss(1) * wss(1) + wss(2) * wss(2));
    facWSS = log10(1 + coeffs->at(0) / (wss_norm + coeffs->at(1))) /
             log10(2);  // empirical function (log law) to account for influence of WSS;
  }
  // else //no WSS influence

  return facWSS;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::integrate_shape_functions(
    const Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& elevec1, const bool addarea)
{
  // access boundary area variable with its actual value
  double boundaryint = params.get<double>("area");

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    const double fac = eval_shape_func_and_int_fac(intpoints, gpid);

    // compute integral of shape functions
    for (int node = 0; node < nen_; ++node)
      for (int k = 0; k < numdofpernode_; ++k)
        elevec1[node * numdofpernode_ + k] += funct_(node) * fac;

    // area calculation
    if (addarea) boundaryint += fac;
  }  // loop over integration points

  // add contribution to the global value
  params.set<double>("area", boundaryint);
}  // Discret::Elements::ScaTraEleBoundaryCalc<distype>::integrate_shape_functions

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType bdistype, Core::FE::CellType pdistype>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::weak_dirichlet(
    Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::shared_ptr<const Core::Mat::Material> material,
    Core::LinAlg::SerialDenseMatrix& elemat_epetra, Core::LinAlg::SerialDenseVector& elevec_epetra)
{
  //------------------------------------------------------------------------
  // Dirichlet boundary condition
  //------------------------------------------------------------------------
  std::shared_ptr<Core::Conditions::Condition> dbc =
      params.get<std::shared_ptr<Core::Conditions::Condition>>("condition");

  // check of total time
  const double time = scatraparamstimint_->time();

  // get values and spatial functions from condition
  // (assumed to be constant on element boundary)
  const auto& val = (*dbc).parameters().get<std::vector<double>>("VAL");
  const auto& func = (*dbc).parameters().get<std::vector<Core::IO::Noneable<int>>>("FUNCT");

  // assign boundary value multiplied by time-curve factor
  double dirichval = val[0];

  //------------------------------------------------------------------------
  // preliminary definitions for (boundary) and parent element and
  // evaluation of nodal values of velocity and scalar based on parent
  // element nodes
  //------------------------------------------------------------------------
  // get the parent element
  Core::Elements::Element* pele = ele->parent_element();

  // number of spatial dimensions regarding (boundary) element
  static const int bnsd = Core::FE::dim<bdistype>;

  // number of spatial dimensions regarding parent element
  static const int pnsd = Core::FE::dim<pdistype>;

  // number of (boundary) element nodes
  static const int bnen = Core::FE::num_nodes<bdistype>;

  // number of parent element nodes
  static const int pnen = Core::FE::num_nodes<pdistype>;

  // parent element location array
  Core::Elements::LocationArray pla(discretization.num_dof_sets());
  pele->location_vector(discretization, pla, false);

  // get number of dofset associated with velocity related dofs
  const int ndsvel = scatraparams_->nds_vel();

  // get convective (velocity - mesh displacement) velocity at nodes
  std::shared_ptr<const Core::LinAlg::Vector<double>> convel =
      discretization.get_state(ndsvel, "convective velocity field");
  if (convel == nullptr) FOUR_C_THROW("Cannot get state vector convective velocity");

  // determine number of velocity related dofs per node
  const int numveldofpernode = pla[ndsvel].lm_.size() / pnen;

  // construct location vector for velocity related dofs
  std::vector<int> plmvel(pnsd * pnen, -1);
  for (int inode = 0; inode < pnen; ++inode)
    for (int idim = 0; idim < pnsd; ++idim)
      plmvel[inode * pnsd + idim] = pla[ndsvel].lm_[inode * numveldofpernode + idim];

  // we deal with a nsd_-dimensional flow field
  Core::LinAlg::Matrix<pnsd, pnen> econvel(true);

  // extract local values of convective velocity field from global state vector
  Core::FE::extract_my_values<Core::LinAlg::Matrix<pnsd, pnen>>(*convel, econvel, plmvel);

  // rotate the vector field in the case of rotationally symmetric boundary conditions
  rotsymmpbc_->template rotate_my_values_if_necessary<pnsd, pnen>(econvel);

  // get scalar values at parent element nodes
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
  if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");

  // extract local values from global vectors for parent element
  std::vector<Core::LinAlg::Matrix<pnen, 1>> ephinp(numscal_);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<pnen, 1>>(*phinp, ephinp, pla[0].lm_);

  //------------------------------------------------------------------------
  // preliminary definitions for integration loop
  //------------------------------------------------------------------------
  // reshape element matrices and vectors and init to zero, construct views
  elemat_epetra.shape(pnen, pnen);
  elevec_epetra.size(pnen);
  Core::LinAlg::Matrix<pnen, pnen> emat(elemat_epetra.values(), true);
  Core::LinAlg::Matrix<pnen, 1> erhs(elevec_epetra.values(), true);

  // (boundary) element local node coordinates
  Core::LinAlg::Matrix<pnsd, bnen> bxyze(true);
  Core::Geo::fill_initial_position_array<bdistype, pnsd, Core::LinAlg::Matrix<pnsd, bnen>>(
      ele, bxyze);

  // parent element local node coordinates
  Core::LinAlg::Matrix<pnsd, pnen> pxyze(true);
  Core::Geo::fill_initial_position_array<pdistype, pnsd, Core::LinAlg::Matrix<pnsd, pnen>>(
      pele, pxyze);

  // coordinates of integration points for (boundary) and parent element
  Core::LinAlg::Matrix<bnsd, 1> bxsi(true);
  Core::LinAlg::Matrix<pnsd, 1> pxsi(true);

  // transposed jacobian "dx/ds" and inverse of transposed jacobian "ds/dx"
  // for parent element
  Core::LinAlg::Matrix<pnsd, pnsd> pxjm(true);
  Core::LinAlg::Matrix<pnsd, pnsd> pxji(true);

  // metric tensor for (boundary) element
  Core::LinAlg::Matrix<bnsd, bnsd> bmetrictensor(true);

  // (outward-pointing) unit normal vector to (boundary) element
  Core::LinAlg::Matrix<pnsd, 1> bnormal(true);

  // velocity vector at integration point
  Core::LinAlg::Matrix<pnsd, 1> velint;

  // gradient of scalar value at integration point
  Core::LinAlg::Matrix<pnsd, 1> gradphi;

  // (boundary) element shape functions, local and global derivatives
  Core::LinAlg::Matrix<bnen, 1> bfunct(true);
  Core::LinAlg::Matrix<bnsd, bnen> bderiv(true);
  Core::LinAlg::Matrix<bnsd, bnen> bderxy(true);

  // parent element shape functions, local and global derivatives
  Core::LinAlg::Matrix<pnen, 1> pfunct(true);
  Core::LinAlg::Matrix<pnsd, pnen> pderiv(true);
  Core::LinAlg::Matrix<pnsd, pnen> pderxy(true);

  //------------------------------------------------------------------------
  // additional matrices and vectors for mixed-hybrid formulation
  //------------------------------------------------------------------------
  // for volume integrals
  Core::LinAlg::Matrix<pnsd * pnen, pnsd * pnen> mat_s_q(true);
  Core::LinAlg::Matrix<pnsd * pnen, pnen> mat_s_gradphi(true);

  Core::LinAlg::Matrix<pnsd * pnen, 1> vec_s_gradphi(true);

  // for boundary integrals
  Core::LinAlg::Matrix<pnen, pnsd * pnen> mat_w_q_o_n(true);
  Core::LinAlg::Matrix<pnsd * pnen, pnen> mat_s_o_n_phi(true);

  Core::LinAlg::Matrix<pnsd * pnen, 1> vec_s_o_n_phi_minus_g(true);

  // inverse matrix
  Core::LinAlg::Matrix<pnsd * pnen, pnsd * pnen> inv_s_q(true);

  //------------------------------------------------------------------------
  // check whether Nitsche (default) or mixed-hybrid formulation as well as
  // preliminary definitions and computations for Nitsche stabilization term
  //------------------------------------------------------------------------
  // default is Nitsche formulation
  bool mixhyb = false;

  // stabilization parameter for Nitsche term
  const auto nitsche_stab_para = dbc->parameters().get<double>("TauBscaling");

  // if stabilization parameter negative: mixed-hybrid formulation
  if (nitsche_stab_para < 0.0) mixhyb = true;

  // pre-factor for adjoint-consistency term:
  // either 1.0 (adjoint-consistent, default) or -1.0 (adjoint-inconsistent)
  double gamma = 1.0;
  const auto consistency = dbc->parameters().get<std::string>("GAMMATYPE");
  if (consistency == "adjoint-consistent")
    gamma = 1.0;
  else if (consistency == "diffusive-optimal")
    gamma = -1.0;
  else
    FOUR_C_THROW("unknown definition for gamma parameter: %s", consistency.c_str());

  // use one-point Gauss rule to do calculations at element center
  const Core::FE::IntPointsAndWeights<bnsd> intpoints_tau(
      ScaTra::DisTypeToStabGaussRule<bdistype>::rule);

  // element surface area (1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double* gpcoord = (intpoints_tau.ip().qxg)[0];
  for (int idim = 0; idim < bnsd; idim++)
  {
    bxsi(idim) = gpcoord[idim];
  }
  Core::FE::shape_function_deriv1<bdistype>(bxsi, bderiv);
  double drs = 0.0;
  Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
      bxyze, bderiv, bmetrictensor, drs, &bnormal);
  const double area = intpoints_tau.ip().qwgt[0] * drs;

  // get number of dimensions for (boundary) element (convert from int to double)
  const auto dim = (double)bnsd;

  // computation of characteristic length of (boundary) element
  // (2D: square root of element area, 1D: element length)
  const double h = std::pow(area, (1.0 / dim));

  //------------------------------------------------------------------------
  // preliminary computations for integration loop
  //------------------------------------------------------------------------
  // integration points and weights for (boundary) element and parent element
  const Core::FE::IntPointsAndWeights<bnsd> bintpoints(
      ScaTra::DisTypeToOptGaussRule<bdistype>::rule);

  const Core::FE::IntPointsAndWeights<pnsd> pintpoints(
      ScaTra::DisTypeToOptGaussRule<pdistype>::rule);

  // transfer integration-point coordinates of (boundary) element to parent element
  Core::LinAlg::SerialDenseMatrix pqxg(pintpoints.ip().nquad, pnsd);
  {
    Core::LinAlg::SerialDenseMatrix gps(bintpoints.ip().nquad, bnsd);

    for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
    {
      const double* gpcoord = (bintpoints.ip().qxg)[iquad];

      for (int idim = 0; idim < bnsd; idim++)
      {
        gps(iquad, idim) = gpcoord[idim];
      }
    }
    if (pnsd == 2)
    {
      Core::FE::boundary_gp_to_parent_gp2(pqxg, gps, pdistype, bdistype, ele->face_parent_number());
    }
    else if (pnsd == 3)
    {
      Core::FE::boundary_gp_to_parent_gp3(pqxg, gps, pdistype, bdistype, ele->face_parent_number());
    }
  }

  //------------------------------------------------------------------------
  // integration loop 1: volume integrals (only for mixed-hybrid formulation)
  //------------------------------------------------------------------------
  if (mixhyb)
  {
    for (int iquad = 0; iquad < pintpoints.ip().nquad; ++iquad)
    {
      // reference coordinates of integration point from (boundary) element
      const double* gpcoord = (pintpoints.ip().qxg)[iquad];
      for (int idim = 0; idim < pnsd; idim++)
      {
        pxsi(idim) = gpcoord[idim];
      }

      // parent element shape functions and local derivatives
      Core::FE::shape_function<pdistype>(pxsi, pfunct);
      Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

      // Jacobian matrix and determinant of parent element (including check)
      pxjm.multiply_nt(pderiv, pxyze);
      const double det = pxji.invert(pxjm);
      if (det < 1E-16)
        FOUR_C_THROW(
            "GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", pele->id(), det);

      // compute integration factor
      const double fac = pintpoints.ip().qwgt[iquad] * det;

      // compute global derivatives
      pderxy.multiply(pxji, pderiv);

      //--------------------------------------------------------------------
      // loop over scalars (not yet implemented for more than one scalar)
      //--------------------------------------------------------------------
      // for(int k=0;k<numdofpernode_;++k)
      int k = 0;
      {
        // get viscosity
        if (material->material_type() == Core::Materials::m_scatra)
        {
          const auto* actmat = static_cast<const Mat::ScatraMat*>(material.get());

          FOUR_C_ASSERT(numdofpernode_ == 1, "more than 1 dof per node for SCATRA material");

          // get constant diffusivity
          diffus_[k] = actmat->diffusivity();
        }
        else
          FOUR_C_THROW("Material type is not supported");

        // gradient of current scalar value
        gradphi.multiply(pderxy, ephinp[k]);

        // integration factor for left-hand side
        const double lhsfac = scatraparamstimint_->time_fac() * fac;

        // integration factor for right-hand side
        double rhsfac = 0.0;
        if (scatraparamstimint_->is_incremental() and scatraparamstimint_->is_gen_alpha())
          rhsfac = lhsfac / scatraparamstimint_->alpha_f();
        else if (not scatraparamstimint_->is_incremental() and scatraparamstimint_->is_gen_alpha())
          rhsfac = lhsfac * (1.0 - scatraparamstimint_->alpha_f()) / scatraparamstimint_->alpha_f();
        else if (scatraparamstimint_->is_incremental() and not scatraparamstimint_->is_gen_alpha())
          rhsfac = lhsfac;

        //--------------------------------------------------------------------
        //  matrix and vector additions due to mixed-hybrid formulation
        //--------------------------------------------------------------------
        /*
                       /         \
                  1   |   h   h  |
              - ----- |  s , q   |
                kappa |          |
                      \          / Omega
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          // const double vlhs = lhsfac*pfunct(vi);
          const double vlhs = lhsfac * (1.0 / diffus_[k]) * pfunct(vi);

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            for (int i = 0; i < pnsd; ++i)
            {
              mat_s_q(fvi * pnsd + i, fui * pnsd + i) -= vlhs * pfunct(ui);
            }
          }
        }

        /*
                       /                  \
                      |  h         /   h\  |
                    + | s  , grad | phi  | |
                      |            \    /  |
                       \                  / Omega
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          // const double vlhs = lhsfac*diffus_[k]*pfunct(vi);
          const double vlhs = lhsfac * pfunct(vi);

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            for (int i = 0; i < pnsd; ++i)
            {
              mat_s_gradphi(fvi * pnsd + i, fui) += vlhs * pderxy(i, ui);
            }
          }

          // const double vrhs = rhsfac*diffus_[k]*pfunct(vi);
          const double vrhs = rhsfac * pfunct(vi);

          for (int i = 0; i < pnsd; ++i)
          {
            vec_s_gradphi(fvi * pnsd + i) += vrhs * gradphi(i);
          }
        }
      }
    }
  }

  //------------------------------------------------------------------------
  // integration loop 2: boundary integrals
  //------------------------------------------------------------------------
  for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
  {
    // reference coordinates of integration point from (boundary) element
    const double* gpcoord = (bintpoints.ip().qxg)[iquad];
    for (int idim = 0; idim < bnsd; idim++)
    {
      bxsi(idim) = gpcoord[idim];
    }

    // (boundary) element shape functions
    Core::FE::shape_function<bdistype>(bxsi, bfunct);
    Core::FE::shape_function_deriv1<bdistype>(bxsi, bderiv);

    // global coordinates of current integration point from (boundary) element
    Core::LinAlg::Matrix<pnsd, 1> coordgp(true);
    for (int A = 0; A < bnen; ++A)
    {
      for (int j = 0; j < pnsd; ++j)
      {
        coordgp(j) += bxyze(j, A) * bfunct(A);
      }
    }

    // reference coordinates of integration point from parent element
    for (int idim = 0; idim < pnsd; idim++)
    {
      pxsi(idim) = pqxg(iquad, idim);
    }

    // parent element shape functions and local derivatives
    Core::FE::shape_function<pdistype>(pxsi, pfunct);
    Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

    // Jacobian matrix and determinant of parent element (including check)
    pxjm.multiply_nt(pderiv, pxyze);
    const double det = pxji.invert(pxjm);
    if (det < 1E-16)
      FOUR_C_THROW(
          "GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", pele->id(), det);

    // compute measure tensor for surface element, infinitesimal area element drs
    // and (outward-pointing) unit normal vector
    Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
        bxyze, bderiv, bmetrictensor, drs, &bnormal);

    // for nurbs elements the normal vector must be scaled with a special orientation factor!!
    if (Core::FE::is_nurbs<distype>) bnormal.scale(normalfac_);

    // compute integration factor
    const double fac = bintpoints.ip().qwgt[iquad] * drs;

    // compute global derivatives
    pderxy.multiply(pxji, pderiv);

    //--------------------------------------------------------------------
    // check whether integration-point coordinates evaluated from
    // (boundary) and parent element match
    //--------------------------------------------------------------------
    Core::LinAlg::Matrix<pnsd, 1> check(true);
    Core::LinAlg::Matrix<pnsd, 1> diff(true);

    for (int A = 0; A < pnen; ++A)
    {
      for (int j = 0; j < pnsd; ++j)
      {
        check(j) += pxyze(j, A) * pfunct(A);
      }
    }

    diff = check;
    diff -= coordgp;

    const double norm = diff.norm2();

    if (norm > 1e-9)
    {
      for (int j = 0; j < pnsd; ++j)
      {
        printf("%12.5e %12.5e\n", check(j), coordgp(j));
      }
      FOUR_C_THROW("Gausspoint matching error %12.5e\n", norm);
    }

    //--------------------------------------------------------------------
    // factor for Dirichlet boundary condition given by spatial function
    //--------------------------------------------------------------------
    double functfac = 1.0;
    if (func[0].has_value() && func[0].value() > 0)
    {
      // evaluate function at current integration point (important: a 3D position vector is
      // required)
      std::array<double, 3> coordgp3D;
      coordgp3D[0] = 0.0;
      coordgp3D[1] = 0.0;
      coordgp3D[2] = 0.0;
      for (int i = 0; i < pnsd; i++) coordgp3D[i] = coordgp(i);

      functfac = Global::Problem::instance()
                     ->function_by_id<Core::Utils::FunctionOfSpaceTime>(func[0].value() - 1)
                     .evaluate(coordgp3D.data(), time, 0);
    }
    else
      functfac = 1.0;
    dirichval *= functfac;

    //--------------------------------------------------------------------
    // loop over scalars (not yet implemented for more than one scalar)
    //--------------------------------------------------------------------
    // for(int k=0;k<numdofpernode_;++k)
    int k = 0;
    {
      // get viscosity
      if (material->material_type() == Core::Materials::m_scatra)
      {
        const auto* actmat = static_cast<const Mat::ScatraMat*>(material.get());

        FOUR_C_ASSERT(numdofpernode_ == 1, "more than 1 dof per node for SCATRA material");

        // get constant diffusivity
        diffus_[k] = actmat->diffusivity();
      }
      else
        FOUR_C_THROW("Material type is not supported");

      // get scalar value at integration point
      const double phi = pfunct.dot(ephinp[k]);

      // integration factor for left-hand side
      const double lhsfac = scatraparamstimint_->time_fac() * fac;

      // integration factor for right-hand side
      double rhsfac = 0.0;
      if (scatraparamstimint_->is_incremental() and scatraparamstimint_->is_gen_alpha())
        rhsfac = lhsfac / scatraparamstimint_->alpha_f();
      else if (not scatraparamstimint_->is_incremental() and scatraparamstimint_->is_gen_alpha())
        rhsfac = lhsfac * (1.0 - scatraparamstimint_->alpha_f()) / scatraparamstimint_->alpha_f();
      else if (scatraparamstimint_->is_incremental() and not scatraparamstimint_->is_gen_alpha())
        rhsfac = lhsfac;

      if (mixhyb)
      {
        //--------------------------------------------------------------------
        //  matrix and vector additions due to mixed-hybrid formulation
        //--------------------------------------------------------------------
        /*  consistency term
                    /           \
                   |  h   h     |
                 - | w , q  o n |
                   |            |
                   \            / Gamma
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          const double vlhs = lhsfac * pfunct(vi);

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            for (int i = 0; i < pnsd; ++i)
            {
              mat_w_q_o_n(fvi, fui * pnsd + i) -= vlhs * pfunct(ui) * bnormal(i);
            }
          }
        }

        /*  adjoint consistency term
                    /                 \
                   |  h          h    |
                 - | s  o n , phi - g |
                   |                  |
                   \                  / Gamma
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          const double vlhs = lhsfac * pfunct(vi);

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            for (int i = 0; i < pnsd; ++i)
            {
              mat_s_o_n_phi(fvi * pnsd + i, fui) -= vlhs * pfunct(ui) * bnormal(i);
            }
          }

          for (int i = 0; i < pnsd; ++i)
          {
            vec_s_o_n_phi_minus_g(fvi * pnsd + i) -=
                pfunct(vi) * bnormal(i) *
                (rhsfac * phi - scatraparamstimint_->time_fac() * fac * dirichval);
          }
        }
      }
      else
      {
        // parameter alpha for Nitsche stabilization term
        const double alpha = nitsche_stab_para * diffus_[k] / h;

        // get velocity at integration point
        velint.multiply(econvel, pfunct);

        // normal velocity
        const double normvel = velint.dot(bnormal);

        // gradient of current scalar value
        gradphi.multiply(pderxy, ephinp[k]);

        // gradient of current scalar value in normal direction
        const double gradphi_norm = bnormal.dot(gradphi);

        //--------------------------------------------------------------------
        //  matrix and vector additions due to Nitsche formulation
        //--------------------------------------------------------------------
        /*  consistency term
                    /                           \
                   |  h                  h      |
                 - | w , kappa * grad(phi ) o n |
                   |                            |
                   \                            / Gamma
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          const double vlhs = lhsfac * pfunct(vi) * diffus_[k];

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            for (int i = 0; i < pnsd; ++i)
            {
              emat(fvi, fui) -= vlhs * pderxy(i, ui) * bnormal(i);
            }
          }

          const double vrhs = rhsfac * diffus_[k];

          erhs(fvi) += vrhs * pfunct(vi) * gradphi_norm;
        }

        /*  adjoint consistency term, inflow/outflow part
              / --          --                                        \
             |  |         h  |                      h           h     |
           - |  |(a o n) w  +| gamma * kappa *grad(w ) o n , phi - g  |
             |  |            |                                        |
             \  --          --                                        / Gamma_in/out
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          // compute diffusive part
          double prefac = 0.0;
          for (int i = 0; i < pnsd; ++i)
          {
            prefac += gamma * diffus_[k] * pderxy(i, vi) * bnormal(i);
          }

          // add convective part in case of inflow boundary
          if (normvel < -0.0001) prefac += normvel * pfunct(vi);

          const double vlhs = lhsfac * prefac;

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            emat(fvi, fui) -= vlhs * pfunct(ui);
          }

          erhs(fvi) += prefac * (rhsfac * phi - scatraparamstimint_->time_fac() * fac * dirichval);
        }

        /*  stabilization term
                            /             \
                           |  h     h     |
                 + alpha * | w , phi - g  |
                           |              |
                           \              / Gamma
        */
        for (int vi = 0; vi < pnen; ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;

          const double prefac = alpha * pfunct(vi);

          for (int ui = 0; ui < pnen; ++ui)
          {
            const int fui = ui * numdofpernode_ + k;

            emat(fvi, fui) += lhsfac * prefac * pfunct(ui);
          }

          erhs(fvi) -= prefac * (rhsfac * phi - scatraparamstimint_->time_fac() * fac * dirichval);
        }
      }
    }
  }

  //------------------------------------------------------------------------
  // local condensation (only for mixed-hybrid formulation)
  //------------------------------------------------------------------------
  if (mixhyb)
  {
    // matrix inversion of flux-flux block
    inv_s_q = mat_s_q;

    Core::LinAlg::FixedSizeSerialDenseSolver<pnsd * pnen, pnsd * pnen> solver;

    solver.set_matrix(inv_s_q);
    solver.invert();

    // computation of matrix-matrix and matrix vector products, local assembly
    for (int vi = 0; vi < pnen; ++vi)
    {
      for (int ui = 0; ui < pnen; ++ui)
      {
        for (int rr = 0; rr < pnsd * pnen; ++rr)
        {
          for (int mm = 0; mm < pnsd * pnen; ++mm)
          {
            emat(vi, ui) -= mat_w_q_o_n(vi, rr) * inv_s_q(rr, mm) *
                            (mat_s_gradphi(mm, ui) + mat_s_o_n_phi(mm, ui));
          }
        }
      }
    }

    for (int vi = 0; vi < pnen; ++vi)
    {
      for (int rr = 0; rr < pnsd * pnen; ++rr)
      {
        for (int mm = 0; mm < pnsd * pnen; ++mm)
        {
          erhs(vi) -= mat_w_q_o_n(vi, rr) * inv_s_q(rr, mm) *
                      (-vec_s_o_n_phi_minus_g(mm) - vec_s_gradphi(mm));
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
template <Core::FE::CellType bdistype, Core::FE::CellType pdistype>
void Discret::Elements::ScaTraEleBoundaryCalc<distype,
    probdim>::reinit_characteristic_galerkin_boundary(Core::Elements::FaceElement*
                                                          ele,  //!< transport element
    Teuchos::ParameterList& params,                             //!< parameter list
    Core::FE::Discretization& discretization,                   //!< discretization
    const Core::Mat::Material& material,                        //!< material
    Core::LinAlg::SerialDenseMatrix& elemat_epetra,             //!< ele sysmat
    Core::LinAlg::SerialDenseVector& elevec_epetra              //!< ele rhs
)
{
  //------------------------------------------------------------------------
  // preliminary definitions for (boundary) and parent element and
  // evaluation of nodal values of velocity and scalar based on parent
  // element nodes
  //------------------------------------------------------------------------
  // get the parent element
  Core::Elements::Element* pele = ele->parent_element();

  // number of spatial dimensions regarding (boundary) element
  static const int bnsd = Core::FE::dim<bdistype>;

  // number of spatial dimensions regarding parent element
  static const int pnsd = Core::FE::dim<pdistype>;

  // number of (boundary) element nodes
  static const int bnen = Core::FE::num_nodes<bdistype>;

  // number of parent element nodes
  static const int pnen = Core::FE::num_nodes<pdistype>;

  // parent element lm vector
  std::vector<int> plm;
  std::vector<int> plmowner;
  std::vector<int> plmstride;
  pele->location_vector(discretization, plm, plmowner, plmstride);

  // get scalar values at parent element nodes
  std::shared_ptr<const Core::LinAlg::Vector<double>> phinp = discretization.get_state("phinp");
  if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phinp'");
  std::shared_ptr<const Core::LinAlg::Vector<double>> phin = discretization.get_state("phin");
  if (phinp == nullptr) FOUR_C_THROW("Cannot get state vector 'phin'");

  // extract local values from global vectors for parent element
  std::vector<Core::LinAlg::Matrix<pnen, 1>> ephinp(numscal_);
  std::vector<Core::LinAlg::Matrix<pnen, 1>> ephin(numscal_);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<pnen, 1>>(*phinp, ephinp, plm);
  Core::FE::extract_my_values<Core::LinAlg::Matrix<pnen, 1>>(*phin, ephin, plm);

  //------------------------------------------------------------------------
  // preliminary definitions for integration loop
  //------------------------------------------------------------------------
  // reshape element matrices and vectors and init to zero, construct views
  elemat_epetra.shape(pnen, pnen);
  elevec_epetra.size(pnen);
  Core::LinAlg::Matrix<pnen, pnen> emat(elemat_epetra.values(), true);
  Core::LinAlg::Matrix<pnen, 1> erhs(elevec_epetra.values(), true);

  // (boundary) element local node coordinates
  Core::LinAlg::Matrix<pnsd, bnen> bxyze(true);
  Core::Geo::fill_initial_position_array<bdistype, pnsd, Core::LinAlg::Matrix<pnsd, bnen>>(
      ele, bxyze);

  // parent element local node coordinates
  Core::LinAlg::Matrix<pnsd, pnen> pxyze(true);
  Core::Geo::fill_initial_position_array<pdistype, pnsd, Core::LinAlg::Matrix<pnsd, pnen>>(
      pele, pxyze);

  // coordinates of integration points for (boundary) and parent element
  Core::LinAlg::Matrix<bnsd, 1> bxsi(true);
  Core::LinAlg::Matrix<pnsd, 1> pxsi(true);

  // transposed jacobian "dx/ds" and inverse of transposed jacobian "ds/dx"
  // for parent element
  Core::LinAlg::Matrix<pnsd, pnsd> pxjm(true);
  Core::LinAlg::Matrix<pnsd, pnsd> pxji(true);

  // metric tensor for (boundary) element
  Core::LinAlg::Matrix<bnsd, bnsd> bmetrictensor(true);

  // (outward-pointing) unit normal vector to (boundary) element
  Core::LinAlg::Matrix<pnsd, 1> bnormal(true);

  // velocity vector at integration point
  Core::LinAlg::Matrix<pnsd, 1> velint;

  // gradient of scalar value at integration point
  Core::LinAlg::Matrix<pnsd, 1> gradphi;

  // (boundary) element shape functions, local and global derivatives
  Core::LinAlg::Matrix<bnen, 1> bfunct(true);
  Core::LinAlg::Matrix<bnsd, bnen> bderiv(true);
  Core::LinAlg::Matrix<bnsd, bnen> bderxy(true);

  // parent element shape functions, local and global derivatives
  Core::LinAlg::Matrix<pnen, 1> pfunct(true);
  Core::LinAlg::Matrix<pnsd, pnen> pderiv(true);
  Core::LinAlg::Matrix<pnsd, pnen> pderxy(true);


  // use one-point Gauss rule to do calculations at element center
  const Core::FE::IntPointsAndWeights<bnsd> intpoints_tau(
      ScaTra::DisTypeToStabGaussRule<bdistype>::rule);

  // element surface area (1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double* gpcoord = (intpoints_tau.IP().qxg)[0];
  for (int idim = 0; idim < bnsd; idim++)
  {
    bxsi(idim) = gpcoord[idim];
  }
  Core::FE::shape_function_deriv1<bdistype>(bxsi, bderiv);
  double drs = 0.0;
  Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
      bxyze, bderiv, bmetrictensor, drs, &bnormal);

  //------------------------------------------------------------------------
  // preliminary computations for integration loop
  //------------------------------------------------------------------------
  // integration points and weights for (boundary) element and parent element
  const Core::FE::IntPointsAndWeights<bnsd> bintpoints(
      ScaTra::DisTypeToOptGaussRule<bdistype>::rule);

  const Core::FE::IntPointsAndWeights<pnsd> pintpoints(
      ScaTra::DisTypeToOptGaussRule<pdistype>::rule);

  // transfer integration-point coordinates of (boundary) element to parent element
  Core::LinAlg::SerialDenseMatrix pqxg(pintpoints.ip().nquad, pnsd);
  {
    Core::LinAlg::SerialDenseMatrix gps(bintpoints.ip().nquad, bnsd);

    for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
    {
      const double* gpcoord = (bintpoints.ip().qxg)[iquad];

      for (int idim = 0; idim < bnsd; idim++)
      {
        gps(iquad, idim) = gpcoord[idim];
      }
    }
    if (pnsd == 2)
    {
      Core::FE::boundary_gp_to_parent_gp2(pqxg, gps, pdistype, bdistype, ele->face_parent_number());
    }
    else if (pnsd == 3)
    {
      Core::FE::boundary_gp_to_parent_gp3(pqxg, gps, pdistype, bdistype, ele->face_parent_number());
    }
  }


  const double reinit_pseudo_timestepsize_factor = params.get<double>("pseudotimestepsize_factor");

  const double meshsize = get_ele_diameter<pdistype>(pxyze);

  const double pseudo_timestep_size = meshsize * reinit_pseudo_timestepsize_factor;

  //------------------------------------------------------------------------
  // integration loop: boundary integrals
  //------------------------------------------------------------------------
  for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
  {
    // reference coordinates of integration point from (boundary) element
    const double* gpcoord = (bintpoints.ip().qxg)[iquad];
    for (int idim = 0; idim < bnsd; idim++)
    {
      bxsi(idim) = gpcoord[idim];
    }

    // (boundary) element shape functions
    Core::FE::shape_function<bdistype>(bxsi, bfunct);
    Core::FE::shape_function_deriv1<bdistype>(bxsi, bderiv);

    // global coordinates of current integration point from (boundary) element
    Core::LinAlg::Matrix<pnsd, 1> coordgp(true);
    for (int A = 0; A < bnen; ++A)
    {
      for (int j = 0; j < pnsd; ++j)
      {
        coordgp(j) += bxyze(j, A) * bfunct(A);
      }
    }

    // reference coordinates of integration point from parent element
    for (int idim = 0; idim < pnsd; idim++)
    {
      pxsi(idim) = pqxg(iquad, idim);
    }

    // parent element shape functions and local derivatives
    Core::FE::shape_function<pdistype>(pxsi, pfunct);
    Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

    // Jacobian matrix and determinant of parent element (including check)
    pxjm.multiply_nt(pderiv, pxyze);
    const double det = pxji.invert(pxjm);
    if (det < 1E-16)
      FOUR_C_THROW(
          "GLOBAL ELEMENT NO.%i\nZERO OR NEGATIVE JACOBIAN DETERMINANT: %f", pele->id(), det);

    // compute measure tensor for surface element, infinitesimal area element drs
    // and (outward-pointing) unit normal vector
    Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
        bxyze, bderiv, bmetrictensor, drs, &bnormal);

    // for nurbs elements the normal vector must be scaled with a special orientation factor!!
    if (Core::FE::is_nurbs<distype>) bnormal.scale(normalfac_);

    // compute integration factor
    const double fac_surface = bintpoints.ip().qwgt[iquad] * drs;

    // compute global derivatives
    pderxy.multiply(pxji, pderiv);

    //--------------------------------------------------------------------
    // loop over scalars (not yet implemented for more than one scalar)
    //--------------------------------------------------------------------
    for (int dofindex = 0; dofindex < numdofpernode_; ++dofindex)
    {
      //----------  --------------      |                    |
      //  mat              -1/4* dtau^2 | w, n*grad(D(psi) ) |
      //--------------------------      |                    |

      Core::LinAlg::Matrix<1, pnen> derxy_normal;
      derxy_normal.clear();
      derxy_normal.multiply_tn(bnormal, pderxy);

      for (int vi = 0; vi < pnen; ++vi)
      {
        const int fvi = vi * numdofpernode_ + dofindex;

        for (int ui = 0; ui < pnen; ++ui)
        {
          const int fui = ui * numdofpernode_ + dofindex;

          emat(fvi, fui) -= pfunct(vi) *
                            (fac_surface * pseudo_timestep_size * pseudo_timestep_size / 4.0) *
                            derxy_normal(0, ui);
        }
      }

      //----------  --------------      |              m     |
      //  rhs               0.5* dtau^2 | w, n*grad(psi )    |
      //--------------------------      |                    |

      // update grad_dist_n
      Core::LinAlg::Matrix<pnsd, 1> grad_dist_n(true);
      grad_dist_n.multiply(pderxy, ephin[dofindex]);

      Core::LinAlg::Matrix<1, 1> grad_dist_n_normal(true);
      grad_dist_n_normal.multiply_tn(bnormal, grad_dist_n);

      for (int vi = 0; vi < pnen; ++vi)
      {
        const int fvi = vi * numdofpernode_ + dofindex;

        erhs(fvi) += pfunct(vi) * pseudo_timestep_size * pseudo_timestep_size * fac_surface / 2.0 *
                     grad_dist_n_normal(0, 0);
      }


      //                    |              m+1     m  |
      //    1/4*delta_tau^2 | w, n*grad(psi   - psi ) |
      //                    |              i          |
      // update grad_dist_n
      Core::LinAlg::Matrix<pnsd, 1> grad_dist_npi(true);
      grad_dist_npi.multiply(pderxy, ephinp[dofindex]);

      Core::LinAlg::Matrix<1, 1> grad_dist_npi_normal;
      grad_dist_npi_normal.clear();
      grad_dist_npi_normal.multiply_tn(bnormal, grad_dist_npi);

      double Grad_Dpsi_normal = grad_dist_npi_normal(0, 0) - grad_dist_n_normal(0, 0);


      for (int vi = 0; vi < pnen; ++vi)
      {
        const int fvi = vi * numdofpernode_ + dofindex;

        erhs(fvi) += pfunct(vi) * Grad_Dpsi_normal * fac_surface * pseudo_timestep_size *
                     pseudo_timestep_size / 4.0;
      }

    }  // loop over scalars
  }  // loop over integration points
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::evaluate_nodal_size(
    const Core::Elements::FaceElement* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseVector& nodalsize)
{
  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac =
        Discret::Elements::ScaTraEleBoundaryCalc<distype, probdim>::eval_shape_func_and_int_fac(
            intpoints, gpid);
    for (int vi = 0; vi < nen_; ++vi)
    {
      nodalsize[numdofpernode_ * vi] += funct_(vi, 0) * fac;
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// explicit instantiation of template methods
template void Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::tri3>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::tri3>(
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&, const int,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseVector&, Core::LinAlg::SerialDenseVector&);
template void Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::tri3>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::quad4>(
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&, const int,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseVector&, Core::LinAlg::SerialDenseVector&);
template void Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::quad4>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::tri3>(
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tri3>, 1>&, const int,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseVector&, Core::LinAlg::SerialDenseVector&);
template void Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::quad4>::
    evaluate_s2_i_coupling_at_integration_point<Core::FE::CellType::quad4>(
        const std::vector<Core::LinAlg::Matrix<nen_, 1>>&,
        const std::vector<Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>>&,
        const double, const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&,
        const Core::LinAlg::Matrix<nen_, 1>&,
        const Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::quad4>, 1>&, const int,
        const Discret::Elements::ScaTraEleParameterBoundary* const, const double, const double,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseMatrix&,
        Core::LinAlg::SerialDenseVector&, Core::LinAlg::SerialDenseVector&);

// template classes
template class Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::quad4, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::quad8, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::quad9, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::tri6, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::line3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::nurbs3, 2>;
template class Discret::Elements::ScaTraEleBoundaryCalc<Core::FE::CellType::nurbs9, 3>;

FOUR_C_NAMESPACE_CLOSE
