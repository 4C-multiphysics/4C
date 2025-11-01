// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_timint_loma.hpp"

#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_turbulence_statistic_manager.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_sutherland.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  Constructor (public)                                       bk 11/13 |
 *----------------------------------------------------------------------*/
FLD::TimIntLoma::TimIntLoma(const std::shared_ptr<Core::FE::Discretization>& actdis,
    const std::shared_ptr<Core::LinAlg::Solver>& solver,
    const std::shared_ptr<Teuchos::ParameterList>& params,
    const std::shared_ptr<Core::IO::DiscretizationWriter>& output, bool alefluid /*= false*/)
    : FluidImplicitTimeInt(actdis, solver, params, output, alefluid),
      thermpressaf_(1.0),
      thermpressam_(1.0),
      thermpressdtaf_(0.0),
      thermpressdtam_(0.0)
{
  return;
}


/*----------------------------------------------------------------------*
 |  initialize algorithm                                rasthofer 04/14 |
 *----------------------------------------------------------------------*/
void FLD::TimIntLoma::init()
{
  // conservative formulation currently not supported in low-Mach-number case
  // when using generalized-alpha time-integration scheme
  if (convform_ == "conservative")
    FOUR_C_THROW(
        "conservative formulation currently not supported for low-Mach-number flow within "
        "generalized-alpha time-integration scheme");

  // ---------------------------------------------------------------------
  // set density variable to 1.0 and get gas constant for low-Mach-number
  // flow and get constant density variable for incompressible flow
  // ---------------------------------------------------------------------

  // get gas constant
  int id =
      Global::Problem::instance()->materials()->first_id_by_type(Core::Materials::m_sutherland);
  if (id == -1)
    FOUR_C_THROW("Could not find sutherland material");
  else
  {
    const Core::Mat::PAR::Parameter* mat =
        Global::Problem::instance()->materials()->parameter_by_id(id);
    const Mat::PAR::Sutherland* actmat = static_cast<const Mat::PAR::Sutherland*>(mat);
    // we need the kinematic viscosity here
    gasconstant_ = actmat->gasconst_;
  }

  // potential check here -> currently not executed
  // if (gasconstant_ < 1e-15) FOUR_C_THROW("received zero or negative gas constant");

  // set some Loma-specific parameters
  set_element_custom_parameter();
  return;
}



/*----------------------------------------------------------------------*
 | set fields for scatra - fluid coupling, esp.                         |
 | set fields for low-Mach-number flow within iteration loop   vg 09/09 |
 *----------------------------------------------------------------------*/
void FLD::TimIntLoma::set_loma_iter_scalar_fields(
    std::shared_ptr<const Core::LinAlg::Vector<double>> scalaraf,
    std::shared_ptr<const Core::LinAlg::Vector<double>> scalaram,
    std::shared_ptr<const Core::LinAlg::Vector<double>> scalardtam,
    std::shared_ptr<const Core::LinAlg::Vector<double>> fsscalaraf, const double thermpressaf,
    const double thermpressam, const double thermpressdtaf, const double thermpressdtam,
    std::shared_ptr<Core::FE::Discretization> scatradis)
{
  // initializations
  double value(0.0);
  std::vector<int> nodedofs;

  //--------------------------------------------------------------------------
  // Filling the scaaf-vector and scaam-vector at time n+alpha_F/n+1 and
  // n+alpha_M/n, respectively, with scalar at pressure dofs
  // Additionally, filling the scaam-vector at time n+alpha_M/n with
  // velocity at time n at velocity dofs for OST/BDF2
  // Filling the accam-vector at time n+alpha_M/n+1, respectively, with
  // scalar time derivative values at pressure dofs
  //--------------------------------------------------------------------------
  // get velocity values at time n in scaam-vector as copy from veln-vector
  scaam_->update(1.0, *veln_, 0.0);

  // loop all nodes on the processor
  for (int lnodeid = 0; lnodeid < discret_->num_my_row_nodes(); lnodeid++)
  {
    // get the processor's local scatra node
    Core::Nodes::Node* lscatranode = scatradis->l_row_node(lnodeid);

    // find out the global dof id of the last(!) dof at the scatra node
    const int numscatradof = scatradis->num_dof(0, lscatranode);
    const int globalscatradofid = scatradis->dof(0, lscatranode, numscatradof - 1);
    const int localscatradofid = scalaraf->get_map().lid(globalscatradofid);
    if (localscatradofid < 0) FOUR_C_THROW("localdofid not found in map for given globaldofid");

    // get the processor's local fluid node
    Core::Nodes::Node* lnode = discret_->l_row_node(lnodeid);
    // get the global ids of degrees of freedom associated with this node
    nodedofs = discret_->dof(0, lnode);
    // get global and processor's local pressure dof id (using the map!)
    const int numdof = discret_->num_dof(0, lnode);
    const int globaldofid = discret_->dof(0, lnode, numdof - 1);
    const int localdofid = scaam_->get_map().lid(globaldofid);
    if (localdofid < 0) FOUR_C_THROW("localdofid not found in map for given globaldofid");

    // now copy the values
    value = scalaraf->local_values_as_span()[localscatradofid];
    scaaf_->replace_local_value(localdofid, value);

    value = scalaram->local_values_as_span()[localscatradofid];
    scaam_->replace_local_value(localdofid, value);

    if (scalardtam != nullptr)
    {
      value = scalardtam->local_values_as_span()[localscatradofid];
    }
    else
    {
      value = 0.0;  // for safety reasons: set zeros in accam_
    }
    accam_->replace_local_value(localdofid, value);

    if (turbmodel_ == Inpar::FLUID::multifractal_subgrid_scales)
    {
      if (fsscalaraf != nullptr)
        value = fsscalaraf->local_values_as_span()[localscatradofid];
      else
        FOUR_C_THROW("Expected fine-scale scalar!");

      fsscaaf_->replace_local_value(localdofid, value);
    }
  }

  //--------------------------------------------------------------------------
  // get thermodynamic pressure at n+alpha_F/n+1 and n+alpha_M/n and
  // time derivative of thermodyn. press. at n+alpha_F/n+1 and n+alpha_M/n+1
  //--------------------------------------------------------------------------
  thermpressaf_ = thermpressaf;
  thermpressam_ = thermpressam;
  thermpressdtaf_ = thermpressdtaf;
  thermpressdtam_ = thermpressdtam;

  return;
}  // TimIntLoma::set_loma_iter_scalar_fields


/*----------------------------------------------------------------------*
 | set scalar fields     vg 09/09 |
 *----------------------------------------------------------------------*/
void FLD::TimIntLoma::set_scalar_fields(
    std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp, const double thermpressnp,
    std::shared_ptr<const Core::LinAlg::Vector<double>> scatraresidual,
    std::shared_ptr<Core::FE::Discretization> scatradis, const int whichscalar)
{
  FluidImplicitTimeInt::set_scalar_fields(
      scalarnp, thermpressnp, scatraresidual, scatradis, whichscalar);
  //--------------------------------------------------------------------------
  // get thermodynamic pressure at n+1
  //--------------------------------------------------------------------------
  thermpressaf_ = thermpressnp;


  return;

}  // TimIntLoma::SetScalarFields

// -------------------------------------------------------------------
// set loma parameters                               rasthofer 03/2012
// -------------------------------------------------------------------
void FLD::TimIntLoma::set_element_custom_parameter()
{
  Teuchos::ParameterList eleparams;

  // set parameters to update material with subgrid-scale temperature
  // potential inclusion of additional subgrid-scale terms in continuity equation
  eleparams.sublist("LOMA") = params_->sublist("LOMA");
  eleparams.sublist("RESIDUAL-BASED STABILIZATION") =
      params_->sublist("RESIDUAL-BASED STABILIZATION");
  eleparams.sublist("MULTIFRACTAL SUBGRID SCALES") =
      params_->sublist("MULTIFRACTAL SUBGRID SCALES");

  Discret::Elements::FluidEleParameterStd::instance()->set_element_loma_parameter(eleparams);
}

/*----------------------------------------------------------------------*
| print info about turbulence model (loma-specific)            bk 11/13 |
*----------------------------------------------------------------------*/
void FLD::TimIntLoma::print_turbulence_model()
{
  FluidImplicitTimeInt::print_turbulence_model();

  if (physicaltype_ == Inpar::FLUID::loma and turbmodel_ == Inpar::FLUID::smagorinsky)
  {
    if (params_->sublist("SUBGRID VISCOSITY").get<bool>("C_INCLUDE_CI"))
    {
      if (params_->sublist("SUBGRID VISCOSITY").get<double>("C_YOSHIZAWA") > 0.0)
      {
        std::cout << "with Yoshizawa constant Ci= ";
        std::cout << params_->sublist("SUBGRID VISCOSITY").get<double>("C_YOSHIZAWA") << "\n";
      }
      else
        FOUR_C_THROW("Ci expected!");
    }
    else
      std::cout << "Yoshizawa constant Ci not included";

    std::cout << &std::endl;
  }
  return;
}

/*----------------------------------------------------------------------*
| set custom ele params in assemble_mat_and_rhs                   bk 11/13 |
*----------------------------------------------------------------------*/
void FLD::TimIntLoma::set_custom_ele_params_assemble_mat_and_rhs(Teuchos::ParameterList& eleparams)
{
  eleparams.set("thermpress at n+alpha_F/n+1", thermpressaf_);
  eleparams.set("thermpress at n+alpha_M/n", thermpressam_);
  eleparams.set("thermpressderiv at n+alpha_F/n+1", thermpressdtaf_);
  eleparams.set("thermpressderiv at n+alpha_M/n+1", thermpressdtam_);
  return;
}

/*----------------------------------------------------------------------*
| set custom ele params in apply_nonlinear_boundary_conditions    bk 11/13 |
*----------------------------------------------------------------------*/
void FLD::TimIntLoma::set_custom_ele_params_apply_nonlinear_boundary_conditions(
    Teuchos::ParameterList& eleparams)
{
  eleparams.set("thermpress at n+alpha_F/n+1", thermpressaf_);
  return;
}

/*----------------------------------------------------------------------*
| set custom ele params in linear_relaxation_solve               bk 11/13 |
*----------------------------------------------------------------------*/
void FLD::TimIntLoma::set_custom_ele_params_linear_relaxation_solve(
    Teuchos::ParameterList& eleparams)
{
  eleparams.set("thermpress at n+alpha_F/n+1", thermpressaf_);
  eleparams.set("thermpress at n+alpha_M/n", thermpressam_);
  eleparams.set("thermpressderiv at n+alpha_F/n+1", thermpressdtaf_);
  eleparams.set("thermpressderiv at n+alpha_M/n+1", thermpressdtam_);
  return;
}

/*----------------------------------------------------------------------*
| call the statistics manager including thermpress parameters  bk 11/13 |
*----------------------------------------------------------------------*/
void FLD::TimIntLoma::call_statistics_manager()
{
  // -------------------------------------------------------------------
  //   add calculated velocity to mean value calculation (statistics)
  // -------------------------------------------------------------------
  // compute equation-of-state factor
  const double eosfac = thermpressaf_ / gasconstant_;
  statisticsmanager_->do_time_sample(
      step_, eosfac, thermpressaf_, thermpressam_, thermpressdtaf_, thermpressdtam_);
  return;
}

/*----------------------------------------------------------------------*
 | prepare AVM3-based scale separation                         vg 10/08 |
 | overloaded in TimIntRedModels and TimIntLoma               bk 12/13 |
 *----------------------------------------------------------------------*/
void FLD::TimIntLoma::avm3_preparation()
{
  // time measurement: avm3
  TEUCHOS_FUNC_TIME_MONITOR("           + avm3");

  // create the parameters for the discretization
  Teuchos::ParameterList eleparams;

  // necessary here, because some application time integrations add something to the residual
  // before the Neumann loads are added
  residual_->put_scalar(0.0);

  eleparams.set("thermpress at n+alpha_F/n+1", thermpressaf_);
  eleparams.set("thermpress at n+alpha_M/n", thermpressam_);
  eleparams.set("thermpressderiv at n+alpha_F/n+1", thermpressdtaf_);
  eleparams.set("thermpressderiv at n+alpha_M/n+1", thermpressdtam_);

  avm3_assemble_mat_and_rhs(eleparams);

  // get scale-separation matrix
  avm3_get_scale_separation_matrix();

  // perform initial separation to initialize fsvelaf_
  // required for loma
  if (physicaltype_ == Inpar::FLUID::loma)
  {
    update_velaf_gen_alpha();
    sep_multiply();
  }

  return;
}  // TimIntLoma::avm3_preparation

FOUR_C_NAMESPACE_CLOSE
