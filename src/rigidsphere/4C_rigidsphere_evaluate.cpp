// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_browniandyn_input.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_geometric_search_bounding_volume.hpp"
#include "4C_geometric_search_params.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_stvenantkirchhoff.hpp"
#include "4C_rigidsphere.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------------------*
 |  evaluate the element (public) meier 02/14|
 *----------------------------------------------------------------------------------------------------------*/
int Discret::Elements::Rigidsphere::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  set_params_interface_ptr(params);

  // start with "none"
  Core::Elements::ActionType act = params_interface().get_action_type();

  switch (act)
  {
    case Core::Elements::struct_calc_linstiff:
    case Core::Elements::struct_calc_nlnstiff:
    case Core::Elements::struct_calc_internalforce:
    case Core::Elements::struct_calc_linstiffmass:
    case Core::Elements::struct_calc_nlnstiffmass:
    case Core::Elements::struct_calc_nlnstifflmass:
    case Core::Elements::struct_calc_internalinertiaforce:
    {
      // need current global displacement and residual forces and get them from discretization
      // making use of the local-to-global map lm one can extract current displacement and residual
      // values for each degree of freedom

      // get element displacements
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

      std::shared_ptr<const Core::LinAlg::Vector<double>> vel;
      std::vector<double> myvel(lm.size());
      myvel.clear();

      // get element acceleration
      std::vector<double> myacc(lm.size());
      myacc.clear();

      if (act == Core::Elements::struct_calc_nlnstiffmass or
          act == Core::Elements::struct_calc_nlnstifflmass or
          act == Core::Elements::struct_calc_linstiffmass)
      {
        nlnstiffmass(params, myacc, myvel, mydisp, &elemat1, &elemat2, &elevec1, &elevec2);
      }
      else if (act == Core::Elements::struct_calc_linstiff or
               act == Core::Elements::struct_calc_nlnstiff)
      {
        nlnstiffmass(params, myacc, myvel, mydisp, &elemat1, nullptr, &elevec1, nullptr);
      }
      else if (act == Core::Elements::struct_calc_internalforce)
      {
        nlnstiffmass(params, myacc, myvel, mydisp, nullptr, nullptr, &elevec1, nullptr);
      }
      else if (act == Core::Elements::struct_calc_internalinertiaforce)
      {
        nlnstiffmass(params, myacc, myvel, mydisp, nullptr, nullptr, &elevec1, &elevec2);
      }
    }
    break;

    case Core::Elements::struct_calc_brownianforce:
    case Core::Elements::struct_calc_brownianstiff:
    {
      // get element displacements
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

      // get element velocity
      std::shared_ptr<const Core::LinAlg::Vector<double>> vel =
          discretization.get_state("velocity");
      if (vel == nullptr) FOUR_C_THROW("Cannot get state vectors 'velocity'");
      std::vector<double> myvel = Core::FE::extract_values(*vel, lm);

      if (act == Core::Elements::struct_calc_brownianforce)
        calc_brownian_forces_and_stiff(params, myvel, mydisp, nullptr, &elevec1);
      else if (act == Core::Elements::struct_calc_brownianstiff)
        calc_brownian_forces_and_stiff(params, myvel, mydisp, &elemat1, &elevec1);
      else
        FOUR_C_THROW("You shouldn't be here.");

      break;
    }

    case Core::Elements::struct_calc_stress:
    {
      FOUR_C_THROW("No stress output implemented for beam3 elements");
      break;
    }
    case Core::Elements::struct_calc_update_istep:
    case Core::Elements::struct_calc_reset_istep:
    case Core::Elements::struct_calc_recover:
    {
      // not necessary since no class variables are modified in predicting steps
      break;
    }

    case Core::Elements::struct_calc_predict:
    {
      // do nothing here
      break;
    }

    // element based PTC scaling
    case Core::Elements::struct_calc_addjacPTC:
    {
      // nothing to do here
      break;
    }

    case Core::Elements::struct_calc_energy:
    {
      // no contribution of rigid sphere to system energy
      break;
    }

    default:
    {
      FOUR_C_THROW("Unknown type of action for Rigidsphere {}", act);
      break;
    }
  }

  return (0);
}

/*------------------------------------------------------------------------------------------------------------*
 | nonlinear stiffness and mass matrix (private) meier 05/12|
 *-----------------------------------------------------------------------------------------------------------*/
void Discret::Elements::Rigidsphere::nlnstiffmass(Teuchos::ParameterList& params,
    std::vector<double>& acc, std::vector<double>& vel, std::vector<double>& disp,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseMatrix* massmatrix,
    Core::LinAlg::SerialDenseVector* force, Core::LinAlg::SerialDenseVector* inertia_force)
{
  // assemble internal force vector if requested
  if (force != nullptr)
  {
    for (int i = 0; i < 3; ++i) (*force)(i) = 0.0;
  }

  // assemble stiffmatrix if requested
  if (stiffmatrix != nullptr)
  {
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) (*stiffmatrix)(i, j) = 0.0;
  }

  // assemble massmatrix if requested
  if (massmatrix != nullptr)
  {
    double m = rho_ * 4.0 / 3.0 * std::numbers::pi * radius_ * radius_ * radius_;
    for (int i = 0; i < 3; ++i) (*massmatrix)(i, i) = m;
  }

  //    //assemble inertia force vector if requested
  //    if ( inertia_force != nullptr and massmatrix != nullptr )
  //    {
  //      for ( int i = 0; i < 3; ++i )
  //        (*inertia_force)(i) = acc[i] * (*massmatrix)(i,i);
  //    }

  return;
}

/*------------------------------------------------------------------------------------------------------------*
 | calculation of thermal (i.e. stochastic) and damping forces according to Brownian dynamics grill
 03/14|
 *------------------------------------------------------------------------------------------------------------*/
void Discret::Elements::Rigidsphere::calc_brownian_forces_and_stiff(Teuchos::ParameterList& params,
    std::vector<double>& vel, std::vector<double>& disp,
    Core::LinAlg::SerialDenseMatrix* stiffmatrix, Core::LinAlg::SerialDenseVector* force)
{
  calc_drag_force(params, vel, disp, stiffmatrix, force);
  calc_stochastic_force(params, vel, disp, stiffmatrix, force);
}

/*------------------------------------------------------------------------------------------------------------*
 | compute drag forces and contribution to stiffness matrix  (private) grill 03/14|
 *-----------------------------------------------------------------------------------------------------------*/
void Discret::Elements::Rigidsphere::calc_drag_force(Teuchos::ParameterList& params,
    const std::vector<double>& vel,                //!< element velocity vector
    const std::vector<double>& disp,               //!< element displacement vector
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
    Core::LinAlg::SerialDenseVector* force)        //!< element internal force vector
{
  double gamma = my_damping_constant();

  // get time step size
  double dt = params_interface().get_delta_time();

  // velocity and gradient of background velocity field
  Core::LinAlg::Matrix<3, 1> velbackground;
  Core::LinAlg::Matrix<3, 3> velbackgroundgrad;  // is a dummy so far

  // Compute background velocity
  get_background_velocity(params, velbackground, velbackgroundgrad);

  // Drag force contribution
  if (force != nullptr)
    for (int i = 0; i < 3; ++i) (*force)(i) += gamma * (vel[i] - velbackground(i));


  // contribution to stiffness matrix
  // depends on TIME INTEGRATION SCHEME (so far, damping is allowed for StatMech only => Backward
  // Euler) GenAlpha would require scaling with gamma_genalpha/beta_genalpha
  if (stiffmatrix != nullptr)
  {
    // StatMech: Backward Euler
    for (int l = 0; l < 3; l++)
    {
      (*stiffmatrix)(l, l) += gamma / dt;
    }
  }

  return;
}

/*-----------------------------------------------------------------------------------------------------------*
 |computes velocity of background fluid and gradient of that velocity at a certain evaluation point
 in       | |the physical space                                                         (public)
 grill   03/14|
 *----------------------------------------------------------------------------------------------------------*/
void Discret::Elements::Rigidsphere::get_background_velocity(
    Teuchos::ParameterList& params,                 //!< parameter list
    Core::LinAlg::Matrix<3, 1>& velbackground,      //!< velocity of background fluid
    Core::LinAlg::Matrix<3, 3>& velbackgroundgrad)  //!< gradient of velocity of background fluid
{
  // only constant background velocity implemented yet. for case of shear flow, see beam3r


  // default values for background velocity and its gradient
  velbackground.put_scalar(0);
  velbackgroundgrad.put_scalar(0);

  //  double time = params.get<double>("total time",0.0);
  //  double starttime = params.get<double>("STARTTIMEACT",0.0);
  //  double dt = params.get<double>("delta time");
  //
  //  std::shared_ptr<std::vector<double> > defvalues = Teuchos::rcp(new
  //  std::vector<double>(3,0.0)); std::shared_ptr<std::vector<double> > periodlength =
  //  params.get("PERIODLENGTH", defvalues);
  //
  //  // check and throw error if shear flow is applied
  //  Inpar::STATMECH::DBCType dbctype = params.get<Inpar::STATMECH::DBCType>("DBCTYPE",
  //  Inpar::STATMECH::dbctype_std); bool shearflow = false;
  //  if(dbctype==Inpar::STATMECH::dbctype_shearfixed ||
  //     dbctype==Inpar::STATMECH::dbctype_shearfixeddel ||
  //     dbctype==Inpar::STATMECH::dbctype_sheartrans ||
  //     dbctype==Inpar::STATMECH::dbctype_affineshear||
  //     dbctype==Inpar::STATMECH::dbctype_affinesheardel)
  //  {
  //    shearflow = true;
  //    FOUR_C_THROW("Shear flow not implemented yet for rigid spherical particles!");
  //  }
  //
  //  // constant background velocity specified in input file?
  //  std::shared_ptr<std::vector<double> > constbackgroundvel = params.get("CONSTBACKGROUNDVEL",
  //  defvalues);
  //
  //  if (constbackgroundvel->size() != 3) FOUR_C_THROW("\nSpecified vector for constant background
  //  velocity has wrong dimension! Check input file!"); bool constflow = false; for (int i=0; i<3;
  //  ++i)
  //  {
  //    if (constbackgroundvel->at(i)!=0.0) constflow=true;
  //  }
  //
  //  if(periodlength->at(0) > 0.0)
  //  {
  //    if(constflow && time>starttime && fabs(time-starttime)>dt/1e4)
  //    {
  //      for (int i=0; i<3; ++i) velbackground(i) = constbackgroundvel->at(i);
  //
  //      // shear flow AND constant background flow not implemented
  //      if(shearflow) FOUR_C_THROW("Conflict in input parameters: shearflow AND constant
  //      background velocity specified. Not implemented!\n");
  //    }
  //  }
}

/*-----------------------------------------------------------------------------------------------------------*
 | computes damping coefficient                                             (private) grill   03/14|
 *----------------------------------------------------------------------------------------------------------*/
double Discret::Elements::Rigidsphere::my_damping_constant()
{
  // (dynamic) viscosity of background fluid
  double eta = params_interface().get_brownian_dyn_param_interface()->get_viscosity();

  // damping/friction coefficient of a rigid sphere (Stokes' law for very small Reynolds numbers)
  return 6 * std::numbers::pi * eta * radius_;
}

/*-----------------------------------------------------------------------------------------------------------*
 |computes the number of different random numbers required in each time step for generation of
 stochastic    | |forces; (public)           grill   03/14|
 *----------------------------------------------------------------------------------------------------------*/
int Discret::Elements::Rigidsphere::how_many_random_numbers_i_need()
{
  /*three randomly excited (translational) DOFs for Rigidsphere element*/
  return 3;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Core::GeometricSearch::BoundingVolume Discret::Elements::Rigidsphere::get_bounding_volume(
    const Core::FE::Discretization& discret,
    const Core::LinAlg::Vector<double>& result_data_dofbased,
    const Core::GeometricSearch::GeometricSearchParams& params) const
{
  // Get the element displacements.
  std::vector<int> lm, lmowner, lmstride;
  this->location_vector(discret, lm, lmowner, lmstride);
  std::vector<double> mydisp = Core::FE::extract_values(result_data_dofbased, lm);

  // Add reference position.
  if (mydisp.size() != 3)
    FOUR_C_THROW("Got unexpected number of DOFs. Expected 3, but received {}", mydisp.size());
  Core::LinAlg::Matrix<3, 1, double> sphere_center;
  for (unsigned int i_dof = 0; i_dof < 3; i_dof++)
    sphere_center(i_dof) = mydisp[i_dof] + nodes()[0]->x()[i_dof];

  Core::GeometricSearch::BoundingVolume bounding_volume;
  bounding_volume.add_point(sphere_center);

  // Add the radius times a safety factor.
  const double safety_factor = params.get_sphere_bounding_volume_scaling();
  const double radius = Rigidsphere::radius();
  bounding_volume.extend_boundaries(radius * safety_factor);

  return bounding_volume;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Discret::Elements::Rigidsphere::get_generalized_interpolation_matrix_variations_at_xi(
    Core::LinAlg::SerialDenseMatrix& Ivar, const double& dummy1,
    const std::vector<double>& dummy2) const
{
  Core::LinAlg::Matrix<6, 3, double> Ivar_fixedsize(&Ivar(0, 0), true);
  for (unsigned int i = 0; i < 3; ++i) Ivar_fixedsize(i, i) = 1.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Discret::Elements::Rigidsphere::get_generalized_interpolation_matrix_increments_at_xi(
    Core::LinAlg::SerialDenseMatrix& Iinc, const double& dummy1,
    const std::vector<double>& dummy2) const
{
  Core::LinAlg::Matrix<6, 3, double> Iinc_fixedsize(&Iinc(0, 0), true);
  for (unsigned int i = 0; i < 3; ++i) Iinc_fixedsize(i, i) = 1.0;
}

/*-----------------------------------------------------------------------------------------------------------*
 | computes stochastic forces and resulting stiffness (public) grill   03/14|
 *----------------------------------------------------------------------------------------------------------*/
void Discret::Elements::Rigidsphere::calc_stochastic_force(
    Teuchos::ParameterList& params,                //!< parameter list
    const std::vector<double>& vel,                //!< element velocity vector
    const std::vector<double>& disp,               //!< element disp vector
    Core::LinAlg::SerialDenseMatrix* stiffmatrix,  //!< element stiffness matrix
    Core::LinAlg::SerialDenseVector* force)        //!< element internal force vector
{
  // damping coefficient
  double gamma = my_damping_constant();

  /*get pointer at Epetra multivector in parameter list linking to random numbers for stochastic
   * forces with zero mean and standard deviation (2*kT / dt)^0.5*/
  std::shared_ptr<Core::LinAlg::MultiVector<double>> randomnumbers =
      params_interface().get_brownian_dyn_param_interface()->get_random_forces();

  if (force != nullptr)
  {
    for (unsigned int k = 0; k < 3; ++k)
    {
      (*force)(k) -= sqrt(gamma) * (*randomnumbers)(k).local_values_as_span()[lid()];
    }
  }

  // no contribution to stiffmatrix

  return;
}

FOUR_C_NAMESPACE_CLOSE
