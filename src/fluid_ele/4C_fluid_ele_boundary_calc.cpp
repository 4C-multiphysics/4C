// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_boundary_calc.hpp"

#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_material.hpp"
#include "4C_mat_carreauyasuda.hpp"
#include "4C_mat_fluid_linear_density_viscosity.hpp"
#include "4C_mat_fluid_murnaghantait.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_herschelbulkley.hpp"
#include "4C_mat_modpowerlaw.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_sutherland.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::FluidBoundaryImpl<distype>::FluidBoundaryImpl()
    :  // Discret::Elements::FluidBoundaryInterface(),
      xyze_(Core::LinAlg::Initialization::zero),
      funct_(Core::LinAlg::Initialization::zero),
      deriv_(Core::LinAlg::Initialization::zero),
      unitnormal_(Core::LinAlg::Initialization::zero),
      velint_(Core::LinAlg::Initialization::zero),
      drs_(0.0),
      fac_(0.0),
      visc_(0.0),
      densaf_(1.0)
{
  // pointer to class FluidImplParameterTimInt for time integration
  fldparatimint_ = Discret::Elements::FluidEleParameterTimInt::instance();
  // initialize also general parameter list, also it will be overwritten in derived subclasses
  fldpara_ = Discret::Elements::FluidEleParameterStd::instance();

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::evaluate_action(
    Discret::Elements::FluidBoundary* ele1, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // get the action required
  const auto act = Teuchos::getIntegralValue<FLD::BoundaryAction>(params, "action");

  // get status of Ale
  const bool isale = ele1->parent_element()->is_ale();

  switch (act)
  {
    case FLD::integrate_Shapefunction:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
      std::vector<double> mydispnp;

      if (isale)
      {
        dispnp = discretization.get_state("dispnp");
        if (dispnp != nullptr)
        {
          mydispnp = Core::FE::extract_values(*dispnp, lm);
        }
      }

      integrate_shape_function(ele1, params, discretization, lm, elevec1, mydispnp);
      break;
    }
    case FLD::calc_area:
    {
      if (ele1->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
        area_calculation(ele1, params, discretization, lm);
      break;
    }
    case FLD::calc_pressure_bou_int:
    {
      if (ele1->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
        pressure_boundary_integral(ele1, params, discretization, lm);
      break;
    }
    // general action to calculate the flow rate
    case FLD::calc_flowrate:
    {
      compute_flow_rate(ele1, params, discretization, lm, elevec1);
      break;
    }
    case FLD::flowratederiv:
    {
      flow_rate_deriv(
          ele1, params, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
    }
    case FLD::Outletimpedance:
    {
      impedance_integration(ele1, params, discretization, lm, elevec1);
      break;
    }

    case FLD::dQdu:
    {
      d_qdu(ele1, params, discretization, lm, elevec1);
      break;
    }
    case FLD::boundary_calc_node_normal:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
      std::vector<double> mydispnp;

      if (isale)
      {
        dispnp = discretization.get_state("dispnp");
        if (dispnp != nullptr)
        {
          mydispnp.resize(lm.size());
          mydispnp = Core::FE::extract_values(*dispnp, lm);
        }
      }

      Core::FE::element_node_normal<distype>(funct_, deriv_, fac_, unitnormal_, drs_, xsi_, xyze_,
          ele1, discretization, elevec1, mydispnp, Core::FE::is_nurbs<distype>,
          ele1->parent_element()->is_ale());
      break;
    }
    case FLD::calc_node_curvature:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
      std::vector<double> mydispnp;

      if (isale)
      {
        dispnp = discretization.get_state("dispnp");
        if (dispnp != nullptr)
        {
          mydispnp.resize(lm.size());
          mydispnp = Core::FE::extract_values(*dispnp, lm);
        }
      }

      std::shared_ptr<const Core::LinAlg::Vector<double>> normals;
      std::vector<double> mynormals;

      normals = discretization.get_state("normals");
      if (normals != nullptr)
      {
        mynormals.resize(lm.size());
        mynormals = Core::FE::extract_values(*normals, lm);
      }

      // what happens, if the mynormals vector is empty? (ehrl)
      FOUR_C_THROW(
          "the action calc_node_curvature has not been called by now. What happens, if the "
          "mynormal vector is empty");

      element_mean_curvature(ele1, params, discretization, lm, elevec1, mydispnp, mynormals);
      break;
    }
    case FLD::calc_Neumann_inflow:
    {
      neumann_inflow(ele1, params, discretization, lm, elemat1, elevec1);
      break;
    }
    case FLD::center_of_mass_calc:
    {
      // evaluate center of mass
      if (ele1->owner() == Core::Communication::my_mpi_rank(discretization.get_comm()))
        center_of_mass_calculation(ele1, params, discretization, lm);
      break;
    }
    case FLD::traction_velocity_component:
    {
      calc_traction_velocity_component(ele1, params, discretization, lm, elevec1);
      break;
    }
    case FLD::traction_Uv_integral_component:
    {
      compute_neumann_uv_integral(ele1, params, discretization, lm, elevec1);
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown type of action for FluidBoundaryImpl!");
      break;
    }
  }  // end of switch(act)
}

/*----------------------------------------------------------------------*
 |  Integrate a Surface Neumann boundary condition (public)  gammi 04/07|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::FluidBoundaryImpl<distype>::evaluate_neumann(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseMatrix* elemat1_epetra)
{
  // find out whether we will use a time curve
  const double time = fldparatimint_->time();

  // get values, switches and spatial functions from the condition
  // (assumed to be constant on element boundary)
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition.parameters().get<std::vector<double>>("VAL");
  const auto func = condition.parameters().get<std::vector<std::optional<int>>>("FUNCT");
  const std::string* type = &condition.parameters().get<std::string>("TYPE");

  // get time factor for Neumann term
  const double timefac = fldparatimint_->time_fac_rhs();
  const double timefacn = (1.0 - fldparatimint_->theta()) * fldparatimint_->dt();

  // get Gaussrule
  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // get local node coordinates
  // (we have a nsd_ dimensional domain, since nsd_ determines the dimension of FluidBoundary
  // element!)
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);


  // THESE ARE NEEDED IF DENSITY OR A SCALAR IS USED in the BC (which normally is NOT the case)
  //========================================================
  // get scalar vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> scaaf = discretization.get_state("scaaf");
  if (scaaf == nullptr) FOUR_C_THROW("Cannot get state vector 'scaaf'");

  // extract local values from global vector
  std::vector<double> myscaaf = Core::FE::extract_values(*scaaf, lm);

  Core::LinAlg::Matrix<bdrynen_, 1> escaaf(Core::LinAlg::Initialization::zero);

  // insert scalar into element array
  // the scalar is stored to the pressure dof
  for (int inode = 0; inode < bdrynen_; ++inode)
  {
    escaaf(inode) = myscaaf[(nsd_) + (inode * numdofpernode_)];
  }

  // get thermodynamic pressure at n+1/n+alpha_F
  const double thermpressaf = params.get<double>("thermpress at n+alpha_F/n+1", 0.0);

  // extract pressure values from global velocity/pressure vector
  // (needed for weakly_compressible fluids)
  Core::LinAlg::Matrix<bdrynen_, 1> epreaf(Core::LinAlg::Initialization::zero);
  Discret::Elements::FluidEleParameter* fldpara =
      Discret::Elements::FluidEleParameterStd::instance();
  if (fldpara->physical_type() == Inpar::FLUID::weakly_compressible or
      fldpara->physical_type() == Inpar::FLUID::weakly_compressible_stokes)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> velaf = discretization.get_state("velaf");
    if (velaf == nullptr) FOUR_C_THROW("Cannot get state vector 'velaf'");

    // extract local values from global vector
    std::vector<double> myvelaf = Core::FE::extract_values(*velaf, lm);

    // insert pressure into element array
    for (int inode = 0; inode < bdrynen_; inode++)
    {
      epreaf(inode) = myvelaf[nsd_ + inode * numdofpernode_];
    }
  }
  else
  {
    // insert pressure into element array
    for (int inode = 0; inode < bdrynen_; inode++)
    {
      epreaf(inode) = 0.0;
    }
  }
  //========================================================


  // add potential ALE displacements
  if (ele->parent_element()->is_ale())
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
    std::vector<double> mydispnp;
    dispnp = discretization.get_state("dispnp");
    if (dispnp != nullptr)
    {
      mydispnp.resize(lm.size());
      mydispnp = Core::FE::extract_values(*dispnp, lm);
    }

    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < (nsd_); ++idim)
      {
        xyze_(idim, inode) += mydispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  // --------------------------------------------------
  // Now do the nurbs specific stuff
  // --------------------------------------------------

  // In the case of nurbs the normal vector is multiplied with normalfac
  double normalfac = 0.0;
  std::vector<Core::LinAlg::SerialDenseVector> myknots(bdrynsd_);
  Core::LinAlg::SerialDenseVector weights(bdrynen_);

  // for isogeometric elements --- get knotvectors for parent
  // element and surface element, get weights
  if (Core::FE::is_nurbs<distype>)
  {
    std::vector<Core::LinAlg::SerialDenseVector> mypknots(nsd_);

    bool zero_size =
        Core::FE::Nurbs::get_knot_vector_and_weights_for_nurbs_boundary(ele, ele->surface_number(),
            ele->parent_element()->id(), discretization, mypknots, myknots, weights, normalfac);
    if (zero_size)
    {
      return 0;
    }
  }
  /*----------------------------------------------------------------------*
  |               start loop over integration points                     |
  *----------------------------------------------------------------------*/
  for (int gpid = 0; gpid < intpoints.ip().nquad; ++gpid)
  {
    // evaluate shape functions and their derivatives,
    // compute unit normal vector and infinitesimal area element drs
    // (evaluation of nurbs-specific stuff not activated here)
    Core::FE::eval_shape_func_at_bou_int_point<distype>(funct_, deriv_, fac_, unitnormal_, drs_,
        xsi_, xyze_, intpoints, gpid, &myknots, &weights, Core::FE::is_nurbs<distype>);

    // get the required material information
    std::shared_ptr<Core::Mat::Material> material = ele->parent_element()->material();

    // get density
    // (evaluation always at integration point, in contrast to parent element)
    get_density(material, escaaf, thermpressaf, epreaf);

    const double tol = 1e-8;
    if (densfac_ < (1.0 - tol) or densfac_ > (1.0 + tol))
    {
      // If you got this warning have a look in GetDensity(...) for clarification!
      std::cout << "                                                                    "
                << std::endl;
      std::cout << "                                                                    "
                << std::endl;
      std::cout << "          WARNING:                                                  "
                << std::endl;
      std::cout << "                  4C scales your NEUMANN BC with the DENSITY!!!   "
                << std::endl;
      std::cout << "                                                                    "
                << std::endl;
      std::cout << "                  Do you really want this?                          "
                << std::endl;
      std::cout << "                  Like, super sure about this?                      "
                << std::endl;
      std::cout << "                  Might want to think this through one more time...."
                << std::endl;
      std::cout << "                                                                    "
                << std::endl;
      std::cout << "                                                                    "
                << std::endl;
      std::cout << "        Fine... Do what you want...                                 "
                << std::endl;
      std::cout << "        densfac_=               " << densfac_ << std::endl;
    }

    const double fac_time_dens = fac_ * timefac * densfac_;
    const double fac_time_densn = fac_ * timefacn;

    // factor given by spatial function
    double functfac = 1.0;
    double functfacn = 1.0;

    // global coordinates of gausspoint
    Core::LinAlg::Matrix<(nsd_), 1> coordgp(Core::LinAlg::Initialization::uninitialized);

    // determine coordinates of current Gauss point
    coordgp.multiply(xyze_, funct_);

    // we need a 3D position vector for function evaluation!
    double coordgp3D[3];
    coordgp3D[0] = 0.0;
    coordgp3D[1] = 0.0;
    coordgp3D[2] = 0.0;
    for (int i = 0; i < nsd_; i++) coordgp3D[i] = coordgp(i);

    const double* coordgpref = coordgp3D;  // needed for function evaluation

    for (int idim = 0; idim < (nsd_); ++idim)
    {
      if (*type == "Live")
      {
        if (onoff[idim])  // Is this dof activated
        {
          if (func[idim].has_value() && func[idim].value() > 0)
          {
            // evaluate function at current gauss point
            functfac = Global::Problem::instance()
                           ->function_by_id<Core::Utils::FunctionOfSpaceTime>(func[idim].value())
                           .evaluate(coordgpref, time, idim);
            if (fldparatimint_->is_new_ost_implementation())
              functfacn = Global::Problem::instance()
                              ->function_by_id<Core::Utils::FunctionOfSpaceTime>(func[idim].value())
                              .evaluate(coordgpref, time - fldparatimint_->dt(), idim);
          }
          else
          {
            functfac = 1.0;
            functfacn = 1.0;
          }

          const double valfac = val[idim] * fac_time_dens * functfac;
          for (int inode = 0; inode < bdrynen_; ++inode)
          {
            elevec1_epetra[inode * numdofpernode_ + idim] += funct_(inode) * valfac;
            if (fldparatimint_->is_new_ost_implementation())
            {
              const double valfacn = val[idim] * fac_time_densn * functfacn;
              elevec1_epetra[inode * numdofpernode_ + idim] += funct_(inode) * valfacn;
            }
          }  // end is_new_ost_implementation
        }  // if onoff
      }
      else if (*type == "pseudo_orthopressure")
      {
        if (idim != 0 and onoff[idim])
          FOUR_C_THROW(
              "If you apply a pseudo_orthopressure load on the fluid, only a load in\n"
              "the first component (which corresponds to the normal direction) is valid!");

        if (onoff[0])  // Do we have a load in normal direction?
        {
          if (func[0].has_value() && func[0].value() > 0)
          {
            // evaluate function at current gauss point
            functfac = Global::Problem::instance()
                           ->function_by_id<Core::Utils::FunctionOfSpaceTime>(func[0].value())
                           .evaluate(coordgpref, time, idim);
            if (fldparatimint_->is_new_ost_implementation())
              functfacn = Global::Problem::instance()
                              ->function_by_id<Core::Utils::FunctionOfSpaceTime>(func[0].value())
                              .evaluate(coordgpref, time - fldparatimint_->dt(), idim);
          }
          else
          {
            functfac = 1.0;
            functfacn = 1.0;
          }

          const double valfac = val[0] * fac_time_dens * functfac;
          for (int inode = 0; inode < bdrynen_; ++inode)
          {
            elevec1_epetra[inode * numdofpernode_ + idim] +=
                funct_(inode) * valfac * (-unitnormal_(idim));

            if (fldparatimint_->is_new_ost_implementation())
            {
              const double valfacn = val[0] * fac_time_densn * functfacn;
              elevec1_epetra[inode * numdofpernode_ + idim] +=
                  funct_(inode) * valfacn * (-unitnormal_(idim));
            }
          }  // end is_new_ost_implementation
        }  // if onoff
      }
      else
        FOUR_C_THROW(
            "The type '{}' is not supported in the fluid neumann condition!", type->c_str());

    }  // for(int idim=0; idim<(nsd_); ++idim)
  }

  return 0;
}



/*----------------------------------------------------------------------*
 | compute additional term at Neumann inflow boundary          vg 01/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::neumann_inflow(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseVector& elevec1)
{
  if (fldparatimint_->is_new_ost_implementation())
  {
    FOUR_C_THROW("NEUMANN INFLOW IS NOT IMPLEMENTED FOR NEW OST AS OF YET!");
  }  // end is_new_ost_implementation

  //----------------------------------------------------------------------
  // get control parameters for time integration
  //----------------------------------------------------------------------
  // get timefactor for left-hand side
  // One-step-Theta:    timefac = theta*dt
  // BDF2:              timefac = 2/3 * dt
  // af-genalpha: timefac = (alpha_F/alpha_M) * gamma * dt
  // np-genalpha: timefac = (alpha_F/alpha_M) * gamma * dt
  // genalpha:    timefac =  alpha_F * gamma * dt
  const double timefac = fldparatimint_->time_fac();

  // get timefactor for right-hand side
  // One-step-Theta:            timefacrhs = theta*dt
  // BDF2:                      timefacrhs = 2/3 * dt
  // af-genalpha:               timefacrhs = (1/alpha_M) * gamma * dt
  // np-genalpha:               timefacrhs = (1/alpha_M) * gamma * dt
  // genalpha:                  timefacrhs = 1.0
  double timefacrhs = fldparatimint_->time_fac_rhs();

  // check ALE status
  const bool isale = ele->parent_element()->is_ale();

  // set flag for type of linearization to default value (fixed-point-like)
  bool is_newton = fldpara_->is_newton();

  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // get global node coordinates for nsd_-dimensional domain
  // (nsd_: number of spatial dimensions of FluidBoundary element!)
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);

  // add potential ALE displacements
  std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
  std::vector<double> mydispnp;
  if (isale)
  {
    dispnp = discretization.get_state("dispnp");
    if (dispnp != nullptr)
    {
      mydispnp.resize(lm.size());
      mydispnp = Core::FE::extract_values(*dispnp, lm);
    }

    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < (nsd_); ++idim)
      {
        xyze_(idim, inode) += mydispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  // get velocity and scalar vector at time n+alpha_F/n+1
  std::shared_ptr<const Core::LinAlg::Vector<double>> velaf = discretization.get_state("velaf");
  std::shared_ptr<const Core::LinAlg::Vector<double>> scaaf = discretization.get_state("scaaf");
  if (velaf == nullptr or scaaf == nullptr)
    FOUR_C_THROW("Cannot get state vector 'velaf' and/or 'scaaf'");

  // extract local values from global vector
  std::vector<double> myvelaf(lm.size());
  std::vector<double> myscaaf(lm.size());
  myvelaf = Core::FE::extract_values(*velaf, lm);
  myscaaf = Core::FE::extract_values(*scaaf, lm);

  // create Epetra objects for scalar array and velocities
  Core::LinAlg::Matrix<nsd_, bdrynen_> evelaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<bdrynen_, 1> epreaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<bdrynen_, 1> escaaf(Core::LinAlg::Initialization::zero);

  // insert velocity, pressure and scalar into element array
  for (int inode = 0; inode < bdrynen_; ++inode)
  {
    for (int idim = 0; idim < (nsd_); ++idim)
    {
      evelaf(idim, inode) = myvelaf[idim + (inode * numdofpernode_)];
    }
    epreaf(inode) = myvelaf[nsd_ + inode * numdofpernode_];
    escaaf(inode) = myscaaf[(nsd_) + (inode * numdofpernode_)];
  }

  // get thermodynamic pressure at n+1/n+alpha_F
  const double thermpressaf = params.get<double>("thermpress at n+alpha_F/n+1", 1.0);

  // --------------------------------------------------
  // nurbs-specific stuff
  // --------------------------------------------------
  // normal vector multiplied by normalfac for nurbs
  double normalfac = 0.0;
  std::vector<Core::LinAlg::SerialDenseVector> mypknots(nsd_);
  std::vector<Core::LinAlg::SerialDenseVector> myknots(bdrynsd_);
  Core::LinAlg::SerialDenseVector weights(bdrynen_);

  // get knotvectors for parent element and surface element as well as weights
  // for isogeometric elements
  if (Core::FE::is_nurbs<distype>)
  {
    bool zero_size =
        Core::FE::Nurbs::get_knot_vector_and_weights_for_nurbs_boundary(ele, ele->surface_number(),
            ele->parent_element()->id(), discretization, mypknots, myknots, weights, normalfac);
    if (zero_size)
    {
      return;
    }
  }

  /*----------------------------------------------------------------------*
   |               start loop over integration points                     |
   *----------------------------------------------------------------------*/
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    // evaluate shape functions and their derivatives,
    // compute unit normal vector and infinitesimal area element drs
    // (evaluation of nurbs-specific stuff not activated here)
    Core::FE::eval_shape_func_at_bou_int_point<distype>(funct_, deriv_, fac_, unitnormal_, drs_,
        xsi_, xyze_, intpoints, gpid, &myknots, &weights, Core::FE::is_nurbs<distype>);

    // normal vector scaled by special factor in case of nurbs
    if (Core::FE::is_nurbs<distype>) unitnormal_.scale(normalfac);

    // compute velocity vector and normal velocity at integration point
    // (values at n+alpha_F for generalized-alpha scheme, n+1 otherwise)
    double normvel = 0.0;
    velint_.multiply(evelaf, funct_);
    normvel = velint_.dot(unitnormal_);

    // check normal velocity -> further computation only required for
    // negative normal velocity, that is, inflow at this Neumann boundary
    if (normvel < -0.0001)
    {
      // get the required material information
      std::shared_ptr<Core::Mat::Material> material = ele->parent_element()->material();

      // get density
      // (evaluation always at integration point, in contrast to parent element)
      get_density(material, escaaf, thermpressaf, epreaf);

      // extended integration factors for left- and right-hand side, respectively
      const double lhsfac = densaf_ * normvel * timefac * fac_;
      const double rhsfac = densaf_ * normvel * timefacrhs * fac_;

      // compute matrix contribution (fill diagonal elements)
      /*
              /                        \
             |                          |
           - |  v , rho * Du ( u o n )  |
             |                          |
              \                        /
      */
      for (int idim = 0; idim < nsd_; ++idim)  // loop over dimensions
      {
        for (int vi = 0; vi < bdrynen_; ++vi)  // loop over rows
        {
          const double vlhs = lhsfac * funct_(vi);

          const int fvi = numdofpernode_ * vi + idim;

          for (int ui = 0; ui < bdrynen_; ++ui)  // loop over columns
          {
            const int fui = numdofpernode_ * ui + idim;

            elemat1(fvi, fui) -= vlhs * funct_(ui);
          }  // end loop over columns
        }  // end loop over rows
      }  // end loop over dimensions

      // compute additional matrix contribution for Newton linearization
      if (is_newton)
      {
        // integration factor
        const double lhsnewtonfac = densaf_ * timefac * fac_;

        // dyadic product of unit normal vector and velocity vector
        Core::LinAlg::Matrix<nsd_, nsd_> n_x_u(Core::LinAlg::Initialization::zero);
        n_x_u.multiply_nt(velint_, unitnormal_);

        /*
                /                        \
               |                          |
             - |  v , rho * u ( Du o n )  |
               |                          |
                \                        /

               rho * v_i * u_i * Du_j * n_j

        */
        for (int vi = 0; vi < bdrynen_; ++vi)  // loop rows
        {
          const double dens_dt_v = lhsnewtonfac * funct_(vi);

          for (int idimrow = 0; idimrow < nsd_; ++idimrow)  // loop row dim.
          {
            const int fvi = numdofpernode_ * vi + idimrow;

            for (int ui = 0; ui < bdrynen_; ++ui)  // loop columns
            {
              const double dens_dt_v_Du = dens_dt_v * funct_(ui);

              for (int idimcol = 0; idimcol < nsd_; ++idimcol)  // loop column dim.
              {
                const int fui = numdofpernode_ * ui + idimcol;

                elemat1(fvi, fui) -= dens_dt_v_Du * n_x_u(idimrow, idimcol);
              }  // end loop row dimensions
            }  // end loop rows
          }  // end loop column dimensions
        }  // end loop columns
      }  // end of Newton loop

      // compute rhs contribution
      Core::LinAlg::Matrix<nsd_, 1> vrhs(velint_, false);
      vrhs.scale(rhsfac);

      for (int vi = 0; vi < bdrynen_; ++vi)  // loop over rows
      {
        for (int idim = 0; idim < nsd_; ++idim)  // loop over dimensions
        {
          const int fvi = numdofpernode_ * vi + idim;

          elevec1(fvi) += funct_(vi) * vrhs(idim);
        }  // end loop over dimensions
      }  // end loop over rows
    }
  }

  return;
}  // Discret::Elements::FluidSurface::neumann_inflow


/*----------------------------------------------------------------------*
 |  Integrate shapefunctions over surface (private)            gjb 07/07|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::integrate_shape_function(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1, const std::vector<double>& edispnp)
{
  // get status of Ale
  const bool isale = ele->parent_element()->is_ale();

  // get Gaussrule
  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // get node coordinates
  // (we have a nsd_ dimensional domain, since nsd_ determines the dimension of FluidBoundary
  // element!)
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);

  if (isale)
  {
    FOUR_C_ASSERT(edispnp.size() != 0, "paranoid");

    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < (nsd_); ++idim)
      {
        xyze_(idim, inode) += edispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  /*----------------------------------------------------------------------*
  |               start loop over integration points                     |
  *----------------------------------------------------------------------*/
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    // Computation of the integration factor & shape function at the Gauss point & derivative of the
    // shape function at the Gauss point Computation of the unit normal vector at the Gauss points
    // is not activated here Computation of nurb specific stuff is not activated here
    Core::FE::eval_shape_func_at_bou_int_point<distype>(funct_, deriv_, fac_, unitnormal_, drs_,
        xsi_, xyze_, intpoints, gpid, nullptr, nullptr, Core::FE::is_nurbs<distype>);

    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < (nsd_); idim++)
      {
        elevec1(inode * numdofpernode_ + idim) += funct_(inode) * fac_;
      }
    }

  } /* end of loop over integration points gpid */


  return;
}  // Discret::Elements::FluidSurface::integrate_shape_function


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::element_mean_curvature(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1, const std::vector<double>& edispnp,
    std::vector<double>& enormals)
{
  // get status of Ale
  const bool isale = ele->parent_element()->is_ale();

  // get Gauss rule
  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // node normals &
  Core::LinAlg::Matrix<nsd_, bdrynen_> norm_elem(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<bdrynsd_, nsd_> dxyzdrs(Core::LinAlg::Initialization::zero);

  // coordinates of current node in reference coordinates
  Core::LinAlg::Matrix<bdrynsd_, 1> xsi_node(Core::LinAlg::Initialization::zero);

  // get node coordinates
  // (we have a nsd_ dimensional domain, since nsd_ determines the dimension of FluidBoundary
  // element!)
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);

  if (isale)
  {
    FOUR_C_ASSERT(edispnp.size() != 0, "paranoid");

    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        xyze_(idim, inode) += edispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  // set normal vectors to length = 1.0
  // normal vector is coming from outside
  for (int inode = 0; inode < bdrynen_; ++inode)
  {
    // double length = 0.0;
    for (int idim = 0; idim < nsd_; ++idim)
    {
      norm_elem(idim, inode) = enormals[numdofpernode_ * inode + idim];
    }
  }
  // compute normalized normal vector
  norm_elem.scale(1 / norm_elem.norm2());

  // get local node coordinates of the element
  // function gives back a matrix with the local node coordinates of the element (nsd_,bdrynen_)
  // the function gives back an Core::LinAlg::SerialDenseMatrix!!!
  Core::LinAlg::SerialDenseMatrix xsi_ele =
      Core::FE::get_ele_node_numbering_nodes_paramspace(distype);

  // ============================== loop over nodes ==========================
  for (int inode = 0; inode < bdrynen_; ++inode)
  {
    // the local node coordinates matrix is split to a vector containing the local coordinates of
    // the actual node
    for (int idim = 0; idim < bdrynsd_; idim++)
    {
      xsi_node(idim) = xsi_ele(idim, inode);
    }

    // get shape derivatives at this node
    Core::FE::shape_function<distype>(xsi_node, funct_);

    // the metric tensor and its determinant
    // Core::LinAlg::SerialDenseMatrix      metrictensor(nsd_,nsd_);
    Core::LinAlg::Matrix<bdrynsd_, bdrynsd_> metrictensor(Core::LinAlg::Initialization::zero);

    // Additionally, compute metric tensor
    Core::FE::compute_metric_tensor_for_boundary_ele<distype>(xyze_, deriv_, metrictensor, drs_);

    dxyzdrs.multiply_nt(deriv_, xyze_);

    // calculate mean curvature H at node.
    double H = 0.0;
    Core::LinAlg::Matrix<bdrynsd_, nsd_> dn123drs(Core::LinAlg::Initialization::uninitialized);

    dn123drs.multiply_nt(deriv_, norm_elem);

    // Acc. to Bronstein ..."mittlere Kruemmung":
    // calculation of the mean curvature for a surface element
    if (bdrynsd_ == 2)
    {
      double L = 0.0, twoM = 0.0, N = 0.0;
      for (int i = 0; i < 3; i++)
      {
        L += (-1.0) * dxyzdrs(0, i) * dn123drs(0, i);
        twoM += (-1.0) * dxyzdrs(0, i) * dn123drs(1, i) - dxyzdrs(1, i) * dn123drs(0, i);
        N += (-1.0) * dxyzdrs(1, i) * dn123drs(1, i);
      }
      // mean curvature: H = 0.5*(k_1+k_2)
      H = 0.5 * (metrictensor(0, 0) * N - twoM * metrictensor(0, 1) + metrictensor(1, 1) * L) /
          (drs_ * drs_);
    }
    else
      FOUR_C_THROW(
          "Calculation of the mean curvature is only implemented for a 2D surface element");


    // get the number of elements adjacent to this node. Find out how many
    // will contribute to the interpolated mean curvature value.
    int contr_elements = 0;
    Core::Nodes::Node* thisNode = (ele->nodes())[inode];
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (thisNode == nullptr) FOUR_C_THROW("No node!\n");
#endif
    int NumElement = thisNode->num_element();
    Core::Elements::Element** ElementsPtr = thisNode->elements();

    // loop over adjacent Fluid elements
    for (int ele = 0; ele < NumElement; ele++)
    {
      Core::Elements::Element* Element = ElementsPtr[ele];

      // get surfaces
      std::vector<std::shared_ptr<Core::Elements::Element>> surfaces = Element->surfaces();

      // loop over surfaces: how many free surfaces with this node on it?
      for (unsigned int surf = 0; surf < surfaces.size(); ++surf)
      {
        std::shared_ptr<Core::Elements::Element> surface = surfaces[surf];
        Core::Nodes::Node** NodesPtr = surface->nodes();
        int numfsnodes = 0;
        bool hasthisnode = false;

        for (int surfnode = 0; surfnode < surface->num_node(); ++surfnode)
        {
          Core::Nodes::Node* checkNode = NodesPtr[surfnode];
          // check whether a free surface condition is active on this node
          if (checkNode->get_condition("FREESURFCoupling") != nullptr)
          {
            numfsnodes++;
          }
          if (checkNode->id() == thisNode->id())
          {
            hasthisnode = true;
          }
        }

        if (numfsnodes == surface->num_node() and hasthisnode)
        {
          // this is a free surface adjacent to this node.
          contr_elements++;
        }
      }
    }
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (!contr_elements) FOUR_C_THROW("No contributing elements found!\n");
#endif

    for (int idim = 0; idim < nsd_; ++idim)
    {
      elevec1[inode * numdofpernode_ + idim] = H / contr_elements;
    }
    elevec1[inode * numdofpernode_ + (numdofpernode_ - 1)] = 0.0;
  }  // END: loop over nodes

}  // Discret::Elements::FluidSurface::element_mean_curvature

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::area_calculation(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm)
{
  //------------------------------------------------------------------
  // get and set density and viscosity (still required for following routines:
  // FluidImpedanceBc/FluidVolumetricSurfaceFlowBc/FluidCouplingBc::Area)
  //------------------------------------------------------------------
  std::shared_ptr<Core::Mat::Material> mat = ele->parent_element()->material();
  if (mat->material_type() == Core::Materials::m_fluid)
  {
    const Mat::NewtonianFluid* actmat = static_cast<const Mat::NewtonianFluid*>(mat.get());
    densaf_ = actmat->density();
    visc_ = actmat->viscosity();
  }
  else if (mat->material_type() == Core::Materials::m_carreauyasuda)
  {
    const Mat::CarreauYasuda* actmat = static_cast<const Mat::CarreauYasuda*>(mat.get());
    densaf_ = actmat->density();

    const double nu_inf = actmat->nu_inf();  // parameter for infinite-shear viscosity

    // dynamic viscosity = kinematic viscosity * density
    visc_ = nu_inf * densaf_;
  }

  params.set<double>("density", densaf_);
  params.set<double>("viscosity", visc_);
  //------------------------------------------------------------------
  // end of get and set density and viscosity
  //------------------------------------------------------------------

  //------------------------------------------------------------------
  // start of actual area calculation
  //------------------------------------------------------------------
  // get node coordinates (nsd_: dimension of boundary element!)
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);

  // add potential ALE displacements
  std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
  std::vector<double> mydispnp;
  if (ele->parent_element()->is_ale())
  {
    dispnp = discretization.get_state("dispnp");
    if (dispnp != nullptr)
    {
      mydispnp.resize(lm.size());
      mydispnp = Core::FE::extract_values(*dispnp, lm);
    }
    FOUR_C_ASSERT(mydispnp.size() != 0, "paranoid");
    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        xyze_(idim, inode) += mydispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  // get initial value for area
  double area = params.get<double>("area");

  // get Gauss rule
  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    Core::FE::eval_shape_func_at_bou_int_point<distype>(funct_, deriv_, fac_, unitnormal_, drs_,
        xsi_, xyze_, intpoints, gpid, nullptr, nullptr, Core::FE::is_nurbs<distype>);

    // add to area integral
    area += fac_;
  }

  // set final value for area
  params.set<double>("area", area);
  //------------------------------------------------------------------
  // end of actual area calculation
  //------------------------------------------------------------------

}  // Discret::Elements::FluidSurface::area_calculation


/*----------------------------------------------------------------------*
 |                                                       ismail 04/2010 |
 |                                                           vg 06/2013 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::pressure_boundary_integral(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm)
{
  // extract pressure values from global velocity/pressure vector
  // renamed to "velaf" to be consistent in fluidimplicitintegration.cpp (krank 12/13)
  std::shared_ptr<const Core::LinAlg::Vector<double>> velnp = discretization.get_state("velaf");
  if (velnp == nullptr) FOUR_C_THROW("Cannot get state vector 'velaf'");

  std::vector<double> myvelnp = Core::FE::extract_values(*velnp, lm);

  Core::LinAlg::Matrix<1, bdrynen_> eprenp(Core::LinAlg::Initialization::zero);
  for (int inode = 0; inode < bdrynen_; inode++)
  {
    eprenp(inode) = myvelnp[nsd_ + inode * numdofpernode_];
  }

  // get node coordinates (nsd_: dimension of boundary element!)
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);

  // add potential ALE displacements
  std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
  std::vector<double> mydispnp;
  if (ele->parent_element()->is_ale())
  {
    dispnp = discretization.get_state("dispnp");
    if (dispnp != nullptr)
    {
      mydispnp.resize(lm.size());
      mydispnp = Core::FE::extract_values(*dispnp, lm);
    }
    FOUR_C_ASSERT(mydispnp.size() != 0, "paranoid");
    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        xyze_(idim, inode) += mydispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  // get initial value for pressure boundary integral
  double press_int = params.get<double>("pressure boundary integral");

  // get Gauss rule
  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    Core::FE::eval_shape_func_at_bou_int_point<distype>(funct_, deriv_, fac_, unitnormal_, drs_,
        xsi_, xyze_, intpoints, gpid, nullptr, nullptr, Core::FE::is_nurbs<distype>);

    // add to pressure boundary integral
    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      press_int += funct_(inode) * eprenp(inode) * fac_;
    }
  }

  // set final value for pressure boundary integral
  params.set<double>("pressure boundary integral", press_int);

}  // Discret::Elements::FluidSurface::pressure_boundary_integral


/*----------------------------------------------------------------------*
 |                                                        ismail 10/2010|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::center_of_mass_calculation(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm)
{
  //------------------------------------------------------------------
  // This calculates the integrated the pressure from the
  // the actual pressure values
  //------------------------------------------------------------------

  // get integration rule
  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // get node coordinates
  // (we have a nsd_ dimensional domain, since nsd_ determines the dimension of FluidBoundary
  // element!)
  // Core::Geo::fill_initial_position_array<distype,nsd_,Core::LinAlg::SerialDenseMatrix>(ele,xyze_);
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);

  // Add the deformation of the ALE mesh to the nodes coordinates
  // displacements
  std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
  std::vector<double> mydispnp;

  if (ele->parent_element()->is_ale())
  {
    dispnp = discretization.get_state("dispnp");
    if (dispnp != nullptr)
    {
      mydispnp.resize(lm.size());
      mydispnp = Core::FE::extract_values(*dispnp, lm);
    }
    FOUR_C_ASSERT(mydispnp.size() != 0, "paranoid");
    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        xyze_(idim, inode) += mydispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  // first evaluate the area of the surface element
  params.set<double>("area", 0.0);
  this->area_calculation(ele, params, discretization, lm);

  // get the surface element area
  const double elem_area = params.get<double>("area");

  Core::LinAlg::Matrix<(nsd_), 1> xyzGe(Core::LinAlg::Initialization::zero);

  for (int i = 0; i < nsd_; i++)
  {
    // const IntegrationPoints2D  intpoints(gaussrule);
    for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
    {
      // Computation of the integration factor & shape function at the Gauss point & derivative of
      // the shape function at the Gauss point Computation of the unit normal vector at the Gauss
      // points Computation of nurb specific stuff is not activated here
      Core::FE::eval_shape_func_at_bou_int_point<distype>(funct_, deriv_, fac_, unitnormal_, drs_,
          xsi_, xyze_, intpoints, gpid, nullptr, nullptr, Core::FE::is_nurbs<distype>);

      // global coordinates of gausspoint
      Core::LinAlg::Matrix<(nsd_), 1> coordgp(Core::LinAlg::Initialization::zero);

      // determine coordinates of current Gauss point
      coordgp.multiply(xyze_, funct_);

      // Compute element center of gravity
      xyzGe(i) += intpoints.ip().qwgt[gpid] * coordgp(i) * drs_;

    }  // end Gauss loop
    xyzGe(i) /= elem_area;
  }

  // Get the center of mass of the already calculate surface elements
  std::shared_ptr<std::vector<double>> xyzG =
      params.get<std::shared_ptr<std::vector<double>>>("center of mass");

  std::shared_ptr<std::vector<double>> normal =
      params.get<std::shared_ptr<std::vector<double>>>("normal");

  // Get the area of the of the already calculate surface elements
  double area = params.get<double>("total area");

  for (int i = 0; i < nsd_; i++)
  {
    (*xyzG)[i] = ((*xyzG)[i] * area + xyzGe(i) * elem_area) / (area + elem_area);
    (*normal)[i] = ((*normal)[i] * area + unitnormal_(i) * elem_area) / (area + elem_area);
  }

  // set new center of mass
  params.set("total area", area + elem_area);

}  // Discret::Elements::FluidSurface::center_of_mass_calculation



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::compute_flow_rate(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  // get integration rule
  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // extract local values from the global vectors
  // renamed to "velaf" to be consistent in fluidimplicitintegration.cpp (krank 12/13)
  std::shared_ptr<const Core::LinAlg::Vector<double>> velnp = discretization.get_state("velaf");

  if (velnp == nullptr) FOUR_C_THROW("Cannot get state vector 'velaf'");

  std::vector<double> myvelnp = Core::FE::extract_values(*velnp, lm);

  // allocate velocity vector
  Core::LinAlg::Matrix<nsd_, bdrynen_> evelnp(Core::LinAlg::Initialization::zero);

  // split velocity and pressure, insert into element arrays
  for (int inode = 0; inode < bdrynen_; inode++)
  {
    for (int idim = 0; idim < nsd_; idim++)
    {
      evelnp(idim, inode) = myvelnp[idim + (inode * numdofpernode_)];
    }
  }

  // get node coordinates
  // (we have a nsd_ dimensional domain, since nsd_ determines the dimension of FluidBoundary
  // element!)
  // Core::Geo::fill_initial_position_array<distype,nsd_,Core::LinAlg::SerialDenseMatrix>(ele,xyze_);
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);

  // Add the deformation of the ALE mesh to the nodes coordinates
  // displacements
  std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
  std::vector<double> mydispnp;

  if (ele->parent_element()->is_ale())
  {
    dispnp = discretization.get_state("dispnp");
    if (dispnp != nullptr)
    {
      mydispnp.resize(lm.size());
      mydispnp = Core::FE::extract_values(*dispnp, lm);
    }
    FOUR_C_ASSERT(mydispnp.size() != 0, "paranoid");
    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        xyze_(idim, inode) += mydispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  // const IntegrationPoints2D  intpoints(gaussrule);
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    // Computation of the integration factor & shape function at the Gauss point & derivative of the
    // shape function at the Gauss point Computation of the unit normal vector at the Gauss points
    // Computation of nurb specific stuff is not activated here
    Core::FE::eval_shape_func_at_bou_int_point<distype>(funct_, deriv_, fac_, unitnormal_, drs_,
        xsi_, xyze_, intpoints, gpid, nullptr, nullptr, Core::FE::is_nurbs<distype>);

    // compute flowrate at gauss point
    velint_.multiply(evelnp, funct_);

    // flowrate = uint o normal
    const double flowrate = velint_.dot(unitnormal_);

    // store flowrate at first dof of each node
    // use negative value so that inflow is positive
    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      // see "A better consistency for low order stabilized finite element methods"
      // Jansen, Collis, Whiting, Shakib
      //
      // Here the principle is used to bring the flow rate to the outside world!!
      //
      // funct_ *  velint * n * fac
      //   |      |________________|
      //   |              |
      //   |         flow rate * fac  -> integral over Gamma
      //   |
      // flow rate is distributed to the single nodes of the element
      // = flow rate per node
      //
      // adding up all nodes (ghost elements are handled by the assembling strategy)
      // -> total flow rate at the desired boundary
      //
      // it can be interpreted as a rhs term
      //
      //  ( v , u o n)
      //               Gamma
      //
      elevec1[inode * numdofpernode_] += funct_(inode) * fac_ * flowrate;

      // alternative way:
      //
      //  velint * n * fac
      // |________________|
      //         |
      //    flow rate * fac  -> integral over Gamma
      //     = flow rate per element
      //
      //  adding up all elements (be aware of ghost elements!!)
      //  -> total flow rate at the desired boundary
      //     (is identical to the total flow rate computed above)
    }
  }
}  // Discret::Elements::FluidSurface::compute_flow_rate


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::flow_rate_deriv(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // This function is only implemented for 3D
  if (bdrynsd_ != 2) FOUR_C_THROW("flow_rate_deriv is only implemented for 3D!");

  // get status of Ale
  const bool isale = ele->parent_element()->is_ale();

  std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
  std::vector<double> edispnp;

  if (isale)
  {
    dispnp = discretization.get_state("dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vectors 'dispnp'");
    edispnp.resize(lm.size());
    edispnp = Core::FE::extract_values(*dispnp, lm);
  }

  // get integration rule
  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // order of accuracy of grid velocity determination
  const Teuchos::ParameterList& fdyn = Global::Problem::instance()->fluid_dynamic_params();
  const auto gridvel = Teuchos::getIntegralValue<Inpar::FLUID::Gridvel>(fdyn, "GRIDVEL");

  // normal vector
  Core::LinAlg::Matrix<nsd_, 1> normal(Core::LinAlg::Initialization::zero);

  // get node coordinates
  // (we have a nsd_ dimensional domain, since nsd_ determines the dimension of FluidBoundary
  // element!)
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);

  if (isale)
  {
    FOUR_C_ASSERT(edispnp.size() != 0, "paranoid");

    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        xyze_(idim, inode) += edispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  // get nodal velocities and pressures
  std::shared_ptr<const Core::LinAlg::Vector<double>> convelnp =
      discretization.get_state("convectivevel");

  if (convelnp == nullptr) FOUR_C_THROW("Cannot get state vector 'convectivevel'");

  // extract local values from the global vectors
  std::vector<double> myconvelnp = Core::FE::extract_values(*convelnp, lm);

  // allocate velocities vector
  Core::LinAlg::Matrix<nsd_, bdrynen_> evelnp(Core::LinAlg::Initialization::zero);

  for (int inode = 0; inode < bdrynen_; ++inode)
  {
    for (int idim = 0; idim < nsd_; ++idim)
    {
      evelnp(idim, inode) = myconvelnp[(numdofpernode_ * inode) + idim];
    }
  }


  /*----------------------------------------------------------------------*
    |               start loop over integration points                     |
    *----------------------------------------------------------------------*/
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    // Computation of the integration factor & shape function at the Gauss point & derivative of the
    // shape function at the Gauss point Computation of the unit normal vector at the Gauss points
    // is not activated here Computation of nurb specific stuff is not activated here
    Core::FE::eval_shape_func_at_bou_int_point<distype>(funct_, deriv_, fac_, unitnormal_, drs_,
        xsi_, xyze_, intpoints, gpid, nullptr, nullptr, Core::FE::is_nurbs<distype>);

    // The integration factor is not multiplied with drs
    // since it is the same as the scaling factor for the unit normal
    // Therefore it cancels out!!
    const double fac = intpoints.ip().qwgt[gpid];

    // dxyzdrs vector -> normal which is not normalized
    Core::LinAlg::Matrix<bdrynsd_, nsd_> dxyzdrs(Core::LinAlg::Initialization::uninitialized);
    dxyzdrs.multiply_nt(deriv_, xyze_);
    normal(0, 0) = dxyzdrs(0, 1) * dxyzdrs(1, 2) - dxyzdrs(0, 2) * dxyzdrs(1, 1);
    normal(1, 0) = dxyzdrs(0, 2) * dxyzdrs(1, 0) - dxyzdrs(0, 0) * dxyzdrs(1, 2);
    normal(2, 0) = dxyzdrs(0, 0) * dxyzdrs(1, 1) - dxyzdrs(0, 1) * dxyzdrs(1, 0);

    //-------------------------------------------------------------------
    //  Q
    Core::LinAlg::Matrix<3, 1> u(Core::LinAlg::Initialization::zero);
    for (int dim = 0; dim < 3; ++dim)
      for (int node = 0; node < bdrynen_; ++node) u(dim) += funct_(node) * evelnp(dim, node);

    for (int dim = 0; dim < 3; ++dim) elevec3[0] += u(dim) * normal(dim, 0) * fac;

    if (params.get<bool>("flowrateonly", false) == false)
    {
      //-------------------------------------------------------------------
      // dQ/du
      for (int node = 0; node < bdrynen_; ++node)
      {
        for (int dim = 0; dim < 3; ++dim)
          elevec1[node * numdofpernode_ + dim] += funct_(node) * normal(dim, 0) * fac;
        elevec1[node * numdofpernode_ + 3] = 0.0;
      }

      //-------------------------------------------------------------------
      // dQ/dd

      // determine derivatives of surface normals wrt mesh displacements
      Core::LinAlg::Matrix<3, bdrynen_ * 3> normalderiv(Core::LinAlg::Initialization::zero);

      for (int node = 0; node < bdrynen_; ++node)
      {
        normalderiv(0, 3 * node) = 0.;
        normalderiv(0, 3 * node + 1) =
            deriv_(0, node) * dxyzdrs(1, 2) - deriv_(1, node) * dxyzdrs(0, 2);
        normalderiv(0, 3 * node + 2) =
            deriv_(1, node) * dxyzdrs(0, 1) - deriv_(0, node) * dxyzdrs(1, 1);

        normalderiv(1, 3 * node) =
            deriv_(1, node) * dxyzdrs(0, 2) - deriv_(0, node) * dxyzdrs(1, 2);
        normalderiv(1, 3 * node + 1) = 0.;
        normalderiv(1, 3 * node + 2) =
            deriv_(0, node) * dxyzdrs(1, 0) - deriv_(1, node) * dxyzdrs(0, 0);

        normalderiv(2, 3 * node) =
            deriv_(0, node) * dxyzdrs(1, 1) - deriv_(1, node) * dxyzdrs(0, 1);
        normalderiv(2, 3 * node + 1) =
            deriv_(1, node) * dxyzdrs(0, 0) - deriv_(0, node) * dxyzdrs(1, 0);
        normalderiv(2, 3 * node + 2) = 0.;
      }

      for (int node = 0; node < bdrynen_; ++node)
      {
        for (int dim = 0; dim < 3; ++dim)
          for (int iterdim = 0; iterdim < 3; ++iterdim)
            elevec2[node * numdofpernode_ + dim] +=
                u(iterdim) * normalderiv(iterdim, 3 * node + dim) * fac;
        elevec2[node * numdofpernode_ + 3] = 0.0;
      }

      // consideration of grid velocity
      if (isale)
      {
        // get time step size
        const double dt = params.get<double>("dt", -1.0);
        if (dt < 0.) FOUR_C_THROW("invalid time step size");

        if (gridvel == Inpar::FLUID::BE)  // BE time discretization
        {
          for (int node = 0; node < bdrynen_; ++node)
          {
            for (int dim = 0; dim < 3; ++dim)
              elevec2[node * numdofpernode_ + dim] -=
                  1.0 / dt * funct_(node) * normal(dim, 0) * fac;
          }
        }
        else
          FOUR_C_THROW(
              "flowrate calculation: higher order of accuracy of grid velocity not implemented");
      }

      //-------------------------------------------------------------------
      // (d^2 Q)/(du dd)

      for (int unode = 0; unode < bdrynen_; ++unode)
      {
        for (int udim = 0; udim < numdofpernode_; ++udim)
        {
          for (int nnode = 0; nnode < bdrynen_; ++nnode)
          {
            for (int ndim = 0; ndim < numdofpernode_; ++ndim)
            {
              if (udim == 3 or ndim == 3)
                elemat1(unode * numdofpernode_ + udim, nnode * numdofpernode_ + ndim) = 0.0;
              else
                elemat1(unode * numdofpernode_ + udim, nnode * numdofpernode_ + ndim) =
                    funct_(unode) * normalderiv(udim, 3 * nnode + ndim) * fac;
            }
          }
        }
      }

      //-------------------------------------------------------------------
      // (d^2 Q)/(dd)^2

      // determine second derivatives of surface normals wrt mesh displacements
      std::vector<Core::LinAlg::Matrix<bdrynen_ * 3, bdrynen_ * 3>> normalderiv2(3);

      for (int node1 = 0; node1 < bdrynen_; ++node1)
      {
        for (int node2 = 0; node2 < bdrynen_; ++node2)
        {
          double temp = deriv_(0, node1) * deriv_(1, node2) - deriv_(1, node1) * deriv_(0, node2);

          normalderiv2[0](node1 * 3 + 1, node2 * 3 + 2) = temp;
          normalderiv2[0](node1 * 3 + 2, node2 * 3 + 1) = -temp;

          normalderiv2[1](node1 * 3, node2 * 3 + 2) = -temp;
          normalderiv2[1](node1 * 3 + 2, node2 * 3) = temp;

          normalderiv2[2](node1 * 3, node2 * 3 + 1) = temp;
          normalderiv2[2](node1 * 3 + 1, node2 * 3) = -temp;
        }
      }

      for (int node1 = 0; node1 < bdrynen_; ++node1)
      {
        for (int dim1 = 0; dim1 < numdofpernode_; ++dim1)
        {
          for (int node2 = 0; node2 < bdrynen_; ++node2)
          {
            for (int dim2 = 0; dim2 < numdofpernode_; ++dim2)
            {
              if (dim1 == 3 or dim2 == 3)
                elemat2(node1 * numdofpernode_ + dim1, node2 * numdofpernode_ + dim2) = 0.0;
              else
              {
                for (int iterdim = 0; iterdim < 3; ++iterdim)
                  elemat2(node1 * numdofpernode_ + dim1, node2 * numdofpernode_ + dim2) +=
                      u(iterdim) * normalderiv2[iterdim](node1 * 3 + dim1, node2 * 3 + dim2) * fac;
              }
            }
          }
        }
      }

      // consideration of grid velocity
      if (isale)
      {
        // get time step size
        const double dt = params.get<double>("dt", -1.0);
        if (dt < 0.) FOUR_C_THROW("invalid time step size");

        if (gridvel == Inpar::FLUID::BE)
        {
          for (int node1 = 0; node1 < bdrynen_; ++node1)
          {
            for (int dim1 = 0; dim1 < 3; ++dim1)
            {
              for (int node2 = 0; node2 < bdrynen_; ++node2)
              {
                for (int dim2 = 0; dim2 < 3; ++dim2)
                {
                  elemat2(node1 * numdofpernode_ + dim1, node2 * numdofpernode_ + dim2) -=
                      (1.0 / dt * funct_(node1) * normalderiv(dim1, 3 * node2 + dim2) +
                          1.0 / dt * funct_(node2) * normalderiv(dim2, 3 * node1 + dim1)) *
                      fac;
                }
              }
            }
          }
        }
        else
          FOUR_C_THROW(
              "flowrate calculation: higher order of accuracy of grid velocity not implemented");
      }

      //-------------------------------------------------------------------
    }
  }
}  // Discret::Elements::FluidSurface::flow_rate_deriv


/*----------------------------------------------------------------------*
 |  Impedance related parameters on boundary elements          AC 03/08  |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::impedance_integration(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  const double tfacrhs = fldparatimint_->time_fac_rhs();

  const double pressure = params.get<double>("WindkesselPressure");

  // get Gaussrule
  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // get node coordinates
  // (we have a nsd_ dimensional domain, since nsd_ determines the dimension of FluidBoundary
  // element!)
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);

  // Add the deformation of the ALE mesh to the nodes coordinates
  // displacements
  if (ele->parent_element()->is_ale())
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
    std::vector<double> mydispnp;

    dispnp = discretization.get_state("dispnp");
    if (dispnp != nullptr)
    {
      mydispnp.resize(lm.size());
      mydispnp = Core::FE::extract_values(*dispnp, lm);
    }
    FOUR_C_ASSERT(mydispnp.size() != 0, "paranoid");
    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        xyze_(idim, inode) += mydispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  // add traction in the inward normal direction with norm 'pressure'
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    // Computation of the integration factor & shape function at the Gauss point & derivative of the
    // shape function at the Gauss point Computation of the unit normal vector at the Gauss points
    // Computation of nurb specific stuff is not activated here
    Core::FE::eval_shape_func_at_bou_int_point<distype>(funct_, deriv_, fac_, unitnormal_, drs_,
        xsi_, xyze_, intpoints, gpid, nullptr, nullptr, Core::FE::is_nurbs<distype>);

    const double fac_facrhs_pres = fac_ * tfacrhs * pressure;

    for (int inode = 0; inode < bdrynen_; ++inode)
      for (int idim = 0; idim < nsd_; ++idim)
        elevec1[inode * numdofpernode_ + idim] +=
            fac_facrhs_pres * funct_(inode) * (-unitnormal_(idim));
  }

  return;
}  // Discret::Elements::FluidSurface::impedance_integration


/*---------------------------------------------------------------------------*
 |  linearization of flux w.r.t velocities on boundary elements  Thon 10/15  |
 *---------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::d_qdu(Discret::Elements::FluidBoundary* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  // get Gaussrule
  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // (we have a nsd_ dimensional domain, since nsd_ determines the dimension of FluidBoundary
  // element!)
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);

  // Add the deformation of the ALE mesh to the nodes coordinates
  // displacements
  if (ele->parent_element()->is_ale())
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
    std::vector<double> mydispnp;

    dispnp = discretization.get_state("dispnp");
    if (dispnp != nullptr)
    {
      mydispnp.resize(lm.size());
      mydispnp = Core::FE::extract_values(*dispnp, lm);
    }
    FOUR_C_ASSERT(mydispnp.size() != 0, "paranoid");
    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        xyze_(idim, inode) += mydispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  // compute dQ/du were (dQ/du)_i= (\phi_i \dot n)_Gamma
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    // Computation of the integration factor & shape function at the Gauss point & derivative of the
    // shape function at the Gauss point Computation of the unit normal vector at the Gauss points
    // Computation of nurb specific stuff is not activated here
    Core::FE::eval_shape_func_at_bou_int_point<distype>(funct_, deriv_, fac_, unitnormal_, drs_,
        xsi_, xyze_, intpoints, gpid, nullptr, nullptr, Core::FE::is_nurbs<distype>);

    for (int node = 0; node < bdrynen_; ++node)
    {
      for (int dim = 0; dim < nsd_; ++dim)
      {
        elevec1[node * numdofpernode_ + dim] += funct_(node) * unitnormal_(dim, 0) * fac_;
      }
      elevec1[node * numdofpernode_ + nsd_] = 0.0;
    }
  }

  params.set<double>("tfaclhs", fldparatimint_->time_fac());
}


/*----------------------------------------------------------------------*
 |  get density                                                vg 06/13 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::get_density(
    std::shared_ptr<const Core::Mat::Material> material,
    const Core::LinAlg::Matrix<bdrynen_, 1>& escaaf, const double thermpressaf,
    const Core::LinAlg::Matrix<bdrynen_, 1>& epreaf)
{
  // initially set density and density factor for Neumann boundary conditions to 1.0
  // (the latter only changed for low-Mach-number flow/combustion problems)
  // (This is due to the nature of the Neumann BC's for these problems as they are usually given as
  // h_N=\rho*h.) (Thus in the input file for these problems, h is set which is then scaled by \rho
  // here.)
  //
  //          See Gravemeier, Wall 2011 IJNMF example 4.1.2
  //
  densaf_ = 1.0;
  densfac_ = 1.0;

  if (material->material_type() == Core::Materials::m_fluid)
  {
    const Mat::NewtonianFluid* actmat = static_cast<const Mat::NewtonianFluid*>(material.get());

    // varying density
    if (fldpara_->physical_type() == Inpar::FLUID::varying_density) densaf_ = funct_.dot(escaaf);
    // Boussinesq approximation: Calculation of delta rho
    else if (fldpara_->physical_type() == Inpar::FLUID::boussinesq)
      FOUR_C_THROW("Boussinesq approximation not yet supported for boundary terms!");
    else
      densaf_ = actmat->density();
  }
  else if (material->material_type() == Core::Materials::m_carreauyasuda)
  {
    const Mat::CarreauYasuda* actmat = static_cast<const Mat::CarreauYasuda*>(material.get());

    densaf_ = actmat->density();
  }
  else if (material->material_type() == Core::Materials::m_modpowerlaw)
  {
    const Mat::ModPowerLaw* actmat = static_cast<const Mat::ModPowerLaw*>(material.get());

    densaf_ = actmat->density();
  }
  else if (material->material_type() == Core::Materials::m_herschelbulkley)
  {
    const Mat::HerschelBulkley* actmat = static_cast<const Mat::HerschelBulkley*>(material.get());

    densaf_ = actmat->density();
  }
  else if (material->material_type() == Core::Materials::m_sutherland)
  {
    const Mat::Sutherland* actmat = static_cast<const Mat::Sutherland*>(material.get());

    // compute temperature at n+alpha_F or n+1
    const double tempaf = funct_.dot(escaaf);

    // compute density at n+alpha_F or n+1 based on temperature
    // and thermodynamic pressure
    densaf_ = actmat->compute_density(tempaf, thermpressaf);

    // set density factor for Neumann boundary conditions to density for present material
    densfac_ = densaf_;
  }
  else if (material->material_type() == Core::Materials::m_fluid_linear_density_viscosity)
  {
    const Mat::LinearDensityViscosity* actmat =
        static_cast<const Mat::LinearDensityViscosity*>(material.get());

    // compute pressure at n+alpha_F or n+1
    const double preaf = funct_.dot(epreaf);

    // compute density at n+alpha_F or n+1 based on pressure
    densaf_ = actmat->compute_density(preaf);

    // set density factor for Neumann boundary conditions to density for present material
    densfac_ = densaf_;
  }
  else if (material->material_type() == Core::Materials::m_fluid_murnaghantait)
  {
    const Mat::MurnaghanTaitFluid* actmat =
        static_cast<const Mat::MurnaghanTaitFluid*>(material.get());

    // compute pressure at n+alpha_F or n+1
    const double preaf = funct_.dot(epreaf);

    // compute density at n+alpha_F or n+1 based on pressure
    densaf_ = actmat->compute_density(preaf);

    // set density factor for Neumann boundary conditions to density for present material
    densfac_ = densaf_;
  }
  else if (material->material_type() == Core::Materials::m_fluidporo)
  {
    const Mat::FluidPoro* actmat = static_cast<const Mat::FluidPoro*>(material.get());

    densaf_ = actmat->density();
  }
  else if (material->material_type() == Core::Materials::m_matlist)
  {
    densaf_ = 1.0;
    densfac_ = 1.0;

    // This is only necessary if the BC is dependant on the density!

  }  // end else if m_matlist
  else
    FOUR_C_THROW("Material type is not supported for density evaluation for boundary element!");

  //  // check whether there is zero or negative density
  if (densaf_ < 1e-15) FOUR_C_THROW("zero or negative density!");



  return;
}  // FluidBoundaryImpl::GetDensity

/*----------------------------------------------------------------------*
 |  Evaluating the velocity component of the traction      ismail 05/11 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::calc_traction_velocity_component(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  // extract local values from the global vectors
  std::shared_ptr<const Core::LinAlg::Vector<double>> velnp = discretization.get_state("velaf");

  if (velnp == nullptr) FOUR_C_THROW("Cannot get state vector 'velaf'");

  std::vector<double> myvelnp = Core::FE::extract_values(*velnp, lm);

  // allocate velocity vector
  Core::LinAlg::Matrix<nsd_, bdrynen_> evelnp(Core::LinAlg::Initialization::zero);

  // split velocity and pressure, insert into element arrays
  for (int inode = 0; inode < bdrynen_; inode++)
  {
    for (int idim = 0; idim < nsd_; idim++)
    {
      evelnp(idim, inode) = myvelnp[idim + (inode * numdofpernode_)];
    }
  }


  std::shared_ptr<Core::LinAlg::Vector<double>> cond_velocities =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("condition velocities");
  std::shared_ptr<Core::LinAlg::Map> cond_dofrowmap =
      params.get<std::shared_ptr<Core::LinAlg::Map>>("condition dofrowmap");

  double density = 0.0;  // inverse density of my parent element

  // get material of volume element this surface belongs to
  std::shared_ptr<Core::Mat::Material> mat = ele->parent_element()->material();

  if (mat->material_type() != Core::Materials::m_carreauyasuda &&
      mat->material_type() != Core::Materials::m_modpowerlaw &&
      mat->material_type() != Core::Materials::m_herschelbulkley &&
      mat->material_type() != Core::Materials::m_fluid)
    FOUR_C_THROW("Material law is not a fluid");

  if (mat->material_type() == Core::Materials::m_fluid)
  {
    const Mat::NewtonianFluid* actmat = static_cast<const Mat::NewtonianFluid*>(mat.get());
    density = actmat->density();
  }
  else if (mat->material_type() == Core::Materials::m_carreauyasuda)
  {
    const Mat::CarreauYasuda* actmat = static_cast<const Mat::CarreauYasuda*>(mat.get());
    density = actmat->density();
  }
  else if (mat->material_type() == Core::Materials::m_modpowerlaw)
  {
    const Mat::ModPowerLaw* actmat = static_cast<const Mat::ModPowerLaw*>(mat.get());
    density = actmat->density();
  }
  else if (mat->material_type() == Core::Materials::m_herschelbulkley)
  {
    const Mat::HerschelBulkley* actmat = static_cast<const Mat::HerschelBulkley*>(mat.get());
    density = actmat->density();
  }
  else
    FOUR_C_THROW("Fluid material expected but got type {}", mat->material_type());

  //-------------------------------------------------------------------
  // get the tractions velocity component
  //-------------------------------------------------------------------

  // get Gaussrule
  //  const Core::FE::IntPointsAndWeights<bdrynsd_>
  //  intpoints(Discret::Elements::DisTypeToGaussRuleForExactSol<distype>::rule);
  const Core::FE::IntPointsAndWeights<bdrynsd_> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  // get node coordinates
  // (we have a nsd_ dimensional domain, since nsd_ determines the dimension of FluidBoundary
  // element!)
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, bdrynen_>>(
      ele, xyze_);

  // Add the deformation of the ALE mesh to the nodes coordinates
  // displacements
  std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp;
  std::vector<double> mydispnp;

  if (ele->parent_element()->is_ale())
  {
    dispnp = discretization.get_state("dispnp");
    if (dispnp != nullptr)
    {
      mydispnp.resize(lm.size());
      mydispnp = Core::FE::extract_values(*dispnp, lm);
    }
    FOUR_C_ASSERT(mydispnp.size() != 0, "paranoid");
    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        xyze_(idim, inode) += mydispnp[numdofpernode_ * inode + idim];
      }
    }
  }

  const double timefac = fldparatimint_->time_fac_rhs();

  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    // Computation of the integration factor & shape function at the Gauss point & derivative of the
    // shape function at the Gauss point Computation of the unit normal vector at the Gauss points
    // Computation of nurb specific stuff is not activated here
    Core::FE::eval_shape_func_at_bou_int_point<distype>(funct_, deriv_, fac_, unitnormal_, drs_,
        xsi_, xyze_, intpoints, gpid, nullptr, nullptr, Core::FE::is_nurbs<distype>);

    // Get the velocity value at the corresponding Gauss point.
    std::vector<double> vel_gps(nsd_, 0.0);
    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        vel_gps[idim] += myvelnp[inode * numdofpernode_ + idim] * funct_(inode);
      }
    }

    // Evaluate the normal velocity at the corresponding Gauss point
    double n_vel = 0.0;
    for (int idim = 0; idim < nsd_; ++idim)
    {
      n_vel += vel_gps[idim] * (unitnormal_(idim));
    }
    // loop over all node and add the corresponding effect of the Neumann-Inflow condition
    for (int inode = 0; inode < bdrynen_; ++inode)
    {
      for (int idim = 0; idim < nsd_; ++idim)
      {
        // evaluate the value of the Un.U at the corresponding Gauss point
        const double uV = n_vel * vel_gps[idim] * density;
        const double fac_thsl_pres_inve = fac_ * timefac * uV;

        // remove the Neumann-inflow contribution only if the normal velocity is an inflow velocity
        // i.e n_vel < 0
        if (n_vel < 0.0)
        {
          elevec1[inode * numdofpernode_ + idim] -= fac_thsl_pres_inve * funct_(inode);
        }
      }
      //      double radius = sqrt(pow(xyze_(0,inode),2.0)+pow(xyze_(1,inode),2.0));
      //      cout<<"n_vel("<<n_vel<<") vel: "<<n_vel<<" rad: "<<radius<<endl;
    }
  }
  return;
}

template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryImpl<distype>::compute_neumann_uv_integral(
    Discret::Elements::FluidBoundary* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseVector& elevec1)
{
}
// Discret::Elements::FluidSurface::compute_neumann_uv_integral

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes
template class Discret::Elements::FluidBoundaryImpl<Core::FE::CellType::quad4>;
template class Discret::Elements::FluidBoundaryImpl<Core::FE::CellType::quad8>;
template class Discret::Elements::FluidBoundaryImpl<Core::FE::CellType::quad9>;
template class Discret::Elements::FluidBoundaryImpl<Core::FE::CellType::tri3>;
template class Discret::Elements::FluidBoundaryImpl<Core::FE::CellType::tri6>;
template class Discret::Elements::FluidBoundaryImpl<Core::FE::CellType::line2>;
template class Discret::Elements::FluidBoundaryImpl<Core::FE::CellType::line3>;
template class Discret::Elements::FluidBoundaryImpl<Core::FE::CellType::nurbs2>;
template class Discret::Elements::FluidBoundaryImpl<Core::FE::CellType::nurbs3>;
template class Discret::Elements::FluidBoundaryImpl<Core::FE::CellType::nurbs4>;
template class Discret::Elements::FluidBoundaryImpl<Core::FE::CellType::nurbs9>;

FOUR_C_NAMESPACE_CLOSE
