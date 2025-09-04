// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_boundary_parent_calc.hpp"

#include "4C_fem_general_element_integration_select.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fluid_ele.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_mat_carreauyasuda.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_herschelbulkley.hpp"
#include "4C_mat_modpowerlaw.hpp"
#include "4C_mat_newtonianfluid.hpp"
#include "4C_mat_sutherland.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::FluidBoundaryParentInterface*
Discret::Elements::FluidBoundaryParentInterface::impl(Core::Elements::FaceElement* ele)
{
  switch (ele->shape())
  {
    case Core::FE::CellType::line2:
    {
      return FluidBoundaryParent<Core::FE::CellType::line2>::instance(
          Core::Utils::SingletonAction::create);
    }
    /*case Core::FE::CellType::line3:
    {
      return FluidBoundaryParent<Core::FE::CellType::line3>::instance();
    }*/
    case Core::FE::CellType::tri3:
    {
      return FluidBoundaryParent<Core::FE::CellType::tri3>::instance();
    }
    case Core::FE::CellType::tri6:
    {
      return FluidBoundaryParent<Core::FE::CellType::tri6>::instance();
    }
    case Core::FE::CellType::quad4:
    {
      return FluidBoundaryParent<Core::FE::CellType::quad4>::instance(
          Core::Utils::SingletonAction::create);
    }
    case Core::FE::CellType::quad8:
    {
      return FluidBoundaryParent<Core::FE::CellType::quad8>::instance();
    }
    case Core::FE::CellType::quad9:
    {
      return FluidBoundaryParent<Core::FE::CellType::quad9>::instance();
    }
    /*case Core::FE::CellType::nurbs2:    // 1D nurbs boundary element
    {
      return FluidBoundaryParent<Core::FE::CellType::nurbs2>::instance();
    }
    case Core::FE::CellType::nurbs3:    // 1D nurbs boundary element
    {
      return FluidBoundaryParent<Core::FE::CellType::nurbs3>::instance();
    }
    case Core::FE::CellType::nurbs4:    // 2D nurbs boundary element
    {
      return FluidBoundaryParent<Core::FE::CellType::nurbs4>::instance();
    }
    case Core::FE::CellType::nurbs9:    // 2D nurbs boundary element
    {
      return FluidBoundaryParent<Core::FE::CellType::nurbs9>::instance();
    }*/
    default:
      FOUR_C_THROW(
          "Element shape {} ({} nodes) not activated for boundary conditions requiring "
          "parent-element evaluations. Just do it.",
          ele->shape(), ele->num_node());
      break;
  }
  return nullptr;
}

template <Core::FE::CellType distype>
Discret::Elements::FluidBoundaryParentInterface*
Discret::Elements::FluidBoundaryParent<distype>::instance(Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<Discret::Elements::FluidBoundaryParent<distype>>(
            new Discret::Elements::FluidBoundaryParent<distype>());
      });

  return singleton_owner.instance(action);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::FluidBoundaryParent<distype>::FluidBoundaryParent()
    : Discret::Elements::FluidBoundaryParentInterface(),
      drs_(0.0),
      fac_(0.0),
      visc_(0.0),
      densaf_(1.0)
{
  // pointer to class FluidParentParameter (access to the general parameter)
  fldpara_ = Discret::Elements::FluidEleParameterStd::instance();
  fldparatimint_ = Discret::Elements::FluidEleParameterTimInt::instance();

  return;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryParent<distype>::flow_dep_pressure_bc(
    Discret::Elements::FluidBoundary* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat, Core::LinAlg::SerialDenseVector::Base& elevec)
{
  switch (surfele->shape())
  {
    // 2D:
    case Core::FE::CellType::line2:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::quad4)
      {
        flow_dep_pressure_bc<Core::FE::CellType::line2, Core::FE::CellType::quad4>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination line2/quad4 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::tri3:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::tet4)
      {
        flow_dep_pressure_bc<Core::FE::CellType::tri3, Core::FE::CellType::tet4>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination tri3/tet4 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::quad4:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex8)
      {
        flow_dep_pressure_bc<Core::FE::CellType::quad4, Core::FE::CellType::hex8>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination quad4/hex8 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::quad8:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex20)
      {
        flow_dep_pressure_bc<Core::FE::CellType::quad8, Core::FE::CellType::hex20>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination quad8/hex20 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::quad9:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex27)
      {
        flow_dep_pressure_bc<Core::FE::CellType::quad9, Core::FE::CellType::hex27>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination quad9/hex27 for surface/parent pair");
      break;
    }
    default:
    {
      FOUR_C_THROW("not implemented yet\n");
      break;
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryParent<distype>::slip_supp_bc(
    Discret::Elements::FluidBoundary* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat, Core::LinAlg::SerialDenseVector::Base& elevec)
{
  switch (surfele->shape())
  {
    // 2D:
    case Core::FE::CellType::line2:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::quad4)
      {
        slip_supp_bc<Core::FE::CellType::line2, Core::FE::CellType::quad4>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination line2/quad4 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::quad4:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex8)
      {
        slip_supp_bc<Core::FE::CellType::quad4, Core::FE::CellType::hex8>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination quad4/hex8 for surface/parent pair");
      break;
    }
    default:
    {
      FOUR_C_THROW("not implemented yet\n");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryParent<distype>::navier_slip_bc(
    Discret::Elements::FluidBoundary* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat, Core::LinAlg::SerialDenseVector::Base& elevec)
{
  switch (surfele->shape())
  {
    // 2D:
    case Core::FE::CellType::line2:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::quad4)
      {
        navier_slip_bc<Core::FE::CellType::line2, Core::FE::CellType::quad4>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination line2/quad4 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::quad4:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex8)
      {
        navier_slip_bc<Core::FE::CellType::quad4, Core::FE::CellType::hex8>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination quad4/hex8 for surface/parent pair");
      break;
    }
    default:
    {
      FOUR_C_THROW("not implemented yet\n");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryParent<distype>::evaluate_weak_dbc(
    Discret::Elements::FluidBoundary* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat, Core::LinAlg::SerialDenseVector::Base& elevec)
{
  switch (surfele->shape())
  {
    // 2D:
    case Core::FE::CellType::line2:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::quad4)
      {
        evaluate_weak_dbc<Core::FE::CellType::line2, Core::FE::CellType::quad4>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination line2/quad4 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::tri3:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::tet4)
      {
        evaluate_weak_dbc<Core::FE::CellType::tri3, Core::FE::CellType::tet4>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination tri3/tet4 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::quad4:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex8)
      {
        evaluate_weak_dbc<Core::FE::CellType::quad4, Core::FE::CellType::hex8>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination quad4/hex8 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::quad8:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex20)
      {
        evaluate_weak_dbc<Core::FE::CellType::quad8, Core::FE::CellType::hex20>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination quad8/hex20 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::quad9:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex27)
      {
        evaluate_weak_dbc<Core::FE::CellType::quad9, Core::FE::CellType::hex27>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination quad9/hex27 for surface/parent pair");
      break;
    }
    default:
    {
      FOUR_C_THROW("not implemented yet\n");
      break;
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryParent<distype>::estimate_nitsche_trace_max_eigenvalue(
    Core::Elements::FaceElement* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2)
{
  switch (surfele->shape())
  {
    // 2D:
    case Core::FE::CellType::line2:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::quad4)
      {
        estimate_nitsche_trace_max_eigenvalue<Core::FE::CellType::line2, Core::FE::CellType::quad4>(
            surfele, params, discretization, lm, elemat1, elemat2);
      }
      else
        FOUR_C_THROW("expected combination line2/quad4 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::tri3:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::tet4)
      {
        estimate_nitsche_trace_max_eigenvalue<Core::FE::CellType::tri3, Core::FE::CellType::tet4>(
            surfele, params, discretization, lm, elemat1, elemat2);
      }
      else
        FOUR_C_THROW("expected combination tri3/tet4 for surface/parent pair");
      break;
    }
    case Core::FE::CellType::tri6:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::tet10)
      {
        estimate_nitsche_trace_max_eigenvalue<Core::FE::CellType::tri6, Core::FE::CellType::tet10>(
            surfele, params, discretization, lm, elemat1, elemat2);
      }
      else
        FOUR_C_THROW("expected combination tri6/tet10 for surface/parent pair");
      break;
    }
    case Core::FE::CellType::quad4:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex8)
      {
        estimate_nitsche_trace_max_eigenvalue<Core::FE::CellType::quad4, Core::FE::CellType::hex8>(
            surfele, params, discretization, lm, elemat1, elemat2);
      }
      else
        FOUR_C_THROW("expected combination quad4/hex8 for surface/parent pair");
      break;
    }
    case Core::FE::CellType::quad8:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex20)
      {
        estimate_nitsche_trace_max_eigenvalue<Core::FE::CellType::quad8, Core::FE::CellType::hex20>(
            surfele, params, discretization, lm, elemat1, elemat2);
      }
      else
        FOUR_C_THROW("expected combination quad8/hex20 for surface/parent pair");
      break;
    }
    case Core::FE::CellType::quad9:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex27)
      {
        estimate_nitsche_trace_max_eigenvalue<Core::FE::CellType::quad9, Core::FE::CellType::hex27>(
            surfele, params, discretization, lm, elemat1, elemat2);
      }
      else
        FOUR_C_THROW("expected combination quad9/hex27 for surface/parent pair");
      break;
    }
    default:
    {
      FOUR_C_THROW("not implemented yet\n");
      break;
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryParent<distype>::mix_hyb_dirichlet(
    Discret::Elements::FluidBoundary* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat, Core::LinAlg::SerialDenseVector::Base& elevec)
{
  switch (surfele->shape())
  {
    // 2D:
    case Core::FE::CellType::line2:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::quad4)
      {
        mix_hyb_dirichlet<Core::FE::CellType::line2, Core::FE::CellType::quad4>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination line2/quad4 for surface/parent pair");
      break;
    }
    // 3D:
    case Core::FE::CellType::quad4:
    {
      if (surfele->parent_element()->shape() == Core::FE::CellType::hex8)
      {
        mix_hyb_dirichlet<Core::FE::CellType::quad4, Core::FE::CellType::hex8>(
            surfele, params, discretization, lm, elemat, elevec);
      }
      else
        FOUR_C_THROW("expected combination quad4/hex8 for surface/parent pair");
      break;
    }
    default:
    {
      FOUR_C_THROW("not implemented yet\n");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
template <Core::FE::CellType bdistype, Core::FE::CellType pdistype>
void Discret::Elements::FluidBoundaryParent<distype>::flow_dep_pressure_bc(
    Discret::Elements::FluidBoundary* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& plm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseVector::Base& elevec1)
{
  // initialize pressure value and pressure derivative at boundary
  double pressure = 0.0;
  double pressder = 0.0;

  //---------------------------------------------------------------------
  // get condition information
  //---------------------------------------------------------------------
  const auto* fdp_cond = params.get<const Core::Conditions::Condition*>("condition");

  // find out whether there is a time curve and get factor:
  // usable as a time-curve factor for fixed pressure as well as
  // for switching off any flow-dependent pressure condition in case
  // time-curve factor being zero
  // (time curve at n+1 applied for all time-integration schemes, but
  //  variable time_ in fluid3_parameter is n+alpha_F in case of
  //  generalized-alpha time-integration scheme -> reset to time n+1)
  const double time =
      fldparatimint_->time() + (1 - fldparatimint_->alpha_f()) * fldparatimint_->dt();
  const auto curvenum = fdp_cond->parameters().get<std::optional<int>>("curve");

  double curvefac = 1.0;
  if (curvenum.has_value() && curvenum.value() > 0 && time >= 0)
    curvefac = Global::Problem::instance()
                   ->function_by_id<Core::Utils::FunctionOfTime>(curvenum.value())
                   .evaluate(time);

  // (temporarily) switch off any flow-dependent pressure condition in case of zero
  // time-curve factor
  if (curvefac > 0.0)
  {
    // decide on whether it is a flow-rate- or flow-volume-based condition
    const std::string& condtype =
        (*fdp_cond).parameters().get<std::string>("TYPE_OF_FLOW_DEPENDENCE");

    // flow-rate-based condition
    if (condtype == "flow_rate")
    {
      // get flow rate in this case
      const double flowrate = params.get<double>("flow rate");

      // get constant and linear coefficient for linear flow rate - pressure relation
      // and compute pressure accordingly
      const double const_coeff = fdp_cond->parameters().get<double>("ConstCoeff");
      const double lin_coeff = fdp_cond->parameters().get<double>("LinCoeff");
      pressure = const_coeff + lin_coeff * flowrate;
      pressder = lin_coeff;
    }
    // flow-volume-based condition
    else if (condtype == "flow_volume")
    {
      // get flow volume in this case
      const double flow_volume = params.get<double>("flow volume");

      // get initial volume, reference pressure and adiabatic exponent
      const double vol0 = fdp_cond->parameters().get<double>("InitialVolume");
      const double ref_pre = fdp_cond->parameters().get<double>("ReferencePressure");
      const double kappa = fdp_cond->parameters().get<double>("AdiabaticExponent");

      // compute rise in pressure due to volume reduction at boundary
      pressure = ref_pre * pow((vol0 / (vol0 - flow_volume)), kappa);

      // subtract reference pressure for usual case of zero ambient pressure
      pressure -= ref_pre;
    }
    // fixed-pressure condition (with potential time curve)
    else if (condtype == "fixed_pressure")
    {
      pressure = fdp_cond->parameters().get<double>("ConstCoeff") * curvefac;
    }
    else
      FOUR_C_THROW("Unknown type of flow-dependent pressure condition: {}", condtype);

    // get thermodynamic pressure at n+1/n+alpha_F
    const double thermpressaf = params.get<double>("thermpress at n+alpha_F/n+1", 1.0);

    //---------------------------------------------------------------------
    // get time-integration parameters and flag for linearization scheme
    //---------------------------------------------------------------------
    const double timefac = fldparatimint_->time_fac();
    const double timefacrhs = fldparatimint_->time_fac_rhs();

    bool is_newton = fldpara_->is_newton();

    //---------------------------------------------------------------------
    // get parent element data
    //---------------------------------------------------------------------
    Core::Elements::Element* parent = surfele->parent_element();

    // parent element id
    int pid = parent->id();

    // number of parent spatial dimensions
    static const int nsd = Core::FE::dim<pdistype>;

    // number of parent element nodes
    static const int piel = Core::FE::num_nodes(pdistype);

    // reshape element matrices and vectors and init to zero, construct views
    const int peledim = (nsd + 1) * piel;
    elemat1.shape(peledim, peledim);
    elevec1.size(peledim);
    Core::LinAlg::Matrix<peledim, peledim> elemat(elemat1.values(), true);
    Core::LinAlg::Matrix<peledim, 1> elevec(elevec1.values(), true);

    // get local node coordinates
    Core::LinAlg::Matrix<nsd, piel> pxyze(Core::LinAlg::Initialization::zero);
    Core::Geo::fill_initial_position_array<pdistype, nsd, Core::LinAlg::Matrix<nsd, piel>>(
        parent, pxyze);

    // get Gaussian integration points
    const Core::FE::IntPointsAndWeights<nsd> pintpoints(
        Discret::Elements::DisTypeToOptGaussRule<pdistype>::rule);

    //---------------------------------------------------------------------
    // get boundary element data
    //---------------------------------------------------------------------
    // local surface id
    int bid = surfele->surface_number();

    // number of boundary spatial dimensions
    static const int bnsd = Core::FE::dim<bdistype>;

    // number of boundary element nodes
    static const int biel = Core::FE::num_nodes(bdistype);

    // get local node coordinates
    Core::LinAlg::Matrix<nsd, biel> bxyze(Core::LinAlg::Initialization::zero);
    Core::Geo::fill_initial_position_array<bdistype, nsd, Core::LinAlg::Matrix<nsd, biel>>(
        surfele, bxyze);

    // get Gaussian integration points
    const Core::FE::IntPointsAndWeights<bnsd> bintpoints(
        Discret::Elements::DisTypeToOptGaussRule<bdistype>::rule);

    // get location vector and ownerships for boundary element
    std::vector<int> blm;
    std::vector<int> blmowner;
    std::vector<int> blmstride;
    surfele->Core::Elements::Element::location_vector(discretization, blm, blmowner, blmstride);

    //---------------------------------------------------------------------
    // map Gaussian integration points to parent element for one-sided
    // derivatives on boundary
    //---------------------------------------------------------------------
    // coordinates of current integration point
    Core::LinAlg::SerialDenseMatrix gps(bintpoints.ip().nquad, bnsd);
    for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
    {
      const double* gpcoord = (bintpoints.ip().qxg)[iquad];
      for (int idim = 0; idim < bnsd; idim++)
      {
        gps(iquad, idim) = gpcoord[idim];
      }
    }

    // distinguish 2- and 3-D case
    Core::LinAlg::SerialDenseMatrix pqxg(pintpoints.ip().nquad, nsd);
    if (nsd == 2)
      Core::FE::boundary_gp_to_parent_gp2(pqxg, gps, pdistype, bdistype, bid);
    else if (nsd == 3)
      Core::FE::boundary_gp_to_parent_gp3(pqxg, gps, pdistype, bdistype, bid);

    //---------------------------------------------------------------------
    // extract parent and boundary values from global distributed vectors
    //---------------------------------------------------------------------
    // parent velocity at n+alpha_F
    std::shared_ptr<const Core::LinAlg::Vector<double>> velaf = discretization.get_state("velaf");
    std::shared_ptr<const Core::LinAlg::Vector<double>> scaaf = discretization.get_state("scaaf");
    if (velaf == nullptr or scaaf == nullptr)
      FOUR_C_THROW("Cannot get state vector 'velaf' and/or 'scaaf'");

    std::vector<double> mypvelaf(plm.size());
    std::vector<double> mypscaaf(plm.size());
    mypvelaf = Core::FE::extract_values(*velaf, plm);
    mypscaaf = Core::FE::extract_values(*scaaf, plm);

    Core::LinAlg::Matrix<nsd, piel> pevelaf(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<piel, 1> pescaaf(Core::LinAlg::Initialization::zero);
    for (int inode = 0; inode < piel; ++inode)
    {
      for (int idim = 0; idim < nsd; ++idim)
      {
        pevelaf(idim, inode) = mypvelaf[(nsd + 1) * inode + idim];
      }
      pescaaf(inode) = mypscaaf[(nsd + 1) * inode + nsd];
    }

    // parent and boundary displacement at n+1
    std::vector<double> mypedispnp((plm).size());
    std::vector<double> mybedispnp((blm).size());
    if (surfele->parent_element()->is_ale())
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("dispnp");
      if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

      mypedispnp = Core::FE::extract_values(*dispnp, plm);
      mybedispnp = Core::FE::extract_values(*dispnp, blm);

      // add parent and boundary displacement at n+1
      for (int idim = 0; idim < nsd; ++idim)
      {
        for (int pnode = 0; pnode < piel; ++pnode)
        {
          pxyze(idim, pnode) += mypedispnp[(nsd + 1) * pnode + idim];
        }
        for (int bnode = 0; bnode < biel; ++bnode)
        {
          bxyze(idim, bnode) += mybedispnp[(nsd + 1) * bnode + idim];
        }
      }
    }

    //---------------------------------------------------------------------
    // definitions and initializations for parent and boundary element
    //---------------------------------------------------------------------
    Core::LinAlg::Matrix<nsd, 1> pxsi(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<piel, 1> pfunct(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd, piel> pderiv(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd, nsd> pxjm(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd, nsd> pxji(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd, 1> unitnormal(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd, piel> pderxy(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd, 1> pvelaf(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd, nsd> pvderxyaf(Core::LinAlg::Initialization::zero);

    Core::LinAlg::Matrix<bnsd, 1> xsi(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<biel, 1> funct(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<bnsd, biel> deriv(Core::LinAlg::Initialization::zero);

    //---------------------------------------------------------------------
    // integration loop
    //---------------------------------------------------------------------
    for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
    {
      // coordinates of integration point w.r.t. parent element
      for (int idim = 0; idim < nsd; idim++)
      {
        pxsi(idim) = pqxg(iquad, idim);
      }

      // coordinates of integration point w.r.t. boundary element
      for (int idim = 0; idim < bnsd; idim++)
      {
        xsi(idim) = gps(iquad, idim);
      }

      // shape functions and derivatives of parent element at integration point
      Core::FE::shape_function<pdistype>(pxsi, pfunct);
      Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

      // shape functions and derivatives of boundary element at integration point
      Core::FE::shape_function<bdistype>(xsi, funct);
      Core::FE::shape_function_deriv1<bdistype>(xsi, deriv);

      // compute (inverse of) Jacobian matrix and determinant for parent element
      // and check its value
      pxjm.multiply_nt(pderiv, pxyze);
      const double pdet = pxji.invert(pxjm);
      if (pdet < 1E-16)
        FOUR_C_THROW("GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", pid, pdet);

      // compute measure tensor, infinitesimal area and outward unit normal
      // for boundary element
      drs_ = 0.0;
      Core::LinAlg::Matrix<bnsd, bnsd> metrictensor(Core::LinAlg::Initialization::zero);
      Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
          bxyze, deriv, metrictensor, drs_, &unitnormal);

      // compute integration factor for boundary element
      fac_ = bintpoints.ip().qwgt[iquad] * drs_;

      // compute velocity vector and normal velocity at integration point
      // (here via parent element)
      double normvel = 0.0;
      pvelaf.multiply(pevelaf, pfunct);
      normvel = pvelaf.dot(unitnormal);

      // compute global first derivatives for parent element
      pderxy.multiply(pxji, pderiv);

      // get velocity derivatives at n+alpha_F at integration point
      pvderxyaf.multiply_nt(pevelaf, pderxy);

      // evaluate material at integration point
      double rateofstrain = 0.0;
      std::shared_ptr<Core::Mat::Material> material = parent->material();

      // compute measure for rate of strain at n+alpha_F or n+1 if required
      // for non-Newtonian fluid
      if (material->material_type() == Core::Materials::m_carreauyasuda or
          material->material_type() == Core::Materials::m_modpowerlaw or
          material->material_type() == Core::Materials::m_herschelbulkley)
      {
        //-------------------------------------------------------------------
        //
        //          +-                                 -+ 1
        //          |          /   \           /   \    | -
        //          | 2 * eps | vel |   * eps | vel |   | 2
        //          |          \   / ij        \   / ij |
        //          +-                                 -+
        //
        //-------------------------------------------------------------------
        // compute (two times) strain-rate tensor
        Core::LinAlg::Matrix<nsd, nsd> two_epsilon;
        for (int rr = 0; rr < nsd; ++rr)
        {
          for (int mm = 0; mm < nsd; ++mm)
          {
            two_epsilon(rr, mm) = pvderxyaf(rr, mm) + pvderxyaf(mm, rr);
          }
        }

        // compute product of (two times) strain-rate tensor with itself
        for (int rr = 0; rr < nsd; ++rr)
        {
          for (int mm = 0; mm < nsd; ++mm)
          {
            rateofstrain += two_epsilon(rr, mm) * two_epsilon(mm, rr);
          }
        }

        // compute square-root of (two times) product
        rateofstrain = sqrt(rateofstrain / 2.0);
      }

      // computations and settings for low-Mach-number flow:
      // 1) compute scalar value at this integration point
      //    and check whether it is positive
      // 2) compute divergence of velocity
      // 3) set pre-factor 1/3 (zero for incompressible flow)
      double pscaaf = 0.0;
      double pvdiv = 0.0;
      double prefac = 0.0;
      if (fldpara_->physical_type() == Inpar::FLUID::loma)
      {
        pscaaf = pfunct.dot(pescaaf);
        if (pscaaf < 0.0)
          FOUR_C_THROW("Negative scalar in boundary computation for low-Mach-number flow!");

        // compute divergence of velocity from previous iteration
        for (int idim = 0; idim < nsd; ++idim)
        {
          pvdiv += pvderxyaf(idim, idim);
        }

        // set pre-factor 1/3
        prefac = 1.0 / 3.0;
      }

      // get density and viscosity at integration point
      get_density_and_viscosity(material, pscaaf, thermpressaf, rateofstrain);

      //---------------------------------------------------------------------
      // contributions to element matrix and element vector
      // (distinguish two- and three-dimensional case)
      //---------------------------------------------------------------------
      if (nsd == 3)
      {
        // Four potential contributions to element matrix on left-hand side:
        // 1) pressure term
        //   (only active if there is a dependence on flow rate and thus velocity,
        //    further non-linear dependence on velocity, e.g., in case of quadratic
        //    dependence of pressure on flow rate, not yet considered by including
        //    additional term for Newton linearization)
        /*
        //             /                           \
        //            |                             |
        //          + |  v , n * pressder * n * Du  |
        //            |                             |
        //             \                           / boundaryele
        //
        */
        const double timefacfacnpredern = timefac * fac_ * 1.0 * pressder;

        for (int ui = 0; ui < piel; ++ui)
        {
          for (int vi = 0; vi < piel; ++vi)
          {
            const double temp = timefacfacnpredern * pfunct(ui) * pfunct(vi);
            elemat(vi * 4, ui * 4) += temp;
            elemat(vi * 4 + 1, ui * 4 + 1) += temp;
            elemat(vi * 4 + 2, ui * 4 + 2) += temp;
          }
        }

        // 2) viscous term
        /*
        //    /                                    \
        //   |            s                         |
        // - |  v , (nabla  Du - (1/3) * div u) * n |
        //   |                                      |
        //    \                                    / boundaryele
        //
        */
        const double timefacmu = timefac * fac_ * 2.0 * visc_;

        for (int ui = 0; ui < piel; ++ui)
        {
          double nabla_u_o_n_lin[3][3];
          nabla_u_o_n_lin[0][0] = timefacmu * ((1.0 - prefac) * pderxy(0, ui) * unitnormal(0) +
                                                  0.5 * pderxy(1, ui) * unitnormal(1) +
                                                  0.5 * pderxy(2, ui) * unitnormal(2));
          nabla_u_o_n_lin[0][1] = timefacmu * (-prefac * pderxy(1, ui) * unitnormal(0) +
                                                  0.5 * pderxy(0, ui) * unitnormal(1));
          nabla_u_o_n_lin[0][2] = timefacmu * (-prefac * pderxy(2, ui) * unitnormal(0) +
                                                  0.5 * pderxy(0, ui) * unitnormal(2));

          nabla_u_o_n_lin[1][0] = timefacmu * (0.5 * pderxy(1, ui) * unitnormal(0) -
                                                  prefac * pderxy(0, ui) * unitnormal(1));
          nabla_u_o_n_lin[1][1] = timefacmu * (0.5 * pderxy(0, ui) * unitnormal(0) +
                                                  (1.0 - prefac) * pderxy(1, ui) * unitnormal(1) +
                                                  0.5 * pderxy(2, ui) * unitnormal(2));
          nabla_u_o_n_lin[1][2] = timefacmu * (-prefac * pderxy(2, ui) * unitnormal(1) +
                                                  0.5 * pderxy(1, ui) * unitnormal(2));

          nabla_u_o_n_lin[2][0] = timefacmu * (0.5 * pderxy(2, ui) * unitnormal(0) -
                                                  prefac * pderxy(0, ui) * unitnormal(2));
          nabla_u_o_n_lin[2][1] = timefacmu * (0.5 * pderxy(2, ui) * unitnormal(1) -
                                                  prefac * pderxy(1, ui) * unitnormal(2));
          nabla_u_o_n_lin[2][2] = timefacmu * (0.5 * pderxy(0, ui) * unitnormal(0) +
                                                  0.5 * pderxy(1, ui) * unitnormal(1) +
                                                  (1.0 - prefac) * pderxy(2, ui) * unitnormal(2));

          for (int vi = 0; vi < piel; ++vi)
          {
            elemat(vi * 4, ui * 4) -= pfunct(vi) * nabla_u_o_n_lin[0][0];
            elemat(vi * 4, ui * 4 + 1) -= pfunct(vi) * nabla_u_o_n_lin[0][1];
            elemat(vi * 4, ui * 4 + 2) -= pfunct(vi) * nabla_u_o_n_lin[0][2];

            elemat(vi * 4 + 1, ui * 4) -= pfunct(vi) * nabla_u_o_n_lin[1][0];
            elemat(vi * 4 + 1, ui * 4 + 1) -= pfunct(vi) * nabla_u_o_n_lin[1][1];
            elemat(vi * 4 + 1, ui * 4 + 2) -= pfunct(vi) * nabla_u_o_n_lin[1][2];

            elemat(vi * 4 + 2, ui * 4) -= pfunct(vi) * nabla_u_o_n_lin[2][0];
            elemat(vi * 4 + 2, ui * 4 + 1) -= pfunct(vi) * nabla_u_o_n_lin[2][1];
            elemat(vi * 4 + 2, ui * 4 + 2) -= pfunct(vi) * nabla_u_o_n_lin[2][2];
          }
        }

        // check normal velocity -> further computation only required for
        // negative normal velocity, that is, inflow at this boundary
        if (normvel < -0.0001)
        {
          // 3) convective term
          /*
          //             /                        \
          //            |                          |
          //          - |  v , rho * Du ( u o n )  |
          //            |                          |
          //             \                        / boundaryele
          //
          */
          const double timefacfacdensnormvel = timefac * fac_ * densaf_ * normvel;

          for (int ui = 0; ui < piel; ++ui)
          {
            for (int vi = 0; vi < piel; ++vi)
            {
              const double temp = timefacfacdensnormvel * pfunct(ui) * pfunct(vi);
              elemat(vi * 4, ui * 4) -= temp;
              elemat(vi * 4 + 1, ui * 4 + 1) -= temp;
              elemat(vi * 4 + 2, ui * 4 + 2) -= temp;
            }
          }

          // 4) additional convective term for Newton linearization
          if (is_newton)
          {
            /*
            //             /                        \
            //            |                          |
            //          - |  v , rho * u ( Du o n )  |
            //            |                          |
            //             \                        / boundaryele
            //
            */
            const double timefacfacdens = timefac * fac_ * densaf_;

            // dyadic product of unit normal vector and velocity vector
            Core::LinAlg::Matrix<nsd, nsd> n_x_u(Core::LinAlg::Initialization::zero);
            n_x_u.multiply_nt(pvelaf, unitnormal);

            for (int ui = 0; ui < piel; ++ui)
            {
              for (int vi = 0; vi < piel; ++vi)
              {
                const double temp = timefacfacdens * pfunct(ui) * pfunct(vi);

                elemat(vi * 4, ui * 4) -= temp * n_x_u(0, 0);
                elemat(vi * 4, ui * 4 + 1) -= temp * n_x_u(0, 1);
                elemat(vi * 4, ui * 4 + 2) -= temp * n_x_u(0, 2);

                elemat(vi * 4 + 1, ui * 4) -= temp * n_x_u(1, 0);
                elemat(vi * 4 + 1, ui * 4 + 1) -= temp * n_x_u(1, 1);
                elemat(vi * 4 + 1, ui * 4 + 2) -= temp * n_x_u(1, 2);

                elemat(vi * 4 + 2, ui * 4) -= temp * n_x_u(2, 0);
                elemat(vi * 4 + 2, ui * 4 + 1) -= temp * n_x_u(2, 1);
                elemat(vi * 4 + 2, ui * 4 + 2) -= temp * n_x_u(2, 2);
              }
            }
          }

          // convective contribution to element vector on right-hand side:
          /*
          //    /                       \
          //   |                         |
          // - |  v , rho * u ( u o n )  |
          //   |                         |
          //    \                       / boundaryele
          //
          */
          const double timefacconvrhs = timefacrhs * fac_ * densaf_ * normvel;

          for (int vi = 0; vi < piel; ++vi)
          {
            elevec(vi * 4) -= pfunct(vi) * timefacconvrhs * pvelaf(0);
            elevec(vi * 4 + 1) -= pfunct(vi) * timefacconvrhs * pvelaf(1);
            elevec(vi * 4 + 2) -= pfunct(vi) * timefacconvrhs * pvelaf(2);
          }
        }

        // pressure and viscous contribution to element vector on right-hand side:
        /*
        //    /                                                \
        //   |                    s  n+af                       |
        // + |  v , - p n + (nabla  u     - (1/3) * div u) * n  |
        //   |                                                  |
        //    \                                                / boundaryele
        //
        */
        const double timefacmurhs = timefacrhs * fac_ * 2.0 * visc_;
        const double timefacprhs = timefacrhs * fac_ * pressure;

        double nabla_u_o_n[3];
        nabla_u_o_n[0] =
            timefacmurhs * ((pvderxyaf(0, 0) - prefac * pvdiv) * unitnormal(0) +
                               0.5 * (pvderxyaf(0, 1) + pvderxyaf(1, 0)) * unitnormal(1) +
                               0.5 * (pvderxyaf(0, 2) + pvderxyaf(2, 0)) * unitnormal(2));
        nabla_u_o_n[1] =
            timefacmurhs * (0.5 * (pvderxyaf(1, 0) + pvderxyaf(0, 1)) * unitnormal(0) +
                               (pvderxyaf(1, 1) - prefac * pvdiv) * unitnormal(1) +
                               0.5 * (pvderxyaf(1, 2) + pvderxyaf(2, 1)) * unitnormal(2));
        nabla_u_o_n[2] =
            timefacmurhs * (0.5 * (pvderxyaf(2, 0) + pvderxyaf(0, 2)) * unitnormal(0) +
                               0.5 * (pvderxyaf(2, 1) + pvderxyaf(1, 2)) * unitnormal(1) +
                               (pvderxyaf(2, 2) - prefac * pvdiv) * unitnormal(2));

        for (int vi = 0; vi < piel; ++vi)
        {
          elevec(vi * 4) += pfunct(vi) * (-timefacprhs * unitnormal(0) + nabla_u_o_n[0]);
          elevec(vi * 4 + 1) += pfunct(vi) * (-timefacprhs * unitnormal(1) + nabla_u_o_n[1]);
          elevec(vi * 4 + 2) += pfunct(vi) * (-timefacprhs * unitnormal(2) + nabla_u_o_n[2]);
        }
      }
      else if (nsd == 2)
      {
        // Four potential contributions to element matrix on left-hand side:
        // 1) pressure term
        //   (only active if there is a dependence on flow rate and thus velocity,
        //    further non-linear dependence on velocity, e.g., in case of quadratic
        //    dependence of pressure on flow rate, not yet considered by including
        //    additional term for Newton linearization)
        /*
        //             /                           \
        //            |                             |
        //          + |  v , n * pressder * n * Du  |
        //            |                             |
        //             \                           / boundaryele
        //
        */
        const double timefacfacnpredern = timefac * fac_ * 1.0 * pressder;

        for (int ui = 0; ui < piel; ++ui)
        {
          for (int vi = 0; vi < piel; ++vi)
          {
            const double temp = timefacfacnpredern * pfunct(ui) * pfunct(vi);
            elemat(vi * 3, ui * 3) += temp;
            elemat(vi * 3 + 1, ui * 3 + 1) += temp;
          }
        }

        // 2) viscous term
        /*
        //    /                                    \
        //   |            s                         |
        // - |  v , (nabla  Du - (1/3) * div u) * n |
        //   |                                      |
        //    \                                    / boundaryele
        //
        */
        const double timefacmu = timefac * fac_ * 2.0 * visc_;

        for (int ui = 0; ui < piel; ++ui)
        {
          double nabla_u_o_n_lin[2][2];
          nabla_u_o_n_lin[0][0] = timefacmu * ((1.0 - prefac) * pderxy(0, ui) * unitnormal(0) +
                                                  0.5 * pderxy(1, ui) * unitnormal(1));
          nabla_u_o_n_lin[0][1] = timefacmu * (-prefac * pderxy(1, ui) * unitnormal(0) +
                                                  0.5 * pderxy(0, ui) * unitnormal(1));

          nabla_u_o_n_lin[1][0] = timefacmu * (0.5 * pderxy(1, ui) * unitnormal(0) -
                                                  prefac * pderxy(0, ui) * unitnormal(1));
          nabla_u_o_n_lin[1][1] = timefacmu * (0.5 * pderxy(0, ui) * unitnormal(0) +
                                                  (1.0 - prefac) * pderxy(1, ui) * unitnormal(1));

          for (int vi = 0; vi < piel; ++vi)
          {
            elemat(vi * 3, ui * 3) -= pfunct(vi) * nabla_u_o_n_lin[0][0];
            elemat(vi * 3, ui * 3 + 1) -= pfunct(vi) * nabla_u_o_n_lin[0][1];

            elemat(vi * 3 + 1, ui * 3) -= pfunct(vi) * nabla_u_o_n_lin[1][0];
            elemat(vi * 3 + 1, ui * 3 + 1) -= pfunct(vi) * nabla_u_o_n_lin[1][1];
          }
        }

        // check normal velocity -> further computation only required for
        // negative normal velocity, that is, inflow at this boundary
        if (normvel < -0.0001)
        {
          // 3) convective term
          /*
          //             /                        \
          //            |                          |
          //          - |  v , rho * Du ( u o n )  |
          //            |                          |
          //             \                        / boundaryele
          //
          */
          const double timefacfacdensnormvel = timefac * fac_ * densaf_ * normvel;

          for (int ui = 0; ui < piel; ++ui)
          {
            for (int vi = 0; vi < piel; ++vi)
            {
              const double temp = timefacfacdensnormvel * pfunct(ui) * pfunct(vi);
              elemat(vi * 3, ui * 3) -= temp;
              elemat(vi * 3 + 1, ui * 3 + 1) -= temp;
            }
          }

          // 4) additional convective term for Newton linearization
          if (is_newton)
          {
            /*
            //             /                        \
            //            |                          |
            //          - |  v , rho * u ( Du o n )  |
            //            |                          |
            //             \                        / boundaryele
            //
            */
            const double timefacfacdens = timefac * fac_ * densaf_;

            // dyadic product of unit normal vector and velocity vector
            Core::LinAlg::Matrix<nsd, nsd> n_x_u(Core::LinAlg::Initialization::zero);
            n_x_u.multiply_nt(pvelaf, unitnormal);

            for (int ui = 0; ui < piel; ++ui)
            {
              for (int vi = 0; vi < piel; ++vi)
              {
                const double temp = timefacfacdens * pfunct(ui) * pfunct(vi);

                elemat(vi * 3, ui * 3) -= temp * n_x_u(0, 0);
                elemat(vi * 3, ui * 3 + 1) -= temp * n_x_u(0, 1);

                elemat(vi * 3 + 1, ui * 3) -= temp * n_x_u(1, 0);
                elemat(vi * 3 + 1, ui * 3 + 1) -= temp * n_x_u(1, 1);
              }
            }
          }

          // convective contribution to element vector on right-hand side:
          /*
          //    /                       \
          //   |                         |
          // - |  v , rho * u ( u o n )  |
          //   |                         |
          //    \                       / boundaryele
          //
          */
          const double timefacconvrhs = timefacrhs * fac_ * densaf_ * normvel;

          for (int vi = 0; vi < piel; ++vi)
          {
            elevec(vi * 3) -= pfunct(vi) * timefacconvrhs * pvelaf(0);
            elevec(vi * 3 + 1) -= pfunct(vi) * timefacconvrhs * pvelaf(1);
          }
        }

        // pressure and viscous contribution to element vector on right-hand side:
        /*
        //    /                                                \
        //   |                    s  n+af                       |
        // + |  v , - p n + (nabla  u     - (1/3) * div u) * n  |
        //   |                                                  |
        //    \                                                / boundaryele
        //
        */
        const double timefacmurhs = timefacrhs * fac_ * 2.0 * visc_;
        const double timefacprhs = timefacrhs * fac_ * pressure;

        double nabla_u_o_n[2];
        nabla_u_o_n[0] =
            timefacmurhs * ((pvderxyaf(0, 0) - prefac * pvdiv) * unitnormal(0) +
                               0.5 * (pvderxyaf(0, 1) + pvderxyaf(1, 0)) * unitnormal(1));
        nabla_u_o_n[1] = timefacmurhs * (0.5 * (pvderxyaf(1, 0) + pvderxyaf(0, 1)) * unitnormal(0) +
                                            (pvderxyaf(1, 1) - prefac * pvdiv) * unitnormal(1));

        for (int vi = 0; vi < piel; ++vi)
        {
          elevec(vi * 3) += pfunct(vi) * (-timefacprhs * unitnormal(0) + nabla_u_o_n[0]);
          elevec(vi * 3 + 1) += pfunct(vi) * (-timefacprhs * unitnormal(1) + nabla_u_o_n[1]);
        }
      }
      else
        FOUR_C_THROW("incorrect number of spatial dimensions for parent element!");
    }  // end of integration loop
  }  // end of (temporarily) switching off of flow-dependent pressure boundary
     // conditions for zero time-curve factor
}  // Discret::Elements::FluidBoundaryParent<distype>::FlowDepPressureBC


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
template <Core::FE::CellType bdistype, Core::FE::CellType pdistype>
void Discret::Elements::FluidBoundaryParent<distype>::slip_supp_bc(
    Discret::Elements::FluidBoundary* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& plm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseVector::Base& elevec1)
{
  //---------------------------------------------------------------------
  // get time-integration parameters
  //---------------------------------------------------------------------
  const double timefac = fldparatimint_->time_fac();
  const double timefacrhs = fldparatimint_->time_fac_rhs();

  //---------------------------------------------------------------------
  // get parent element data
  //---------------------------------------------------------------------
  Core::Elements::Element* parent = surfele->parent_element();

  // parent element id
  int pid = parent->id();

  // number of parent spatial dimensions
  static const int nsd = Core::FE::dim<pdistype>;

  // number of parent element nodes
  static const int piel = Core::FE::num_nodes(pdistype);

  // reshape element matrices and vectors and init to zero, construct views
  const int peledim = (nsd + 1) * piel;
  elemat1.shape(peledim, peledim);
  elevec1.size(peledim);
  Core::LinAlg::Matrix<peledim, peledim> elemat(elemat1.values(), true);
  Core::LinAlg::Matrix<peledim, 1> elevec(elevec1.values(), true);

  // get local node coordinates
  Core::LinAlg::Matrix<nsd, piel> pxyze(Core::LinAlg::Initialization::zero);
  Core::Geo::fill_initial_position_array<pdistype, nsd, Core::LinAlg::Matrix<nsd, piel>>(
      parent, pxyze);

  // get Gaussian integration points
  const Core::FE::IntPointsAndWeights<nsd> pintpoints(
      Discret::Elements::DisTypeToOptGaussRule<pdistype>::rule);

  //---------------------------------------------------------------------
  // get boundary element data
  //---------------------------------------------------------------------
  // local surface id
  int bid = surfele->surface_number();

  // number of boundary spatial dimensions
  static const int bnsd = Core::FE::dim<bdistype>;

  // number of boundary element nodes
  static const int biel = Core::FE::num_nodes(bdistype);

  // get local node coordinates
  Core::LinAlg::Matrix<nsd, biel> bxyze(Core::LinAlg::Initialization::zero);
  Core::Geo::fill_initial_position_array<bdistype, nsd, Core::LinAlg::Matrix<nsd, biel>>(
      surfele, bxyze);

  // get Gaussian integration points
  const Core::FE::IntPointsAndWeights<bnsd> bintpoints(
      Discret::Elements::DisTypeToOptGaussRule<bdistype>::rule);

  // get location vector and ownerships for boundary element
  std::vector<int> blm;
  std::vector<int> blmowner;
  std::vector<int> blmstride;
  surfele->Core::Elements::Element::location_vector(discretization, blm, blmowner, blmstride);

  //---------------------------------------------------------------------
  // map Gaussian integration points to parent element for one-sided
  // derivatives on boundary
  //---------------------------------------------------------------------
  // coordinates of current integration point
  Core::LinAlg::SerialDenseMatrix gps(bintpoints.ip().nquad, bnsd);
  for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
  {
    const double* gpcoord = (bintpoints.ip().qxg)[iquad];
    for (int idim = 0; idim < bnsd; idim++)
    {
      gps(iquad, idim) = gpcoord[idim];
    }
  }

  // distinguish 2- and 3-D case
  Core::LinAlg::SerialDenseMatrix pqxg(pintpoints.ip().nquad, nsd);
  if (nsd == 2)
    Core::FE::boundary_gp_to_parent_gp2(pqxg, gps, pdistype, bdistype, bid);
  else if (nsd == 3)
    Core::FE::boundary_gp_to_parent_gp3(pqxg, gps, pdistype, bdistype, bid);

  //---------------------------------------------------------------------
  // extract parent and boundary values from global distributed vectors
  //---------------------------------------------------------------------
  // parent velocity at n+alpha_F
  std::shared_ptr<const Core::LinAlg::Vector<double>> velaf = discretization.get_state("velaf");
  if (velaf == nullptr) FOUR_C_THROW("Cannot get state vector 'velaf'");

  std::vector<double> mypvelaf = Core::FE::extract_values(*velaf, plm);

  Core::LinAlg::Matrix<nsd, piel> pevelaf(Core::LinAlg::Initialization::zero);
  for (int inode = 0; inode < piel; ++inode)
  {
    for (int idim = 0; idim < nsd; ++idim)
    {
      pevelaf(idim, inode) = mypvelaf[(nsd + 1) * inode + idim];
    }
  }

  // boundary pressure
  std::vector<double> mybvelaf = Core::FE::extract_values(*velaf, blm);

  Core::LinAlg::Matrix<1, biel> epressnp(Core::LinAlg::Initialization::zero);

  for (int inode = 0; inode < biel; ++inode)
  {
    epressnp(inode) = mybvelaf[nsd + ((nsd + 1) * inode)];
  }

  // parent and boundary displacement at n+1
  std::vector<double> mypedispnp((plm).size());
  std::vector<double> mybedispnp((blm).size());
  if (surfele->parent_element()->is_ale())
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp = discretization.get_state("dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

    mypedispnp = Core::FE::extract_values(*dispnp, plm);
    mybedispnp = Core::FE::extract_values(*dispnp, blm);

    // add parent and boundary displacement at n+1
    for (int idim = 0; idim < nsd; ++idim)
    {
      for (int pnode = 0; pnode < piel; ++pnode)
      {
        pxyze(idim, pnode) += mypedispnp[(nsd + 1) * pnode + idim];
      }
      for (int bnode = 0; bnode < biel; ++bnode)
      {
        bxyze(idim, bnode) += mybedispnp[(nsd + 1) * bnode + idim];
      }
    }
  }

  //---------------------------------------------------------------------
  // definitions and initializations for parent and boundary element
  //---------------------------------------------------------------------
  Core::LinAlg::Matrix<nsd, 1> pxsi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<piel, 1> pfunct(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, piel> pderiv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, nsd> pxjm(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, nsd> pxji(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, 1> boundaryNormal(
      Core::LinAlg::Initialization::zero);  // outward unit ('surface/line') normal of boundary
                                            // element at integration point
  Core::LinAlg::Matrix<nsd, piel> pderxy(
      Core::LinAlg::Initialization::zero);  // nabla of parent element at integration point
  Core::LinAlg::Matrix<nsd, nsd> pvderxyaf(
      Core::LinAlg::Initialization::zero);  // nabla*u of parent element at integration point
  Core::LinAlg::Matrix<1, 1> pressint(
      Core::LinAlg::Initialization::zero);  // pressure of boundary element at integration point
                                            // (N_C * p_C)

  Core::LinAlg::Matrix<bnsd, 1> xsi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<biel, 1> funct(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<bnsd, biel> deriv(Core::LinAlg::Initialization::zero);

  //---------------------------------------------------------------------
  // integration loop
  //---------------------------------------------------------------------
  for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
  {
    // coordinates of integration point w.r.t. parent element
    for (int idim = 0; idim < nsd; idim++)
    {
      pxsi(idim) = pqxg(iquad, idim);
    }

    // coordinates of integration point w.r.t. boundary element
    for (int idim = 0; idim < bnsd; idim++)
    {
      xsi(idim) = gps(iquad, idim);
    }

    // shape functions and derivatives of parent element at integration point
    Core::FE::shape_function<pdistype>(pxsi, pfunct);
    Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

    // shape functions and derivatives of boundary element at integration point
    Core::FE::shape_function<bdistype>(xsi, funct);
    Core::FE::shape_function_deriv1<bdistype>(xsi, deriv);

    // Compute pressure of boundary element at integration point
    pressint.multiply(epressnp, funct);

    // compute (inverse of) Jacobian matrix and determinant for parent element
    // and check its value
    pxjm.multiply_nt(pderiv, pxyze);
    const double pdet = pxji.invert(pxjm);
    if (pdet < 1E-16)
      FOUR_C_THROW("GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", pid, pdet);

    // compute measure tensor, infinitesimal area and outward unit normal
    // for boundary element
    drs_ = 0.0;
    Core::LinAlg::Matrix<bnsd, bnsd> metrictensor(Core::LinAlg::Initialization::zero);
    Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
        bxyze, deriv, metrictensor, drs_, &boundaryNormal);

    // compute integration factor for boundary element
    fac_ = bintpoints.ip().qwgt[iquad] * drs_;

    // compute global first derivatives for parent element
    pderxy.multiply(pxji, pderiv);

    // get velocity derivatives at n+alpha_F at integration point
    pvderxyaf.multiply_nt(pevelaf, pderxy);

    // evaluate material at integration point
    double rateofstrain = 0.0;  // Only Newtonian-fluids supported
    std::shared_ptr<Core::Mat::Material> material = parent->material();

    // get viscosity at integration point
    get_density_and_viscosity(material, 0.0, 0.0, rateofstrain);

    //---------------------------------------------------------------------
    // Contributions to element matrix and element vector
    //---------------------------------------------------------------------
    if ((nsd != 3) && (nsd != 2))
      FOUR_C_THROW("Incorrect number of spatial dimensions for parent element!");

    // Contributions to element matrix on left-hand side
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    const double timefacFac = timefac * fac_;
    const double timefacFacNu = timefac * fac_ * visc_;

    for (int ui = 0; ui < piel; ++ui)  // Columns
    {
      for (int vi = 0; vi < piel; ++vi)  // Rows
      {
        // Pressure term
        // *************
        double normalProd = 0;
        for (int sdim = 0; sdim < nsd; ++sdim)
        {
          normalProd += boundaryNormal(sdim) * boundaryNormal(sdim);
        }

        const double temp = timefacFac * pfunct(ui) * pfunct(vi);
        for (int idim = 0; idim < nsd; ++idim)
        {  // Write into 'pressure' column (nsd) of each 'spatial' row (idim)
          elemat(vi * (nsd + 1) + idim, ui * (nsd + 1) + nsd) +=
              temp * (normalProd * boundaryNormal(idim));
        }

        // Velocity terms
        // **************
        double sumNablaBoundaryNormal = 0;
        for (int sdim = 0; sdim < nsd; ++sdim)
        {
          sumNablaBoundaryNormal += pderxy(sdim, ui) * boundaryNormal(sdim);
        }

        for (int idim = 0; idim < nsd; ++idim)
        {
          for (int jdim = 0; jdim < nsd; ++jdim)
          {
            double velTerm = 0;
            for (int sdim = 0; sdim < nsd; ++sdim)
            {
              velTerm += boundaryNormal(sdim) * boundaryNormal(idim) * pderxy(sdim, ui) *
                         boundaryNormal(jdim);
            }
            velTerm += boundaryNormal(jdim) * boundaryNormal(idim) * sumNablaBoundaryNormal;

            elemat(vi * (nsd + 1) + idim, ui * (nsd + 1) + jdim) +=
                timefacFacNu * pfunct(vi) * (-1.0) * velTerm;
          }
        }
      }
    }

    // Contribution to element vector on right-hand side
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    const double timefacrhsFacPress = timefacrhs * fac_ * pressint(0);
    const double timefacrhsFacNu = timefacrhs * fac_ * visc_;

    for (int vi = 0; vi < piel; ++vi)
    {
      double normalProd = 0;
      for (int sdim = 0; sdim < nsd; ++sdim)
      {
        normalProd += boundaryNormal(sdim) * boundaryNormal(sdim);
      }

      // Pressure term
      // *************
      for (int idim = 0; idim < nsd; ++idim)
      {
        elevec(vi * (nsd + 1) + idim) -=
            pfunct(vi) * timefacrhsFacPress * (normalProd * boundaryNormal(idim));
      }

      // Velocity terms
      // **************
      for (int idim = 0; idim < nsd; ++idim)
      {
        // velTerm_i = n_k_j * [n_a_s * n_a_i * (f_s_j + f_j_s)]
        double velTerm = 0;
        for (int jdim = 0; jdim < nsd; ++jdim)
        {
          double tmp = 0;
          for (int sdim = 0; sdim < nsd; ++sdim)
          {
            tmp += boundaryNormal(sdim) * (pvderxyaf(sdim, jdim) + pvderxyaf(jdim, sdim));
          }
          velTerm += boundaryNormal(jdim) * boundaryNormal(idim) * tmp;
        }

        // Sum up final term at i: N_A * nu * (-velTerm)
        elevec(vi * (nsd + 1) + idim) -= pfunct(vi) * timefacrhsFacNu * (-1.0) * velTerm;
      }
    }
  }  // end of integration loop

  return;
}  // Discret::Elements::FluidBoundaryParent<distype>::SlipSuppBC


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
template <Core::FE::CellType bdistype, Core::FE::CellType pdistype>
void Discret::Elements::FluidBoundaryParent<distype>::navier_slip_bc(
    Discret::Elements::FluidBoundary* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& plm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseVector::Base& elevec1)
{
  //---------------------------------------------------------------------
  // get time-integration parameters
  //---------------------------------------------------------------------
  const double timefac = fldparatimint_->time_fac();
  const double timefacrhs = fldparatimint_->time_fac_rhs();

  //---------------------------------------------------------------------
  // get parent element data
  //---------------------------------------------------------------------
  Core::Elements::Element* parent = surfele->parent_element();

  // parent element id
  int pid = parent->id();

  // number of parent spatial dimensions
  static const int nsd = Core::FE::dim<pdistype>;

  // number of parent element nodes
  static const int piel = Core::FE::num_nodes(pdistype);

  // reshape element matrices and vectors and init to zero, construct views
  const int peledim = (nsd + 1) * piel;
  elemat1.shape(peledim, peledim);
  elevec1.size(peledim);
  Core::LinAlg::Matrix<peledim, peledim> elemat(elemat1.values(), true);
  Core::LinAlg::Matrix<peledim, 1> elevec(elevec1.values(), true);

  // get local node coordinates
  Core::LinAlg::Matrix<nsd, piel> pxyze(Core::LinAlg::Initialization::zero);
  Core::Geo::fill_initial_position_array<pdistype, nsd, Core::LinAlg::Matrix<nsd, piel>>(
      parent, pxyze);

  // get Gaussian integration points
  const Core::FE::IntPointsAndWeights<nsd> pintpoints(
      Discret::Elements::DisTypeToOptGaussRule<pdistype>::rule);

  //---------------------------------------------------------------------
  // get boundary element data
  //---------------------------------------------------------------------
  // local surface id
  int bid = surfele->surface_number();

  // number of boundary spatial dimensions
  static const int bnsd = Core::FE::dim<bdistype>;

  // number of boundary element nodes
  static const int biel = Core::FE::num_nodes(bdistype);

  // get local node coordinates
  Core::LinAlg::Matrix<nsd, biel> bxyze(Core::LinAlg::Initialization::zero);
  Core::Geo::fill_initial_position_array<bdistype, nsd, Core::LinAlg::Matrix<nsd, biel>>(
      surfele, bxyze);

  // get Gaussian integration points
  const Core::FE::IntPointsAndWeights<bnsd> bintpoints(
      Discret::Elements::DisTypeToOptGaussRule<bdistype>::rule);

  // get location vector and ownerships for boundary element
  std::vector<int> blm;
  std::vector<int> blmowner;
  std::vector<int> blmstride;
  surfele->Core::Elements::Element::location_vector(discretization, blm, blmowner, blmstride);

  // get slip coefficient
  const double beta = params.get<double>("beta");

  //---------------------------------------------------------------------
  // map Gaussian integration points to parent element for one-sided
  // derivatives on boundary
  //---------------------------------------------------------------------
  // coordinates of current integration point
  Core::LinAlg::SerialDenseMatrix gps(bintpoints.ip().nquad, bnsd);
  for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
  {
    const double* gpcoord = (bintpoints.ip().qxg)[iquad];
    for (int idim = 0; idim < bnsd; idim++)
    {
      gps(iquad, idim) = gpcoord[idim];
    }
  }

  // distinguish 2- and 3-D case
  Core::LinAlg::SerialDenseMatrix pqxg(pintpoints.ip().nquad, nsd);
  if (nsd == 2)
    Core::FE::boundary_gp_to_parent_gp2(pqxg, gps, pdistype, bdistype, bid);
  else if (nsd == 3)
    Core::FE::boundary_gp_to_parent_gp3(pqxg, gps, pdistype, bdistype, bid);

  //---------------------------------------------------------------------
  // extract parent and boundary values from global distributed vectors
  //---------------------------------------------------------------------
  // parent velocity at n+alpha_F
  std::shared_ptr<const Core::LinAlg::Vector<double>> velaf = discretization.get_state("velaf");
  if (velaf == nullptr) FOUR_C_THROW("Cannot get state vector 'velaf'");

  std::vector<double> mypvelaf = Core::FE::extract_values(*velaf, plm);

  Core::LinAlg::Matrix<nsd, piel> pevelaf(Core::LinAlg::Initialization::zero);
  for (int inode = 0; inode < piel; ++inode)
  {
    for (int idim = 0; idim < nsd; ++idim)
    {
      pevelaf(idim, inode) = mypvelaf[(nsd + 1) * inode + idim];
    }
  }

  // parent and boundary displacement at n+1
  std::vector<double> mypedispnp((plm).size());
  std::vector<double> mybedispnp((blm).size());
  if (surfele->parent_element()->is_ale())
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp = discretization.get_state("dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

    mypedispnp = Core::FE::extract_values(*dispnp, plm);
    mybedispnp = Core::FE::extract_values(*dispnp, blm);

    // add parent and boundary displacement at n+1
    for (int idim = 0; idim < nsd; ++idim)
    {
      for (int pnode = 0; pnode < piel; ++pnode)
      {
        pxyze(idim, pnode) += mypedispnp[(nsd + 1) * pnode + idim];
      }
      for (int bnode = 0; bnode < biel; ++bnode)
      {
        bxyze(idim, bnode) += mybedispnp[(nsd + 1) * bnode + idim];
      }
    }
  }

  //---------------------------------------------------------------------
  // definitions and initializations for parent and boundary element
  //---------------------------------------------------------------------
  Core::LinAlg::Matrix<nsd, 1> pxsi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<piel, 1> pfunct(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, piel> pderiv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, nsd> pxjm(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, nsd> pxji(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, 1> boundaryNormal(
      Core::LinAlg::Initialization::zero);  // outward unit ('surface/line') normal of boundary
                                            // element at integration point
  Core::LinAlg::Matrix<nsd, piel> pderxy(
      Core::LinAlg::Initialization::zero);  // nabla of parent element at integration point
  Core::LinAlg::Matrix<nsd, nsd> pvderxyaf(
      Core::LinAlg::Initialization::zero);  // nabla*u of parent element at integration point
  Core::LinAlg::Matrix<1, 1> pressint(
      Core::LinAlg::Initialization::zero);  // pressure of boundary element at integration point
                                            // (N_C * p_C)
  Core::LinAlg::Matrix<nsd, 1> pvelint(
      Core::LinAlg::Initialization::zero);  // velocity of parent element at integration point
                                            // (N_B * U_B_i)

  Core::LinAlg::Matrix<bnsd, 1> xsi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<biel, 1> funct(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<bnsd, biel> deriv(Core::LinAlg::Initialization::zero);

  //---------------------------------------------------------------------
  // integration loop
  //---------------------------------------------------------------------
  for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
  {
    // coordinates of integration point w.r.t. parent element
    for (int idim = 0; idim < nsd; idim++)
    {
      pxsi(idim) = pqxg(iquad, idim);
    }

    // coordinates of integration point w.r.t. boundary element
    for (int idim = 0; idim < bnsd; idim++)
    {
      xsi(idim) = gps(iquad, idim);
    }

    // shape functions and derivatives of parent element at integration point
    Core::FE::shape_function<pdistype>(pxsi, pfunct);
    Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

    // shape functions and derivatives of boundary element at integration point
    Core::FE::shape_function<bdistype>(xsi, funct);
    Core::FE::shape_function_deriv1<bdistype>(xsi, deriv);

    // Compute velocity of parent element at integration point
    pvelint.multiply(pevelaf, pfunct);

    // compute (inverse of) Jacobian matrix and determinant for parent element
    // and check its value
    pxjm.multiply_nt(pderiv, pxyze);
    const double pdet = pxji.invert(pxjm);
    if (pdet < 1E-16)
      FOUR_C_THROW("GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", pid, pdet);

    // compute measure tensor, infinitesimal area and outward unit normal
    // for boundary element
    drs_ = 0.0;
    Core::LinAlg::Matrix<bnsd, bnsd> metrictensor(Core::LinAlg::Initialization::zero);
    Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
        bxyze, deriv, metrictensor, drs_, &boundaryNormal);

    // compute integration factor for boundary element
    fac_ = bintpoints.ip().qwgt[iquad] * drs_;

    //---------------------------------------------------------------------
    // Contributions to element matrix and element vector
    //---------------------------------------------------------------------
    if ((nsd != 3) && (nsd != 2))
      FOUR_C_THROW("Incorrect number of spatial dimensions for parent element!");

    // Contributions to element matrix on left-hand side
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    const double timefacFacBeta = timefac * fac_ * beta;

    for (int ui = 0; ui < piel; ++ui)  // Columns
    {
      for (int vi = 0; vi < piel; ++vi)  // Rows
      {
        // Navier slip term
        // ****************
        for (int idim = 0; idim < nsd; ++idim)
        {
          for (int jdim = 0; jdim < nsd; ++jdim)
          {
            if (idim == jdim)
            {
              elemat(vi * (nsd + 1) + idim, ui * (nsd + 1) + jdim) +=
                  timefacFacBeta * pfunct(vi) * pfunct(ui);
            }
          }
        }
      }
    }

    // Contribution to element vector on right-hand side
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    const double timefacrhsFacBeta = timefacrhs * fac_ * beta;

    for (int vi = 0; vi < piel; ++vi)
    {
      // Navier slip term
      // ****************
      for (int idim = 0; idim < nsd; ++idim)
      {
        elevec(vi * (nsd + 1) + idim) -= pfunct(vi) * timefacrhsFacBeta * pvelint(idim, 0);
      }
    }
  }  // end of integration loop

  return;
}  // Discret::Elements::FluidBoundaryParent<distype>::NavierSlipBC

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
template <Core::FE::CellType bdistype, Core::FE::CellType pdistype>
void Discret::Elements::FluidBoundaryParent<distype>::evaluate_weak_dbc(
    Discret::Elements::FluidBoundary* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& plm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseVector::Base& elevec1)
{
  //---------------------------------------------------------------------
  // get condition information
  //---------------------------------------------------------------------
  const auto* wdbc_cond = params.get<const Core::Conditions::Condition*>("condition");

  // type of consistency (default: adjoint-consistent)
  const std::string& consistency = (*wdbc_cond).parameters().get<std::string>("GAMMATYPE");
  double wd_gamma = 0.0;
  if (consistency == "adjoint-consistent")
    wd_gamma = 1.0;
  else if (consistency == "diffusive-optimal")
    wd_gamma = -1.0;
  else
    FOUR_C_THROW("unknown type of consistency for weak DBC: {}", consistency);

  // decide whether to use it or not
  const std::string& deftauB = (*wdbc_cond).parameters().get<std::string>("PENTYPE");
  bool spalding = false;
  if (deftauB == "Spalding")
    spalding = true;
  else if (deftauB == "constant")
    spalding = false;
  else
    FOUR_C_THROW("unknown PENTYPE tauB for weak DBC: {}", deftauB);

  // linearisation of adjoint convective flux
  const std::string& linearisation_approach =
      (*wdbc_cond).parameters().get<std::string>("LINEARISATION");
  bool complete_linearisation = false;
  if (linearisation_approach == "lin_all")
    complete_linearisation = true;
  else if (linearisation_approach == "no_lin_conv_inflow")
    complete_linearisation = false;
  else
    FOUR_C_THROW("unknown linearisation for weak DBC: {}", linearisation_approach);

  // find out whether there is a time curve and get factor
  // (time curve at n+1 applied for all time-integration schemes, but
  //  variable time_ in fluid3_parameter is n+alpha_F in case of
  //  generalized-alpha time-integration scheme -> reset to time n+1)

  const double time =
      fldparatimint_->time() + (1 - fldparatimint_->alpha_f()) * fldparatimint_->dt();

  // get values and switches from condition
  // (assumed to be constant on element boundary)
  const auto functions = wdbc_cond->parameters().get<std::vector<int>>("FUNCT");

  // find out whether to apply weak DBC only in normal direction
  bool onlynormal = false;
  const std::string& active_components = wdbc_cond->parameters().get<std::string>("DIR");
  if (active_components == "all_directions")
    onlynormal = false;
  else if (active_components == "only_in_normal_direction")
    onlynormal = true;
  else
    FOUR_C_THROW("unknown definition of active components for weak DBC: {}", active_components);

  // optional scaling of penalty parameter
  const double scaling = wdbc_cond->parameters().get<double>("TauBscaling");
  if (spalding && fabs(scaling - 1.0) > 1e-9)
    FOUR_C_THROW(
        "Parameter tauB for weak DBC will be computed according to Spaldings law. Do not apply "
        "scaling factor != 1.0!\n");

  // default value for C_b is 4.0
  const double Cb = 4.0 * scaling;

  // get value for boundary condition and
  // check for Spalding's law in case of prescribed non-zero velocity
  const auto& val = wdbc_cond->parameters().get<std::vector<double>>("VAL");
  if (spalding)
  {
    for (int i = 0; i < 3; ++i)
    {
      if (val[i] * val[i] > 1e-9)
        FOUR_C_THROW("Applying Spaldings law to a wall with non-zero velocity!\n");
    }
  }

  //---------------------------------------------------------------------
  // get time-integration parameters
  //---------------------------------------------------------------------
  const double timefac = fldparatimint_->time_fac();
  const double timefacpre = fldparatimint_->time_fac_pre();
  const double timefacrhs = fldparatimint_->time_fac_rhs();

  //---------------------------------------------------------------------
  // get parent element data
  //---------------------------------------------------------------------
  Core::Elements::Element* parent = surfele->parent_element();

  // parent element id
  int pid = parent->id();

  // number of parent spatial dimensions
  static const int nsd = Core::FE::dim<pdistype>;

  // number of parent element nodes
  static const int piel = Core::FE::num_nodes(pdistype);

  // reshape element matrices and vectors and init to zero, construct views
  const int peledim = (nsd + 1) * piel;
  elemat1.shape(peledim, peledim);
  elevec1.size(peledim);
  Core::LinAlg::Matrix<peledim, peledim> elemat(elemat1.values(), true);
  Core::LinAlg::Matrix<peledim, 1> elevec(elevec1.values(), true);

  // get local node coordinates
  Core::LinAlg::Matrix<nsd, piel> pxyze(Core::LinAlg::Initialization::zero);
  Core::Geo::fill_initial_position_array<pdistype, nsd, Core::LinAlg::Matrix<nsd, piel>>(
      parent, pxyze);

  // get Gaussian integration points
  const Core::FE::IntPointsAndWeights<nsd> pintpoints(
      Discret::Elements::DisTypeToOptGaussRule<pdistype>::rule);

  //---------------------------------------------------------------------
  // get boundary element data
  //---------------------------------------------------------------------
  // local surface id
  int bid = surfele->surface_number();

  // number of boundary spatial dimensions
  static const int bnsd = Core::FE::dim<bdistype>;

  // number of boundary element nodes
  static const int biel = Core::FE::num_nodes(bdistype);

  // get local node coordinates
  Core::LinAlg::Matrix<nsd, biel> bxyze(Core::LinAlg::Initialization::zero);
  Core::Geo::fill_initial_position_array<bdistype, nsd, Core::LinAlg::Matrix<nsd, biel>>(
      surfele, bxyze);

  // get Gaussian integration points
  const Core::FE::IntPointsAndWeights<bnsd> bintpoints(
      Discret::Elements::DisTypeToOptGaussRule<bdistype>::rule);

  // get location vector and ownerships for boundary element
  std::vector<int> blm;
  std::vector<int> blmowner;
  std::vector<int> blmstride;
  surfele->Core::Elements::Element::location_vector(discretization, blm, blmowner, blmstride);

  //---------------------------------------------------------------------
  // map Gaussian integration points to parent element for one-sided
  // derivatives on boundary
  //---------------------------------------------------------------------
  // coordinates of current integration point
  Core::LinAlg::SerialDenseMatrix gps(bintpoints.ip().nquad, bnsd);
  for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
  {
    const double* gpcoord = (bintpoints.ip().qxg)[iquad];
    for (int idim = 0; idim < bnsd; idim++)
    {
      gps(iquad, idim) = gpcoord[idim];
    }
  }

  // distinguish 2- and 3-D case
  Core::LinAlg::SerialDenseMatrix pqxg(pintpoints.ip().nquad, nsd);
  if (nsd == 2)
    Core::FE::boundary_gp_to_parent_gp2(pqxg, gps, pdistype, bdistype, bid);
  else if (nsd == 3)
    Core::FE::boundary_gp_to_parent_gp3(pqxg, gps, pdistype, bdistype, bid);

  //---------------------------------------------------------------------
  // extract parent and boundary values from global distributed vectors
  //---------------------------------------------------------------------
  // parent velocity at n+alpha_F
  std::shared_ptr<const Core::LinAlg::Vector<double>> velaf = discretization.get_state("velaf");
  if (velaf == nullptr) FOUR_C_THROW("Cannot get state vector 'velaf'");

  std::vector<double> mypvelaf = Core::FE::extract_values(*velaf, plm);

  Core::LinAlg::Matrix<nsd, piel> pevelaf(Core::LinAlg::Initialization::zero);
  for (int inode = 0; inode < piel; ++inode)
  {
    for (int idim = 0; idim < nsd; ++idim)
    {
      pevelaf(idim, inode) = mypvelaf[(nsd + 1) * inode + idim];
    }
  }

  // parent velocity at n+1
  std::vector<double> mypvelnp(plm.size());

  if (fldparatimint_->time_algo() == Inpar::FLUID::timeint_npgenalpha)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> velnp = discretization.get_state("velnp");
    if (velnp == nullptr) FOUR_C_THROW("Cannot get state vector 'velnp'");

    mypvelnp = Core::FE::extract_values(*velnp, plm);
  }
  else
    mypvelnp = Core::FE::extract_values(*velaf, plm);

  Core::LinAlg::Matrix<nsd, piel> pevelnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<piel, 1> peprenp(Core::LinAlg::Initialization::zero);
  for (int inode = 0; inode < piel; ++inode)
  {
    for (int idim = 0; idim < nsd; ++idim)
    {
      pevelnp(idim, inode) = mypvelnp[(nsd + 1) * inode + idim];
    }
    peprenp(inode) = mypvelnp[(nsd + 1) * inode + nsd];
  }

  // parent and boundary displacement at n+1
  std::vector<double> mypedispnp((plm).size());
  std::vector<double> mybedispnp((blm).size());
  if (surfele->parent_element()->is_ale())
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp = discretization.get_state("dispnp");
    if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

    mypedispnp = Core::FE::extract_values(*dispnp, plm);
    mybedispnp = Core::FE::extract_values(*dispnp, blm);

    // add parent and boundary displacement at n+1
    for (int idim = 0; idim < nsd; ++idim)
    {
      for (int pnode = 0; pnode < piel; ++pnode)
      {
        pxyze(idim, pnode) += mypedispnp[(nsd + 1) * pnode + idim];
      }
      for (int bnode = 0; bnode < biel; ++bnode)
      {
        bxyze(idim, bnode) += mybedispnp[(nsd + 1) * bnode + idim];
      }
    }
  }

  //---------------------------------------------------------------------
  // definitions and initializations for parent and boundary element
  //---------------------------------------------------------------------
  Core::LinAlg::Matrix<nsd, 1> pxsi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<piel, 1> pfunct(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, piel> pderiv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, nsd> pxjm(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, nsd> pxji(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, 1> unitnormal(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, piel> pderxy(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, 1> pvelintaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, 1> pvelintnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, nsd> pvderxyaf(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<bnsd, 1> xsi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<biel, 1> funct(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<bnsd, biel> deriv(Core::LinAlg::Initialization::zero);

  //---------------------------------------------------------------------
  // integration loop
  //---------------------------------------------------------------------
  for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
  {
    // coordinates of integration point w.r.t. parent element
    for (int idim = 0; idim < nsd; idim++)
    {
      pxsi(idim) = pqxg(iquad, idim);
    }

    // coordinates of integration point w.r.t. boundary element
    for (int idim = 0; idim < bnsd; idim++)
    {
      xsi(idim) = gps(iquad, idim);
    }

    // shape functions and derivatives of parent element at integration point
    Core::FE::shape_function<pdistype>(pxsi, pfunct);
    Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

    // shape functions and derivatives of boundary element at integration point
    Core::FE::shape_function<bdistype>(xsi, funct);
    Core::FE::shape_function_deriv1<bdistype>(xsi, deriv);

    // compute (inverse of) Jacobian matrix and determinant for parent element
    // and check its value
    pxjm.multiply_nt(pderiv, pxyze);
    const double pdet = pxji.invert(pxjm);
    if (pdet < 1E-16)
      FOUR_C_THROW("GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", pid, pdet);

    // compute measure tensor, infinitesimal area and outward unit normal
    // for boundary element
    drs_ = 0.0;
    Core::LinAlg::Matrix<bnsd, bnsd> metrictensor(Core::LinAlg::Initialization::zero);
    Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
        bxyze, deriv, metrictensor, drs_, &unitnormal);

    // compute integration factor for boundary element
    fac_ = bintpoints.ip().qwgt[iquad] * drs_;

    // compute metric tensor G
    /*            +-           -+   +-           -+   +-           -+
                  |             |   |             |   |             |
                  |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
            G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
             ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
                  |    i     j  |   |    i     j  |   |    i     j  |
                  +-           -+   +-           -+   +-           -+
    */
    Core::LinAlg::Matrix<nsd, nsd> G;
    for (int nn = 0; nn < nsd; ++nn)
    {
      for (int rr = 0; rr < nsd; ++rr)
      {
        G(nn, rr) = pxji(nn, 0) * pxji(rr, 0);

        for (int mm = 1; mm < nsd; ++mm)
        {
          G(nn, rr) += pxji(nn, mm) * pxji(rr, mm);
        }
      }
    }

    // compute characteristic boundary element length
    //
    //                           2.0
    //             h  = ---------------------
    //              b        +-------------+
    //                      / /  T       \ |
    //                   \ / |  n * G * n |
    //                    +   \          /
    //

    double nGn = 0;
    for (int nn = 0; nn < nsd; ++nn)
    {
      for (int rr = 0; rr < nsd; ++rr)
      {
        nGn += unitnormal(rr) * G(rr, nn) * unitnormal(nn);
      }
    }
    const double h = 2.0 / sqrt(nGn);

    // compute factor due to spatial function
    Core::LinAlg::Matrix<nsd, 1> functionfac;
    for (int i = 0; i < nsd; ++i)
    {
      functionfac(i) = 1.0;
    }

    // determine global coordinates of integration point
    Core::LinAlg::Matrix<nsd, 1> coordgp(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < biel; ++i)
    {
      for (int idim = 0; idim < nsd; idim++)
      {
        coordgp(idim) += bxyze(idim, i) * funct(i);
      }
    }

    for (int idim = 0; idim < nsd; idim++)
    {
      // factor given by spatial function
      if (functions[idim] > 0)
      {
        // evaluate function at current gauss point
        // (important: requires 3D position vector)
        functionfac(idim) = Global::Problem::instance()
                                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functions[idim])
                                .evaluate(coordgp.data(), time, idim);
      }
      else
        functionfac(idim) = 1.0;
    }

    // compute global first derivatives for parent element
    pderxy.multiply(pxji, pderiv);

    // get velocity at n+alpha_F at integration point
    pvelintaf.multiply(pevelaf, pfunct);

    // get velocity at n+1/n+alpha_F at integration point
    pvelintnp.multiply(pevelnp, pfunct);

    // get velocity derivatives at n+1 at integration point
    pvderxyaf.multiply_nt(pevelaf, pderxy);

    // get pressure at n+1/n+alpha_F at integration point
    const double preintnp = pfunct.dot(peprenp);

    // evaluate material at integration point
    double rateofstrain = 0.0;
    std::shared_ptr<Core::Mat::Material> material = parent->material();

    // compute measure for rate of strain at n+alpha_F or n+1 if required
    // for non-Newtonian fluid
    if (material->material_type() == Core::Materials::m_carreauyasuda or
        material->material_type() == Core::Materials::m_modpowerlaw or
        material->material_type() == Core::Materials::m_herschelbulkley)
    {
      //-------------------------------------------------------------------
      //
      //          +-                                 -+ 1
      //          |          /   \           /   \    | -
      //          | 2 * eps | vel |   * eps | vel |   | 2
      //          |          \   / ij        \   / ij |
      //          +-                                 -+
      //
      //-------------------------------------------------------------------
      // compute (two times) strain-rate tensor
      Core::LinAlg::Matrix<nsd, nsd> two_epsilon;
      for (int rr = 0; rr < nsd; ++rr)
      {
        for (int mm = 0; mm < nsd; ++mm)
        {
          two_epsilon(rr, mm) = pvderxyaf(rr, mm) + pvderxyaf(mm, rr);
        }
      }

      // compute product of (two times) strain-rate tensor with itself
      for (int rr = 0; rr < nsd; ++rr)
      {
        for (int mm = 0; mm < nsd; ++mm)
        {
          rateofstrain += two_epsilon(rr, mm) * two_epsilon(mm, rr);
        }
      }

      // compute square-root of (two times) product
      rateofstrain = sqrt(rateofstrain / 2.0);
    }

    // get viscosity at integration point
    get_density_and_viscosity(material, 0.0, 0.0, rateofstrain);

    // define (initial) penalty parameter
    /*
    //
    //                         /    \
    //                        |  nu  |
    //         tau      = C * | ---- |
    //            B(,0)    b  |  h   |
    //                         \  b /
    */
    double tau_B = Cb * visc_ / h;

    // ---------------------------------------------------
    // update penalty parameter for Spalding law
    // ---------------------------------------------------
    if (spalding)
    {
      if (nsd != 3)
        FOUR_C_THROW(
            "Application of Spalding's law only reasonable for three-dimensional problems!");

      // only constant density of 1.0 allowed, for the time being
      if (densaf_ != 1.0)
        FOUR_C_THROW(
            "Only incompressible flow with density 1.0 allowed for Spalding's law so far!");

      //                             +--------------------+
      //                            /  +---+              |
      //       ||  n+af ||         /    \    n+af    n+af
      //       || u     ||  =     /      +  u     * u
      //                         /      /    j       j
      //                      \ /      +---+
      //                       +       dim j

      const double normu = pvelintaf.norm2();

      /*
      // the penalty term could be interpreted as a traction
      // boundary condition (normally g=0 for wall bounded flows)
      //
      //      /                       \
      //     |             / n+af   \  |
      //     | v , tau  * | u    - g | |
      //     |        B    \        /  |
      //      \                       / boundaryele
      //
      //           |                 |
      //           +-----------------+
      //                t  /   "
      //               " W/ rho
      */

      /*
      // this gives rise to the following definition of the
      // friction velocity
      //
      //                 +---------+     +----------------+
      //      u    =    / t   /      =  / tau  * || n+af||
      //       tau     v   W / rho     v     B   ||u    ||
      */

      /*
      // and hence the following dimensionless value for y
      //
      //                             +-------------+
      //           h           y *  / tau  * ||u||
      //            b      +       v     B
      //      y = ---- -> y  = ----------------------
      //           C                     nu
      //            b
      //                              +
      // note that y is constant but y  is depending on tau !
      //                                                   B
      //
      //              +-------------+       +-------------+
      //        h *  / tau  * ||u||        / tau  * ||u||
      //    +    b  v     B               v     B
      //   y  = ---------------------- = -------------------
      //              C  * nu                  tau
      //               b                          B,0
      */

      /*
      // accordingly, we are able to define the dimensioneless velocity
      //                                            +-------------+
      //                  ||  n+af ||              /  ||  n+af||
      //        +         || u     ||             /   || u    ||
      //       u  = ---------------------- =     /   -------------
      //              +------------------+    \ /        tau
      //             / tau  * ||  n+af ||      v            B
      //            v     B   || u     ||
      */

      // we assume a boundary layer thickness of
      //
      //
      //                    h
      //                     b       nu
      //               y = ---- = --------
      //                    C      tau
      //                     b        B,0
      //
      // (proportional to the grid size normal to the wall)
      const double y = h / Cb;

      // iterate until the residual of the Spalding equation is 0
      double res = spalding_residual(y, visc_, tau_B, normu);

      int count = 0;
      while (res * res > 1e-6)
      {
        const double drdtauB = jacobian_spalding_residual(y, visc_, tau_B, normu);

        if (drdtauB < 1e-10) FOUR_C_THROW("(Nearly) singular Jacobian of Spaldings equation");

        double inc = res / drdtauB;

        // do damping to avoid negative values of tau_B (robustness)
        while (tau_B - inc < 0)
        {
          inc /= 2.0;
        }

        // get jacobian, do damped Newton step
        tau_B -= inc;

        // get residual of Spaldings equation (law of the wall)
        res = spalding_residual(y, visc_, tau_B, normu);

        ++count;
        if (count > 100)
          FOUR_C_THROW(
              "no convergence in 100 steps in Newton iteration during solution of Spaldings "
              "equation\n");
      }
    }

    // normal velocity and residual (i.e., difference between velocity and
    // prescribed value) on boundary
    double normvel = 0.0;
    Core::LinAlg::Matrix<nsd, 1> bvres;
    for (int idim = 0; idim < nsd; idim++)
    {
      normvel += pvelintaf(idim) * unitnormal(idim);
      bvres(idim) = pvelintaf(idim) - val[idim] * functionfac(idim);
    }

    //---------------------------------------------------------------------
    // contributions to element matrix and element vector
    // (distinguish two- and three-dimensional case)
    //---------------------------------------------------------------------
    if (nsd == 3)
    {
      // factors used in the following
      const double timefacfacpre = fac_ * timefacpre;
      const double timefacfacrhs = fac_ * timefacrhs;

      //--------------------------------------------------
      // partially integrated pressure term, rescaled by gamma*dt
      /*
      // factor: 1.0
      //
      //             /            \
      //            |              |
      //          + |  v , Dp * n  |
      //            |              |
      //             \            / boundaryele
      //
      */

      for (int ui = 0; ui < piel; ++ui)
      {
        for (int vi = 0; vi < piel; ++vi)
        {
          elemat(vi * 4, ui * 4 + 3) += timefacfacpre * pfunct(vi) * pfunct(ui) * unitnormal(0);
          elemat(vi * 4 + 1, ui * 4 + 3) += timefacfacpre * pfunct(vi) * pfunct(ui) * unitnormal(1);
          elemat(vi * 4 + 2, ui * 4 + 3) += timefacfacpre * pfunct(vi) * pfunct(ui) * unitnormal(2);
        }
      }

      for (int vi = 0; vi < piel; ++vi)
      {
        elevec(vi * 4) -= timefacfacrhs * pfunct(vi) * unitnormal(0) * preintnp;
        elevec(vi * 4 + 1) -= timefacfacrhs * pfunct(vi) * unitnormal(1) * preintnp;
        elevec(vi * 4 + 2) -= timefacfacrhs * pfunct(vi) * unitnormal(2) * preintnp;
      }

      //--------------------------------------------------
      // adjoint consistency term, pressure/continuity part
      /*
      // factor: gdt
      //
      //             /            \
      //            |              |
      //          - |  q , Du * n  |
      //            |              |
      //             \            / boundaryele
      //
      */
      for (int ui = 0; ui < piel; ++ui)
      {
        for (int vi = 0; vi < piel; ++vi)
        {
          elemat(vi * 4 + 3, ui * 4) -= timefacfacpre * pfunct(vi) * pfunct(ui) * unitnormal(0);
          elemat(vi * 4 + 3, ui * 4 + 1) -= timefacfacpre * pfunct(vi) * pfunct(ui) * unitnormal(1);
          elemat(vi * 4 + 3, ui * 4 + 2) -= timefacfacpre * pfunct(vi) * pfunct(ui) * unitnormal(2);
        }
      }

      /*
      // factor: 1.0
      //
      //             /                       \
      //            |       / n+1     \       |
      //          + |  q , | u   - u   | * n  |
      //            |       \ (i)   B /       |
      //             \                       / boundaryele
      //
      */
      for (int vi = 0; vi < piel; ++vi)
      {
        elevec(vi * 4 + 3) += timefacfacrhs * pfunct(vi) *
                              ((pvelintnp(0) - val[0] * functionfac(0)) * unitnormal(0) +
                                  (pvelintnp(1) - val[1] * functionfac(1)) * unitnormal(1) +
                                  (pvelintnp(2) - val[2] * functionfac(2)) * unitnormal(2));
      }

      //---------------------------------------------------------------------
      //---------------------------------------------------------------------
      //           weak boundary conditions in all directions
      //---------------------------------------------------------------------
      //---------------------------------------------------------------------
      if (!onlynormal)
      {
        //--------------------------------------------------
        // partially integrated viscous term
        /*
        // factor: 2*mu*afgdt
        //
        //    /                   \
        //   |           s         |
        // - |  v , nabla  Du * n  |
        //   |                     |
        //    \                   / boundaryele
        //
        */
        const double timefacmu = fac_ * 2.0 * visc_ * timefac;

        for (int ui = 0; ui < piel; ++ui)
        {
          double nabla_u_o_n_lin[3][3];

          nabla_u_o_n_lin[0][0] =
              timefacmu * (pderxy(0, ui) * unitnormal(0) + 0.5 * pderxy(1, ui) * unitnormal(1) +
                              0.5 * pderxy(2, ui) * unitnormal(2));
          nabla_u_o_n_lin[0][1] = timefacmu * (0.5 * pderxy(0, ui) * unitnormal(1));
          nabla_u_o_n_lin[0][2] = timefacmu * (0.5 * pderxy(0, ui) * unitnormal(2));

          nabla_u_o_n_lin[1][0] = timefacmu * (0.5 * pderxy(1, ui) * unitnormal(0));
          nabla_u_o_n_lin[1][1] =
              timefacmu * (0.5 * pderxy(0, ui) * unitnormal(0) + pderxy(1, ui) * unitnormal(1) +
                              0.5 * pderxy(2, ui) * unitnormal(2));
          nabla_u_o_n_lin[1][2] = timefacmu * (0.5 * pderxy(1, ui) * unitnormal(2));

          nabla_u_o_n_lin[2][0] = timefacmu * (0.5 * pderxy(2, ui) * unitnormal(0));
          nabla_u_o_n_lin[2][1] = timefacmu * (0.5 * pderxy(2, ui) * unitnormal(1));
          nabla_u_o_n_lin[2][2] =
              timefacmu * (0.5 * pderxy(0, ui) * unitnormal(0) +
                              0.5 * pderxy(1, ui) * unitnormal(1) + pderxy(2, ui) * unitnormal(2));


          for (int vi = 0; vi < piel; ++vi)
          {
            elemat(vi * 4, ui * 4) -= pfunct(vi) * nabla_u_o_n_lin[0][0];
            elemat(vi * 4, ui * 4 + 1) -= pfunct(vi) * nabla_u_o_n_lin[0][1];
            elemat(vi * 4, ui * 4 + 2) -= pfunct(vi) * nabla_u_o_n_lin[0][2];

            elemat(vi * 4 + 1, ui * 4) -= pfunct(vi) * nabla_u_o_n_lin[1][0];
            elemat(vi * 4 + 1, ui * 4 + 1) -= pfunct(vi) * nabla_u_o_n_lin[1][1];
            elemat(vi * 4 + 1, ui * 4 + 2) -= pfunct(vi) * nabla_u_o_n_lin[1][2];

            elemat(vi * 4 + 2, ui * 4) -= pfunct(vi) * nabla_u_o_n_lin[2][0];
            elemat(vi * 4 + 2, ui * 4 + 1) -= pfunct(vi) * nabla_u_o_n_lin[2][1];
            elemat(vi * 4 + 2, ui * 4 + 2) -= pfunct(vi) * nabla_u_o_n_lin[2][2];
          }
        }

        /*
        // factor: 2*mu
        //
        //    /                     \
        //   |           s  n+af     |
        // + |  v , nabla  u    * n  |
        //   |                       |
        //    \                     / boundaryele
        //
        */
        const double timefacmurhs = fac_ * 2.0 * visc_ * timefacrhs;

        {
          double nabla_u_o_n[3];
          nabla_u_o_n[0] =
              timefacmurhs * (pvderxyaf(0, 0) * unitnormal(0) +
                                 0.5 * (pvderxyaf(0, 1) + pvderxyaf(1, 0)) * unitnormal(1) +
                                 0.5 * (pvderxyaf(0, 2) + pvderxyaf(2, 0)) * unitnormal(2));
          nabla_u_o_n[1] =
              timefacmurhs * (0.5 * (pvderxyaf(1, 0) + pvderxyaf(0, 1)) * unitnormal(0) +
                                 pvderxyaf(1, 1) * unitnormal(1) +
                                 0.5 * (pvderxyaf(1, 2) + pvderxyaf(2, 1)) * unitnormal(2));
          nabla_u_o_n[2] =
              timefacmurhs * (0.5 * (pvderxyaf(2, 0) + pvderxyaf(0, 2)) * unitnormal(0) +
                                 0.5 * (pvderxyaf(2, 1) + pvderxyaf(1, 2)) * unitnormal(1) +
                                 pvderxyaf(2, 2) * unitnormal(2));

          for (int vi = 0; vi < piel; ++vi)
          {
            elevec(vi * 4) += pfunct(vi) * nabla_u_o_n[0];
            elevec(vi * 4 + 1) += pfunct(vi) * nabla_u_o_n[1];
            elevec(vi * 4 + 2) += pfunct(vi) * nabla_u_o_n[2];
          }
        }

        //--------------------------------------------------
        // (adjoint) consistency term, viscous part
        /*
        // factor: 2*mu*gamma_wd*afgdt
        //
        //    /                   \
        //   |        s            |
        // - |   nabla  w * n , Du |
        //   |                     |
        //    \                   / boundaryele
        //
        */
        const double consistencytimefac = fac_ * 2.0 * visc_ * wd_gamma * timefac;

        Core::LinAlg::Matrix<3, 3> nabla_s_w_o_n;
        for (int vi = 0; vi < piel; ++vi)
        {
          nabla_s_w_o_n(0, 0) = consistencytimefac * (unitnormal(0) * (pderxy(0, vi)) +
                                                         unitnormal(1) * (0.5 * pderxy(1, vi)) +
                                                         unitnormal(2) * (0.5 * pderxy(2, vi)));
          nabla_s_w_o_n(0, 1) = consistencytimefac * (unitnormal(0) * (0.5 * pderxy(1, vi)));
          nabla_s_w_o_n(0, 2) = consistencytimefac * (unitnormal(0) * (0.5 * pderxy(2, vi)));

          nabla_s_w_o_n(1, 0) = consistencytimefac * (unitnormal(1) * (0.5 * pderxy(0, vi)));
          nabla_s_w_o_n(1, 1) = consistencytimefac * (unitnormal(0) * (0.5 * pderxy(0, vi)) +
                                                         unitnormal(1) * (pderxy(1, vi)) +
                                                         unitnormal(2) * (0.5 * pderxy(2, vi)));
          nabla_s_w_o_n(1, 2) = consistencytimefac * (unitnormal(1) * (0.5 * pderxy(2, vi)));

          nabla_s_w_o_n(2, 0) = consistencytimefac * (unitnormal(2) * (0.5 * pderxy(0, vi)));
          nabla_s_w_o_n(2, 1) = consistencytimefac * (unitnormal(2) * (0.5 * pderxy(1, vi)));
          nabla_s_w_o_n(2, 2) = consistencytimefac * (unitnormal(0) * (0.5 * pderxy(0, vi)) +
                                                         unitnormal(1) * (0.5 * pderxy(1, vi)) +
                                                         unitnormal(2) * (pderxy(2, vi)));


          for (int ui = 0; ui < piel; ++ui)
          {
            elemat(vi * 4, ui * 4) -= pfunct(ui) * nabla_s_w_o_n(0, 0);
            elemat(vi * 4, ui * 4 + 1) -= pfunct(ui) * nabla_s_w_o_n(0, 1);
            elemat(vi * 4, ui * 4 + 2) -= pfunct(ui) * nabla_s_w_o_n(0, 2);

            elemat(vi * 4 + 1, ui * 4) -= pfunct(ui) * nabla_s_w_o_n(1, 0);
            elemat(vi * 4 + 1, ui * 4 + 1) -= pfunct(ui) * nabla_s_w_o_n(1, 1);
            elemat(vi * 4 + 1, ui * 4 + 2) -= pfunct(ui) * nabla_s_w_o_n(1, 2);

            elemat(vi * 4 + 2, ui * 4) -= pfunct(ui) * nabla_s_w_o_n(2, 0);
            elemat(vi * 4 + 2, ui * 4 + 1) -= pfunct(ui) * nabla_s_w_o_n(2, 1);
            elemat(vi * 4 + 2, ui * 4 + 2) -= pfunct(ui) * nabla_s_w_o_n(2, 2);
          }
        }

        /*
        // factor: 2*mu*gamma_wd
        //
        //    /                           \
        //   |        s          n+af      |
        // + |   nabla  w * n , u    - u   |
        //   |                          b  |
        //    \                           / boundaryele
        //
        */
        const double consistencytimefacrhs = fac_ * 2.0 * visc_ * wd_gamma * timefacrhs;

        for (int vi = 0; vi < piel; ++vi)
        {
          elevec(vi * 4) +=
              consistencytimefacrhs * (unitnormal(0) * bvres(0) * (pderxy(0, vi)) +
                                          unitnormal(1) * bvres(0) * (0.5 * pderxy(1, vi)) +
                                          unitnormal(2) * bvres(0) * (0.5 * pderxy(2, vi)) +
                                          unitnormal(0) * bvres(1) * (0.5 * pderxy(1, vi)) +
                                          unitnormal(0) * bvres(2) * (0.5 * pderxy(2, vi)));

          elevec(vi * 4 + 1) +=
              consistencytimefacrhs * (unitnormal(1) * bvres(0) * (0.5 * pderxy(0, vi)) +
                                          unitnormal(0) * bvres(1) * (0.5 * pderxy(0, vi)) +
                                          unitnormal(1) * bvres(1) * (pderxy(1, vi)) +
                                          unitnormal(2) * bvres(1) * (0.5 * pderxy(2, vi)) +
                                          unitnormal(1) * bvres(2) * (0.5 * pderxy(2, vi)));

          elevec(vi * 4 + 2) +=
              consistencytimefacrhs * (unitnormal(2) * bvres(0) * (0.5 * pderxy(0, vi)) +
                                          unitnormal(2) * bvres(1) * (0.5 * pderxy(1, vi)) +
                                          unitnormal(0) * bvres(2) * (0.5 * pderxy(0, vi)) +
                                          unitnormal(1) * bvres(2) * (0.5 * pderxy(1, vi)) +
                                          unitnormal(2) * bvres(2) * (pderxy(2, vi)));
        }

        //--------------------------------------------------
        // adjoint consistency term, convective part (only on inflow)

        if (normvel < 0)
        {
          if (complete_linearisation)
          {
            /*
            // This linearisation has only to be included if
            // u*n is negative --- otherwise it's nonsense
            //
            // factor: afgdt
            //
            //    /                                \
            //   |         /      \       n+af      |
            // - | rho *  | Du * n | w , u    - u   |
            //   |         \      /              b  |
            //    \                                 / boundaryele, inflow
            //
            */
            const double timefacfacdens = fac_ * timefac * densaf_;

            for (int ui = 0; ui < piel; ++ui)
            {
              for (int vi = 0; vi < piel; ++vi)
              {
                elemat(vi * 4, ui * 4) -= timefacfacdens * pfunct(vi) * unitnormal(0) * bvres(0);
                elemat(vi * 4, ui * 4 + 1) -=
                    timefacfacdens * pfunct(vi) * unitnormal(1) * bvres(0);
                elemat(vi * 4, ui * 4 + 2) -=
                    timefacfacdens * pfunct(vi) * unitnormal(2) * bvres(0);

                elemat(vi * 4 + 1, ui * 4) -=
                    timefacfacdens * pfunct(vi) * unitnormal(0) * bvres(1);
                elemat(vi * 4 + 1, ui * 4 + 1) -=
                    timefacfacdens * pfunct(vi) * unitnormal(1) * bvres(1);
                elemat(vi * 4 + 1, ui * 4 + 2) -=
                    timefacfacdens * pfunct(vi) * unitnormal(2) * bvres(1);

                elemat(vi * 4 + 2, ui * 4) -=
                    timefacfacdens * pfunct(vi) * unitnormal(0) * bvres(2);
                elemat(vi * 4 + 2, ui * 4 + 1) -=
                    timefacfacdens * pfunct(vi) * unitnormal(1) * bvres(2);
                elemat(vi * 4 + 2, ui * 4 + 2) -=
                    timefacfacdens * pfunct(vi) * unitnormal(2) * bvres(2);
              }
            }

            /*
            // factor: afgdt
            //
            //    /                          \
            //   |         / n+af   \         |
            // - | rho *  | u    * n | w , Du |
            //   |         \        /         |
            //    \       |          |       / boundaryele, inflow
            //            +----------+
            //                 <0
            */
            const double normveltimefacfacdens = fac_ * timefac * normvel * densaf_;

            for (int ui = 0; ui < piel; ++ui)
            {
              for (int vi = 0; vi < piel; ++vi)
              {
                elemat(vi * 4, ui * 4) -= normveltimefacfacdens * pfunct(ui) * pfunct(vi);
                elemat(vi * 4 + 1, ui * 4 + 1) -= normveltimefacfacdens * pfunct(ui) * pfunct(vi);
                elemat(vi * 4 + 2, ui * 4 + 2) -= normveltimefacfacdens * pfunct(ui) * pfunct(vi);
              }
            }
          }  // end if full_linearisation

          /*
          // factor: 1
          //
          //    /                                  \
          //   |         / n+af   \       n+af      |
          // - | rho *  | u    * n | w , u    - u   |
          //   |         \        /              b  |
          //    \       |          |               / boundaryele, inflow
          //            +----------+
          //                <0
          */
          const double normveltimefacfacdensrhs = fac_ * timefacrhs * normvel * densaf_;

          for (int vi = 0; vi < piel; ++vi)
          {
            elevec(vi * 4) += normveltimefacfacdensrhs * pfunct(vi) * bvres(0);
            elevec(vi * 4 + 1) += normveltimefacfacdensrhs * pfunct(vi) * bvres(1);
            elevec(vi * 4 + 2) += normveltimefacfacdensrhs * pfunct(vi) * bvres(2);
          }
        }  // end if normvel<0, i.e. boundary is an inflow boundary

        //--------------------------------------------------
        // penalty term
        /*
        // factor: nu*Cb/h*afgdt
        //
        //    /        \
        //   |          |
        // + |  w , Du  |
        //   |          |
        //    \        / boundaryele
        //
        */
        const double timefacfactaub = fac_ * timefac * tau_B;

        for (int ui = 0; ui < piel; ++ui)
        {
          for (int vi = 0; vi < piel; ++vi)
          {
            const double temp = timefacfactaub * pfunct(ui) * pfunct(vi);

            elemat(vi * 4, ui * 4) += temp;
            elemat(vi * 4 + 1, ui * 4 + 1) += temp;
            elemat(vi * 4 + 2, ui * 4 + 2) += temp;
          }
        }

        /*
        // factor: nu*Cb/h
        //
        //    /                \
        //   |        n+af      |
        // + |   w , u    - u   |
        //   |               b  |
        //    \                / boundaryele
        //
        */
        const double timefacfactaubrhs = fac_ * timefacrhs * tau_B;

        for (int vi = 0; vi < piel; ++vi)
        {
          elevec(vi * 4) -= timefacfactaubrhs * pfunct(vi) * bvres(0);
          elevec(vi * 4 + 1) -= timefacfactaubrhs * pfunct(vi) * bvres(1);
          elevec(vi * 4 + 2) -= timefacfactaubrhs * pfunct(vi) * bvres(2);
        }
      }  // !onlynormal
      //---------------------------------------------------------------------
      //---------------------------------------------------------------------
      //          weak boundary conditions only in normal direction
      //---------------------------------------------------------------------
      //---------------------------------------------------------------------
      else
      {
        //--------------------------------------------------
        // partially integrated viscous term
        /*
        // factor: 2*mu
        //
        //    /                           \
        //   |                   s         |
        // + |  v * n , n * nabla  Du * n  |
        //   |                             |
        //    \                           / boundaryele
        //
        */
        const double timefacmu = fac_ * 2.0 * visc_ * timefac;

        for (int ui = 0; ui < piel; ++ui)
        {
          const double aux =
              timefacmu * (pderxy(0, ui) * unitnormal(0) + pderxy(1, ui) * unitnormal(1) +
                              pderxy(2, ui) * unitnormal(2));
          for (int vi = 0; vi < piel; ++vi)
          {
            elemat(vi * 4, ui * 4) -= pfunct(vi) * unitnormal(0) * unitnormal(0) * aux;
            elemat(vi * 4, ui * 4 + 1) -= pfunct(vi) * unitnormal(0) * unitnormal(1) * aux;
            elemat(vi * 4, ui * 4 + 2) -= pfunct(vi) * unitnormal(0) * unitnormal(2) * aux;

            elemat(vi * 4 + 1, ui * 4) -= pfunct(vi) * unitnormal(1) * unitnormal(0) * aux;
            elemat(vi * 4 + 1, ui * 4 + 1) -= pfunct(vi) * unitnormal(1) * unitnormal(1) * aux;
            elemat(vi * 4 + 1, ui * 4 + 2) -= pfunct(vi) * unitnormal(1) * unitnormal(2) * aux;

            elemat(vi * 4 + 2, ui * 4) -= pfunct(vi) * unitnormal(2) * unitnormal(0) * aux;
            elemat(vi * 4 + 2, ui * 4 + 1) -= pfunct(vi) * unitnormal(2) * unitnormal(1) * aux;
            elemat(vi * 4 + 2, ui * 4 + 2) -= pfunct(vi) * unitnormal(2) * unitnormal(2) * aux;
          }
        }

        /*
        // factor: 2*mu
        //
        //    /                             \
        //   |                   s  n+af     |
        // + |  v * n , n * nabla  u    * n  |
        //   |                               |
        //    \                             / boundaryele
        //
        */
        const double timefacmurhs = fac_ * 2.0 * visc_ * timefacrhs;

        double n_o_nabla_u_o_n =
            pvderxyaf(0, 0) * unitnormal(0) * unitnormal(0) +
            pvderxyaf(1, 1) * unitnormal(1) * unitnormal(1) +
            pvderxyaf(2, 2) * unitnormal(2) * unitnormal(2) +
            (pvderxyaf(0, 1) + pvderxyaf(1, 0)) * unitnormal(0) * unitnormal(1) +
            (pvderxyaf(0, 2) + pvderxyaf(2, 0)) * unitnormal(0) * unitnormal(2) +
            (pvderxyaf(1, 2) + pvderxyaf(2, 1)) * unitnormal(2) * unitnormal(1);

        for (int vi = 0; vi < piel; ++vi)
        {
          elevec(vi * 4) += timefacmurhs * pfunct(vi) * unitnormal(0) * n_o_nabla_u_o_n;
          elevec(vi * 4 + 1) += timefacmurhs * pfunct(vi) * unitnormal(1) * n_o_nabla_u_o_n;
          elevec(vi * 4 + 2) += timefacmurhs * pfunct(vi) * unitnormal(2) * n_o_nabla_u_o_n;
        }

        //--------------------------------------------------
        // (adjoint) consistency term, viscous part
        /*
        // factor: 2*mu*gamma_wd*afgdt
        //
        //    /                             \
        //   |            s                  |
        // + |   n * nabla  w * n ,  Du * n  |
        //   |                               |
        //    \                             / boundaryele
        //
        */
        const double consistencytimefac = fac_ * 2.0 * visc_ * wd_gamma * timefac;

        for (int vi = 0; vi < piel; ++vi)
        {
          for (int ui = 0; ui < piel; ++ui)
          {
            const double aux = pderxy(0, vi) * unitnormal(0) + pderxy(1, vi) * unitnormal(1) +
                               pderxy(2, vi) * unitnormal(2);

            elemat(vi * 4, ui * 4) -=
                aux * unitnormal(0) * unitnormal(0) * pfunct(ui) * consistencytimefac;
            elemat(vi * 4, ui * 4 + 1) -=
                aux * unitnormal(0) * unitnormal(1) * pfunct(ui) * consistencytimefac;
            elemat(vi * 4, ui * 4 + 2) -=
                aux * unitnormal(0) * unitnormal(2) * pfunct(ui) * consistencytimefac;

            elemat(vi * 4 + 1, ui * 4) -=
                aux * unitnormal(1) * unitnormal(0) * pfunct(ui) * consistencytimefac;
            elemat(vi * 4 + 1, ui * 4 + 1) -=
                aux * unitnormal(1) * unitnormal(1) * pfunct(ui) * consistencytimefac;
            elemat(vi * 4 + 1, ui * 4 + 2) -=
                aux * unitnormal(1) * unitnormal(2) * pfunct(ui) * consistencytimefac;

            elemat(vi * 4 + 2, ui * 4) -=
                aux * unitnormal(2) * unitnormal(0) * pfunct(ui) * consistencytimefac;
            elemat(vi * 4 + 2, ui * 4 + 1) -=
                aux * unitnormal(2) * unitnormal(1) * pfunct(ui) * consistencytimefac;
            elemat(vi * 4 + 2, ui * 4 + 2) -=
                aux * unitnormal(2) * unitnormal(2) * pfunct(ui) * consistencytimefac;
          }
        }

        /*
        // factor: 2*mu*gamma_wd
        //
        //    /                                        \
        //   |            s          / n+af     \       |
        // + |   n * nabla  w * n , | u    - u   | * n  |
        //   |                       \        b /       |
        //    \                                        / boundaryele
        //
        */
        const double consistencytimefacrhs = fac_ * 2.0 * visc_ * wd_gamma * timefacrhs;

        double bvres_o_n =
            bvres(0) * unitnormal(0) + bvres(1) * unitnormal(1) + bvres(2) * unitnormal(2);

        for (int vi = 0; vi < piel; ++vi)
        {
          double aux = (pderxy(0, vi) * unitnormal(0) + pderxy(1, vi) * unitnormal(1) +
                        pderxy(2, vi) * unitnormal(2));

          elevec(vi * 4) += consistencytimefacrhs * aux * unitnormal(0) * bvres_o_n;
          elevec(vi * 4 + 1) += consistencytimefacrhs * aux * unitnormal(1) * bvres_o_n;
          elevec(vi * 4 + 2) += consistencytimefacrhs * aux * unitnormal(2) * bvres_o_n;
        }

        //--------------------------------------------------
        // penalty term
        /*
        // factor:mu*Cb/h*afgdt
        //
        //    /                \
        //   |                  |
        // + |  w o n , Du o n  |
        //   |                  |
        //    \                / boundaryele
        //
        */
        const double timefacfactaub = fac_ * timefac * tau_B;

        for (int ui = 0; ui < piel; ++ui)
        {
          for (int vi = 0; vi < piel; ++vi)
          {
            elemat(vi * 4, ui * 4) +=
                timefacfactaub * pfunct(vi) * unitnormal(0) * unitnormal(0) * pfunct(ui);
            elemat(vi * 4, ui * 4 + 1) +=
                timefacfactaub * pfunct(vi) * unitnormal(0) * unitnormal(1) * pfunct(ui);
            elemat(vi * 4, ui * 4 + 2) +=
                timefacfactaub * pfunct(vi) * unitnormal(0) * unitnormal(2) * pfunct(ui);

            elemat(vi * 4 + 1, ui * 4) +=
                timefacfactaub * pfunct(vi) * unitnormal(1) * unitnormal(0) * pfunct(ui);
            elemat(vi * 4 + 1, ui * 4 + 1) +=
                timefacfactaub * pfunct(vi) * unitnormal(1) * unitnormal(1) * pfunct(ui);
            elemat(vi * 4 + 1, ui * 4 + 2) +=
                timefacfactaub * pfunct(vi) * unitnormal(1) * unitnormal(2) * pfunct(ui);

            elemat(vi * 4 + 2, ui * 4) +=
                timefacfactaub * pfunct(vi) * unitnormal(2) * unitnormal(0) * pfunct(ui);
            elemat(vi * 4 + 2, ui * 4 + 1) +=
                timefacfactaub * pfunct(vi) * unitnormal(2) * unitnormal(1) * pfunct(ui);
            elemat(vi * 4 + 2, ui * 4 + 2) +=
                timefacfactaub * pfunct(vi) * unitnormal(2) * unitnormal(2) * pfunct(ui);
          }
        }

        /*
        // factor: mu*Cb/h
        //
        //    /                           \
        //   |            / n+af   \       |
        // + |   w o n , | u    - u | o n  |
        //   |            \  b     /       |
        //    \                           / boundaryele
        //
        */
        const double timefacfactaubrhs = fac_ * timefacrhs * tau_B;

        for (int vi = 0; vi < piel; ++vi)
        {
          elevec(vi * 4) -= timefacfactaubrhs * pfunct(vi) * unitnormal(0) * bvres_o_n;
          elevec(vi * 4 + 1) -= timefacfactaubrhs * pfunct(vi) * unitnormal(1) * bvres_o_n;
          elevec(vi * 4 + 2) -= timefacfactaubrhs * pfunct(vi) * unitnormal(2) * bvres_o_n;
        }

        //--------------------------------------------------
        // adjoint consistency term, convective part (only on inflow)

        if (normvel < 0)
        {
          if (complete_linearisation)
          {
            /*
            // These linearisations have only to be included if
            // u*n is negative --- otherwise they're nonsense
            //
            // factor: afgdt
            //
            //    /                                                 \
            //   |        /       \    /     \     / n+af     \      |
            // - | rho *  | Du * n |  | w o n | , | u    - u   | o n |
            //   |         \      /    \     /     \        b /      |
            //    \                                                 / boundaryele, inflow
            //
            */
            const double timefacfacdensbvresn = fac_ * timefac * densaf_ * bvres_o_n;

            for (int ui = 0; ui < piel; ++ui)
            {
              for (int vi = 0; vi < piel; ++vi)
              {
                elemat(vi * 4, ui * 4) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(0) * unitnormal(0) * pfunct(ui);
                elemat(vi * 4, ui * 4 + 1) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(0) * unitnormal(1) * pfunct(ui);
                elemat(vi * 4, ui * 4 + 2) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(0) * unitnormal(2) * pfunct(ui);

                elemat(vi * 4 + 1, ui * 4) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(1) * unitnormal(0) * pfunct(ui);
                elemat(vi * 4 + 1, ui * 4 + 1) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(1) * unitnormal(1) * pfunct(ui);
                elemat(vi * 4 + 1, ui * 4 + 2) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(1) * unitnormal(2) * pfunct(ui);

                elemat(vi * 4 + 2, ui * 4) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(2) * unitnormal(0) * pfunct(ui);
                elemat(vi * 4 + 2, ui * 4 + 1) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(2) * unitnormal(1) * pfunct(ui);
                elemat(vi * 4 + 2, ui * 4 + 2) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(2) * unitnormal(2) * pfunct(ui);
              }
            }

            /*
            // factor: afgdt
            //
            //    /                                      \
            //   |         / n+af   \   /     \           |
            // - | rho *  | u    * n | | w o n | , Du o n |
            //   |         \        /   \     /           |
            //    \        |          |                   / boundaryele, inflow
            //             +----------+
            //                 <0
            */
            const double normveltimefacfacdens = fac_ * timefac * normvel * densaf_;

            for (int ui = 0; ui < piel; ++ui)
            {
              for (int vi = 0; vi < piel; ++vi)
              {
                elemat(vi * 4, ui * 4) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(0) * unitnormal(0) * pfunct(ui);
                elemat(vi * 4, ui * 4 + 1) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(0) * unitnormal(1) * pfunct(ui);
                elemat(vi * 4, ui * 4 + 2) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(0) * unitnormal(2) * pfunct(ui);

                elemat(vi * 4 + 1, ui * 4) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(1) * unitnormal(0) * pfunct(ui);
                elemat(vi * 4 + 1, ui * 4 + 1) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(1) * unitnormal(1) * pfunct(ui);
                elemat(vi * 4 + 1, ui * 4 + 2) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(1) * unitnormal(2) * pfunct(ui);

                elemat(vi * 4 + 2, ui * 4) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(2) * unitnormal(0) * pfunct(ui);
                elemat(vi * 4 + 2, ui * 4 + 1) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(2) * unitnormal(1) * pfunct(ui);
                elemat(vi * 4 + 2, ui * 4 + 2) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(2) * unitnormal(2) * pfunct(ui);
              }
            }

          }  // end complete_linearisation

          /*
          // factor: 1
          //
          //    /                                            \
          //   |          / n+af   \   /     \    / n+af     \      |
          // - |  rho *  | u    * n | | w o n |, | u    - u   | o n |
          //   |          \        /   \     /    \        b /      |
          //    \        |          |                              / boundaryele, inflow
          //             +----------+
          //                 <0
          */
          const double normveltimefacfacdensrhs = fac_ * timefacrhs * normvel * densaf_ * bvres_o_n;

          for (int vi = 0; vi < piel; ++vi)
          {
            elevec(vi * 4) += normveltimefacfacdensrhs * pfunct(vi) * unitnormal(0);
            elevec(vi * 4 + 1) += normveltimefacfacdensrhs * pfunct(vi) * unitnormal(1);
            elevec(vi * 4 + 2) += normveltimefacfacdensrhs * pfunct(vi) * unitnormal(2);
          }
        }  // end if normvel<0, i.e. boundary is an inflow boundary
      }  // onlynormal
    }
    else if (nsd == 2)
    {
      // factors used in the following
      const double timefacfacpre = fac_ * timefacpre;
      const double timefacfacrhs = fac_ * timefacrhs;

      //--------------------------------------------------
      // partially integrated pressure term, rescaled by gamma*dt
      /*
      // factor: 1.0
      //
      //             /            \
      //            |              |
      //          + |  v , Dp * n  |
      //            |              |
      //             \            / boundaryele
      //
      */
      for (int ui = 0; ui < piel; ++ui)
      {
        for (int vi = 0; vi < piel; ++vi)
        {
          elemat(vi * 3, ui * 3 + 2) += timefacfacpre * pfunct(vi) * pfunct(ui) * unitnormal(0);
          elemat(vi * 3 + 1, ui * 3 + 2) += timefacfacpre * pfunct(vi) * pfunct(ui) * unitnormal(1);
        }
      }

      for (int vi = 0; vi < piel; ++vi)
      {
        elevec(vi * 3) -= timefacfacrhs * pfunct(vi) * unitnormal(0) * preintnp;
        elevec(vi * 3 + 1) -= timefacfacrhs * pfunct(vi) * unitnormal(1) * preintnp;
      }

      //--------------------------------------------------
      // adjoint consistency term, pressure/continuity part
      /*
      // factor: gdt
      //
      //             /              \
      //            |                |
      //          - |  q , Dacc * n  |
      //            |                |
      //             \              / boundaryele
      //
      */
      for (int ui = 0; ui < piel; ++ui)
      {
        for (int vi = 0; vi < piel; ++vi)
        {
          elemat(vi * 3 + 2, ui * 3) -= timefacfacpre * pfunct(vi) * pfunct(ui) * unitnormal(0);
          elemat(vi * 3 + 2, ui * 3 + 1) -= timefacfacpre * pfunct(vi) * pfunct(ui) * unitnormal(1);
        }
      }

      /*
      // factor: 1.0
      //
      //             /                       \
      //            |       / n+1     \       |
      //          + |  q , | u   - u   | * n  |
      //            |       \ (i)   B /       |
      //             \                       / boundaryele
      //
      */
      for (int vi = 0; vi < piel; ++vi)
      {
        elevec(vi * 3 + 2) += timefacfacrhs * pfunct(vi) *
                              ((pvelintnp(0) - val[0] * functionfac(0)) * unitnormal(0) +
                                  (pvelintnp(1) - val[1] * functionfac(1)) * unitnormal(1));
      }

      //---------------------------------------------------------------------
      //---------------------------------------------------------------------
      //           weak boundary conditions in all directions
      //---------------------------------------------------------------------
      //---------------------------------------------------------------------
      if (!onlynormal)
      {
        //--------------------------------------------------
        // partially integrated viscous term
        /*
        // factor: 2*mu*afgdt
        //
        //    /                   \
        //   |           s         |
        // - |  v , nabla  Du * n  |
        //   |                     |
        //    \                   / boundaryele
        //
        */
        const double timefacmu = fac_ * 2.0 * visc_ * timefac;

        for (int ui = 0; ui < piel; ++ui)
        {
          double nabla_u_o_n_lin[2][2];

          nabla_u_o_n_lin[0][0] =
              timefacmu * (pderxy(0, ui) * unitnormal(0) + 0.5 * pderxy(1, ui) * unitnormal(1));
          nabla_u_o_n_lin[0][1] = timefacmu * (0.5 * pderxy(0, ui) * unitnormal(1));

          nabla_u_o_n_lin[1][0] = timefacmu * (0.5 * pderxy(1, ui) * unitnormal(0));
          nabla_u_o_n_lin[1][1] =
              timefacmu * (0.5 * pderxy(0, ui) * unitnormal(0) + pderxy(1, ui) * unitnormal(1));


          for (int vi = 0; vi < piel; ++vi)
          {
            elemat(vi * 3, ui * 3) -= pfunct(vi) * nabla_u_o_n_lin[0][0];
            elemat(vi * 3, ui * 3 + 1) -= pfunct(vi) * nabla_u_o_n_lin[0][1];

            elemat(vi * 3 + 1, ui * 3) -= pfunct(vi) * nabla_u_o_n_lin[1][0];
            elemat(vi * 3 + 1, ui * 3 + 1) -= pfunct(vi) * nabla_u_o_n_lin[1][1];
          }
        }

        /*
        // factor: 2*mu
        //
        //    /                     \
        //   |           s  n+af     |
        // + |  v , nabla  u    * n  |
        //   |                       |
        //    \                     / boundaryele
        //
        */
        const double timefacmurhs = fac_ * 2.0 * visc_ * timefacrhs;

        {
          double nabla_u_o_n[2];
          nabla_u_o_n[0] =
              timefacmurhs * (pvderxyaf(0, 0) * unitnormal(0) +
                                 0.5 * (pvderxyaf(0, 1) + pvderxyaf(1, 0)) * unitnormal(1));
          nabla_u_o_n[1] =
              timefacmurhs * (0.5 * (pvderxyaf(1, 0) + pvderxyaf(0, 1)) * unitnormal(0) +
                                 pvderxyaf(1, 1) * unitnormal(1));

          for (int vi = 0; vi < piel; ++vi)
          {
            elevec(vi * 3) += pfunct(vi) * nabla_u_o_n[0];
            elevec(vi * 3 + 1) += pfunct(vi) * nabla_u_o_n[1];
          }
        }

        //--------------------------------------------------
        // (adjoint) consistency term, viscous part
        /*
        // factor: 2*mu*gamma_wd*afgdt
        //
        //    /                   \
        //   |        s            |
        // - |   nabla  w * n , Du |
        //   |                     |
        //    \                   / boundaryele
        //
        */
        const double consistencytimefac = fac_ * 2.0 * visc_ * wd_gamma * timefac;

        Core::LinAlg::Matrix<2, 2> nabla_s_w_o_n;
        for (int vi = 0; vi < piel; ++vi)
        {
          nabla_s_w_o_n(0, 0) = consistencytimefac * (unitnormal(0) * (pderxy(0, vi)) +
                                                         unitnormal(1) * (0.5 * pderxy(1, vi)));
          nabla_s_w_o_n(0, 1) = consistencytimefac * (unitnormal(0) * (0.5 * pderxy(1, vi)));

          nabla_s_w_o_n(1, 0) = consistencytimefac * (unitnormal(1) * (0.5 * pderxy(0, vi)));
          nabla_s_w_o_n(1, 1) = consistencytimefac * (unitnormal(0) * (0.5 * pderxy(0, vi)) +
                                                         unitnormal(1) * (pderxy(1, vi)));


          for (int ui = 0; ui < piel; ++ui)
          {
            elemat(vi * 3, ui * 3) -= pfunct(ui) * nabla_s_w_o_n(0, 0);
            elemat(vi * 3, ui * 3 + 1) -= pfunct(ui) * nabla_s_w_o_n(0, 1);

            elemat(vi * 3 + 1, ui * 3) -= pfunct(ui) * nabla_s_w_o_n(1, 0);
            elemat(vi * 3 + 1, ui * 3 + 1) -= pfunct(ui) * nabla_s_w_o_n(1, 1);
          }
        }

        /*
        // factor: 2*mu*gamma_wd
        //
        //    /                           \
        //   |        s          n+af      |
        // + |   nabla  w * n , u    - u   |
        //   |                          b  |
        //    \                           / boundaryele
        //
        */
        const double consistencytimefacrhs = fac_ * 2.0 * visc_ * wd_gamma * timefacrhs;

        for (int vi = 0; vi < piel; ++vi)
        {
          elevec(vi * 3) +=
              consistencytimefacrhs * (unitnormal(0) * bvres(0) * (pderxy(0, vi)) +
                                          unitnormal(1) * bvres(0) * (0.5 * pderxy(1, vi)) +
                                          unitnormal(0) * bvres(1) * (0.5 * pderxy(1, vi)));

          elevec(vi * 3 + 1) +=
              consistencytimefacrhs * (unitnormal(1) * bvres(0) * (0.5 * pderxy(0, vi)) +
                                          unitnormal(0) * bvres(1) * (0.5 * pderxy(0, vi)) +
                                          unitnormal(1) * bvres(1) * (pderxy(1, vi)));
        }

        //--------------------------------------------------
        // adjoint consistency term, convective part (only on inflow)

        if (normvel < 0)
        {
          if (complete_linearisation)
          {
            /*
            // This linearisation has only to be included if
            // u*n is negative --- otherwise it's nonsense
            //
            // factor: afgdt
            //
            //    /                                \
            //   |         /      \       n+af      |
            // - | rho *  | Du * n | w , u    - u   |
            //   |         \      /              b  |
            //    \                                 / boundaryele, inflow
            //
            */
            const double timefacfacdens = fac_ * timefac * densaf_;

            for (int ui = 0; ui < piel; ++ui)
            {
              for (int vi = 0; vi < piel; ++vi)
              {
                elemat(vi * 3, ui * 3) -= timefacfacdens * pfunct(vi) * unitnormal(0) * bvres(0);
                elemat(vi * 3, ui * 3 + 1) -=
                    timefacfacdens * pfunct(vi) * unitnormal(1) * bvres(0);

                elemat(vi * 3 + 1, ui * 3) -=
                    timefacfacdens * pfunct(vi) * unitnormal(0) * bvres(1);
                elemat(vi * 3 + 1, ui * 3 + 1) -=
                    timefacfacdens * pfunct(vi) * unitnormal(1) * bvres(1);
              }
            }

            /*
            // factor: afgdt
            //
            //    /                          \
            //   |         / n+af   \         |
            // - | rho *  | u    * n | w , Du |
            //   |         \        /         |
            //    \       |          |       / boundaryele, inflow
            //            +----------+
            //                 <0
            */
            const double normveltimefacfacdens = fac_ * timefac * normvel * densaf_;

            for (int ui = 0; ui < piel; ++ui)
            {
              for (int vi = 0; vi < piel; ++vi)
              {
                elemat(vi * 3, ui * 3) -= normveltimefacfacdens * pfunct(ui) * pfunct(vi);
                elemat(vi * 3 + 1, ui * 3 + 1) -= normveltimefacfacdens * pfunct(ui) * pfunct(vi);
              }
            }
          }  // end if full_linearisation

          /*
          // factor: 1
          //
          //    /                                  \
          //   |         / n+af   \       n+af      |
          // - | rho *  | u    * n | w , u    - u   |
          //   |         \        /              b  |
          //    \       |          |               / boundaryele, inflow
          //            +----------+
          //                <0
          */
          const double normveltimefacfacdensrhs = fac_ * timefacrhs * normvel * densaf_;

          for (int vi = 0; vi < piel; ++vi)
          {
            elevec(vi * 3) += normveltimefacfacdensrhs * pfunct(vi) * bvres(0);
            elevec(vi * 3 + 1) += normveltimefacfacdensrhs * pfunct(vi) * bvres(1);
          }
        }  // end if normvel<0, i.e. boundary is an inflow boundary

        //--------------------------------------------------
        // penalty term
        /*
        // factor: nu*Cb/h*afgdt
        //
        //    /        \
        //   |          |
        // + |  w , Du  |
        //   |          |
        //    \        / boundaryele
        //
        */
        const double timefacfactaub = fac_ * timefac * tau_B;

        for (int ui = 0; ui < piel; ++ui)
        {
          for (int vi = 0; vi < piel; ++vi)
          {
            const double temp = timefacfactaub * pfunct(ui) * pfunct(vi);

            elemat(vi * 3, ui * 3) += temp;
            elemat(vi * 3 + 1, ui * 3 + 1) += temp;
          }
        }

        /*
        // factor: nu*Cb/h
        //
        //    /                \
        //   |        n+af      |
        // + |   w , u    - u   |
        //   |               b  |
        //    \                / boundaryele
        //
        */
        const double timefacfactaubrhs = fac_ * timefacrhs * tau_B;

        for (int vi = 0; vi < piel; ++vi)
        {
          elevec(vi * 3) -= timefacfactaubrhs * pfunct(vi) * bvres(0);
          elevec(vi * 3 + 1) -= timefacfactaubrhs * pfunct(vi) * bvres(1);
        }
      }  // !onlynormal
      //---------------------------------------------------------------------
      //---------------------------------------------------------------------
      //          weak boundary conditions only in normal direction
      //---------------------------------------------------------------------
      //---------------------------------------------------------------------
      else
      {
        //--------------------------------------------------
        // partially integrated viscous term
        /*
        // factor: 2*mu
        //
        //    /                           \
        //   |                   s         |
        // + |  v * n , n * nabla  Du * n  |
        //   |                             |
        //    \                           / boundaryele
        //
        */
        const double timefacmu = fac_ * 2.0 * visc_ * timefac;

        for (int ui = 0; ui < piel; ++ui)
        {
          const double aux =
              timefacmu * (pderxy(0, ui) * unitnormal(0) + pderxy(1, ui) * unitnormal(1));

          for (int vi = 0; vi < piel; ++vi)
          {
            elemat(vi * 3, ui * 3) -= pfunct(vi) * unitnormal(0) * unitnormal(0) * aux;
            elemat(vi * 3, ui * 3 + 1) -= pfunct(vi) * unitnormal(0) * unitnormal(1) * aux;

            elemat(vi * 3 + 1, ui * 3) -= pfunct(vi) * unitnormal(1) * unitnormal(0) * aux;
            elemat(vi * 3 + 1, ui * 3 + 1) -= pfunct(vi) * unitnormal(1) * unitnormal(1) * aux;
          }
        }

        /*
        // factor: 2*mu
        //
        //    /                             \
        //   |                   s  n+af     |
        // + |  v * n , n * nabla  u    * n  |
        //   |                               |
        //    \                             / boundaryele
        //
        */
        const double timefacmurhs = fac_ * 2.0 * visc_ * timefacrhs;

        double n_o_nabla_u_o_n =
            pvderxyaf(0, 0) * unitnormal(0) * unitnormal(0) +
            pvderxyaf(1, 1) * unitnormal(1) * unitnormal(1) +
            (pvderxyaf(0, 1) + pvderxyaf(1, 0)) * unitnormal(0) * unitnormal(1);

        for (int vi = 0; vi < piel; ++vi)
        {
          elevec(vi * 3) += timefacmurhs * pfunct(vi) * unitnormal(0) * n_o_nabla_u_o_n;
          elevec(vi * 3 + 1) += timefacmurhs * pfunct(vi) * unitnormal(1) * n_o_nabla_u_o_n;
        }

        //--------------------------------------------------
        // (adjoint) consistency term, viscous part
        /*
        // factor: 2*mu*gamma_wd*afgdt
        //
        //    /                             \
        //   |            s                  |
        // + |   n * nabla  w * n ,  Du * n  |
        //   |                               |
        //    \                             / boundaryele
        //
        */
        const double consistencytimefac = fac_ * 2.0 * visc_ * wd_gamma * timefac;

        for (int vi = 0; vi < piel; ++vi)
        {
          for (int ui = 0; ui < piel; ++ui)
          {
            const double aux = pderxy(0, vi) * unitnormal(0) + pderxy(1, vi) * unitnormal(1);

            elemat(vi * 3, ui * 3) -=
                aux * unitnormal(0) * unitnormal(0) * pfunct(ui) * consistencytimefac;
            elemat(vi * 3, ui * 3 + 1) -=
                aux * unitnormal(0) * unitnormal(1) * pfunct(ui) * consistencytimefac;

            elemat(vi * 3 + 1, ui * 3) -=
                aux * unitnormal(1) * unitnormal(0) * pfunct(ui) * consistencytimefac;
            elemat(vi * 3 + 1, ui * 3 + 1) -=
                aux * unitnormal(1) * unitnormal(1) * pfunct(ui) * consistencytimefac;
          }
        }

        /*
        // factor: 2*mu*gamma_wd
        //
        //    /                                        \
        //   |            s          / n+af     \       |
        // + |   n * nabla  w * n , | u    - u   | * n  |
        //   |                       \        b /       |
        //    \                                        / boundaryele
        //
        */
        const double consistencytimefacrhs = fac_ * 2.0 * visc_ * wd_gamma * timefacrhs;

        double bvres_o_n = bvres(0) * unitnormal(0) + bvres(1) * unitnormal(1);

        for (int vi = 0; vi < piel; ++vi)
        {
          double aux = pderxy(0, vi) * unitnormal(0) + pderxy(1, vi) * unitnormal(1);

          elevec(vi * 3) += consistencytimefacrhs * aux * unitnormal(0) * bvres_o_n;
          elevec(vi * 3 + 1) += consistencytimefacrhs * aux * unitnormal(1) * bvres_o_n;
        }

        //--------------------------------------------------
        // penalty term
        /*
        // factor: mu*Cb/h*afgdt
        //
        //    /                \
        //   |                  |
        // + |  w o n , Du o n  |
        //   |                  |
        //    \                / boundaryele
        //
        */
        const double timefacfactaub = fac_ * timefac * tau_B;

        for (int ui = 0; ui < piel; ++ui)
        {
          for (int vi = 0; vi < piel; ++vi)
          {
            elemat(vi * 3, ui * 3) +=
                timefacfactaub * pfunct(vi) * unitnormal(0) * unitnormal(0) * pfunct(ui);
            elemat(vi * 3, ui * 3 + 1) +=
                timefacfactaub * pfunct(vi) * unitnormal(0) * unitnormal(1) * pfunct(ui);

            elemat(vi * 3 + 1, ui * 3) +=
                timefacfactaub * pfunct(vi) * unitnormal(1) * unitnormal(0) * pfunct(ui);
            elemat(vi * 3 + 1, ui * 3 + 1) +=
                timefacfactaub * pfunct(vi) * unitnormal(1) * unitnormal(1) * pfunct(ui);
          }
        }

        /*
        // factor: mu*Cb/h
        //
        //    /                           \
        //   |            / n+af   \       |
        // + |   w o n , | u    - u | o n  |
        //   |            \  b     /       |
        //    \                           / boundaryele
        //
        */
        const double timefacfactaubrhs = fac_ * timefacrhs * tau_B;

        for (int vi = 0; vi < piel; ++vi)
        {
          elevec(vi * 3) -= timefacfactaubrhs * pfunct(vi) * unitnormal(0) * bvres_o_n;
          elevec(vi * 3 + 1) -= timefacfactaubrhs * pfunct(vi) * unitnormal(1) * bvres_o_n;
        }

        //--------------------------------------------------
        // adjoint consistency term, convective part (only on inflow)

        if (normvel < 0)
        {
          if (complete_linearisation)
          {
            /*
            // These linearisations have only to be included if
            // u*n is negative --- otherwise they're nonsense
            //
            // factor: afgdt
            //
            //    /                                                 \
            //   |        /       \    /     \     / n+af     \      |
            // - | rho *  | Du * n |  | w o n | , | u    - u   | o n |
            //   |         \      /    \     /     \        b /      |
            //    \                                                 / boundaryele, inflow
            //
            */
            const double timefacfacdensbvresn = fac_ * timefac * densaf_ * bvres_o_n;

            for (int ui = 0; ui < piel; ++ui)
            {
              for (int vi = 0; vi < piel; ++vi)
              {
                elemat(vi * 3, ui * 3) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(0) * unitnormal(0) * pfunct(ui);
                elemat(vi * 3, ui * 3 + 1) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(0) * unitnormal(1) * pfunct(ui);

                elemat(vi * 3 + 1, ui * 3) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(1) * unitnormal(0) * pfunct(ui);
                elemat(vi * 3 + 1, ui * 3 + 1) -=
                    timefacfacdensbvresn * pfunct(vi) * unitnormal(1) * unitnormal(1) * pfunct(ui);
              }
            }

            /*
            // factor: afgdt
            //
            //    /                                      \
            //   |         / n+af   \   /     \           |
            // - | rho *  | u    * n | | w o n | , Du o n |
            //   |         \        /   \     /           |
            //    \        |          |                   / boundaryele, inflow
            //             +----------+
            //                 <0
            */
            const double normveltimefacfacdens = fac_ * timefac * normvel * densaf_;

            for (int ui = 0; ui < piel; ++ui)
            {
              for (int vi = 0; vi < piel; ++vi)
              {
                elemat(vi * 3, ui * 3) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(0) * unitnormal(0) * pfunct(ui);
                elemat(vi * 3, ui * 3 + 1) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(0) * unitnormal(1) * pfunct(ui);

                elemat(vi * 3 + 1, ui * 3) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(1) * unitnormal(0) * pfunct(ui);
                elemat(vi * 3 + 1, ui * 3 + 1) -=
                    normveltimefacfacdens * pfunct(vi) * unitnormal(1) * unitnormal(1) * pfunct(ui);
              }
            }

          }  // end complete_linearisation

          /*
          // factor: 1
          //
          //    /                                            \
          //   |          / n+af   \   /     \    / n+af     \      |
          // - |  rho *  | u    * n | | w o n |, | u    - u   | o n |
          //   |          \        /   \     /    \        b /      |
          //    \        |          |                              / boundaryele, inflow
          //             +----------+
          //                 <0
          */
          const double normveltimefacfacdensrhs = fac_ * timefacrhs * normvel * densaf_ * bvres_o_n;

          for (int vi = 0; vi < piel; ++vi)
          {
            elevec(vi * 3) += normveltimefacfacdensrhs * pfunct(vi) * unitnormal(0);
            elevec(vi * 3 + 1) += normveltimefacfacdensrhs * pfunct(vi) * unitnormal(1);
          }
        }  // end if normvel<0, i.e. boundary is an inflow boundary

      }  // onlynormal
    }
    else
      FOUR_C_THROW("incorrect number of spatial dimensions for parent element!");
  }  // end integration loop

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
template <Core::FE::CellType bdistype, Core::FE::CellType pdistype>
void Discret::Elements::FluidBoundaryParent<distype>::estimate_nitsche_trace_max_eigenvalue(
    Core::Elements::FaceElement* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& blm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2)
{
  //---------------------------------------------------------------------
  // get parent element data
  //---------------------------------------------------------------------
  Core::Elements::Element* parent = surfele->parent_element();

  // parent element id
  int pid = parent->id();

  // number of parent spatial dimensions
  static const int nsd = Core::FE::dim<pdistype>;

  // number of parent element nodes
  static const int piel = Core::FE::num_nodes(pdistype);

  // reshape element matrices and vectors and init to zero, construct views
  const int peledim = nsd * piel;
  elemat1.shape(peledim, peledim);
  elemat2.shape(peledim, peledim);
  Core::LinAlg::Matrix<peledim, peledim> Amat(elemat1.values(), true);
  Core::LinAlg::Matrix<peledim, peledim> Bmat(elemat2.values(), true);

  // get local node coordinates
  Core::LinAlg::Matrix<nsd, piel> pxyze(Core::LinAlg::Initialization::zero);
  Core::Geo::fill_initial_position_array<pdistype, nsd, Core::LinAlg::Matrix<nsd, piel>>(
      parent, pxyze);

  // get Gaussian integration points
  const Core::FE::IntPointsAndWeights<nsd> pintpoints(
      Discret::Elements::DisTypeToOptGaussRule<pdistype>::rule);

  // get location vector and ownerships for parent element
  std::vector<int> plm;
  std::vector<int> plmowner;
  std::vector<int> plmstride;
  parent->Core::Elements::Element::location_vector(discretization, plm, plmowner, plmstride);

  //---------------------------------------------------------------------
  // get boundary element data
  //---------------------------------------------------------------------
  // local surface id
  int bid = surfele->face_master_number();

  // number of boundary spatial dimensions
  static const int bnsd = Core::FE::dim<bdistype>;

  // number of boundary element nodes
  static const int biel = Core::FE::num_nodes(bdistype);

  // get local node coordinates
  Core::LinAlg::Matrix<nsd, biel> bxyze(Core::LinAlg::Initialization::zero);
  Core::Geo::fill_initial_position_array<bdistype, nsd, Core::LinAlg::Matrix<nsd, biel>>(
      surfele, bxyze);

  // get Gaussian integration points
  const Core::FE::IntPointsAndWeights<bnsd> bintpoints(
      Discret::Elements::DisTypeToOptGaussRule<bdistype>::rule);

  //---------------------------------------------------------------------
  // map Gaussian integration points to parent element for one-sided
  // derivatives on boundary
  //---------------------------------------------------------------------
  // coordinates of current integration point
  Core::LinAlg::SerialDenseMatrix gps(bintpoints.ip().nquad, bnsd);
  for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
  {
    const double* gpcoord = (bintpoints.ip().qxg)[iquad];
    for (int idim = 0; idim < bnsd; idim++)
    {
      gps(iquad, idim) = gpcoord[idim];
    }
  }

  Core::LinAlg::SerialDenseMatrix pqxg(pintpoints.ip().nquad, nsd);
  if (nsd == 3)
    Core::FE::boundary_gp_to_parent_gp3(pqxg, gps, pdistype, bdistype, bid);
  else if (nsd == 2)
    Core::FE::boundary_gp_to_parent_gp2(pqxg, gps, pdistype, bdistype, bid);
  else
    FOUR_C_THROW("only 2D and 3D");
  //---------------------------------------------------------------------
  // extract parent and boundary values from global distributed vectors
  //---------------------------------------------------------------------
  Discret::Elements::Fluid* fele =
      dynamic_cast<Discret::Elements::Fluid*>(surfele->parent_element());
  if (fele)
    if (fele->is_ale())
    {
      // parent and boundary displacement at n+1
      std::vector<double> mypedispnp((plm).size());
      std::vector<double> mybedispnp((blm).size());
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("dispnp");
      if (dispnp == nullptr) FOUR_C_THROW("Cannot get state vector 'dispnp'");

      mypedispnp = Core::FE::extract_values(*dispnp, plm);
      mybedispnp = Core::FE::extract_values(*dispnp, blm);

      // add parent and boundary displacement at n+1
      for (int idim = 0; idim < nsd; ++idim)
      {
        for (int pnode = 0; pnode < piel; ++pnode)
        {
          pxyze(idim, pnode) += mypedispnp[(nsd + 1) * pnode + idim];
        }
        for (int bnode = 0; bnode < biel; ++bnode)
        {
          bxyze(idim, bnode) += mybedispnp[(nsd + 1) * bnode + idim];
        }
      }
    }

  //---------------------------------------------------------------------
  // definitions and initializations for parent and boundary element
  //---------------------------------------------------------------------
  Core::LinAlg::Matrix<nsd, 1> pxsi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<piel, 1> pfunct(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, piel> pderiv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, nsd> pxjm(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, nsd> pxji(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, 1> unitnormal(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, piel> pderxy(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, 1> pvelintaf(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, 1> pvelintnp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<nsd, nsd> pvderxyaf(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<bnsd, 1> xsi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<biel, 1> funct(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<bnsd, biel> deriv(Core::LinAlg::Initialization::zero);

  //  double meas_surf = 0.0;
  //  double meas_vol = 0.0;

  //---------------------------------------------------------------------
  // part I : integration loop for boundary element
  //---------------------------------------------------------------------
  for (int iquad = 0; iquad < bintpoints.ip().nquad; ++iquad)
  {
    // coordinates of integration point w.r.t. parent element
    for (int idim = 0; idim < nsd; idim++)
    {
      pxsi(idim) = pqxg(iquad, idim);
    }

    // coordinates of integration point w.r.t. boundary element
    for (int idim = 0; idim < bnsd; idim++)
    {
      xsi(idim) = gps(iquad, idim);
    }

    // shape functions and derivatives of parent element at integration point
    Core::FE::shape_function<pdistype>(pxsi, pfunct);
    Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

    // shape functions and derivatives of boundary element at integration point
    Core::FE::shape_function<bdistype>(xsi, funct);
    Core::FE::shape_function_deriv1<bdistype>(xsi, deriv);

    // compute (inverse of) Jacobian matrix and determinant for parent element
    // and check its value
    pxjm.multiply_nt(pderiv, pxyze);
    const double pdet = pxji.invert(pxjm);
    if (pdet < 1E-16)
      FOUR_C_THROW("GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", pid, pdet);

    // compute measure tensor, infinitesimal area and outward unit normal
    // for boundary element
    drs_ = 0.0;
    Core::LinAlg::Matrix<bnsd, bnsd> metrictensor(Core::LinAlg::Initialization::zero);
    Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
        bxyze, deriv, metrictensor, drs_, &unitnormal);

    // compute integration factor for boundary element
    fac_ = bintpoints.ip().qwgt[iquad] * drs_;

    //    meas_surf += fac_;

    // compute global first derivatives for parent element
    pderxy.multiply(pxji, pderiv);

    const unsigned Velx = 0;
    const unsigned Vely = 1;
    const unsigned Velz = 2;

    /*
     //
     //    /                     \
     //   |                       |
     // + |  eps(v)*n , eps(u)*n  |
     //   |                       |
     //    \                     / boundaryele
     */

    for (int vi = 0; vi < piel; ++vi)
    {
      int ivx = vi * (nsd) + 0;
      int ivy = vi * (nsd) + 1;
      int ivz = vi * (nsd) + 2;

      for (int ui = 0; ui < piel; ++ui)
      {
        int iux = ui * (nsd) + 0;
        int iuy = ui * (nsd) + 1;
        int iuz = ui * (nsd) + 2;

        //(x,x)
        Amat(ivx, iux) +=
            fac_ * (pderxy(Velx, vi) * unitnormal(Velx) * pderxy(Velx, ui) * unitnormal(Velx)  // 1
                       + 0.5 * pderxy(Velx, vi) * unitnormal(Velx) * pderxy(Vely, ui) *
                             unitnormal(Vely)  // 2
                       + 0.5 * pderxy(Vely, vi) * unitnormal(Vely) * pderxy(Velx, ui) *
                             unitnormal(Velx)  // 4
                       + 0.25 * pderxy(Vely, vi) * unitnormal(Vely) * pderxy(Vely, ui) *
                             unitnormal(Vely)  // 5
                   );


        //(x,y)
        Amat(ivx, iuy) += fac_ * (0.5 * pderxy(Velx, vi) * unitnormal(Velx) * pderxy(Velx, ui) *
                                         unitnormal(Vely)  // 1
                                     + 0.25 * pderxy(Vely, vi) * unitnormal(Vely) *
                                           pderxy(Velx, ui) * unitnormal(Vely)  // 2
                                     + 0.25 * pderxy(Vely, vi) * unitnormal(Velx) *
                                           pderxy(Velx, ui) * unitnormal(Velx)  // 3
                                     + 0.5 * pderxy(Vely, vi) * unitnormal(Velx) *
                                           pderxy(Vely, ui) * unitnormal(Vely)  // 4
                                 );

        //(y,x)
        Amat(ivy, iux) +=
            fac_ *
            (0.5 * pderxy(Velx, vi) * unitnormal(Vely) * pderxy(Velx, ui) * unitnormal(Velx) +
                0.25 * pderxy(Velx, vi) * unitnormal(Vely) * pderxy(Vely, ui) * unitnormal(Vely) +
                0.25 * pderxy(Velx, vi) * unitnormal(Velx) * pderxy(Vely, ui) * unitnormal(Velx) +
                0.5 * pderxy(Vely, vi) * unitnormal(Vely) * pderxy(Vely, ui) * unitnormal(Velx));

        //(y,y)
        Amat(ivy, iuy) +=
            fac_ *
            (0.25 * pderxy(Velx, vi) * unitnormal(Vely) * pderxy(Velx, ui) * unitnormal(Vely)  // 1
                + 0.25 * pderxy(Velx, vi) * unitnormal(Velx) * pderxy(Velx, ui) *
                      unitnormal(Velx)  // 2
                + 0.5 * pderxy(Velx, vi) * unitnormal(Velx) * pderxy(Vely, ui) *
                      unitnormal(Vely)  // 3
                + 0.5 * pderxy(Vely, vi) * unitnormal(Vely) * pderxy(Velx, ui) *
                      unitnormal(Velx)                                                       // 5
                + pderxy(Vely, vi) * unitnormal(Vely) * pderxy(Vely, ui) * unitnormal(Vely)  // 6
            );

        if (nsd == 3)
        {
          //(x,x)
          Amat(ivx, iux) += fac_ * (+0.5 * pderxy(Velx, vi) * unitnormal(Velx) * pderxy(Velz, ui) *
                                           unitnormal(Velz)  // 3
                                       + 0.25 * pderxy(Vely, vi) * unitnormal(Vely) *
                                             pderxy(Velz, ui) * unitnormal(Velz)  // 6
                                       + 0.5 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Velx, ui) * unitnormal(Velx)  // 7
                                       + 0.25 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Vely, ui) * unitnormal(Vely)  // 8
                                       + 0.25 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Velz, ui) * unitnormal(Velz)  // 9
                                       + 0.25 * pderxy(Vely, vi) * unitnormal(Velx) *
                                             pderxy(Vely, ui) * unitnormal(Velx)  // 10
                                       + 0.25 * pderxy(Velz, vi) * unitnormal(Velx) *
                                             pderxy(Velz, ui) * unitnormal(Velx)  // 11
                                   );


          //(x,y)
          Amat(ivx, iuy) += fac_ * (+0.25 * pderxy(Vely, vi) * unitnormal(Velx) * pderxy(Velz, ui) *
                                           unitnormal(Velz)  // 5
                                       + 0.25 * pderxy(Velz, vi) * unitnormal(Velx) *
                                             pderxy(Velz, ui) * unitnormal(Vely)  // 6
                                       + 0.25 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Velx, ui) * unitnormal(Vely)  // 7
                                   );

          //(y,x)
          Amat(ivy, iux) +=
              fac_ *
              (+0.25 * pderxy(Velx, vi) * unitnormal(Vely) * pderxy(Velz, ui) * unitnormal(Velz) +
                  0.25 * pderxy(Velz, vi) * unitnormal(Velz) * pderxy(Vely, ui) * unitnormal(Velx) +
                  0.25 * pderxy(Velz, vi) * unitnormal(Vely) * pderxy(Velz, ui) * unitnormal(Velx));

          //(y,y)
          Amat(ivy, iuy) += fac_ * (+0.25 * pderxy(Velx, vi) * unitnormal(Velx) * pderxy(Velz, ui) *
                                           unitnormal(Velz)  // 4
                                       + 0.5 * pderxy(Vely, vi) * unitnormal(Vely) *
                                             pderxy(Velz, ui) * unitnormal(Velz)  // 7
                                       + 0.25 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Velx, ui) * unitnormal(Velx)  // 8
                                       + 0.5 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Vely, ui) * unitnormal(Vely)  // 9
                                       + 0.25 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Velz, ui) * unitnormal(Velz)  // 10
                                       + 0.25 * pderxy(Velz, vi) * unitnormal(Vely) *
                                             pderxy(Velz, ui) * unitnormal(Vely)  // 11
                                   );

          //(x,z)
          Amat(ivx, iuz) +=
              fac_ *
              (0.5 * pderxy(Velx, vi) * unitnormal(Velx) * pderxy(Velx, ui) * unitnormal(Velz) +
                  0.25 * pderxy(Vely, vi) * unitnormal(Vely) * pderxy(Velx, ui) * unitnormal(Velz) +
                  0.25 * pderxy(Velz, vi) * unitnormal(Velz) * pderxy(Velx, ui) * unitnormal(Velz) +
                  0.25 * pderxy(Vely, vi) * unitnormal(Velx) * pderxy(Vely, ui) * unitnormal(Velz) +
                  0.25 * pderxy(Velz, vi) * unitnormal(Velx) * pderxy(Velx, ui) * unitnormal(Velx) +
                  0.25 * pderxy(Velz, vi) * unitnormal(Velx) * pderxy(Vely, ui) * unitnormal(Vely) +
                  0.5 * pderxy(Velz, vi) * unitnormal(Velx) * pderxy(Velz, ui) * unitnormal(Velz));

          //(y,z)
          Amat(ivy, iuz) += fac_ * (0.25 * pderxy(Velx, vi) * unitnormal(Vely) * pderxy(Velx, ui) *
                                           unitnormal(Velz)  // 1
                                       + 0.25 * pderxy(Velx, vi) * unitnormal(Velx) *
                                             pderxy(Vely, ui) * unitnormal(Velz)  // 2
                                       + 0.5 * pderxy(Vely, vi) * unitnormal(Vely) *
                                             pderxy(Vely, ui) * unitnormal(Velz)  // 3
                                       + 0.25 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Vely, ui) * unitnormal(Velz)  // 4
                                       + 0.25 * pderxy(Velz, vi) * unitnormal(Vely) *
                                             pderxy(Velx, ui) * unitnormal(Velx)  // 5
                                       + 0.25 * pderxy(Velz, vi) * unitnormal(Vely) *
                                             pderxy(Vely, ui) * unitnormal(Vely)  // 6
                                       + 0.5 * pderxy(Velz, vi) * unitnormal(Vely) *
                                             pderxy(Velz, ui) * unitnormal(Velz)  // 7
                                   );

          //(z,x)
          Amat(ivz, iux) += fac_ * (0.5 * pderxy(Velx, vi) * unitnormal(Velz) * pderxy(Velx, ui) *
                                           unitnormal(Velx)  // 1
                                       + 0.25 * pderxy(Velx, vi) * unitnormal(Velz) *
                                             pderxy(Vely, ui) * unitnormal(Vely)  // 2
                                       + 0.25 * pderxy(Velx, vi) * unitnormal(Velz) *
                                             pderxy(Velz, ui) * unitnormal(Velz)  // 3
                                       + 0.25 * pderxy(Vely, vi) * unitnormal(Velz) *
                                             pderxy(Vely, ui) * unitnormal(Velx)  // 4
                                       + 0.25 * pderxy(Velx, vi) * unitnormal(Velx) *
                                             pderxy(Velz, ui) * unitnormal(Velx)  // 5
                                       + 0.25 * pderxy(Vely, vi) * unitnormal(Vely) *
                                             pderxy(Velz, ui) * unitnormal(Velx)  // 6
                                       + 0.5 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Velz, ui) * unitnormal(Velx)  // 7
                                   );

          //(z,y)
          Amat(ivz, iuy) += fac_ * (0.25 * pderxy(Velx, vi) * unitnormal(Velz) * pderxy(Velx, ui) *
                                           unitnormal(Vely)  // 1
                                       + 0.25 * pderxy(Vely, vi) * unitnormal(Velz) *
                                             pderxy(Velx, ui) * unitnormal(Velx)  // 2
                                       + 0.5 * pderxy(Vely, vi) * unitnormal(Velz) *
                                             pderxy(Vely, ui) * unitnormal(Vely)  // 3
                                       + 0.25 * pderxy(Vely, vi) * unitnormal(Velz) *
                                             pderxy(Velz, ui) * unitnormal(Velz)  // 4
                                       + 0.25 * pderxy(Velx, vi) * unitnormal(Velx) *
                                             pderxy(Velz, ui) * unitnormal(Vely)  // 5
                                       + 0.25 * pderxy(Vely, vi) * unitnormal(Vely) *
                                             pderxy(Velz, ui) * unitnormal(Vely)  // 6
                                       + 0.5 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Velz, ui) * unitnormal(Vely)  // 7
                                   );

          //(z,z)
          Amat(ivz, iuz) += fac_ * (0.25 * pderxy(Velx, vi) * unitnormal(Velz) * pderxy(Velx, ui) *
                                           unitnormal(Velz)  // 1
                                       + 0.25 * pderxy(Vely, vi) * unitnormal(Velz) *
                                             pderxy(Vely, ui) * unitnormal(Velz)  // 2
                                       + 0.25 * pderxy(Velx, vi) * unitnormal(Velx) *
                                             pderxy(Velx, ui) * unitnormal(Velx)  // 3
                                       + 0.25 * pderxy(Velx, vi) * unitnormal(Velx) *
                                             pderxy(Vely, ui) * unitnormal(Vely)  // 4
                                       + 0.5 * pderxy(Velx, vi) * unitnormal(Velx) *
                                             pderxy(Velz, ui) * unitnormal(Velz)  // 5
                                       + 0.25 * pderxy(Vely, vi) * unitnormal(Vely) *
                                             pderxy(Velx, ui) * unitnormal(Velx)  // 6
                                       + 0.25 * pderxy(Vely, vi) * unitnormal(Vely) *
                                             pderxy(Vely, ui) * unitnormal(Vely)  // 7
                                       + 0.5 * pderxy(Vely, vi) * unitnormal(Vely) *
                                             pderxy(Velz, ui) * unitnormal(Velz)  // 8
                                       + 0.5 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Velx, ui) * unitnormal(Velx)  // 9
                                       + 0.5 * pderxy(Velz, vi) * unitnormal(Velz) *
                                             pderxy(Vely, ui) * unitnormal(Vely)  // 10
                                       + pderxy(Velz, vi) * unitnormal(Velz) * pderxy(Velz, ui) *
                                             unitnormal(Velz)  // 11
                                   );
        }
      }
    }
  }  // GP over boundary element - End of PART I

  // is Amat symmetric?
  for (int vi = 0; vi < piel * nsd; ++vi)
  {
    for (int ui = 0; ui < piel * nsd; ++ui)
    {
      if (fabs(Amat(vi, ui) - Amat(ui, vi)) > 1e-8)
      {
        std::cout << "Warning 1: " << Amat(vi, ui) << " " << Amat(ui, vi) << " "
                  << fabs(Amat(vi, ui) - Amat(ui, vi)) << std::endl;
        FOUR_C_THROW("Amat is not symmetric!!!");
      }
    }
  }
  //  std::cout << Amat << std::endl;

  double pdet = 0.0;

  //---------------------------------------------------------------------
  // part II : integration loop for parent element
  //---------------------------------------------------------------------
  for (int iquad = 0; iquad < pintpoints.ip().nquad; ++iquad)
  {
    // coordinates of parent-element integration point
    const double* gpcoord = (pintpoints.ip().qxg)[iquad];
    for (int idim = 0; idim < nsd; idim++)
    {
      pxsi(idim) = gpcoord[idim];
    }

    // shape functions and derivatives of parent element at integration point
    Core::FE::shape_function<pdistype>(pxsi, pfunct);
    Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

    // compute (inverse of) Jacobian matrix and determinant for parent element
    // and check its value
    pxjm.multiply_nt(pderiv, pxyze);
    //    const double pdet = pxji.invert(pxjm);
    pdet = pxji.invert(pxjm);
    if (pdet < 1E-16)
      FOUR_C_THROW("GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", pid, pdet);

    // compute integration factor for boundary element
    fac_ = pintpoints.ip().qwgt[iquad] * pdet;


    //    meas_vol += fac_;

    // compute global first derivatives for parent element
    pderxy.multiply(pxji, pderiv);

    /*
  //    /                \
  //   |                  |
  //   |  eps(v), eps(u)  |
  //   |                  |
  //    \                / volume
     */

    const unsigned Velx = 0;
    const unsigned Vely = 1;
    const unsigned Velz = 2;

    for (int vi = 0; vi < piel; ++vi)
    {
      int ivx = vi * (nsd) + 0;
      int ivy = vi * (nsd) + 1;
      int ivz = vi * (nsd) + 2;

      for (int ui = 0; ui < piel; ++ui)
      {
        int iux = ui * (nsd) + 0;
        int iuy = ui * (nsd) + 1;
        int iuz = ui * (nsd) + 2;

        //(x,x)
        Bmat(ivx, iux) += fac_ * (pderxy(Velx, vi) * pderxy(Velx, ui) +
                                     0.5 * pderxy(Vely, vi) * pderxy(Vely, ui));

        //(x,y)
        Bmat(ivx, iuy) += fac_ * (0.5 * pderxy(Vely, vi) * pderxy(Velx, ui));

        //(y,x)
        Bmat(ivy, iux) += fac_ * (0.5 * pderxy(Velx, vi) * pderxy(Vely, ui));

        //(y,y)
        Bmat(ivy, iuy) += fac_ * (pderxy(Vely, vi) * pderxy(Vely, ui) +
                                     0.5 * pderxy(Velx, vi) * pderxy(Velx, ui));

        if (nsd == 3)
        {
          //(x,x)
          Bmat(ivx, iux) += fac_ * (+0.5 * pderxy(Velz, vi) * pderxy(Velz, ui));

          //(y,y)
          Bmat(ivy, iuy) += fac_ * (+0.5 * pderxy(Velz, vi) * pderxy(Velz, ui));

          //(x,z)
          Bmat(ivx, iuz) += fac_ * (0.5 * pderxy(Velz, vi) * pderxy(Velx, ui));
          //(y,z)
          Bmat(ivy, iuz) += fac_ * (0.5 * pderxy(Velz, vi) * pderxy(Vely, ui));

          //(z,x)
          Bmat(ivz, iux) += fac_ * (0.5 * pderxy(Velx, vi) * pderxy(Velz, ui));

          //(z,y)
          Bmat(ivz, iuy) += fac_ * (0.5 * pderxy(Vely, vi) * pderxy(Velz, ui));

          //(z,z)
          Bmat(ivz, iuz) += fac_ * (pderxy(Velz, vi) * pderxy(Velz, ui) +
                                       0.5 * pderxy(Velx, vi) * pderxy(Velx, ui) +
                                       0.5 * pderxy(Vely, vi) * pderxy(Vely, ui));
        }
      }
    }

  }  // gauss loop

  // is Bmat symmetric?
  for (int vi = 0; vi < piel * nsd; ++vi)
  {
    for (int ui = 0; ui < piel * nsd; ++ui)
    {
      if (fabs(Bmat(vi, ui) - Bmat(ui, vi)) > 1E-14)
      {
        std::cout << "Warning: " << Bmat(vi, ui) << " " << Bmat(ui, vi) << std::endl;
        FOUR_C_THROW("Bmat is not symmetric!!!");
      }
    }
  }

  // Solve the local eigen value problem Ax = lambda Bx. The function generalized_eigen
  // returns the maximum Eigenvalue of the problem.
  const double maxeigenvalue =
      Core::LinAlg::generalized_eigen_max_real_eigenvalue(elemat1, elemat2);

  // fill the map: every side id has it's own parameter beta
  (*params.get<std::shared_ptr<std::map<int, double>>>(
      "trace_estimate_max_eigenvalue_map"))[surfele->id()] = maxeigenvalue;

  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
template <Core::FE::CellType bdistype, Core::FE::CellType pdistype>
void Discret::Elements::FluidBoundaryParent<distype>::mix_hyb_dirichlet(
    Discret::Elements::FluidBoundary* surfele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& plm,
    Core::LinAlg::SerialDenseMatrix& elemat, Core::LinAlg::SerialDenseVector::Base& elevec)
{
  //--------------------------------------------------
  // get my parent element
  Core::Elements::Element* parent = surfele->parent_element();

  // evaluate material at integration point
  const double rateofstrain = 0.0;
  std::shared_ptr<Core::Mat::Material> material = parent->material();

  if (material->material_type() == Core::Materials::m_carreauyasuda or
      material->material_type() == Core::Materials::m_modpowerlaw or
      material->material_type() == Core::Materials::m_herschelbulkley)
    FOUR_C_THROW("No non-Newtonian fluid allowed for mixed/hybrid DBCs so far!");

  // get viscosity
  get_density_and_viscosity(material, 0.0, 0.0, rateofstrain);

  // only constant density of 1.0 allowed, for the time being
  if (densaf_ != 1.0)
    FOUR_C_THROW("Only incompressible flow with density 1.0 allowed for weak DBCs so far!");

  /// number of parentnodes
  static const int piel = Core::FE::num_nodes(pdistype);

  /// number of surfacenodes
  static const int biel = Core::FE::num_nodes(bdistype);

  /// number of spatial dimensions
  static const int nsd = Core::FE::dim<pdistype>;

  static const int bnsd = Core::FE::dim<bdistype>;

  // number of internal stress dofs is equivalent to number of second derivatives
  static const int numstressdof_ = Core::FE::DisTypeToNumDeriv2<pdistype>::numderiv2;

  if (fldparatimint_->time_algo() == Inpar::FLUID::timeint_afgenalpha)
    FOUR_C_THROW(
        "The use of mixed hybrid boundary conditions and Afgenalpha has not been verified so far!");

  // --------------------------------------------------
  // Reshape element matrices and vectors and init to zero, construct views
  const int peledim = (nsd + 1) * piel;

  elemat.shape(peledim, peledim);
  elevec.size(peledim);

  Core::LinAlg::Matrix<peledim, peledim> elemat1(elemat.values(), true);
  Core::LinAlg::Matrix<peledim, 1> elevec1(elevec.values(), true);

  //--------------------------------------------------
  // get the condition information
  const auto* hixhybdbc_cond = params.get<const Core::Conditions::Condition*>("condition");

  // get value for boundary condition
  const auto& val = (*hixhybdbc_cond).parameters().get<std::vector<double>>("val");

  const int myid = (*((*hixhybdbc_cond).get_nodes()))[0];

  // TODO: which time (n+1) or (n+alphaF)
  // find out whether we will use a time curve
  const double time = fldparatimint_->time();

  // initialise scaling for distance to wall (Spalding)
  double hB_divided_by = 1.0;

  // get a characteristic velocity
  double u_C = hixhybdbc_cond->parameters().get<double>("u_C");

  // decide whether to use it or not
  const std::string deftauB = hixhybdbc_cond->parameters().get<std::string>("PENTYPE");

  bool spalding = false;

  if (deftauB == "Spalding")
  {
    spalding = true;

    // get actual scaling
    hB_divided_by = hixhybdbc_cond->parameters().get<double>("hB_divided_by");
  }
  else if (deftauB == "constant")
  {
    spalding = false;
  }
  else
  {
    FOUR_C_THROW("Unknown definition of penalty parameter: {}", deftauB);
  }

  // flag for utau computation (viscous tangent or at wall (a la Michler))
  const std::string utau_computation =
      hixhybdbc_cond->parameters().get<std::string>("utau_computation");

  // get values and switches from the condition
  // (assumed to be constant on element boundary)
  const auto functions = hixhybdbc_cond->parameters().get<std::vector<std::optional<int>>>("funct");

  Core::LinAlg::Matrix<nsd, 1> u_dirich(Core::LinAlg::Initialization::zero);

  for (int rr = 0; rr < nsd; ++rr)
  {
    u_dirich(rr) = val[rr];
  }

  // --------------------------------------------------
  // Extra matrices

  // for r / sigma: indices ordered according to
  //
  //      0    1    2    (2D)
  //     11 , 22 , 12
  //

  //      0    1    2    3    4   5  (3D)
  //     11 , 22 , 33 , 12 , 13 ,23

  // for volume integrals

  Core::LinAlg::Matrix<numstressdof_ * piel, piel> mat_r_p(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numstressdof_ * piel, numstressdof_ * piel> mat_r_sigma(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numstressdof_ * piel, nsd * piel> mat_r_epsu(
      Core::LinAlg::Initialization::zero);

  // for boundary integrals

  Core::LinAlg::Matrix<nsd * piel, numstressdof_ * piel> mat_v_sigma_o_n(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numstressdof_ * piel, nsd * piel> mat_r_o_n_u(
      Core::LinAlg::Initialization::zero);

  // rearranging and computational arrays

  Core::LinAlg::Matrix<numstressdof_ * piel, (nsd + 1) * piel> mat_r_up_block(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numstressdof_ * piel, numstressdof_ * piel> inv_r_sigma(
      Core::LinAlg::Initialization::zero);


  // --------------------------------------------------
  // Extra vectors

  // for volume integrals

  Core::LinAlg::Matrix<numstressdof_ * piel, 1> vec_r_p(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numstressdof_ * piel, 1> vec_r_epsu(Core::LinAlg::Initialization::zero);

  // for boundary integrals
  Core::LinAlg::Matrix<numstressdof_ * piel, 1> vec_r_o_n_u_minus_g(
      Core::LinAlg::Initialization::zero);

  // extract local velocities and pressure from the global vectors
  Core::LinAlg::Matrix<nsd, piel> pevel(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<piel, 1> pepres(Core::LinAlg::Initialization::zero);

  std::shared_ptr<const Core::LinAlg::Vector<double>> vel = discretization.get_state("velaf");
  if (vel == nullptr) FOUR_C_THROW("Cannot get state vector 'velaf'");

  // extract local node values for pressure and velocities from global vectors
  if (fldparatimint_->time_algo() == Inpar::FLUID::timeint_npgenalpha)
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> velnp = discretization.get_state("velnp");
    if (velnp == nullptr) FOUR_C_THROW("Cannot get state vector 'velnp'");

    double maxvel = 0;

    std::vector<double> mypvelaf((plm).size());
    std::vector<double> mypvelnp((plm).size());

    mypvelaf = Core::FE::extract_values(*vel, plm);
    mypvelnp = Core::FE::extract_values(*velnp, plm);

    for (int inode = 0; inode < piel; ++inode)
    {
      double normvel = 0;

      for (int idim = 0; idim < nsd; ++idim)
      {
        pevel(idim, inode) = mypvelaf[(nsd + 1) * inode + idim];

        normvel += pevel(idim, inode) * pevel(idim, inode);
      }
      normvel = sqrt(normvel);

      if (normvel > maxvel)
      {
        maxvel = normvel;
      }

      pepres(inode) = mypvelnp[(nsd + 1) * inode + nsd];
    }

    // eventually set the characteristic velocity to the maximum value on that element
    if (u_C < 0)
    {
      u_C = maxvel;
    }
  }
  else
  {
    std::vector<double> mypvel = Core::FE::extract_values(*vel, plm);


    for (int inode = 0; inode < piel; ++inode)
    {
      for (int idim = 0; idim < nsd; ++idim)
      {
        pevel(idim, inode) = mypvel[(nsd + 1) * inode + idim];
      }
      pepres(inode) = mypvel[(nsd + 1) * inode + nsd];
    }
  }


  /*<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
         PART 1: Gaussloop for volume integrals of parent element
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>*/
  {
    // allocate vector for shape functions and matrix for derivatives
    Core::LinAlg::Matrix<piel, 1> pfunct(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd, piel> pderiv(Core::LinAlg::Initialization::zero);

    // get local node coordinates
    Core::LinAlg::Matrix<nsd, piel> pxyze(Core::LinAlg::Initialization::zero);
    Core::Geo::fill_initial_position_array<pdistype, nsd, Core::LinAlg::Matrix<nsd, piel>>(
        parent, pxyze);

    //--------------------------------------------------
    // Gaussian integration points
    const Core::FE::IntPointsAndWeights<nsd> pintpoints(
        Discret::Elements::DisTypeToOptGaussRule<pdistype>::rule);

    //--------------------------------------------------
    // vectors/scalars for Gausspoint values

    // velocity at gausspoint
    Core::LinAlg::Matrix<nsd, 1> pvelint(Core::LinAlg::Initialization::zero);
    // velocity derivatives at gausspoint
    Core::LinAlg::Matrix<nsd, nsd> pvderxy(Core::LinAlg::Initialization::zero);
    // pressure at gausspoint
    double ppressure = 0.0;

    // global derivatives of shape functions w.r.t x,y,z
    Core::LinAlg::Matrix<nsd, piel> pderxy(Core::LinAlg::Initialization::zero);
    // transposed jacobian "dx/ds"
    Core::LinAlg::Matrix<nsd, nsd> pxjm(Core::LinAlg::Initialization::zero);
    // inverse of transposed jacobian "ds/dx"
    Core::LinAlg::Matrix<nsd, nsd> pxji(Core::LinAlg::Initialization::zero);

    Core::LinAlg::Matrix<nsd, 1> pxsi(Core::LinAlg::Initialization::zero);

    //--------------------------------------------------
    // the actual loop
    for (int iquad = 0; iquad < pintpoints.ip().nquad; ++iquad)
    {
      // coordinates of the current integration point
      const double* gpcoord = (pintpoints.ip().qxg)[iquad];
      for (int idim = 0; idim < nsd; idim++)
      {
        pxsi(idim) = gpcoord[idim];
      }

      // get parent elements shape functions
      Core::FE::shape_function<pdistype>(pxsi, pfunct);
      Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

      // get Jacobian matrix and determinant
      // actually compute its transpose....
      /*
        +-            -+ T      +-            -+
        | dx   dx   dx |        | dx   dy   dz |
        | --   --   -- |        | --   --   -- |
        | dr   ds   dt |        | dr   dr   dr |
        |              |        |              |
        | dy   dy   dy |        | dx   dy   dz |
        | --   --   -- |   =    | --   --   -- |
        | dr   ds   dt |        | ds   ds   ds |
        |              |        |              |
        | dz   dz   dz |        | dx   dy   dz |
        | --   --   -- |        | --   --   -- |
        | dr   ds   dt |        | dt   dt   dt |
        +-            -+        +-            -+
      */
      pxjm.multiply_nt(pderiv, pxyze);
      const double det = pxji.invert(pxjm);

      if (det < 1E-16)
        FOUR_C_THROW(
            "GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", parent->id(), det);

      // compute integration factor
      fac_ = pintpoints.ip().qwgt[iquad] * det;

      // compute global first derivatives
      pderxy.multiply(pxji, pderiv);

      // interpolate to gausspoint
      pvelint.multiply(pevel, pfunct);

      // get velocity derivatives at integration point
      pvderxy.multiply_nt(pevel, pderxy);

      // interpolate pressure to gausspoint
      ppressure = pfunct.dot(pepres);

      /*
                            /          \
                      1    |  h       h |
                  - ---- * | r : sigma  |
                    2*nu   |            |
                            \          / Omega
      */

      const double fac_twoviscinv = fac_ / (2.0 * visc_);
      const double fac_viscinv = fac_ / visc_;
      for (int A = 0; A < piel; ++A)
      {
        const double fac_twoviscinv_pfunctA = fac_twoviscinv * pfunct(A);
        const double fac_viscinv_pfunctA = fac_viscinv * pfunct(A);

        for (int B = 0; B < piel; ++B)
        {
          for (int i = 0; i < nsd; ++i)
          {
            mat_r_sigma(A * numstressdof_ + i, B * numstressdof_ + i) -=
                fac_twoviscinv_pfunctA * pfunct(B);
          }
          for (int i = nsd; i < numstressdof_; ++i)
          {
            mat_r_sigma(A * numstressdof_ + i, B * numstressdof_ + i) -=
                fac_viscinv_pfunctA * pfunct(B);
          }
        }
      }

      /*
                            /         \
                      1    |  h   h    |
                  - ---- * | r : p * I |
                    2*nu   |           |
                            \         / Omega
      */
      for (int A = 0; A < piel; ++A)
      {
        const double fac_twoviscinv_pfunctA = fac_twoviscinv * pfunct(A);
        for (int B = 0; B < piel; ++B)
        {
          const double aux = fac_twoviscinv_pfunctA * pfunct(B);

          for (int i = 0; i < nsd; ++i)
          {
            mat_r_p(A * numstressdof_ + i, B) -= aux;
          }
        }
      }

      for (int A = 0; A < piel; ++A)
      {
        const double fac_twoviscinv_pfunctA_pressure = fac_twoviscinv * pfunct(A) * ppressure;
        for (int i = 0; i < nsd; ++i)
        {
          vec_r_p(A * numstressdof_ + i) -= fac_twoviscinv_pfunctA_pressure;
        }
      }


      /*
                     /              \
                    |  h       / h\  |
                  + | r : eps | u  | |
                    |          \  /  |
                     \              / Omega
      */
      if (nsd == 2)
      {
        for (int A = 0; A < piel; ++A)
        {
          for (int B = 0; B < piel; ++B)
          {
            mat_r_epsu(A * numstressdof_, B * nsd) += fac_ * pfunct(A) * pderxy(0, B);
            mat_r_epsu(A * numstressdof_ + 1, B * nsd + 1) += fac_ * pfunct(A) * pderxy(1, B);

            mat_r_epsu(A * numstressdof_ + 2, B * nsd) += fac_ * pfunct(A) * pderxy(1, B);
            mat_r_epsu(A * numstressdof_ + 2, B * nsd + 1) += fac_ * pfunct(A) * pderxy(0, B);
          }
        }
      }
      else if (nsd == 3)
      {
        for (int A = 0; A < piel; ++A)
        {
          const double fac_pfunctA = fac_ * pfunct(A);

          for (int B = 0; B < piel; ++B)
          {
            mat_r_epsu(A * numstressdof_, B * nsd) += fac_pfunctA * pderxy(0, B);
            mat_r_epsu(A * numstressdof_ + 1, B * nsd + 1) += fac_pfunctA * pderxy(1, B);
            mat_r_epsu(A * numstressdof_ + 2, B * nsd + 2) += fac_pfunctA * pderxy(2, B);

            mat_r_epsu(A * numstressdof_ + 3, B * nsd) += fac_pfunctA * pderxy(1, B);
            mat_r_epsu(A * numstressdof_ + 3, B * nsd + 1) += fac_pfunctA * pderxy(0, B);

            mat_r_epsu(A * numstressdof_ + 4, B * nsd) += fac_pfunctA * pderxy(2, B);
            mat_r_epsu(A * numstressdof_ + 4, B * nsd + 2) += fac_pfunctA * pderxy(0, B);

            mat_r_epsu(A * numstressdof_ + 5, B * nsd + 1) += fac_pfunctA * pderxy(2, B);
            mat_r_epsu(A * numstressdof_ + 5, B * nsd + 2) += fac_pfunctA * pderxy(1, B);
          }
        }
      }

      if (nsd == 2)
      {
        for (int A = 0; A < piel; ++A)
        {
          vec_r_epsu(A * numstressdof_) += fac_ * pfunct(A) * pvderxy(0, 0);
          vec_r_epsu(A * numstressdof_ + 1) += fac_ * pfunct(A) * pvderxy(1, 1);

          vec_r_epsu(A * numstressdof_ + 2) += fac_ * pfunct(A) * (pvderxy(0, 1) + pvderxy(1, 0));
        }
      }
      else if (nsd == 3)
      {
        Core::LinAlg::Matrix<numstressdof_, 1> temp(Core::LinAlg::Initialization::zero);

        temp(0) = fac_ * pvderxy(0, 0);
        temp(1) = fac_ * pvderxy(1, 1);
        temp(2) = fac_ * pvderxy(2, 2);
        temp(3) = fac_ * (pvderxy(0, 1) + pvderxy(1, 0));
        temp(4) = fac_ * (pvderxy(0, 2) + pvderxy(2, 0));
        temp(5) = fac_ * (pvderxy(1, 2) + pvderxy(2, 1));

        for (int A = 0; A < piel; ++A)
        {
          vec_r_epsu(A * numstressdof_) += temp(0) * pfunct(A);
          vec_r_epsu(A * numstressdof_ + 1) += temp(1) * pfunct(A);
          vec_r_epsu(A * numstressdof_ + 2) += temp(2) * pfunct(A);

          vec_r_epsu(A * numstressdof_ + 3) += temp(3) * pfunct(A);
          vec_r_epsu(A * numstressdof_ + 4) += temp(4) * pfunct(A);
          vec_r_epsu(A * numstressdof_ + 5) += temp(5) * pfunct(A);
        }
      }
    }
  }

  /*<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
         PART 2: Matrix inversion
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>*/

  // matrix inversion of stress-stress block
  inv_r_sigma = mat_r_sigma;

  Core::LinAlg::FixedSizeSerialDenseSolver<numstressdof_ * piel, numstressdof_ * piel> solver;

  solver.set_matrix(inv_r_sigma);
  solver.invert();

  /*<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
         PART 2.1: Include Spaldings law
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>*/

  double normtraction = 0.0;
  {
    if (nsd == 3)
    {
      // for boundary integrals
      Core::LinAlg::Matrix<numstressdof_ * piel, 1> vec_r_o_n_u_minus_g_SPALDING(
          Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<numstressdof_ * piel, 1> SPALDING_stresses(
          Core::LinAlg::Initialization::zero);

      // allocate vector/matrix for shape functions and derivatives
      Core::LinAlg::Matrix<biel, 1> funct(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<bnsd, biel> deriv(Core::LinAlg::Initialization::zero);

      // allocate vector for parents shape functions and matrix for derivatives
      Core::LinAlg::Matrix<piel, 1> pfunct(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd, piel> pderiv(Core::LinAlg::Initialization::zero);

      // get local node coordinates
      Core::LinAlg::Matrix<nsd, biel> bxyze(Core::LinAlg::Initialization::zero);
      Core::Geo::fill_initial_position_array<bdistype, nsd, Core::LinAlg::Matrix<nsd, biel>>(
          surfele, bxyze);

      // get local node coordinates
      Core::LinAlg::Matrix<nsd, piel> pxyze(Core::LinAlg::Initialization::zero);
      Core::Geo::fill_initial_position_array<pdistype, nsd, Core::LinAlg::Matrix<nsd, piel>>(
          parent, pxyze);

      //--------------------------------------------------
      // Gaussian integration points
      const Core::FE::IntPointsAndWeights<bnsd> intpoints(
          Discret::Elements::DisTypeToOptGaussRule<bdistype>::rule);

      const Core::FE::IntPointsAndWeights<nsd> pintpoints(
          Discret::Elements::DisTypeToOptGaussRule<pdistype>::rule);

      // coordinates of current integration point in reference coordinates
      Core::LinAlg::Matrix<bnsd, 1> xsi(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd, 1> pxsi(Core::LinAlg::Initialization::zero);


      Core::LinAlg::SerialDenseMatrix pqxg(pintpoints.ip().nquad, nsd);

      {
        Core::LinAlg::SerialDenseMatrix gps(intpoints.ip().nquad, bnsd);


        // coordinates of the current integration point
        for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
        {
          const double* gpcoord = (intpoints.ip().qxg)[iquad];

          for (int idim = 0; idim < bnsd; idim++)
          {
            gps(iquad, idim) = gpcoord[idim];
          }
        }
        Core::FE::boundary_gp_to_parent_gp3(
            pqxg, gps, pdistype, bdistype, surfele->surface_number());
      }


      //--------------------------------------------------
      // vectors/scalars for Gausspoint values

      // the element's normal vector
      Core::LinAlg::Matrix<nsd, 1> unitnormal(Core::LinAlg::Initialization::zero);
      // velocity at gausspoint
      Core::LinAlg::Matrix<nsd, 1> velint(Core::LinAlg::Initialization::zero);

      // transposed jacobian "dx/ds"
      Core::LinAlg::Matrix<nsd, nsd> xjm(Core::LinAlg::Initialization::zero);
      // inverse of transposed jacobian "ds/dx"
      Core::LinAlg::Matrix<nsd, nsd> xji(Core::LinAlg::Initialization::zero);

      // transposed jacobian "dx/ds" for parent
      Core::LinAlg::Matrix<nsd, nsd> pxjm(Core::LinAlg::Initialization::zero);
      // inverse of transposed jacobian "ds/dx" for parent
      Core::LinAlg::Matrix<nsd, nsd> pxji(Core::LinAlg::Initialization::zero);


      //--------------------------------------------------
      // the actual integration loop to compute the current normalised stresses
      for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
      {
        // coordinates of the current integration point
        const double* gpcoord = (intpoints.ip().qxg)[iquad];
        for (int idim = 0; idim < bnsd; idim++)
        {
          xsi(idim) = gpcoord[idim];
        }

        Core::FE::shape_function<bdistype>(xsi, funct);
        Core::FE::shape_function_deriv1<bdistype>(xsi, deriv);

        for (int idim = 0; idim < nsd; idim++)
        {
          pxsi(idim) = pqxg(iquad, idim);
        }

        Core::FE::shape_function<pdistype>(pxsi, pfunct);
        Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

        drs_ = 0.0;

        // compute measure tensor for surface element and the infinitesimal
        // area element drs for the integration
        Core::LinAlg::Matrix<bnsd, bnsd> metrictensor(Core::LinAlg::Initialization::zero);

        Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
            bxyze, deriv, metrictensor, drs_, &unitnormal);

        // compute integration factor
        fac_ = intpoints.ip().qwgt[iquad] * drs_;

        // interpolate to gausspoint
        velint.multiply(pevel, pfunct);

        // ------------------------------------------------
        // factor given by spatial function
        Core::LinAlg::Matrix<nsd, 1> functionfac(Core::LinAlg::Initialization::zero);
        for (int i = 0; i < nsd; ++i)
        {
          functionfac(i) = 1.0;
        }

        // determine coordinates of current Gauss point
        Core::LinAlg::Matrix<3, 1> coordgp(Core::LinAlg::Initialization::zero);

        for (int A = 0; A < biel; ++A)
        {
          for (int j = 0; j < nsd; ++j)
          {
            coordgp(j) += bxyze(j, A) * funct(A);
          }
        }

        for (int dim = 0; dim < nsd; ++dim)
        {
          // factor given by spatial function
          if (functions[dim].has_value() && functions[dim].value() > 0)
          {
            // evaluate function at current gauss point (important: requires 3D position vector)
            functionfac(dim) =
                Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functions[dim].value())
                    .evaluate(coordgp.data(), time, dim);
          }
          else
          {
            functionfac(dim) = 1.0;
          }
        }

        Core::LinAlg::Matrix<nsd, 1> delta_vel(Core::LinAlg::Initialization::zero);

        for (int rr = 0; rr < nsd; ++rr)
        {
          delta_vel(rr) = velint(rr) - u_dirich(rr) * functionfac(rr);
        }


        //--------------------------------------------------
        // adjoint consistency term, tangential stress part (normalised)

        /*
                     /                        \
                    |  h       /         \   h |
                  - | r o n , | 1 - n x n | u  |
                    |          \         /     |
                     \                        / Gamma
        */

        for (int A = 0; A < piel; ++A)
        {
          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_) -=
              fac_ * pfunct(A) * unitnormal(0) * delta_vel(0);
          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_ + 1) -=
              fac_ * pfunct(A) * unitnormal(1) * delta_vel(1);
          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_ + 2) -=
              fac_ * pfunct(A) * unitnormal(2) * delta_vel(2);

          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_ + 3) -=
              fac_ * pfunct(A) * (unitnormal(1) * delta_vel(0) + unitnormal(0) * delta_vel(1));
          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_ + 4) -=
              fac_ * pfunct(A) * (unitnormal(2) * delta_vel(0) + unitnormal(0) * delta_vel(2));
          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_ + 5) -=
              fac_ * pfunct(A) * (unitnormal(2) * delta_vel(1) + unitnormal(1) * delta_vel(2));
        }

        double u_o_n = unitnormal(0) * delta_vel(0) + unitnormal(1) * delta_vel(1) +
                       unitnormal(2) * delta_vel(2);

        for (int A = 0; A < piel; ++A)
        {
          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_) +=
              fac_ * pfunct(A) * unitnormal(0) * unitnormal(0) * u_o_n;
          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_ + 1) +=
              fac_ * pfunct(A) * unitnormal(1) * unitnormal(1) * u_o_n;
          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_ + 2) +=
              fac_ * pfunct(A) * unitnormal(2) * unitnormal(2) * u_o_n;

          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_ + 3) +=
              fac_ * pfunct(A) * 2 * unitnormal(0) * unitnormal(1) * u_o_n;
          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_ + 4) +=
              fac_ * pfunct(A) * 2 * unitnormal(0) * unitnormal(2) * u_o_n;
          vec_r_o_n_u_minus_g_SPALDING(A * numstressdof_ + 5) +=
              fac_ * pfunct(A) * 2 * unitnormal(1) * unitnormal(2) * u_o_n;
        }
      }

      for (int rr = 0; rr < numstressdof_ * piel; ++rr)
      {
        for (int mm = 0; mm < numstressdof_ * piel; ++mm)
        {
          SPALDING_stresses(rr) += inv_r_sigma(rr, mm) * vec_r_o_n_u_minus_g_SPALDING(mm);
        }
      }

      double area = 0.0;

      //--------------------------------------------------
      // compute the norm of the tangential traction
      for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
      {
        // traction and stress at gausspoint
        Core::LinAlg::Matrix<numstressdof_, 1> GP_stress(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nsd, 1> traction(Core::LinAlg::Initialization::zero);

        // coordinates of the current integration point
        const double* gpcoord = (intpoints.ip().qxg)[iquad];
        for (int idim = 0; idim < bnsd; idim++)
        {
          xsi(idim) = gpcoord[idim];
        }

        Core::FE::shape_function<bdistype>(xsi, funct);
        Core::FE::shape_function_deriv1<bdistype>(xsi, deriv);

        for (int idim = 0; idim < nsd; idim++)
        {
          pxsi(idim) = pqxg(iquad, idim);
        }

        Core::FE::shape_function<pdistype>(pxsi, pfunct);
        Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

        drs_ = 0.0;

        // compute measure tensor for surface element and the infinitesimal
        // area element drs for the integration
        Core::LinAlg::Matrix<bnsd, bnsd> metrictensor(Core::LinAlg::Initialization::zero);

        Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
            bxyze, deriv, metrictensor, drs_, &unitnormal);

        // compute integration factor
        fac_ = intpoints.ip().qwgt[iquad] * drs_;

        // interpolate to gausspoint
        for (int A = 0; A < piel; ++A)
        {
          for (int i = 0; i < numstressdof_; ++i)
          {
            GP_stress(i) += SPALDING_stresses(A * numstressdof_ + i) * pfunct(A);
          }
        }

        // multiply by normal to obtain the traction
        traction(0) = GP_stress(0) * unitnormal(0) + GP_stress(3) * unitnormal(1) +
                      GP_stress(4) * unitnormal(2);
        traction(1) = GP_stress(3) * unitnormal(0) + GP_stress(1) * unitnormal(1) +
                      GP_stress(5) * unitnormal(2);
        traction(2) = GP_stress(4) * unitnormal(0) + GP_stress(5) * unitnormal(1) +
                      GP_stress(2) * unitnormal(2);

        if (parent->id() == 1)
        {
          printf("traction (%12.5e,%12.5e,%12.5e)\n", traction(0), traction(1), traction(2));
        }

        //             /
        //            |
        // || t ||  = | sqrt ( t_1**2 + t_2**2 + t_3**2 ) dOmega
        //            |
        //           / Omega
        normtraction += fac_ * sqrt(traction(0) * traction(0) + traction(1) * traction(1) +
                                    traction(2) * traction(2));

        area += fac_;
      }

      // compute averaged norm of traction by division by area element
      if (area < 1e-13) FOUR_C_THROW("Area too small, zero or even negative!");
      normtraction /= area;
    }  // if (nsd==3)
  }

  /*<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
         PART 3: Gaussloop for integrals on boundary element
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>*/
  {
    // allocate vector/matrix for shape functions and derivatives
    Core::LinAlg::Matrix<biel, 1> funct(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<bnsd, biel> deriv(Core::LinAlg::Initialization::zero);

    // allocate vector for parents shape functions and matrix for derivatives
    Core::LinAlg::Matrix<piel, 1> pfunct(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd, piel> pderiv(Core::LinAlg::Initialization::zero);


    // get local node coordinates
    Core::LinAlg::Matrix<nsd, biel> bxyze(Core::LinAlg::Initialization::zero);
    Core::Geo::fill_initial_position_array<bdistype, nsd, Core::LinAlg::Matrix<nsd, biel>>(
        surfele, bxyze);

    // get local node coordinates
    Core::LinAlg::Matrix<nsd, piel> pxyze(Core::LinAlg::Initialization::zero);
    Core::Geo::fill_initial_position_array<pdistype, nsd, Core::LinAlg::Matrix<nsd, piel>>(
        parent, pxyze);

    //--------------------------------------------------
    // Gaussian integration points
    const Core::FE::IntPointsAndWeights<bnsd> intpoints(
        Discret::Elements::DisTypeToOptGaussRule<bdistype>::rule);

    const Core::FE::IntPointsAndWeights<nsd> pintpoints(
        Discret::Elements::DisTypeToOptGaussRule<pdistype>::rule);

    // coordinates of current integration point in reference coordinates
    Core::LinAlg::Matrix<bnsd, 1> xsi(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd, 1> pxsi(Core::LinAlg::Initialization::zero);


    Core::LinAlg::SerialDenseMatrix pqxg(pintpoints.ip().nquad, nsd);

    {
      Core::LinAlg::SerialDenseMatrix gps(intpoints.ip().nquad, bnsd);


      // coordinates of the current integration point
      for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
      {
        const double* gpcoord = (intpoints.ip().qxg)[iquad];

        for (int idim = 0; idim < bnsd; idim++)
        {
          gps(iquad, idim) = gpcoord[idim];
        }
      }
      if (nsd == 2)
      {
        Core::FE::boundary_gp_to_parent_gp2(
            pqxg, gps, pdistype, bdistype, surfele->surface_number());
      }
      else if (nsd == 3)
      {
        Core::FE::boundary_gp_to_parent_gp3(
            pqxg, gps, pdistype, bdistype, surfele->surface_number());
      }
    }


    //--------------------------------------------------
    // vectors/scalars for Gausspoint values

    // the element's normal vector
    Core::LinAlg::Matrix<nsd, 1> unitnormal(Core::LinAlg::Initialization::zero);
    // velocity at gausspoint
    Core::LinAlg::Matrix<nsd, 1> velint(Core::LinAlg::Initialization::zero);

    // transposed jacobian "dx/ds"
    Core::LinAlg::Matrix<nsd, nsd> xjm(Core::LinAlg::Initialization::zero);
    // inverse of transposed jacobian "ds/dx"
    Core::LinAlg::Matrix<nsd, nsd> xji(Core::LinAlg::Initialization::zero);

    // transposed jacobian "dx/ds" for parent
    Core::LinAlg::Matrix<nsd, nsd> pxjm(Core::LinAlg::Initialization::zero);
    // inverse of transposed jacobian "ds/dx" for parent
    Core::LinAlg::Matrix<nsd, nsd> pxji(Core::LinAlg::Initialization::zero);


    //--------------------------------------------------
    // the actual loop
    for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
    {
      // coordinates of the current integration point
      const double* gpcoord = (intpoints.ip().qxg)[iquad];
      for (int idim = 0; idim < bnsd; idim++)
      {
        xsi(idim) = gpcoord[idim];
      }

      Core::FE::shape_function<bdistype>(xsi, funct);
      Core::FE::shape_function_deriv1<bdistype>(xsi, deriv);

      for (int idim = 0; idim < nsd; idim++)
      {
        pxsi(idim) = pqxg(iquad, idim);
      }

      Core::FE::shape_function<pdistype>(pxsi, pfunct);
      Core::FE::shape_function_deriv1<pdistype>(pxsi, pderiv);

      drs_ = 0.0;

      // compute measure tensor for surface element and the infinitesimal
      // area element drs for the integration
      Core::LinAlg::Matrix<bnsd, bnsd> metrictensor(Core::LinAlg::Initialization::zero);

      Core::FE::compute_metric_tensor_for_boundary_ele<bdistype>(
          bxyze, deriv, metrictensor, drs_, &unitnormal);

      // compute integration factor
      fac_ = intpoints.ip().qwgt[iquad] * drs_;

      // get Jacobian matrix and determinant
      // actually compute its transpose....
      /*
        +-            -+ T      +-            -+
        | dx   dx   dx |        | dx   dy   dz |
        | --   --   -- |        | --   --   -- |
        | dr   ds   dt |        | dr   dr   dr |
        |              |        |              |
        | dy   dy   dy |        | dx   dy   dz |
        | --   --   -- |   =    | --   --   -- |
        | dr   ds   dt |        | ds   ds   ds |
        |              |        |              |
        | dz   dz   dz |        | dx   dy   dz |
        | --   --   -- |        | --   --   -- |
        | dr   ds   dt |        | dt   dt   dt |
        +-            -+        +-            -+
      */
      pxjm.multiply_nt(pderiv, pxyze);
      const double det = pxji.invert(pxjm);

      if (det < 1E-16)
        FOUR_C_THROW(
            "GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", parent->id(), det);

      //-----------------------------------------------------
      /*          +-           -+   +-           -+   +-           -+
                  |             |   |             |   |             |
                  |  dr    dr   |   |  ds    ds   |   |  dt    dt   |
            G   = |  --- * ---  | + |  --- * ---  | + |  --- * ---  |
             ij   |  dx    dx   |   |  dx    dx   |   |  dx    dx   |
                  |    i     j  |   |    i     j  |   |    i     j  |
                  +-           -+   +-           -+   +-           -+
      */
      Core::LinAlg::Matrix<nsd, nsd> G;

      for (int nn = 0; nn < nsd; ++nn)
      {
        for (int rr = 0; rr < nsd; ++rr)
        {
          G(nn, rr) = pxji(nn, 0) * pxji(rr, 0);
          for (int mm = 1; mm < nsd; ++mm)
          {
            G(nn, rr) += pxji(nn, mm) * pxji(rr, mm);
          }
        }
      }

      //
      //                           2.0
      //             h  = ---------------------
      //              b        +-------------+
      //                      / /  T       \ |
      //                   \ / |  n * G * n |
      //                    +   \          /
      //

      double nGn = 0;
      for (int nn = 0; nn < nsd; ++nn)
      {
        for (int rr = 0; rr < nsd; ++rr)
        {
          nGn += unitnormal(rr) * G(rr, nn) * unitnormal(nn);
        }
      }
      if (nGn < 1e-14) FOUR_C_THROW("nGn is zero or negative!");
      const double h = 2.0 / sqrt(nGn);

      // interpolate to gausspoint
      velint.multiply(pevel, pfunct);

      // ------------------------------------------------
      // factor given by spatial function
      Core::LinAlg::Matrix<nsd, 1> functionfac(Core::LinAlg::Initialization::zero);
      for (int i = 0; i < nsd; ++i)
      {
        functionfac(i) = 1.0;
      }

      // determine coordinates of current Gauss point
      Core::LinAlg::Matrix<3, 1> coordgp(Core::LinAlg::Initialization::zero);

      for (int A = 0; A < biel; ++A)
      {
        for (int j = 0; j < nsd; ++j)
        {
          coordgp(j) += bxyze(j, A) * funct(A);
        }
      }

      for (int dim = 0; dim < nsd; ++dim)
      {
        // factor given by spatial function
        if (functions[dim].has_value() && functions[dim].value() > 0)
        {
          // evaluate function at current gauss point (important: requires 3D position vector)
          functionfac(dim) =
              Global::Problem::instance()
                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(functions[dim].value())
                  .evaluate(coordgp.data(), time, dim);
        }
        else
        {
          functionfac(dim) = 1.0;
        }
      }

      Core::LinAlg::Matrix<nsd, 1> delta_vel(Core::LinAlg::Initialization::zero);

      for (int rr = 0; rr < nsd; ++rr)
      {
        delta_vel(rr) = velint(rr) - u_dirich(rr) * functionfac(rr);
      }


      /*
                              /              \
                             |  h       h     |
                           - | v , sigma  o n |
                             |                |
                              \              / Gamma
      */
      if (nsd == 2)
      {
        for (int A = 0; A < piel; ++A)
        {
          for (int B = 0; B < piel; ++B)
          {
            mat_v_sigma_o_n(A * nsd, B * numstressdof_) -=
                fac_ * pfunct(A) * pfunct(B) * unitnormal(0);
            mat_v_sigma_o_n(A * nsd + 1, B * numstressdof_ + 1) -=
                fac_ * pfunct(A) * pfunct(B) * unitnormal(1);

            mat_v_sigma_o_n(A * nsd, B * numstressdof_ + 2) -=
                fac_ * pfunct(A) * pfunct(B) * unitnormal(1);
            mat_v_sigma_o_n(A * nsd + 1, B * numstressdof_ + 2) -=
                fac_ * pfunct(A) * pfunct(B) * unitnormal(0);
          }
        }
      }
      else if (nsd == 3)
      {
        Core::LinAlg::Matrix<nsd, 1> temp(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<nsd, 1> tempA(Core::LinAlg::Initialization::zero);

        for (int dim = 0; dim < nsd; ++dim)
        {
          temp(dim) = fac_ * unitnormal(dim);
        }

        for (int A = 0; A < piel; ++A)
        {
          for (int dim = 0; dim < nsd; ++dim)
          {
            tempA(dim) = temp(dim) * pfunct(A);
          }
          for (int B = 0; B < piel; ++B)
          {
            mat_v_sigma_o_n(A * nsd, B * numstressdof_) -= tempA(0) * pfunct(B);
            mat_v_sigma_o_n(A * nsd, B * numstressdof_ + 3) -= tempA(1) * pfunct(B);
            mat_v_sigma_o_n(A * nsd, B * numstressdof_ + 4) -= tempA(2) * pfunct(B);

            mat_v_sigma_o_n(A * nsd + 1, B * numstressdof_ + 3) -= tempA(0) * pfunct(B);
            mat_v_sigma_o_n(A * nsd + 1, B * numstressdof_ + 1) -= tempA(1) * pfunct(B);
            mat_v_sigma_o_n(A * nsd + 1, B * numstressdof_ + 5) -= tempA(2) * pfunct(B);

            mat_v_sigma_o_n(A * nsd + 2, B * numstressdof_ + 4) -= tempA(0) * pfunct(B);
            mat_v_sigma_o_n(A * nsd + 2, B * numstressdof_ + 5) -= tempA(1) * pfunct(B);
            mat_v_sigma_o_n(A * nsd + 2, B * numstressdof_ + 2) -= tempA(2) * pfunct(B);
          }
        }
      }

      //--------------------------------------------------
      // adjoint consistency term, stress part

      /*
                     /          \
                    |  h       h |
                  - | r o n , u  |
                    |            |
                     \          / Gamma
      */
      double tau_tangential = 1.0;

      if (nsd == 3)
      {
        // Spaldings law to compute u_tau
        {
          //
          const double y = h / hB_divided_by;


          // get velocity norm
          double normu = velint.norm2();

          // compute friction velocity u_tau
          double utau = visc_ / y;

          double res = spalding_residual_utau(y, visc_, utau, normu);

          int count = 0;

          while ((res * res) > 1.0e-12)
          {
            const double SpaldingJ = jacobian_spalding_residual_utau(y, visc_, utau, normu);

            if (SpaldingJ < 1e-10)
            {
              FOUR_C_THROW("(Nearly) singular Jacobian of Spaldings equation");
            }

            double inc = res / SpaldingJ;

            // do Newton step
            utau -= inc;
            if (abs(utau) < 1.0E-14)
            {
              // If |u_tau| approaches zero, subtract only 99,99% of inc from utau.
              // This heuristics prevents a zero utau, which is problematic
              // within spalding_residual(), where a division takes place.
              // Seems to be required only for the first time step of a simulation.
              // gjb 12/12
              utau += 0.001 * inc;
            }

            // get residual of Spaldings equation (law of the wall)
            res = spalding_residual_utau(y, visc_, utau, normu);

            if (count > 1000)
            {
              printf("WARNING: no convergence in 1000 steps in Newton iteration\n");
              printf("         in solution of Spaldings equation (res %12.5e), utau= %12.5e)\n",
                  res, utau);
              FOUR_C_THROW("Newton iteration diverged.");
            }

            ++count;
          }

          if (spalding)
          {
            const double dres_duplus = jacobian_spalding_residual_uplus(y, visc_, utau, normu);

            if (abs(dres_duplus) < 1e-12) FOUR_C_THROW("prevent division by zero");
            const double visc_dudy = -utau * utau / dres_duplus;

            if (fabs(normtraction) > 0.001 * visc_ / y)
            {
              // based on viscous stresses
              if (utau_computation == "viscous_tangent")
              {
                tau_tangential *= visc_dudy / normtraction;
              }
              else if (utau_computation == "at_wall")
              {
                // a la Michler
                tau_tangential *= utau * utau / normtraction;
              }
            }
          }  // if(spalding)
        }
      }


      const double eleRey = u_C * h / visc_;

      double tau_normal = 1.0 + 2.0 * eleRey;

      const double C1 = tau_tangential;

      const double C2 = tau_normal - C1;

      if (parent->id() == myid)
      {
        printf("u_C  %12.5e ", u_C);
        printf("Re_e %12.5e ", eleRey);
        printf("C2   %12.5e ", C2);
        printf("tau_normal  %12.5e \n", tau_normal);
      }
      /*
                     /                   \
                    |  h       /  h    \  |
             - C1 * | r o n , |  u - g  | |
                    |          \       /  |
                     \                   / Gamma
      */

      if (nsd == 2)
      {
        for (int A = 0; A < piel; ++A)
        {
          for (int B = 0; B < piel; ++B)
          {
            mat_r_o_n_u(A * numstressdof_, B * nsd) -=
                fac_ * C1 * pfunct(A) * pfunct(B) * unitnormal(0);
            mat_r_o_n_u(A * numstressdof_ + 1, B * nsd + 1) -=
                fac_ * C1 * pfunct(A) * pfunct(B) * unitnormal(1);

            mat_r_o_n_u(A * numstressdof_ + 2, B * nsd) -=
                fac_ * C1 * pfunct(A) * pfunct(B) * unitnormal(1);
            mat_r_o_n_u(A * numstressdof_ + 2, B * nsd + 1) -=
                fac_ * C1 * pfunct(A) * pfunct(B) * unitnormal(0);
          }
        }


        for (int A = 0; A < piel; ++A)
        {
          vec_r_o_n_u_minus_g(A * numstressdof_) -=
              fac_ * C1 * pfunct(A) * unitnormal(0) * delta_vel(0);
          vec_r_o_n_u_minus_g(A * numstressdof_ + 1) -=
              fac_ * C1 * pfunct(A) * unitnormal(1) * delta_vel(1);

          vec_r_o_n_u_minus_g(A * numstressdof_ + 2) -=
              fac_ * C1 * pfunct(A) * (unitnormal(1) * delta_vel(0) + unitnormal(0) * delta_vel(1));
        }
      }
      else if (nsd == 3)
      {
        Core::LinAlg::Matrix<nsd, 1> temp;
        Core::LinAlg::Matrix<nsd, 1> tempA;

        for (int dim = 0; dim < nsd; ++dim)
        {
          temp(dim) = fac_ * C1 * unitnormal(dim);
        }

        for (int A = 0; A < piel; ++A)
        {
          for (int dim = 0; dim < nsd; ++dim)
          {
            tempA(dim) = temp(dim) * pfunct(A);
          }

          for (int B = 0; B < piel; ++B)
          {
            mat_r_o_n_u(A * numstressdof_, B * nsd) -= tempA(0) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 1, B * nsd + 1) -= tempA(1) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 2, B * nsd + 2) -= tempA(2) * pfunct(B);

            mat_r_o_n_u(A * numstressdof_ + 3, B * nsd) -= tempA(1) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 3, B * nsd + 1) -= tempA(0) * pfunct(B);

            mat_r_o_n_u(A * numstressdof_ + 4, B * nsd) -= tempA(2) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 4, B * nsd + 2) -= tempA(0) * pfunct(B);

            mat_r_o_n_u(A * numstressdof_ + 5, B * nsd + 1) -= tempA(2) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 5, B * nsd + 2) -= tempA(1) * pfunct(B);
          }
        }

        for (int A = 0; A < piel; ++A)
        {
          vec_r_o_n_u_minus_g(A * numstressdof_) -=
              fac_ * C1 * pfunct(A) * unitnormal(0) * delta_vel(0);
          vec_r_o_n_u_minus_g(A * numstressdof_ + 1) -=
              fac_ * C1 * pfunct(A) * unitnormal(1) * delta_vel(1);
          vec_r_o_n_u_minus_g(A * numstressdof_ + 2) -=
              fac_ * C1 * pfunct(A) * unitnormal(2) * delta_vel(2);

          vec_r_o_n_u_minus_g(A * numstressdof_ + 3) -=
              fac_ * C1 * pfunct(A) * (unitnormal(1) * delta_vel(0) + unitnormal(0) * delta_vel(1));
          vec_r_o_n_u_minus_g(A * numstressdof_ + 4) -=
              fac_ * C1 * pfunct(A) * (unitnormal(2) * delta_vel(0) + unitnormal(0) * delta_vel(2));
          vec_r_o_n_u_minus_g(A * numstressdof_ + 5) -=
              fac_ * C1 * pfunct(A) * (unitnormal(2) * delta_vel(1) + unitnormal(1) * delta_vel(2));
        }
      }

      /*
                     /              /             \  \
                    |  h           |  /  h  \      |  |
             - C2 * | r o n ,  n * | | u - g | * n |  |
                    |              |  \     /      |  |
                     \              \             /  / Gamma
      */
      if (nsd == 2)
      {
        for (int A = 0; A < piel; ++A)
        {
          for (int B = 0; B < piel; ++B)
          {
            mat_r_o_n_u(A * numstressdof_, B * nsd) -=
                fac_ * C2 * pfunct(A) * unitnormal(0) * unitnormal(0) * pfunct(B) * unitnormal(0);
            mat_r_o_n_u(A * numstressdof_, B * nsd + 1) -=
                fac_ * C2 * pfunct(A) * unitnormal(0) * unitnormal(0) * pfunct(B) * unitnormal(1);

            mat_r_o_n_u(A * numstressdof_ + 1, B * nsd) -=
                fac_ * C2 * pfunct(A) * unitnormal(1) * unitnormal(1) * pfunct(B) * unitnormal(0);
            mat_r_o_n_u(A * numstressdof_ + 1, B * nsd + 1) -=
                fac_ * C2 * pfunct(A) * unitnormal(1) * unitnormal(1) * pfunct(B) * unitnormal(1);

            mat_r_o_n_u(A * numstressdof_ + 2, B * nsd) -=
                fac_ * C2 * pfunct(A) * unitnormal(0) * unitnormal(1) * pfunct(B) * unitnormal(0);
            mat_r_o_n_u(A * numstressdof_ + 2, B * nsd + 1) -=
                fac_ * C2 * pfunct(A) * unitnormal(0) * unitnormal(1) * pfunct(B) * unitnormal(1);
          }
        }

        double u_o_n = unitnormal(0) * delta_vel(0) + unitnormal(1) * delta_vel(1);

        for (int A = 0; A < piel; ++A)
        {
          vec_r_o_n_u_minus_g(A * numstressdof_) -=
              fac_ * C2 * pfunct(A) * unitnormal(0) * unitnormal(0) * u_o_n;
          vec_r_o_n_u_minus_g(A * numstressdof_ + 1) -=
              fac_ * C2 * pfunct(A) * unitnormal(1) * unitnormal(1) * u_o_n;

          vec_r_o_n_u_minus_g(A * numstressdof_ + 2) -=
              fac_ * C2 * pfunct(A) * 2 * unitnormal(0) * unitnormal(1) * u_o_n;
        }
      }
      else if (nsd == 3)
      {
        Core::LinAlg::Matrix<numstressdof_, nsd> temp;
        Core::LinAlg::Matrix<numstressdof_, nsd> tempA;

        temp(0, 0) = fac_ * C2 * unitnormal(0) * unitnormal(0) * unitnormal(0);
        temp(0, 1) = fac_ * C2 * unitnormal(0) * unitnormal(0) * unitnormal(1);
        temp(0, 2) = fac_ * C2 * unitnormal(0) * unitnormal(0) * unitnormal(2);

        temp(1, 0) = fac_ * C2 * unitnormal(1) * unitnormal(1) * unitnormal(0);
        temp(1, 1) = fac_ * C2 * unitnormal(1) * unitnormal(1) * unitnormal(1);
        temp(1, 2) = fac_ * C2 * unitnormal(1) * unitnormal(1) * unitnormal(2);

        temp(2, 0) = fac_ * C2 * unitnormal(2) * unitnormal(2) * unitnormal(0);
        temp(2, 1) = fac_ * C2 * unitnormal(2) * unitnormal(2) * unitnormal(1);
        temp(2, 2) = fac_ * C2 * unitnormal(2) * unitnormal(2) * unitnormal(2);

        temp(3, 0) = fac_ * C2 * 2 * unitnormal(0) * unitnormal(1) * unitnormal(0);
        temp(3, 1) = fac_ * C2 * 2 * unitnormal(0) * unitnormal(1) * unitnormal(1);
        temp(3, 2) = fac_ * C2 * 2 * unitnormal(0) * unitnormal(1) * unitnormal(2);

        temp(4, 0) = fac_ * C2 * 2 * unitnormal(0) * unitnormal(2) * unitnormal(0);
        temp(4, 1) = fac_ * C2 * 2 * unitnormal(0) * unitnormal(2) * unitnormal(1);
        temp(4, 2) = fac_ * C2 * 2 * unitnormal(0) * unitnormal(2) * unitnormal(2);

        temp(5, 0) = fac_ * C2 * 2 * unitnormal(1) * unitnormal(2) * unitnormal(0);
        temp(5, 1) = fac_ * C2 * 2 * unitnormal(1) * unitnormal(2) * unitnormal(1);
        temp(5, 2) = fac_ * C2 * 2 * unitnormal(1) * unitnormal(2) * unitnormal(2);


        for (int A = 0; A < piel; ++A)
        {
          for (int sdof = 0; sdof < numstressdof_; ++sdof)
          {
            for (int dim = 0; dim < nsd; ++dim)
            {
              tempA(sdof, dim) = temp(sdof, dim) * pfunct(A);
            }
          }

          for (int B = 0; B < piel; ++B)
          {
            mat_r_o_n_u(A * numstressdof_, B * nsd) -= tempA(0, 0) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_, B * nsd + 1) -= tempA(0, 1) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_, B * nsd + 2) -= tempA(0, 2) * pfunct(B);

            mat_r_o_n_u(A * numstressdof_ + 1, B * nsd) -= tempA(1, 0) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 1, B * nsd + 1) -= tempA(1, 1) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 1, B * nsd + 2) -= tempA(1, 2) * pfunct(B);

            mat_r_o_n_u(A * numstressdof_ + 2, B * nsd) -= tempA(2, 0) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 2, B * nsd + 1) -= tempA(2, 1) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 2, B * nsd + 2) -= tempA(2, 2) * pfunct(B);

            mat_r_o_n_u(A * numstressdof_ + 3, B * nsd) -= tempA(3, 0) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 3, B * nsd + 1) -= tempA(3, 1) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 3, B * nsd + 2) -= tempA(3, 2) * pfunct(B);

            mat_r_o_n_u(A * numstressdof_ + 4, B * nsd) -= tempA(4, 0) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 4, B * nsd + 1) -= tempA(4, 1) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 4, B * nsd + 2) -= tempA(4, 2) * pfunct(B);

            mat_r_o_n_u(A * numstressdof_ + 5, B * nsd) -= tempA(5, 0) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 5, B * nsd + 1) -= tempA(5, 1) * pfunct(B);
            mat_r_o_n_u(A * numstressdof_ + 5, B * nsd + 2) -= tempA(5, 2) * pfunct(B);
          }
        }

        double u_o_n = unitnormal(0) * delta_vel(0) + unitnormal(1) * delta_vel(1) +
                       unitnormal(2) * delta_vel(2);

        for (int A = 0; A < piel; ++A)
        {
          vec_r_o_n_u_minus_g(A * numstressdof_) -=
              fac_ * C2 * pfunct(A) * unitnormal(0) * unitnormal(0) * u_o_n;
          vec_r_o_n_u_minus_g(A * numstressdof_ + 1) -=
              fac_ * C2 * pfunct(A) * unitnormal(1) * unitnormal(1) * u_o_n;
          vec_r_o_n_u_minus_g(A * numstressdof_ + 2) -=
              fac_ * C2 * pfunct(A) * unitnormal(2) * unitnormal(2) * u_o_n;

          vec_r_o_n_u_minus_g(A * numstressdof_ + 3) -=
              fac_ * C2 * pfunct(A) * 2 * unitnormal(0) * unitnormal(1) * u_o_n;
          vec_r_o_n_u_minus_g(A * numstressdof_ + 4) -=
              fac_ * C2 * pfunct(A) * 2 * unitnormal(0) * unitnormal(2) * u_o_n;
          vec_r_o_n_u_minus_g(A * numstressdof_ + 5) -=
              fac_ * C2 * pfunct(A) * 2 * unitnormal(1) * unitnormal(2) * u_o_n;
        }
      }
    }
  }
  /*<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
         PART 4: Local condensation
    <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>*/

  // --------------------------------
  // organise the timefac business

  // One-step-Theta:            timefacrhs = theta*dt
  // BDF2:                      timefacrhs = 2/3 * dt
  // af-generalized-alpha:      timefacrhs = (1.0/alpha_M) * gamma * dt
  // Peters-generalized-alpha:  timefacrhs = 1.0
  double timefacrhs = fldparatimint_->time_fac_rhs();
  // One-step-Theta:            timefacmat_u = theta*dt
  // BDF2:                      timefacmat_u = 2/3 * dt
  // af-generalized-alpha:      timefacmat_u = (alphaF/alpha_M) * gamma * dt
  // Peters-generalized-alpha:  timefacmat_u = alphaF* gamma * dt
  double timefacmat_u = fldparatimint_->time_fac();
  // One-step-Theta:            timefacmat_p = theta*dt
  // BDF2:                      timefacmat_p = 2/3 * dt
  // af-generalized-alpha:      timefacmat_p = (alphaF/alpha_M) * gamma * dt
  // Peters-generalized-alpha:  timefacmat_p = gamma * dt
  // double timefacmat_p= fldpara_->timefacmat_p_;
  double timefacmat_p = fldparatimint_->time_fac_pre();

  // --------------------------------
  // rearrange to pattern uvwp uvwp ...
  for (int A = 0; A < piel; ++A)
  {
    for (int B = 0; B < piel; ++B)
    {
      for (int i = 0; i < numstressdof_; ++i)
      {
        for (int j = 0; j < nsd; ++j)
        {
          mat_r_up_block(A * numstressdof_ + i, B * (nsd + 1) + j) +=
              timefacmat_u * mat_r_epsu(A * numstressdof_ + i, B * nsd + j);
          mat_r_up_block(A * numstressdof_ + i, B * (nsd + 1) + j) +=
              timefacmat_u * mat_r_o_n_u(A * numstressdof_ + i, B * nsd + j);
        }
        mat_r_up_block(A * numstressdof_ + i, B * (nsd + 1) + nsd) +=
            timefacmat_p * mat_r_p(A * numstressdof_ + i, B);
      }
    }
  }

  // computation of matrix-matrix and matrix vector products, local assembly
  for (int A = 0; A < piel; ++A)
  {
    for (int i = 0; i < nsd; ++i)
    {
      for (int B = 0; B < piel; ++B)
      {
        for (int rr = 0; rr < numstressdof_ * piel; ++rr)
        {
          for (int mm = 0; mm < numstressdof_ * piel; ++mm)
          {
            for (int j = 0; j < nsd + 1; ++j)
            {
              elemat1(A * (nsd + 1) + i, B * (nsd + 1) + j) -=
                  mat_v_sigma_o_n(A * nsd + i, rr) * inv_r_sigma(rr, mm) *
                  mat_r_up_block(mm, B * (nsd + 1) + j);
            }
          }
        }
      }
    }
  }

  for (int A = 0; A < piel; ++A)
  {
    for (int i = 0; i < nsd; ++i)
    {
      for (int rr = 0; rr < numstressdof_ * piel; ++rr)
      {
        for (int mm = 0; mm < numstressdof_ * piel; ++mm)
        {
          elevec1(A * (nsd + 1) + i) -= timefacrhs * mat_v_sigma_o_n(A * nsd + i, rr) *
                                        inv_r_sigma(rr, mm) *
                                        (-vec_r_o_n_u_minus_g(mm) - vec_r_epsu(mm) - vec_r_p(mm));
        }
      }
    }
  }

  return;
}  // Discret::Elements::FluidBoundaryParent<distype>::MixHybDirichlet


/*----------------------------------------------------------------------*
 |  get density and viscosity                                  vg 07/13 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::FluidBoundaryParent<distype>::get_density_and_viscosity(
    std::shared_ptr<const Core::Mat::Material> material, const double pscaaf,
    const double thermpressaf, const double rateofstrain)
{
  // initially set density to 1.0
  densaf_ = 1.0;

  if (material->material_type() == Core::Materials::m_fluid)
  {
    const Mat::NewtonianFluid* actmat = static_cast<const Mat::NewtonianFluid*>(material.get());

    densaf_ = actmat->density();

    // get constant viscosity
    visc_ = actmat->viscosity();
  }
  else if (material->material_type() == Core::Materials::m_carreauyasuda)
  {
    const Mat::CarreauYasuda* actmat = static_cast<const Mat::CarreauYasuda*>(material.get());

    densaf_ = actmat->density();

    double nu_0 = actmat->nu0();       // parameter for zero-shear viscosity
    double nu_inf = actmat->nu_inf();  // parameter for infinite-shear viscosity
    double lambda = actmat->lambda();  // parameter for characteristic time
    double a = actmat->a_param();      // constant parameter
    double b = actmat->b_param();      // constant parameter

    // compute viscosity according to the Carreau-Yasuda model for shear-thinning fluids
    // see Dhruv Arora, Computational Hemodynamics: Hemolysis and Viscoelasticity,PhD, 2005
    const double tmp = std::pow(lambda * rateofstrain, b);
    // kinematic viscosity
    visc_ = nu_inf + ((nu_0 - nu_inf) / pow((1 + tmp), a));
    // dynamic viscosity
    visc_ *= densaf_;
  }
  else if (material->material_type() == Core::Materials::m_modpowerlaw)
  {
    const Mat::ModPowerLaw* actmat = static_cast<const Mat::ModPowerLaw*>(material.get());

    densaf_ = actmat->density();

    // get material parameters
    double m = actmat->m_cons();     // consistency constant
    double delta = actmat->delta();  // safety factor
    double a = actmat->a_exp();      // exponent

    // compute viscosity according to a modified power law model for shear-thinning fluids
    // see Dhruv Arora, Computational Hemodynamics: Hemolysis and Viscoelasticity,PhD, 2005
    // kinematic viscosity
    visc_ = m * pow((delta + rateofstrain), (-1) * a);
    // dynamic viscosity
    visc_ *= densaf_;
  }
  else if (material->material_type() == Core::Materials::m_herschelbulkley)
  {
    const Mat::HerschelBulkley* actmat = static_cast<const Mat::HerschelBulkley*>(material.get());

    densaf_ = actmat->density();

    double tau0 = actmat->tau0();                         // yield stress
    double kfac = actmat->k_fac();                        // constant factor
    double nexp = actmat->n_exp();                        // exponent
    double mexp = actmat->m_exp();                        // exponent
    double uplimshearrate = actmat->up_lim_shear_rate();  // upper limit of shear rate
    double lolimshearrate = actmat->lo_lim_shear_rate();  // lower limit of shear rate

    // calculate dynamic viscosity according to Herschel-Bulkley model
    // (within lower and upper limit of shear rate)
    if (rateofstrain < lolimshearrate)
      visc_ = tau0 * ((1.0 - exp(-mexp * lolimshearrate)) / lolimshearrate) +
              kfac * pow(lolimshearrate, (nexp - 1.0));
    else if (rateofstrain > uplimshearrate)
      visc_ = tau0 * ((1.0 - exp(-mexp * uplimshearrate)) / uplimshearrate) +
              kfac * pow(uplimshearrate, (nexp - 1.0));
    else
      visc_ = tau0 * ((1.0 - exp(-mexp * rateofstrain)) / rateofstrain) +
              kfac * pow(rateofstrain, (nexp - 1.0));
  }
  else if (material->material_type() == Core::Materials::m_sutherland)
  {
    const Mat::Sutherland* actmat = static_cast<const Mat::Sutherland*>(material.get());

    // compute viscosity according to Sutherland law
    visc_ = actmat->compute_viscosity(pscaaf);

    // compute density at n+alpha_F or n+1 based on temperature
    // and thermodynamic pressure
    densaf_ = actmat->compute_density(pscaaf, thermpressaf);
  }
  else if (material->material_type() == Core::Materials::m_fluidporo)
  {
    const Mat::FluidPoro* actmat = static_cast<const Mat::FluidPoro*>(material.get());

    densaf_ = actmat->density();

    // get constant viscosity
    visc_ = actmat->viscosity();
  }
  else
    FOUR_C_THROW(
        "Material type is not supported for boundary element with parent-element evaluation!");

  // check whether there is zero or negative density
  if (densaf_ < 1e-15) FOUR_C_THROW("zero or negative density!");

  // check whether there is zero or negative (physical) viscosity
  if (visc_ < 1e-15) FOUR_C_THROW("zero or negative (physical) diffusivity!");

  return;
}  // FluidBoundaryParent::get_density_and_viscosity

FOUR_C_NAMESPACE_CLOSE
