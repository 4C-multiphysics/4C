// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_thermo_ele_impl.hpp"

#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_plasticelasthyper.hpp"
#include "4C_mat_thermoplastichyperelast.hpp"
#include "4C_mat_thermoplasticlinelast.hpp"
#include "4C_mat_thermostvenantkirchhoff.hpp"
#include "4C_mat_trait_thermo_solid.hpp"
#include "4C_thermo_ele_action.hpp"
#include "4C_thermo_element.hpp"  // only for visualization of element data
#include "4C_thermo_input.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <algorithm>

FOUR_C_NAMESPACE_OPEN

Discret::Elements::TemperImplInterface* Discret::Elements::TemperImplInterface::impl(
    const Core::Elements::Element* ele)
{
  switch (ele->shape())
  {
    case Core::FE::CellType::hex8:
    {
      return TemperImpl<Core::FE::CellType::hex8>::instance();
    }
    case Core::FE::CellType::hex20:
    {
      return TemperImpl<Core::FE::CellType::hex20>::instance();
    }
    case Core::FE::CellType::hex27:
    {
      return TemperImpl<Core::FE::CellType::hex27>::instance();
    }
    case Core::FE::CellType::tet4:
    {
      return TemperImpl<Core::FE::CellType::tet4>::instance();
    }
    case Core::FE::CellType::tet10:
    {
      return TemperImpl<Core::FE::CellType::tet10>::instance();
    }
    case Core::FE::CellType::wedge6:
    {
      return TemperImpl<Core::FE::CellType::wedge6>::instance();
    }
    case Core::FE::CellType::pyramid5:
    {
      return TemperImpl<Core::FE::CellType::pyramid5>::instance();
    }
    case Core::FE::CellType::quad4:
    {
      return TemperImpl<Core::FE::CellType::quad4>::instance();
    }
    case Core::FE::CellType::quad8:
    {
      return TemperImpl<Core::FE::CellType::quad8>::instance();
    }
    case Core::FE::CellType::quad9:
    {
      return TemperImpl<Core::FE::CellType::quad9>::instance();
    }
    case Core::FE::CellType::tri3:
    {
      return TemperImpl<Core::FE::CellType::tri3>::instance();
    }
    case Core::FE::CellType::line2:
    {
      return TemperImpl<Core::FE::CellType::line2>::instance();
    }
    case Core::FE::CellType::nurbs27:
    {
      return TemperImpl<Core::FE::CellType::nurbs27>::instance();
    }
    default:
      FOUR_C_THROW("Element shape {} ({} nodes) not activated. Just do it.",
          Core::FE::cell_type_to_string(ele->shape()), ele->num_node());
      break;
  }
  return nullptr;

}  // TemperImperInterface::Impl()

template <Core::FE::CellType distype>
Discret::Elements::TemperImpl<distype>* Discret::Elements::TemperImpl<distype>::instance(
    Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<Discret::Elements::TemperImpl<distype>>(
            new Discret::Elements::TemperImpl<distype>());
      });

  return singleton_owner.instance(action);
}

template <Core::FE::CellType distype>
Discret::Elements::TemperImpl<distype>::TemperImpl()
    : etempn_(Core::LinAlg::Initialization::uninitialized),
      xyze_(Core::LinAlg::Initialization::zero),
      radiation_(Core::LinAlg::Initialization::uninitialized),
      xsi_(Core::LinAlg::Initialization::zero),
      funct_(Core::LinAlg::Initialization::zero),
      deriv_(Core::LinAlg::Initialization::zero),
      xjm_(Core::LinAlg::Initialization::zero),
      xij_(Core::LinAlg::Initialization::zero),
      derxy_(Core::LinAlg::Initialization::zero),
      fac_(0.0),
      gradtemp_(Core::LinAlg::Initialization::zero),
      heatflux_(Core::LinAlg::Initialization::uninitialized),
      cmat_(Core::LinAlg::Initialization::uninitialized),
      dercmat_(Core::LinAlg::Initialization::zero),
      capacoeff_(0.0),
      dercapa_(0.0),
      plasticmat_(false)

{
}

template <Core::FE::CellType distype>
int Discret::Elements::TemperImpl<distype>::evaluate(
    const Core::Elements::Element* ele, Teuchos::ParameterList& params,
    const Core::FE::Discretization& discretization, const Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1,  // Tangent ("stiffness")
    Core::LinAlg::SerialDenseMatrix& elemat2,  // Capacity ("mass")
    Core::LinAlg::SerialDenseVector& elevec1,  // internal force vector
    Core::LinAlg::SerialDenseVector& elevec2,  // external force vector
    Core::LinAlg::SerialDenseVector& elevec3   // capacity vector
)
{
  prepare_nurbs_eval(ele, discretization);

  const auto action = Teuchos::getIntegralValue<Thermo::Action>(params, "action");

  // check length
  if (la[0].size() != nen_ * numdofpernode_) FOUR_C_THROW("Location vector length does not match!");

  // disassemble temperature
  if (discretization.has_state(0, "temperature"))
  {
    std::vector<double> mytempnp((la[0].lm_).size());
    std::shared_ptr<const Core::LinAlg::Vector<double>> tempnp =
        discretization.get_state(0, "temperature");
    if (tempnp == nullptr) FOUR_C_THROW("Cannot get state vector 'tempnp'");
    mytempnp = Core::FE::extract_values(*tempnp, la[0].lm_);
    // build the element temperature
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> etempn(mytempnp.data(), true);  // view only!
    etempn_.update(etempn);                                                        // copy
  }

  if (discretization.has_state(0, "last temperature"))
  {
    std::vector<double> mytempn((la[0].lm_).size());
    std::shared_ptr<const Core::LinAlg::Vector<double>> tempn =
        discretization.get_state(0, "last temperature");
    if (tempn == nullptr) FOUR_C_THROW("Cannot get state vector 'tempn'");
    mytempn = Core::FE::extract_values(*tempn, la[0].lm_);
    // build the element temperature
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> etemp(mytempn.data(), true);  // view only!
    etemp_.update(etemp);                                                        // copy
  }

  double time = 0.0;

  if (action != Thermo::calc_thermo_energy)
  {
    // extract time
    time = params.get<double>("total time");
  }

  // ---------------------------------------------------------------- TSI

  // if it's a TSI problem with displacementcoupling_ --> go on here!
  // todo: fix for volmortar (not working with plasticity)
  if (la.size() > 1)
  {
    // ------------------------------------------------ structural material
    std::shared_ptr<Core::Mat::Material> structmat = get_str_material(ele);

    // call ThermoStVenantKirchhoff material and get the temperature dependent
    // tangent ctemp
    plasticmat_ = false;
    if ((structmat->material_type() == Core::Materials::m_thermopllinelast) or
        (structmat->material_type() == Core::Materials::m_thermoplhyperelast))
      plasticmat_ = true;
  }  // (la.Size > 1)

  //============================================================================
  // calculate tangent K and internal force F_int = K * Theta
  // --> for static case
  if (action == Thermo::calc_thermo_fintcond)
  {
    // set views
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_> etang(
        elemat1.values(), true);                                                   // view only!
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> efint(elevec1.values(), true);  // view only!
    // ecapa, efext, efcap not needed for this action
    // econd: conductivity matrix
    // etang: tangent of thermal problem.
    // --> If dynamic analysis, i.e. T' != 0 --> etang consists of econd AND ecapa

    evaluate_tang_capa_fint(
        ele, time, discretization, la, &etang, nullptr, nullptr, &efint, params);
  }
  //============================================================================
  // calculate only the internal force F_int, needed for restart
  else if (action == Thermo::calc_thermo_fint)
  {
    // set views
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> efint(elevec1.values(), true);  // view only!
    // etang, ecapa, efext, efcap not needed for this action

    evaluate_tang_capa_fint(
        ele, time, discretization, la, nullptr, nullptr, nullptr, &efint, params);
  }

  //============================================================================
  // calculate the capacity matrix and the internal force F_int
  // --> for dynamic case, called only once in determine_capa_consist_temp_rate()
  else if (action == Thermo::calc_thermo_fintcapa)
  {
    // set views
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_> ecapa(
        elemat2.values(), true);                                                   // view only!
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> efint(elevec1.values(), true);  // view only!
    // etang, efext, efcap not needed for this action

    evaluate_tang_capa_fint(
        ele, time, discretization, la, nullptr, &ecapa, nullptr, &efint, params);

    // lumping
    if (params.get<bool>("lump capa matrix", false))
    {
      const auto timint =
          params.get<Thermo::DynamicType>("time integrator", Thermo::DynamicType::Undefined);
      switch (timint)
      {
        case Thermo::DynamicType::OneStepTheta:
        {
          calculate_lump_matrix(&ecapa);

          break;
        }
        case Thermo::DynamicType::GenAlpha:
        case Thermo::DynamicType::Statics:
        {
          FOUR_C_THROW("Lumped capacity matrix has not yet been tested");
          break;
        }
        case Thermo::DynamicType::Undefined:
        default:
        {
          FOUR_C_THROW("Undefined time integration scheme for thermal problem!");
          break;
        }
      }
    }
  }

  //============================================================================
  // called from overloaded function apply_force_tang_internal(), exclusively for
  // dynamic-timint (as OST, GenAlpha)
  // calculate effective dynamic tangent matrix K_{T, effdyn},
  // i.e. sum consistent capacity matrix C + its linearization and scaled conductivity matrix
  // --> for dynamic case
  else if (action == Thermo::calc_thermo_finttang)
  {
    // set views
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_> etang(
        elemat1.values(), true);  // view only!
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_> ecapa(
        Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> efint(elevec1.values(), true);  // view only!
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> efcap(elevec3.values(), true);  // view only!

    // etang: effective dynamic tangent of thermal problem
    // --> etang == k_{T,effdyn}^{(e)} = timefac_capa ecapa + timefac_cond econd
    // econd: conductivity matrix
    // ecapa: capacity matrix
    // --> If dynamic analysis, i.e. T' != 0 --> etang consists of econd AND ecapa

    // helper matrix to store partial dC/dT*(T_{n+1} - T_n) linearization of capacity
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_> ecapalin(
        Core::LinAlg::Initialization::zero);

    evaluate_tang_capa_fint(
        ele, time, discretization, la, &etang, &ecapa, &ecapalin, &efint, params);


    if (params.get<bool>("lump capa matrix", false))
    {
      calculate_lump_matrix(&ecapa);
    }

    // explicitly insert capacity matrix into corresponding matrix if existing
    if (elemat2.values() != nullptr)
    {
      Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_> ecapa_export(
          elemat2.values(), true);  // view only!
      ecapa_export.update(ecapa);
    }

    // BUILD EFFECTIVE TANGENT AND RESIDUAL ACCORDING TO TIME INTEGRATOR
    // combine capacity and conductivity matrix to one global tangent matrix
    // check the time integrator
    // K_T = fac_capa . C + fac_cond . K
    const auto timint =
        params.get<Thermo::DynamicType>("time integrator", Thermo::DynamicType::Undefined);
    switch (timint)
    {
      case Thermo::DynamicType::Statics:
      {
        // continue
        break;
      }
      case Thermo::DynamicType::OneStepTheta:
      {
        // extract time values from parameter list
        const double theta = params.get<double>("theta");
        const double stepsize = params.get<double>("delta time");

        // ---------------------------------------------------------- etang
        // combine capacity and conductivity matrix to one global tangent matrix
        // etang = 1/Dt . ecapa + theta . econd
        // fac_capa = 1/Dt
        // fac_cond = theta
        etang.update(1.0 / stepsize, ecapa, theta);
        // add additional linearization term from variable capacity
        // + 1/Dt. ecapalin
        etang.update(1.0 / stepsize, ecapalin, 1.0);

        // ---------------------------------------------------------- efcap
        // fcapn = ecapa(T_{n+1}) .  (T_{n+1} -T_n) /Dt
        efcap.multiply(ecapa, etempn_);
        efcap.multiply(-1.0, ecapa, etemp_, 1.0);
        efcap.scale(1.0 / stepsize);
        break;
      }  // ost

      case Thermo::DynamicType::GenAlpha:
      {
        // extract time values from parameter list
        const double alphaf = params.get<double>("alphaf");
        const double alpham = params.get<double>("alpham");
        const double gamma = params.get<double>("gamma");
        const double stepsize = params.get<double>("delta time");

        // ---------------------------------------------------------- etang
        // combined tangent and conductivity matrix to one global matrix
        // etang = alpham/(gamma . Dt) . ecapa + alphaf . econd
        // fac_capa = alpham/(gamma . Dt)
        // fac_cond = alphaf
        double fac_capa = alpham / (gamma * stepsize);
        etang.update(fac_capa, ecapa, alphaf);

        // ---------------------------------------------------------- efcap
        // efcap = ecapa . R_{n+alpham}
        if (discretization.has_state(0, "mid-temprate"))
        {
          std::shared_ptr<const Core::LinAlg::Vector<double>> ratem =
              discretization.get_state(0, "mid-temprate");
          if (ratem == nullptr) FOUR_C_THROW("Cannot get mid-temprate state vector for fcap");
          std::vector<double> myratem((la[0].lm_).size());
          // fill the vector myratem with the global values of ratem
          myratem = Core::FE::extract_values(*ratem, la[0].lm_);
          // build the element mid-temperature rates
          Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> eratem(
              myratem.data(), true);  // view only!
          efcap.multiply(ecapa, eratem);
        }  // ratem != nullptr
        break;

      }  // genalpha
      case Thermo::DynamicType::Undefined:
      default:
      {
        FOUR_C_THROW("Don't know what to do...");
        break;
      }
    }  // end of switch(timint)
  }  // action == Thermo::calc_thermo_finttang

  //============================================================================
  // Calculate/ evaluate heatflux q and temperature gradients gradtemp at
  // gauss points
  else if (action == Thermo::calc_thermo_heatflux)
  {
    // set views
    // efext, efcap not needed for this action, elemat1+2,elevec1-3 are not used anyway

    // get storage arrays of Gauss-point-wise vectors
    std::shared_ptr<std::vector<char>> heatfluxdata =
        params.get<std::shared_ptr<std::vector<char>>>("heatflux");
    std::shared_ptr<std::vector<char>> tempgraddata =
        params.get<std::shared_ptr<std::vector<char>>>("tempgrad");
    // working arrays
    Core::LinAlg::Matrix<nquad_, nsd_> eheatflux(Core::LinAlg::Initialization::uninitialized);
    Core::LinAlg::Matrix<nquad_, nsd_> etempgrad(Core::LinAlg::Initialization::uninitialized);

    // if ele is a thermo element --> the Thermo element method KinType() exists
    const auto* therm = dynamic_cast<const Thermo::Element*>(ele);
    const Inpar::Solid::KinemType kintype = therm->kin_type();
    // thermal problem or geometrically linear TSI problem
    if (kintype == Inpar::Solid::KinemType::linear)
    {
      linear_heatflux_tempgrad(ele, &eheatflux, &etempgrad);
    }  // TSI: (kintype_ == Inpar::Solid::KinemType::linear)

    // geometrically nonlinear TSI problem
    if (kintype == Inpar::Solid::KinemType::nonlinearTotLag)
    {
      // if it's a TSI problem and there are current displacements/velocities
      if (la.size() > 1)
      {
        if ((discretization.has_state(1, "displacement")) and
            (discretization.has_state(1, "velocity")))
        {
          std::vector<double> mydisp(((la[0].lm_).size()) * nsd_, 0.0);
          std::vector<double> myvel(((la[0].lm_).size()) * nsd_, 0.0);

          extract_disp_vel(discretization, la, mydisp, myvel);

          nonlinear_heatflux_tempgrad(ele, mydisp, myvel, &eheatflux, &etempgrad, params);
        }
      }
    }

    copy_matrix_into_char_vector(*heatfluxdata, eheatflux);
    copy_matrix_into_char_vector(*tempgraddata, etempgrad);
  }  // action == Thermo::calc_thermo_heatflux

  //============================================================================
  // Calculate heatflux q and temperature gradients gradtemp at gauss points
  else if (action == Thermo::postproc_thermo_heatflux)
  {
    // set views
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_> etang(
        elemat1.values(), true);  // view only!
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_> ecapa(
        elemat2.values(), true);                                                   // view only!
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> efint(elevec1.values(), true);  // view only!
    // efext, efcap not needed for this action

    const std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>
        gpheatfluxmap = params.get<
            std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>>(
            "gpheatfluxmap");
    std::string heatfluxtype = params.get<std::string>("heatfluxtype", "ndxyz");
    const int gid = ele->id();
    Core::LinAlg::Matrix<nquad_, nsd_> gpheatflux(
        ((*gpheatfluxmap)[gid])->values(), true);  // view only!

    // set views to components
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> efluxx(elevec1, true);  // view only!
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> efluxy(elevec2, true);  // view only!
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> efluxz(elevec3, true);  // view only!

    // catch unknown heatflux types
    bool processed = false;

    // nodally
    // extrapolate heatflux q and temperature gradient gradtemp stored at GP
    if ((heatfluxtype == "ndxyz") or (heatfluxtype == "cxyz_ndxyz"))
    {
      processed = true;
      // extrapolate heatfluxes/temperature gradients at Gauss points to nodes
      // and store results in
      extrapolate_from_gauss_points_to_nodes(ele, gpheatflux, efluxx, efluxy, efluxz);
      // method only applicable if number GP == number nodes
    }  // end "ndxyz" or "cxyz_ndxyz"

    // centered
    if ((heatfluxtype == "cxyz") or (heatfluxtype == "cxyz_ndxyz"))
    {
      processed = true;

      std::shared_ptr<Core::LinAlg::MultiVector<double>> eleheatflux =
          params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("eleheatflux");
      const Core::LinAlg::Map& elemap = eleheatflux->get_map();
      int lid = elemap.lid(gid);
      if (lid != -1)
      {
        for (int idim = 0; idim < nsd_; ++idim)
        {
          // double& s = ; // resolve pointer for faster access
          double s = 0.0;
          // nquad_: number of Gauss points
          for (int jquad = 0; jquad < nquad_; ++jquad) s += gpheatflux(jquad, idim);
          s /= nquad_;
          (((*eleheatflux)(idim))).get_values()[lid] = s;
        }
      }
    }  // end "cxyz" or "cxyz_ndxyz"

    // catch unknown heatflux types
    if (not processed)
      FOUR_C_THROW("unknown type of heatflux/temperature gradient output on element level");

  }  // action == Thermo::postproc_thermo_heatflux

  //============================================================================
  else if (action == Thermo::integrate_shape_functions)
  {
    // calculate integral of shape functions
    const auto dofids = params.get<std::shared_ptr<Core::LinAlg::IntSerialDenseVector>>("dofids");
    integrate_shape_functions(ele, elevec1, *dofids);
  }

  //============================================================================
  else if (action == Thermo::calc_thermo_update_istep)
  {
    // call material specific update
    std::shared_ptr<Core::Mat::Material> material = ele->material();
    // we have to have a thermo-capable material here -> throw error if not
    std::shared_ptr<Mat::Trait::Thermo> thermoMat =
        std::dynamic_pointer_cast<Mat::Trait::Thermo>(material);

    Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
    if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");
  }

  //==================================================================================
  // allowing the predictor TangTemp in input file --> can be decisive in compressible case!
  else if (action == Thermo::calc_thermo_reset_istep)
  {
    // we have to have a thermo-capable material here -> throw error if not
    std::shared_ptr<Mat::Trait::Thermo> thermoMat =
        std::dynamic_pointer_cast<Mat::Trait::Thermo>(ele->material());
    thermoMat->reset_current_state();
  }

  //============================================================================
  // evaluation of internal thermal energy
  else if (action == Thermo::calc_thermo_energy)
  {
    // check length of elevec1
    if (elevec1.length() < 1) FOUR_C_THROW("The given result vector is too short.");

    // get node coordinates
    Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
        ele, xyze_);

    // declaration of internal variables
    double intenergy = 0.0;

    // ----------------------------- integration loop for one element

    // integrations points and weights
    Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
    if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

    // --------------------------------------- loop over Gauss Points
    for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
    {
      eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

      // call material law => sets capacoeff_
      materialize(ele, iquad);

      Core::LinAlg::Matrix<1, 1> temp(Core::LinAlg::Initialization::uninitialized);
      temp.multiply_tn(funct_, etempn_);

      // internal energy
      intenergy += capacoeff_ * fac_ * temp(0, 0);

    }  // -------------------------------- end loop over Gauss Points

    elevec1(0) = intenergy;

  }  // evaluation of internal energy

  //============================================================================
  // add linearistion of velocity for dynamic time integration to the stiffness term
  // calculate thermal mechanical tangent matrix K_Td
  else if (action == Thermo::calc_thermo_coupltang)
  {
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * nsd_ * numdofpernode_> etangcoupl(
        elemat1.values(), true);

    // if it's a TSI problem and there are the current displacements/velocities
    evaluate_coupled_tang(ele, discretization, la, &etangcoupl, params);

  }  // action == "calc_thermo_coupltang"
  //============================================================================
  else if (action == Thermo::calc_thermo_error)
  {
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> evector(elevec1.values(), true);  // view only!

    compute_error(ele, evector, params);
  }
  //============================================================================
  else
  {
    FOUR_C_THROW("Unknown type of action for Temperature Implementation: {}", action);
  }


  return 0;
}

template <Core::FE::CellType distype>
int Discret::Elements::TemperImpl<distype>::evaluate_neumann(const Core::Elements::Element* ele,
    const Teuchos::ParameterList& params, const Core::FE::Discretization& discretization,
    const std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  // prepare nurbs
  prepare_nurbs_eval(ele, discretization);

  // check length
  if (lm.size() != nen_ * numdofpernode_) FOUR_C_THROW("Location vector length does not match!");
  // set views
  Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> efext(elevec1, true);  // view only!
  // disassemble temperature
  if (discretization.has_state(0, "temperature"))
  {
    std::vector<double> mytempnp(lm.size());
    std::shared_ptr<const Core::LinAlg::Vector<double>> tempnp =
        discretization.get_state("temperature");
    if (tempnp == nullptr) FOUR_C_THROW("Cannot get state vector 'tempnp'");
    mytempnp = Core::FE::extract_values(*tempnp, lm);
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1> etemp(mytempnp.data(), true);  // view only!
    etempn_.update(etemp);                                                        // copy
  }
  // check for the action parameter
  const auto action = Teuchos::getIntegralValue<Thermo::Action>(params, "action");
  // extract time
  const double time = params.get<double>("total time");

  // perform actions
  if (action == Thermo::calc_thermo_fext)
  {
    // so far we assume deformation INdependent external loads, i.e. NO
    // difference between geometrically (non)linear TSI

    // we prescribe a scalar value on the volume, constant for (non)linear analysis
    evaluate_fext(ele, time, efext);
  }
  else
  {
    FOUR_C_THROW("Unknown type of action for Temperature Implementation: {}", action);
  }

  return 0;
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::evaluate_tang_capa_fint(
    const Core::Elements::Element* ele, const double time,
    const Core::FE::Discretization& discretization, const Core::Elements::LocationArray& la,
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>* etang,
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>* ecapa,
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>* ecapalin,
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1>* efint, Teuchos::ParameterList& params)
{
  const auto* therm = dynamic_cast<const Thermo::Element*>(ele);
  const Inpar::Solid::KinemType kintype = therm->kin_type();

  // initialise the vectors
  // evaluate() is called the first time in Thermo::BaseAlgorithm: at this stage
  // the coupling field is not yet known. Pass coupling vectors filled with zeros
  // the size of the vectors is the length of the location vector*nsd_
  std::vector<double> mydisp(((la[0].lm_).size()) * nsd_, 0.0);
  std::vector<double> myvel(((la[0].lm_).size()) * nsd_, 0.0);

  // if it's a TSI problem with displacementcoupling_ --> go on here!
  if (la.size() > 1)
  {
    extract_disp_vel(discretization, la, mydisp, myvel);
  }  // la.Size>1

  // geometrically linear TSI problem
  if ((kintype == Inpar::Solid::KinemType::linear))
  {
    // purely thermal contributions
    linear_thermo_contribution(ele, time, etang,
        ecapa,     // capa matric
        ecapalin,  // capa linearization
        efint);

    if (la.size() > 1)
    {
      // coupled displacement dependent terms
      linear_disp_contribution(ele, time, mydisp, myvel, etang, efint, params);

      // if structural material is plastic --> calculate the mechanical dissipation terms
      // A_k * a_k - (d^2 psi / dT da_k) * a_k'
      if (plasticmat_) linear_dissipation_fint(ele, efint, params);
    }
  }  // TSI: (kintype_ == Inpar::Solid::KinemType::linear)

  // geometrically nonlinear TSI problem
  else if (kintype == Inpar::Solid::KinemType::nonlinearTotLag)
  {
    nonlinear_thermo_disp_contribution(
        ele, time, mydisp, myvel, etang, ecapa, ecapalin, efint, params);

    if (plasticmat_) nonlinear_dissipation_fint_tang(ele, mydisp, etang, efint, params);
  }  // TSI: (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::evaluate_coupled_tang(
    const Core::Elements::Element* ele, const Core::FE::Discretization& discretization,
    const Core::Elements::LocationArray& la,
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * nsd_ * numdofpernode_>* etangcoupl,
    Teuchos::ParameterList& params)
{
  const auto* therm = dynamic_cast<const Thermo::Element*>(ele);
  const Inpar::Solid::KinemType kintype = therm->kin_type();

  if (la.size() > 1)
  {
    std::vector<double> mydisp(((la[0].lm_).size()) * nsd_, 0.0);
    std::vector<double> myvel(((la[0].lm_).size()) * nsd_, 0.0);

    extract_disp_vel(discretization, la, mydisp, myvel);

    // if there is a strucutural vector available go on here
    // --> calculate coupling stiffness term in case of monolithic TSI

    // geometrically linear TSI problem
    if (kintype == Inpar::Solid::KinemType::linear)
    {
      linear_coupled_tang(ele, mydisp, myvel, etangcoupl, params);

      // calculate Dmech_d
      if (plasticmat_) linear_dissipation_coupled_tang(ele, etangcoupl, params);
      // --> be careful: so far only implicit Euler for time integration
      //                 of the evolution equation available!!!
    }  // TSI: (kintype_ == Inpar::Solid::KinemType::linear)

    // geometrically nonlinear TSI problem
    if (kintype == Inpar::Solid::KinemType::nonlinearTotLag)
    {
      nonlinear_coupled_tang(ele, mydisp, myvel, etangcoupl, params);

      // calculate Dmech_d
      if (plasticmat_) nonlinear_dissipation_coupled_tang(ele, mydisp, myvel, etangcoupl, params);
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::evaluate_fext(
    const Core::Elements::Element* ele,                    // the element whose matrix is calculated
    const double time,                                     // current time
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1>& efext  // external force
)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // ------------------------------- integration loop for one element

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // ----------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // ---------------------------------------------------------------------
    // call routine for calculation of radiation in element nodes
    // (time n+alpha_F for generalized-alpha scheme, at time n+1 otherwise)
    // ---------------------------------------------------------------------
    radiation(ele, time);
    // fext = fext + N . r. detJ . w(gp)
    // with funct_: shape functions, fac_:detJ . w(gp)
    efext.multiply_nn(fac_, funct_, radiation_, 1.0);
  }
}


/*----------------------------------------------------------------------*
 | calculate system matrix and rhs r_T(T), k_TT(T) (public) g.bau 08/08 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::linear_thermo_contribution(
    const Core::Elements::Element* ele,  // the element whose matrix is calculated
    const double time,                   // current time
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>*
        econd,  // conductivity matrix
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>* ecapa,  // capacity matrix
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>*
        ecapalin,                                          // linearization contribution of capacity
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1>* efint  // internal force
)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // ------------------------------- integration loop for one element

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // ----------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // gradient of current temperature value
    // grad T = d T_j / d x_i = L . N . T = B_ij T_j
    gradtemp_.multiply_nn(derxy_, etempn_);

    // call material law => cmat_,heatflux_
    // negative q is used for balance equation: -q = -(-k gradtemp)= k * gradtemp
    materialize(ele, iquad);


    // internal force vector
    if (efint != nullptr)
    {
      // fint = fint + B^T . q . detJ . w(gp)
      efint->multiply_tn(fac_, derxy_, heatflux_, 1.0);
    }

    // conductivity matrix
    if (econd != nullptr)
    {
      // ke = ke + ( B^T . C_mat . B ) * detJ * w(gp)  with C_mat = k * I
      Core::LinAlg::Matrix<nsd_, nen_> aop(Core::LinAlg::Initialization::uninitialized);  // (3x8)
      // -q = C * B
      aop.multiply_nn(cmat_, derxy_);              //(nsd_xnsd_)(nsd_xnen_)
      econd->multiply_tn(fac_, derxy_, aop, 1.0);  //(nen_xnen_)=(nen_xnsd_)(nsd_xnen_)

      // linearization of non-constant conductivity
      Core::LinAlg::Matrix<nen_, 1> dNgradT(Core::LinAlg::Initialization::uninitialized);
      dNgradT.multiply_tn(derxy_, gradtemp_);
      // TODO only valid for isotropic case
      econd->multiply_nt(dercmat_(0, 0) * fac_, dNgradT, funct_, 1.0);
    }

    // capacity matrix (equates the mass matrix in the structural field)
    if (ecapa != nullptr)
    {
      // ce = ce + ( N^T .  (rho * C_V) . N ) * detJ * w(gp)
      // (8x8)      (8x1)               (1x8)
      // caution: funct_ implemented as (8,1)--> use transposed in code for
      // theoretic part
      ecapa->multiply_nt((fac_ * capacoeff_), funct_, funct_, 1.0);
    }

    if (ecapalin != nullptr)
    {
      // calculate additional linearization d(C(T))/dT (3-tensor!)
      // multiply with temperatures to obtain 2-tensor
      //
      // ecapalin = dC/dT*(T_{n+1} -T_{n})
      //          = fac . dercapa . (T_{n+1} -T_{n}) . (N . N^T . T)^T
      Core::LinAlg::Matrix<1, 1> Netemp(Core::LinAlg::Initialization::uninitialized);
      Core::LinAlg::Matrix<numdofpernode_ * nen_, 1> difftemp(
          Core::LinAlg::Initialization::uninitialized);
      Core::LinAlg::Matrix<numdofpernode_ * nen_, 1> NNetemp(
          Core::LinAlg::Initialization::uninitialized);
      // T_{n+1} - T_{n}
      difftemp.update(1.0, etempn_, -1.0, etemp_);
      Netemp.multiply_tn(funct_, difftemp);
      NNetemp.multiply_nn(funct_, Netemp);
      ecapalin->multiply_nt((fac_ * dercapa_), NNetemp, funct_, 1.0);
    }

  }  // --------------------------------- end loop over Gauss Points

}  // linear_thermo_contribution


/*----------------------------------------------------------------------*
 | calculate coupled fraction for the system matrix          dano 05/10 |
 | and rhs: r_T(d), k_TT(d) (public)                                    |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::linear_disp_contribution(
    const Core::Elements::Element* ele, const double time, const std::vector<double>& disp,
    const std::vector<double>& vel,
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>* econd,
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1>* efint, const Teuchos::ParameterList& params)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // now get current element displacements
  Core::LinAlg::Matrix<nen_ * nsd_, 1> edisp(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<nen_ * nsd_, 1> evel(Core::LinAlg::Initialization::uninitialized);
  for (int i = 0; i < nen_ * nsd_; i++)
  {
    edisp(i, 0) = disp[i + 0];
    evel(i, 0) = vel[i + 0];
  }


  // ------------------------------------------------ initialise material

  // thermal material tangent
  Core::LinAlg::SymmetricTensor<double, 3, 3> ctemp_t{};
  Core::LinAlg::Matrix<6, 1> ctemp = Core::LinAlg::make_stress_like_voigt_view(ctemp_t);
  // get scalar-valued element temperature
  // build the product of the shapefunctions and element temperatures T = N . T
  Core::LinAlg::Matrix<1, 1> NT(Core::LinAlg::Initialization::uninitialized);


  // ------------------------------------------------ structural material
  std::shared_ptr<Core::Mat::Material> structmat = get_str_material(ele);

  if (structmat->material_type() == Core::Materials::m_thermostvenant)
  {
    std::shared_ptr<Mat::ThermoStVenantKirchhoff> thrstvk =
        std::dynamic_pointer_cast<Mat::ThermoStVenantKirchhoff>(structmat);
  }  // m_thermostvenant

  Core::LinAlg::Matrix<nen_, 1> Ndctemp_dTBvNT(Core::LinAlg::Initialization::zero);

  // --------------------------------------------------- time integration
  // get the time step size
  const double stepsize = params.get<double>("delta time");

  // ----------------------------------- integration loop for one element

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // --------------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // compute inverse Jacobian matrix and derivatives at GP w.r.t. material
    // coordinates
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // calculate the linear B-operator
    Core::LinAlg::Matrix<6, nsd_ * nen_ * numdofpernode_> boplin(
        Core::LinAlg::Initialization::uninitialized);
    calculate_boplin(&boplin, &derxy_);

    // now build the strain rates / velocities
    Core::LinAlg::Matrix<6, 1> strainvel(Core::LinAlg::Initialization::uninitialized);
    // e' = B . d' = B . v = 0.5 * (Grad u' + Grad^T u')
    strainvel.multiply(boplin, evel);  // (6x24)(24x1)=(6x1)

    // calculate scalar-valued temperature
    NT.multiply_tn(funct_, etempn_);

    std::shared_ptr<Mat::Trait::ThermoSolid> thermoSolid =
        std::dynamic_pointer_cast<Mat::Trait::ThermoSolid>(structmat);
    if (thermoSolid != nullptr)
    {
      Core::LinAlg::SymmetricTensor<double, 3, 3> dctemp_dT_t{};
      Core::LinAlg::Matrix<6, 1> dctemp_dT = Core::LinAlg::make_stress_like_voigt_view(dctemp_dT_t);
      thermoSolid->reinit(nullptr, Core::LinAlg::TensorGenerators::full<3, 3>(0.0), NT(0), iquad);
      thermoSolid->stress_temperature_modulus_and_deriv(ctemp_t, dctemp_dT_t, iquad);

      Core::LinAlg::Matrix<nen_, 6> Ndctemp_dT(
          Core::LinAlg::Initialization::uninitialized);  // (8x1)(1x6)
      Ndctemp_dT.multiply_nt(funct_, dctemp_dT);

      Core::LinAlg::Matrix<nen_, 1> Ndctemp_dTBv(Core::LinAlg::Initialization::uninitialized);
      Ndctemp_dTBv.multiply(Ndctemp_dT, strainvel);

      Ndctemp_dTBvNT.multiply(Ndctemp_dTBv, NT);
    }
    else if (structmat->material_type() == Core::Materials::m_thermopllinelast)
    {
      std::shared_ptr<Mat::ThermoPlasticLinElast> thrpllinelast =
          std::dynamic_pointer_cast<Mat::ThermoPlasticLinElast>(structmat);
      // get the temperature-dependent material tangent
      thrpllinelast->setup_cthermo(ctemp_t);

      // thermoELASTIC heating term f_Td = T . (m . I) : strain',
      // thermoPLASTICITY:               = T . (m . I) : strain_e'
      // in case of a thermo-elasto-plastic solid material, strainvel != elastic strains
      // e' = (e^e)' + (e^p)'
      // split strainvel (=total strain) into elastic and plastic terms
      // --> thermomechanical coupling term requires elastic strain rates and
      // --> dissipation term requires the plastic strain rates
      // call the structural material

      // extract elastic part of the total strain
      thrpllinelast->strain_rate_split(iquad, stepsize, strainvel);
      // overwrite strainvel, strainvel has to include only elastic strain rates
      strainvel.update(thrpllinelast->elastic_strain_rate(iquad));

    }  // m_thermopllinelast


    // N_T^T . (- ctemp) : ( B_L .  (d^e)' )
    Core::LinAlg::Matrix<nen_, 6> Nctemp(
        Core::LinAlg::Initialization::uninitialized);  // (8x1)(1x6)
    Nctemp.multiply_nt(funct_, ctemp);
    Core::LinAlg::Matrix<nen_, 1> ncBv(Core::LinAlg::Initialization::uninitialized);
    ncBv.multiply(Nctemp, strainvel);

    // integrate internal force vector (coupling fraction towards displacements)
    if (efint != nullptr)
    {
      // fintdisp += - N_T^T . ctemp : (B_L .  (d^e)') . N_T . T
      efint->multiply((-fac_), ncBv, NT, 1.0);

    }  // if (efint != nullptr)

    // update conductivity matrix (with displacement dependent term)
    if (econd != nullptr)
    {
      // k^e += - ( N_T^T . (-m . I) . (B_L . (d^e)') . N_T ) . detJ . w(gp)
      // --> negative term enters the tangent (cf. L923) ctemp.scale(-1.0);
      econd->multiply_nt((-fac_), ncBv, funct_, 1.0);

      // in case of temperature-dependent Young's modulus, additional term for
      // conductivity matrix
      {
        // k_TT += - N_T^T . dC_T/dT : B_L . d' . N_T . T . N_T
        econd->multiply_nt(-fac_, Ndctemp_dTBvNT, funct_, 1.0);
      }

    }  // if (econd != nullptr)


  }  // ---------------------------------- end loop over Gauss Points
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::linear_coupled_tang(
    const Core::Elements::Element* ele,  // the element whose matrix is calculated
    const std::vector<double>& disp,     // current displacements
    const std::vector<double>& vel,      // current velocities
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nsd_ * nen_ * numdofpernode_>* etangcoupl,  // k_Td
    const Teuchos::ParameterList& params)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // now get current element displacements and velocities
  Core::LinAlg::Matrix<nen_ * nsd_, 1> edisp(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<nen_ * nsd_, 1> evel(Core::LinAlg::Initialization::uninitialized);
  for (int i = 0; i < nen_ * nsd_; i++)
  {
    edisp(i, 0) = disp[i + 0];
    evel(i, 0) = vel[i + 0];
  }

  // ------------------------------------------------ initialise material

  // in case of thermo-elasto-plastic material: elasto-plastic tangent modulus
  Core::LinAlg::Matrix<6, 6> cmat(Core::LinAlg::Initialization::zero);
  // thermal material tangent
  Core::LinAlg::SymmetricTensor<double, 3, 3> ctemp_t{};
  Core::LinAlg::Matrix<6, 1> ctemp = Core::LinAlg::make_stress_like_voigt_view(ctemp_t);
  // get scalar-valued element temperature
  // build the product of the shapefunctions and element temperatures T = N . T
  Core::LinAlg::Matrix<1, 1> NT(Core::LinAlg::Initialization::uninitialized);
  // get constant initial temperature from the material

  // ------------------------------------------------ structural material
  std::shared_ptr<Core::Mat::Material> structmat = get_str_material(ele);

  // --------------------------------------------------- time integration
  // check the time integrator and add correct time factor
  Thermo::DynamicType timint = Thermo::DynamicType::Undefined;
  if (params.isParameter("time integrator"))
  {
    timint = Teuchos::getIntegralValue<Thermo::DynamicType>(params, "time integrator");
  }
  // get step size dt
  const double stepsize = params.get<double>("delta time");
  // initialise time_factor
  double timefac_d = 0.0;
  double timefac = 0.0;

  // consider linearisation of velocities due to displacements
  switch (timint)
  {
    case Thermo::DynamicType::Statics:
    {
      // k_Td = k_Td^e . time_fac_d'
      timefac = 1.0;
      // timefac_d' = Lin (v_n+1) . \Delta d_n+1 = 1/dt
      // cf. Diss N. Karajan (2009) for quasistatic approach
      timefac_d = 1.0 / stepsize;
      break;
    }
    case Thermo::DynamicType::OneStepTheta:
    {
      // k_Td = theta . k_Td^e . time_fac_d'
      timefac = params.get<double>("theta");
      // timefac_d' = Lin (v_n+1) . \Delta d_n+1 = 1/(theta . dt)
      // initialise timefac_d of velocity discretisation w.r.t. displacements
      double str_theta = params.get<double>("str_theta");
      timefac_d = 1.0 / (str_theta * stepsize);
      break;
    }
    case Thermo::DynamicType::GenAlpha:
    {
      // k_Td = alphaf . k_Td^e . time_fac_d'
      timefac = params.get<double>("alphaf");
      // timefac_d' = Lin (v_n+1) . \Delta d_n+1 = gamma/(beta . dt)
      const double str_beta = params.get<double>("str_beta");
      const double str_gamma = params.get<double>("str_gamma");
      // Lin (v_n+1) . \Delta d_n+1 = (gamma) / (beta . dt)
      timefac_d = str_gamma / (str_beta * stepsize);
      break;
    }
    case Thermo::DynamicType::Undefined:
    default:
    {
      FOUR_C_THROW("Add correct temporal coefficient here!");
      break;
    }
  }  // end of switch(timint)

  // ----------------------------------- integration loop for one element

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // --------------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // GEOMETRIC LINEAR problem the deformation gradient is equal to identity

    // calculate the linear B-operator
    Core::LinAlg::Matrix<6, nsd_ * nen_ * numdofpernode_> boplin(
        Core::LinAlg::Initialization::uninitialized);
    calculate_boplin(&boplin, &derxy_);

    // non-symmetric stiffness matrix
    // current element temperatures
    NT.multiply_tn(funct_, etempn_);  // (1x8)(8x1)= (1x1)


    std::shared_ptr<Mat::Trait::ThermoSolid> thermoSolid =
        std::dynamic_pointer_cast<Mat::Trait::ThermoSolid>(structmat);
    if (thermoSolid != nullptr)
    {
      Core::LinAlg::SymmetricTensor<double, 3, 3> dctemp_dT;
      thermoSolid->reinit(nullptr, Core::LinAlg::TensorGenerators::full<3, 3>(0.0), NT(0), iquad);
      thermoSolid->stress_temperature_modulus_and_deriv(ctemp_t, dctemp_dT, iquad);
    }
    else if (structmat->material_type() == Core::Materials::m_thermopllinelast)
    {
      std::shared_ptr<Mat::ThermoPlasticLinElast> thrpllinelast =
          std::dynamic_pointer_cast<Mat::ThermoPlasticLinElast>(structmat);

      // get the temperature-dependent material tangent
      thrpllinelast->setup_cthermo(ctemp_t);
    }  // m_thermopllinelast

    // N_temp^T . N_temp . temp
    Core::LinAlg::Matrix<nen_, 1> NNT(Core::LinAlg::Initialization::uninitialized);
    NNT.multiply(funct_, NT);  // (8x1)(1x1) = (8x1)

    // N_T^T . N_T . T . ctemp
    Core::LinAlg::Matrix<nen_, 6> NNTC(Core::LinAlg::Initialization::uninitialized);  // (8x1)(1x6)
    NNTC.multiply_nt(NNT, ctemp);                                                     // (8x6)

    // coupling stiffness matrix
    if (etangcoupl != nullptr)
    {
      // k_Td^e = k_Td^e - timefac . ( N_T^T . N_T . T . C_T/str_timefac . B_L )
      //                   . detJ . w(gp)
      // with C_T = m . I
      // (8x24) = (8x6) . (6x24)
      etangcoupl->multiply_nn((-timefac * fac_ * timefac_d), NNTC, boplin, 1.0);
    }  // (etangcoupl != nullptr)

  }  //-------------------------------------- end loop over Gauss Points

}  // linear_coupled_tang()


template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::nonlinear_thermo_disp_contribution(
    const Core::Elements::Element* ele,  // the element whose matrix is calculated
    const double time,                   // current time
    const std::vector<double>& disp,     // current displacements
    const std::vector<double>& vel,      // current velocities
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>*
        econd,  // conductivity matrix
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>* ecapa,  // capacity matrix
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>*
        ecapalin,  //!< partial linearization dC/dT of capacity matrix
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1>* efint,  // internal force
    Teuchos::ParameterList& params)
{
  // update element geometry
  Core::LinAlg::Matrix<nen_, nsd_> xcurr;      // current  coord. of element
  Core::LinAlg::Matrix<nen_, nsd_> xcurrrate;  // current  coord. of element
  initial_and_current_nodal_position_velocity(ele, disp, vel, xcurr, xcurrrate);

  // ------------------------------------------------ initialise material

  // thermal material tangent
  Core::LinAlg::SymmetricTensor<double, 3, 3> ctemp_t{};
  Core::LinAlg::Matrix<6, 1> ctemp = Core::LinAlg::make_stress_like_voigt_view(ctemp_t);
  // get scalar-valued element temperature
  // build the product of the shapefunctions and element temperatures T = N . T
  Core::LinAlg::Matrix<1, 1> NT(Core::LinAlg::Initialization::uninitialized);
  // extract step size
  const double stepsize = params.get<double>("delta time");

  // ------------------------------------------------ structural material
  std::shared_ptr<Core::Mat::Material> structmat = get_str_material(ele);

  Core::LinAlg::Matrix<nen_, 1> Ndctemp_dTCrateNT(Core::LinAlg::Initialization::zero);

  // build the deformation gradient w.r.t. material configuration
  Core::LinAlg::Matrix<nsd_, nsd_> defgrd(Core::LinAlg::Initialization::uninitialized);
  // build the rate of the deformation gradient w.r.t. material configuration
  Core::LinAlg::Matrix<nsd_, nsd_> defgrdrate(Core::LinAlg::Initialization::uninitialized);
  // inverse of deformation gradient
  Core::LinAlg::Matrix<nsd_, nsd_> invdefgrd(Core::LinAlg::Initialization::uninitialized);

  // ----------------------------------- integration loop for one element

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // --------------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // compute inverse Jacobian matrix and derivatives at GP w.r.t. material
    // coordinates
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // scalar-valued current element temperature T_{n+1} = N . T
    NT.multiply_tn(funct_, etempn_);

    // ------------------------------------------------- thermal gradient
    // gradient of current temperature value
    // Grad T = d T_j / d x_i = L . N . T = B_ij T_j
    gradtemp_.multiply_nn(derxy_, etempn_);

    // ---------------------------------------- call thermal material law
    // call material law => cmat_,heatflux_ and dercmat_
    // negative q is used for balance equation:
    // heatflux_ = k_0 . Grad T
    materialize(ele, iquad);
    // heatflux_ := qintermediate = k_0 . Grad T

    // -------------------------------------------- coupling to mechanics
    // (material) deformation gradient F
    // F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
    defgrd.multiply_tt(xcurr, derxy_);
    // rate of (material) deformation gradient F'
    // F' = d xcurr' / d xrefe = (xcurr')^T * N_XYZ^T
    defgrdrate.multiply_tt(xcurrrate, derxy_);
    // inverse of deformation gradient
    invdefgrd.invert(defgrd);

    // ----------- derivatives of right Cauchy-Green deformation tensor C
    // build the rate of C: C'= F^T . F' + (F')^T . F
    // OR: C' = F^T . F' if applied to symmetric tensor
    // save C' as rate vector Crate
    // C' = { C11', C22', C33', C12', C23', C31' }
    Core::LinAlg::Matrix<6, 1> Cratevct(Core::LinAlg::Initialization::uninitialized);
    // build the inverse C: C^{-1} = F^{-1} . F^{-T}
    Core::LinAlg::Matrix<nsd_, nsd_> Cinv(Core::LinAlg::Initialization::uninitialized);
    // Cinvvct: C^{-1} in Voight-/vector notation
    // C^{-1} = { C11^{-1}, C22^{-1}, C33^{-1}, C12^{-1}, C23^{-1}, C31^{-1} }
    Core::LinAlg::SymmetricTensor<double, 3, 3> Cinv_t{};
    Core::LinAlg::Matrix<6, 1> Cinvvct = Core::LinAlg::make_stress_like_voigt_view(Cinv_t);
    calculate_cauchy_greens(Cratevct, Cinvvct, Cinv, &defgrd, &defgrdrate, &invdefgrd);

    // initial heatflux Q = C^{-1} . qintermediate = k_0 . C^{-1} . B_T . T
    // the current heatflux q = detF . F^{-1} . q
    // store heatflux
    // (3x1)  (3x3) . (3x1)
    Core::LinAlg::Matrix<nsd_, 1> initialheatflux(Core::LinAlg::Initialization::uninitialized);
    initialheatflux.multiply(Cinv, heatflux_);
    // put the initial, material heatflux onto heatflux_
    heatflux_.update(initialheatflux);
    // from here on heatflux_ == -Q

    std::shared_ptr<Mat::Trait::ThermoSolid> thermoSolid =
        std::dynamic_pointer_cast<Mat::Trait::ThermoSolid>(structmat);
    if (thermoSolid != nullptr)
    {
      Core::LinAlg::SymmetricTensor<double, 3, 3> dctemp_dT_t{};
      Core::LinAlg::Matrix<6, 1> dctemp_dT = Core::LinAlg::make_stress_like_voigt_view(dctemp_dT_t);
      thermoSolid->reinit(nullptr, Core::LinAlg::TensorGenerators::full<3, 3>(0.0), NT(0), iquad);
      thermoSolid->stress_temperature_modulus_and_deriv(ctemp_t, dctemp_dT_t, iquad);
      // scalar product: dctemp_dTCdot = dC_T/dT : 1/2 C'
      double dctemp_dTCdot = 0.0;
      for (int i = 0; i < 6; ++i)
        dctemp_dTCdot += dctemp_dT(i, 0) * (1 / 2.0) * Cratevct(i, 0);  // (6x1)(6x1)

      Core::LinAlg::Matrix<nen_, 1> Ndctemp_dTCratevct(Core::LinAlg::Initialization::uninitialized);
      Ndctemp_dTCratevct.update(dctemp_dTCdot, funct_);
      Ndctemp_dTCrateNT.multiply(Ndctemp_dTCratevct, NT);  // (8x1)(1x1)

      // ------------------------------------ special terms due to material law
      // if young's modulus is temperature-dependent, E(T), additional terms arise
      // for the stiffness matrix k_TT
      if (econd != nullptr)
      {
        // k_TT += - N_T^T . dC_T/dT : C' . N_T . T . N_T
        // with dC_T/dT = d(m . I)/dT = d (m(T) . I)/dT
        //
        // k_TT += - N_T^T . dC_T/dT : C' . N_T . T . N_T
        econd->multiply_nt(-fac_, Ndctemp_dTCrateNT, funct_, 1.0);
      }  // (econd != nullptr)
    }
    if (structmat->material_type() == Core::Materials::m_thermoplhyperelast)
    {
      std::shared_ptr<Mat::ThermoPlasticHyperElast> thermoplhyperelast =
          std::dynamic_pointer_cast<Mat::ThermoPlasticHyperElast>(structmat);

      // insert matrices into parameter list which are only required for thrplasthyperelast
      params.set<Core::LinAlg::Matrix<nsd_, nsd_>>("defgrd", defgrd);
      params.set<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("Cinv_vct", Cinvvct);

      // ------------ (non-dissipative) thermoelastic and -plastic heating term
      // H_ep := H_e + H_p = T . dsigma/dT . E' + T . dkappa/dT . astrain^p'

      // --------------------(non-dissipative) thermoelastic heating term
      // H_e := N_T^T . N_T . T . (-C_T) : 1/2 C'
      thermoplhyperelast->setup_cthermo(ctemp_t, defgrd.determinant(), Cinv_t);

      // --------------------(non-dissipative) thermoplastic heating term
      // H_p := - N^T_T . N_T . T . dkappa/dT . sqrt(2/3) . Dgamma/Dt
      // H_p := - N^T_T . N_T . T . thrplheat . 1/Dt
      double thrplheat = thermoplhyperelast->thermo_plast_heating(iquad);

      if (efint != nullptr)
      {
        // fint += - N^T_T . N_T . T . thrplheat . 1/Dt . detJ . w(gp)
        efint->multiply((-thrplheat / stepsize * fac_), funct_, NT, 1.0);
      }

      if (econd != nullptr)
      {
        // k_TT += - N^T_T . thrplheat . 1/Dt . N_T . detJ . w(gp)
        econd->multiply_nt((-thrplheat / stepsize * fac_), funct_, funct_, 1.0);
        // k_TT += - N^T_T . N_T . T . 1/Dt . dH_p/dT . N_T . detJ . w(gp)
        double thrplheat_kTT = thermoplhyperelast->thermo_plast_heating_k_tt(iquad);
        econd->multiply_nt((-NT(0, 0) * thrplheat_kTT / stepsize * fac_), funct_, funct_, 1.0);
      }
    }  // m_thermoplhyperelast

    // --------------------------------------------- terms for r_T / k_TT
    // scalar product: ctempcdot = C_T : 1/2 C'
    double ctempCdot = 0.0;
    for (int i = 0; i < 6; ++i) ctempCdot += ctemp(i, 0) * (1 / 2.0) * Cratevct(i, 0);

    // ------------------------------ integrate internal force vector r_T
    // add the displacement-dependent terms to fint
    // fint = fint + fint_{Td}
    if (efint != nullptr)
    {
      // fint += B_T^T . Q . detJ * w(gp)
      //      += B_T^T . (k_0) . C^{-1} . B_T . T . detJ . w(gp)
      // (8x1)   (8x3) (3x1)
      efint->multiply_tn(fac_, derxy_, heatflux_, 1.0);

#ifndef TSISLMNOGOUGHJOULE
      // fint_{Td} = - N^T . ctemp : (1/2 . C') . N . T
      //              (1x8)  (6x1)       (6x1)(8x1)(8x1)
      //              (1x8)        (1x1)        (1x1)
      // fint = fint + fint_{Td}
      // with fint_{Td} += - N^T . ctemp : (1/2 . C') . N . T +
      //                   + B^T . k_0 . F^{-1} . F^{-T} . B . T
      if (structmat->material_type() == Core::Materials::m_plelasthyper)
      {
        std::shared_ptr<Mat::PlasticElastHyper> plmat =
            std::dynamic_pointer_cast<Mat::PlasticElastHyper>(structmat);
        double He = plmat->hep_diss(iquad);
        efint->update((-fac_ * He), funct_, 1.0);
      }
      else
        efint->multiply((-fac_ * ctempCdot), funct_, NT, 1.0);
#endif
      // efint += H_p term is added to fint within material call

    }  // (efint != nullptr)

    // ------------------------------- integrate conductivity matrix k_TT
    // update conductivity matrix k_TT (with displacement dependent term)
    if (econd != nullptr)
    {
      // k^e_TT += ( B_T^T . C^{-1} . C_mat . B_T ) . detJ . w(gp)
      // 3D:        (8x3)    (3x3)    (3x3)   (3x8)
      // with C_mat = k_0 . I
      // -q = C_mat . C^{-1} . B
      Core::LinAlg::Matrix<nsd_, nen_> aop(Core::LinAlg::Initialization::uninitialized);  // (3x8)
      aop.multiply_nn(cmat_, derxy_);  // (nsd_xnsd_)(nsd_xnen_)
      Core::LinAlg::Matrix<nsd_, nen_> aop1(Core::LinAlg::Initialization::uninitialized);  // (3x8)
      aop1.multiply_nn(Cinv, aop);  // (nsd_xnsd_)(nsd_xnen_)

      // k^e_TT += ( B_T^T . C^{-1} . C_mat . B_T ) . detJ . w(gp)
      econd->multiply_tn(fac_, derxy_, aop1, 1.0);  //(8x8)=(8x3)(3x8)

      // linearization of non-constant conductivity
      // k^e_TT += ( B_T^T . C^{-1} . dC_mat . B_T . T . N) . detJ . w(gp)
      Core::LinAlg::Matrix<nsd_, 1> dCmatGradT(Core::LinAlg::Initialization::uninitialized);
      dCmatGradT.multiply_nn(dercmat_, gradtemp_);
      Core::LinAlg::Matrix<nsd_, 1> CinvdCmatGradT(Core::LinAlg::Initialization::uninitialized);
      CinvdCmatGradT.multiply_nn(Cinv, dCmatGradT);
      Core::LinAlg::Matrix<nsd_, nen_> CinvdCmatGradTN(Core::LinAlg::Initialization::uninitialized);
      CinvdCmatGradTN.multiply_nt(CinvdCmatGradT, funct_);
      econd->multiply_tn(fac_, derxy_, CinvdCmatGradTN, 1.0);  //(8x8)=(8x3)(3x8)
#ifndef TSISLMNOGOUGHJOULE
      // linearization of thermo-mechanical effects
      if (structmat->material_type() == Core::Materials::m_plelasthyper)
      {
        std::shared_ptr<Mat::PlasticElastHyper> plmat =
            std::dynamic_pointer_cast<Mat::PlasticElastHyper>(structmat);
        double dHeDT = plmat->d_hep_dt(iquad);
        econd->multiply_nt((-fac_ * dHeDT), funct_, funct_, 1.0);
        if (plmat->d_hep_d_teas() != nullptr)
          Core::LinAlg::DenseFunctions::multiply_nt<double, nen_, 1, nen_>(
              1., econd->data(), -fac_, funct_.data(), plmat->d_hep_d_teas()->at(iquad).values());
      }
      else
        econd->multiply_nt((-fac_ * ctempCdot), funct_, funct_, 1.0);
#endif
      // be aware: special terms of materials are added within material call
    }  // (econd != nullptr)

    // --------------------------------------- capacity matrix m_capa
    // capacity matrix is independent of deformation
    // m_capa corresponds to the mass matrix of the structural field
    if (ecapa != nullptr)
    {
      // m_capa = m_capa + ( N_T^T .  (rho_0 . C_V) . N_T ) . detJ . w(gp)
      //           (8x8)     (8x1)                 (1x8)
      // caution: funct_ implemented as (8,1)--> use transposed in code for
      // theoretic part
      ecapa->multiply_nt((fac_ * capacoeff_), funct_, funct_, 1.0);
    }  // (ecapa != nullptr)
    if (ecapalin != nullptr)
    {
      // calculate additional linearization d(C(T))/dT (3-tensor!)
      // multiply with temperatures to obtain 2-tensor
      //
      // ecapalin = dC/dT*(T_{n+1} -T_{n})
      //          = fac . dercapa . (T_{n+1} -T_{n}) . (N . N^T . T)^T
      Core::LinAlg::Matrix<1, 1> Netemp(Core::LinAlg::Initialization::uninitialized);
      Core::LinAlg::Matrix<numdofpernode_ * nen_, 1> difftemp(
          Core::LinAlg::Initialization::uninitialized);
      Core::LinAlg::Matrix<numdofpernode_ * nen_, 1> NNetemp(
          Core::LinAlg::Initialization::uninitialized);
      // T_{n+1} - T_{n}
      difftemp.update(1.0, etempn_, -1.0, etemp_);
      Netemp.multiply_tn(funct_, difftemp);
      NNetemp.multiply_nn(funct_, Netemp);
      ecapalin->multiply_nt((fac_ * dercapa_), NNetemp, funct_, 1.0);
    }

  }  // ---------------------------------- end loop over Gauss Points
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::nonlinear_coupled_tang(
    const Core::Elements::Element* ele,  // the element whose matrix is calculated
    const std::vector<double>& disp,     // current displacements
    const std::vector<double>& vel,      // current velocities
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nsd_ * nen_ * numdofpernode_>* etangcoupl,
    Teuchos::ParameterList& params  // parameter list
)
{
  // update element geometry
  Core::LinAlg::Matrix<nen_, nsd_> xcurr(
      Core::LinAlg::Initialization::uninitialized);  // current  coord. of element
  Core::LinAlg::Matrix<nen_, nsd_> xcurrrate(
      Core::LinAlg::Initialization::uninitialized);  // current  velocity of element
  initial_and_current_nodal_position_velocity(ele, disp, vel, xcurr, xcurrrate);

  // --------------------------------------------------- time integration

  // get step size dt
  const double stepsize = params.get<double>("delta time");
  // initialise time_fac of velocity discretisation w.r.t. displacements
  double timefac_d = 0.0;
  double timefac = 0.0;
  // check the time integrator and add correct time factor
  const auto timint =
      params.get<Thermo::DynamicType>("time integrator", Thermo::DynamicType::Undefined);
  switch (timint)
  {
    case Thermo::DynamicType::Statics:
    {
      timefac = 1.0;
      break;
    }
    case Thermo::DynamicType::OneStepTheta:
    {
      // k^e_Td += + theta . N_T^T . (-C_T) . 1/2 dC'/dd . N_T . T . detJ . w(gp) -
      //           - theta . ( B_T^T . C_mat . dC^{-1}/dd . B_T . T . detJ . w(gp) )
      //           - theta . N^T_T . N_T . T . 1/Dt . dthplheat_kTd/dd
      const double theta = params.get<double>("theta");
      // K_Td = theta . K_Td
      timefac = theta;
      break;
    }
    case Thermo::DynamicType::GenAlpha:
    {
      timefac = params.get<double>("alphaf");
      break;
    }
    case Thermo::DynamicType::Undefined:
    default:
    {
      FOUR_C_THROW("Add correct temporal coefficient here!");
      break;
    }
  }  // end of switch(timint)

  const auto s_timint =
      Teuchos::getIntegralValue<Inpar::Solid::DynamicType>(params, "structural time integrator");
  switch (s_timint)
  {
    case Inpar::Solid::DynamicType::Statics:
    {
      timefac_d = 1.0 / stepsize;
      break;
    }
    case Inpar::Solid::DynamicType::GenAlpha:
    {
      const double str_beta = params.get<double>("str_beta");
      const double str_gamma = params.get<double>("str_gamma");
      timefac_d = str_gamma / (str_beta * stepsize);
      break;
    }
    case Inpar::Solid::DynamicType::OneStepTheta:
    {
      const double str_theta = params.get<double>("str_theta");
      timefac_d = 1.0 / (stepsize * str_theta);
      break;
    }
    default:
      FOUR_C_THROW("unknown structural time integrator type");
  }

  // ------------------------------------------------ initialise material

  // get scalar-valued element temperature
  // build the product of the shapefunctions and element temperatures T = N . T
  Core::LinAlg::Matrix<1, 1> NT(Core::LinAlg::Initialization::uninitialized);
  // N_T^T . N_T . T
  Core::LinAlg::Matrix<nen_, 1> NNT(Core::LinAlg::Initialization::uninitialized);
  // thermal material tangent
  Core::LinAlg::SymmetricTensor<double, 3, 3> ctemp_t{};
  Core::LinAlg::Matrix<6, 1> ctemp = Core::LinAlg::make_stress_like_voigt_view(ctemp_t);

  // ------------------------------------------------ structural material
  std::shared_ptr<Core::Mat::Material> structmat = get_str_material(ele);

  // build the deformation gradient w.r.t. material configuration
  Core::LinAlg::Matrix<nsd_, nsd_> defgrd(Core::LinAlg::Initialization::uninitialized);
  // build the rate of the deformation gradient w.r.t. material configuration
  Core::LinAlg::Matrix<nsd_, nsd_> defgrdrate(Core::LinAlg::Initialization::uninitialized);
  // inverse of deformation gradient
  Core::LinAlg::Matrix<nsd_, nsd_> invdefgrd(Core::LinAlg::Initialization::zero);
  // initialise Jacobi-determinant
  double J = 0.0;

  // ------------------------------- integration loop for one element

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // ----------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // compute inverse Jacobian matrix and derivatives at GP w.r.t. material
    // coordinates
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // ------------------------------------------------ thermal terms

    // gradient of current temperature value
    // grad T = d T_j / d x_i = L . N . T = B_ij T_j
    gradtemp_.multiply_nn(derxy_, etempn_);

    // call material law => cmat_,heatflux_
    // negative q is used for balance equation: -q = -(-k gradtemp)= k * gradtemp
    materialize(ele, iquad);

    // put thermal material tangent in vector notation
    Core::LinAlg::Matrix<6, 1> cmat_vct(Core::LinAlg::Initialization::zero);
    for (unsigned i = 0; i < nsd_; ++i) cmat_vct(i) = cmat_(i, i);

    // B_T^T . B_T . T
    Core::LinAlg::Matrix<nen_, 1> bgradT(Core::LinAlg::Initialization::uninitialized);
    bgradT.multiply_tn(derxy_, gradtemp_);  // (8x1)(1x1) = (8x1)
    // B_T^T . B_T . T . Cmat_
    Core::LinAlg::Matrix<nen_, 6> bgradTcmat(
        Core::LinAlg::Initialization::uninitialized);  // (8x1)(1x6)
    bgradTcmat.multiply_nt(bgradT, cmat_vct);          // (8x6)

    // current element temperatures
    // N_T . T (funct_ defined as <nen,1>)
    NT.multiply_tn(funct_, etempn_);  // (1x8)(8x1)
    NNT.multiply(funct_, NT);         // (8x1)(1x1)

    // ---------------------------------------- coupling to mechanics
    // (material) deformation gradient F
    // F = d xcurr / d xrefe = xcurr^T . N_XYZ^T
    defgrd.multiply_tt(xcurr, derxy_);
    // rate of (material) deformation gradient F'
    // F' = d xcurr' / d xrefe = (xcurr')^T . N_XYZ^T
    defgrdrate.multiply_tt(xcurrrate, derxy_);
    // inverse of deformation gradient
    invdefgrd.invert(defgrd);
    // build the linear B-operator
    Core::LinAlg::Matrix<6, nsd_ * nen_ * numdofpernode_> boplin(
        Core::LinAlg::Initialization::uninitialized);
    calculate_boplin(&boplin, &derxy_);
    // build the nonlinear B-operator
    Core::LinAlg::Matrix<6, nen_ * nsd_ * numdofpernode_> bop(
        Core::LinAlg::Initialization::uninitialized);
    calculate_bop(&bop, &defgrd, &derxy_);

    // ------- derivatives of right Cauchy-Green deformation tensor C

    // build the rate of C: C'= F^T . F' + (F')^T . F
    // save C' as rate vector Crate
    // C' = { C11', C22', C33', C12', C23', C31 }
    Core::LinAlg::Matrix<6, 1> Cratevct(Core::LinAlg::Initialization::uninitialized);
    // build the inverse C: C^{-1} = F^{-1} . F^{-T}
    Core::LinAlg::Matrix<nsd_, nsd_> Cinv(Core::LinAlg::Initialization::uninitialized);
    // Cinvvct: C^{-1} in Voight-/vector notation
    // C^{-1} = { C11^{-1}, C22^{-1}, C33^{-1}, C12^{-1}, C23^{-1}, C31^{-1} }
    Core::LinAlg::SymmetricTensor<double, 3, 3> Cinv_t{};
    Core::LinAlg::Matrix<6, 1> Cinvvct = Core::LinAlg::make_stress_like_voigt_view(Cinv_t);
    // calculation is done in calculate_cauchy_greens, return C', C^{-1} in vector
    // notation, NO Voigt-notation
    calculate_cauchy_greens(Cratevct, Cinvvct, Cinv, &defgrd, &defgrdrate, &invdefgrd);

    // ------------------------------------ calculate linearisation of C'

    // C_T : 1/2 dC'/dd --> symmetric part of dC'/dd is sufficient
    // dC'/dd = dCrate/dd = 1/2 . [ timefac_d . (B^T + B) + (F')^T . B_L + B_L^T . F' ]
    //        = timefac_d [ B^T + B ] + [ (F')^T . B_L + ( (F')^T . B_L )^T ]
    // C_T : 1/2 dC'/dd = C_T : sym[ timefac_d B + B' ]
    // --> use only the symmetric part of dC'/dd

    // with B' = (F')^T . B_L: calculate rate of B
    Core::LinAlg::Matrix<6, nen_ * nsd_> boprate(
        Core::LinAlg::Initialization::uninitialized);  // (6x24)
    calculate_bop(&boprate, &defgrdrate, &derxy_);

    // -------------------------------- calculate linearisation of C^{-1}

    // calculate linearisation of C^{-1} according to so3_poro_evaluate: compute_auxiliary_values()
    // dC^{-1}/dd = dCinv_dd = - F^{-1} . ( B_L . F^{-1} + F^{-T} . B_L^T ) . F^{-T}
    //                       = - F^{-1} . ( B_L . F^{-1} + (B_L . F^{-1})^T ) . F^{-T}
    Core::LinAlg::Matrix<6, nen_ * nsd_> dCinv_dd(Core::LinAlg::Initialization::zero);
    for (int n = 0; n < nen_; ++n)
    {
      for (int k = 0; k < nsd_; ++k)
      {
        const int gid = n * nsd_ + k;
        for (int i = 0; i < nsd_; ++i)
        {
          dCinv_dd(0, gid) += -2 * Cinv(0, i) * derxy_(i, n) * invdefgrd(0, k);
          if constexpr (nsd_ == 2)
          {
            dCinv_dd(1, gid) += -2 * Cinv(1, i) * derxy_(i, n) * invdefgrd(1, k);
            dCinv_dd(2, gid) += -Cinv(0, i) * derxy_(i, n) * invdefgrd(1, k) -
                                invdefgrd(0, k) * derxy_(i, n) * Cinv(1, i);
          }
          else if constexpr (nsd_ == 3)
          {
            dCinv_dd(1, gid) += -2 * Cinv(1, i) * derxy_(i, n) * invdefgrd(1, k);
            dCinv_dd(2, gid) += -2 * Cinv(2, i) * derxy_(i, n) * invdefgrd(2, k);
            dCinv_dd(3, gid) += -Cinv(0, i) * derxy_(i, n) * invdefgrd(1, k) -
                                invdefgrd(0, k) * derxy_(i, n) * Cinv(1, i);
            dCinv_dd(4, gid) += -Cinv(1, i) * derxy_(i, n) * invdefgrd(2, k) -
                                invdefgrd(1, k) * derxy_(i, n) * Cinv(2, i);
            dCinv_dd(5, gid) += -Cinv(2, i) * derxy_(i, n) * invdefgrd(0, k) -
                                invdefgrd(2, k) * derxy_(i, n) * Cinv(0, i);
          }
        }
      }
    }  // end DCinv_dd

    std::shared_ptr<Mat::Trait::ThermoSolid> thermoSolid =
        std::dynamic_pointer_cast<Mat::Trait::ThermoSolid>(structmat);
    if (thermoSolid != nullptr)
    {
      Core::LinAlg::SymmetricTensor<double, 3, 3> dctemp_dT;
      thermoSolid->reinit(nullptr, Core::LinAlg::TensorGenerators::full<3, 3>(0.0), NT(0), iquad);
      thermoSolid->stress_temperature_modulus_and_deriv(ctemp_t, dctemp_dT, iquad);
    }
    if (structmat->material_type() == Core::Materials::m_thermoplhyperelast)
    {
      // C_T = m_0 . (J + 1/J) . C^{-1}
      // thermoelastic heating term
      std::shared_ptr<Mat::ThermoPlasticHyperElast> thermoplhyperelast =
          std::dynamic_pointer_cast<Mat::ThermoPlasticHyperElast>(structmat);

      // insert matrices into parameter list which are only required for thrplasthyperelast
      params.set<Core::LinAlg::Matrix<nsd_, nsd_>>("defgrd", defgrd);
      params.set<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("Cinv_vct", Cinvvct);
      // calculate Jacobi-determinant
      J = defgrd.determinant();

      // H_e := - N_T^T . N_T . T . C_T : 1/2 C'
      thermoplhyperelast->setup_cthermo(ctemp_t, J, Cinv_t);
    }
    // N_T^T . N_T . T . ctemp
    Core::LinAlg::Matrix<nen_, 6> NNTC(Core::LinAlg::Initialization::uninitialized);  // (8x1)(1x6)
    NNTC.multiply_nt(NNT, ctemp);                                                     // (8x6)

    // ----------------- coupling matrix k_Td only for monolithic TSI
    if (etangcoupl != nullptr)
    {
      // for PlasticElastHyper materials (i.e. Semi-smooth Newton type plasticity)
      // these coupling terms have already been computed during the structural
      // evaluate to efficiently combine it with the condensation of plastic
      // deformation DOFs
      if (structmat->material_type() == Core::Materials::m_plelasthyper)
      {
        std::shared_ptr<Mat::PlasticElastHyper> plmat =
            std::dynamic_pointer_cast<Mat::PlasticElastHyper>(structmat);
        Core::LinAlg::DenseFunctions::multiply_nt<double, nen_, 1, nsd_ * nen_>(
            1., etangcoupl->data(), -fac_, funct_.data(), plmat->d_hep_diss_dd(iquad).values());
      }
      // other materials do specific computations here
      else
      {
        // B_T: thermal gradient matrix
        // B_L: linear B-operator, gradient matrix == B_T
        // B: nonlinear B-operator, i.e. B = F^T . B_L
        // dC'/dd = timefac_d ( B^T + B ) + F'T . B_L + B_L^T . F'
        // --> 1/2 dC'/dd = sym dC'/dd = 1/(theta . Dt) . B + B'
        // with boprate := B' = F'^T . B_L
        // dC^{-1}/dd = - F^{-1} . (B_L . F^{-1} + B_L^{T} . F^{-T}) . F^{-T}
        //
        // C_mat = k_0 . I

        // k^e_Td += - timefac . N_T^T . N_T . T . C_T : 1/2 dC'/dd . detJ . w(gp)
        // (8x24)                (8x3) (3x8)(8x1)   (6x1)       (6x24)
        // (8x24)                   (8x8)   (8x1)   (1x6)       (6x24)
        // (8x24)                       (8x1)       (1x6)       (6x24)
        // (8x24)                             (8x6)             (6x24)
        etangcoupl->multiply(-fac_, NNTC, boprate, 1.0);
        etangcoupl->multiply((-fac_ * timefac_d), NNTC, bop, 1.0);
      }
      // k^e_Td += timefac . ( B_T^T . C_mat . dC^{-1}/dd . B_T . T . detJ . w(gp) )
      //        += timefac . ( B_T^T . C_mat . B_T . T . dC^{-1}/dd . detJ . w(gp) )
      // (8x24)                        (8x3)   (3x3)  (3x8)(8x1)  (6x24)
      //                                 (8x3)        (3x1)
      //                                       (8x1) (1x24)
      // k^e_Td += timefac . ( B_T^T . B_T . T . C_mat . dC^{-1}/dd . detJ . w(gp) )
      // (8x24)                (8x3)  (3x8)(8x1) (1x6) (6x24)

      Core::LinAlg::Matrix<nen_, Mat::NUM_STRESS_3D> bgradTcmat(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<nsd_, 1> G;
      G.multiply(cmat_, gradtemp_);
      for (int i = 0; i < nen_; i++)
      {
        bgradTcmat(i, 0) = derxy_(0, i) * G(0);
        if (nsd_ == 2)
        {
          bgradTcmat(i, 1) = derxy_(1, i) * G(1);
          bgradTcmat(i, 2) = (derxy_(0, i) * G(1) + derxy_(1, i) * G(0));
        }
        if (nsd_ == 3)
        {
          bgradTcmat(i, 1) = derxy_(1, i) * G(1);
          bgradTcmat(i, 2) = derxy_(2, i) * G(2);
          bgradTcmat(i, 3) = (derxy_(0, i) * G(1) + derxy_(1, i) * G(0));
          bgradTcmat(i, 4) = (derxy_(2, i) * G(1) + derxy_(1, i) * G(2));
          bgradTcmat(i, 5) = (derxy_(0, i) * G(2) + derxy_(2, i) * G(0));
        }
      }

      etangcoupl->multiply_nn(fac_, bgradTcmat, dCinv_dd, 1.0);
    }  // (etangcoupl != nullptr)

    if (structmat->material_type() == Core::Materials::m_thermoplhyperelast)
    {
      // --------- additional terms due to linearisation of H_ep w.r.t. d_{n+1}

      // k_Td += - timefac . N^T_T . dH_ep/dd
      //       = - timefac . N^T_T . dH_e/dd - timefac . N^T_T . dH_p/dd
      //       = - timefac . N^T_T [ m_0 . (1 - 1/J^2) dJ/dd . C^{-1} +
      //                             + (J + 1/J) . dC^{-1}/dd ] : 1/2 C' . N_T . T
      //         - timefac . N^T_T . N_T . T . 1/Dt . thrplheat_kTd . dE/dd ]

      // get material
      std::shared_ptr<Mat::ThermoPlasticHyperElast> thermoplhyperelast =
          std::dynamic_pointer_cast<Mat::ThermoPlasticHyperElast>(structmat);

      // dJ/dd (1x24)
      Core::LinAlg::Matrix<1, nsd_ * nen_ * numdofpernode_> dJ_dd(
          Core::LinAlg::Initialization::zero);
      calculate_linearisation_of_jacobian(dJ_dd, J, derxy_, invdefgrd);

      // --------------------------------- thermoelastic heating term H_e

      // k_Td += - timefac . N^T_T . N_T . T .
      //         [ m_0 . (1 - 1/J^2) dJ/dd . C^{-1}
      //           + m_0 . (J + 1/J) . dC^{-1}/dd ] : 1/2 C' . N_T . T ]

      // m_0 . (1 - 1/J^2) . C^{-1} . dJ/dd + m_0 . (J + 1/J) . dC^{-1}/dd
      //                     (6x1)    (1x24)                     (6x24)
      const double m_0 = thermoplhyperelast->st_modulus();
      double fac_He_dJ = m_0 * (1.0 - 1.0 / (J * J));
      double fac_He_dCinv = m_0 * (J + 1.0 / J);

      Core::LinAlg::Matrix<6, nsd_ * nen_ * numdofpernode_> dC_T_dd(
          Core::LinAlg::Initialization::uninitialized);  // (6x24)
      dC_T_dd.multiply(fac_He_dJ, Cinvvct, dJ_dd);
      dC_T_dd.update(fac_He_dCinv, dCinv_dd, 1.0);
      // dC_T_dd : 1/2 C'
      Core::LinAlg::Matrix<1, nsd_ * nen_ * numdofpernode_> dC_T_ddCdot(
          Core::LinAlg::Initialization::uninitialized);  // (1x24)
      dC_T_ddCdot.multiply_tn(0.5, Cratevct, dC_T_dd);

      // dC_T/dd
      // k_Td += - timefac . N^T_T . N_T . T . [ m_0 . (1 - 1/J^2) . dJ/dd . C^{-1}
      //               + m_0 . (J + 1/J) . dC^{-1}/dd ] : 1/2 C' . detJ . w(gp)
      etangcoupl->multiply_nn((-fac_ * NT(0.0)), funct_, dC_T_ddCdot, 1.0);

      // ---------------- linearisation of thermoplastic heating term H_p

      // k_Td += - timefac . N_T^T . N_T . T . 1/Dt . thrplheat_kTd . dE/dd

      // dH_p/dE = 1/Dt . [ ddkappa/dTdastrain . 2/3 . Dgamma + dkappa/dT . sqrt(2/3) ] . dDgamma/dE
      Core::LinAlg::Matrix<1, nsd_ * nen_ * numdofpernode_> dHp_dd(
          Core::LinAlg::Initialization::uninitialized);
      dHp_dd.multiply_tn(thermoplhyperelast->thermo_plast_heating_k_td(iquad), bop);
      // k_Td += - timefac . N_T . T . 1/Dt . N_T^T . dH_p/dd . detJ . w(gp)
      etangcoupl->multiply((-fac_ * NT(0.0) / stepsize), funct_, dHp_dd, 1.0);

    }  // m_thermoplhyperelast

  }  // ---------------------------------- end loop over Gauss Points

  // scale total tangent with timefac
  if (etangcoupl != nullptr)
  {
    etangcoupl->scale(timefac);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::linear_dissipation_fint(
    const Core::Elements::Element* ele,  // the element whose matrix is calculated
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1>* efint,  // internal force
    Teuchos::ParameterList& params)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // --------------------------------------------------------- initialise
  // thermal material tangent
  Core::LinAlg::Matrix<6, 1> ctemp(Core::LinAlg::Initialization::zero);

  // ------------------------------------------------ structural material
  std::shared_ptr<Core::Mat::Material> structmat = get_str_material(ele);

  if (structmat->material_type() != Core::Materials::m_thermopllinelast)
  {
    FOUR_C_THROW("So far dissipation only for ThermoPlasticLinElast material!");
  }
  std::shared_ptr<Mat::ThermoPlasticLinElast> thrpllinelast =
      std::dynamic_pointer_cast<Mat::ThermoPlasticLinElast>(structmat);
  // true: error if cast fails

  // --------------------------------------------------- time integration
  // get step size dt
  const double stepsize = params.get<double>("delta time");

  // ----------------------------------- integration loop for one element

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // --------------------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // compute inverse Jacobian matrix and derivatives at GP w.r.t. material
    // coordinates
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // GEOMETRIC LINEAR problem the deformation gradient is equal to identity

    // build the linear B-operator
    Core::LinAlg::Matrix<6, nsd_ * nen_ * numdofpernode_> boplin(
        Core::LinAlg::Initialization::uninitialized);
    calculate_boplin(&boplin, &derxy_);

    // ------------------------------------------------------------ dissipation

    // D_mech = - N_T^T . (sigma_{d,T} - beta) . strain^p'
    //          + N_T^T Hiso strainbar^p . strainbar^p'
    // with eta = sigma_d - beta
    // --> consider sigma_T separately

    // --------------------------------------- Dmech due to kinematic hardening
    // Dmech_kin = N_T^T . (sigma_{d,T} - beta) . strain^p'

    // for a thermo-elasto-plastic solid material: strainvel == total strain e'
    // split strainvel into elastic and plastic terms
    // additive split of strains: e' = (e^e)' + (e^p)'

    // ------------------------------------------------ mechanical contribution
    // Dmech_kin = (sigma_d - beta) : strain^p_{n+1}'

    // --------------------------------------- Dmech due to isotropic hardening
    // N_T^T . kappa . strainbar^p' = N_T^T . Hiso . strainbar^p . Dgamma/dt
    // kappa = kappa(strainbar^p): isotropic work hardening von Mises stress

    // Dmech += Hiso . strainbar^p . Dgamma
    double Dmech = thrpllinelast->mechanical_kinematic_dissipation(iquad) / stepsize;

    // CAUTION: (tr(strain^p) == 0) and sigma_T(i,i)=const.
    // --> neglect: Dmech = -sigma_{T,n+1} : strain^p_{n+1}' == 0: (vol:dev == 0)
    // --> no additional terms for fint, nor for econd!

    // update/integrate internal force vector (coupling fraction towards displacements)
    if (efint != nullptr)
    {
      // update of the internal "force" vector
      // fint += N_T^T . 1/Dt . Dmech . detJ . w(gp)
      efint->update((fac_ * Dmech), funct_, 1.0);
    }

  }  // -------------------------------------------- end loop over Gauss Points
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::linear_dissipation_coupled_tang(
    const Core::Elements::Element* ele,  // the element whose matrix is calculated
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nsd_ * nen_ * numdofpernode_>* etangcoupl,  // k_Td
    Teuchos::ParameterList& params)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // ------------------------------------------------ structural material
  std::shared_ptr<Core::Mat::Material> structmat = get_str_material(ele);
  if (structmat->material_type() != Core::Materials::m_thermopllinelast)
  {
    FOUR_C_THROW("So far dissipation only available for ThermoPlasticLinElast material!");
  }
  std::shared_ptr<Mat::ThermoPlasticLinElast> thrpllinelast =
      std::dynamic_pointer_cast<Mat::ThermoPlasticLinElast>(structmat);
  // true: error if cast fails

  // --------------------------------------------------- time integration
  // get step size dt
  const double stepsize = params.get<double>("delta time");

  // check the time integrator and add correct time factor
  const auto timint =
      params.get<Thermo::DynamicType>("time integrator", Thermo::DynamicType::Undefined);
  // initialise time_fac of velocity discretisation w.r.t. displacements
  double timefac = 0.0;
  switch (timint)
  {
    case Thermo::DynamicType::Statics:
    {
      // evolution equation of plastic material use implicit Euler
      // put str_timefac = 1.0
      timefac = 1.0;
      break;
    }
    case Thermo::DynamicType::OneStepTheta:
    {
      // k_Td = theta . k_Td^e . timefac_Dgamma = theta . k_Td / Dt
      double theta = params.get<double>("theta");
      timefac = theta;
      break;
    }
    case Thermo::DynamicType::GenAlpha:
    {
      // k_Td = alphaf . k_Td^e . timefac_Dgamma = alphaf . k_Td / Dt
      double alphaf = params.get<double>("alphaf");
      timefac = alphaf;
      break;
    }
    case Thermo::DynamicType::Undefined:
    default:
    {
      FOUR_C_THROW("Add correct temporal coefficient here!");
      break;
    }
  }  // end of switch(timint)

  // ----------------------------------- integration loop for one element

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // --------------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // compute inverse Jacobian matrix and derivatives at GP w.r.t. material
    // coordinates
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // GEOMETRIC LINEAR problem the deformation gradient is equal to identity

    // calculate the linear B-operator
    Core::LinAlg::Matrix<6, nsd_ * nen_ * numdofpernode_> boplin(
        Core::LinAlg::Initialization::uninitialized);
    calculate_boplin(&boplin, &derxy_);

    // --------------------------- calculate linearisation of dissipation

    // k_Td = Lin [D_mech] . Inc_d
    //      = (dD_mech/dstrain) : Lin [ strain ] . Inc_d
    //      = (dD_mech/dstrain) . B_d . Inc_d
    //
    // --> perform the linearisation w.r.t. to strains, NOT w.r.t. to displacements

    // calculate the derivation of the dissipation w.r.t. to the strains
    // (dD_mech/dstrain)
    // = N_T^T . (- dDmech_kin/ dstrain + dDmech_iso/ dstrain )
    // = - N_T^T . (d [ (sigma_{d,T} - beta) . strain^p' ]/ dstrain)
    //   + N_T^T . (d [ kappa(strainbar^p) . strainbar^p' ]/ dstrain)

    // ---------------------- linearisation of KINEMATIC hardening for k_Td

    // (dD_mech_kin/dstrain) = (d [ (sigma_{d,T} - beta) . strain^p' ]/ dstrain)
    //
    // d[ (sigma_{d,T} - beta) . strain^p' ]/ dstrain
    // = d(sigma_{d,T} - beta)/dstrain . strain^p'
    //   + (sigma_{d,T} - beta) . (dstrain^p')/ dstrain)
    //
    // sigma_T is independent of deformation, i.e. strains: dsigma_T/dstrain = 0
    //
    // = d(sigma_d - beta)/dstrain . strain^p'
    //   + (sigma_{d,T} - beta) . [(dstrain^p')/ dstrain]
    //
    // thermal contribution can be neglected because [(vol : dev) == 0]
    // sigma_T: vol, plasticity: deviatoric!!
    // (dDthr/dstrain) = sigma_T : (dstrain^p'/dstrain) == 0,

    // --------- calculate (sigma_{d,T} - beta) . [(dstrain^p')/ dstrain]

    // ---------------------------------calculate [(dstrain^p')/ dstrain]
    // strain^p_{n+1}' = (strain^p_{n+1}-strain^p_n)/Dt = Dgamma/Dt N_n+1
    // strain^p_{n+1} = strain^p_n + Dgamma N_n+1
    //
    // [(dstrain^p')/ dstrain] = 1/Dt (dstrain^p_{n+1}/dstrain)
    //                         = 1/Dt (dDgamma/dstrain) \otimes N_{n+1} + Dgamma .
    //                         (dN_{n+1}/dstrain)

    // (dDgamma/dstrain^{trial}_{n+1}) \otimes N_{n+1}
    // = 2G/(3G + Hiso + Hkin) N_{n+1} \otimes N_{n+1}

    // (dN_{n+1}/dstrain) = 2G / || eta || [sqrt{3/2} I_d - N_{n+1} \otimes N_{n+1}]

    // ----------------------------------------linearisation of Dmech_iso
    // (dD_mech/dstrain) += N_T^T . Hiso . (d [ strainbar^p . strainbar^p' ]/ dstrain)
    Core::LinAlg::Matrix<6, 1> Dmech_d(Core::LinAlg::Initialization::uninitialized);
    Dmech_d.update(thrpllinelast->dissipation_linearised_for_coupl_cond(iquad));
    Core::LinAlg::Matrix<1, nsd_ * nen_ * numdofpernode_> DBop(
        Core::LinAlg::Initialization::uninitialized);
    DBop.multiply_tn(Dmech_d, boplin);

    // coupling stiffness matrix
    if (etangcoupl != nullptr)
    {
      // k_Td^e += timefac . N_T^T . 1/Dt . Dmech_d . B_L . detJ . w(gp)
      // with C_T = m . I
      // (8x24) = (8x1) . (1x24)
      etangcoupl->multiply_nn((fac_ * timefac / stepsize), funct_, DBop, 1.0);
    }  // (etangcoupl != nullptr)

  }  //---------------------------------- end loop over Gauss Points
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::nonlinear_dissipation_fint_tang(
    const Core::Elements::Element* ele,  // the element whose matrix is calculated
    const std::vector<double>& disp,     // current displacements
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>*
        econd,                                              // conductivity matrix
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1>* efint,  // internal force
    Teuchos::ParameterList& params)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // update element geometry
  Core::LinAlg::Matrix<nen_, nsd_> xrefe;  // material coord. of element
  Core::LinAlg::Matrix<nen_, nsd_> xcurr;  // current  coord. of element

  // now get current element displacements and velocities
  auto nodes = ele->nodes();
  for (int i = 0; i < nen_; ++i)
  {
    const auto& x = nodes[i]->x();
    // (8x3) = (nen_xnsd_)
    for (int j = 0; j < nsd_; ++j)
    {
      xrefe(i, j) = x[j];
      xcurr(i, j) = x[j] + disp[i * nsd_ + j];
    }
  }

  // --------------------------------------------------------------- initialise
  // thermal material tangent
  Core::LinAlg::Matrix<6, 1> ctemp(Core::LinAlg::Initialization::zero);

  // ------------------------------------------------------ structural material
  std::shared_ptr<Core::Mat::Material> structmat = get_str_material(ele);

  if (structmat->material_type() != Core::Materials::m_thermoplhyperelast)
  {
    FOUR_C_THROW("So far dissipation only for ThermoPlasticHyperElast material!");
  }
  std::shared_ptr<Mat::ThermoPlasticHyperElast> thermoplhyperelast =
      std::dynamic_pointer_cast<Mat::ThermoPlasticHyperElast>(structmat);
  // true: error if cast fails

  // --------------------------------------------------------- time integration
  // get step size dt
  const double stepsize = params.get<double>("delta time");

  // ----------------------------------------- integration loop for one element

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // initialise the deformation gradient w.r.t. material configuration
  Core::LinAlg::Matrix<nsd_, nsd_> defgrd(Core::LinAlg::Initialization::uninitialized);

  // --------------------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // compute inverse Jacobian matrix and derivatives at GP w.r.t. material
    // coordinates
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // ------------------------------------------------------------ dissipation
    // plastic contribution thermoplastichyperelastic material

    // mechanical Dissipation
    // Dmech := sqrt(2/3) . sigma_y(T_{n+1}) . Dgamma/Dt
    // with MechDiss := sqrt(2/3) . sigma_y(T_{n+1}) . Dgamma
    const double Dmech = thermoplhyperelast->mech_diss(iquad) / stepsize;

    // update/integrate internal force vector (coupling fraction towards displacements)
    if (efint != nullptr)
    {
      // update of the internal "force" vector
      // fint += - N_T^T . Dmech/Dt . detJ . w(gp)
      efint->update((-fac_ * Dmech), funct_, 1.0);
    }

    if (econd != nullptr)
    {
      // Contribution of dissipation to cond matrix
      // econd += - N_T^T . dDmech_dT/Dt . N_T
      econd->multiply_nt(
          (-fac_ * thermoplhyperelast->mech_diss_k_tt(iquad) / stepsize), funct_, funct_, 1.0);
    }

  }  // ---------------------------------- end loop over Gauss Points
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::nonlinear_dissipation_coupled_tang(
    const Core::Elements::Element* ele,  // the element whose matrix is calculated
    const std::vector<double>& disp,     //!< current displacements
    const std::vector<double>& vel,      //!< current velocities
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nsd_ * nen_ * numdofpernode_>* etangcoupl,  // k_Td
    Teuchos::ParameterList& params)
{
  // update element geometry
  Core::LinAlg::Matrix<nen_, nsd_> xcurr;      // current  coord. of element
  Core::LinAlg::Matrix<nen_, nsd_> xcurrrate;  // current  velocity of element

  initial_and_current_nodal_position_velocity(ele, disp, vel, xcurr, xcurrrate);

  // build the deformation gradient w.r.t. material configuration
  Core::LinAlg::Matrix<nsd_, nsd_> defgrd(Core::LinAlg::Initialization::uninitialized);
  // inverse of deformation gradient
  Core::LinAlg::Matrix<nsd_, nsd_> invdefgrd(Core::LinAlg::Initialization::uninitialized);

  // ------------------------------------------------ structural material
  std::shared_ptr<Core::Mat::Material> structmat = get_str_material(ele);
  std::shared_ptr<Mat::ThermoPlasticHyperElast> thermoplhyperelast =
      std::dynamic_pointer_cast<Mat::ThermoPlasticHyperElast>(structmat);
  // true: error if cast fails

  // --------------------------------------------------- time integration
  // get step size dt
  const double stepsize = params.get<double>("delta time");

  // check the time integrator and add correct time factor
  const auto timint =
      params.get<Thermo::DynamicType>("time integrator", Thermo::DynamicType::Undefined);
  // initialise time_fac of velocity discretisation w.r.t. displacements
  double timefac = 0.0;
  switch (timint)
  {
    case Thermo::DynamicType::Statics:
    {
      // evolution equation of plastic material use implicit Euler
      // put str_timefac = 1.0
      timefac = 1.0;
      break;
    }
    case Thermo::DynamicType::OneStepTheta:
    {
      // k_Td = theta . k_Td^e . timefac_Dgamma = theta . k_Td / Dt
      double theta = params.get<double>("theta");
      timefac = theta;
      break;
    }
    case Thermo::DynamicType::GenAlpha:
    {
      // k_Td = alphaf . k_Td^e . timefac_Dgamma = alphaf . k_Td / Dt
      double alphaf = params.get<double>("alphaf");
      timefac = alphaf;
      break;
    }
    case Thermo::DynamicType::Undefined:
    default:
    {
      FOUR_C_THROW("Add correct temporal coefficient here!");
      break;
    }
  }  // end of switch(timint)

  // ----------------------------------------- integration loop for one element
  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // --------------------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // compute inverse Jacobian matrix and derivatives at GP w.r.t. material
    // coordinates
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // (material) deformation gradient F
    // F = d xcurr / d xrefe = xcurr^T . N_XYZ^T
    defgrd.multiply_tt(xcurr, derxy_);

    // calculate the nonlinear B-operator
    Core::LinAlg::Matrix<6, nsd_ * nen_ * numdofpernode_> bop(
        Core::LinAlg::Initialization::uninitialized);
    calculate_bop(&bop, &defgrd, &derxy_);

    // ----------------------------------------------- linearisation of Dmech_d
    // k_Td += - timefac . N_T^T . 1/Dt . mechdiss_kTd . dE/dd
    Core::LinAlg::Matrix<6, 1> dDmech_dE(Core::LinAlg::Initialization::uninitialized);
    dDmech_dE.update(thermoplhyperelast->mech_diss_k_td(iquad));
    Core::LinAlg::Matrix<1, nsd_ * nen_ * numdofpernode_> dDmech_dd(
        Core::LinAlg::Initialization::uninitialized);
    dDmech_dd.multiply_tn(dDmech_dE, bop);

    // coupling stiffness matrix
    if (etangcoupl != nullptr)
    {
      // k_Td^e += - timefac . N_T^T . 1/Dt . dDmech_dE . B . detJ . w(gp)
      // (8x24)  = (8x1) .        (1x6)  (6x24)
      etangcoupl->multiply_nn(-fac_ * timefac / stepsize, funct_, dDmech_dd, 1.0);
    }  // (etangcoupl != nullptr)

  }  //--------------------------------------------- end loop over Gauss Points
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::linear_heatflux_tempgrad(
    const Core::Elements::Element* ele,
    Core::LinAlg::Matrix<nquad_, nsd_>* eheatflux,  // heat fluxes at Gauss points
    Core::LinAlg::Matrix<nquad_, nsd_>* etempgrad   // temperature gradients at Gauss points
)
{
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // ----------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // gradient of current temperature value
    // grad T = d T_j / d x_i = L . N . T = B_ij T_j
    gradtemp_.multiply_nn(derxy_, etempn_);

    // store the temperature gradient for postprocessing
    if (etempgrad != nullptr)
      for (int idim = 0; idim < nsd_; ++idim)
        // (8x3)                    (3x1)
        (*etempgrad)(iquad, idim) = gradtemp_(idim);

    // call material law => cmat_,heatflux_
    // negative q is used for balance equation: -q = -(-k gradtemp)= k * gradtemp
    materialize(ele, iquad);

    // store the heat flux for postprocessing
    if (eheatflux != nullptr)
      // negative sign for heat flux introduced here
      for (int idim = 0; idim < nsd_; ++idim) (*eheatflux)(iquad, idim) = -heatflux_(idim);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::nonlinear_heatflux_tempgrad(
    const Core::Elements::Element* ele,             // the element whose matrix is calculated
    const std::vector<double>& disp,                // current displacements
    const std::vector<double>& vel,                 // current velocities
    Core::LinAlg::Matrix<nquad_, nsd_>* eheatflux,  // heat fluxes at Gauss points
    Core::LinAlg::Matrix<nquad_, nsd_>* etempgrad,  // temperature gradients at Gauss points
    Teuchos::ParameterList& params)
{
  // specific choice of heat flux / temperature gradient
  const auto ioheatflux = params.get<Thermo::HeatFluxType>("ioheatflux", Thermo::heatflux_none);
  const auto iotempgrad = params.get<Thermo::TempGradType>("iotempgrad", Thermo::tempgrad_none);

  // update element geometry
  Core::LinAlg::Matrix<nen_, nsd_> xcurr;      // current  coord. of element
  Core::LinAlg::Matrix<nen_, nsd_> xcurrrate;  // current  coord. of element
  initial_and_current_nodal_position_velocity(ele, disp, vel, xcurr, xcurrrate);

  // build the deformation gradient w.r.t. material configuration
  Core::LinAlg::Matrix<nsd_, nsd_> defgrd(Core::LinAlg::Initialization::uninitialized);
  // inverse of deformation gradient
  Core::LinAlg::Matrix<nsd_, nsd_> invdefgrd(Core::LinAlg::Initialization::uninitialized);

  // ----------------------------------- integration loop for one element
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // --------------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // compute inverse Jacobian matrix and derivatives at GP w.r.t. material
    // coordinates
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    gradtemp_.multiply_nn(derxy_, etempn_);

    // ---------------------------------------- call thermal material law
    // call material law => cmat_,heatflux_ and dercmat_
    // negative q is used for balance equation:
    // heatflux_ = k_0 . Grad T
    materialize(ele, iquad);
    // heatflux_ := qintermediate = k_0 . Grad T

    // -------------------------------------------- coupling to mechanics
    // (material) deformation gradient F
    // F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
    defgrd.multiply_tt(xcurr, derxy_);
    // inverse of deformation gradient
    invdefgrd.invert(defgrd);

    Core::LinAlg::Matrix<nsd_, nsd_> Cinv(Core::LinAlg::Initialization::uninitialized);
    // build the inverse of the right Cauchy-Green deformation gradient C^{-1}
    // C^{-1} = F^{-1} . F^{-T}
    Cinv.multiply_nt(invdefgrd, invdefgrd);

    switch (iotempgrad)
    {
      case Thermo::tempgrad_initial:
      {
        if (etempgrad == nullptr) FOUR_C_THROW("tempgrad data not available");
        // etempgrad = Grad T
        for (int idim = 0; idim < nsd_; ++idim) (*etempgrad)(iquad, idim) = gradtemp_(idim);
        break;
      }
      case Thermo::tempgrad_current:
      {
        if (etempgrad == nullptr) FOUR_C_THROW("tempgrad data not available");
        // etempgrad = grad T = Grad T . F^{-1} =  F^{-T} . Grad T
        // (8x3)        (3x1)   (3x1)    (3x3)     (3x3)    (3x1)
        // spatial temperature gradient
        Core::LinAlg::Matrix<nsd_, 1> currentgradT(Core::LinAlg::Initialization::uninitialized);
        currentgradT.multiply_tn(invdefgrd, gradtemp_);
        for (int idim = 0; idim < nsd_; ++idim) (*etempgrad)(iquad, idim) = currentgradT(idim);
        break;
      }
      case Thermo::tempgrad_none:
      {
        // no postprocessing of temperature gradients
        break;
      }
      default:
        FOUR_C_THROW("requested tempgrad type not available");
        break;
    }  // iotempgrad

    switch (ioheatflux)
    {
      case Thermo::heatflux_initial:
      {
        if (eheatflux == nullptr) FOUR_C_THROW("heat flux data not available");
        Core::LinAlg::Matrix<nsd_, 1> initialheatflux(Core::LinAlg::Initialization::uninitialized);
        // eheatflux := Q = -k_0 . Cinv . Grad T
        initialheatflux.multiply(Cinv, heatflux_);
        for (int idim = 0; idim < nsd_; ++idim) (*eheatflux)(iquad, idim) = -initialheatflux(idim);
        break;
      }
      case Thermo::heatflux_current:
      {
        if (eheatflux == nullptr) FOUR_C_THROW("heat flux data not available");
        // eheatflux := q = - k_0 . 1/(detF) . F^{-T} . Grad T
        // (8x3)     (3x1)            (3x3)  (3x1)
        const double detF = defgrd.determinant();
        Core::LinAlg::Matrix<nsd_, 1> spatialq;
        spatialq.multiply_tn((1.0 / detF), invdefgrd, heatflux_);
        for (int idim = 0; idim < nsd_; ++idim) (*eheatflux)(iquad, idim) = -spatialq(idim);
        break;
      }
      case Thermo::heatflux_none:
      {
        // no postprocessing of heat fluxes, continue!
        break;
      }
      default:
        FOUR_C_THROW("requested heat flux type not available");
        break;
    }  // ioheatflux
  }
}


template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::extract_disp_vel(
    const Core::FE::Discretization& discretization, const Core::Elements::LocationArray& la,
    std::vector<double>& mydisp, std::vector<double>& myvel) const
{
  if ((discretization.has_state(1, "displacement")) and (discretization.has_state(1, "velocity")))
  {
    // get the displacements
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
        discretization.get_state(1, "displacement");
    if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
    // extract the displacements
    mydisp = Core::FE::extract_values(*disp, la[1].lm_);

    // get the velocities
    std::shared_ptr<const Core::LinAlg::Vector<double>> vel =
        discretization.get_state(1, "velocity");
    if (vel == nullptr) FOUR_C_THROW("Cannot get state vectors 'velocity'");
    // extract the displacements
    myvel = Core::FE::extract_values(*vel, la[1].lm_);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::calculate_lump_matrix(
    Core::LinAlg::Matrix<nen_ * numdofpernode_, nen_ * numdofpernode_>* ecapa) const
{
  // lump capacity matrix
  if (ecapa != nullptr)
  {
    // we assume #elemat2 is a square matrix
    for (unsigned int c = 0; c < (*ecapa).n(); ++c)  // parse columns
    {
      double d = 0.0;
      for (unsigned int r = 0; r < (*ecapa).m(); ++r)  // parse rows
      {
        d += (*ecapa)(r, c);  // accumulate row entries
        (*ecapa)(r, c) = 0.0;
      }
      (*ecapa)(c, c) = d;  // apply sum of row entries on diagonal
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::radiation(
    const Core::Elements::Element* ele, const double time)
{
  std::vector<const Core::Conditions::Condition*> myneumcond;

  // check whether all nodes have a unique VolumeNeumann condition
  switch (nsd_)
  {
    case 3:
      Core::Conditions::find_element_conditions(ele, "VolumeNeumann", myneumcond);
      break;
    case 2:
      Core::Conditions::find_element_conditions(ele, "SurfaceNeumann", myneumcond);
      break;
    case 1:
      Core::Conditions::find_element_conditions(ele, "LineNeumann", myneumcond);
      break;
    default:
      FOUR_C_THROW("Illegal number of space dimensions: {}", nsd_);
      break;
  }

  if (myneumcond.size() > 1) FOUR_C_THROW("more than one VolumeNeumann cond on one node");

  if (myneumcond.size() == 1)
  {
    // get node coordinates
    Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
        ele, xyze_);

    // update element geometry
    Core::LinAlg::Matrix<nen_, nsd_> xrefe;  // material coord. of element
    auto nodes = ele->nodes();
    for (int i = 0; i < nen_; ++i)
    {
      const auto& x = nodes[i]->x();
      // (8x3) = (nen_xnsd_)
      for (int j = 0; j < nsd_; j++) xrefe(i, j) = x[j];
    }


    // integrations points and weights
    Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
    if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

    radiation_.clear();

    // compute the Jacobian matrix
    Core::LinAlg::Matrix<nsd_, nsd_> jac;
    jac.multiply(derxy_, xrefe);

    // compute determinant of Jacobian
    const double detJ = jac.determinant();
    if (detJ == 0.0)
      FOUR_C_THROW("ZERO JACOBIAN DETERMINANT");
    else if (detJ < 0.0)
      FOUR_C_THROW("NEGATIVE JACOBIAN DETERMINANT");

    const auto funct = myneumcond[0]->parameters().get<std::vector<std::optional<int>>>("FUNCT");

    Core::LinAlg::Matrix<nsd_, 1> xrefegp(Core::LinAlg::Initialization::uninitialized);
    // material/reference co-ordinates of Gauss point
    for (int dim = 0; dim < nsd_; dim++)
    {
      xrefegp(dim) = 0.0;
      for (int nodid = 0; nodid < nen_; ++nodid) xrefegp(dim) += funct_(nodid) * xrefe(nodid, dim);
    }

    // function evaluation
    FOUR_C_ASSERT(funct.size() == 1, "Need exactly one function.");

    double functfac = 1.0;
    if (funct[0].has_value() && funct[0].value() > 0)
      // evaluate function at current gauss point (3D position vector required!)
      functfac = Global::Problem::instance()
                     ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funct[0].value())
                     .evaluate(xrefegp.as_span(), time, 0);

    // get values and switches from the condition
    const auto onoff = myneumcond[0]->parameters().get<std::vector<int>>("ONOFF");
    const auto val = myneumcond[0]->parameters().get<std::vector<double>>("VAL");

    // set this condition to the radiation array
    for (int idof = 0; idof < numdofpernode_; idof++)
    {
      radiation_(idof) = onoff[idof] * val[idof] * functfac;
    }
  }
  else
  {
    radiation_.clear();
  }
}


template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::materialize(
    const Core::Elements::Element* ele, const int gp)
{
  auto material = ele->material();

  // calculate the current temperature at the integration point
  Core::LinAlg::Matrix<1, 1> temp;
  temp.multiply_tn(1.0, funct_, etempn_, 0.0);

  auto thermoMaterial = std::dynamic_pointer_cast<Mat::Trait::Thermo>(material);
  thermoMaterial->reinit(temp(0), gp);
  thermoMaterial->evaluate(gradtemp_, cmat_, heatflux_, ele->id());
  capacoeff_ = thermoMaterial->capacity();
  thermoMaterial->conductivity_deriv_t(dercmat_);
  dercapa_ = thermoMaterial->capacity_deriv_t();
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::eval_shape_func_and_derivs_at_int_point(
    const Core::FE::IntPointsAndWeights<nsd_>& intpoints,  // integration points
    const int iquad,                                       // id of current Gauss point
    const int eleid                                        // the element id
)
{
  // coordinates of the current (Gauss) integration point (xsi_)
  const double* gpcoord = (intpoints.ip().qxg)[iquad];
  for (int idim = 0; idim < nsd_; idim++)
  {
    xsi_(idim) = gpcoord[idim];
  }

  // shape functions (funct_) and their first derivatives (deriv_)
  // N, N_{,xsi}
  if (myknots_.size() == 0)
  {
    Core::FE::shape_function<distype>(xsi_, funct_);
    Core::FE::shape_function_deriv1<distype>(xsi_, deriv_);
  }
  else
    Core::FE::Nurbs::nurbs_get_3d_funct_deriv(funct_, deriv_, xsi_, myknots_, weights_, distype);

  // compute Jacobian matrix and determinant (as presented in FE lecture notes)
  // actually compute its transpose (compared to J in NiliFEM lecture notes)
  // J = dN/dxsi . x^{-}
  /*
   *   J-NiliFEM               J-FE
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

  // derivatives at gp w.r.t. material coordinates (N_XYZ in solid)
  xjm_.multiply_nt(deriv_, xyze_);
  // xij_ = J^{-T}
  // det = J^{-T} *
  // J = (N_rst * X)^T (6.24 NiliFEM)
  const double det = xij_.invert(xjm_);

  if (det < 1e-16)
    FOUR_C_THROW("GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", eleid, det);

  // set integration factor: fac = Gauss weight * det(J)
  fac_ = intpoints.ip().qwgt[iquad] * det;

  // compute global derivatives
  derxy_.multiply(xij_, deriv_);
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::initial_and_current_nodal_position_velocity(
    const Core::Elements::Element* ele, const std::vector<double>& disp,
    const std::vector<double>& vel, Core::LinAlg::Matrix<nen_, nsd_>& xcurr,
    Core::LinAlg::Matrix<nen_, nsd_>& xcurrrate)
{
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);
  for (int i = 0; i < nen_; ++i)
  {
    for (int j = 0; j < nsd_; ++j)
    {
      xcurr(i, j) = xyze_(j, i) + disp[i * nsd_ + j];
      xcurrrate(i, j) = vel[i * nsd_ + j];
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::prepare_nurbs_eval(
    const Core::Elements::Element* ele,             // the element whose matrix is calculated
    const Core::FE::Discretization& discretization  // current discretisation
)
{
  if (ele->shape() != Core::FE::CellType::nurbs27)
  {
    myknots_.resize(0);
    return;
  }

  myknots_.resize(3);  // fixme: dimension
                       // get nurbs specific infos
  // cast to nurbs discretization
  const auto* nurbsdis =
      dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(&(discretization));
  if (nurbsdis == nullptr) FOUR_C_THROW("So_nurbs27 appeared in non-nurbs discretisation\n");

  // zero-sized element
  if ((*((*nurbsdis).get_knot_vector())).get_ele_knots(myknots_, ele->id())) return;

  // get weights from cp's
  for (int inode = 0; inode < nen_; inode++)
    weights_(inode) = dynamic_cast<const Core::FE::Nurbs::ControlPoint*>(ele->nodes()[inode])->w();
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::integrate_shape_functions(
    const Core::Elements::Element* ele, Core::LinAlg::SerialDenseVector& elevec1,
    const Core::LinAlg::IntSerialDenseVector& dofids)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int gpid = 0; gpid < intpoints.ip().nquad; gpid++)
  {
    eval_shape_func_and_derivs_at_int_point(intpoints, gpid, ele->id());

    // compute integral of shape functions (only for dofid)
    for (int k = 0; k < numdofpernode_; k++)
    {
      if (dofids[k] >= 0)
      {
        for (int node = 0; node < nen_; node++)
        {
          elevec1[node * numdofpernode_ + k] += funct_(node) * fac_;
        }
      }
    }
  }  // loop over integration points

}  // TemperImpl<distype>::integrate_shape_function


template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::extrapolate_from_gauss_points_to_nodes(
    const Core::Elements::Element* ele,  // the element whose matrix is calculated
    const Core::LinAlg::Matrix<nquad_, nsd_>& gpheatflux,
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1>& efluxx,
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1>& efluxy,
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1>& efluxz)
{
  // this quick'n'dirty hack functions only for elements which has the same
  // number of gauss points AND same number of nodes
  if (not((distype == Core::FE::CellType::hex8) or (distype == Core::FE::CellType::hex27) or
          (distype == Core::FE::CellType::tet4) or (distype == Core::FE::CellType::quad4) or
          (distype == Core::FE::CellType::line2)))
    FOUR_C_THROW("Sorry, not implemented for element shape");

  // another check
  if (nen_ * numdofpernode_ != nquad_)
    FOUR_C_THROW("Works only if number of gauss points and nodes match");

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  if (intpoints.ip().nquad != nquad_) FOUR_C_THROW("Trouble with number of Gauss points");

  // build matrix of shape functions at Gauss points
  Core::LinAlg::Matrix<nquad_, nquad_> shpfctatgps;
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // coordinates of the current integration point
    const double* gpcoord = (intpoints.ip().qxg)[iquad];
    for (int idim = 0; idim < nsd_; idim++) xsi_(idim) = gpcoord[idim];

    // shape functions and their first derivatives
    Core::FE::shape_function<distype>(xsi_, funct_);

    for (int inode = 0; inode < nen_; ++inode) shpfctatgps(iquad, inode) = funct_(inode);
  }

  // extrapolation
  Core::LinAlg::Matrix<nquad_, nsd_> ndheatflux;  //  objective nodal heatflux
  Core::LinAlg::Matrix<nquad_, nsd_> gpheatflux2(
      gpheatflux);  // copy the heatflux at the Gauss point
  {
    Core::LinAlg::FixedSizeSerialDenseSolver<nquad_, nquad_, nsd_> solver;  // must be quadratic
    solver.set_matrix(shpfctatgps);
    solver.set_vectors(ndheatflux, gpheatflux2);
    solver.solve();
  }

  // copy into component vectors
  for (int idof = 0; idof < nen_ * numdofpernode_; ++idof)
  {
    efluxx(idof) = ndheatflux(idof, 0);
    if (nsd_ > 1) efluxy(idof) = ndheatflux(idof, 1);
    if (nsd_ > 2) efluxz(idof) = ndheatflux(idof, 2);
  }
}

template <Core::FE::CellType distype>
double Discret::Elements::TemperImpl<distype>::calculate_char_ele_length() const
{
  // volume of the element (2D: element surface area; 1D: element length)
  // (Integration of f(x) = 1 gives exactly the volume/surface/length of element)
  const double vol = fac_;

  // as shown in calc_char_ele_length() in ScaTraImpl
  // c) cubic/square root of element volume/area or element length (3-/2-/1-D)
  // cast dimension to a double variable -> pow()

  // get characteristic element length as cubic root of element volume
  // (2D: square root of element area, 1D: element length)
  // h = vol^(1/dim)
  double h = std::pow(vol, (1.0 / nsd_));

  return h;
}


template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::calculate_boplin(
    Core::LinAlg::Matrix<6, nsd_ * nen_ * numdofpernode_>* boplin,
    const Core::LinAlg::Matrix<nsd_, nen_>* N_XYZ) const
{
  // in thermo element derxy_ == N_XYZ in structural element (i.e. So3_Thermo)
  // lump mass matrix
  if (boplin != nullptr)
  {
    // linear B-operator B_L = N_XYZ
    // disperse global derivatives to bop-lines
    // boplin is arranged as usual (refer to script FE or elsewhere):
    // [ N1,X  0  0  | N2,X  0  0  | ... | Ni,X  0  0  ]
    // [ 0  N1,Y  0  | 0  N2,Y  0  | ... | 0  Ni,Y  0  ]
    // [ 0  0  N1,Z  | 0  0  N2,Z  | ... | 0  0  Ni,Z  ]
    // [ N1,Y N1,X 0 | N2,Y N2,X 0 | ... | Ni,Y Ni,X 0 ]
    // [ 0 N1,Z N1,Y | 0 N2,Z N2,Y | ... | 0 Ni,Z Ni,Y ]
    // [ N1,Z 0 N1,X | N2,Z 0 N2,X | ... | Ni,Z 0 Ni,X ]
    for (int i = 0; i < nen_; ++i)
    {
      (*boplin)(0, nsd_* numdofpernode_* i + 0) = (*N_XYZ)(0, i);
      (*boplin)(0, nsd_* numdofpernode_* i + 1) = 0.0;
      (*boplin)(0, nsd_* numdofpernode_* i + 2) = 0.0;
      (*boplin)(1, nsd_* numdofpernode_* i + 0) = 0.0;
      (*boplin)(1, nsd_* numdofpernode_* i + 1) = (*N_XYZ)(1, i);
      (*boplin)(1, nsd_* numdofpernode_* i + 2) = 0.0;
      (*boplin)(2, nsd_* numdofpernode_* i + 0) = 0.0;
      (*boplin)(2, nsd_* numdofpernode_* i + 1) = 0.0;
      (*boplin)(2, nsd_* numdofpernode_* i + 2) = (*N_XYZ)(2, i);
      /* ~~~ */
      (*boplin)(3, nsd_* numdofpernode_* i + 0) = (*N_XYZ)(1, i);
      (*boplin)(3, nsd_* numdofpernode_* i + 1) = (*N_XYZ)(0, i);
      (*boplin)(3, nsd_* numdofpernode_* i + 2) = 0.0;
      (*boplin)(4, nsd_* numdofpernode_* i + 0) = 0.0;
      (*boplin)(4, nsd_* numdofpernode_* i + 1) = (*N_XYZ)(2, i);
      (*boplin)(4, nsd_* numdofpernode_* i + 2) = (*N_XYZ)(1, i);
      (*boplin)(5, nsd_* numdofpernode_* i + 0) = (*N_XYZ)(2, i);
      (*boplin)(5, nsd_* numdofpernode_* i + 1) = 0.0;
      (*boplin)(5, nsd_* numdofpernode_* i + 2) = (*N_XYZ)(0, i);
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::calculate_bop(
    Core::LinAlg::Matrix<6, nsd_ * nen_ * numdofpernode_>* bop,
    const Core::LinAlg::Matrix<nsd_, nsd_>* defgrd,
    const Core::LinAlg::Matrix<nsd_, nen_>* N_XYZ) const
{
  // lump mass matrix
  if (bop != nullptr)
  {
    /* non-linear B-operator (may so be called, meaning of B-operator is not so
    ** sharp in the non-linear realm) *
    ** B = F . B_L *
    ** with linear B-operator B_L =  N_XYZ (6x24) = (3x8)
    **
    **   B    =   F  . N_XYZ
    ** (6x24)   (3x3) (3x8)
    **
    **      [ ... | F_11*N_{,1}^k  F_21*N_{,1}^k  F_31*N_{,1}^k | ... ]
    **      [ ... | F_12*N_{,2}^k  F_22*N_{,2}^k  F_32*N_{,2}^k | ... ]
    **      [ ... | F_13*N_{,3}^k  F_23*N_{,3}^k  F_33*N_{,3}^k | ... ]
    ** B =  [ ~~~   ~~~~~~~~~~~~~  ~~~~~~~~~~~~~  ~~~~~~~~~~~~~   ~~~ ]
    **      [       F_11*N_{,2}^k+F_12*N_{,1}^k                       ]
    **      [ ... |          F_21*N_{,2}^k+F_22*N_{,1}^k        | ... ]
    **      [                       F_31*N_{,2}^k+F_32*N_{,1}^k       ]
    **      [                                                         ]
    **      [       F_12*N_{,3}^k+F_13*N_{,2}^k                       ]
    **      [ ... |          F_22*N_{,3}^k+F_23*N_{,2}^k        | ... ]
    **      [                       F_32*N_{,3}^k+F_33*N_{,2}^k       ]
    **      [                                                         ]
    **      [       F_13*N_{,1}^k+F_11*N_{,3}^k                       ]
    **      [ ... |          F_23*N_{,1}^k+F_21*N_{,3}^k        | ... ]
    **      [                       F_33*N_{,1}^k+F_31*N_{,3}^k       ]
    */
    for (int i = 0; i < nen_; ++i)
    {
      (*bop)(0, nsd_* numdofpernode_* i + 0) = (*defgrd)(0, 0) * (*N_XYZ)(0, i);
      (*bop)(0, nsd_* numdofpernode_* i + 1) = (*defgrd)(1, 0) * (*N_XYZ)(0, i);
      (*bop)(0, nsd_* numdofpernode_* i + 2) = (*defgrd)(2, 0) * (*N_XYZ)(0, i);
      (*bop)(1, nsd_* numdofpernode_* i + 0) = (*defgrd)(0, 1) * (*N_XYZ)(1, i);
      (*bop)(1, nsd_* numdofpernode_* i + 1) = (*defgrd)(1, 1) * (*N_XYZ)(1, i);
      (*bop)(1, nsd_* numdofpernode_* i + 2) = (*defgrd)(2, 1) * (*N_XYZ)(1, i);
      (*bop)(2, nsd_* numdofpernode_* i + 0) = (*defgrd)(0, 2) * (*N_XYZ)(2, i);
      (*bop)(2, nsd_* numdofpernode_* i + 1) = (*defgrd)(1, 2) * (*N_XYZ)(2, i);
      (*bop)(2, nsd_* numdofpernode_* i + 2) = (*defgrd)(2, 2) * (*N_XYZ)(2, i);
      /* ~~~ */
      (*bop)(3, nsd_* numdofpernode_* i + 0) =
          (*defgrd)(0, 0) * (*N_XYZ)(1, i) + (*defgrd)(0, 1) * (*N_XYZ)(0, i);
      (*bop)(3, nsd_* numdofpernode_* i + 1) =
          (*defgrd)(1, 0) * (*N_XYZ)(1, i) + (*defgrd)(1, 1) * (*N_XYZ)(0, i);
      (*bop)(3, nsd_* numdofpernode_* i + 2) =
          (*defgrd)(2, 0) * (*N_XYZ)(1, i) + (*defgrd)(2, 1) * (*N_XYZ)(0, i);
      (*bop)(4, nsd_* numdofpernode_* i + 0) =
          (*defgrd)(0, 1) * (*N_XYZ)(2, i) + (*defgrd)(0, 2) * (*N_XYZ)(1, i);
      (*bop)(4, nsd_* numdofpernode_* i + 1) =
          (*defgrd)(1, 1) * (*N_XYZ)(2, i) + (*defgrd)(1, 2) * (*N_XYZ)(1, i);
      (*bop)(4, nsd_* numdofpernode_* i + 2) =
          (*defgrd)(2, 1) * (*N_XYZ)(2, i) + (*defgrd)(2, 2) * (*N_XYZ)(1, i);
      (*bop)(5, nsd_* numdofpernode_* i + 0) =
          (*defgrd)(0, 2) * (*N_XYZ)(0, i) + (*defgrd)(0, 0) * (*N_XYZ)(2, i);
      (*bop)(5, nsd_* numdofpernode_* i + 1) =
          (*defgrd)(1, 2) * (*N_XYZ)(0, i) + (*defgrd)(1, 0) * (*N_XYZ)(2, i);
      (*bop)(5, nsd_* numdofpernode_* i + 2) =
          (*defgrd)(2, 2) * (*N_XYZ)(0, i) + (*defgrd)(2, 0) * (*N_XYZ)(2, i);
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::calculate_linearisation_of_jacobian(
    Core::LinAlg::Matrix<1, nsd_ * nen_ * numdofpernode_>& dJ_dd, const double J,
    const Core::LinAlg::Matrix<nsd_, nen_>& N_XYZ,
    const Core::LinAlg::Matrix<nsd_, nsd_>& defgrd_inv) const
{
  if (nsd_ != 3)
    FOUR_C_THROW("TSI only implemented for fully three dimensions!");
  else
  {
    // ----------------------------------------- build F^{-1} as vector 9x1
    // F != F^T, i.e. Voigt notation (6x1) NOT admissible
    // F (3x3) --> (9x1)
    Core::LinAlg::Matrix<nsd_ * nsd_, 1> defgrd_inv_vec(
        Core::LinAlg::Initialization::uninitialized);
    defgrd_inv_vec(0) = defgrd_inv(0, 0);
    defgrd_inv_vec(1) = defgrd_inv(0, 1);
    defgrd_inv_vec(2) = defgrd_inv(0, 2);
    defgrd_inv_vec(3) = defgrd_inv(1, 0);
    defgrd_inv_vec(4) = defgrd_inv(1, 1);
    defgrd_inv_vec(5) = defgrd_inv(1, 2);
    defgrd_inv_vec(6) = defgrd_inv(2, 0);
    defgrd_inv_vec(7) = defgrd_inv(2, 1);
    defgrd_inv_vec(8) = defgrd_inv(2, 2);

    // ------------------------ build N_X operator (w.r.t. material config)
    Core::LinAlg::Matrix<nsd_ * nsd_, nsd_ * nen_ * numdofpernode_> N_X(
        Core::LinAlg::Initialization::zero);  // set to zero
    for (int i = 0; i < nen_; ++i)
    {
      N_X(0, 3 * i + 0) = N_XYZ(0, i);
      N_X(1, 3 * i + 1) = N_XYZ(0, i);
      N_X(2, 3 * i + 2) = N_XYZ(0, i);

      N_X(3, 3 * i + 0) = N_XYZ(1, i);
      N_X(4, 3 * i + 1) = N_XYZ(1, i);
      N_X(5, 3 * i + 2) = N_XYZ(1, i);

      N_X(6, 3 * i + 0) = N_XYZ(2, i);
      N_X(7, 3 * i + 1) = N_XYZ(2, i);
      N_X(8, 3 * i + 2) = N_XYZ(2, i);
    }

    // ------linearisation of Jacobi determinant detF = J w.r.t. displacements
    // dJ/dd = dJ/dF : dF/dd = J . F^{-T} . N,X  = J . F^{-T} . B_L
    // (1x24)                                          (9x1)   (9x8)
    dJ_dd.multiply_tn(J, defgrd_inv_vec, N_X);

  }  // method only implemented for fully three dimensional analysis
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::calculate_cauchy_greens(
    Core::LinAlg::Matrix<6, 1>& Cratevct,                // (io) C' in vector notation
    Core::LinAlg::Matrix<6, 1>& Cinvvct,                 // (io) C^{-1} in vector notation
    Core::LinAlg::Matrix<nsd_, nsd_>& Cinv,              // (io) C^{-1} in tensor notation
    const Core::LinAlg::Matrix<nsd_, nsd_>* defgrd,      // (i) deformation gradient
    const Core::LinAlg::Matrix<nsd_, nsd_>* defgrdrate,  // (i) rate of deformation gradient
    const Core::LinAlg::Matrix<nsd_, nsd_>* invdefgrd    // (i) inverse of deformation gradient
) const
{
  // calculate the rate of the right Cauchy-Green deformation gradient C'
  // rate of right Cauchy-Green tensor C' = F^T . F' + (F')^T . F
  // C'= F^T . F' + (F')^T . F
  Core::LinAlg::Matrix<nsd_, nsd_> Crate(Core::LinAlg::Initialization::uninitialized);
  Crate.multiply_tn((*defgrd), (*defgrdrate));
  Crate.multiply_tn(1.0, (*defgrdrate), (*defgrd), 1.0);
  // Or alternative use: C' = 2 . (F^T . F') when applied to symmetric tensor

  // copy to matrix notation
  // rate vector Crate C'
  // C' = { C11', C22', C33', C12', C23', C31' }
  if constexpr (nsd_ == 1)
  {
    Cratevct(0) = Crate(0, 0);
  }
  else if constexpr (nsd_ == 2)
  {
    Cratevct(0) = Crate(0, 0);
    Cratevct(1) = Crate(1, 1);
    Cratevct(2) = Crate(0, 1);
  }
  else if constexpr (nsd_ == 3)
  {
    Cratevct(0) = Crate(0, 0);
    Cratevct(1) = Crate(1, 1);
    Cratevct(2) = Crate(2, 2);
    Cratevct(3) = Crate(0, 1);
    Cratevct(4) = Crate(1, 2);
    Cratevct(5) = Crate(2, 0);
  }

  // build the inverse of the right Cauchy-Green deformation gradient C^{-1}
  // C^{-1} = F^{-1} . F^{-T}
  Cinv.multiply_nt((*invdefgrd), (*invdefgrd));
  // Cinvvct: C^{-1} in Voigt-/vector notation
  // C^{-1} = { C11^{-1}, C22^{-1}, C33^{-1}, C12^{-1}, C23^{-1}, C31^{-1} }

  if constexpr (nsd_ == 1)
  {
    Cinvvct(0) = Cinv(0, 0);
  }
  else if constexpr (nsd_ == 2)
  {
    Cinvvct(0) = Cinv(0, 0);
    Cinvvct(1) = Cinv(1, 1);
    Cinvvct(2) = Cinv(0, 1);
  }
  else if constexpr (nsd_ == 3)
  {
    Cinvvct(0) = Cinv(0, 0);
    Cinvvct(1) = Cinv(1, 1);
    Cinvvct(2) = Cinv(2, 2);
    Cinvvct(3) = Cinv(0, 1);
    Cinvvct(4) = Cinv(1, 2);
    Cinvvct(5) = Cinv(2, 0);
  }
}

template <Core::FE::CellType distype>
std::shared_ptr<Core::Mat::Material> Discret::Elements::TemperImpl<distype>::get_str_material(
    const Core::Elements::Element* ele  // the element whose matrix is calculated
) const
{
  std::shared_ptr<Core::Mat::Material> structmat = nullptr;

  // access second material in thermo element
  if (ele->num_material() > 1)
    structmat = ele->material(1);
  else
    FOUR_C_THROW("no second material defined for element {}", ele->id());

  return structmat;
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::compute_error(
    const Core::Elements::Element* ele,  // the element whose matrix is calculated
    Core::LinAlg::Matrix<nen_ * numdofpernode_, 1>& elevec1,
    Teuchos::ParameterList& params  // parameter list
)
{
  // get node coordinates
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      ele, xyze_);

  // get scalar-valued element temperature
  // build the product of the shapefunctions and element temperatures T = N . T
  Core::LinAlg::Matrix<1, 1> NT(Core::LinAlg::Initialization::uninitialized);

  // analytical solution
  Core::LinAlg::Matrix<1, 1> T_analytical(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 1> deltaT(Core::LinAlg::Initialization::zero);
  // ------------------------------- integration loop for one element

  // integrations points and weights
  Core::FE::IntPointsAndWeights<nsd_> intpoints(Thermo::DisTypeToOptGaussRule<distype>::rule);
  //  if (intpoints.ip().nquad != nquad_)
  //    FOUR_C_THROW("Trouble with number of Gauss points");

  const auto calcerr = Teuchos::getIntegralValue<Thermo::CalcError>(params, "calculate error");
  const int errorfunctno = params.get<int>("error function number");
  const double t = params.get<double>("total time");

  // ----------------------------------------- loop over Gauss Points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // compute inverse Jacobian matrix and derivatives
    eval_shape_func_and_derivs_at_int_point(intpoints, iquad, ele->id());

    // ------------------------------------------------ thermal terms

    // gradient of current temperature value
    // grad T = d T_j / d x_i = L . N . T = B_ij T_j
    gradtemp_.multiply_nn(derxy_, etempn_);

    // current element temperatures
    // N_T . T (funct_ defined as <nen,1>)
    NT.multiply_tn(funct_, etempn_);  // (1x8)(8x1)

    // H1 -error norm
    // compute first derivative of the displacement
    Core::LinAlg::Matrix<nsd_, 1> derT(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<nsd_, 1> deltaderT(Core::LinAlg::Initialization::zero);

    // Compute analytical solution
    switch (calcerr)
    {
      case Thermo::calcerror_byfunct:
      {
        // get coordinates at integration point
        // gp reference coordinates
        Core::LinAlg::Matrix<nsd_, 1> xyzint(Core::LinAlg::Initialization::zero);
        xyzint.multiply(xyze_, funct_);

        // function evaluation requires a 3D position vector!!
        double position[3] = {0.0, 0.0, 0.0};

        for (int dim = 0; dim < nsd_; ++dim) position[dim] = xyzint(dim);

        const double T_exact = Global::Problem::instance()
                                   ->function_by_id<Core::Utils::FunctionOfSpaceTime>(errorfunctno)
                                   .evaluate(position, t, 0);

        T_analytical(0, 0) = T_exact;

        std::vector<double> Tder_exact =
            Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(errorfunctno)
                .evaluate_spatial_derivative(position, t, 0);

        if (Tder_exact.size())
        {
          for (int dim = 0; dim < nsd_; ++dim) derT(dim) = Tder_exact[dim];
        }
      }
      break;
      default:
        FOUR_C_THROW("analytical solution is not defined");
        break;
    }

    // compute difference between analytical solution and numerical solution
    deltaT.update(1.0, NT, -1.0, T_analytical);

    // H1 -error norm
    // compute error for first velocity derivative
    deltaderT.update(1.0, gradtemp_, -1.0, derT);

    // 0: delta temperature for L2-error norm
    // 1: delta temperature for H1-error norm
    // 2: analytical temperature for L2 norm
    // 3: analytical temperature for H1 norm

    // the error for the L2 and H1 norms are evaluated at the Gauss point

    // integrate delta velocity for L2-error norm
    elevec1(0) += deltaT(0, 0) * deltaT(0, 0) * fac_;
    // integrate delta velocity for H1-error norm
    elevec1(1) += deltaT(0, 0) * deltaT(0, 0) * fac_;
    // integrate analytical velocity for L2 norm
    elevec1(2) += T_analytical(0, 0) * T_analytical(0, 0) * fac_;
    // integrate analytical velocity for H1 norm
    elevec1(3) += T_analytical(0, 0) * T_analytical(0, 0) * fac_;

    // integrate delta velocity derivative for H1-error norm
    elevec1(1) += deltaderT.dot(deltaderT) * fac_;
    // integrate analytical velocity for H1 norm
    elevec1(3) += derT.dot(derT) * fac_;
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::TemperImpl<distype>::copy_matrix_into_char_vector(
    std::vector<char>& data, const Core::LinAlg::Matrix<nquad_, nsd_>& stuff) const
{
  Core::Communication::PackBuffer tempBuffer;
  add_to_pack(tempBuffer, stuff);
  std::copy(tempBuffer().begin(), tempBuffer().end(), std::back_inserter(data));
}

FOUR_C_NAMESPACE_CLOSE
