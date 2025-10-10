// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_fixedsizematrix_tensor_transformation.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_membrane_elasthyper.hpp"
#include "4C_mat_membrane_material_interfaces.hpp"
#include "4C_material_base.hpp"
#include "4C_membrane.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_function_of_time.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace Internal
{
  namespace
  {
    /*!
     * @brief Get stress like voigt notation stress tensor from 3D stress matrix assuming plane
     * stress in the x-y plane
     *
     * @param stress (in) : Stress in Matrix notation
     * @param planeStressStressLike (out) : Stress in voigt notation assuming plane stress in the
     * x-y plane
     */
    inline void local_plane_stress_to_stress_like_voigt(
        const Core::LinAlg::Matrix<3, 3>& stress, Core::LinAlg::Matrix<3, 1>& planeStressStressLike)
    {
      planeStressStressLike(0) = stress(0, 0);
      planeStressStressLike(1) = stress(1, 1);
      planeStressStressLike(2) = 0.5 * (stress(0, 1) + stress(1, 0));
    }

    /*!
     * @brief Subpart of the linearization assuming plane stress in the x-y plane
     *
     * @param cmat (in) : Full linearization
     * @param cmatred (out) : Reduced linearization assuming plane stress in the x-y plane
     */
    inline void local_fourth_tensor_plane_stress_to_stress_like_voigt(
        const Core::LinAlg::Matrix<6, 6>& cmat, Core::LinAlg::Matrix<3, 3>& cmatred)
    {
      cmatred(0, 0) = cmat(0, 0);
      cmatred(0, 1) = cmat(0, 1);
      cmatred(0, 2) = cmat(0, 3);
      cmatred(1, 0) = cmat(1, 0);
      cmatred(1, 1) = cmat(1, 1);
      cmatred(1, 2) = cmat(1, 3);
      cmatred(2, 0) = cmat(3, 0);
      cmatred(2, 1) = cmat(3, 1);
      cmatred(2, 2) = cmat(3, 3);
    }
  }  // namespace
}  // namespace Internal


/*----------------------------------------------------------------------*
 |  evaluate the element (public)                          fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::Membrane<distype>::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // determine size of each element matrix
  Core::LinAlg::Matrix<numdof_, numdof_> elemat_1(elemat1.values(), true);
  Core::LinAlg::Matrix<numdof_, numdof_> elemat_2(elemat2.values(), true);
  Core::LinAlg::Matrix<numdof_, 1> elevec_1(elevec1.values(), true);
  Core::LinAlg::Matrix<numdof_, 1> elevec_2(elevec2.values(), true);
  Core::LinAlg::Matrix<numdof_, 1> elevec_3(elevec3.values(), true);

  // set params interface pointer
  set_params_interface_ptr(params);

  // start with ActionType none
  Core::Elements::ActionType act = Core::Elements::none;

  if (is_params_interface())  // new structural time integration
  {
    act = params_interface().get_action_type();
  }
  else  // old structural time integration
  {
    // get the action required
    std::string action = params.get<std::string>("action", "none");
    if (action == "none")
      FOUR_C_THROW("No action supplied");
    else if (action == "calc_struct_nlnstiff")
      act = Core::Elements::struct_calc_nlnstiff;
    else if (action == "calc_struct_nlnstiffmass")
      act = Core::Elements::struct_calc_nlnstiffmass;
    else if (action == "calc_struct_update_istep")
      act = Core::Elements::struct_calc_update_istep;
    else if (action == "calc_struct_reset_istep")
      act = Core::Elements::struct_calc_reset_istep;
    else if (action == "calc_struct_stress")
      act = Core::Elements::struct_calc_stress;
    else if (action == "calc_struct_energy")
      act = Core::Elements::struct_calc_energy;
    else if (action == "postprocess_thickness")
      act = Core::Elements::struct_postprocess_thickness;
    else
    {
      FOUR_C_THROW("Unknown type of action for Membrane: {}", action);
    }
  }

  switch (act)
  {
    /*===============================================================================*
     | struct_calc_nlnstiff                                                          |
     *===============================================================================*/
    case Core::Elements::struct_calc_nlnstiff:
    {
      // need current displacement
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);
      Core::LinAlg::Matrix<numdof_, numdof_>* matptr = nullptr;
      if (elemat_1.is_initialized()) matptr = &elemat_1;

      mem_nlnstiffmass(lm, mydisp, matptr, nullptr, &elevec_1, nullptr, nullptr, params,
          Inpar::Solid::stress_none, Inpar::Solid::strain_none);
    }
    break;

    /*===============================================================================*
     | struct_calc_nlnstiffmass                                                      |
     *===============================================================================*/
    case Core::Elements::struct_calc_nlnstiffmass:  // do mass, stiffness and internal forces
    {
      // need current displacement
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);
      Core::LinAlg::Matrix<numdof_, numdof_>* matptr = nullptr;
      if (elemat_1.is_initialized()) matptr = &elemat_1;

      mem_nlnstiffmass(lm, mydisp, matptr, &elemat_2, &elevec_1, nullptr, nullptr, params,
          Inpar::Solid::stress_none, Inpar::Solid::strain_none);
    }
    break;

    /*===============================================================================*
     | struct_calc_internalforce                                                     |
     *===============================================================================*/
    case Core::Elements::struct_calc_internalforce:
    {
      // need current displacement
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

      mem_nlnstiffmass(lm, mydisp, nullptr, nullptr, &elevec_1, nullptr, nullptr, params,
          Inpar::Solid::stress_none, Inpar::Solid::strain_none);
    }
    break;

    /*===============================================================================*
     | struct_calc_update_istep                                                      |
     *===============================================================================*/
    case Core::Elements::struct_calc_update_istep:
    {
      // Update materials
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);
      update_element(mydisp, params, *material());
    }
    break;

    /*===============================================================================*
     | struct_calc_reset_istep                                                       |
     *===============================================================================*/
    case Core::Elements::struct_calc_reset_istep:
    {
      // Reset of history (if needed)
      solid_material()->reset_step();
    }
    break;

    /*===============================================================================*
     | struct_calc_stress                                                            |
     *===============================================================================*/
    case Core::Elements::struct_calc_stress:
    {
      // need current displacement
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

      std::shared_ptr<std::vector<char>> stressdata = nullptr;
      std::shared_ptr<std::vector<char>> straindata = nullptr;

      Inpar::Solid::StressType iostress = Inpar::Solid::stress_none;
      Inpar::Solid::StrainType iostrain = Inpar::Solid::strain_none;

      if (is_params_interface())  // new structural time integration
      {
        stressdata = str_params_interface().stress_data_ptr();
        straindata = str_params_interface().strain_data_ptr();

        iostress = str_params_interface().get_stress_output_type();
        iostrain = str_params_interface().get_strain_output_type();
      }
      else  // old structural time integration
      {
        stressdata = params.get<std::shared_ptr<std::vector<char>>>("stress", nullptr);
        straindata = params.get<std::shared_ptr<std::vector<char>>>("strain", nullptr);

        iostress = params.get<Inpar::Solid::StressType>("iostress", Inpar::Solid::stress_none);
        iostrain = params.get<Inpar::Solid::StrainType>("iostrain", Inpar::Solid::strain_none);
      }

      if (stressdata == nullptr) FOUR_C_THROW("Cannot get 'stress' data");
      if (straindata == nullptr) FOUR_C_THROW("Cannot get 'strain' data");

      Core::LinAlg::Matrix<numgpt_post_, 6> stress;
      Core::LinAlg::Matrix<numgpt_post_, 6> strain;

      // determine strains and/or stresses
      mem_nlnstiffmass(
          lm, mydisp, nullptr, nullptr, nullptr, &stress, &strain, params, iostress, iostrain);

      // add data to pack
      {
        Core::Communication::PackBuffer data;
        Core::LinAlg::SerialDenseMatrix stress_view(
            Teuchos::View, stress.values(), numgpt_post_, numgpt_post_, 6);
        add_to_pack(data, stress_view);
        std::copy(data().begin(), data().end(), std::back_inserter(*stressdata));
      }

      {
        Core::Communication::PackBuffer data;
        Core::LinAlg::SerialDenseMatrix strain_view(
            Teuchos::View, strain.values(), numgpt_post_, numgpt_post_, 6);
        add_to_pack(data, strain_view);
        std::copy(data().begin(), data().end(), std::back_inserter(*straindata));
      }
    }
    break;

    /*===============================================================================*
     | struct_calc_thickness                                                         |
     *===============================================================================*/
    case Core::Elements::struct_calc_thickness:
    {
      // nothing to do for ghost elements
      if (Core::Communication::my_mpi_rank(discretization.get_comm()) == owner())
      {
        auto thickdata = str_params_interface().opt_quantity_data_ptr();

        if (thickdata == nullptr) FOUR_C_THROW("Cannot get 'thickness' data");

        Core::LinAlg::Matrix<numgpt_post_, 1> thickness;
        for (int i = 0; i < numgpt_post_; ++i) thickness(i) = cur_thickness_[i];
        Core::LinAlg::SerialDenseMatrix thickness_view(
            Teuchos::View, thickness.values(), numgpt_post_, numgpt_post_, 1);

        // add data to pack
        {
          Core::Communication::PackBuffer data;
          add_to_pack(data, thickness_view);
          std::copy(data().begin(), data().end(), std::back_inserter(*thickdata));
        }
      }
    }
    break;

    /*===============================================================================*
     | struct_calc_energy                                                            |
     *===============================================================================*/
    case Core::Elements::struct_calc_energy:
    {
      // initialization of internal energy
      double intenergy = 0.0;

      // need current displacement
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

      // get reference configuration and determine current configuration
      Core::LinAlg::Matrix<numnod_, noddof_> xrefe(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<numnod_, noddof_> xcurr(Core::LinAlg::Initialization::zero);

      mem_configuration(mydisp, xrefe, xcurr);

      /*===============================================================================*
       | loop over the gauss points                                                    |
       *===============================================================================*/

      // allocate matrix for shape function derivatives at gp
      Core::LinAlg::Matrix<numdim_, numnod_> derivs(Core::LinAlg::Initialization::zero);

      for (int gp = 0; gp < intpoints_.nquad; ++gp)
      {
        // get gauss points from integration rule
        double xi_gp = intpoints_.qxg[gp][0];
        double eta_gp = intpoints_.qxg[gp][1];

        // get gauss weight at current gp
        double gpweight = intpoints_.qwgt[gp];

        // get shape function derivatives in the plane of the element
        Core::FE::shape_function_2d_deriv1(derivs, xi_gp, eta_gp, shape());

        /*===============================================================================*
         | orthonormal base (t1,t2,tn) in the undeformed configuration at current GP     |
         *===============================================================================*/

        Core::LinAlg::Matrix<numdim_, numnod_> derivs_ortho(Core::LinAlg::Initialization::zero);
        double G1G2_cn;
        Core::LinAlg::Matrix<noddof_, 1> dXds1(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<noddof_, 1> dXds2(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<noddof_, 1> dxds1(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<noddof_, 1> dxds2(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<noddof_, noddof_> Q_localToGlobal(Core::LinAlg::Initialization::zero);

        mem_orthonormalbase(xrefe, xcurr, derivs, derivs_ortho, G1G2_cn, dXds1, dXds2, dxds1, dxds2,
            Q_localToGlobal);

        /*===============================================================================*
         | surface deformation gradient                                                  |
         *===============================================================================*/

        // surface deformation gradient in 3 dimensions in global coordinates
        Core::LinAlg::Matrix<noddof_, noddof_> defgrd_glob(Core::LinAlg::Initialization::zero);

        // surface deformation gradient in 3 dimensions in local coordinates
        Core::LinAlg::Matrix<noddof_, noddof_> defgrd_loc(Core::LinAlg::Initialization::zero);

        // principle stretch in thickness direction
        double lambda3 = 1.0;

        // standard evaluation (incompressible, plane stress)
        if (material()->material_type() == Core::Materials::m_membrane_elasthyper)
        {
          // incompressibility condition to get principle stretch in thickness direction
          lambda3 = std::sqrt(
              1.0 / (dxds1.dot(dxds1) * dxds2.dot(dxds2) - std::pow(dxds1.dot(dxds2), 2.0)));
        }
        else
        {
          FOUR_C_THROW(
              "Type of material not implemented for evaluation of strain energy for membranes!");
        }

        // surface deformation gradient in 3 dimensions in global coordinates
        mem_defgrd_global(dXds1, dXds2, dxds1, dxds2, lambda3, defgrd_glob);

        // surface deformation gradient in 3 dimensions in local coordinates
        Core::LinAlg::FourTensorOperations::inverse_tensor_rotation<3>(
            Q_localToGlobal, defgrd_glob, defgrd_loc);

        /*===============================================================================*
         | right cauchygreen tensor in local coordinates                                 |
         *===============================================================================*/

        // calculate three-dimensional right cauchy-green strain tensor in orthonormal base
        Core::LinAlg::Matrix<noddof_, noddof_> cauchygreen_loc(Core::LinAlg::Initialization::zero);
        cauchygreen_loc.multiply_tn(1.0, defgrd_loc, defgrd_loc, 0.0);

        /*===============================================================================*
         | call material law for evaluation of strain energy                            |
         *===============================================================================*/

        double psi = 0.0;

        // standard evaluation (incompressible, plane stress)
        if (material()->material_type() == Core::Materials::m_membrane_elasthyper)
        {
          std::dynamic_pointer_cast<Mat::MembraneElastHyper>(Core::Elements::Element::material())
              ->strain_energy(cauchygreen_loc, psi, gp, id());
        }
        else
        {
          FOUR_C_THROW(
              "Type of material not implemented for evaluation of strain energy for membranes!");
        }

        // add gauss point contribution to internal energy
        double fac = gpweight * thickness_ * G1G2_cn;
        intenergy += fac * psi;
      }

      if (is_params_interface())  // new structural time integration
      {
        // only add contributions from row elements to avoid counting them on more than one proc
        if (Core::Communication::my_mpi_rank(discretization.get_comm()) == owner())
          str_params_interface().add_contribution_to_energy_type(intenergy, Solid::internal_energy);
      }
      else  // old structural time integration
      {
        // check length of elevec1
        if (elevec1.length() < 1) FOUR_C_THROW("The given result vector is too short.");

        elevec1(0) = intenergy;
      }
    }
    break;

    /*===============================================================================*
     | struct_postprocess_thickness                                                  |
     *===============================================================================*/
    case Core::Elements::struct_postprocess_thickness:
    {
      const std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>
          gpthickmap = params.get<
              std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>>(
              "gpthickmap", nullptr);
      if (gpthickmap == nullptr) FOUR_C_THROW("no gp thickness map available for postprocessing");

      std::string optquantitytype = params.get<std::string>("optquantitytype", "ndxyz");

      int gid = id();
      Core::LinAlg::Matrix<numgpt_post_, 1> gpthick(((*gpthickmap)[gid])->values(), true);

      std::shared_ptr<Core::LinAlg::MultiVector<double>> postthick =
          params.get<std::shared_ptr<Core::LinAlg::MultiVector<double>>>("postthick", nullptr);
      if (postthick == nullptr) FOUR_C_THROW("No element thickness vector available");

      if (optquantitytype == "ndxyz")
      {
        // extrapolation matrix: static because equal for all elements of the same discretization
        // type
        static Core::LinAlg::Matrix<numnod_, numgpt_post_> extrapol(mem_extrapolmat());

        // extrapolate the nodal thickness for current element
        Core::LinAlg::Matrix<numnod_, 1> nodalthickness;
        nodalthickness.multiply(1.0, extrapol, gpthick, 0.0);

        // "assembly" of extrapolated nodal thickness
        for (int i = 0; i < numnod_; ++i)
        {
          int gid = node_ids()[i];
          if (postthick->get_map().my_gid(node_ids()[i]))  // rownode
          {
            int lid = postthick->get_map().lid(gid);
            int myadjele = nodes()[i]->num_element();
            (*postthick)(0).get_values()[lid] += nodalthickness(i) / myadjele;
          }
        }
      }
      else
      {
        FOUR_C_THROW("unknown type of thickness output on element level");
      }
    }
    break;

    /*===============================================================================*
     | struct_calc_recover                                                           |
     *===============================================================================*/
    case Core::Elements::struct_calc_recover:
    {
      // do nothing here
    }
    break;

    case Core::Elements::struct_calc_predict:
    {
      // do nothing here
      break;
    }

    /*===============================================================================*
     | default                                                                       |
     *===============================================================================*/
    default:
      FOUR_C_THROW("Unknown type of action for Membrane: {}", action_type_to_string(act));
      break;
  }

  return 0;
}


/*-----------------------------------------------------------------------*
 |  Integrate a Surface Neumann boundary condition (public) fbraeu 06/16 |
 *-----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::Membrane<distype>::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  // set params interface pointer
  set_params_interface_ptr(params);

  // get values and switches from the condition
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition.parameters().get<std::vector<double>>("VAL");

  // find out whether we will use a time curve
  double time = -1.0;

  if (is_params_interface())  // new structural time integration
    time = params_interface().get_total_time();
  else  // old structural time integration
    time = params.get("total time", -1.0);

  // ensure that at least as many curves/functs as dofs are available
  if (int(onoff.size()) < noddof_)
    FOUR_C_THROW("Fewer functions or curves defined than the element has dofs.");

  // check membrane pressure input
  for (int checkdof = 1; checkdof < int(onoff.size()); ++checkdof)
    if (onoff[checkdof] != 0) FOUR_C_THROW("membrane pressure on 1st dof only!");

  // find out whether we will use time curves and get the factors
  const auto tmp_funct = condition.parameters().get<std::vector<std::optional<int>>>("FUNCT");
  std::vector<double> functfacs(noddof_, 1.0);
  for (int i = 0; i < noddof_; ++i)
  {
    if (tmp_funct[i].has_value() && tmp_funct[i].value() > 0)
    {
      functfacs[i] = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfTime>(tmp_funct[i].value())
                         .evaluate(time);
    }
  }

  // determine current pressure
  double pressure;
  if (onoff[0])
    pressure = val[0] * functfacs[0];
  else
    pressure = 0.0;

  // need displacement new
  std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
      discretization.get_state("displacement new");
  if (disp == nullptr) FOUR_C_THROW("Cannot get state vector 'displacement new'");
  std::vector<double> mydisp = Core::FE::extract_values(*disp, lm);

  // get reference configuration and determine current configuration
  Core::LinAlg::Matrix<numnod_, noddof_> xrefe(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numnod_, noddof_> xcurr(Core::LinAlg::Initialization::zero);

  mem_configuration(mydisp, xrefe, xcurr);

  /*===============================================================================*
   | loop over the gauss points                                                    |
   *===============================================================================*/

  // allocate vector for shape functions and matrix for derivatives at gp
  Core::LinAlg::Matrix<numnod_, 1> shapefcts(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdim_, numnod_> derivs(Core::LinAlg::Initialization::zero);

  for (int gp = 0; gp < intpoints_.nquad; ++gp)
  {
    // get gauss points from integration rule
    double xi_gp = intpoints_.qxg[gp][0];
    double eta_gp = intpoints_.qxg[gp][1];

    // get gauss weight at current gp
    double gpweight = intpoints_.qwgt[gp];

    // get shape functions and derivatives in the plane of the element
    Core::FE::shape_function_2d(shapefcts, xi_gp, eta_gp, shape());
    Core::FE::shape_function_2d_deriv1(derivs, xi_gp, eta_gp, shape());

    /*===============================================================================*
     | orthonormal base (t1,t2,tn) in the undeformed configuration at current GP     |
     *===============================================================================*/

    Core::LinAlg::Matrix<numdim_, numnod_> derivs_ortho(Core::LinAlg::Initialization::zero);
    double G1G2_cn;
    Core::LinAlg::Matrix<noddof_, 1> dXds1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<noddof_, 1> dXds2(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<noddof_, 1> dxds1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<noddof_, 1> dxds2(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<noddof_, noddof_> Q_localToGlobal(Core::LinAlg::Initialization::zero);

    mem_orthonormalbase(
        xrefe, xcurr, derivs, derivs_ortho, G1G2_cn, dXds1, dXds2, dxds1, dxds2, Q_localToGlobal);

    // determine cross product x,1 x x,2
    Core::LinAlg::Matrix<noddof_, 1> xcurr_cross(Core::LinAlg::Initialization::zero);
    xcurr_cross(0) = dxds1(1) * dxds2(2) - dxds1(2) * dxds2(1);
    xcurr_cross(1) = dxds1(2) * dxds2(0) - dxds1(0) * dxds2(2);
    xcurr_cross(2) = dxds1(0) * dxds2(1) - dxds1(1) * dxds2(0);

    // determine cross product X,1 x X,2
    Core::LinAlg::Matrix<noddof_, 1> xrefe_cross(Core::LinAlg::Initialization::zero);
    xrefe_cross(0) = dXds1(1) * dXds2(2) - dXds1(2) * dXds2(1);
    xrefe_cross(1) = dXds1(2) * dXds2(0) - dXds1(0) * dXds2(2);
    xrefe_cross(2) = dXds1(0) * dXds2(1) - dXds1(1) * dXds2(0);

    // euclidean norm of xref_cross
    double xrefe_cn = xrefe_cross.norm2();

    // integration factor
    double fac = (pressure * G1G2_cn * gpweight) / xrefe_cn;

    // loop over all 4 nodes
    for (int i = 0; i < numnod_; ++i)
    {
      // assemble external force vector
      elevec1[noddof_ * i + 0] += fac * xcurr_cross(0) * (shapefcts)(i);
      elevec1[noddof_ * i + 1] += fac * xcurr_cross(1) * (shapefcts)(i);
      elevec1[noddof_ * i + 2] += fac * xcurr_cross(2) * (shapefcts)(i);

      // evaluate external stiffness matrix if needed
      if (elemat1 != nullptr)
      {
        // determine P matrix for all 4 nodes, Gruttmann92 equation (41) and directly fill up elemat
        for (int j = 0; j < numnod_; ++j)
        {
          double p1_ij =
              (dxds1(0) * derivs_ortho(1, i) - dxds2(0) * derivs_ortho(0, i)) * (shapefcts)(j);
          double p2_ij =
              (dxds1(1) * derivs_ortho(1, i) - dxds2(1) * derivs_ortho(0, i)) * (shapefcts)(j);
          double p3_ij =
              (dxds1(2) * derivs_ortho(1, i) - dxds2(2) * derivs_ortho(0, i)) * (shapefcts)(j);

          // entries of P matrix are in round brackets
          (*elemat1)(noddof_* i + 0, noddof_ * j + 1) += fac * -p3_ij;
          (*elemat1)(noddof_* i + 0, noddof_ * j + 2) += fac * +p2_ij;
          (*elemat1)(noddof_* i + 1, noddof_ * j + 0) += fac * +p3_ij;
          (*elemat1)(noddof_* i + 1, noddof_ * j + 2) += fac * -p1_ij;
          (*elemat1)(noddof_* i + 2, noddof_ * j + 0) += fac * -p2_ij;
          (*elemat1)(noddof_* i + 2, noddof_ * j + 1) += fac * +p1_ij;
        }
      }
    }
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  evaluate the element (private)                         fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Membrane<distype>::mem_nlnstiffmass(
    std::vector<int>& lm,                                 // location matrix
    std::vector<double>& disp,                            // current displacements
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,  // element stiffness matrix
    Core::LinAlg::Matrix<numdof_, numdof_>* massmatrix,   // element mass matrix
    Core::LinAlg::Matrix<numdof_, 1>* force,              // element internal force vector
    Core::LinAlg::Matrix<numgpt_post_, 6>* elestress,     // stresses at GP
    Core::LinAlg::Matrix<numgpt_post_, 6>* elestrain,     // strains at GP
    Teuchos::ParameterList& params,                       // algorithmic parameters e.g. time
    const Inpar::Solid::StressType iostress,              // stress output option
    const Inpar::Solid::StrainType iostrain)              // strain output option
{
  // get reference configuration and determine current configuration
  Core::LinAlg::Matrix<numnod_, noddof_> xrefe(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numnod_, noddof_> xcurr(Core::LinAlg::Initialization::zero);

  auto material_local_coordinates =
      std::dynamic_pointer_cast<Mat::MembraneMaterialLocalCoordinates>(
          Core::Elements::Element::material());
  auto material_global_coordinates =
      std::dynamic_pointer_cast<Mat::MembraneMaterialGlobalCoordinates>(
          Core::Elements::Element::material());
  auto material_inelastic_thickness =
      std::dynamic_pointer_cast<Mat::MembraneMaterialInelasticThickness>(
          Core::Elements::Element::material());

  mem_configuration(disp, xrefe, xcurr);

  /*===============================================================================*
   | loop over the gauss points                                                    |
   *===============================================================================*/

  // allocate vector for shape functions and matrix for derivatives at gp
  Core::LinAlg::Matrix<numnod_, 1> shapefcts(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdim_, numnod_> derivs(Core::LinAlg::Initialization::zero);

  const double* total_time =
      params.isParameter("total time") ? &params.get<double>("total time") : nullptr;
  const double* time_step_size =
      params.isParameter("delta time") ? &params.get<double>("delta time") : nullptr;
  for (int gp = 0; gp < intpoints_.nquad; ++gp)
  {
    // get gauss points from integration rule
    double xi_gp = intpoints_.qxg[gp][0];
    double eta_gp = intpoints_.qxg[gp][1];

    // get gauss weight at current gp
    double gpweight = intpoints_.qwgt[gp];

    // get shape functions and derivatives in the plane of the element
    Core::FE::shape_function_2d(shapefcts, xi_gp, eta_gp, shape());
    Core::FE::shape_function_2d_deriv1(derivs, xi_gp, eta_gp, shape());

    /*===============================================================================*
     | orthonormal base (t1,t2,tn) in the undeformed configuration at current GP     |
     *===============================================================================*/

    Core::LinAlg::Matrix<numdim_, numnod_> derivs_ortho(Core::LinAlg::Initialization::zero);
    double G1G2_cn;
    Core::LinAlg::Matrix<noddof_, 1> dXds1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<noddof_, 1> dXds2(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<noddof_, 1> dxds1(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<noddof_, 1> dxds2(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<noddof_, noddof_> Q_localToGlobal(Core::LinAlg::Initialization::zero);

    mem_orthonormalbase(
        xrefe, xcurr, derivs, derivs_ortho, G1G2_cn, dXds1, dXds2, dxds1, dxds2, Q_localToGlobal);

    /*===============================================================================*
     | surface deformation gradient                                                  |
     *===============================================================================*/

    // surface deformation gradient in 3 dimensions in global coordinates
    Core::LinAlg::Matrix<noddof_, noddof_> defgrd_glob(Core::LinAlg::Initialization::zero);

    // surface deformation gradient in 3 dimensions in local coordinates
    Core::LinAlg::Matrix<noddof_, noddof_> defgrd_loc(Core::LinAlg::Initialization::zero);

    // principle stretch in thickness direction
    double lambda3 = 1.0;

    if (material_inelastic_thickness != nullptr)
    {
      // incompressibility is just valid for the elastic quantities, therefore
      // use thickness from previous iteration step to get principle stretch in thickness
      // direction.
      // Stretch in thickness direction is evaluated later by the material
      lambda3 = cur_thickness_[gp] / thickness_;
    }
    else
    {
      // standard evaluation (incompressible, plane stress)
      // incompressibility condition to get principle stretch in thickness direction
      lambda3 =
          std::sqrt(1.0 / (dxds1.dot(dxds1) * dxds2.dot(dxds2) - std::pow(dxds1.dot(dxds2), 2.0)));
    }

    // surface deformation gradient in 3 dimensions in global coordinates
    mem_defgrd_global(dXds1, dXds2, dxds1, dxds2, lambda3, defgrd_glob);

    // surface deformation gradient in 3 dimensions in local coordinates
    Core::LinAlg::FourTensorOperations::inverse_tensor_rotation<3>(
        Q_localToGlobal, defgrd_glob, defgrd_loc);

    /*===============================================================================*
     | right cauchygreen tensor in local coordinates                                 |
     *===============================================================================*/

    // calculate three dimensional right cauchy-green strain tensor in orthonormal base
    Core::LinAlg::Matrix<noddof_, noddof_> cauchygreen_loc(Core::LinAlg::Initialization::zero);
    cauchygreen_loc.multiply_tn(1.0, defgrd_loc, defgrd_loc, 0.0);

    /*===============================================================================*
     | call material law                                                             |
     *===============================================================================*/

    // 2nd piola kirchhoff stress vector under plane stress assumption
    Core::LinAlg::Matrix<3, 1> pk2red_loc(Core::LinAlg::Initialization::zero);

    // material tangent matrix for plane stress
    Core::LinAlg::Matrix<3, 3> cmatred_loc(Core::LinAlg::Initialization::zero);

    // Gauss-point coordinates in reference configuration
    Core::LinAlg::Tensor<double, noddof_> gprefecoord{};
    Core::LinAlg::make_matrix_view<noddof_, 1>(gprefecoord).multiply_tn(xrefe, shapefcts);

    // The growth remodel elast hyper material needs some special quantities for its evaluation
    if (material()->material_type() == Core::Materials::m_growthremodel_elasthyper)
    {
      // center of element in reference configuration
      Core::LinAlg::Matrix<numnod_, 1> funct_center;
      Core::FE::shape_function_2d(funct_center, 0.0, 0.0, distype);
      Core::LinAlg::Tensor<double, noddof_> midpoint;
      Core::LinAlg::make_matrix_view<3, 1>(midpoint).multiply_tn(xrefe, funct_center);
      params.set("elecenter_coords_ref", midpoint);
    }

    Core::LinAlg::Tensor<double, 3> xi = {{xi_gp, eta_gp, 0.0}};
    Mat::EvaluationContext context{.total_time = total_time,
        .time_step_size = time_step_size,
        .xi = &xi,
        .ref_coords = &gprefecoord};
    if (material_inelastic_thickness != nullptr)
    {
      // Let material decide the total stretch in thickness direction
      lambda3 = material_inelastic_thickness->evaluate_membrane_thickness_stretch(
          defgrd_glob, params, context, gp, id());

      // update surface deformation gradient in 3 dimensions in global coordinates
      mem_defgrd_global(dXds1, dXds2, dxds1, dxds2, lambda3, defgrd_glob);

      // update surface deformation gradient in 3 dimensions in local coordinates
      Core::LinAlg::FourTensorOperations::inverse_tensor_rotation<3>(
          Q_localToGlobal, defgrd_glob, defgrd_loc);

      // update three dimensional right cauchy-green strain tensor in orthonormal base
      cauchygreen_loc.multiply_tn(1.0, defgrd_loc, defgrd_loc, 0.0);
    }

    // standard evaluation (incompressible, plane stress)
    if (material_local_coordinates != nullptr)
    {
      material_local_coordinates->evaluate_membrane(defgrd_loc, cauchygreen_loc, params, context,
          Q_localToGlobal, pk2red_loc, cmatred_loc, gp, id());
    }
    else if (material_global_coordinates != nullptr)
    {
      Core::LinAlg::Matrix<3, 3> pk2M_glob(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<6, 6> cmat_glob(Core::LinAlg::Initialization::zero);

      // Evaluate material with quantities in the global coordinate system
      material_global_coordinates->evaluate_membrane(
          defgrd_glob, params, context, pk2M_glob, cmat_glob, gp, id());

      // Transform stress and elasticity into the local membrane coordinate system
      Core::LinAlg::Matrix<3, 3> pk2M_loc(Core::LinAlg::Initialization::zero);
      Core::LinAlg::FourTensorOperations::inverse_tensor_rotation<3>(
          Q_localToGlobal, pk2M_glob, pk2M_loc);
      Internal::local_plane_stress_to_stress_like_voigt(pk2M_loc, pk2red_loc);

      Core::LinAlg::Matrix<6, 6> cmat_loc(Core::LinAlg::Initialization::zero);
      Core::LinAlg::FourTensorOperations::inverse_fourth_tensor_rotation(
          Q_localToGlobal, cmat_glob, cmat_loc);
      Internal::local_fourth_tensor_plane_stress_to_stress_like_voigt(cmat_loc, cmatred_loc);
    }
    else
    {
      FOUR_C_THROW("The material does not support the evaluation of membranes");
    }

    /*===============================================================================*
     | update current thickness at gp                                                |
     *===============================================================================*/
    cur_thickness_[gp] = lambda3 * thickness_;

    /*===============================================================================*
     | calculate force, stiffness matrix and mass matrix                             |
     *===============================================================================*/
    // evaluate just force vector (stiffness matrix not needed)
    if (stiffmatrix == nullptr && force != nullptr)
    {
      // determine B matrix for all 4 nodes, Gruttmann1992 equation (36)
      Core::LinAlg::Matrix<noddof_, numdof_> B_matrix(Core::LinAlg::Initialization::zero);

      for (int i = 0; i < numnod_; ++i)
      {
        B_matrix(0, noddof_ * i + 0) = derivs_ortho(0, i) * dxds1(0);
        B_matrix(1, noddof_ * i + 0) = derivs_ortho(1, i) * dxds2(0);
        B_matrix(2, noddof_ * i + 0) =
            derivs_ortho(0, i) * dxds2(0) + derivs_ortho(1, i) * dxds1(0);

        B_matrix(0, noddof_ * i + 1) = derivs_ortho(0, i) * dxds1(1);
        B_matrix(1, noddof_ * i + 1) = derivs_ortho(1, i) * dxds2(1);
        B_matrix(2, noddof_ * i + 1) =
            derivs_ortho(0, i) * dxds2(1) + derivs_ortho(1, i) * dxds1(1);

        B_matrix(0, noddof_ * i + 2) = derivs_ortho(0, i) * dxds1(2);
        B_matrix(1, noddof_ * i + 2) = derivs_ortho(1, i) * dxds2(2);
        B_matrix(2, noddof_ * i + 2) =
            derivs_ortho(0, i) * dxds2(2) + derivs_ortho(1, i) * dxds1(2);
      }

      double fac = gpweight * thickness_ * G1G2_cn;

      // determine force and stiffness matrix, Gruttmann1992 equation (37) and (39)
      force->multiply_tn(fac, B_matrix, pk2red_loc, 1.0);
    }

    // evaluate stiffness matrix and force vector if needed
    if (stiffmatrix != nullptr && force != nullptr)
    {
      // determine B matrix and G matrix for all 4 nodes, Gruttmann1992 equation (36) and (40)
      Core::LinAlg::Matrix<noddof_, numdof_> B_matrix(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<numdof_, numdof_> G_matrix(Core::LinAlg::Initialization::zero);
      double g_ij;

      for (int i = 0; i < numnod_; ++i)
      {
        B_matrix(0, noddof_ * i + 0) = derivs_ortho(0, i) * dxds1(0);
        B_matrix(1, noddof_ * i + 0) = derivs_ortho(1, i) * dxds2(0);
        B_matrix(2, noddof_ * i + 0) =
            derivs_ortho(0, i) * dxds2(0) + derivs_ortho(1, i) * dxds1(0);

        B_matrix(0, noddof_ * i + 1) = derivs_ortho(0, i) * dxds1(1);
        B_matrix(1, noddof_ * i + 1) = derivs_ortho(1, i) * dxds2(1);
        B_matrix(2, noddof_ * i + 1) =
            derivs_ortho(0, i) * dxds2(1) + derivs_ortho(1, i) * dxds1(1);

        B_matrix(0, noddof_ * i + 2) = derivs_ortho(0, i) * dxds1(2);
        B_matrix(1, noddof_ * i + 2) = derivs_ortho(1, i) * dxds2(2);
        B_matrix(2, noddof_ * i + 2) =
            derivs_ortho(0, i) * dxds2(2) + derivs_ortho(1, i) * dxds1(2);

        for (int j = 0; j < numnod_; ++j)
        {
          g_ij = pk2red_loc(0) * derivs_ortho(0, i) * derivs_ortho(0, j) +
                 pk2red_loc(1) * derivs_ortho(1, i) * derivs_ortho(1, j) +
                 pk2red_loc(2) * (derivs_ortho(0, i) * derivs_ortho(1, j) +
                                     derivs_ortho(1, i) * derivs_ortho(0, j));

          G_matrix(noddof_ * i + 0, noddof_ * j + 0) = g_ij;
          G_matrix(noddof_ * i + 1, noddof_ * j + 1) = g_ij;
          G_matrix(noddof_ * i + 2, noddof_ * j + 2) = g_ij;
        }
      }

      double fac = gpweight * thickness_ * G1G2_cn;

      // determine force and stiffness matrix, Gruttmann1992 equation (37) and (39)
      force->multiply_tn(fac, B_matrix, pk2red_loc, 1.0);

      Core::LinAlg::Matrix<numdof_, noddof_> temp(Core::LinAlg::Initialization::zero);
      temp.multiply_tn(1.0, B_matrix, cmatred_loc, 0.0);
      Core::LinAlg::Matrix<numdof_, numdof_> temp2(Core::LinAlg::Initialization::zero);
      temp2.multiply(1.0, temp, B_matrix, 0.0);
      temp2.update(1.0, G_matrix, 1.0);

      stiffmatrix->update(fac, temp2, 1.0);
    }

    // evaluate massmatrix if needed, just valid for a constant density
    if (massmatrix != nullptr)
    {
      // get density
      double density = solid_material()->density();

      // integrate consistent mass matrix
      const double factor = gpweight * thickness_ * G1G2_cn * density;
      double ifactor = 0.0;
      double massfactor = 0.0;

      for (int i = 0; i < numnod_; ++i)
      {
        ifactor = shapefcts(i) * factor;

        for (int j = 0; j < numnod_; ++j)
        {
          massfactor = shapefcts(j) * ifactor;  // intermediate factor

          (*massmatrix)(noddof_* i + 0, noddof_ * j + 0) += massfactor;
          (*massmatrix)(noddof_* i + 1, noddof_ * j + 1) += massfactor;
          (*massmatrix)(noddof_* i + 2, noddof_ * j + 2) += massfactor;
        }
      }

      // check for non constant mass matrix
      if (solid_material()->varying_density())
      {
        FOUR_C_THROW("Varying Density not supported for Membrane");
      }
    }

    /*===============================================================================*
     | return gp strains (only in case of stress/strain output)                      |
     *===============================================================================*/
    switch (iostrain)
    {
      // Green-Lagrange strains
      case Inpar::Solid::strain_gl:
      {
        if (elestrain == nullptr) FOUR_C_THROW("strain data not available");

        // transform local cauchygreen to global coordinates
        Core::LinAlg::Matrix<noddof_, noddof_> cauchygreen_glob(Core::LinAlg::Initialization::zero);
        Core::LinAlg::FourTensorOperations::tensor_rotation<3>(
            Q_localToGlobal, cauchygreen_loc, cauchygreen_glob);

        // green-lagrange strain tensor in global coordinates
        Core::LinAlg::Matrix<noddof_, noddof_> glstrain_glob(Core::LinAlg::Initialization::zero);
        glstrain_glob(0, 0) = 0.5 * (cauchygreen_glob(0, 0) - 1.0);
        glstrain_glob(1, 1) = 0.5 * (cauchygreen_glob(1, 1) - 1.0);
        glstrain_glob(2, 2) = 0.5 * (cauchygreen_glob(2, 2) - 1.0);
        glstrain_glob(0, 1) = 0.5 * cauchygreen_glob(0, 1);
        glstrain_glob(0, 2) = 0.5 * cauchygreen_glob(0, 2);
        glstrain_glob(1, 2) = 0.5 * cauchygreen_glob(1, 2);
        glstrain_glob(1, 0) = glstrain_glob(0, 1);
        glstrain_glob(2, 0) = glstrain_glob(0, 2);
        glstrain_glob(2, 1) = glstrain_glob(1, 2);

        (*elestrain)(gp, 0) = glstrain_glob(0, 0);
        (*elestrain)(gp, 1) = glstrain_glob(1, 1);
        (*elestrain)(gp, 2) = glstrain_glob(2, 2);
        (*elestrain)(gp, 3) = glstrain_glob(0, 1);
        (*elestrain)(gp, 4) = glstrain_glob(1, 2);
        (*elestrain)(gp, 5) = glstrain_glob(0, 2);
      }
      break;
      // Euler-Almansi strains
      case Inpar::Solid::strain_ea:
      {
        if (elestrain == nullptr) FOUR_C_THROW("strain data not available");

        // transform local cauchygreen to global coordinates
        Core::LinAlg::Matrix<noddof_, noddof_> cauchygreen_glob(Core::LinAlg::Initialization::zero);
        Core::LinAlg::FourTensorOperations::tensor_rotation<3>(
            Q_localToGlobal, cauchygreen_loc, cauchygreen_glob);

        // green-lagrange strain tensor in global coordinates
        Core::LinAlg::Matrix<noddof_, noddof_> glstrain_glob(Core::LinAlg::Initialization::zero);
        glstrain_glob(0, 0) = 0.5 * (cauchygreen_glob(0, 0) - 1);
        glstrain_glob(1, 1) = 0.5 * (cauchygreen_glob(1, 1) - 1);
        glstrain_glob(2, 2) = 0.5 * (cauchygreen_glob(2, 2) - 1);
        glstrain_glob(0, 1) = 0.5 * cauchygreen_glob(0, 1);
        glstrain_glob(0, 2) = 0.5 * cauchygreen_glob(0, 2);
        glstrain_glob(1, 2) = 0.5 * cauchygreen_glob(1, 2);
        glstrain_glob(1, 0) = glstrain_glob(0, 1);
        glstrain_glob(2, 0) = glstrain_glob(0, 2);
        glstrain_glob(2, 1) = glstrain_glob(1, 2);

        // pushforward of gl strains to ea strains
        Core::LinAlg::Matrix<noddof_, noddof_> euler_almansi(Core::LinAlg::Initialization::zero);
        mem_g_lto_ea(glstrain_glob, defgrd_glob, euler_almansi);

        (*elestrain)(gp, 0) = euler_almansi(0, 0);
        (*elestrain)(gp, 1) = euler_almansi(1, 1);
        (*elestrain)(gp, 2) = euler_almansi(2, 2);
        (*elestrain)(gp, 3) = euler_almansi(0, 1);
        (*elestrain)(gp, 4) = euler_almansi(1, 2);
        (*elestrain)(gp, 5) = euler_almansi(0, 2);
      }
      break;
      // Logarithmic strains
      case Inpar::Solid::strain_log:
      {
        if (elestrain == nullptr) FOUR_C_THROW("strain data not available");

        // the Eularian logarithmic strain is defined as the natural logarithm of the left stretch
        // tensor [1,2]: e_{log} = e_{hencky} = ln (\mathbf{V}) = \sum_{i=1}^3 (ln \lambda_i)
        // \mathbf{n}_i \otimes \mathbf{n}_i References: [1] H. Xiao, Beijing, China, O. T. Bruhns
        // and A. Meyers (1997) Logarithmic strain, logarithmic spin and logarithmic rate, Eq. 5 [2]
        // Caminero et al. (2011) Modeling large strain anisotropic elasto-plasticity with
        // logarithmic strain and stress measures, Eq. 70

        // transform local cauchygreen to global coordinates
        Core::LinAlg::Matrix<noddof_, noddof_> cauchygreen_glob(Core::LinAlg::Initialization::zero);
        Core::LinAlg::FourTensorOperations::tensor_rotation<3>(
            Q_localToGlobal, cauchygreen_loc, cauchygreen_glob);

        // eigenvalue decomposition (from elasthyper.cpp)
        Core::LinAlg::Matrix<noddof_, noddof_> prstr2(
            Core::LinAlg::Initialization::zero);  // squared principal stretches
        Core::LinAlg::Matrix<noddof_, 1> prstr(
            Core::LinAlg::Initialization::zero);  // principal stretch
        Core::LinAlg::Matrix<noddof_, noddof_> prdir(
            Core::LinAlg::Initialization::zero);  // principal directions
        Core::LinAlg::syev(cauchygreen_glob, prstr2, prdir);

        // THE principal stretches
        for (int al = 0; al < 3; ++al) prstr(al) = std::sqrt(prstr2(al, al));

        // populating the logarithmic strain matrix
        Core::LinAlg::Matrix<noddof_, noddof_> lnv(Core::LinAlg::Initialization::zero);

        // checking if cauchy green is correctly determined to ensure eigenvectors in correct
        // direction i.e. a flipped eigenvector is also a valid solution C = \sum_{i=1}^3
        // (\lambda_i^2) \mathbf{n}_i \otimes \mathbf{n}_i
        Core::LinAlg::Matrix<noddof_, noddof_> tempCG(Core::LinAlg::Initialization::zero);

        for (int k = 0; k < 3; ++k)
        {
          double n_00, n_01, n_02, n_11, n_12, n_22 = 0.0;

          n_00 = prdir(0, k) * prdir(0, k);
          n_01 = prdir(0, k) * prdir(1, k);
          n_02 = prdir(0, k) * prdir(2, k);
          n_11 = prdir(1, k) * prdir(1, k);
          n_12 = prdir(1, k) * prdir(2, k);
          n_22 = prdir(2, k) * prdir(2, k);

          // only compute the symmetric components from a single eigenvector,
          // because eigenvalue directions are not consistent (it can be flipped)
          tempCG(0, 0) += (prstr(k)) * (prstr(k)) * n_00;
          tempCG(0, 1) += (prstr(k)) * (prstr(k)) * n_01;
          tempCG(0, 2) += (prstr(k)) * (prstr(k)) * n_02;
          tempCG(1, 0) += (prstr(k)) * (prstr(k)) * n_01;  // symmetry
          tempCG(1, 1) += (prstr(k)) * (prstr(k)) * n_11;
          tempCG(1, 2) += (prstr(k)) * (prstr(k)) * n_12;
          tempCG(2, 0) += (prstr(k)) * (prstr(k)) * n_02;  // symmetry
          tempCG(2, 1) += (prstr(k)) * (prstr(k)) * n_12;  // symmetry
          tempCG(2, 2) += (prstr(k)) * (prstr(k)) * n_22;

          // Computation of the Logarithmic strain tensor

          lnv(0, 0) += (std::log(prstr(k))) * n_00;
          lnv(0, 1) += (std::log(prstr(k))) * n_01;
          lnv(0, 2) += (std::log(prstr(k))) * n_02;
          lnv(1, 0) += (std::log(prstr(k))) * n_01;  // symmetry
          lnv(1, 1) += (std::log(prstr(k))) * n_11;
          lnv(1, 2) += (std::log(prstr(k))) * n_12;
          lnv(2, 0) += (std::log(prstr(k))) * n_02;  // symmetry
          lnv(2, 1) += (std::log(prstr(k))) * n_12;  // symmetry
          lnv(2, 2) += (std::log(prstr(k))) * n_22;
        }

        // compare CG computed with deformation gradient with CG computed
        // with eigenvalues and -vectors to determine/ensure the correct
        // orientation of the eigen vectors
        Core::LinAlg::Matrix<noddof_, noddof_> diffCG(Core::LinAlg::Initialization::zero);

        for (int i = 0; i < 3; ++i)
        {
          for (int j = 0; j < 3; ++j)
          {
            diffCG(i, j) = cauchygreen_glob(i, j) - tempCG(i, j);
            // the solution to this problem is to evaluate the cauchygreen tensor with
            // tempCG computed with every combination of eigenvector orientations -- up to nine
            // comparisons
            if (diffCG(i, j) > 1e-10)
            {
              FOUR_C_THROW(
                  "eigenvector orientation error with the diffCG giving problems: {:10.5e} \n "
                  "BUILD "
                  "SOLUTION TO FIX IT",
                  diffCG(i, j));
            }
          }
        }

        (*elestrain)(gp, 0) = lnv(0, 0);
        (*elestrain)(gp, 1) = lnv(1, 1);
        (*elestrain)(gp, 2) = lnv(2, 2);
        (*elestrain)(gp, 3) = lnv(0, 1);
        (*elestrain)(gp, 4) = lnv(1, 2);
        (*elestrain)(gp, 5) = lnv(0, 2);
      }
      break;
      // no strain output
      case Inpar::Solid::strain_none:
        break;
      default:
        FOUR_C_THROW("requested strain type not available");
        break;
    }

    /*===============================================================================*
     | return gp stresses (only in case of stress/strain output)                     |
     *===============================================================================*/
    switch (iostress)
    {
      // 2nd Piola-Kirchhoff stresses
      case Inpar::Solid::stress_2pk:
      {
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");

        // 2nd Piola-Kirchhoff stress in tensor notation, plane stress meaning entries in 2i and i2
        // are zero for i=0,1,2
        Core::LinAlg::Matrix<noddof_, noddof_> pkstressM_local(Core::LinAlg::Initialization::zero);
        pkstressM_local(0, 0) = pk2red_loc(0);
        pkstressM_local(1, 1) = pk2red_loc(1);
        pkstressM_local(0, 1) = pk2red_loc(2);
        pkstressM_local(1, 0) = pk2red_loc(2);

        // determine 2nd Piola-Kirchhoff stresses in global coordinates
        Core::LinAlg::Matrix<noddof_, noddof_> pkstress_glob(Core::LinAlg::Initialization::zero);
        Core::LinAlg::FourTensorOperations::tensor_rotation<3>(
            Q_localToGlobal, pkstressM_local, pkstress_glob);

        (*elestress)(gp, 0) = pkstress_glob(0, 0);
        (*elestress)(gp, 1) = pkstress_glob(1, 1);
        (*elestress)(gp, 2) = pkstress_glob(2, 2);
        (*elestress)(gp, 3) = pkstress_glob(0, 1);
        (*elestress)(gp, 4) = pkstress_glob(1, 2);
        (*elestress)(gp, 5) = pkstress_glob(0, 2);
      }
      break;
      // Cauchy stresses
      case Inpar::Solid::stress_cauchy:
      {
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");

        // 2nd Piola-Kirchhoff stress in tensor notation, plane stress meaning entries in 2i and i2
        // are zero for i=0,1,2
        Core::LinAlg::Matrix<noddof_, noddof_> pkstressM_loc(Core::LinAlg::Initialization::zero);
        pkstressM_loc(0, 0) = pk2red_loc(0);
        pkstressM_loc(1, 1) = pk2red_loc(1);
        pkstressM_loc(0, 1) = pk2red_loc(2);
        pkstressM_loc(1, 0) = pk2red_loc(2);

        // determine 2nd Piola-Kirchhoff stresses in global coordinates
        Core::LinAlg::Matrix<noddof_, noddof_> pkstress_glob(Core::LinAlg::Initialization::zero);
        Core::LinAlg::FourTensorOperations::tensor_rotation<3>(
            Q_localToGlobal, pkstressM_loc, pkstress_glob);

        Core::LinAlg::Matrix<noddof_, noddof_> cauchy_glob(Core::LinAlg::Initialization::zero);
        mem_p_k2to_cauchy(pkstress_glob, defgrd_glob, cauchy_glob);

        (*elestress)(gp, 0) = cauchy_glob(0, 0);
        (*elestress)(gp, 1) = cauchy_glob(1, 1);
        (*elestress)(gp, 2) = cauchy_glob(2, 2);
        (*elestress)(gp, 3) = cauchy_glob(0, 1);
        (*elestress)(gp, 4) = cauchy_glob(1, 2);
        (*elestress)(gp, 5) = cauchy_glob(0, 2);
      }
      break;
      // no stress output
      case Inpar::Solid::stress_none:
        break;
      default:
        FOUR_C_THROW("requested stress type not available");
        break;
    }
  }
}  // Discret::Elements::Membrane::membrane_nlnstiffmass

/*----------------------------------------------------------------------*
 |  Return names of visualization data (public)                fb 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Membrane<distype>::vis_names(std::map<std::string, int>& names)
{
  std::string result_thickness = "thickness";

  names[result_thickness] = 1;


  solid_material()->vis_names(names);

}  // Discret::Elements::Membrane::vis_names

/*----------------------------------------------------------------------*
 |  Return visualization data (public)                     fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Discret::Elements::Membrane<distype>::vis_data(
    const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Core::Elements::Element::vis_data(name, data)) return true;

  if (name == "thickness")
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (int gp = 0; gp < intpoints_.nquad; gp++)
    {
      data[0] += cur_thickness_[gp];
    }
    data[0] = data[0] / intpoints_.nquad;

    return true;
  }

  return solid_material()->vis_data(name, data, intpoints_.nquad, this->id());

}  // Discret::Elements::Membrane::vis_data

/*----------------------------------------------------------------------*
 |  get reference and current configuration                fbraeu 06/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Membrane<distype>::mem_configuration(const std::vector<double>& disp,
    Core::LinAlg::Matrix<numnod_, noddof_>& xrefe, Core::LinAlg::Matrix<numnod_, noddof_>& xcurr)
{
  // get reference configuration and determine current configuration
  Core::Nodes::Node** nodes = Membrane::nodes();
  if (!nodes) FOUR_C_THROW("Nodes() returned null pointer");

  for (int i = 0; i < numnod_; ++i)
  {
    const auto& x = nodes[i]->x();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];

    xcurr(i, 0) = xrefe(i, 0) + disp[i * noddof_ + 0];
    xcurr(i, 1) = xrefe(i, 1) + disp[i * noddof_ + 1];
    xcurr(i, 2) = xrefe(i, 2) + disp[i * noddof_ + 2];
  }
}  // Discret::Elements::Membrane::mem_configuration

/*------------------------------------------------------------------------------------------------------*
 |  introduce an orthonormal base in the undeformed configuration at current Gauss point   fbraeu
 06/16 |
 *------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Membrane<distype>::mem_orthonormalbase(
    const Core::LinAlg::Matrix<numnod_, noddof_>& xrefe,
    const Core::LinAlg::Matrix<numnod_, noddof_>& xcurr,
    const Core::LinAlg::Matrix<numdim_, numnod_>& derivs,
    Core::LinAlg::Matrix<numdim_, numnod_>& derivs_ortho, double& G1G2_cn,
    Core::LinAlg::Matrix<noddof_, 1>& dXds1, Core::LinAlg::Matrix<noddof_, 1>& dXds2,
    Core::LinAlg::Matrix<noddof_, 1>& dxds1, Core::LinAlg::Matrix<noddof_, 1>& dxds2,
    Core::LinAlg::Matrix<noddof_, noddof_>& Q_localToGlobal) const
{
  /*===============================================================================*
   | introduce an orthonormal base in the undeformed configuration as proposed in: |
   | Gruttmann, "Theory and finite element formulation of rubberlike membrane      |
   | shells using principal stretches", 1992                                       |
   *===============================================================================*/

  Core::LinAlg::Matrix<noddof_, numdim_> G12(Core::LinAlg::Initialization::zero);
  G12.multiply_tt(1.0, xrefe, derivs, 0.0);

  // G1 and G2 Gruttmann1992 equation (43)
  Core::LinAlg::Matrix<noddof_, 1> G1(Core::LinAlg::Initialization::zero);
  G1(0) = G12(0, 0);
  G1(1) = G12(1, 0);
  G1(2) = G12(2, 0);

  Core::LinAlg::Matrix<noddof_, 1> G2(Core::LinAlg::Initialization::zero);
  G2(0) = G12(0, 1);
  G2(1) = G12(1, 1);
  G2(2) = G12(2, 1);

  // cross product G1xG2
  Core::LinAlg::Matrix<noddof_, 1> G1G2_cross(Core::LinAlg::Initialization::zero);
  G1G2_cross(0) = G1(1) * G2(2) - G1(2) * G2(1);
  G1G2_cross(1) = G1(2) * G2(0) - G1(0) * G2(2);
  G1G2_cross(2) = G1(0) * G2(1) - G1(1) * G2(0);

  // 2 norm of vectors
  G1G2_cn = G1G2_cross.norm2();
  double G1_n = G1.norm2();

  // Gruttmann1992 equation (44), orthonormal base vectors
  Core::LinAlg::Matrix<noddof_, 1> tn(Core::LinAlg::Initialization::zero);
  tn(0) = G1G2_cross(0) / G1G2_cn;
  tn(1) = G1G2_cross(1) / G1G2_cn;
  tn(2) = G1G2_cross(2) / G1G2_cn;

  Core::LinAlg::Matrix<noddof_, 1> t1(Core::LinAlg::Initialization::zero);
  t1(0) = G1(0) / G1_n;
  t1(1) = G1(1) / G1_n;
  t1(2) = G1(2) / G1_n;

  Core::LinAlg::Matrix<noddof_, 1> t2(Core::LinAlg::Initialization::zero);
  t2(0) = tn(1) * t1(2) - tn(2) * t1(1);
  t2(1) = tn(2) * t1(0) - tn(0) * t1(2);
  t2(2) = tn(0) * t1(1) - tn(1) * t1(0);

  Core::LinAlg::Matrix<noddof_, numdim_> t12(Core::LinAlg::Initialization::zero);
  t12(0, 0) = t1(0);
  t12(1, 0) = t1(1);
  t12(2, 0) = t1(2);
  t12(0, 1) = t2(0);
  t12(1, 1) = t2(1);
  t12(2, 1) = t2(2);

  // Jacobian transformation matrix and its inverse, Gruttmann1992 equation (44b)
  // for the Trafo from local membrane orthonormal coordinates to global coordinates
  // It is not the Jacobian for the Trafo from the parameter space xi, eta to the global coords!
  Core::LinAlg::Matrix<numdim_, numdim_> J(Core::LinAlg::Initialization::zero);
  J.multiply_tn(1.0, G12, t12, 0.0);

  Core::LinAlg::Matrix<numdim_, numdim_> Jinv(Core::LinAlg::Initialization::zero);
  Jinv.invert(J);

  // calclate derivatives of shape functions in orthonormal base, Gruttmann1992 equation (42)
  derivs_ortho.multiply(1.0, Jinv, derivs, 0.0);

  // derivative of the reference position wrt the orthonormal base
  Core::LinAlg::Matrix<noddof_, numdim_> dXds(Core::LinAlg::Initialization::zero);
  dXds.multiply_tt(1.0, xrefe, derivs_ortho, 0.0);

  dXds1(0) = dXds(0, 0);
  dXds1(1) = dXds(1, 0);
  dXds1(2) = dXds(2, 0);

  dXds2(0) = dXds(0, 1);
  dXds2(1) = dXds(1, 1);
  dXds2(2) = dXds(2, 1);

  // derivative of the current position wrt the orthonormal base
  Core::LinAlg::Matrix<noddof_, numdim_> dxds(Core::LinAlg::Initialization::zero);
  dxds.multiply_tt(1.0, xcurr, derivs_ortho, 0.0);

  dxds1(0) = dxds(0, 0);
  dxds1(1) = dxds(1, 0);
  dxds1(2) = dxds(2, 0);

  dxds2(0) = dxds(0, 1);
  dxds2(1) = dxds(1, 1);
  dxds2(2) = dxds(2, 1);

  // determine Trafo from local membrane orthonormal coordinates to global coordinates
  Q_localToGlobal(0, 0) = t1(0);
  Q_localToGlobal(1, 0) = t1(1);
  Q_localToGlobal(2, 0) = t1(2);
  Q_localToGlobal(0, 1) = t2(0);
  Q_localToGlobal(1, 1) = t2(1);
  Q_localToGlobal(2, 1) = t2(2);
  Q_localToGlobal(0, 2) = tn(0);
  Q_localToGlobal(1, 2) = tn(1);
  Q_localToGlobal(2, 2) = tn(2);

}  // Discret::Elements::Membrane::mem_orthonormalbase

/*-------------------------------------------------------------------------------------------------*
 |  pushforward of 2nd PK stresses to Cauchy stresses at gp                           fbraeu 06/16 |
 *-------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Membrane<distype>::mem_p_k2to_cauchy(
    const Core::LinAlg::Matrix<noddof_, noddof_>& pkstress_global,
    const Core::LinAlg::Matrix<noddof_, noddof_>& defgrd,
    Core::LinAlg::Matrix<noddof_, noddof_>& cauchy) const
{
  // calculate the Jacobi-deterinant
  const double detF = defgrd.determinant();

  // check determinant of deformation gradient
  if (detF == 0) FOUR_C_THROW("Zero determinant of Deformation Gradient.");

  // determine the cauchy stresses
  Core::LinAlg::Matrix<noddof_, noddof_> temp;
  temp.multiply((1.0 / detF), defgrd, pkstress_global, 0.0);
  cauchy.multiply_nt(1.0, temp, defgrd, 1.0);

}  // Discret::Elements::Membrane::mem_p_k2to_cauchy

/*-------------------------------------------------------------------------------------------------*
 |  pushforward of Green-Lagrange to Euler-Almansi strains at gp                      fbraeu 06/16 |
 *-------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Membrane<distype>::mem_g_lto_ea(
    const Core::LinAlg::Matrix<noddof_, noddof_>& glstrain_global,
    const Core::LinAlg::Matrix<noddof_, noddof_>& defgrd,
    Core::LinAlg::Matrix<noddof_, noddof_>& euler_almansi) const
{
  // check determinant of deformation gradient
  if (defgrd.determinant() == 0)
    FOUR_C_THROW(
        "Inverse of Deformation Gradient can not be calculated due to a zero determinant.");

  // inverse of deformation gradient
  Core::LinAlg::Matrix<noddof_, noddof_> invdefgrd(Core::LinAlg::Initialization::zero);
  invdefgrd.invert(defgrd);

  // determine the euler-almansi strains
  Core::LinAlg::Matrix<noddof_, noddof_> temp;
  temp.multiply(1.0, glstrain_global, invdefgrd, 0.0);
  euler_almansi.multiply_tn(1.0, invdefgrd, temp, 1.0);

}  // Discret::Elements::Membrane::mem_g_lto_ea

/*-------------------------------------------------------------------------------------------------*
 |  determine deformation gradient in global coordinates                              fbraeu 06/16 |
 *-------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Membrane<distype>::mem_defgrd_global(
    const Core::LinAlg::Matrix<noddof_, 1>& dXds1, const Core::LinAlg::Matrix<noddof_, 1>& dXds2,
    const Core::LinAlg::Matrix<noddof_, 1>& dxds1, const Core::LinAlg::Matrix<noddof_, 1>& dxds2,
    const double& lambda3, Core::LinAlg::Matrix<noddof_, noddof_>& defgrd_glob) const
{
  // clear
  defgrd_glob.clear();

  // determine cross product x,1 x x,2
  Core::LinAlg::Matrix<noddof_, 1> xcurr_cross(Core::LinAlg::Initialization::zero);
  xcurr_cross(0) = dxds1(1) * dxds2(2) - dxds1(2) * dxds2(1);
  xcurr_cross(1) = dxds1(2) * dxds2(0) - dxds1(0) * dxds2(2);
  xcurr_cross(2) = dxds1(0) * dxds2(1) - dxds1(1) * dxds2(0);

  // normalize the cross product for the current configuration
  xcurr_cross.scale(1.0 / xcurr_cross.norm2());

  // determine cross product X,1 x X,2, has unit length due to orthonormal basis
  Core::LinAlg::Matrix<noddof_, 1> xrefe_cross(Core::LinAlg::Initialization::zero);
  xrefe_cross(0) = dXds1(1) * dXds2(2) - dXds1(2) * dXds2(1);
  xrefe_cross(1) = dXds1(2) * dXds2(0) - dXds1(0) * dXds2(2);
  xrefe_cross(2) = dXds1(0) * dXds2(1) - dXds1(1) * dXds2(0);

  defgrd_glob.multiply_nt(1.0, dxds1, dXds1, 0.0);
  defgrd_glob.multiply_nt(1.0, dxds2, dXds2, 1.0);
  // scale third dimension by sqrt(rcg33), that equals the principle stretch lambda_3
  defgrd_glob.multiply_nt(lambda3, xcurr_cross, xrefe_cross, 1.0);

}  // Discret::Elements::Membrane::mem_defgrd_global

/*-------------------------------------------------------------------------------------------------*
 |  determine extrapolation matrix                                                                 |
 *-------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Core::LinAlg::Matrix<Core::FE::num_nodes(distype), Thermo::DisTypeToNumGaussPoints<distype>::nquad>
Discret::Elements::Membrane<distype>::mem_extrapolmat() const
{
  // extrapolation matrix
  // note: equal for all elements of the same discretization type
  Core::LinAlg::Matrix<numnod_, numgpt_post_> extrapol;

  // check for correct gaussrule
  if (intpoints_.nquad != numgpt_post_)
    FOUR_C_THROW(
        "number of gauss points of gaussrule_ does not match numgpt_post_ used for postprocessing");

  // allocate vector for shape functions and matrix for derivatives at gp
  Core::LinAlg::Matrix<numnod_, 1> shapefcts(Core::LinAlg::Initialization::zero);

  // loop over the nodes and gauss points
  // interpolation matrix, inverted later to be the extrapolation matrix
  for (int nd = 0; nd < numnod_; ++nd)
  {
    // gaussian coordinates
    const double e1 = intpoints_.qxg[nd][0];
    const double e2 = intpoints_.qxg[nd][1];

    // shape functions for the extrapolated coordinates
    Core::LinAlg::Matrix<numgpt_post_, 1> funct;
    Core::FE::shape_function_2d(funct, e1, e2, shape());

    for (int i = 0; i < numgpt_post_; ++i) extrapol(nd, i) = funct(i);
  }

  // fixedsizesolver for inverting extrapol
  Core::LinAlg::FixedSizeSerialDenseSolver<numnod_, numgpt_post_, 1> solver;
  solver.set_matrix(extrapol);
  int err = solver.invert();
  if (err != 0.) FOUR_C_THROW("Matrix extrapol is not invertible");

  return extrapol;

}  // Discret::Elements::Membrane::mem_extrapolmat

/*---------------------------------------------------------------------------------------------*
 |  Update history variables (e.g. remodeling of fiber directions) (protected)      braeu 07/16|
 *---------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Membrane<distype>::update_element(
    std::vector<double>& disp, Teuchos::ParameterList& params, Core::Mat::Material& mat)
{
  // Calculate current deformation gradient
  if (solid_material()->uses_extended_update())
  {
    // get reference configuration and determine current configuration
    Core::LinAlg::Matrix<numnod_, noddof_> xrefe(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<numnod_, noddof_> xcurr(Core::LinAlg::Initialization::zero);

    mem_configuration(disp, xrefe, xcurr);

    /*===============================================================================*
     | loop over the gauss points                                                    |
     *===============================================================================*/

    // allocate vector for shape functions and matrix for derivatives at gp
    Core::LinAlg::Matrix<numdim_, numnod_> derivs(Core::LinAlg::Initialization::zero);

    const double* total_time =
        params.isParameter("total time") ? &params.get<double>("total time") : nullptr;
    const double* time_step_size =
        params.isParameter("delta time") ? &params.get<double>("delta time") : nullptr;

    for (int gp = 0; gp < intpoints_.nquad; ++gp)
    {
      // get gauss points from integration rule
      double xi_gp = intpoints_.qxg[gp][0];
      double eta_gp = intpoints_.qxg[gp][1];

      // get derivatives in the plane of the element
      Core::FE::shape_function_2d_deriv1(derivs, xi_gp, eta_gp, shape());

      /*===============================================================================*
       | orthonormal base (t1,t2,tn) in the undeformed configuration at current GP     |
       *===============================================================================*/

      Core::LinAlg::Matrix<numdim_, numnod_> derivs_ortho(Core::LinAlg::Initialization::zero);
      double G1G2_cn;
      Core::LinAlg::Matrix<noddof_, 1> dXds1(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<noddof_, 1> dXds2(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<noddof_, 1> dxds1(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<noddof_, 1> dxds2(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<noddof_, noddof_> Q_localToGlobal(Core::LinAlg::Initialization::zero);

      mem_orthonormalbase(
          xrefe, xcurr, derivs, derivs_ortho, G1G2_cn, dXds1, dXds2, dxds1, dxds2, Q_localToGlobal);

      /*===============================================================================*
       | surface deformation gradient                                                  |
       *===============================================================================*/

      // surface deformation gradient in 3 dimensions in global coordinates
      Core::LinAlg::Matrix<noddof_, noddof_> defgrd_glob(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<noddof_, noddof_> defgrd_loc(Core::LinAlg::Initialization::zero);

      // principle stretch in thickness direction
      double lambda3 = cur_thickness_[gp] / thickness_;

      // surface deformation gradient in 3 dimensions in global coordinates
      mem_defgrd_global(dXds1, dXds2, dxds1, dxds2, lambda3, defgrd_glob);

      Core::LinAlg::FourTensorOperations::inverse_tensor_rotation<3>(
          Q_localToGlobal, defgrd_glob, defgrd_loc);

      auto material_local_coordinates =
          std::dynamic_pointer_cast<Mat::MembraneMaterialLocalCoordinates>(
              Core::Elements::Element::material());
      auto material_global_coordinates =
          std::dynamic_pointer_cast<Mat::MembraneMaterialGlobalCoordinates>(
              Core::Elements::Element::material());
      Core::LinAlg::Tensor<double, 3> xi = {{xi_gp, eta_gp, 0.0}};
      Mat::EvaluationContext context{.total_time = total_time,
          .time_step_size = time_step_size,
          .xi = &xi,
          .ref_coords = nullptr};
      if (material_local_coordinates != nullptr)
      {
        material_local_coordinates->update_membrane(
            defgrd_loc, params, context, Q_localToGlobal, gp, id());
      }
      else if (material_global_coordinates != nullptr)
      {
        solid_material()->update(Core::LinAlg::make_tensor(defgrd_glob), gp, params, context, id());
      }
    }
  }

  solid_material()->update();
}

template class Discret::Elements::Membrane<Core::FE::CellType::tri3>;
template class Discret::Elements::Membrane<Core::FE::CellType::tri6>;
template class Discret::Elements::Membrane<Core::FE::CellType::quad4>;
template class Discret::Elements::Membrane<Core::FE::CellType::quad9>;

FOUR_C_NAMESPACE_CLOSE
