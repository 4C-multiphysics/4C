// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_elements_jacobian.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_gauss_point_extrapolation.hpp"
#include "4C_fem_general_utils_gauss_point_postprocess.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mat_constraintmixture.hpp"
#include "4C_mat_elasthyper.hpp"
#include "4C_mat_growthremodel_elasthyper.hpp"
#include "4C_mat_robinson.hpp"
#include "4C_mat_service.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mat_thermoplastichyperelast.hpp"
#include "4C_mat_thermostvenantkirchhoff.hpp"
#include "4C_so3_defines.hpp"
#include "4C_so3_element_service.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_so3_hex8_determinant_analysis.hpp"
#include "4C_so3_prestress.hpp"
#include "4C_so3_prestress_service.hpp"
#include "4C_so3_utils.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_structure_new_gauss_point_data_output_manager.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_SerialDenseSolver.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

using VoigtMapping = Core::LinAlg::Voigt::IndexMappings;

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                              maf 04/07|
 *----------------------------------------------------------------------*/
int Discret::Elements::SoHex8::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // Check whether the solid material post_setup() routine has already been called and call it if
  // not
  ensure_material_post_setup(params);

  set_params_interface_ptr(params);

  Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8> elemat1(elemat1_epetra.values(), true);
  Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8> elemat2(elemat2_epetra.values(), true);
  Core::LinAlg::Matrix<NUMDOF_SOH8, 1> elevec1(elevec1_epetra.values(), true);
  Core::LinAlg::Matrix<NUMDOF_SOH8, 1> elevec2(elevec2_epetra.values(), true);
  Core::LinAlg::Matrix<NUMDOF_SOH8, 1> elevec3(elevec3_epetra.values(), true);

  // start with "none"
  Core::Elements::ActionType act = Core::Elements::none;

  if (is_params_interface())
  {
    act = params_interface().get_action_type();
  }
  else
  {
    // get the required action
    std::string action = params.get<std::string>("action", "none");
    act = Core::Elements::string_to_action_type(action);
  }


  // what should the element do
  switch (act)
  {
    //==================================================================================
    // nonlinear stiffness and internal force vector
    case Core::Elements::struct_calc_nlnstiff:
    case Core::Elements::struct_calc_linstiff:
    {
      // need current displacement and residual forces
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      if (disp == nullptr || res == nullptr)
        FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);
      Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8>* matptr = nullptr;
      if (elemat1.is_initialized()) matptr = &elemat1;

      nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, matptr, nullptr, &elevec1, nullptr,
          &elevec3, nullptr, nullptr, nullptr, params, Inpar::Solid::stress_none,
          Inpar::Solid::strain_none, Inpar::Solid::strain_none);

      break;
    }
    //==================================================================================
    // internal force vector only
    case Core::Elements::struct_calc_internalforce:
    {
      // need current displacement and residual forces
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      if (disp == nullptr || res == nullptr)
        FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);
      // create a dummy element matrix to apply linearised EAS-stuff onto
      Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8> myemat(true);

      nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, &myemat, nullptr, &elevec1, nullptr,
          nullptr, nullptr, nullptr, nullptr, params, Inpar::Solid::stress_none,
          Inpar::Solid::strain_none, Inpar::Solid::strain_none);

      break;
    }
    //==================================================================================
    // nonlinear stiffness, internal force vector, and consistent mass matrix
    case Core::Elements::struct_calc_nlnstiffmass:
    case Core::Elements::struct_calc_nlnstifflmass:
    case Core::Elements::struct_calc_linstiffmass:
    case Core::Elements::struct_calc_internalinertiaforce:
    {
      // need current displacement and residual forces
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      // need current velocities and accelerations (for non constant mass matrix)
      std::shared_ptr<const Core::LinAlg::Vector<double>> vel =
          discretization.get_state("velocity");
      std::shared_ptr<const Core::LinAlg::Vector<double>> acc =
          discretization.get_state("acceleration");
      if (disp == nullptr || res == nullptr)
        FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
      if (vel == nullptr) FOUR_C_THROW("Cannot get state vectors 'velocity'");
      if (acc == nullptr) FOUR_C_THROW("Cannot get state vectors 'acceleration'");

      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myvel(lm.size());
      Core::FE::extract_my_values(*vel, myvel, lm);
      std::vector<double> myacc(lm.size());
      Core::FE::extract_my_values(*acc, myacc, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);

      // This matrix is used in the evaluation functions to store the mass matrix. If the action
      // type is Core::Elements::struct_calc_internalinertiaforce we do not want to actually
      // populate the elemat2 variable, since the inertia terms will be directly added to the right
      // hand side. Therefore, a view is only set in cases where the evaluated mass matrix should
      // also be exported in elemat2.
      Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8> mass_matrix_evaluate;
      if (act != Core::Elements::struct_calc_internalinertiaforce)
        mass_matrix_evaluate.set_view(elemat2);

      if (act == Core::Elements::struct_calc_internalinertiaforce)
      {
        nlnstiffmass(lm, mydisp, &myvel, &myacc, myres, nullptr, &mass_matrix_evaluate, &elevec1,
            &elevec2, nullptr, nullptr, nullptr, nullptr, params, Inpar::Solid::stress_none,
            Inpar::Solid::strain_none, Inpar::Solid::strain_none);
      }
      else  // standard analysis
      {
        nlnstiffmass(lm, mydisp, &myvel, &myacc, myres, &elemat1, &mass_matrix_evaluate, &elevec1,
            &elevec2, &elevec3, nullptr, nullptr, nullptr, params, Inpar::Solid::stress_none,
            Inpar::Solid::strain_none, Inpar::Solid::strain_none);
      }
      if (act == Core::Elements::struct_calc_nlnstifflmass) soh8_lumpmass(&elemat2);

      Inpar::Solid::MassLin mass_lin = Inpar::Solid::MassLin::ml_none;
      auto modelevaluator_data =
          std::dynamic_pointer_cast<Solid::ModelEvaluator::Data>(params_interface_ptr());
      if (modelevaluator_data != nullptr)
        mass_lin = modelevaluator_data->sdyn().get_mass_lin_type();
      if (mass_lin == Inpar::Solid::MassLin::ml_rotations)
      {
        // In case of Lie group time integration, we need to explicitly add the inertia terms to the
        // force vector, as the global mass matrix is never multiplied with the global acceleration
        // vector.
        Core::LinAlg::Matrix<NUMDOF_SOH8, 1> acceleration(true);
        for (unsigned int i_dof = 0; i_dof < NUMDOF_SOH8; i_dof++)
          acceleration(i_dof) = myacc[i_dof];
        Core::LinAlg::Matrix<NUMDOF_SOH8, 1> internal_inertia(true);
        internal_inertia.multiply(mass_matrix_evaluate, acceleration);
        elevec2 += internal_inertia;
      }

      break;
    }
    //==================================================================================
    case Core::Elements::struct_calc_mass_volume:
    {
      // declaration of variables
      double volume_ref = 0.0;
      double volume_mat = 0.0;
      double volume_cur = 0.0;
      double mass_ref = 0.0;
      double mass_mat = 0.0;
      double mass_cur = 0.0;

      // get displacements and extract values of this element
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state displacement vector");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      // reference and current geometry (nodal positions)
      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xref;  // reference coord. of element
      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xcur;  // current  coord. of element

      for (int k = 0; k < NUMNOD_SOH8; ++k)
      {
        xref(k, 0) = nodes()[k]->x()[0];
        xref(k, 1) = nodes()[k]->x()[1];
        xref(k, 2) = nodes()[k]->x()[2];
        xcur(k, 0) = xref(k, 0) + mydisp[k * NODDOF_SOH8 + 0];
        xcur(k, 1) = xref(k, 1) + mydisp[k * NODDOF_SOH8 + 1];
        xcur(k, 2) = xref(k, 2) + mydisp[k * NODDOF_SOH8 + 2];
      }

      // safety check before the actual evaluation starts
      const double min_detJ_curr = soh8_get_min_det_jac_at_corners(xcur);
      if (min_detJ_curr <= 0.0)
      {
        soh8_error_handling(
            min_detJ_curr, params, __LINE__, Solid::Elements::ele_error_determinant_at_corner);
        elevec1_epetra(0) = 0.0;
        return 1;
      }

      const static std::vector<Core::LinAlg::Matrix<NUMNOD_SOH8, 1>> shapefcts = soh8_shapefcts();
      const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs =
          soh8_derivs();
      const static std::vector<double> gpweights = soh8_weights();

      // MAT ------------------------
      // build new jacobian mapping with respect to the material configuration
      int err = 0;

      std::vector<double> detJmat = detJ_;
      std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>> invJmat = invJ_;

      err = init_jacobian_mapping(mydisp);
      if (err)
      {
        // reset class variable before leaving
        init_jacobian_mapping();
        return err;
      }

      std::vector<double> detJcur = detJ_;
      std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>> invJcur = invJ_;

      init_jacobian_mapping();

      std::vector<double> detJref = detJ_;
      std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>> invJref = invJ_;


      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_XYZ;
      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd(true);

      //----------------------------------------------------------------
      // loop over all Gauss points
      //----------------------------------------------------------------
      for (unsigned ip = 0; ip < NUMGPT_SOH8; ++ip)
      {
        const double density = material()->density(ip);
        const double wgt = gpweights[ip];

        /*------------------------------------ integration factor  -------*/
        double fac = wgt * detJref[ip];
        volume_ref += fac;
        fac = wgt * detJref[ip] * density;
        mass_ref += fac;

        // MAT ------------------------
        {
          N_XYZ.multiply(invJref[ip], derivs[ip]);
          defgrd.multiply_tt(xcur, N_XYZ);
          double detFref = defgrd.determinant();

          /*------------------------------------ integration factor  -------*/
          fac = wgt * detJcur[ip];
          volume_cur += fac;
          fac = wgt * detJcur[ip] * density * 1 / detFref;
          mass_cur += fac;
        }
      }

      //----------------------------------------------------------------

      // return results

      volume_mat = volume_ref;
      mass_mat = mass_ref;

      elevec1(0) = volume_ref;
      elevec1(1) = volume_mat;
      elevec1(2) = volume_cur;
      elevec1(3) = mass_ref;
      elevec1(4) = mass_mat;
      elevec1(5) = mass_cur;

      break;
    }

    case Core::Elements::analyse_jacobian_determinant:
    {
      // get displacements and extract values of this element
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state displacement vector");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      std::vector<double> mydispmat(lm.size(), 0.0);
      // reference and current geometry (nodal positions)
      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> xcurr;  // current  coord. of element

      for (int k = 0; k < NUMNOD_SOH8; ++k)
      {
        xcurr(0, k) = nodes()[k]->x()[0] + mydisp[k * NODDOF_SOH8 + 0];
        xcurr(1, k) = nodes()[k]->x()[1] + mydisp[k * NODDOF_SOH8 + 1];
        xcurr(2, k) = nodes()[k]->x()[2] + mydisp[k * NODDOF_SOH8 + 2];
      }

      std::shared_ptr<SoHex8DeterminantAnalysis> det_analyser = SoHex8DeterminantAnalysis::create();
      if (not det_analyser->is_valid(xcurr))
        soh8_error_handling(
            -1.0, params, __LINE__, Solid::Elements::ele_error_determinant_analysis);

      break;
    }

    //==================================================================================
    // recover elementwise stored quantities (e.g. EAS)
    case Core::Elements::struct_calc_recover:
    {
      // need current displacement and residual forces
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");

      if (disp == nullptr || res == nullptr)
        FOUR_C_THROW(
            "Cannot get state vectors \"displacement\" "
            "and/or \"residual displacement\"");

      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);

      soh8_recover(lm, myres);
      /* ToDo Probably we have to recover the history information of some special
       * materials as well.                                 hiermeier 04/2016  */

      break;
    }
    //==================================================================================
    // evaluate stresses and strains at gauss points
    case Core::Elements::struct_calc_stress:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      std::shared_ptr<std::vector<char>> stressdata = nullptr;
      std::shared_ptr<std::vector<char>> straindata = nullptr;
      std::shared_ptr<std::vector<char>> plstraindata = nullptr;
      Inpar::Solid::StressType iostress = Inpar::Solid::stress_none;
      Inpar::Solid::StrainType iostrain = Inpar::Solid::strain_none;
      Inpar::Solid::StrainType ioplstrain = Inpar::Solid::strain_none;
      if (is_params_interface())
      {
        stressdata = str_params_interface().stress_data_ptr();
        straindata = str_params_interface().strain_data_ptr();
        plstraindata = str_params_interface().plastic_strain_data_ptr();

        iostress = str_params_interface().get_stress_output_type();
        iostrain = str_params_interface().get_strain_output_type();
        ioplstrain = str_params_interface().get_plastic_strain_output_type();
      }
      else
      {
        stressdata = params.get<std::shared_ptr<std::vector<char>>>("stress", nullptr);
        straindata = params.get<std::shared_ptr<std::vector<char>>>("strain", nullptr);
        iostress = params.get<Inpar::Solid::StressType>("iostress", Inpar::Solid::stress_none);
        iostrain = params.get<Inpar::Solid::StrainType>("iostrain", Inpar::Solid::strain_none);
        // in case of small strain materials calculate plastic strains for post processing
        plstraindata = params.get<std::shared_ptr<std::vector<char>>>("plstrain", nullptr);
        ioplstrain = params.get<Inpar::Solid::StrainType>("ioplstrain", Inpar::Solid::strain_none);
      }
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      if (stressdata == nullptr) FOUR_C_THROW("Cannot get 'stress' data");
      if (straindata == nullptr) FOUR_C_THROW("Cannot get 'strain' data");
      if (plstraindata == nullptr) FOUR_C_THROW("Cannot get 'plastic strain' data");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);
      Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D> stress;
      Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D> strain;
      Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D> plstrain;

      nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, nullptr, nullptr, nullptr, nullptr, nullptr,
          &stress, &strain, &plstrain, params, iostress, iostrain, ioplstrain);
      {
        Core::Communication::PackBuffer data;

        Core::LinAlg::SerialDenseMatrix stress_view(
            Teuchos::View, stress.values(), NUMGPT_SOH8, NUMGPT_SOH8, Mat::NUM_STRESS_3D);
        add_to_pack(data, stress_view);
        std::copy(data().begin(), data().end(), std::back_inserter(*stressdata));
      }

      {
        Core::Communication::PackBuffer data;

        Core::LinAlg::SerialDenseMatrix strain_view(
            Teuchos::View, strain.values(), NUMGPT_SOH8, NUMGPT_SOH8, Mat::NUM_STRESS_3D);
        add_to_pack(data, strain_view);
        std::copy(data().begin(), data().end(), std::back_inserter(*straindata));
      }

      {
        Core::Communication::PackBuffer data;

        Core::LinAlg::SerialDenseMatrix plstrain_view(
            Teuchos::View, plstrain.values(), NUMGPT_SOH8, NUMGPT_SOH8, Mat::NUM_STRESS_3D);
        add_to_pack(data, plstrain_view);
        std::copy(data().begin(), data().end(), std::back_inserter(*plstraindata));
      }
    }
    break;
    case Core::Elements::struct_init_gauss_point_data_output:
    {
      FOUR_C_ASSERT(is_params_interface(),
          "This action type should only be called from the new time integration framework!");

      // Save number of Gauss of the element for gauss point data output
      str_params_interface()
          .gauss_point_data_output_manager_ptr()
          ->add_element_number_of_gauss_points(NUMGPT_SOH8);

      // holder for output quantity names and their size
      std::unordered_map<std::string, int> quantities_map{};

      // Ask material for the output quantity names and sizes
      solid_material()->register_output_data_names(quantities_map);

      // Add quantities to the Gauss point output data manager (if they do not already exist)
      str_params_interface().gauss_point_data_output_manager_ptr()->merge_quantities(
          quantities_map);
    }
    break;
    case Core::Elements::struct_gauss_point_data_output:
    {
      FOUR_C_ASSERT(is_params_interface(),
          "This action type should only be called from the new time integration framework!");

      // Collection and assembly of gauss point data
      for (const auto& quantity :
          str_params_interface().gauss_point_data_output_manager_ptr()->get_quantities())
      {
        const std::string& quantity_name = quantity.first;
        const int quantity_size = quantity.second;

        // Step 1: Collect the data for each Gauss point for the material
        Core::LinAlg::SerialDenseMatrix gp_data(NUMGPT_SOH8, quantity_size, true);
        bool data_available = solid_material()->evaluate_output_data(quantity_name, gp_data);

        // Step 3: Assemble data based on output type (elecenter, postprocessed to nodes, Gauss
        // point)
        if (data_available)
        {
          switch (str_params_interface().gauss_point_data_output_manager_ptr()->get_output_type())
          {
            case Inpar::Solid::GaussPointDataOutputType::element_center:
            {
              // compute average of the quantities
              std::shared_ptr<Core::LinAlg::MultiVector<double>> global_data =
                  str_params_interface()
                      .gauss_point_data_output_manager_ptr()
                      ->get_element_center_data()
                      .at(quantity_name);
              Core::FE::assemble_averaged_element_values(*global_data, gp_data, *this);
              break;
            }
            case Inpar::Solid::GaussPointDataOutputType::nodes:
            {
              std::shared_ptr<Core::LinAlg::MultiVector<double>> global_data =
                  str_params_interface().gauss_point_data_output_manager_ptr()->get_nodal_data().at(
                      quantity_name);

              Core::LinAlg::Vector<int>& global_nodal_element_count =
                  *str_params_interface()
                       .gauss_point_data_output_manager_ptr()
                       ->get_nodal_data_count()
                       .at(quantity_name);

              static auto gauss_integration = Core::FE::IntegrationPoints3D(
                  Core::FE::num_gauss_points_to_gauss_rule<Core::FE::CellType::hex8>(NUMGPT_SOH8));
              Core::FE::extrapolate_gp_quantity_to_nodes_and_assemble<Core::FE::CellType::hex8>(
                  *this, gp_data, *global_data, false, gauss_integration);
              Discret::Elements::assemble_nodal_element_count(global_nodal_element_count, *this);
              break;
            }
            case Inpar::Solid::GaussPointDataOutputType::gauss_points:
            {
              std::vector<std::shared_ptr<Core::LinAlg::MultiVector<double>>>& global_data =
                  str_params_interface()
                      .gauss_point_data_output_manager_ptr()
                      ->get_gauss_point_data()
                      .at(quantity_name);
              Discret::Elements::assemble_gauss_point_values(global_data, gp_data, *this);
              break;
            }
            case Inpar::Solid::GaussPointDataOutputType::none:
              FOUR_C_THROW(
                  "You specified a Gauss point data output type of none, so you should not end up "
                  "here.");
            default:
              FOUR_C_THROW("Unknown Gauss point data output type.");
          }
        }
      }
    }
    break;
    //==================================================================================
    case Core::Elements::struct_calc_eleload:
      FOUR_C_THROW("this method is not supposed to evaluate a load, use evaluate_neumann(...)");
      break;
    //==================================================================================
    case Core::Elements::struct_calc_fsiload:
      FOUR_C_THROW("Case not yet implemented");
      break;
    //==================================================================================
    case Core::Elements::struct_calc_update_istep:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      update_element(mydisp, params, *material());
    }
    break;
    //==================================================================================
    case Core::Elements::struct_calc_reset_istep:
    {
      // restore EAS parameters
      if (eastype_ != soh8_easnone)
      {
        soh8_easrestore();

        // reset EAS internal force
        Core::LinAlg::SerialDenseMatrix* oldfeas = &easdata_.feas;
        oldfeas->putScalar(0.0);
      }
      // Reset of history (if needed)
      solid_material()->reset_step();
    }
    break;
    //==================================================================================
    case Core::Elements::struct_calc_energy:
    {
      // initialization of internal energy
      double intenergy = 0.0;

      // shape functions and Gauss weights
      const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs =
          soh8_derivs();
      const static std::vector<double> weights = soh8_weights();

      // get displacements of this processor
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state displacement vector");

      // get displacements of this element
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      // update element geometry
      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe;  // material coord. of element
      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xcurr;  // current  coord. of element
      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xdisp;

      Utils::evaluate_nodal_coordinates<Core::FE::CellType::hex8, 3>(nodes(), xrefe);
      Utils::evaluate_nodal_displacements<Core::FE::CellType::hex8, 3>(mydisp, xdisp);
      Utils::evaluate_current_nodal_coordinates<Core::FE::CellType::hex8, 3>(xrefe, xdisp, xcurr);

      // safety check before the actual evaluation starts
      const double min_detJ_curr = soh8_get_min_det_jac_at_corners(xcurr);
      if (min_detJ_curr <= 0.0)
      {
        soh8_error_handling(
            min_detJ_curr, params, __LINE__, Solid::Elements::ele_error_determinant_at_corner);
        elevec1_epetra(0) = 0.0;
        return 0;
      }

      // prepare EAS data
      Core::LinAlg::SerialDenseMatrix* alpha = nullptr;              // EAS alphas
      std::vector<Core::LinAlg::SerialDenseMatrix>* M_GP = nullptr;  // EAS matrix M at all GPs
      Core::LinAlg::SerialDenseMatrix M;                             // EAS matrix M at current GP
      double detJ0;                                                  // detJ(origin)
      Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> T0invT;  // trafo matrix
      if (eastype_ != soh8_easnone)
      {
        alpha = &easdata_.alpha;  // get alpha of previous iteration
        soh8_eassetup(&M_GP, detJ0, T0invT, xrefe);
      }

      // loop over all Gauss points
      for (unsigned gp = 0; gp < NUMGPT_SOH8; gp++)
      {
        // Gauss weights and Jacobian determinant
        double fac = detJ_[gp] * weights[gp];

        /* get the inverse of the Jacobian matrix which looks like:
        **            [ x_,r  y_,r  z_,r ]^-1
        **     J^-1 = [ x_,s  y_,s  z_,s ]
        **            [ x_,t  y_,t  z_,t ]
        */
        // compute derivatives N_XYZ at gp w.r.t. material coordinates
        // by N_XYZ = J^-1 * N_rst
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_XYZ(true);
        N_XYZ.multiply(invJ_[gp], derivs[gp]);

        // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd(true);

        // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
        // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> glstrain(true);

        if (Prestress::is_mulf(pstype_))
        {
          // get Jacobian mapping wrt to the stored configuration
          Core::LinAlg::Matrix<3, 3> invJdef;
          prestress_->storageto_matrix(gp, invJdef, prestress_->j_history());
          // get derivatives wrt to last spatial configuration
          Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_xyz;
          N_xyz.multiply(invJdef, derivs[gp]);

          // build multiplicative incremental defgrd
          defgrd.multiply_tt(xdisp, N_xyz);
          defgrd(0, 0) += 1.0;
          defgrd(1, 1) += 1.0;
          defgrd(2, 2) += 1.0;

          // get stored old incremental F
          Core::LinAlg::Matrix<3, 3> Fhist;
          prestress_->storageto_matrix(gp, Fhist, prestress_->f_history());

          // build total defgrd = delta F * F_old
          Core::LinAlg::Matrix<3, 3> Fnew;
          Fnew.multiply(defgrd, Fhist);
          defgrd = Fnew;

          // right Cauchy-Green tensor = F^T * F
          Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> cauchygreen;
          cauchygreen.multiply_tn(defgrd, defgrd);

          glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
          glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
          glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
          glstrain(3) = cauchygreen(0, 1);
          glstrain(4) = cauchygreen(1, 2);
          glstrain(5) = cauchygreen(2, 0);
        }
        else if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
        {
          // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
          defgrd.multiply_tt(xcurr, N_XYZ);

          // right Cauchy-Green tensor = F^T * F
          Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> cauchygreen;
          cauchygreen.multiply_tn(defgrd, defgrd);

          glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
          glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
          glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
          glstrain(3) = cauchygreen(0, 1);
          glstrain(4) = cauchygreen(1, 2);
          glstrain(5) = cauchygreen(2, 0);
        }
        else if (kintype_ == Inpar::Solid::KinemType::linear)
        {
          // in kinematically linear analysis the deformation gradient is equal to identity
          // no difference between reference and current state
          for (int i = 0; i < 3; ++i) defgrd(i, i) = 1.0;

          // nodal displacement vector
          Core::LinAlg::Matrix<NUMDOF_SOH8, 1> nodaldisp;
          for (int i = 0; i < NUMDOF_SOH8; ++i) nodaldisp(i, 0) = mydisp[i];
          // compute linear B-operator
          Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, NUMDOF_SOH8> bop;
          for (int i = 0; i < NUMNOD_SOH8; ++i)
          {
            bop(0, NODDOF_SOH8 * i + 0) = N_XYZ(0, i);
            bop(0, NODDOF_SOH8 * i + 1) = 0.0;
            bop(0, NODDOF_SOH8 * i + 2) = 0.0;
            bop(1, NODDOF_SOH8 * i + 0) = 0.0;
            bop(1, NODDOF_SOH8 * i + 1) = N_XYZ(1, i);
            bop(1, NODDOF_SOH8 * i + 2) = 0.0;
            bop(2, NODDOF_SOH8 * i + 0) = 0.0;
            bop(2, NODDOF_SOH8 * i + 1) = 0.0;
            bop(2, NODDOF_SOH8 * i + 2) = N_XYZ(2, i);

            bop(3, NODDOF_SOH8 * i + 0) = N_XYZ(1, i);
            bop(3, NODDOF_SOH8 * i + 1) = N_XYZ(0, i);
            bop(3, NODDOF_SOH8 * i + 2) = 0.0;
            bop(4, NODDOF_SOH8 * i + 0) = 0.0;
            bop(4, NODDOF_SOH8 * i + 1) = N_XYZ(2, i);
            bop(4, NODDOF_SOH8 * i + 2) = N_XYZ(1, i);
            bop(5, NODDOF_SOH8 * i + 0) = N_XYZ(2, i);
            bop(5, NODDOF_SOH8 * i + 1) = 0.0;
            bop(5, NODDOF_SOH8 * i + 2) = N_XYZ(0, i);
          }

          // compute linear strain at GP
          glstrain.multiply(bop, nodaldisp);
        }
        else
          FOUR_C_THROW("unknown kinematic type for energy calculation");

        // EAS technology: "enhance the strains"  ----------------------------- EAS
        if (eastype_ != soh8_easnone)
        {
          M.shape(Mat::NUM_STRESS_3D, neas_);
          // map local M to global, also enhancement is referred to element origin
          // M = detJ0/detJ T0^{-T} . M
          // Core::LinAlg::SerialDenseMatrix Mtemp(M); // temp M for Matrix-Matrix-Product
          // add enhanced strains = M . alpha to GL strains to "unlock" element
          switch (eastype_)
          {
            case Discret::Elements::SoHex8::soh8_easfull:
              Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
                  soh8_easfull>(
                  M.values(), detJ0 / detJ_[gp], T0invT.data(), (M_GP->at(gp)).values());
              Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, soh8_easfull, 1>(
                  1.0, glstrain.data(), 1.0, M.values(), alpha->values());
              break;
            case Discret::Elements::SoHex8::soh8_easmild:
              Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
                  soh8_easmild>(
                  M.values(), detJ0 / detJ_[gp], T0invT.data(), (M_GP->at(gp)).values());
              Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, soh8_easmild, 1>(
                  1.0, glstrain.data(), 1.0, M.values(), alpha->values());
              break;
            case Discret::Elements::SoHex8::soh8_eassosh8:
              Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
                  soh8_eassosh8>(
                  M.values(), detJ0 / detJ_[gp], T0invT.data(), (M_GP->at(gp)).values());
              Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, soh8_eassosh8, 1>(
                  1.0, glstrain.data(), 1.0, M.values(), alpha->values());
              break;
            case Discret::Elements::SoHex8::soh8_easnone:
              break;
            default:
              FOUR_C_THROW("Don't know what to do with EAS type %d", eastype_);
              break;
          }
        }  // ------------------------------------------------------------------ EAS

        if (defgrd.determinant() <= 0.0)
        {
          if (is_params_interface() and str_params_interface().is_tolerate_errors())
          {
            str_params_interface().set_ele_eval_error_flag(
                Solid::Elements::ele_error_negative_det_of_def_gradient);
            return 0;
          }
          else
          {
            FOUR_C_THROW("Negative deformation gradient!");
          }
        }

        // call material for evaluation of strain energy function
        double psi = 0.0;
        solid_material()->strain_energy(glstrain, psi, gp, id());

        // sum up GP contribution to internal energy
        intenergy += fac * psi;
      }

      if (is_params_interface())  // new structural time integration
      {
        str_params_interface().add_contribution_to_energy_type(intenergy, Solid::internal_energy);
      }
      else  // old structural time integration
      {
        // check length of elevec1
        if (elevec1_epetra.length() < 1) FOUR_C_THROW("The given result vector is too short.");

        elevec1_epetra(0) = intenergy;
      }
    }
    break;
    //==================================================================================
    case Core::Elements::multi_calc_dens:
    {
      soh8_homog(params);
    }
    break;
      //==================================================================================
    case Core::Elements::struct_interpolate_velocity_to_point:
    {
      // get displacements and extract values of this element (set in prepare_fluid_op())
      std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp =
          discretization.get_state("displacement");
      std::shared_ptr<const Core::LinAlg::Vector<double>> velnp =
          discretization.get_state("velocity");

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (dispnp == nullptr) FOUR_C_THROW("Cannot get state displacement vector");
      if (velnp == nullptr) FOUR_C_THROW("Cannot get state velocity vector");
#endif

      std::vector<double> mydispnp(lm.size());
      Core::FE::extract_my_values(*dispnp, mydispnp, lm);

      std::vector<double> myvelnp(lm.size());
      Core::FE::extract_my_values(*velnp, myvelnp, lm);

      // update element geometry
      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe;  // reference coord. of element
      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8>
          xdisp;  // current displacements of element nodes
      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xcurr;  // current coord. of element
      Utils::evaluate_nodal_coordinates<Core::FE::CellType::hex8, 3>(nodes(), xrefe);
      Utils::evaluate_nodal_displacements<Core::FE::CellType::hex8, 3>(mydispnp, xdisp);
      Utils::evaluate_current_nodal_coordinates<Core::FE::CellType::hex8, 3>(xrefe, xdisp, xcurr);

      // shape functions and derivatives w.r.t. r,s,t
      Core::LinAlg::Matrix<NUMNOD_SOH8, 1> shapefcts;
      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> deriv1;
      // coordinates of given point in reference coordinates
      Core::LinAlg::Matrix<NUMDIM_SOH8, 1> xsi;
      xsi(0) = elevec2_epetra(0);
      xsi(1) = elevec2_epetra(1);
      xsi(2) = elevec2_epetra(2);
      // evaluate shape functions and derivatives at given point w.r.t r,s,t
      Core::FE::shape_function<Core::FE::CellType::hex8>(xsi, shapefcts);
      Core::FE::shape_function_deriv1<Core::FE::CellType::hex8>(xsi, deriv1);

      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> myvelocitynp;
      for (int node = 0; node < NUMNOD_SOH8; ++node)
      {
        for (int dim = 0; dim < NUMDIM_SOH8; ++dim)
        {
          myvelocitynp(node, dim) = myvelnp[node * 3 + dim];
        }
      }

      if (params.get("calculate_velocity", 1))
      {
        //************************************************************************
        // 1.) interpolation of velocity at n+1 to given point
        //************************************************************************

        // give back velocity at given point
        Core::LinAlg::Matrix<3, 1> result;
        result.multiply_tn(myvelocitynp, shapefcts);
        for (int i = 0; i < 3; ++i) elevec1_epetra(i) = result(i, 0);
      }
      else
      {
        //************************************************************************
        // 2.) calculation of divergence of structural velocity
        //************************************************************************

        // get Jacobian matrix
        // actually compute its transpose....
        /*
                 +-            -+ -1
                 | dx   dy   dz |
                 | --   --   -- |
                 | dr   dr   dr |
                 |              |
                 | dx   dy   dz |
        J^{-T} = | --   --   -- |
                 | ds   ds   ds |
                 |              |
                 | dx   dy   dz |
                 | --   --   -- |
                 | dt   dt   dt |
                 +-            -+
        */

        // global first derivatives of shape functions w.r.t x,y,z (by derxy1 = J^-T * deriv1)
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> derxy1;
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> transJ;
        transJ.multiply(deriv1, xrefe);
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invJ(transJ);
        invJ.invert();
        derxy1.multiply(invJ, deriv1);

        // build (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * derxy1^T
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrdnp(true);
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrdnp_inv(true);
        defgrdnp.multiply_tt(xcurr, derxy1);
        defgrdnp_inv.invert(defgrdnp);

        // evaluate divergence of structural velocity
        double velocitydivergence = 0.0;

        // div(v) = derxy1_A,I (dX_I/dx_i) v_Ai
        for (int I = 0; I < NUMDIM_SOH8; ++I)
        {
          for (int A = 0; A < NUMNOD_SOH8; ++A)
          {
            for (int i = 0; i < NUMDIM_SOH8; ++i)
            {
              velocitydivergence += derxy1(I, A) * (defgrdnp_inv(I, i) * myvelocitynp(A, i));
            }
          }
        }

        // vector to fill
        elevec1_epetra(3) = velocitydivergence;
      }
    }
    break;
    //==================================================================================
    // in case of multi-scale problems, possible EAS internal data on microscale
    // have to be stored in every macroscopic Gauss point
    // allocation and initialization of these data arrays can only be
    // done in the elements that know the number of EAS parameters
    case Core::Elements::multi_init_eas:
    {
      soh8_eas_init_multi(params);
    }
    break;
    //==================================================================================
    // in case of multi-scale problems, possible EAS internal data on microscale
    // have to be stored in every macroscopic Gauss point
    // before any microscale simulation, EAS internal data has to be
    // set accordingly
    case Core::Elements::multi_set_eas:
    {
      soh8_set_eas_multi(params);
    }
    break;
    //==================================================================================
    // read restart of microscale
    case Core::Elements::multi_readrestart:
    {
      soh8_read_restart_multi();
    }
    break;
    //==================================================================================
    case Core::Elements::struct_update_prestress:
    {
      time_ = params.get<double>("total time");
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get displacement state");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      switch (pstype_)
      {
        case Inpar::Solid::PreStress::mulf:
        {
          // build def gradient for every gauss point
          Core::LinAlg::SerialDenseMatrix gpdefgrd(NUMGPT_SOH8, 9);
          def_gradient(mydisp, gpdefgrd, *prestress_);

          // update deformation gradient and put back to storage
          Core::LinAlg::Matrix<3, 3> deltaF;
          Core::LinAlg::Matrix<3, 3> Fhist;
          Core::LinAlg::Matrix<3, 3> Fnew;
          for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
          {
            prestress_->storageto_matrix(gp, deltaF, gpdefgrd);
            prestress_->storageto_matrix(gp, Fhist, prestress_->f_history());
            Fnew.multiply(deltaF, Fhist);
            prestress_->matrixto_storage(gp, Fnew, prestress_->f_history());
          }

          // push-forward invJ for every gaussian point
          update_jacobian_mapping(mydisp, *prestress_);

          // Update constraintmixture material
          if (material()->material_type() == Core::Materials::m_constraintmixture)
          {
            solid_material()->update();
          }
          break;
        }
        default:
          FOUR_C_THROW(
              "You should either not be here, or the prestressing method you are using is not "
              "implemented for HEX8 elements!");
      }
    }
    break;

    //==================================================================================
    // evaluate stresses and strains at gauss points and store gpstresses in map <EleId, gpstresses
    // >
    case Core::Elements::struct_calc_global_gpstresses_map:
    {
      // nothing to do for ghost elements
      if (Core::Communication::my_mpi_rank(discretization.get_comm()) == owner())
      {
        std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
            discretization.get_state("displacement");
        std::shared_ptr<const Core::LinAlg::Vector<double>> res =
            discretization.get_state("residual displacement");
        std::shared_ptr<std::vector<char>> stressdata =
            params.get<std::shared_ptr<std::vector<char>>>("stress", nullptr);
        std::shared_ptr<std::vector<char>> straindata =
            params.get<std::shared_ptr<std::vector<char>>>("strain", nullptr);
        std::shared_ptr<std::vector<char>> plstraindata =
            params.get<std::shared_ptr<std::vector<char>>>("plstrain", nullptr);
        if (disp == nullptr) FOUR_C_THROW("Cannot get state vectors 'displacement'");
        if (stressdata == nullptr) FOUR_C_THROW("Cannot get 'stress' data");
        if (straindata == nullptr) FOUR_C_THROW("Cannot get 'strain' data");
        if (plstraindata == nullptr) FOUR_C_THROW("Cannot get 'plastic strain' data");
        const std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>
            gpstressmap = params.get<
                std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>>(
                "gpstressmap", nullptr);
        if (gpstressmap == nullptr)
          FOUR_C_THROW("no gp stress map available for writing gpstresses");
        const std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>
            gpstrainmap = params.get<
                std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>>>(
                "gpstrainmap", nullptr);
        if (gpstrainmap == nullptr)
          FOUR_C_THROW("no gp strain map available for writing gpstrains");
        std::vector<double> mydisp(lm.size());
        Core::FE::extract_my_values(*disp, mydisp, lm);
        std::vector<double> myres(lm.size());
        Core::FE::extract_my_values(*res, myres, lm);

        Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D> stress;
        Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D> strain;
        Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D> plstrain;
        auto iostress = params.get<Inpar::Solid::StressType>("iostress", Inpar::Solid::stress_none);
        auto iostrain = params.get<Inpar::Solid::StrainType>("iostrain", Inpar::Solid::strain_none);
        auto ioplstrain =
            params.get<Inpar::Solid::StrainType>("ioplstrain", Inpar::Solid::strain_none);

        nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, nullptr, nullptr, nullptr, nullptr,
            nullptr, &stress, &strain, &plstrain, params, iostress, iostrain, ioplstrain);

        // add stresses to global map
        // get EleID Id()
        int gid = id();
        std::shared_ptr<Core::LinAlg::SerialDenseMatrix> gpstress =
            std::make_shared<Core::LinAlg::SerialDenseMatrix>();
        gpstress->shape(NUMGPT_SOH8, Mat::NUM_STRESS_3D);

        // move stresses to serial dense matrix
        for (unsigned i = 0; i < NUMGPT_SOH8; i++)
        {
          for (int j = 0; j < Mat::NUM_STRESS_3D; j++)
          {
            (*gpstress)(i, j) = stress(i, j);
          }
        }

        // strains
        std::shared_ptr<Core::LinAlg::SerialDenseMatrix> gpstrain =
            std::make_shared<Core::LinAlg::SerialDenseMatrix>();
        gpstrain->shape(NUMGPT_SOH8, Mat::NUM_STRESS_3D);

        // move stresses to serial dense matrix
        for (unsigned i = 0; i < NUMGPT_SOH8; i++)
        {
          for (int j = 0; j < Mat::NUM_STRESS_3D; j++)
          {
            (*gpstrain)(i, j) = strain(i, j);
          }
        }

        // add to map
        (*gpstressmap)[gid] = gpstress;
        (*gpstrainmap)[gid] = gpstrain;

        {
          Core::Communication::PackBuffer data;

          add_to_pack(data, stress);
          std::copy(data().begin(), data().end(), std::back_inserter(*stressdata));
        }

        {
          Core::Communication::PackBuffer data;

          add_to_pack(data, strain);
          std::copy(data().begin(), data().end(), std::back_inserter(*straindata));
        }

        {
          Core::Communication::PackBuffer data;
          add_to_pack(data, plstrain);
          std::copy(data().begin(), data().end(), std::back_inserter(*plstraindata));
        }
      }
    }
    break;
    //==================================================================================
    case Core::Elements::struct_calc_predict:
    {
      // do nothing here
      break;
    }
    //==================================================================================
    // create a backup state for all internally stored variables (e.g. EAS)
    case Core::Elements::struct_create_backup:
    {
      std::shared_ptr<const Core::LinAlg::Vector<double>> res =
          discretization.get_state("residual displacement");
      if (!res) FOUR_C_THROW("Cannot get state vector \"residual displacement\"");

      // extract the part for this element
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);

      soh8_create_eas_backup_state(myres);

      break;
    }
    //==================================================================================
    /* recover internally stored state variables from a previously created backup
     * state (e.g. EAS) */
    case Core::Elements::struct_recover_from_backup:
    {
      soh8_recover_from_eas_backup_state();

      break;
    }
    default:
      FOUR_C_THROW("Unknown type of action for So_hex8: %s",
          Core::Elements::action_type_to_string(act).c_str());
      break;
  }
  return 0;
}


/*----------------------------------------------------------------------*
 |  Integrate a Volume Neumann boundary condition (public)     maf 04/07|
 *----------------------------------------------------------------------*/
int Discret::Elements::SoHex8::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  set_params_interface_ptr(params);
  // get values and switches from the condition
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition.parameters().get<std::vector<double>>("VAL");

  /*
  **    TIME CURVE BUSINESS
  */
  // find out whether we will use a time curve
  double time = -1.0;
  if (is_params_interface())
    time = params_interface().get_total_time();
  else
    time = params.get("total time", -1.0);

  // ensure that at least as many curves/functs as dofs are available
  if (int(onoff.size()) < NUMDIM_SOH8)
    FOUR_C_THROW("Fewer functions or curves defined than the element has dofs.");

  for (int checkdof = NUMDIM_SOH8; checkdof < int(onoff.size()); ++checkdof)
  {
    if (onoff[checkdof] != 0)
      FOUR_C_THROW(
          "Number of Dimensions in Neumann_Evaluation is 3. Further DoFs are not considered.");
  }

  // (SPATIAL) FUNCTION BUSINESS
  const auto funct = condition.parameters().get<std::vector<Core::IO::Noneable<int>>>("FUNCT");
  Core::LinAlg::Matrix<NUMDIM_SOH8, 1> xrefegp(false);

  /* ============================================================================*
  ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for HEX_8 with 8 GAUSS POINTS*
  ** ============================================================================*/
  const static std::vector<Core::LinAlg::Matrix<NUMNOD_SOH8, 1>> shapefcts = soh8_shapefcts();
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs = soh8_derivs();
  const static std::vector<double> gpweights = soh8_weights();
  /* ============================================================================*/

  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe;  // material coord. of element
  Utils::evaluate_nodal_coordinates<Core::FE::CellType::hex8, 3>(nodes(), xrefe);
  /* ================================================= Loop over Gauss Points */
  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
  {
    // compute the Jacobian matrix
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> jac;
    jac.multiply(derivs[gp], xrefe);

    // compute determinant of Jacobian
    const double detJ = jac.determinant();
    if (detJ == 0.0)
      FOUR_C_THROW("ZERO JACOBIAN DETERMINANT");
    else if (detJ < 0.0)
      FOUR_C_THROW("NEGATIVE JACOBIAN DETERMINANT");

    // material/reference co-ordinates of Gauss point
    for (int dim = 0; dim < NUMDIM_SOH8; dim++)
    {
      xrefegp(dim) = 0.0;
      for (int nodid = 0; nodid < NUMNOD_SOH8; ++nodid)
        xrefegp(dim) += shapefcts[gp](nodid) * xrefe(nodid, dim);
    }

    // integration factor
    const double fac = gpweights[gp] * detJ;
    // distribute/add over element load vector
    for (int dim = 0; dim < NUMDIM_SOH8; dim++)
    {
      if (onoff[dim])
      {
        // function evaluation
        double functfac = 1.0;
        if (funct[dim].has_value() && funct[dim].value() > 0)
        {
          functfac = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funct[dim].value() - 1)
                         .evaluate(xrefegp.data(), time, dim);
        }

        const double dim_fac = val[dim] * fac * functfac;
        for (int nodid = 0; nodid < NUMNOD_SOH8; ++nodid)
        {
          elevec1[nodid * NUMDIM_SOH8 + dim] += shapefcts[gp](nodid) * dim_fac;
        }
      }
    }

  } /* ==================================================== end of Loop over GP */

  return 0;
}  // Discret::Elements::So_hex8::evaluate_neumann

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const double* Discret::Elements::SoHex8::soh8_get_coordinate_of_gausspoints(
    const unsigned dim) const
{
  static Core::LinAlg::Matrix<NUMGPT_SOH8, NUMDIM_SOH8> coordinates_of_gps(false);
  static bool init = false;

  if (not init)
  {
    if (gp_rule_.num_points() != NUMGPT_SOH8)
      FOUR_C_THROW(
          "Inconsistent number of GPs: "
          "%d != %d",
          gp_rule_.num_points(), NUMGPT_SOH8);

    for (unsigned gp = 0; gp < gp_rule_.num_points(); ++gp)
      for (unsigned d = 0; d < NUMDIM_SOH8; ++d) coordinates_of_gps(gp, d) = gp_rule_.point(gp)[d];
    // do it only once
    init = true;
  }

  return &coordinates_of_gps(0, dim);
}

/*----------------------------------------------------------------------*
 |  init the element jacobian mapping (protected)              gee 04/08|
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::init_jacobian_mapping()
{
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs = soh8_derivs();
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe;
  for (int i = 0; i < NUMNOD_SOH8; ++i)
  {
    xrefe(i, 0) = nodes()[i]->x()[0];
    xrefe(i, 1) = nodes()[i]->x()[1];
    xrefe(i, 2) = nodes()[i]->x()[2];
  }
  invJ_.resize(NUMGPT_SOH8);
  detJ_.resize(NUMGPT_SOH8);
  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
  {
    // invJ_[gp].Shape(NUMDIM_SOH8,NUMDIM_SOH8);
    invJ_[gp].multiply(derivs[gp], xrefe);
    detJ_[gp] = invJ_[gp].invert();
    if (detJ_[gp] <= 0.0) FOUR_C_THROW("Element Jacobian mapping %10.5e <= 0.0", detJ_[gp]);

    if (Prestress::is_mulf_active(time_, pstype_, pstime_))
    {
      if (!(prestress_->is_init()))
        prestress_->matrixto_storage(gp, invJ_[gp], prestress_->j_history());
    }
  }

  if (Prestress::is_mulf_active(time_, pstype_, pstime_)) prestress_->is_init() = true;
}
/*----------------------------------------------------------------------*
 |  init the element jacobian mapping with respect to the    farah 06/13|
 |  material configuration.                                             |
 *----------------------------------------------------------------------*/
int Discret::Elements::SoHex8::init_jacobian_mapping(std::vector<double>& dispmat)
{
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs = soh8_derivs();
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xmat;

  for (int i = 0; i < NUMNOD_SOH8; ++i)
  {
    xmat(i, 0) = nodes()[i]->x()[0] + dispmat[i * NODDOF_SOH8 + 0];
    xmat(i, 1) = nodes()[i]->x()[1] + dispmat[i * NODDOF_SOH8 + 1];
    xmat(i, 2) = nodes()[i]->x()[2] + dispmat[i * NODDOF_SOH8 + 2];
  }
  invJ_.clear();
  detJ_.clear();
  invJ_.resize(NUMGPT_SOH8);
  detJ_.resize(NUMGPT_SOH8);
  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
  {
    // invJ_[gp].Shape(NUMDIM_SOH8,NUMDIM_SOH8);
    invJ_[gp].multiply(derivs[gp], xmat);
    detJ_[gp] = invJ_[gp].invert();

    if (detJ_[gp] <= 0.0)
    {
      if (is_params_interface() and str_params_interface().is_tolerate_errors())
      {
        str_params_interface().set_ele_eval_error_flag(
            Solid::Elements::ele_error_negative_det_of_def_gradient);
        return 1;
      }
      else
        FOUR_C_THROW("Element Jacobian mapping %10.5e <= 0.0", detJ_[gp]);
    }
  }

  return 0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Discret::Elements::SoHex8::soh8_get_min_det_jac_at_corners(
    const Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8>& xcurr) const
{
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> xcurr_t(false);
  xcurr_t.update_t(xcurr);
  return Core::Elements::get_minimal_jac_determinant_at_nodes<Core::FE::CellType::hex8>(xcurr_t);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Discret::Elements::SoHex8::soh8_error_handling(const double& det_curr,
    Teuchos::ParameterList& params, const int line_id, const Solid::Elements::EvalErrorFlag flag)
{
  error_handling(det_curr, params, line_id, flag);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::soh8_compute_eas_inc(
    const std::vector<double>& residual, Core::LinAlg::SerialDenseMatrix* const eas_inc)
{
  auto* oldKaainv = &easdata_.invKaa;
  auto* oldKda = &easdata_.Kda;
  auto* oldfeas = &easdata_.feas;
  if (!oldKaainv || !oldKda || !oldfeas) FOUR_C_THROW("Missing EAS history data");

  // we need the (residual) displacement at the previous step
  Core::LinAlg::SerialDenseVector res_d_eas(NUMDOF_SOH8);
  for (int i = 0; i < NUMDOF_SOH8; ++i) res_d_eas(i) = residual[i];
  // --- EAS default update ---------------------------
  Core::LinAlg::SerialDenseMatrix eashelp(neas_, 1);
  /*----------- make multiplication eashelp = oldLt * disp_incr[kstep] */
  Core::LinAlg::multiply(eashelp, *oldKda, res_d_eas);
  /*---------------------------------------- add old Rtilde to eashelp */
  eashelp += *oldfeas;
  /*--------- make multiplication alpha_inc = - old Dtildinv * eashelp */
  Core::LinAlg::multiply(*eas_inc, *oldKaainv, eashelp);
  eas_inc->scale(-1.0);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::soh8_recover(
    const std::vector<int>& lm, const std::vector<double>& residual)
{
  // for eas
  Core::LinAlg::SerialDenseMatrix* alpha = nullptr;
  Core::LinAlg::SerialDenseMatrix* eas_inc = nullptr;
  // get access to the interface parameters
  const double step_length = str_params_interface().get_step_length();
  const bool iseas = (eastype_ != soh8_easnone);

  // have eas?
  if (iseas)
  {
    // access general eas history stuff stored in element
    // get alpha of previous iteration
    alpha = &easdata_.alpha;
    // get the old eas increment
    eas_inc = &easdata_.eas_inc;
    if (!alpha || !eas_inc) FOUR_C_THROW("Missing EAS history data (eas_inc and/or alpha)");
  }

  /* if it is a default step, we have to recover the condensed
   * solution vectors */
  if (str_params_interface().is_default_step())
  {
    /* recovery of the enhanced assumed strain increment and
     * update of the eas dofs. */
    if (iseas)
    {
      // first, store the eas state of the previous accepted Newton step
      str_params_interface().sum_into_my_previous_sol_norm(
          NOX::Nln::StatusTest::quantity_eas, neas_, (*alpha)[0], owner());

      // compute the eas increment
      soh8_compute_eas_inc(residual, eas_inc);

      /*--------------------------- update alpha += step_length * alfa_inc */
      for (int i = 0; i < neas_; ++i) (*alpha)(i, 0) += step_length * (*eas_inc)(i, 0);
    }  // if (iseas)
  }  // if (*isdefault_step_ptr_)
  /* if it is no default step, we can correct the update and the current eas
   * state without the need for any matrix-vector products. */
  else
  {
    // The first step has to be a default step!
    if (old_step_length_ < 0.0) FOUR_C_THROW("The old step length was not defined!");
    /* if this is no full step, we have to adjust the length of the
     * enhanced assumed strain incremental step. */
    if (iseas)
    {
      /* undo the previous step:
       *            alpha_new = alpha_old - old_step * alpha_inc
       * and update the solution variable with the new step length:
       *            alpha_new = alpha_new + new_step * alpha_inc */
      for (int i = 0; i < neas_; ++i)
        (*alpha)(i, 0) += (step_length - old_step_length_) * (*eas_inc)(i, 0);

      //      {
      //        std::cout << "EAS #" << Id() << ":\n";
      //        alpha->print( std::cout );
      //        eas_inc->print( std::cout );
      //        std::cout << "\n";
      //      }
    }  // if (nhyb_)
  }  // else

  // save the old step length
  old_step_length_ = step_length;

  // Check if the eas incr is tested and if yes, calculate the element
  // contribution to the norm
  if (iseas)
    str_params_interface().sum_into_my_update_norm(NOX::Nln::StatusTest::quantity_eas, neas_,
        (*eas_inc)[0], (*alpha)[0], step_length, owner());

  // the element internal stuff should be up-to-date for now...
  return;
}



/*----------------------------------------------------------------------*
 |  evaluate the element (private)                             maf 04/07|
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::nlnstiffmass(std::vector<int>& lm,  // location matrix
    std::vector<double>& disp,                                      // current displacements
    std::vector<double>* vel,                                       // current velocities
    std::vector<double>* acc,                                       // current accelerations
    std::vector<double>& residual,                                  // current residual displ
    Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8>* stiffmatrix,    // element stiffness matrix
    Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8>* massmatrix,     // element mass matrix
    Core::LinAlg::Matrix<NUMDOF_SOH8, 1>* force,                    // element internal force vector
    Core::LinAlg::Matrix<NUMDOF_SOH8, 1>* forceinert,               // element inertial force vector
    Core::LinAlg::Matrix<NUMDOF_SOH8, 1>* force_str,  // element structural force vector
    Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D>* elestress,    // stresses at GP
    Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D>* elestrain,    // strains at GP
    Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D>* eleplstrain,  // plastic strains at GP
    Teuchos::ParameterList& params,             // algorithmic parameters e.g. time
    const Inpar::Solid::StressType iostress,    // stress output option
    const Inpar::Solid::StrainType iostrain,    // strain output option
    const Inpar::Solid::StrainType ioplstrain)  // plastic strain output option
{
  /* ============================================================================*
  ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for HEX_8 with 8 GAUSS POINTS*
  ** ============================================================================*/
  const static std::vector<Core::LinAlg::Matrix<NUMNOD_SOH8, 1>> shapefcts = soh8_shapefcts();
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs = soh8_derivs();
  const static std::vector<double> gpweights = soh8_weights();
  /* ============================================================================*/

  // check for prestressing
  if (Prestress::is_any(pstype_) && eastype_ != soh8_easnone)
    FOUR_C_THROW("No way you can do mulf or id prestressing with EAS turned on!");

  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe(false);  // reference coord. of element
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xcurr(false);  // current  coord. of element
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xdisp(false);

  Utils::evaluate_nodal_coordinates<Core::FE::CellType::hex8, 3>(nodes(), xrefe);
  Utils::evaluate_nodal_displacements<Core::FE::CellType::hex8, 3>(disp, xdisp);
  Utils::evaluate_current_nodal_coordinates<Core::FE::CellType::hex8, 3>(xrefe, xdisp, xcurr);

  // safety check before the actual evaluation starts
  const double min_detJ_curr = soh8_get_min_det_jac_at_corners(xcurr);
  if (min_detJ_curr <= 0.0)
  {
    soh8_error_handling(
        min_detJ_curr, params, __LINE__, Solid::Elements::ele_error_determinant_at_corner);
    return;
  }

  double elediagonallength = 0.0;
  if (!analyticalmaterialtangent_)
    elediagonallength = sqrt(pow(xrefe(0, 0) - xrefe(6, 0), 2) + pow(xrefe(0, 1) - xrefe(6, 1), 2) +
                             pow(xrefe(0, 2) - xrefe(6, 2), 2));

  // we need the (residual) displacement at the previous step
  Core::LinAlg::Matrix<NUMDOF_SOH8, 1> nodaldisp;
  for (int i = 0; i < NUMDOF_SOH8; ++i) nodaldisp(i, 0) = disp[i];

  /*
  ** EAS Technology: declare, initialize, set up, and alpha history -------- EAS
  */
  // in any case declare variables, sizes etc. only in eascase
  Core::LinAlg::SerialDenseMatrix* alpha = nullptr;              // EAS alphas
  std::vector<Core::LinAlg::SerialDenseMatrix>* M_GP = nullptr;  // EAS matrix M at all GPs
  Core::LinAlg::SerialDenseMatrix M;                             // EAS matrix M at current GP
  Core::LinAlg::SerialDenseVector feas;                          // EAS portion of internal forces
  Core::LinAlg::SerialDenseMatrix Kaa;                           // EAS matrix Kaa
  Core::LinAlg::SerialDenseMatrix Kda;                           // EAS matrix Kda
  double detJ0;                                                  // detJ(origin)
  Core::LinAlg::SerialDenseMatrix* oldfeas = nullptr;            // EAS history
  Core::LinAlg::SerialDenseMatrix* oldKaainv = nullptr;          // EAS history
  Core::LinAlg::SerialDenseMatrix* oldKda = nullptr;             // EAS history
  Core::LinAlg::SerialDenseMatrix* eas_inc = nullptr;            // EAS increment

  // transformation matrix T0, maps M-matrix evaluated at origin
  // between local element coords and global coords
  // here we already get the inverse transposed T0
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> T0invT;  // trafo matrix

  if (eastype_ != soh8_easnone)
  {
    /*
    ** EAS Update of alphas:
    ** the current alphas are (re-)evaluated out of
    ** Kaa and Kda of previous step to avoid additional element call.
    ** This corresponds to the (innermost) element update loop
    ** in the nonlinear FE-Skript page 120 (load-control alg. with EAS)
    */
    alpha = &easdata_.alpha;  // get alpha of previous iteration
    oldfeas = &easdata_.feas;
    oldKaainv = &easdata_.invKaa;
    oldKda = &easdata_.Kda;
    eas_inc = &easdata_.eas_inc;

    if (!alpha || !oldKaainv || !oldKda || !oldfeas || !eas_inc)
      FOUR_C_THROW("Missing EAS history-data");

    // ============================== DEPRECATED ==============================
    // FixMe deprecated implementation
    if (not is_params_interface())
    {
      // we need the (residual) displacement at the previous step
      Core::LinAlg::SerialDenseVector res_d_eas(NUMDOF_SOH8);
      for (int i = 0; i < NUMDOF_SOH8; ++i) res_d_eas(i) = residual[i];

      // this is a line search step, i.e. the direction of the eas increments
      // has been calculated by a Newton step and now it is only scaled
      if (params.isParameter("alpha_ls"))
      {
        double alpha_ls = params.get<double>("alpha_ls");
        // undo step
        eas_inc->scale(-1.);
        alpha->operator+=(*eas_inc);
        // scale increment
        eas_inc->scale(-1. * alpha_ls);
        // add reduced increment
        alpha->operator+=(*eas_inc);
      }
      // add Kda . res_d to feas
      // new alpha is: - Kaa^-1 . (feas + Kda . old_d), here: - Kaa^-1 . feas
      else
      {
        switch (eastype_)
        {
          case Discret::Elements::SoHex8::soh8_easfull:
            Core::LinAlg::DenseFunctions::multiply<double, soh8_easfull, NUMDOF_SOH8, 1>(
                1.0, oldfeas->values(), 1.0, oldKda->values(), res_d_eas.values());
            Core::LinAlg::DenseFunctions::multiply<double, soh8_easfull, soh8_easfull, 1>(
                0.0, eas_inc->values(), -1.0, oldKaainv->values(), oldfeas->values());
            Core::LinAlg::DenseFunctions::update<double, soh8_easfull, 1>(
                1., alpha->values(), 1., eas_inc->values());
            break;
          case Discret::Elements::SoHex8::soh8_easmild:
            Core::LinAlg::DenseFunctions::multiply<double, soh8_easmild, NUMDOF_SOH8, 1>(
                1.0, oldfeas->values(), 1.0, oldKda->values(), res_d_eas.values());
            Core::LinAlg::DenseFunctions::multiply<double, soh8_easmild, soh8_easmild, 1>(
                0.0, eas_inc->values(), -1.0, oldKaainv->values(), oldfeas->values());
            Core::LinAlg::DenseFunctions::update<double, soh8_easmild, 1>(
                1., alpha->values(), 1., eas_inc->values());
            break;
          case Discret::Elements::SoHex8::soh8_eassosh8:
            Core::LinAlg::DenseFunctions::multiply<double, soh8_eassosh8, NUMDOF_SOH8, 1>(
                1.0, oldfeas->values(), 1.0, oldKda->values(), res_d_eas.values());
            Core::LinAlg::DenseFunctions::multiply<double, soh8_eassosh8, soh8_eassosh8, 1>(
                0.0, eas_inc->values(), -1.0, oldKaainv->values(), oldfeas->values());
            Core::LinAlg::DenseFunctions::update<double, soh8_eassosh8, 1>(
                1., alpha->values(), 1., eas_inc->values());
            break;
          case Discret::Elements::SoHex8::soh8_easnone:
            break;
          default:
            FOUR_C_THROW("Don't know what to do with EAS type %d", eastype_);
            break;
        }
      }
    }  // if (not IsInterface())
    // ============================== DEPRECATED ==============================

    /* end of EAS Update ******************/

    // EAS portion of internal forces, also called enhancement vector s or Rtilde
    feas.size(neas_);

    // EAS matrix K_{alpha alpha}, also called Dtilde
    Kaa.shape(neas_, neas_);

    // EAS matrix K_{d alpha}
    Kda.shape(neas_, NUMDOF_SOH8);

    /* evaluation of EAS variables (which are constant for the following):
    ** -> M defining interpolation of enhanced strains alpha, evaluated at GPs
    ** -> determinant of Jacobi matrix at element origin (r=s=t=0.0)
    ** -> T0^{-T}
    */
    soh8_eassetup(&M_GP, detJ0, T0invT, xrefe);
  }  // -------------------------------------------------------------------- EAS

  // check if we need to split the residuals (for Newton line search)
  // if true an additional global vector is assembled containing
  // the internal forces without the condensed EAS entries and the norm
  // of the EAS residual is calculated
  bool split_res = params.isParameter("cond_rhs_norm");

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_XYZ;

  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd(true);
  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
  {
    /* get the inverse of the Jacobian matrix which looks like:
    **            [ x_,r  y_,r  z_,r ]^-1
    **     J^-1 = [ x_,s  y_,s  z_,s ]
    **            [ x_,t  y_,t  z_,t ]
    */
    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 * N_rst
    double detJ = detJ_[gp];
    N_XYZ.multiply(invJ_[gp], derivs[gp]);

    if (Prestress::is_mulf(pstype_))
    {
      // get Jacobian mapping wrt to the stored configuration
      Core::LinAlg::Matrix<3, 3> invJdef;
      prestress_->storageto_matrix(gp, invJdef, prestress_->j_history());
      // get derivatives wrt to last spatial configuration
      Core::LinAlg::Matrix<3, 8> N_xyz;
      N_xyz.multiply(invJdef, derivs[gp]);

      // build multiplicative incremental defgrd
      defgrd.multiply_tt(xdisp, N_xyz);
      defgrd(0, 0) += 1.0;
      defgrd(1, 1) += 1.0;
      defgrd(2, 2) += 1.0;

      // get stored old incremental F
      Core::LinAlg::Matrix<3, 3> Fhist;
      prestress_->storageto_matrix(gp, Fhist, prestress_->f_history());

      // build total defgrd = delta F * F_old
      Core::LinAlg::Matrix<3, 3> Fnew;
      Fnew.multiply(defgrd, Fhist);
      defgrd = Fnew;
    }
    else if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
    {
      // standard kinematically nonlinear analysis
      defgrd.multiply_tt(xcurr, N_XYZ);
    }
    else
    {
      // in kinematically linear analysis the deformation gradient is equal to identity
      // no difference between reference and current state
      for (int i = 0; i < 3; ++i) defgrd(i, i) = 1.0;
    }

    if (is_params_interface())
    {
      double det_defgrd = defgrd.determinant();
      if (det_defgrd < 0.0)
      {
        if (str_params_interface().is_tolerate_errors())
        {
          str_params_interface().set_ele_eval_error_flag(
              Solid::Elements::ele_error_negative_det_of_def_gradient);
          stiffmatrix->clear();
          force->clear();
          return;
        }
        else
          FOUR_C_THROW("negative deformation gradient determinant");
      }  // if (det_defgrd<0.0)
    }

    /* non-linear B-operator (may so be called, meaning
    ** of B-operator is not so sharp in the non-linear realm) *
    ** B = F . Bl *
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
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, NUMDOF_SOH8> bop;
    for (int i = 0; i < NUMNOD_SOH8; ++i)
    {
      bop(0, NODDOF_SOH8 * i + 0) = defgrd(0, 0) * N_XYZ(0, i);
      bop(0, NODDOF_SOH8 * i + 1) = defgrd(1, 0) * N_XYZ(0, i);
      bop(0, NODDOF_SOH8 * i + 2) = defgrd(2, 0) * N_XYZ(0, i);
      bop(1, NODDOF_SOH8 * i + 0) = defgrd(0, 1) * N_XYZ(1, i);
      bop(1, NODDOF_SOH8 * i + 1) = defgrd(1, 1) * N_XYZ(1, i);
      bop(1, NODDOF_SOH8 * i + 2) = defgrd(2, 1) * N_XYZ(1, i);
      bop(2, NODDOF_SOH8 * i + 0) = defgrd(0, 2) * N_XYZ(2, i);
      bop(2, NODDOF_SOH8 * i + 1) = defgrd(1, 2) * N_XYZ(2, i);
      bop(2, NODDOF_SOH8 * i + 2) = defgrd(2, 2) * N_XYZ(2, i);
      /* ~~~ */
      bop(3, NODDOF_SOH8 * i + 0) = defgrd(0, 0) * N_XYZ(1, i) + defgrd(0, 1) * N_XYZ(0, i);
      bop(3, NODDOF_SOH8 * i + 1) = defgrd(1, 0) * N_XYZ(1, i) + defgrd(1, 1) * N_XYZ(0, i);
      bop(3, NODDOF_SOH8 * i + 2) = defgrd(2, 0) * N_XYZ(1, i) + defgrd(2, 1) * N_XYZ(0, i);
      bop(4, NODDOF_SOH8 * i + 0) = defgrd(0, 1) * N_XYZ(2, i) + defgrd(0, 2) * N_XYZ(1, i);
      bop(4, NODDOF_SOH8 * i + 1) = defgrd(1, 1) * N_XYZ(2, i) + defgrd(1, 2) * N_XYZ(1, i);
      bop(4, NODDOF_SOH8 * i + 2) = defgrd(2, 1) * N_XYZ(2, i) + defgrd(2, 2) * N_XYZ(1, i);
      bop(5, NODDOF_SOH8 * i + 0) = defgrd(0, 2) * N_XYZ(0, i) + defgrd(0, 0) * N_XYZ(2, i);
      bop(5, NODDOF_SOH8 * i + 1) = defgrd(1, 2) * N_XYZ(0, i) + defgrd(1, 0) * N_XYZ(2, i);
      bop(5, NODDOF_SOH8 * i + 2) = defgrd(2, 2) * N_XYZ(0, i) + defgrd(2, 0) * N_XYZ(2, i);
    }

    // Right Cauchy-Green tensor = F^T * F
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> cauchygreen;
    cauchygreen.multiply_tn(defgrd, defgrd);

    // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
    Core::LinAlg::SerialDenseVector glstrain_epetra(Mat::NUM_STRESS_3D);
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> glstrain(glstrain_epetra.values(), true);
    if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
    {
      // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
      glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
      glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
      glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
      glstrain(3) = cauchygreen(0, 1);
      glstrain(4) = cauchygreen(1, 2);
      glstrain(5) = cauchygreen(2, 0);
    }
    else
    {
      // build the linearised strain epsilon = B_L . d
      glstrain.multiply(bop, nodaldisp);
    }

    // deformation gradient consistent with (potentially EAS-modified) GL strains
    // without eas this is equal to the regular defgrd.
    Core::LinAlg::Matrix<3, 3> defgrd_mod(defgrd);

    // EAS technology: "enhance the strains"  ----------------------------- EAS
    if (eastype_ != soh8_easnone)
    {
      M.shape(Mat::NUM_STRESS_3D, neas_);
      // map local M to global, also enhancement is referred to element origin
      // M = detJ0/detJ T0^{-T} . M
      // Core::LinAlg::SerialDenseMatrix Mtemp(M); // temp M for Matrix-Matrix-Product
      // add enhanced strains = M . alpha to GL strains to "unlock" element
      switch (eastype_)
      {
        case Discret::Elements::SoHex8::soh8_easfull:
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
              soh8_easfull>(M.values(), detJ0 / detJ, T0invT.data(), (M_GP->at(gp)).values());
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, soh8_easfull, 1>(
              1.0, glstrain.data(), 1.0, M.values(), alpha->values());
          break;
        case Discret::Elements::SoHex8::soh8_easmild:
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
              soh8_easmild>(M.values(), detJ0 / detJ, T0invT.data(), (M_GP->at(gp)).values());
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, soh8_easmild, 1>(
              1.0, glstrain.data(), 1.0, M.values(), alpha->values());
          break;
        case Discret::Elements::SoHex8::soh8_eassosh8:
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
              soh8_eassosh8>(M.values(), detJ0 / detJ, T0invT.data(), (M_GP->at(gp)).values());
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, soh8_eassosh8, 1>(
              1.0, glstrain.data(), 1.0, M.values(), alpha->values());
          break;
        case Discret::Elements::SoHex8::soh8_easnone:
          break;
        default:
          FOUR_C_THROW("Don't know what to do with EAS type %d", eastype_);
          break;
      }

      // calculate deformation gradient consistent with modified GL strain tensor
      if (std::static_pointer_cast<Mat::So3Material>(material())->needs_defgrd())
        calc_consistent_defgrd(defgrd, glstrain, defgrd_mod);

      const double det_defgrd_mod = defgrd_mod.determinant();
      if (det_defgrd_mod <= 0.0)
      {
        soh8_error_handling(det_defgrd_mod, params, __LINE__,
            Solid::Elements::ele_error_negative_det_of_def_gradient);
        return;
      }
    }  // ------------------------------------------------------------------ EAS

    // return gp strains (only in case of stress/strain output)
    switch (iostrain)
    {
      case Inpar::Solid::strain_gl:
      {
        if (elestrain == nullptr) FOUR_C_THROW("strain data not available");
        for (int i = 0; i < 3; ++i) (*elestrain)(gp, i) = glstrain(i);
        for (int i = 3; i < 6; ++i) (*elestrain)(gp, i) = 0.5 * glstrain(i);
      }
      break;
      case Inpar::Solid::strain_ea:
      {
        if (eastype_ != soh8_easnone)
        {
          FOUR_C_THROW(
              "EA strains are computed with the 'normal' deformation gradient from GL strains, and "
              "not with the deformation gradient that is consistent with EAS!\n"
              "Use the new solid elements instead!");
        }

        if (elestrain == nullptr) FOUR_C_THROW("strain data not available");
        // rewriting Green-Lagrange strains in matrix format
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> gl;
        gl(0, 0) = glstrain(0);
        gl(0, 1) = 0.5 * glstrain(3);
        gl(0, 2) = 0.5 * glstrain(5);
        gl(1, 0) = gl(0, 1);
        gl(1, 1) = glstrain(1);
        gl(1, 2) = 0.5 * glstrain(4);
        gl(2, 0) = gl(0, 2);
        gl(2, 1) = gl(1, 2);
        gl(2, 2) = glstrain(2);

        // inverse of deformation gradient
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invdefgrd;
        invdefgrd.invert(defgrd);

        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> temp;
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> euler_almansi;
        temp.multiply(gl, invdefgrd);
        euler_almansi.multiply_tn(invdefgrd, temp);

        (*elestrain)(gp, 0) = euler_almansi(0, 0);
        (*elestrain)(gp, 1) = euler_almansi(1, 1);
        (*elestrain)(gp, 2) = euler_almansi(2, 2);
        (*elestrain)(gp, 3) = euler_almansi(0, 1);
        (*elestrain)(gp, 4) = euler_almansi(1, 2);
        (*elestrain)(gp, 5) = euler_almansi(0, 2);
      }
      break;
      case Inpar::Solid::strain_log:
      {
        if (elestrain == nullptr) FOUR_C_THROW("strain data not available");

        /// the Eularian logarithmic strain is defined as the natural logarithm of the left stretch
        /// tensor [1,2]: \f[
        ///    e_{log} = e_{hencky} = ln (\mathbf{V}) = \sum_{i=1}^3 (ln \lambda_i) \mathbf{n}_i
        ///    \otimes \mathbf{n}_i
        /// \f]
        ///< h3>References</h3>
        /// <ul>
        /// <li> [1] H. Xiao, Beijing, China, O. T. Bruhns and A. Meyers (1997) Logarithmic strain,
        /// logarithmic spin and logarithmic rate, Eq. 5 <li> [2] Caminero et al. (2011) Modeling
        /// large strain anisotropic elasto-plasticity with logarithmic strain and stress measures,
        /// Eq. 70
        /// </ul>
        ///
        /// \author HdV
        /// \date 08/13

        // eigenvalue decomposition (from elasthyper.cpp)
        Core::LinAlg::Matrix<3, 3> prstr2(true);  // squared principal stretches
        Core::LinAlg::Matrix<3, 1> prstr(true);   // principal stretch
        Core::LinAlg::Matrix<3, 3> prdir(true);   // principal directions
        Core::LinAlg::syev(cauchygreen, prstr2, prdir);

        // THE principal stretches
        for (int al = 0; al < 3; ++al) prstr(al) = std::sqrt(prstr2(al, al));

        // populating the logarithmic strain matrix
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> lnv(true);

        // checking if cauchy green is correctly determined to ensure eigen vectors in correct
        // direction i.e. a flipped eigenvector is also a valid solution C = \sum_{i=1}^3
        // (\lambda_i^2) \mathbf{n}_i \otimes \mathbf{n}_i
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> tempCG(true);

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
          tempCG(0, 0) += (prstr(k)) * (prstr(k))*n_00;
          tempCG(0, 1) += (prstr(k)) * (prstr(k))*n_01;
          tempCG(0, 2) += (prstr(k)) * (prstr(k))*n_02;
          tempCG(1, 0) += (prstr(k)) * (prstr(k))*n_01;  // symmetry
          tempCG(1, 1) += (prstr(k)) * (prstr(k))*n_11;
          tempCG(1, 2) += (prstr(k)) * (prstr(k))*n_12;
          tempCG(2, 0) += (prstr(k)) * (prstr(k))*n_02;  // symmetry
          tempCG(2, 1) += (prstr(k)) * (prstr(k))*n_12;  // symmetry
          tempCG(2, 2) += (prstr(k)) * (prstr(k))*n_22;

          // Computation of the Logarithmic strain tensor

          lnv(0, 0) += (std::log(prstr(k)))*n_00;
          lnv(0, 1) += (std::log(prstr(k)))*n_01;
          lnv(0, 2) += (std::log(prstr(k)))*n_02;
          lnv(1, 0) += (std::log(prstr(k)))*n_01;  // symmetry
          lnv(1, 1) += (std::log(prstr(k)))*n_11;
          lnv(1, 2) += (std::log(prstr(k)))*n_12;
          lnv(2, 0) += (std::log(prstr(k)))*n_02;  // symmetry
          lnv(2, 1) += (std::log(prstr(k)))*n_12;  // symmetry
          lnv(2, 2) += (std::log(prstr(k)))*n_22;
        }

        // compare CG computed with deformation gradient with CG computed
        // with eigenvalues and -vectors to determine/ensure the correct
        // orientation of the eigen vectors
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> diffCG(true);

        for (int i = 0; i < 3; ++i)
        {
          for (int j = 0; j < 3; ++j)
          {
            diffCG(i, j) = cauchygreen(i, j) - tempCG(i, j);
            // the solution to this problem is to evaluate the cauchygreen tensor with tempCG
            // computed with every combination of eigenvector orientations -- up to nine comparisons
            if (diffCG(i, j) > 1e-10)
              FOUR_C_THROW(
                  "eigenvector orientation error with the diffCG giving problems: %10.5e \n BUILD "
                  "SOLUTION TO FIX IT",
                  diffCG(i, j));
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
      case Inpar::Solid::strain_none:
        break;
      default:
        FOUR_C_THROW("requested strain type not available");
        break;
    }


    /* call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc
    ** Here all possible material laws need to be incorporated,
    ** the stress vector, a C-matrix must be retrieved,
    ** all necessary data must be passed.
    */
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> cmat(true);
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> stress(true);

    Utils::get_temperature_for_structural_material<Core::FE::CellType::hex8>(shapefcts[gp], params);

    if (material()->material_type() == Core::Materials::m_constraintmixture ||
        material()->material_type() == Core::Materials::m_growthremodel_elasthyper ||
        material()->material_type() == Core::Materials::m_mixture)
    {
      Core::LinAlg::Matrix<NUMDIM_SOH8, 1> point(true);
      soh8_gauss_point_refe_coords(point, xrefe, gp);
      params.set("gp_coords_ref", point);

      // center of element in reference configuration
      point.clear();
      soh8_element_center_refe_coords(point, xrefe);
      params.set("elecenter_coords_ref", point);
    }

    // if output is requested only active stresses are written.
    params.set<Inpar::Solid::StressType>("iostress", iostress);

    std::shared_ptr<Mat::So3Material> so3mat =
        std::static_pointer_cast<Mat::So3Material>(material());
    so3mat->evaluate(&defgrd_mod, &glstrain, params, &stress, &cmat, gp, id());

    // stop if the material evaluation fails
    if (is_params_interface() and str_params_interface().is_tolerate_errors())
      if (str_params_interface().get_ele_eval_error_flag() != Solid::Elements::ele_error_none)
        return;

    // end of call material law ccccccccccccccccccccccccccccccccccccccccccccccc

    // return gp plastic strains (only in case of plastic strain output)
    switch (ioplstrain)
    {
      case Inpar::Solid::strain_gl:
      {
        if (eleplstrain == nullptr) FOUR_C_THROW("plastic strain data not available");
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> plglstrain =
            params.get<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("plglstrain");
        for (int i = 0; i < 3; ++i) (*eleplstrain)(gp, i) = plglstrain(i);
        for (int i = 3; i < 6; ++i) (*eleplstrain)(gp, i) = 0.5 * plglstrain(i);
        break;
      }
      case Inpar::Solid::strain_ea:
      {
        if (eastype_ != soh8_easnone)
        {
          FOUR_C_THROW(
              "EA strains are computed with the 'normal' deformation gradient from GL strains, and "
              "not with the deformation gradient that is consistent with EAS!\n"
              "Use the new solid elements instead!");
        }

        if (eleplstrain == nullptr) FOUR_C_THROW("plastic strain data not available");
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> plglstrain =
            params.get<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("plglstrain");
        // rewriting Green-Lagrange strains in matrix format
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> gl;
        gl(0, 0) = plglstrain(0);
        gl(0, 1) = 0.5 * plglstrain(3);
        gl(0, 2) = 0.5 * plglstrain(5);
        gl(1, 0) = gl(0, 1);
        gl(1, 1) = plglstrain(1);
        gl(1, 2) = 0.5 * plglstrain(4);
        gl(2, 0) = gl(0, 2);
        gl(2, 1) = gl(1, 2);
        gl(2, 2) = plglstrain(2);

        // inverse of deformation gradient
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invdefgrd;
        invdefgrd.invert(defgrd);

        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> temp;
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> euler_almansi;
        temp.multiply(gl, invdefgrd);
        euler_almansi.multiply_tn(invdefgrd, temp);

        (*eleplstrain)(gp, 0) = euler_almansi(0, 0);
        (*eleplstrain)(gp, 1) = euler_almansi(1, 1);
        (*eleplstrain)(gp, 2) = euler_almansi(2, 2);
        (*eleplstrain)(gp, 3) = euler_almansi(0, 1);
        (*eleplstrain)(gp, 4) = euler_almansi(1, 2);
        (*eleplstrain)(gp, 5) = euler_almansi(0, 2);
        break;
      }
      case Inpar::Solid::strain_none:
        break;
      default:
        FOUR_C_THROW("requested plastic strain type not available");
        break;
    }

    // return gp stresses
    switch (iostress)
    {
      case Inpar::Solid::stress_2pk:
      {
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");
        for (int i = 0; i < Mat::NUM_STRESS_3D; ++i) (*elestress)(gp, i) = stress(i);
      }
      break;
      case Inpar::Solid::stress_cauchy:
      {
        if (eastype_ != soh8_easnone)
        {
          FOUR_C_THROW(
              "Cauchy stresses are computed with the 'normal' deformation gradient from 2PK "
              "stresses and not with the deformation gradient that is consistent with EAS!\n"
              "Use the new solid elements instead!");
        }

        if (elestress == nullptr) FOUR_C_THROW("stress data not available");
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> cauchystress(false);
        p_k2to_cauchy(&stress, &defgrd, &cauchystress);

        (*elestress)(gp, 0) = cauchystress(0, 0);
        (*elestress)(gp, 1) = cauchystress(1, 1);
        (*elestress)(gp, 2) = cauchystress(2, 2);
        (*elestress)(gp, 3) = cauchystress(0, 1);
        (*elestress)(gp, 4) = cauchystress(1, 2);
        (*elestress)(gp, 5) = cauchystress(0, 2);
      }
      break;
      case Inpar::Solid::stress_none:
        break;
      default:
        FOUR_C_THROW("requested stress type not available");
        break;
    }

    const double detJ_w = detJ * gpweights[gp];
    // update internal force vector
    if (force != nullptr)
    {
      // integrate internal force vector f = f + (B^T . sigma) * detJ * w(gp)
      force->multiply_tn(detJ_w, bop, stress, 1.0);
    }

    // structural force vector
    if (split_res && force_str != nullptr) force_str->multiply_tn(detJ_w, bop, stress, 1.);

    // update stiffness matrix
    if (stiffmatrix != nullptr)
    {
      // integrate `elastic' and `initial-displacement' stiffness matrix
      // keu = keu + (B^T . C . B) * detJ * w(gp)
      Core::LinAlg::Matrix<6, NUMDOF_SOH8> cb;
      cb.multiply(cmat, bop);

      if (analyticalmaterialtangent_)
        stiffmatrix->multiply_tn(detJ_w, bop, cb, 1.0);  // standard hex8 evaluation
      else
      {
        evaluate_finite_difference_material_tangent(stiffmatrix, stress, disp, detJ_w, detJ, detJ0,
            elediagonallength, bop, cb, N_XYZ, T0invT, M_GP, alpha, M, gp, params);
      }


      if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
      {
        // integrate `geometric' stiffness matrix and add to keu *****************
        Core::LinAlg::Matrix<6, 1> sfac(stress);  // auxiliary integrated stress
        sfac.scale(detJ_w);            // detJ*w(gp)*[S11,S22,S33,S12=S21,S23=S32,S13=S31]
        std::vector<double> SmB_L(3);  // intermediate Sm.B_L
        // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
        for (int inod = 0; inod < NUMNOD_SOH8; ++inod)
        {
          SmB_L[0] = sfac(0) * N_XYZ(0, inod) + sfac(3) * N_XYZ(1, inod) + sfac(5) * N_XYZ(2, inod);
          SmB_L[1] = sfac(3) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod) + sfac(4) * N_XYZ(2, inod);
          SmB_L[2] = sfac(5) * N_XYZ(0, inod) + sfac(4) * N_XYZ(1, inod) + sfac(2) * N_XYZ(2, inod);
          for (int jnod = 0; jnod < NUMNOD_SOH8; ++jnod)
          {
            double bopstrbop = 0.0;  // intermediate value
            for (int idim = 0; idim < NUMDIM_SOH8; ++idim)
              bopstrbop += N_XYZ(idim, jnod) * SmB_L[idim];
            (*stiffmatrix)(3 * inod + 0, 3 * jnod + 0) += bopstrbop;
            (*stiffmatrix)(3 * inod + 1, 3 * jnod + 1) += bopstrbop;
            (*stiffmatrix)(3 * inod + 2, 3 * jnod + 2) += bopstrbop;
          }

        }  // end of integrate `geometric' stiffness******************************
      }

      // EAS technology: integrate matrices --------------------------------- EAS
      if (eastype_ != soh8_easnone)
      {
        // integrate Kaa: Kaa += (M^T . cmat . M) * detJ * w(gp)
        // integrate Kda: Kda += (M^T . cmat . B) * detJ * w(gp)
        // integrate feas: feas += (M^T . sigma) * detJ *wp(gp)
        Core::LinAlg::SerialDenseMatrix cM(Mat::NUM_STRESS_3D, neas_);  // temporary c . M
        switch (eastype_)
        {
          case Discret::Elements::SoHex8::soh8_easfull:
            Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
                soh8_easfull>(cM.values(), cmat.data(), M.values());
            Core::LinAlg::DenseFunctions::multiply_tn<double, soh8_easfull, Mat::NUM_STRESS_3D,
                soh8_easfull>(1.0, Kaa.values(), detJ_w, M.values(), cM.values());
            Core::LinAlg::DenseFunctions::multiply_tn<double, soh8_easfull, Mat::NUM_STRESS_3D,
                NUMDOF_SOH8>(1.0, Kda.values(), detJ_w, M.values(), cb.data());
            Core::LinAlg::DenseFunctions::multiply_tn<double, soh8_easfull, Mat::NUM_STRESS_3D, 1>(
                1.0, feas.values(), detJ_w, M.values(), stress.data());
            break;
          case Discret::Elements::SoHex8::soh8_easmild:
            Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
                soh8_easmild>(cM.values(), cmat.data(), M.values());
            Core::LinAlg::DenseFunctions::multiply_tn<double, soh8_easmild, Mat::NUM_STRESS_3D,
                soh8_easmild>(1.0, Kaa.values(), detJ_w, M.values(), cM.values());
            Core::LinAlg::DenseFunctions::multiply_tn<double, soh8_easmild, Mat::NUM_STRESS_3D,
                NUMDOF_SOH8>(1.0, Kda.values(), detJ_w, M.values(), cb.data());
            Core::LinAlg::DenseFunctions::multiply_tn<double, soh8_easmild, Mat::NUM_STRESS_3D, 1>(
                1.0, feas.values(), detJ_w, M.values(), stress.data());
            break;
          case Discret::Elements::SoHex8::soh8_eassosh8:
            Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
                soh8_eassosh8>(cM.values(), cmat.data(), M.values());
            Core::LinAlg::DenseFunctions::multiply_tn<double, soh8_eassosh8, Mat::NUM_STRESS_3D,
                soh8_eassosh8>(1.0, Kaa.values(), detJ_w, M.values(), cM.values());
            Core::LinAlg::DenseFunctions::multiply_tn<double, soh8_eassosh8, Mat::NUM_STRESS_3D,
                NUMDOF_SOH8>(1.0, Kda.values(), detJ_w, M.values(), cb.data());
            Core::LinAlg::DenseFunctions::multiply_tn<double, soh8_eassosh8, Mat::NUM_STRESS_3D, 1>(
                1.0, feas.values(), detJ_w, M.values(), stress.data());
            break;
          case Discret::Elements::SoHex8::soh8_easnone:
            break;
          default:
            FOUR_C_THROW("Don't know what to do with EAS type %d", eastype_);
            break;
        }
      }  // ---------------------------------------------------------------- EAS
    }

    if (massmatrix != nullptr)  // evaluate mass matrix +++++++++++++++++++++++++
    {
      const double density = material()->density(gp);

      // integrate consistent mass matri
      const double factor = detJ_w * density;
      double ifactor, massfactor;
      for (int inod = 0; inod < NUMNOD_SOH8; ++inod)
      {
        ifactor = shapefcts[gp](inod) * factor;
        for (int jnod = 0; jnod < NUMNOD_SOH8; ++jnod)
        {
          massfactor = shapefcts[gp](jnod) * ifactor;  // intermediate factor
          (*massmatrix)(NUMDIM_SOH8* inod + 0, NUMDIM_SOH8 * jnod + 0) += massfactor;
          (*massmatrix)(NUMDIM_SOH8* inod + 1, NUMDIM_SOH8 * jnod + 1) += massfactor;
          (*massmatrix)(NUMDIM_SOH8* inod + 2, NUMDIM_SOH8 * jnod + 2) += massfactor;
        }
      }

      // check for non-constant mass matrix
      if (so3mat->varying_density())
      {
        /*
         If the density, i.e. the mass matrix, is not constant, a linearization is necessary.
         In general, the mass matrix can be dependent on the displacements, the velocities and the
         accelerations. We write all the additional terms into the mass matrix, hence, conversion
         from accelerations to velocities and displacements are needed. As those conversions depend
         on the time integration scheme, the factors are set within the respective time integrators
         and read from the parameter list inside the element (this is a little ugly...). */
        double timintfac_dis = 0.0;
        double timintfac_vel = 0.0;
        if (is_params_interface())
        {
          timintfac_dis = str_params_interface().get_tim_int_factor_disp();
          timintfac_vel = str_params_interface().get_tim_int_factor_vel();
        }
        else
        {
          timintfac_dis = params.get<double>("timintfac_dis");
          timintfac_vel = params.get<double>("timintfac_vel");
        }
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> linmass_disp(true);
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> linmass_vel(true);
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> linmass(true);

        // evaluate derivative of mass w.r.t. to right cauchy green tensor
        so3mat->evaluate_non_lin_mass(
            &defgrd, &glstrain, params, &linmass_disp, &linmass_vel, gp, id());

        // multiply by 2.0 to get derivative w.r.t green lagrange strains and multiply by time
        // integration factor
        linmass_disp.scale(2.0 * timintfac_dis);
        linmass_vel.scale(2.0 * timintfac_vel);
        linmass.update(1.0, linmass_disp, 1.0, linmass_vel, 0.0);

        // evaluate accelerations at time n+1 at gauss point
        Core::LinAlg::Matrix<NUMDIM_SOH8, 1> myacc(true);
        for (int idim = 0; idim < NUMDIM_SOH8; ++idim)
          for (int inod = 0; inod < NUMNOD_SOH8; ++inod)
            myacc(idim) += shapefcts[gp](inod) * (*acc)[idim + (inod * NUMDIM_SOH8)];

        if (stiffmatrix != nullptr)
        {
          // integrate linearisation of mass matrix
          //(B^T . d\rho/d disp . a) * detJ * w(gp)
          Core::LinAlg::Matrix<1, NUMDOF_SOH8> cb;
          cb.multiply_tn(linmass_disp, bop);
          for (int inod = 0; inod < NUMNOD_SOH8; ++inod)
          {
            double factor = detJ_w * shapefcts[gp](inod);
            for (int idim = 0; idim < NUMDIM_SOH8; ++idim)
            {
              double massfactor = factor * myacc(idim);
              for (int jnod = 0; jnod < NUMNOD_SOH8; ++jnod)
                for (int jdim = 0; jdim < NUMDIM_SOH8; ++jdim)
                  (*massmatrix)(inod* NUMDIM_SOH8 + idim, jnod * NUMDIM_SOH8 + jdim) +=
                      massfactor * cb(jnod * NUMDIM_SOH8 + jdim);
            }
          }
        }

        // internal force vector without EAS terms
        if (forceinert != nullptr)
        {
          // integrate nonlinear inertia force term
          for (int inod = 0; inod < NUMNOD_SOH8; ++inod)
          {
            double forcefactor = shapefcts[gp](inod) * detJ_w;
            for (int idim = 0; idim < NUMDIM_SOH8; ++idim)
              (*forceinert)(inod* NUMDIM_SOH8 + idim) += forcefactor * density * myacc(idim);
          }
        }
      }

    }  // end of mass matrix +++++++++++++++++++++++++++++++++++++++++++++++++++
    /* =========================================================================*/
  } /* ==================================================== end of Loop over GP */
  /* =========================================================================*/

  // rhs norm of eas equations
  if (eastype_ != soh8_easnone && split_res && force != nullptr)
    // only add for row-map elements
    if (params.get<int>("MyPID") == owner())
      params.get<double>("cond_rhs_norm") += pow(Core::LinAlg::norm2(feas), 2.);

  if (force != nullptr && stiffmatrix != nullptr)
  {
    // EAS technology: ------------------------------------------------------ EAS
    // subtract EAS matrices from disp-based Kdd to "soften" element
    if (eastype_ != soh8_easnone)
    {
      // we need the inverse of Kaa. Catch Inf/NaN case
      const double norm1 = Kaa.normOne();
      if (std::isnan(norm1) || std::isinf(norm1) || norm1 == 0.)
      {
        for (int i = 0; i < Kaa.numCols(); ++i)
          for (int j = 0; j < Kaa.numRows(); ++j)
            Kaa(j, i) = std::numeric_limits<double>::quiet_NaN();
      }
      else
      {
        using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
        using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
        Teuchos::SerialDenseSolver<ordinalType, scalarType> solve_for_inverseKaa;
        solve_for_inverseKaa.setMatrix(Teuchos::rcpFromRef(Kaa));
        solve_for_inverseKaa.invert();
      }

      // EAS-stiffness matrix is: Kdd - Kda^T . Kaa^-1 . Kda
      // EAS-internal force is: fint - Kda^T . Kaa^-1 . feas

      Core::LinAlg::SerialDenseMatrix KdaKaa(NUMDOF_SOH8, neas_);  // temporary Kda.Kaa^{-1}
      switch (eastype_)
      {
        case Discret::Elements::SoHex8::soh8_easfull:
          Core::LinAlg::DenseFunctions::multiply_tn<double, NUMDOF_SOH8, soh8_easfull,
              soh8_easfull>(KdaKaa.values(), Kda.values(), Kaa.values());
          Core::LinAlg::DenseFunctions::multiply<double, NUMDOF_SOH8, soh8_easfull, NUMDOF_SOH8>(
              1.0, stiffmatrix->data(), -1.0, KdaKaa.values(), Kda.values());
          Core::LinAlg::DenseFunctions::multiply<double, NUMDOF_SOH8, soh8_easfull, 1>(
              1.0, force->data(), -1.0, KdaKaa.values(), feas.values());
          break;
        case Discret::Elements::SoHex8::soh8_easmild:
          Core::LinAlg::DenseFunctions::multiply_tn<double, NUMDOF_SOH8, soh8_easmild,
              soh8_easmild>(KdaKaa.values(), Kda.values(), Kaa.values());
          Core::LinAlg::DenseFunctions::multiply<double, NUMDOF_SOH8, soh8_easmild, NUMDOF_SOH8>(
              1.0, stiffmatrix->data(), -1.0, KdaKaa.values(), Kda.values());
          Core::LinAlg::DenseFunctions::multiply<double, NUMDOF_SOH8, soh8_easmild, 1>(
              1.0, force->data(), -1.0, KdaKaa.values(), feas.values());
          break;
        case Discret::Elements::SoHex8::soh8_eassosh8:
          Core::LinAlg::DenseFunctions::multiply_tn<double, NUMDOF_SOH8, soh8_eassosh8,
              soh8_eassosh8>(KdaKaa.values(), Kda.values(), Kaa.values());
          Core::LinAlg::DenseFunctions::multiply<double, NUMDOF_SOH8, soh8_eassosh8, NUMDOF_SOH8>(
              1.0, stiffmatrix->data(), -1.0, KdaKaa.values(), Kda.values());
          Core::LinAlg::DenseFunctions::multiply<double, NUMDOF_SOH8, soh8_eassosh8, 1>(
              1.0, force->data(), -1.0, KdaKaa.values(), feas.values());
          break;
        case Discret::Elements::SoHex8::soh8_easnone:
          break;
        default:
          FOUR_C_THROW("Don't know what to do with EAS type %d", eastype_);
          break;
      }

      // store current EAS data in history
      for (int i = 0; i < neas_; ++i)
      {
        for (int j = 0; j < neas_; ++j) (*oldKaainv)(i, j) = Kaa(i, j);
        for (int j = 0; j < NUMDOF_SOH8; ++j) (*oldKda)(i, j) = Kda(i, j);
        (*oldfeas)(i, 0) = feas(i);
      }
    }  // -------------------------------------------------------------------- EAS
  }
  return;
}  // Discret::Elements::So_hex8::nlnstiffmass


/*----------------------------------------------------------------------*
 |  lump mass matrix (private)                               bborn 07/08|
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::soh8_lumpmass(Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8>* emass)
{
  // lump mass matrix
  if (emass != nullptr)
  {
    // we assume #elemat2 is a square matrix
    for (unsigned int c = 0; c < (*emass).num_cols(); ++c)  // parse columns
    {
      double d = 0.0;
      for (unsigned int r = 0; r < (*emass).num_rows(); ++r)  // parse rows
      {
        d += (*emass)(r, c);  // accumulate row entries
        (*emass)(r, c) = 0.0;
      }
      (*emass)(c, c) = d;  // apply sum of row entries on diagonal
    }
  }
}

/*----------------------------------------------------------------------*
 |  Evaluate Hex8 Shape fcts at all 8 Gauss Points             maf 05/08|
 *----------------------------------------------------------------------*/
std::vector<Core::LinAlg::Matrix<NUMNOD_SOH8, 1>> Discret::Elements::SoHex8::soh8_shapefcts() const
{
  std::vector<Core::LinAlg::Matrix<NUMNOD_SOH8, 1>> shapefcts(NUMGPT_SOH8);

  // fill up nodal f at each gp
  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
  {
    const Core::LinAlg::Matrix<NUMDIM_SOH8, 1> rst_gp(gp_rule_.point(gp), true);
    Core::FE::shape_function<Core::FE::CellType::hex8>(rst_gp, shapefcts[gp]);
  }

  return shapefcts;
}


/*----------------------------------------------------------------------*
 |  Evaluate Hex8 Shape fct derivs at all 8 Gauss Points       maf 05/08|
 *----------------------------------------------------------------------*/
std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> Discret::Elements::SoHex8::soh8_derivs()
    const
{
  std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs(NUMGPT_SOH8);

  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
  {
    soh8_derivs(derivs[gp], gp);
  }

  return derivs;
}

// Evaluate the derivatives of the shape functions for a specific Gauss point
void Discret::Elements::SoHex8::soh8_derivs(
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>& derivs, const int gp) const
{
  const Core::LinAlg::Matrix<NUMDIM_SOH8, 1> rst_gp(gp_rule_.point(gp), true);
  Core::FE::shape_function_deriv1<Core::FE::CellType::hex8>(rst_gp, derivs);
}

/*----------------------------------------------------------------------*
 |  Evaluate Hex8 Weights at all 8 Gauss Points                maf 05/08|
 *----------------------------------------------------------------------*/
std::vector<double> Discret::Elements::SoHex8::soh8_weights() const
{
  std::vector<double> weights(NUMGPT_SOH8);
  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp) weights[gp] = gp_rule_.weight(gp);

  return weights;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Discret::Elements::SoHex8::soh8_create_eas_backup_state(const std::vector<double>& displ_incr)
{
  if (eastype_ == soh8_easnone) return;

  // --- create EAS state backup ----------------------------------------------
  {
    const auto* alpha = &easdata_.alpha;
    if (not alpha) FOUR_C_THROW("Can't access the current enhanced strain state.");

    auto* alpha_backup_ptr = &easdata_.alpha_backup;
    if (alpha_backup_ptr)
      *alpha_backup_ptr = *alpha;
    else
      easdata_.alpha_backup = *alpha;
  }

  // --- create EAS increment backup ------------------------------------------
  {
    // compute the current eas increment
    Core::LinAlg::SerialDenseMatrix eas_inc(neas_, 1);
    soh8_compute_eas_inc(displ_incr, &eas_inc);

    auto* eas_inc_backup_ptr = &easdata_.eas_inc_backup;
    if (eas_inc_backup_ptr)
      *eas_inc_backup_ptr = eas_inc;
    else
      easdata_.eas_inc_backup = eas_inc;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Discret::Elements::SoHex8::soh8_recover_from_eas_backup_state()
{
  if (eastype_ == soh8_easnone) return;

  Core::LinAlg::SerialDenseMatrix* alpha = nullptr;
  Core::LinAlg::SerialDenseMatrix* eas_inc = nullptr;

  // --- recover state from EAS backup ----------------------------------------
  {
    const auto* alpha_backup = &easdata_.alpha_backup;
    if (not alpha_backup)
      FOUR_C_THROW(
          "Can't access the enhanced strain backup state. Did you "
          "create a backup? See soh8_create_eas_backup_state().");

    alpha = &easdata_.alpha;
    if (not alpha) FOUR_C_THROW("Can't access the enhanced strain state.");

    *alpha = *alpha_backup;
  }

  // --- recover increment from EAS backup ------------------------------------
  {
    const auto* eas_inc_backup = &easdata_.eas_inc_backup;
    if (not eas_inc_backup)
      FOUR_C_THROW(
          "Can't access the enhanced strain increment backup. Did you "
          "create a backup? See soh8_create_eas_backup_state().");

    eas_inc = &easdata_.eas_inc;
    if (not eas_inc) FOUR_C_THROW("Can't access the enhanced strain increment.");

    *eas_inc = *eas_inc_backup;
  }

  // Finally, we have to update the backup state, otherwise a follow-up
  // step length adaption will lead to a wrong eas state.
  {
    old_step_length_ = 0.0;
  }
}

/*----------------------------------------------------------------------*
 |  init the element (public)                                  gee 04/08|
 *----------------------------------------------------------------------*/
int Discret::Elements::SoHex8Type::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<Discret::Elements::SoHex8*>(dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex8* failed");
    actele->init_jacobian_mapping();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  compute def gradient at every gaussian point (protected)   gee 07/08|
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::def_gradient(const std::vector<double>& disp,
    Core::LinAlg::SerialDenseMatrix& gpdefgrd, Discret::Elements::PreStress& prestress)
{
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs = soh8_derivs();

  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xdisp;  // current  coord. of element
  Utils::evaluate_nodal_displacements<Core::FE::CellType::hex8, 3>(disp, xdisp);

  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
  {
    // get Jacobian mapping wrt to the stored deformed configuration
    Core::LinAlg::Matrix<3, 3> invJdef;
    prestress.storageto_matrix(gp, invJdef, prestress.j_history());

    // by N_XYZ = J^-1 * N_rst
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_xyz;
    N_xyz.multiply(invJdef, derivs[gp]);

    // build defgrd (independent of xrefe!)
    Core::LinAlg::Matrix<3, 3> defgrd(true);
    if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
    {
      defgrd.multiply_tt(xdisp, N_xyz);
    }
    defgrd(0, 0) += 1.0;
    defgrd(1, 1) += 1.0;
    defgrd(2, 2) += 1.0;

    prestress.matrixto_storage(gp, defgrd, gpdefgrd);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  compute Jac.mapping wrt deformed configuration (protected) gee 07/08|
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::update_jacobian_mapping(
    const std::vector<double>& disp, Discret::Elements::PreStress& prestress)
{
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs = soh8_derivs();

  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xdisp(false);
  Utils::evaluate_nodal_displacements<Core::FE::CellType::hex8, 3>(disp, xdisp);

  Core::LinAlg::Matrix<3, 3> invJhist;
  Core::LinAlg::Matrix<3, 3> invJ;
  Core::LinAlg::Matrix<3, 3> defgrd(true);
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_xyz;
  Core::LinAlg::Matrix<3, 3> invJnew;
  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
  {
    // get the invJ old state
    prestress.storageto_matrix(gp, invJhist, prestress.j_history());
    // get derivatives wrt to invJhist
    N_xyz.multiply(invJhist, derivs[gp]);
    // build defgrd \partial x_new / \parial x_old , where x_old != X
    if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
    {
      defgrd.multiply_tt(xdisp, N_xyz);
    }
    defgrd(0, 0) += 1.0;
    defgrd(1, 1) += 1.0;
    defgrd(2, 2) += 1.0;
    // make inverse of this defgrd
    defgrd.invert();
    // push-forward of Jinv
    invJnew.multiply_tn(defgrd, invJhist);
    // store new reference configuration
    prestress.matrixto_storage(gp, invJnew, prestress.j_history());
  }  // for (int gp=0; gp<NUMGPT_SOH8; ++gp)

  return;
}

/*---------------------------------------------------------------------------------------------*
 |  Update history variables (e.g. remodeling of fiber directions) (protected)      braeu 07/16|
 *---------------------------------------------------------------------------------------------*/
void Discret::Elements::SoHex8::update_element(
    std::vector<double>& disp, Teuchos::ParameterList& params, Core::Mat::Material& mat)
{
  // Calculate current deformation gradient
  if ((mat.material_type() == Core::Materials::m_constraintmixture) ||
      (mat.material_type() == Core::Materials::m_elasthyper) ||
      (mat.material_type() == Core::Materials::m_growthremodel_elasthyper) ||
      (solid_material()->uses_extended_update()))
  {
    Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe(false);
    Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xdisp(false);
    Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xcurr(false);

    Utils::evaluate_nodal_coordinates<Core::FE::CellType::hex8, 3>(nodes(), xrefe);
    Utils::evaluate_nodal_displacements<Core::FE::CellType::hex8, 3>(disp, xdisp);
    Utils::evaluate_current_nodal_coordinates<Core::FE::CellType::hex8, 3>(xrefe, xdisp, xcurr);

    /* =========================================================================*/
    /* ================================================= Loop over Gauss Points */
    /* =========================================================================*/
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_XYZ;
    // interpolated values of stress and defgrd for remodeling
    Core::LinAlg::Matrix<3, 3> avg_stress(true);
    Core::LinAlg::Matrix<3, 3> avg_defgrd(true);

    // build deformation gradient wrt to material configuration
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd(false);
    params.set<int>("numgp", static_cast<int>(NUMGPT_SOH8));

    // center of element in reference configuration
    Core::LinAlg::Matrix<NUMDIM_SOH8, 1> point(false);
    point.clear();
    soh8_element_center_refe_coords(point, xrefe);
    params.set("elecenter_coords_ref", point);

    for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
    {
      soh8_gauss_point_refe_coords(point, xrefe, gp);
      params.set("gp_coords_ref", point);
      Core::LinAlg::Matrix<3, 8> derivs(false);
      soh8_derivs(derivs, gp);

      // Compute deformation gradient
      Utils::compute_deformation_gradient<Core::FE::CellType::hex8>(
          defgrd, kintype_, xdisp, xcurr, invJ_[gp], derivs, pstype_, *prestress_, gp);

      // call material update if material = m_growthremodel_elasthyper (calculate and update
      // inelastic deformation gradient)
      if (solid_material()->uses_extended_update())
      {
        solid_material()->update(defgrd, gp, params, id());
      }
    }  // end loop over gauss points
  }

  // store EAS parameters
  if (eastype_ != soh8_easnone)
  {
    soh8_easupdate();

    // reset EAS internal force
    Core::LinAlg::SerialDenseMatrix* oldfeas = &easdata_.feas;
    oldfeas->putScalar(0.0);
  }
  solid_material()->update();

  return;
}


/*----------------------------------------------------------------------*
 | push forward of material to spatial stresses              dano 11/12 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::g_lto_ea(Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>* glstrain,
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>* defgrd,
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>* euler_almansi)
{
  // e = F^{T-1} . E . F^{-1}

  // rewrite Green-Lagrange strain in tensor notation
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> gl;
  gl(0, 0) = (*glstrain)(0);
  gl(0, 1) = 0.5 * (*glstrain)(3);
  gl(0, 2) = 0.5 * (*glstrain)(5);
  gl(1, 0) = gl(0, 1);
  gl(1, 1) = (*glstrain)(1);
  gl(1, 2) = 0.5 * (*glstrain)(4);
  gl(2, 0) = gl(0, 2);
  gl(2, 1) = gl(1, 2);
  gl(2, 2) = (*glstrain)(2);

  // inverse of deformation gradient
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invdefgrd;
  invdefgrd.invert((*defgrd));

  // (3x3) = (3x3) (3x3) (3x3)
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> temp;
  temp.multiply(gl, invdefgrd);
  (*euler_almansi).multiply_tn(invdefgrd, temp);

}  // GLtoEdata()


/*----------------------------------------------------------------------*
 | push forward of material to spatial stresses              dano 11/12 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::p_k2to_cauchy(Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>* stress,
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>* defgrd,
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8>* cauchystress)
{
  // calculate the Jacobi-deterinant
  const double detF = (*defgrd).determinant();

  // sigma = 1/J . F . S . F^T
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> pkstress;
  pkstress(0, 0) = (*stress)(0);
  pkstress(0, 1) = (*stress)(3);
  pkstress(0, 2) = (*stress)(5);
  pkstress(1, 0) = pkstress(0, 1);
  pkstress(1, 1) = (*stress)(1);
  pkstress(1, 2) = (*stress)(4);
  pkstress(2, 0) = pkstress(0, 2);
  pkstress(2, 1) = pkstress(1, 2);
  pkstress(2, 2) = (*stress)(2);

  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> temp;
  temp.multiply((1.0 / detF), (*defgrd), pkstress);
  (*cauchystress).multiply_nt(temp, (*defgrd));

}  // PK2toCauchy()

/*----------------------------------------------------------------------*
 |  Calculate consistent deformation gradient               seitz 04/14 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::calc_consistent_defgrd(
    const Core::LinAlg::Matrix<3, 3>& defgrd_disp, Core::LinAlg::Matrix<6, 1> glstrain_mod,
    Core::LinAlg::Matrix<3, 3>& defgrd_mod) const
{
  Core::LinAlg::Matrix<3, 3> R;       // rotation tensor
  Core::LinAlg::Matrix<3, 3> U_mod;   // modified right stretch tensor
  Core::LinAlg::Matrix<3, 3> U_disp;  // displacement-based right stretch tensor
  Core::LinAlg::Matrix<3, 3> EW;      // temporarily store eigenvalues
  Core::LinAlg::Matrix<3, 3> tmp;     // temporary matrix for matrix matrix matrix products
  Core::LinAlg::Matrix<3, 3> tmp2;    // temporary matrix for matrix matrix matrix products

  // ******************************************************************
  // calculate modified right stretch tensor
  // ******************************************************************
  for (int i = 0; i < 3; i++) U_mod(i, i) = 2. * glstrain_mod(i) + 1.;
  U_mod(0, 1) = glstrain_mod(3);
  U_mod(1, 0) = glstrain_mod(3);
  U_mod(1, 2) = glstrain_mod(4);
  U_mod(2, 1) = glstrain_mod(4);
  U_mod(0, 2) = glstrain_mod(5);
  U_mod(2, 0) = glstrain_mod(5);

  Core::LinAlg::syev(U_mod, EW, U_mod);
  for (int i = 0; i < 3; ++i) EW(i, i) = sqrt(EW(i, i));
  tmp.multiply(U_mod, EW);
  tmp2.multiply_nt(tmp, U_mod);
  U_mod.update(tmp2);

  // ******************************************************************
  // calculate displacement-based right stretch tensor
  // ******************************************************************
  U_disp.multiply_tn(defgrd_disp, defgrd_disp);

  Core::LinAlg::syev(U_disp, EW, U_disp);
  for (int i = 0; i < 3; ++i) EW(i, i) = sqrt(EW(i, i));
  tmp.multiply(U_disp, EW);
  tmp2.multiply_nt(tmp, U_disp);
  U_disp.update(tmp2);

  // ******************************************************************
  // compose consistent deformation gradient
  // ******************************************************************
  U_disp.invert();
  R.multiply(defgrd_disp, U_disp);
  defgrd_mod.multiply(R, U_mod);

  // you're done here
  return;
}

/*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | check the constitutive tensor and/or use the approximation as        |
 | elastic stiffness matrix                                  rauch 07/13|
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::evaluate_finite_difference_material_tangent(
    Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8>* stiffmatrix,
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>& stress, std::vector<double>& disp,
    const double detJ_w, const double detJ, const double detJ0, const double charelelength,
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, NUMDOF_SOH8>& bop,
    const Core::LinAlg::Matrix<6, NUMDOF_SOH8>& cb,
    const Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>& N_XYZ,
    const Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& T0invT,
    const std::vector<Core::LinAlg::SerialDenseMatrix>* M_GP,
    const Core::LinAlg::SerialDenseMatrix* alpha, Core::LinAlg::SerialDenseMatrix& M, const int gp,
    Teuchos::ParameterList& params)
{
  // build elastic stiffness matrix directly by finite differences

  std::shared_ptr<Mat::So3Material> so3mat = std::static_pointer_cast<Mat::So3Material>(material());

#ifdef MATERIALFDCHECK
  static Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8> stiffmatrix_analytical;
  static Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8> stiffmatrix_fd;
  if (gp == 0)
  {
    stiffmatrix_analytical.put_scalar(0.0);
    stiffmatrix_fd.put_scalar(0.0);
  }
  stiffmatrix_analytical.multiply_tn(detJ_w, bop, cb, 1.0);
#endif

  const double delta = charelelength * 1.0e-08;

  // matrices and vectors
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, NUMDOF_SOH8> cb_fd(true);
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> stress_fd(true);
  Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> finitedifference(true);

  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe;  // reference coord. of element
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xcurr;  // current   coord. of element
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xdisp;

  // get nodes
  Core::Nodes::Node** nodes = SoHex8::nodes();

  //////////////////////////////////////////////////////////////////////////////
  ////// evaluate partial derivatives of stress (S(d_n+delta) - S(d_n))/delta
  /////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////
  // loop over columns
  /////////////////////////////////////////////
  for (int i = 0; i < NUMDOF_SOH8; ++i)
  {
    // undo disturbance for disp[i-1]
    if (i > 0) disp[i - 1] -= delta;

    // disturb displacements
    disp[i] += delta;

    for (int k = 0; k < NUMNOD_SOH8; ++k)
    {
      const auto& x = nodes[k]->x();
      xrefe(k, 0) = x[0];
      xrefe(k, 1) = x[1];
      xrefe(k, 2) = x[2];

      xcurr(k, 0) = xrefe(k, 0) + disp[k * NODDOF_SOH8 + 0];
      xcurr(k, 1) = xrefe(k, 1) + disp[k * NODDOF_SOH8 + 1];
      xcurr(k, 2) = xrefe(k, 2) + disp[k * NODDOF_SOH8 + 2];
    }

    // build deformation gradient wrt to material configuration

    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd_fd(true);
    defgrd_fd.multiply_tt(xcurr, N_XYZ);


    // Right Cauchy-Green tensor = F^T * F
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> cauchygreen_fd(true);
    cauchygreen_fd.multiply_tn(defgrd_fd, defgrd_fd);

    // Green-Lagrange strains matrix E = 0.5 * (cauchygreen_fd - Identity)
    // GL strain vector glstrain_fd={E11,E22,E33,2*E12,2*E23,2*E31}
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> glstrain_fd(true);
    glstrain_fd(0) = 0.5 * (cauchygreen_fd(0, 0) - 1.0);
    glstrain_fd(1) = 0.5 * (cauchygreen_fd(1, 1) - 1.0);
    glstrain_fd(2) = 0.5 * (cauchygreen_fd(2, 2) - 1.0);
    glstrain_fd(3) = cauchygreen_fd(0, 1);
    glstrain_fd(4) = cauchygreen_fd(1, 2);
    glstrain_fd(5) = cauchygreen_fd(2, 0);

    // deformation gradient consistent with (potentially EAS-modified) GL strains
    // without eas this is equal to the regular defgrd_fd.
    Core::LinAlg::Matrix<3, 3> defgrd_fd_mod(defgrd_fd);

    // EAS technology: "enhance the strains"  ----------------------------- EAS
    if (eastype_ != soh8_easnone)
    {
      FOUR_C_THROW("be careful ! fdcheck has not been tested with EAS, yet! ");
      M.shape(Mat::NUM_STRESS_3D, neas_);
      // map local M to global, also enhancement is referred to element origin
      // M = detJ0/detJ T0^{-T} . M
      // Core::LinAlg::SerialDenseMatrix Mtemp(M); // temp M for Matrix-Matrix-Product
      // add enhanced strains = M . alpha to GL strains to "unlock" element
      switch (eastype_)
      {
        case Discret::Elements::SoHex8::soh8_easfull:
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
              soh8_easfull>(M.values(), detJ0 / detJ, T0invT.data(), (M_GP->at(gp)).values());
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, soh8_easfull, 1>(
              1.0, glstrain_fd.data(), 1.0, M.values(), alpha->values());
          break;
        case Discret::Elements::SoHex8::soh8_easmild:
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
              soh8_easmild>(M.values(), detJ0 / detJ, T0invT.data(), (M_GP->at(gp)).values());
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, soh8_easmild, 1>(
              1.0, glstrain_fd.data(), 1.0, M.values(), alpha->values());
          break;
        case Discret::Elements::SoHex8::soh8_eassosh8:
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D,
              soh8_eassosh8>(M.values(), detJ0 / detJ, T0invT.data(), (M_GP->at(gp)).values());
          Core::LinAlg::DenseFunctions::multiply<double, Mat::NUM_STRESS_3D, soh8_eassosh8, 1>(
              1.0, glstrain_fd.data(), 1.0, M.values(), alpha->values());
          break;
        case Discret::Elements::SoHex8::soh8_easnone:
          break;
        default:
          FOUR_C_THROW("Don't know what to do with EAS type %d", eastype_);
          break;
      }

      // calculate deformation gradient consistent with modified GL strain tensor
      if (std::static_pointer_cast<Mat::So3Material>(material())->needs_defgrd())
        calc_consistent_defgrd(defgrd_fd, glstrain_fd, defgrd_fd_mod);
    }  // ------------------------------------------------------------------ EAS

    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> cmat_fd;
    so3mat->evaluate(&defgrd_fd_mod, &glstrain_fd, params, &stress_fd, &cmat_fd, gp, id());

    // finite difference approximation of partial derivative
    //
    //      d S_ij,columnindex
    //
    //
    finitedifference.update(1.0, stress_fd, -1.0, stress);
    finitedifference.scale(1.0 / delta);

    /////////////////////////
    // loop over rows
    ////////////////////////
    for (int j = 0; j < Mat::NUM_STRESS_3D; ++j)
    {
      cb_fd(j, i) = finitedifference(j, 0);
    }  // j-loop (rows)


    // reset disp at last loop execution
    if (i == (NUMDOF_SOH8 - 1))
    {
      disp[i] -= delta;

      // reset xcurr
      for (int k = 0; k < NUMNOD_SOH8; ++k)
      {
        const auto& x = nodes[k]->x();
        xrefe(k, 0) = x[0];
        xrefe(k, 1) = x[1];
        xrefe(k, 2) = x[2];

        xcurr(k, 0) = xrefe(k, 0) + disp[k * NODDOF_SOH8 + 0];
        xcurr(k, 1) = xrefe(k, 1) + disp[k * NODDOF_SOH8 + 1];
        xcurr(k, 2) = xrefe(k, 2) + disp[k * NODDOF_SOH8 + 2];
      }
    }

  }  // i-loop (columns)
  /////////////////////////////// FD LOOP


  ///////////////////////////////////////
  // build approximated stiffness matrix
  ///////////////////////////////////////
  stiffmatrix->multiply_tn(detJ_w, bop, cb_fd, 1.0);
#ifdef MATERIALFDCHECK
  stiffmatrix_fd.multiply_tn(detJ_w, bop, cb_fd, 1.0);
#endif

  ///////////////////////////////////////
#ifdef MATERIALFDCHECK
  // after last gp was evaluated
  if (gp == (NUMGPT_SOH8 - 1))
  {
    Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8> errormatrix(true);

    // calc error (subtraction stiffmatrix - stiffmatrix_analytical)
    errormatrix.update(1.0, stiffmatrix_fd, -1.0, stiffmatrix_analytical);

    for (int i = 0; i < NUMDOF_SOH8; ++i)
    {
      for (int j = 0; j < NUMDOF_SOH8; ++j)
      {
        double relerror = abs(errormatrix(i, j)) / abs((stiffmatrix_analytical)(i, j));
        if (std::min(abs(errormatrix(i, j)), relerror) > delta * 1000.0)
        {
          std::cout << "ELEGID:" << this->Id() << "  gp: " << gp << "  ROW: " << i << "  COL: " << j
                    << "    REL. ERROR: " << relerror
                    << "    ABS. ERROR: " << abs(errormatrix(i, j))
                    << "    stiff. val: " << stiffmatrix_analytical(i, j)
                    << "    approx. val: " << stiffmatrix_fd(i, j) << std::endl;
        }
      }
    }  // check errors
  }  // if last gp of element is reached
#endif
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex8::get_cauchy_n_dir_and_derivatives_at_xi(
    const Core::LinAlg::Matrix<3, 1>& xi, const std::vector<double>& disp,
    const Core::LinAlg::Matrix<3, 1>& n, const Core::LinAlg::Matrix<3, 1>& dir,
    double& cauchy_n_dir, Core::LinAlg::SerialDenseMatrix* d_cauchyndir_dd,
    Core::LinAlg::SerialDenseMatrix* d2_cauchyndir_dd2,
    Core::LinAlg::SerialDenseMatrix* d2_cauchyndir_dd_dn,
    Core::LinAlg::SerialDenseMatrix* d2_cauchyndir_dd_ddir,
    Core::LinAlg::SerialDenseMatrix* d2_cauchyndir_dd_dxi,
    Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dn, Core::LinAlg::Matrix<3, 1>* d_cauchyndir_ddir,
    Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dxi, const std::vector<double>* temp,
    Core::LinAlg::SerialDenseMatrix* d_cauchyndir_dT,
    Core::LinAlg::SerialDenseMatrix* d2_cauchyndir_dd_dT, const double* concentration,
    double* d_cauchyndir_dc)
{
  FOUR_C_ASSERT_ALWAYS(eastype_ == soh8_easnone && !Prestress::is_mulf(),
      "Evaluation of the Cauchy stress is not possible for EAS-elements or MULF prestressing.");
  if (temp || d_cauchyndir_dT || d2_cauchyndir_dd_dT)
    FOUR_C_THROW("Thermo-elastic Nitsche contact not yet implemented in so hex8");

  cauchy_n_dir = 0.0;

  static Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe(true);  // reference coord. of element
  static Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xcurr(true);  // current  coord. of element
  xrefe.clear();
  xcurr.clear();

  for (int i = 0; i < NUMNOD_SOH8; ++i)
  {
    const auto& x = nodes()[i]->x();
    for (int d = 0; d < NUMDIM_SOH8; ++d)
    {
      xrefe(i, d) = x[d];
      xcurr(i, d) = xrefe(i, d) + disp[i * NODDOF_SOH8 + d];
    }
  }

  static Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> deriv(true);
  deriv.clear();
  Core::FE::shape_function_deriv1<Core::FE::CellType::hex8>(xi, deriv);

  static Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_XYZ(true);
  static Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invJ(true);
  invJ.multiply(1.0, deriv, xrefe, 0.0);
  invJ.invert();
  N_XYZ.multiply(1.0, invJ, deriv, 0.0);
  static Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd(true);
  defgrd.multiply_tt(1.0, xcurr, N_XYZ, 0.0);

  // linearization of deformation gradient F w.r.t. displacements
  static Core::LinAlg::Matrix<9, NUMDOF_SOH8> d_F_dd(true);
  d_F_dd.clear();
  if (d_cauchyndir_dd || d2_cauchyndir_dd_dn || d2_cauchyndir_dd_ddir || d2_cauchyndir_dd2 ||
      d2_cauchyndir_dd_dxi)
  {
    for (int i = 0; i < NUMNOD_SOH8; ++i)
    {
      d_F_dd(0, NODDOF_SOH8 * i + 0) = N_XYZ(0, i);
      d_F_dd(1, NODDOF_SOH8 * i + 1) = N_XYZ(1, i);
      d_F_dd(2, NODDOF_SOH8 * i + 2) = N_XYZ(2, i);
      d_F_dd(3, NODDOF_SOH8 * i + 0) = N_XYZ(1, i);
      d_F_dd(4, NODDOF_SOH8 * i + 1) = N_XYZ(2, i);
      d_F_dd(5, NODDOF_SOH8 * i + 0) = N_XYZ(2, i);
      d_F_dd(6, NODDOF_SOH8 * i + 1) = N_XYZ(0, i);
      d_F_dd(7, NODDOF_SOH8 * i + 2) = N_XYZ(1, i);
      d_F_dd(8, NODDOF_SOH8 * i + 2) = N_XYZ(0, i);
    }
  }

  static Core::LinAlg::Matrix<9, 1> d_cauchyndir_dF(true);
  static Core::LinAlg::Matrix<9, 9> d2_cauchyndir_dF2(true);
  static Core::LinAlg::Matrix<9, NUMDIM_SOH8> d2_cauchyndir_dF_dn(true);
  static Core::LinAlg::Matrix<9, NUMDIM_SOH8> d2_cauchyndir_dF_ddir(true);

  solid_material()->evaluate_cauchy_n_dir_and_derivatives(defgrd, n, dir, cauchy_n_dir,
      d_cauchyndir_dn, d_cauchyndir_ddir, &d_cauchyndir_dF, &d2_cauchyndir_dF2,
      &d2_cauchyndir_dF_dn, &d2_cauchyndir_dF_ddir, -1, id(), concentration, nullptr, nullptr,
      nullptr);

  if (d_cauchyndir_dd)
  {
    d_cauchyndir_dd->reshape(NUMDOF_SOH8, 1);
    Core::LinAlg::Matrix<NUMDOF_SOH8, 1> d_cauchyndir_dd_mat(d_cauchyndir_dd->values(), true);
    d_cauchyndir_dd_mat.multiply_tn(1.0, d_F_dd, d_cauchyndir_dF, 0.0);
  }

  if (d2_cauchyndir_dd_dn)
  {
    d2_cauchyndir_dd_dn->reshape(NUMDOF_SOH8, NUMDIM_SOH8);
    Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDIM_SOH8> d2_cauchyndir_dd_dn_mat(
        d2_cauchyndir_dd_dn->values(), true);
    d2_cauchyndir_dd_dn_mat.multiply_tn(1.0, d_F_dd, d2_cauchyndir_dF_dn, 0.0);
  }

  if (d2_cauchyndir_dd_ddir)
  {
    d2_cauchyndir_dd_ddir->reshape(NUMDOF_SOH8, NUMDIM_SOH8);
    Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDIM_SOH8> d2_cauchyndir_dd_ddir_mat(
        d2_cauchyndir_dd_ddir->values(), true);
    d2_cauchyndir_dd_ddir_mat.multiply_tn(1.0, d_F_dd, d2_cauchyndir_dF_ddir, 0.0);
  }

  if (d2_cauchyndir_dd2)
  {
    d2_cauchyndir_dd2->reshape(NUMDOF_SOH8, NUMDOF_SOH8);
    Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8> d2_cauchyndir_dd2_mat(
        d2_cauchyndir_dd2->values(), true);
    static Core::LinAlg::Matrix<9, NUMDOF_SOH8> d2_cauchyndir_dF_2d_F_dd(true);
    d2_cauchyndir_dF_2d_F_dd.multiply(1.0, d2_cauchyndir_dF2, d_F_dd, 0.0);
    d2_cauchyndir_dd2_mat.multiply_tn(1.0, d_F_dd, d2_cauchyndir_dF_2d_F_dd, 0.0);
  }

  // prepare evaluation of d_cauchyndir_dxi or d2_cauchyndir_dd_dxi
  static Core::LinAlg::Matrix<9, NUMDIM_SOH8> d_F_dxi(true);
  static Core::LinAlg::Matrix<Core::FE::DisTypeToNumDeriv2<Core::FE::CellType::hex8>::numderiv2,
      NUMNOD_SOH8>
      deriv2(true);
  d_F_dxi.clear();
  deriv2.clear();

  if (d_cauchyndir_dxi or d2_cauchyndir_dd_dxi)
  {
    Core::FE::shape_function_deriv2<Core::FE::CellType::hex8>(xi, deriv2);

    static Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xXF(true);
    static Core::LinAlg::Matrix<NUMDIM_SOH8,
        Core::FE::DisTypeToNumDeriv2<Core::FE::CellType::hex8>::numderiv2>
        xXFsec(true);
    xXF.update(1.0, xcurr, 0.0);
    xXF.multiply_nt(-1.0, xrefe, defgrd, 1.0);
    xXFsec.multiply_tt(1.0, xXF, deriv2, 0.0);

    for (int a = 0; a < NUMDIM_SOH8; ++a)
    {
      for (int b = 0; b < NUMDIM_SOH8; ++b)
      {
        d_F_dxi(VoigtMapping::non_symmetric_tensor_to_voigt9_index(a, b), 0) +=
            xXFsec(a, 0) * invJ(b, 0) + xXFsec(a, 3) * invJ(b, 1) + xXFsec(a, 4) * invJ(b, 2);
        d_F_dxi(VoigtMapping::non_symmetric_tensor_to_voigt9_index(a, b), 1) +=
            xXFsec(a, 3) * invJ(b, 0) + xXFsec(a, 1) * invJ(b, 1) + xXFsec(a, 5) * invJ(b, 2);
        d_F_dxi(VoigtMapping::non_symmetric_tensor_to_voigt9_index(a, b), 2) +=
            xXFsec(a, 4) * invJ(b, 0) + xXFsec(a, 5) * invJ(b, 1) + xXFsec(a, 2) * invJ(b, 2);
      }
    }
  }

  if (d_cauchyndir_dxi)
  {
    d_cauchyndir_dxi->multiply_tn(1.0, d_F_dxi, d_cauchyndir_dF, 0.0);
  }

  if (d2_cauchyndir_dd_dxi)
  {
    d2_cauchyndir_dd_dxi->reshape(NUMDOF_SOH8, NUMDIM_SOH8);
    Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDIM_SOH8> d2_cauchyndir_dd_dxi_mat(
        d2_cauchyndir_dd_dxi->values(), true);

    static Core::LinAlg::Matrix<Core::FE::DisTypeToNumDeriv2<Core::FE::CellType::hex8>::numderiv2,
        NUMNOD_SOH8>
        deriv2(true);
    deriv2.clear();
    Core::FE::shape_function_deriv2<Core::FE::CellType::hex8>(xi, deriv2);

    static Core::LinAlg::Matrix<Core::FE::DisTypeToNumDeriv2<Core::FE::CellType::hex8>::numderiv2,
        NUMDIM_SOH8>
        Xsec(true);
    static Core::LinAlg::Matrix<NUMNOD_SOH8,
        Core::FE::DisTypeToNumDeriv2<Core::FE::CellType::hex8>::numderiv2>
        N_XYZ_Xsec(true);
    Xsec.multiply(1.0, deriv2, xrefe, 0.0);
    N_XYZ_Xsec.multiply_tt(1.0, N_XYZ, Xsec, 0.0);

    static Core::LinAlg::Matrix<9, NUMDOF_SOH8> d2_cauchyndir_dF_2d_F_dd(true);
    d2_cauchyndir_dF_2d_F_dd.multiply(1.0, d2_cauchyndir_dF2, d_F_dd, 0.0);
    d2_cauchyndir_dd_dxi_mat.multiply_tn(1.0, d2_cauchyndir_dF_2d_F_dd, d_F_dxi, 0.0);

    static Core::LinAlg::Matrix<9, NUMDIM_SOH8 * NUMDOF_SOH8> d2_F_dxi_dd(true);
    d2_F_dxi_dd.clear();
    for (int i = 0; i < NUMDIM_SOH8; ++i)
    {
      for (int j = 0; j < NUMDIM_SOH8; ++j)
      {
        for (int k = 0; k < NUMNOD_SOH8; ++k)
        {
          d2_F_dxi_dd(VoigtMapping::non_symmetric_tensor_to_voigt9_index(i, j),
              NODDOF_SOH8 * (NODDOF_SOH8 * k + i) + 0) +=
              deriv2(0, k) * invJ(j, 0) + deriv2(3, k) * invJ(j, 1) + deriv2(4, k) * invJ(j, 2) -
              N_XYZ_Xsec(k, 0) * invJ(j, 0) - N_XYZ_Xsec(k, 3) * invJ(j, 1) -
              N_XYZ_Xsec(k, 4) * invJ(j, 2);

          d2_F_dxi_dd(VoigtMapping::non_symmetric_tensor_to_voigt9_index(i, j),
              NODDOF_SOH8 * (NODDOF_SOH8 * k + i) + 1) +=
              deriv2(3, k) * invJ(j, 0) + deriv2(1, k) * invJ(j, 1) + deriv2(5, k) * invJ(j, 2) -
              N_XYZ_Xsec(k, 3) * invJ(j, 0) - N_XYZ_Xsec(k, 1) * invJ(j, 1) -
              N_XYZ_Xsec(k, 5) * invJ(j, 2);

          d2_F_dxi_dd(VoigtMapping::non_symmetric_tensor_to_voigt9_index(i, j),
              NODDOF_SOH8 * (NODDOF_SOH8 * k + i) + 2) +=
              deriv2(4, k) * invJ(j, 0) + deriv2(5, k) * invJ(j, 1) + deriv2(2, k) * invJ(j, 2) -
              N_XYZ_Xsec(k, 4) * invJ(j, 0) - N_XYZ_Xsec(k, 5) * invJ(j, 1) -
              N_XYZ_Xsec(k, 2) * invJ(j, 2);

          for (int l = 0; l < NUMDIM_SOH8; ++l)
          {
            d2_cauchyndir_dd_dxi_mat(k * 3 + i, l) +=
                d_cauchyndir_dF(VoigtMapping::non_symmetric_tensor_to_voigt9_index(i, j), 0) *
                d2_F_dxi_dd(VoigtMapping::non_symmetric_tensor_to_voigt9_index(i, j),
                    NODDOF_SOH8 * (NODDOF_SOH8 * k + i) + l);
          }
        }
      }
    }
  }

  if (d_cauchyndir_dc != nullptr)
  {
    static Core::LinAlg::Matrix<9, 1> d_F_dc(true);
    solid_material()->evaluate_linearization_od(defgrd, *concentration, &d_F_dc);
    *d_cauchyndir_dc = d_cauchyndir_dF.dot(d_F_dc);
  }
}

FOUR_C_NAMESPACE_CLOSE
