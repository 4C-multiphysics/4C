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
#include "4C_fem_general_utils_integration.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_so3_hex27.hpp"
#include "4C_so3_prestress.hpp"
#include "4C_so3_prestress_service.hpp"
#include "4C_so3_utils.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_structure_new_model_evaluator_data.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                                       |
 *----------------------------------------------------------------------*/
int Discret::Elements::SoHex27::evaluate(Teuchos::ParameterList& params,
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

  Core::LinAlg::Matrix<NUMDOF_SOH27, NUMDOF_SOH27> elemat1(elemat1_epetra.values(), true);
  Core::LinAlg::Matrix<NUMDOF_SOH27, NUMDOF_SOH27> elemat2(elemat2_epetra.values(), true);
  Core::LinAlg::Matrix<NUMDOF_SOH27, 1> elevec1(elevec1_epetra.values(), true);
  Core::LinAlg::Matrix<NUMDOF_SOH27, 1> elevec2(elevec2_epetra.values(), true);
  Core::LinAlg::Matrix<NUMDOF_SOH27, 1> elevec3(elevec3_epetra.values(), true);

  // start with "none"
  Discret::Elements::SoHex27::ActionType act = SoHex27::none;

  // get the required action
  std::string action = params.get<std::string>("action", "none");
  if (action == "none")
    FOUR_C_THROW("No action supplied");
  else if (action == "calc_struct_linstiff")
    act = SoHex27::calc_struct_linstiff;
  else if (action == "calc_struct_nlnstiff")
    act = SoHex27::calc_struct_nlnstiff;
  else if (action == "calc_struct_internalforce")
    act = SoHex27::calc_struct_internalforce;
  else if (is_params_interface() &&
           params_interface().get_action_type() == Core::Elements::struct_calc_internalinertiaforce)
    act = SoHex27::calc_struct_internalinertiaforce;
  else if (action == "calc_struct_linstiffmass")
    act = SoHex27::calc_struct_linstiffmass;
  else if (action == "calc_struct_nlnstiffmass")
    act = SoHex27::calc_struct_nlnstiffmass;
  else if (action == "calc_struct_nlnstifflmass")
    act = SoHex27::calc_struct_nlnstifflmass;
  else if (action == "calc_struct_stress")
    act = SoHex27::calc_struct_stress;
  else if (action == "calc_struct_eleload")
    act = SoHex27::calc_struct_eleload;
  else if (action == "calc_struct_fsiload")
    act = SoHex27::calc_struct_fsiload;
  else if (action == "calc_struct_update_istep")
    act = SoHex27::calc_struct_update_istep;
  else if (action == "calc_struct_prestress_update")
    act = SoHex27::prestress_update;
  else if (action == "calc_struct_reset_istep")
    act = SoHex27::calc_struct_reset_istep;
  else if (action == "calc_struct_energy")
    act = SoHex27::calc_struct_energy;
  else if (action == "multi_readrestart")
    act = SoHex27::multi_readrestart;
  else if (action == "multi_calc_dens")
    act = SoHex27::multi_calc_dens;
  else if (action == "calc_struct_recover")
    return 0;
  else if (action == "calc_struct_predict")
    return 0;
  else
    FOUR_C_THROW("Unknown type of action for So_hex27: %s", action.c_str());
  // what should the element do
  switch (act)
  {
    // linear stiffness
    case calc_struct_linstiff:
    {
      // need current displacement and residual forces
      std::vector<double> mydisp(lm.size());
      for (double& i : mydisp) i = 0.0;
      std::vector<double> myres(lm.size());
      for (double& myre : myres) myre = 0.0;

      std::vector<double> mydispmat(lm.size(), 0.0);

      soh27_nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, &elemat1, nullptr,
          &elevec1, nullptr, nullptr, nullptr, nullptr, nullptr, params, Inpar::Solid::stress_none,
          Inpar::Solid::strain_none, Inpar::Solid::strain_none);
    }
    break;

    // nonlinear stiffness and internal force vector
    case calc_struct_nlnstiff:
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
      Core::LinAlg::Matrix<NUMDOF_SOH27, NUMDOF_SOH27>* matptr = nullptr;
      if (elemat1.is_initialized()) matptr = &elemat1;

      std::vector<double> mydispmat(lm.size(), 0.0);

      // special case: geometrically linear
      if (kintype_ == Inpar::Solid::KinemType::linear)
      {
        soh27_linstiffmass(lm, mydisp, myres, matptr, nullptr, &elevec1, nullptr, nullptr, nullptr,
            params, Inpar::Solid::stress_none, Inpar::Solid::strain_none,
            Inpar::Solid::strain_none);
      }
      // standard is: geometrically non-linear with Total Lagrangean approach
      else if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
      {
        soh27_nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, matptr, nullptr,
            &elevec1, nullptr, &elevec3, nullptr, nullptr, nullptr, params,
            Inpar::Solid::stress_none, Inpar::Solid::strain_none, Inpar::Solid::strain_none);
      }
      else
        FOUR_C_THROW("unknown kinematic type");
    }
    break;

    // internal force vector only
    case calc_struct_internalforce:
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
      Core::LinAlg::Matrix<NUMDOF_SOH27, NUMDOF_SOH27> myemat(true);

      std::vector<double> mydispmat(lm.size(), 0.0);

      // special case: geometrically linear
      if (kintype_ == Inpar::Solid::KinemType::linear)
      {
        soh27_linstiffmass(lm, mydisp, myres, &myemat, nullptr, &elevec1, nullptr, nullptr, nullptr,
            params, Inpar::Solid::stress_none, Inpar::Solid::strain_none,
            Inpar::Solid::strain_none);
      }
      // standard is: geometrically non-linear with Total Lagrangean approach
      else if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
      {
        soh27_nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, &myemat, nullptr,
            &elevec1, nullptr, nullptr, nullptr, nullptr, nullptr, params,
            Inpar::Solid::stress_none, Inpar::Solid::strain_none, Inpar::Solid::strain_none);
      }
      else
        FOUR_C_THROW("unknown kinematic type");
    }
    break;

    // linear stiffness and consistent mass matrix
    case calc_struct_linstiffmass:
      FOUR_C_THROW("Case 'calc_struct_linstiffmass' not yet implemented");
      break;

    // nonlinear stiffness, internal force vector, and consistent mass matrix
    case calc_struct_nlnstiffmass:
    case calc_struct_nlnstifflmass:
    case calc_struct_internalinertiaforce:
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
      Core::LinAlg::Matrix<NUMDOF_SOH27, NUMDOF_SOH27> mass_matrix_evaluate;
      if (act != SoHex27::calc_struct_internalinertiaforce) mass_matrix_evaluate.set_view(elemat2);

      std::vector<double> mydispmat(lm.size(), 0.0);

      if (act == SoHex27::calc_struct_internalinertiaforce)
      {
        soh27_nlnstiffmass(lm, mydisp, &myvel, &myacc, myres, mydispmat, nullptr,
            &mass_matrix_evaluate, &elevec1, &elevec2, nullptr, nullptr, nullptr, nullptr, params,
            Inpar::Solid::stress_none, Inpar::Solid::strain_none, Inpar::Solid::strain_none);
      }
      else
      {
        // special case: geometrically linear
        if (kintype_ == Inpar::Solid::KinemType::linear)
        {
          soh27_linstiffmass(lm, mydisp, myres, &elemat1, &elemat2, &elevec1, nullptr, nullptr,
              nullptr, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none,
              Inpar::Solid::strain_none);
        }
        // standard is: geometrically non-linear with Total Lagrangean approach
        else if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
        {
          soh27_nlnstiffmass(lm, mydisp, &myvel, &myacc, myres, mydispmat, &elemat1, &elemat2,
              &elevec1, &elevec2, &elevec3, nullptr, nullptr, nullptr, params,
              Inpar::Solid::stress_none, Inpar::Solid::strain_none, Inpar::Solid::strain_none);
        }
        else
          FOUR_C_THROW("unknown kinematic type");
      }

      if (act == calc_struct_nlnstifflmass) soh27_lumpmass(&elemat2);

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
        Core::LinAlg::Matrix<NUMDOF_SOH27, 1> acceleration(true);
        for (unsigned int i_dof = 0; i_dof < NUMDOF_SOH27; i_dof++)
          acceleration(i_dof) = myacc[i_dof];
        Core::LinAlg::Matrix<NUMDOF_SOH27, 1> internal_inertia(true);
        internal_inertia.multiply(mass_matrix_evaluate, acceleration);
        elevec2 += internal_inertia;
      }
    }
    break;

    // evaluate stresses and strains at gauss points
    case calc_struct_stress:
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
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);
      Core::LinAlg::Matrix<NUMGPT_SOH27, Mat::NUM_STRESS_3D> stress;
      Core::LinAlg::Matrix<NUMGPT_SOH27, Mat::NUM_STRESS_3D> strain;
      Core::LinAlg::Matrix<NUMGPT_SOH27, Mat::NUM_STRESS_3D> plstrain;
      auto iostress = params.get<Inpar::Solid::StressType>("iostress", Inpar::Solid::stress_none);
      auto iostrain = params.get<Inpar::Solid::StrainType>("iostrain", Inpar::Solid::strain_none);
      auto ioplstrain =
          params.get<Inpar::Solid::StrainType>("ioplstrain", Inpar::Solid::strain_none);

      std::vector<double> mydispmat(lm.size(), 0.0);

      // special case: geometrically linear
      if (kintype_ == Inpar::Solid::KinemType::linear)
      {
        soh27_linstiffmass(lm, mydisp, myres, nullptr, nullptr, nullptr, &stress, &strain,
            &plstrain, params, iostress, iostrain, ioplstrain);
      }
      // standard is: geometrically non-linear with Total Lagrangean approach
      else if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
      {
        soh27_nlnstiffmass(lm, mydisp, nullptr, nullptr, myres, mydispmat, nullptr, nullptr,
            nullptr, nullptr, nullptr, &stress, &strain, &plstrain, params, iostress, iostrain,
            ioplstrain);
      }
      else
        FOUR_C_THROW("unknown kinematic type");

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
    break;

    case prestress_update:
    {
      time_ = params.get<double>("total time");
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get displacement state");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      // build incremental def gradient for every gauss point
      Core::LinAlg::SerialDenseMatrix gpdefgrd(NUMGPT_SOH27, 9);
      def_gradient(mydisp, gpdefgrd, *prestress_);

      // update deformation gradient and put back to storage
      Core::LinAlg::Matrix<3, 3> deltaF;
      Core::LinAlg::Matrix<3, 3> Fhist;
      Core::LinAlg::Matrix<3, 3> Fnew;
      for (int gp = 0; gp < NUMGPT_SOH27; ++gp)
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
    }
    break;

    case calc_struct_eleload:
      FOUR_C_THROW("this method is not supposed to evaluate a load, use evaluate_neumann(...)");
      break;

    case calc_struct_fsiload:
      FOUR_C_THROW("Case not yet implemented");
      break;

    case calc_struct_update_istep:
    {
      // Update of history for materials
      solid_material()->update();
    }
    break;

    case calc_struct_reset_istep:
    {
      // Reset of history (if needed)
      solid_material()->reset_step();
    }
    break;

    //==================================================================================
    case calc_struct_energy:
    {
      // initialization of internal energy
      double intenergy = 0.0;

      // shape functions and Gauss weights
      const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27>> derivs =
          soh27_derivs();
      const static std::vector<double> weights = soh27_weights();

      // get displacements of this processor
      std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
          discretization.get_state("displacement");
      if (disp == nullptr) FOUR_C_THROW("Cannot get state displacement vector");

      // get displacements of this element
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      // update element geometry
      Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xrefe;  // material coord. of element
      Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xcurr;  // current  coord. of element


      for (int i = 0; i < NUMNOD_SOH27; ++i)
      {
        xrefe(i, 0) = nodes()[i]->x()[0];
        xrefe(i, 1) = nodes()[i]->x()[1];
        xrefe(i, 2) = nodes()[i]->x()[2];

        xcurr(i, 0) = xrefe(i, 0) + mydisp[i * NODDOF_SOH27 + 0];
        xcurr(i, 1) = xrefe(i, 1) + mydisp[i * NODDOF_SOH27 + 1];
        xcurr(i, 2) = xrefe(i, 2) + mydisp[i * NODDOF_SOH27 + 2];
      }

      // loop over all Gauss points
      for (int gp = 0; gp < NUMGPT_SOH27; gp++)
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
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27> N_XYZ(true);
        N_XYZ.multiply(invJ_[gp], derivs[gp]);

        // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> defgrd(true);
        defgrd.multiply_tt(xcurr, N_XYZ);

        // right Cauchy-Green tensor = F^T * F
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> cauchygreen;
        cauchygreen.multiply_tn(defgrd, defgrd);

        // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
        // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> glstrain;
        glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
        glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
        glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
        glstrain(3) = cauchygreen(0, 1);
        glstrain(4) = cauchygreen(1, 2);
        glstrain(5) = cauchygreen(2, 0);

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

    case multi_calc_dens:
    {
      soh27_homog(params);
    }
    break;


    // read restart of microscale
    case multi_readrestart:
    {
      soh27_read_restart_multi();
    }
    break;

    default:
      FOUR_C_THROW("Unknown type of action for So_hex27");
  }
  return 0;
}



/*----------------------------------------------------------------------*
  |  Integrate a Volume Neumann boundary condition (public)               |
  *----------------------------------------------------------------------*/
int Discret::Elements::SoHex27::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  // get values and switches from the condition
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition.parameters().get<std::vector<double>>("VAL");

  /*
  **    TIME CURVE BUSINESS
  */
  // find out whether we will use a time curve
  const double time = std::invoke(
      [&]()
      {
        if (is_params_interface())
          return str_params_interface().get_total_time();
        else
          return params.get("total time", -1.0);
      });

  // ensure that at least as many curves/functs as dofs are available
  if (int(onoff.size()) < NUMDIM_SOH27)
    FOUR_C_THROW("Fewer functions or curves defined than the element has dofs.");

  for (int checkdof = NUMDIM_SOH27; checkdof < int(onoff.size()); ++checkdof)
  {
    if (onoff[checkdof] != 0)
      FOUR_C_THROW(
          "Number of Dimensions in Neumann_Evaluation is 3. Further DoFs are not considered.");
  }

  // (SPATIAL) FUNCTION BUSINESS
  const auto funct = condition.parameters().get<std::vector<Core::IO::Noneable<int>>>("FUNCT");
  Core::LinAlg::Matrix<NUMDIM_SOH27, 1> xrefegp(false);

  /* ============================================================================*
  ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for HEX_27 with 27 GAUSS POINTS*
  ** ============================================================================*/
  const static std::vector<Core::LinAlg::Matrix<NUMNOD_SOH27, 1>> shapefcts = soh27_shapefcts();
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27>> derivs =
      soh27_derivs();
  const static std::vector<double> gpweights = soh27_weights();
  /* ============================================================================*/

  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xrefe;  // material coord. of element
  for (int i = 0; i < NUMNOD_SOH27; ++i)
  {
    const auto& x = nodes()[i]->x();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];
  }
  /* ================================================= Loop over Gauss Points */
  for (int gp = 0; gp < NUMGPT_SOH27; ++gp)
  {
    // compute the Jacobian matrix
    Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> jac;
    jac.multiply(derivs[gp], xrefe);

    // compute determinant of Jacobian
    const double detJ = jac.determinant();
    if (detJ == 0.0)
      FOUR_C_THROW("ZERO JACOBIAN DETERMINANT");
    else if (detJ < 0.0)
      FOUR_C_THROW("NEGATIVE JACOBIAN DETERMINANT");

    // material/reference co-ordinates of Gauss point
    for (int dim = 0; dim < NUMDIM_SOH27; dim++)
    {
      xrefegp(dim) = 0.0;
      for (int nodid = 0; nodid < NUMNOD_SOH27; ++nodid)
        xrefegp(dim) += shapefcts[gp](nodid) * xrefe(nodid, dim);
    }

    // integration factor
    const double fac = gpweights[gp] * detJ;
    // distribute/add over element load vector
    for (int dim = 0; dim < NUMDIM_SOH27; dim++)
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

        double dim_fac = val[dim] * fac * functfac;
        for (int nodid = 0; nodid < NUMNOD_SOH27; ++nodid)
        {
          elevec1[nodid * NUMDIM_SOH27 + dim] += shapefcts[gp](nodid) * dim_fac;
        }
      }
    }

  } /* ==================================================== end of Loop over GP */

  return 0;
}  // Discret::Elements::So_hex27::evaluate_neumann


/*----------------------------------------------------------------------*
 |  init the element jacobian mapping (protected)                       |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex27::init_jacobian_mapping()
{
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27>> derivs =
      soh27_derivs();
  Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xrefe;
  for (int i = 0; i < NUMNOD_SOH27; ++i)
  {
    xrefe(i, 0) = nodes()[i]->x()[0];
    xrefe(i, 1) = nodes()[i]->x()[1];
    xrefe(i, 2) = nodes()[i]->x()[2];
  }
  invJ_.resize(NUMGPT_SOH27);
  detJ_.resize(NUMGPT_SOH27);
  for (int gp = 0; gp < NUMGPT_SOH27; ++gp)
  {
    // invJ_[gp].Shape(NUMDIM_SOH27,NUMDIM_SOH27);
    invJ_[gp].multiply(derivs[gp], xrefe);
    detJ_[gp] = invJ_[gp].invert();
    if (detJ_[gp] == 0.0)
      FOUR_C_THROW("ZERO JACOBIAN DETERMINANT");
    else if (detJ_[gp] < 0.0)
      FOUR_C_THROW("NEGATIVE JACOBIAN DETERMINANT");

    if (Prestress::is_mulf_active(time_, pstype_, pstime_))
      if (!(prestress_->is_init()))
        prestress_->matrixto_storage(gp, invJ_[gp], prestress_->j_history());
  }

  if (Prestress::is_mulf_active(time_, pstype_, pstime_)) prestress_->is_init() = true;

  return;
}


/*----------------------------------------------------------------------*
 |  evaluate the element (private)                           popp 09/11 |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex27::soh27_linstiffmass(std::vector<int>& lm,  // location matrix
    std::vector<double>& disp,                                             // current displacements
    std::vector<double>& residual,                                         // current residual displ
    Core::LinAlg::Matrix<NUMDOF_SOH27, NUMDOF_SOH27>* stiffmatrix,  // element stiffness matrix
    Core::LinAlg::Matrix<NUMDOF_SOH27, NUMDOF_SOH27>* massmatrix,   // element mass matrix
    Core::LinAlg::Matrix<NUMDOF_SOH27, 1>* force,                   // element internal force vector
    Core::LinAlg::Matrix<NUMGPT_SOH27, Mat::NUM_STRESS_3D>* elestress,    // stresses at GP
    Core::LinAlg::Matrix<NUMGPT_SOH27, Mat::NUM_STRESS_3D>* elestrain,    // strains at GP
    Core::LinAlg::Matrix<NUMGPT_SOH27, Mat::NUM_STRESS_3D>* eleplstrain,  // plastic strains at GP
    Teuchos::ParameterList& params,             // algorithmic parameters e.g. time
    const Inpar::Solid::StressType iostress,    // stress output option
    const Inpar::Solid::StrainType iostrain,    // strain output option
    const Inpar::Solid::StrainType ioplstrain)  // plastic strain output option
{
  /* ============================================================================*
  ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for HEX_27 with 27 GAUSS POINTS*
  ** ============================================================================*/
  const static std::vector<Core::LinAlg::Matrix<NUMNOD_SOH27, 1>> shapefcts = soh27_shapefcts();
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27>> derivs =
      soh27_derivs();
  const static std::vector<double> gpweights = soh27_weights();
  /* ============================================================================*/

  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xrefe;  // material coord. of element
  Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xcurr;  // current  coord. of element
  Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xdisp;


  for (int i = 0; i < NUMNOD_SOH27; ++i)
  {
    const auto& x = nodes()[i]->x();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];

    xcurr(i, 0) = xrefe(i, 0) + disp[i * NODDOF_SOH27 + 0];
    xcurr(i, 1) = xrefe(i, 1) + disp[i * NODDOF_SOH27 + 1];
    xcurr(i, 2) = xrefe(i, 2) + disp[i * NODDOF_SOH27 + 2];
  }

  Core::LinAlg::Matrix<NUMDOF_SOH27, 1> nodaldisp;
  for (int i = 0; i < NUMDOF_SOH27; ++i)
  {
    nodaldisp(i, 0) = disp[i];
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27> N_XYZ;
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> defgrd(true);
  for (int gp = 0; gp < NUMGPT_SOH27; ++gp)
  {
    /* get the inverse of the Jacobian matrix which looks like:
    **            [ x_,r  y_,r  z_,r ]^-1
    **     J^-1 = [ x_,s  y_,s  z_,s ]
    **            [ x_,t  y_,t  z_,t ]
    */
    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 * N_rst
    N_XYZ.multiply(invJ_[gp], derivs[gp]);
    double detJ = detJ_[gp];

    // set to initial state as test to receive a linear solution
    for (int i = 0; i < 3; ++i) defgrd(i, i) = 1.0;

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
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, NUMDOF_SOH27> bop;
    for (int i = 0; i < NUMNOD_SOH27; ++i)
    {
      bop(0, NODDOF_SOH27 * i + 0) = defgrd(0, 0) * N_XYZ(0, i);
      bop(0, NODDOF_SOH27 * i + 1) = defgrd(1, 0) * N_XYZ(0, i);
      bop(0, NODDOF_SOH27 * i + 2) = defgrd(2, 0) * N_XYZ(0, i);
      bop(1, NODDOF_SOH27 * i + 0) = defgrd(0, 1) * N_XYZ(1, i);
      bop(1, NODDOF_SOH27 * i + 1) = defgrd(1, 1) * N_XYZ(1, i);
      bop(1, NODDOF_SOH27 * i + 2) = defgrd(2, 1) * N_XYZ(1, i);
      bop(2, NODDOF_SOH27 * i + 0) = defgrd(0, 2) * N_XYZ(2, i);
      bop(2, NODDOF_SOH27 * i + 1) = defgrd(1, 2) * N_XYZ(2, i);
      bop(2, NODDOF_SOH27 * i + 2) = defgrd(2, 2) * N_XYZ(2, i);
      /* ~~~ */
      bop(3, NODDOF_SOH27 * i + 0) = defgrd(0, 0) * N_XYZ(1, i) + defgrd(0, 1) * N_XYZ(0, i);
      bop(3, NODDOF_SOH27 * i + 1) = defgrd(1, 0) * N_XYZ(1, i) + defgrd(1, 1) * N_XYZ(0, i);
      bop(3, NODDOF_SOH27 * i + 2) = defgrd(2, 0) * N_XYZ(1, i) + defgrd(2, 1) * N_XYZ(0, i);
      bop(4, NODDOF_SOH27 * i + 0) = defgrd(0, 1) * N_XYZ(2, i) + defgrd(0, 2) * N_XYZ(1, i);
      bop(4, NODDOF_SOH27 * i + 1) = defgrd(1, 1) * N_XYZ(2, i) + defgrd(1, 2) * N_XYZ(1, i);
      bop(4, NODDOF_SOH27 * i + 2) = defgrd(2, 1) * N_XYZ(2, i) + defgrd(2, 2) * N_XYZ(1, i);
      bop(5, NODDOF_SOH27 * i + 0) = defgrd(0, 2) * N_XYZ(0, i) + defgrd(0, 0) * N_XYZ(2, i);
      bop(5, NODDOF_SOH27 * i + 1) = defgrd(1, 2) * N_XYZ(0, i) + defgrd(1, 0) * N_XYZ(2, i);
      bop(5, NODDOF_SOH27 * i + 2) = defgrd(2, 2) * N_XYZ(0, i) + defgrd(2, 0) * N_XYZ(2, i);
    }

    // now build the linear strain
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> strainlin(true);
    strainlin.multiply(bop, nodaldisp);

    // and rename it as glstrain to use the common methods further on

    // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
    // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
    Core::LinAlg::SerialDenseVector glstrain_epetra(Mat::NUM_STRESS_3D);
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> glstrain(glstrain_epetra.values(), true);
    glstrain.update(1.0, strainlin);

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
        if (elestrain == nullptr) FOUR_C_THROW("strain data not available");
        // rewriting Green-Lagrange strains in matrix format
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> gl;
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
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> invdefgrd;
        invdefgrd.invert(defgrd);

        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> temp;
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> euler_almansi;
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
      case Inpar::Solid::strain_none:
        break;
      default:
        FOUR_C_THROW("requested strain type not available");
    }

    if (material()->material_type() == Core::Materials::m_constraintmixture ||
        material()->material_type() == Core::Materials::m_growthremodel_elasthyper ||
        material()->material_type() == Core::Materials::m_mixture)
    {
      // gp reference coordinates
      Core::LinAlg::Matrix<NUMNOD_SOH27, 1> funct(true);
      funct = shapefcts[gp];
      Core::LinAlg::Matrix<NUMDIM_SOH27, 1> point(true);
      point.multiply_tn(xrefe, funct);
      params.set("gp_coords_ref", point);
    }

    // call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> cmat(true);
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> stress(true);
    solid_material()->evaluate(&defgrd, &glstrain, params, &stress, &cmat, gp, id());
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
        if (eleplstrain == nullptr) FOUR_C_THROW("plastic strain data not available");
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> plglstrain =
            params.get<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("plglstrain");
        // rewriting Green-Lagrange strains in matrix format
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> gl;
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
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> invdefgrd;
        invdefgrd.invert(defgrd);

        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> temp;
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> euler_almansi;
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
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");
        const double detF = defgrd.determinant();

        Core::LinAlg::Matrix<3, 3> pkstress;
        pkstress(0, 0) = stress(0);
        pkstress(0, 1) = stress(3);
        pkstress(0, 2) = stress(5);
        pkstress(1, 0) = pkstress(0, 1);
        pkstress(1, 1) = stress(1);
        pkstress(1, 2) = stress(4);
        pkstress(2, 0) = pkstress(0, 2);
        pkstress(2, 1) = pkstress(1, 2);
        pkstress(2, 2) = stress(2);

        Core::LinAlg::Matrix<3, 3> temp;
        Core::LinAlg::Matrix<3, 3> cauchystress;
        temp.multiply(1.0 / detF, defgrd, pkstress);
        cauchystress.multiply_nt(temp, defgrd);

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
    }

    double detJ_w = detJ * gpweights[gp];
    // update internal force vector
    if (force != nullptr)
    {
      // integrate internal force vector f = f + (B^T . sigma) * detJ * w(gp)
      force->multiply_tn(detJ_w, bop, stress, 1.0);
    }
    // update stiffness matrix
    if (stiffmatrix != nullptr)
    {
      // integrate `elastic' and `initial-displacement' stiffness matrix
      // keu = keu + (B^T . C . B) * detJ * w(gp)
      Core::LinAlg::Matrix<6, NUMDOF_SOH27> cb;
      cb.multiply(cmat, bop);
      stiffmatrix->multiply_tn(detJ_w, bop, cb, 1.0);
    }

    if (massmatrix != nullptr)  // evaluate mass matrix +++++++++++++++++++++++++
    {
      double density = material()->density(gp);
      // integrate consistent mass matrix
      const double factor = detJ_w * density;
      double ifactor, massfactor;
      for (int inod = 0; inod < NUMNOD_SOH27; ++inod)
      {
        ifactor = shapefcts[gp](inod) * factor;
        for (int jnod = 0; jnod < NUMNOD_SOH27; ++jnod)
        {
          massfactor = shapefcts[gp](jnod) * ifactor;  // intermediate factor
          (*massmatrix)(NUMDIM_SOH27* inod + 0, NUMDIM_SOH27 * jnod + 0) += massfactor;
          (*massmatrix)(NUMDIM_SOH27* inod + 1, NUMDIM_SOH27 * jnod + 1) += massfactor;
          (*massmatrix)(NUMDIM_SOH27* inod + 2, NUMDIM_SOH27 * jnod + 2) += massfactor;
        }
      }

    }  // end of mass matrix +++++++++++++++++++++++++++++++++++++++++++++++++++
    /* =========================================================================*/
  } /* ==================================================== end of Loop over GP */
  /* =========================================================================*/

  return;
}  // Discret::Elements::So_hex27::soh27_linstiffmass


/*----------------------------------------------------------------------*
 |  evaluate the element (private)                                      |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex27::soh27_nlnstiffmass(std::vector<int>& lm,  // location matrix
    std::vector<double>& disp,                                             // current displacements
    std::vector<double>* vel,                                              // current velocities
    std::vector<double>* acc,                                              // current accelerations
    std::vector<double>& residual,                                         // current residual displ
    std::vector<double>& dispmat,  // current material displacements
    Core::LinAlg::Matrix<NUMDOF_SOH27, NUMDOF_SOH27>* stiffmatrix,  // element stiffness matrix
    Core::LinAlg::Matrix<NUMDOF_SOH27, NUMDOF_SOH27>* massmatrix,   // element mass matrix
    Core::LinAlg::Matrix<NUMDOF_SOH27, 1>* force,                   // element internal force vector
    Core::LinAlg::Matrix<NUMDOF_SOH27, 1>* forceinert,              // element inertial force vector
    Core::LinAlg::Matrix<NUMDOF_SOH27, 1>* force_str,  // element structural force vector
    Core::LinAlg::Matrix<NUMGPT_SOH27, Mat::NUM_STRESS_3D>* elestress,    // stresses at GP
    Core::LinAlg::Matrix<NUMGPT_SOH27, Mat::NUM_STRESS_3D>* elestrain,    // strains at GP
    Core::LinAlg::Matrix<NUMGPT_SOH27, Mat::NUM_STRESS_3D>* eleplstrain,  // plastic strains at GP
    Teuchos::ParameterList& params,             // algorithmic parameters e.g. time
    const Inpar::Solid::StressType iostress,    // stress output option
    const Inpar::Solid::StrainType iostrain,    // strain output option
    const Inpar::Solid::StrainType ioplstrain)  // plastic strain output option
{
  /* ============================================================================*
  ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for HEX_27 with 27 GAUSS POINTS*
  ** ============================================================================*/
  const static std::vector<Core::LinAlg::Matrix<NUMNOD_SOH27, 1>> shapefcts = soh27_shapefcts();
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27>> derivs =
      soh27_derivs();
  const static std::vector<double> gpweights = soh27_weights();
  /* ============================================================================*/

  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xrefe;  // material coord. of element
  Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xcurr;  // current  coord. of element
  Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xdisp;
  for (int i = 0; i < NUMNOD_SOH27; ++i)
  {
    const auto& x = nodes()[i]->x();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];

    xcurr(i, 0) = xrefe(i, 0) + disp[i * NODDOF_SOH27 + 0];
    xcurr(i, 1) = xrefe(i, 1) + disp[i * NODDOF_SOH27 + 1];
    xcurr(i, 2) = xrefe(i, 2) + disp[i * NODDOF_SOH27 + 2];

    if (Prestress::is_mulf(pstype_))
    {
      xdisp(i, 0) = disp[i * NODDOF_SOH27 + 0];
      xdisp(i, 1) = disp[i * NODDOF_SOH27 + 1];
      xdisp(i, 2) = disp[i * NODDOF_SOH27 + 2];
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27> N_XYZ;
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> defgrd(false);
  for (int gp = 0; gp < NUMGPT_SOH27; ++gp)
  {
    /* get the inverse of the Jacobian matrix which looks like:
    **            [ x_,r  y_,r  z_,r ]^-1
    **     J^-1 = [ x_,s  y_,s  z_,s ]
    **            [ x_,t  y_,t  z_,t ]
    */
    // compute derivatives N_XYZ at gp w.r.t. material coordinates
    // by N_XYZ = J^-1 * N_rst
    N_XYZ.multiply(invJ_[gp], derivs[gp]);
    double detJ = detJ_[gp];

    if (Prestress::is_mulf(pstype_))
    {
      // get Jacobian mapping wrt to the stored configuration
      Core::LinAlg::Matrix<3, 3> invJdef;
      prestress_->storageto_matrix(gp, invJdef, prestress_->j_history());
      // get derivatives wrt to last spatial configuration
      Core::LinAlg::Matrix<3, 27> N_xyz;
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
    else
    {
      defgrd.multiply_tt(xcurr, N_XYZ);
    }

    // Right Cauchy-Green tensor = F^T * F
    Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> cauchygreen;
    cauchygreen.multiply_tn(defgrd, defgrd);

    // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
    // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
    Core::LinAlg::SerialDenseVector glstrain_epetra(Mat::NUM_STRESS_3D);
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> glstrain(glstrain_epetra.values(), true);
    glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
    glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
    glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
    glstrain(3) = cauchygreen(0, 1);
    glstrain(4) = cauchygreen(1, 2);
    glstrain(5) = cauchygreen(2, 0);

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
        if (elestrain == nullptr) FOUR_C_THROW("strain data not available");
        // rewriting Green-Lagrange strains in matrix format
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> gl;
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
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> invdefgrd;
        invdefgrd.invert(defgrd);

        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> temp;
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> euler_almansi;
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
      case Inpar::Solid::strain_none:
        break;
      default:
        FOUR_C_THROW("requested strain type not available");
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
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, NUMDOF_SOH27> bop;
    for (int i = 0; i < NUMNOD_SOH27; ++i)
    {
      bop(0, NODDOF_SOH27 * i + 0) = defgrd(0, 0) * N_XYZ(0, i);
      bop(0, NODDOF_SOH27 * i + 1) = defgrd(1, 0) * N_XYZ(0, i);
      bop(0, NODDOF_SOH27 * i + 2) = defgrd(2, 0) * N_XYZ(0, i);
      bop(1, NODDOF_SOH27 * i + 0) = defgrd(0, 1) * N_XYZ(1, i);
      bop(1, NODDOF_SOH27 * i + 1) = defgrd(1, 1) * N_XYZ(1, i);
      bop(1, NODDOF_SOH27 * i + 2) = defgrd(2, 1) * N_XYZ(1, i);
      bop(2, NODDOF_SOH27 * i + 0) = defgrd(0, 2) * N_XYZ(2, i);
      bop(2, NODDOF_SOH27 * i + 1) = defgrd(1, 2) * N_XYZ(2, i);
      bop(2, NODDOF_SOH27 * i + 2) = defgrd(2, 2) * N_XYZ(2, i);
      /* ~~~ */
      bop(3, NODDOF_SOH27 * i + 0) = defgrd(0, 0) * N_XYZ(1, i) + defgrd(0, 1) * N_XYZ(0, i);
      bop(3, NODDOF_SOH27 * i + 1) = defgrd(1, 0) * N_XYZ(1, i) + defgrd(1, 1) * N_XYZ(0, i);
      bop(3, NODDOF_SOH27 * i + 2) = defgrd(2, 0) * N_XYZ(1, i) + defgrd(2, 1) * N_XYZ(0, i);
      bop(4, NODDOF_SOH27 * i + 0) = defgrd(0, 1) * N_XYZ(2, i) + defgrd(0, 2) * N_XYZ(1, i);
      bop(4, NODDOF_SOH27 * i + 1) = defgrd(1, 1) * N_XYZ(2, i) + defgrd(1, 2) * N_XYZ(1, i);
      bop(4, NODDOF_SOH27 * i + 2) = defgrd(2, 1) * N_XYZ(2, i) + defgrd(2, 2) * N_XYZ(1, i);
      bop(5, NODDOF_SOH27 * i + 0) = defgrd(0, 2) * N_XYZ(0, i) + defgrd(0, 0) * N_XYZ(2, i);
      bop(5, NODDOF_SOH27 * i + 1) = defgrd(1, 2) * N_XYZ(0, i) + defgrd(1, 0) * N_XYZ(2, i);
      bop(5, NODDOF_SOH27 * i + 2) = defgrd(2, 2) * N_XYZ(0, i) + defgrd(2, 0) * N_XYZ(2, i);
    }

    if (material()->material_type() == Core::Materials::m_constraintmixture ||
        material()->material_type() == Core::Materials::m_growthremodel_elasthyper ||
        material()->material_type() == Core::Materials::m_mixture)
    {
      // gp reference coordinates
      Core::LinAlg::Matrix<NUMNOD_SOH27, 1> funct(true);
      funct = shapefcts[gp];
      Core::LinAlg::Matrix<NUMDIM_SOH27, 1> point(true);
      point.multiply_tn(xrefe, funct);
      params.set("gp_coords_ref", point);
    }

    // call material law cccccccccccccccccccccccccccccccccccccccccccccccccccccc
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> cmat(true);
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> stress(true);
    Utils::get_temperature_for_structural_material<Core::FE::CellType::hex27>(
        shapefcts[gp], params);
    solid_material()->evaluate(&defgrd, &glstrain, params, &stress, &cmat, gp, id());
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
        if (eleplstrain == nullptr) FOUR_C_THROW("plastic strain data not available");
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> plglstrain =
            params.get<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("plglstrain");
        // rewriting Green-Lagrange strains in matrix format
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> gl;
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
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> invdefgrd;
        invdefgrd.invert(defgrd);

        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> temp;
        Core::LinAlg::Matrix<NUMDIM_SOH27, NUMDIM_SOH27> euler_almansi;
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
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");
        const double detF = defgrd.determinant();

        Core::LinAlg::Matrix<3, 3> pkstress;
        pkstress(0, 0) = stress(0);
        pkstress(0, 1) = stress(3);
        pkstress(0, 2) = stress(5);
        pkstress(1, 0) = pkstress(0, 1);
        pkstress(1, 1) = stress(1);
        pkstress(1, 2) = stress(4);
        pkstress(2, 0) = pkstress(0, 2);
        pkstress(2, 1) = pkstress(1, 2);
        pkstress(2, 2) = stress(2);

        Core::LinAlg::Matrix<3, 3> temp;
        Core::LinAlg::Matrix<3, 3> cauchystress;
        temp.multiply(1.0 / detF, defgrd, pkstress);
        cauchystress.multiply_nt(temp, defgrd);

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
    }

    double detJ_w = detJ * gpweights[gp];
    // update internal force vector
    if (force != nullptr)
    {
      // integrate internal force vector f = f + (B^T . sigma) * detJ * w(gp)
      force->multiply_tn(detJ_w, bop, stress, 1.0);
    }
    // update stiffness matrix
    if (stiffmatrix != nullptr)
    {
      // integrate `elastic' and `initial-displacement' stiffness matrix
      // keu = keu + (B^T . C . B) * detJ * w(gp)
      Core::LinAlg::Matrix<6, NUMDOF_SOH27> cb;
      cb.multiply(cmat, bop);
      stiffmatrix->multiply_tn(detJ_w, bop, cb, 1.0);

      // integrate `geometric' stiffness matrix and add to keu *****************
      Core::LinAlg::Matrix<6, 1> sfac(stress);  // auxiliary integrated stress
      sfac.scale(detJ_w);                       // detJ*w(gp)*[S11,S22,S33,S12=S21,S23=S32,S13=S31]
      std::vector<double> SmB_L(3);             // intermediate Sm.B_L
      // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
      for (int inod = 0; inod < NUMNOD_SOH27; ++inod)
      {
        SmB_L[0] = sfac(0) * N_XYZ(0, inod) + sfac(3) * N_XYZ(1, inod) + sfac(5) * N_XYZ(2, inod);
        SmB_L[1] = sfac(3) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod) + sfac(4) * N_XYZ(2, inod);
        SmB_L[2] = sfac(5) * N_XYZ(0, inod) + sfac(4) * N_XYZ(1, inod) + sfac(2) * N_XYZ(2, inod);
        for (int jnod = 0; jnod < NUMNOD_SOH27; ++jnod)
        {
          double bopstrbop = 0.0;  // intermediate value
          for (int idim = 0; idim < NUMDIM_SOH27; ++idim)
            bopstrbop += N_XYZ(idim, jnod) * SmB_L[idim];
          (*stiffmatrix)(3 * inod + 0, 3 * jnod + 0) += bopstrbop;
          (*stiffmatrix)(3 * inod + 1, 3 * jnod + 1) += bopstrbop;
          (*stiffmatrix)(3 * inod + 2, 3 * jnod + 2) += bopstrbop;
        }
      }  // end of integrate `geometric' stiffness******************************
    }

    if (massmatrix != nullptr)  // evaluate mass matrix +++++++++++++++++++++++++
    {
      double density = material()->density(gp);
      // integrate consistent mass matrix
      const double factor = detJ_w * density;
      double ifactor, massfactor;
      for (int inod = 0; inod < NUMNOD_SOH27; ++inod)
      {
        ifactor = shapefcts[gp](inod) * factor;
        for (int jnod = 0; jnod < NUMNOD_SOH27; ++jnod)
        {
          massfactor = shapefcts[gp](jnod) * ifactor;  // intermediate factor
          (*massmatrix)(NUMDIM_SOH27* inod + 0, NUMDIM_SOH27 * jnod + 0) += massfactor;
          (*massmatrix)(NUMDIM_SOH27* inod + 1, NUMDIM_SOH27 * jnod + 1) += massfactor;
          (*massmatrix)(NUMDIM_SOH27* inod + 2, NUMDIM_SOH27 * jnod + 2) += massfactor;
        }
      }

      // check for non constant mass matrix
      if (solid_material()->varying_density())
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
        solid_material()->evaluate_non_lin_mass(
            &defgrd, &glstrain, params, &linmass_disp, &linmass_vel, gp, id());

        // multiply by 2.0 to get derivative w.r.t green lagrange strains and multiply by time
        // integration factor
        linmass_disp.scale(2.0 * timintfac_dis);
        linmass_vel.scale(2.0 * timintfac_vel);
        linmass.update(1.0, linmass_disp, 1.0, linmass_vel, 0.0);

        // evaluate accelerations at time n+1 at gauss point
        Core::LinAlg::Matrix<NUMDIM_SOH27, 1> myacc(true);
        for (int idim = 0; idim < NUMDIM_SOH27; ++idim)
          for (int inod = 0; inod < NUMNOD_SOH27; ++inod)
            myacc(idim) += shapefcts[gp](inod) * (*acc)[idim + (inod * NUMDIM_SOH27)];

        if (stiffmatrix != nullptr)
        {
          // integrate linearisation of mass matrix
          //(B^T . d\rho/d disp . a) * detJ * w(gp)
          Core::LinAlg::Matrix<1, NUMDOF_SOH27> cb;
          cb.multiply_tn(linmass_disp, bop);
          for (int inod = 0; inod < NUMNOD_SOH27; ++inod)
          {
            double factor = detJ_w * shapefcts[gp](inod);
            for (int idim = 0; idim < NUMDIM_SOH27; ++idim)
            {
              double massfactor = factor * myacc(idim);
              for (int jnod = 0; jnod < NUMNOD_SOH27; ++jnod)
                for (int jdim = 0; jdim < NUMDIM_SOH27; ++jdim)
                  (*massmatrix)(inod* NUMDIM_SOH27 + idim, jnod * NUMDIM_SOH27 + jdim) +=
                      massfactor * cb(jnod * NUMDIM_SOH27 + jdim);
            }
          }
        }

        // internal force vector without EAS terms
        if (forceinert != nullptr)
        {
          // integrate nonlinear inertia force term
          for (int inod = 0; inod < NUMNOD_SOH27; ++inod)
          {
            double forcefactor = shapefcts[gp](inod) * detJ_w;
            for (int idim = 0; idim < NUMDIM_SOH27; ++idim)
              (*forceinert)(inod* NUMDIM_SOH27 + idim) += forcefactor * density * myacc(idim);
          }
        }
      }
    }  // end of mass matrix +++++++++++++++++++++++++++++++++++++++++++++++++++

  } /* ==================================================== end of Loop over GP */

  return;
}  // Discret::Elements::So_hex27::soh27_nlnstiffmass

/*----------------------------------------------------------------------*
 |  lump mass matrix (private)                                          |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex27::soh27_lumpmass(
    Core::LinAlg::Matrix<NUMDOF_SOH27, NUMDOF_SOH27>* emass)
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
 |  Evaluate Hex27 Shape fcts at all 27 Gauss Points                     |
 *----------------------------------------------------------------------*/
std::vector<Core::LinAlg::Matrix<NUMNOD_SOH27, 1>> Discret::Elements::SoHex27::soh27_shapefcts()
{
  std::vector<Core::LinAlg::Matrix<NUMNOD_SOH27, 1>> shapefcts(NUMGPT_SOH27);
  // (r,s,t) gp-locations of fully integrated quadratic Hex 27
  // fill up nodal f at each gp
  const Core::FE::GaussRule3D gaussrule = Core::FE::GaussRule3D::hex_27point;
  const Core::FE::IntegrationPoints3D intpoints(gaussrule);
  for (int igp = 0; igp < intpoints.nquad; ++igp)
  {
    const double r = intpoints.qxg[igp][0];
    const double s = intpoints.qxg[igp][1];
    const double t = intpoints.qxg[igp][2];

    Core::FE::shape_function_3d(shapefcts[igp], r, s, t, Core::FE::CellType::hex27);
  }
  return shapefcts;
}


/*----------------------------------------------------------------------*
 |  Evaluate Hex27 Shape fct derivs at all 27 Gauss Points              |
 *----------------------------------------------------------------------*/
std::vector<Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27>>
Discret::Elements::SoHex27::soh27_derivs()
{
  std::vector<Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27>> derivs(NUMGPT_SOH27);
  // (r,s,t) gp-locations of fully integrated quadratic Hex 27
  // fill up df w.r.t. rst directions (NUMDIM) at each gp
  const Core::FE::GaussRule3D gaussrule = Core::FE::GaussRule3D::hex_27point;
  const Core::FE::IntegrationPoints3D intpoints(gaussrule);
  for (int igp = 0; igp < intpoints.nquad; ++igp)
  {
    const double r = intpoints.qxg[igp][0];
    const double s = intpoints.qxg[igp][1];
    const double t = intpoints.qxg[igp][2];

    Core::FE::shape_function_3d_deriv1(derivs[igp], r, s, t, Core::FE::CellType::hex27);
  }
  return derivs;
}

/*----------------------------------------------------------------------*
 |  Evaluate Hex27 Weights at all 27 Gauss Points                       |
 *----------------------------------------------------------------------*/
std::vector<double> Discret::Elements::SoHex27::soh27_weights()
{
  std::vector<double> weights(NUMGPT_SOH27);
  const Core::FE::GaussRule3D gaussrule = Core::FE::GaussRule3D::hex_27point;
  const Core::FE::IntegrationPoints3D intpoints(gaussrule);
  for (int i = 0; i < NUMGPT_SOH27; ++i)
  {
    weights[i] = intpoints.qwgt[i];
  }
  return weights;
}

/*----------------------------------------------------------------------*
 |  shape functions and derivatives for So_hex27                         |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex27::soh27_shapederiv(
    Core::LinAlg::Matrix<NUMNOD_SOH27, NUMGPT_SOH27>** shapefct,  // pointer to pointer of shapefct
    Core::LinAlg::Matrix<NUMDOF_SOH27, NUMNOD_SOH27>** deriv,     // pointer to pointer of derivs
    Core::LinAlg::Matrix<NUMGPT_SOH27, 1>** weights)              // pointer to pointer of weights
{
  // static matrix objects, kept in memory
  static Core::LinAlg::Matrix<NUMNOD_SOH27, NUMGPT_SOH27> f;   // shape functions
  static Core::LinAlg::Matrix<NUMDOF_SOH27, NUMNOD_SOH27> df;  // derivatives
  static Core::LinAlg::Matrix<NUMGPT_SOH27, 1> weightfactors;  // weights for each gp
  static bool fdf_eval;                                        // flag for re-evaluate everything

  if (fdf_eval == true)  // if true f,df already evaluated
  {
    *shapefct = &f;             // return address of static object to target of pointer
    *deriv = &df;               // return address of static object to target of pointer
    *weights = &weightfactors;  // return address of static object to target of pointer
    return;
  }
  else
  {
    // (r,s,t) gp-locations of fully integrated quadratic Hex 27
    // fill up nodal f at each gp
    // fill up df w.r.t. rst directions (NUMDIM) at each gp
    const Core::FE::GaussRule3D gaussrule_ = Core::FE::GaussRule3D::hex_27point;
    const Core::FE::IntegrationPoints3D intpoints(gaussrule_);
    for (int igp = 0; igp < intpoints.nquad; ++igp)
    {
      const double r = intpoints.qxg[igp][0];
      const double s = intpoints.qxg[igp][1];
      const double t = intpoints.qxg[igp][2];

      Core::LinAlg::Matrix<NUMNOD_SOH27, 1> funct;
      Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27> deriv;
      Core::FE::shape_function_3d(funct, r, s, t, Core::FE::CellType::hex27);
      Core::FE::shape_function_3d_deriv1(deriv, r, s, t, Core::FE::CellType::hex27);
      for (int inode = 0; inode < NUMNOD_SOH27; ++inode)
      {
        f(inode, igp) = funct(inode);
        df(igp * NUMDIM_SOH27 + 0, inode) = deriv(0, inode);
        df(igp * NUMDIM_SOH27 + 1, inode) = deriv(1, inode);
        df(igp * NUMDIM_SOH27 + 2, inode) = deriv(2, inode);
        weightfactors(igp) = intpoints.qwgt[igp];
      }
    }
    // return addresses of just evaluated matrices
    *shapefct = &f;             // return address of static object to target of pointer
    *deriv = &df;               // return address of static object to target of pointer
    *weights = &weightfactors;  // return address of static object to target of pointer
    fdf_eval = true;            // now all arrays are filled statically
  }
  return;
}  // of soh27_shapederiv


/*----------------------------------------------------------------------*
 |  init the element (public)                                           |
 *----------------------------------------------------------------------*/
int Discret::Elements::SoHex27Type::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<Discret::Elements::SoHex27*>(dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex27* failed");
    actele->init_jacobian_mapping();
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  compute def gradient at every gaussian point (protected)            |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex27::def_gradient(const std::vector<double>& disp,
    Core::LinAlg::SerialDenseMatrix& gpdefgrd, Discret::Elements::PreStress& prestress)
{
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27>> derivs =
      soh27_derivs();

  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xdisp;  // current  coord. of element
  for (int i = 0; i < NUMNOD_SOH27; ++i)
  {
    xdisp(i, 0) = disp[i * NODDOF_SOH27 + 0];
    xdisp(i, 1) = disp[i * NODDOF_SOH27 + 1];
    xdisp(i, 2) = disp[i * NODDOF_SOH27 + 2];
  }

  for (int gp = 0; gp < NUMGPT_SOH27; ++gp)
  {
    // get Jacobian mapping wrt to the stored deformed configuration
    Core::LinAlg::Matrix<3, 3> invJdef;
    prestress.storageto_matrix(gp, invJdef, prestress.j_history());

    // by N_XYZ = J^-1 * N_rst
    Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27> N_xyz;
    N_xyz.multiply(invJdef, derivs[gp]);

    // build defgrd (independent of xrefe!)
    Core::LinAlg::Matrix<3, 3> defgrd;
    defgrd.multiply_tt(xdisp, N_xyz);
    defgrd(0, 0) += 1.0;
    defgrd(1, 1) += 1.0;
    defgrd(2, 2) += 1.0;

    prestress.matrixto_storage(gp, defgrd, gpdefgrd);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  compute Jac.mapping wrt deformed configuration (protected)          |
 *----------------------------------------------------------------------*/
void Discret::Elements::SoHex27::update_jacobian_mapping(
    const std::vector<double>& disp, Discret::Elements::PreStress& prestress)
{
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27>> derivs =
      soh27_derivs();

  // get incremental disp
  Core::LinAlg::Matrix<NUMNOD_SOH27, NUMDIM_SOH27> xdisp;
  for (int i = 0; i < NUMNOD_SOH27; ++i)
  {
    xdisp(i, 0) = disp[i * NODDOF_SOH27 + 0];
    xdisp(i, 1) = disp[i * NODDOF_SOH27 + 1];
    xdisp(i, 2) = disp[i * NODDOF_SOH27 + 2];
  }

  Core::LinAlg::Matrix<3, 3> invJhist;
  Core::LinAlg::Matrix<3, 3> invJ;
  Core::LinAlg::Matrix<3, 3> defgrd;
  Core::LinAlg::Matrix<NUMDIM_SOH27, NUMNOD_SOH27> N_xyz;
  Core::LinAlg::Matrix<3, 3> invJnew;
  for (int gp = 0; gp < NUMGPT_SOH27; ++gp)
  {
    // get the invJ old state
    prestress.storageto_matrix(gp, invJhist, prestress.j_history());
    // get derivatives wrt to invJhist
    N_xyz.multiply(invJhist, derivs[gp]);
    // build defgrd \partial x_new / \parial x_old , where x_old != X
    defgrd.multiply_tt(xdisp, N_xyz);
    defgrd(0, 0) += 1.0;
    defgrd(1, 1) += 1.0;
    defgrd(2, 2) += 1.0;
    // make inverse of this defgrd
    defgrd.invert();
    // push-forward of Jinv
    invJnew.multiply_tn(defgrd, invJhist);
    // store new reference configuration
    prestress.matrixto_storage(gp, invJnew, prestress.j_history());
  }  // for (int gp=0; gp<NUMGPT_SOH27; ++gp)

  return;
}

FOUR_C_NAMESPACE_CLOSE
