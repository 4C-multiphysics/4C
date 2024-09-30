/*----------------------------------------------------------------------*/
/*! \file

\brief Evaluate routines for Solid Hex8 element with F-bar modification

\level 1

*/
/*----------------------------------------------------------------------*/

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_eigen.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_mat_growthremodel_elasthyper.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mat_thermoplastichyperelast.hpp"
#include "4C_so3_hex8fbar.hpp"
#include "4C_so3_prestress.hpp"
#include "4C_so3_prestress_service.hpp"
#include "4C_so3_utils.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  evaluate the element (public)                                       |
 *----------------------------------------------------------------------*/
int Discret::ELEMENTS::SoHex8fbar::evaluate(Teuchos::ParameterList& params,
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
  // elevec3 is not used anyway

  // start with "none"
  Core::Elements::ActionType act = Core::Elements::none;

  if (is_params_interface())
    act = params_interface().get_action_type();
  else
  {
    // get the required action
    std::string action = params.get<std::string>("action", "none");
    if (action == "none")
      FOUR_C_THROW("No action supplied");
    else if (action == "calc_struct_linstiff")
      act = Core::Elements::struct_calc_linstiff;
    else if (action == "calc_struct_nlnstiff")
      act = Core::Elements::struct_calc_nlnstiff;
    else if (action == "calc_struct_internalforce")
      act = Core::Elements::struct_calc_internalforce;
    else if (action == "calc_struct_linstiffmass")
      act = Core::Elements::struct_calc_linstiffmass;
    else if (action == "calc_struct_nlnstiffmass")
      act = Core::Elements::struct_calc_nlnstiffmass;
    else if (action == "calc_struct_nlnstifflmass")
      act = Core::Elements::struct_calc_nlnstifflmass;
    else if (action == "calc_struct_stress")
      act = Core::Elements::struct_calc_stress;
    else if (action == "calc_struct_eleload")
      act = Core::Elements::struct_calc_eleload;
    else if (action == "calc_struct_fsiload")
      act = Core::Elements::struct_calc_fsiload;
    else if (action == "calc_struct_update_istep")
      act = Core::Elements::struct_calc_update_istep;
    else if (action == "calc_struct_reset_istep")
      act = Core::Elements::struct_calc_reset_istep;
    else if (action == "multi_readrestart")
      act = Core::Elements::multi_readrestart;
    else if (action == "multi_calc_dens")
      act = Core::Elements::multi_calc_dens;
    else if (action == "calc_struct_prestress_update")
      act = Core::Elements::struct_update_prestress;
    else if (action == "calc_struct_energy")
      act = Core::Elements::struct_calc_energy;
    else if (action == "calc_struct_predict")
      return 0;
    else if (action == "calc_struct_recover")
      return 0;
    else
      FOUR_C_THROW("Unknown type of action for So_hex8fbar");
  }

  // what should the element do
  switch (act)
  {
    // linear stiffness
    case Core::Elements::struct_calc_linstiff:
    {
      // need current displacement and residual forces
      std::vector<double> mydisp(lm.size());
      for (double& i : mydisp) i = 0.0;
      std::vector<double> myres(lm.size());
      for (double& myre : myres) myre = 0.0;
      nlnstiffmass(lm, mydisp, nullptr, myres, &elemat1, nullptr, &elevec1, nullptr, nullptr,
          nullptr, nullptr, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none,
          Inpar::Solid::strain_none);
    }
    break;

    // nonlinear stiffness and internal force vector
    case Core::Elements::struct_calc_nlnstiff:
    {
      // need current displacement and residual forces
      Teuchos::RCP<const Core::LinAlg::Vector> disp = discretization.get_state("displacement");
      Teuchos::RCP<const Core::LinAlg::Vector> res =
          discretization.get_state("residual displacement");
      if (disp == Teuchos::null || res == Teuchos::null)
        FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);
      Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8>* matptr = nullptr;
      if (elemat1.is_initialized()) matptr = &elemat1;

      nlnstiffmass(lm, mydisp, nullptr, myres, matptr, nullptr, &elevec1, nullptr, nullptr, nullptr,
          nullptr, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none,
          Inpar::Solid::strain_none);
    }
    break;

    // internal force vector only
    case Core::Elements::struct_calc_internalforce:
    {
      // need current displacement and residual forces
      Teuchos::RCP<const Core::LinAlg::Vector> disp = discretization.get_state("displacement");
      Teuchos::RCP<const Core::LinAlg::Vector> res =
          discretization.get_state("residual displacement");
      if (disp == Teuchos::null || res == Teuchos::null)
        FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);
      // create a dummy element matrix to apply linearised EAS-stuff onto
      Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8> myemat(true);
      nlnstiffmass(lm, mydisp, nullptr, myres, &myemat, nullptr, &elevec1, nullptr, nullptr,
          nullptr, nullptr, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none,
          Inpar::Solid::strain_none);
    }
    break;

    // linear stiffness and consistent mass matrix
    case Core::Elements::struct_calc_linstiffmass:
      FOUR_C_THROW("Case 'calc_struct_linstiffmass' not yet implemented");
      break;

    // nonlinear stiffness, internal force vector, and consistent mass matrix
    case Core::Elements::struct_calc_nlnstiffmass:
    case Core::Elements::struct_calc_nlnstifflmass:
    {
      // need current displacement and residual forces
      Teuchos::RCP<const Core::LinAlg::Vector> disp = discretization.get_state("displacement");
      Teuchos::RCP<const Core::LinAlg::Vector> res =
          discretization.get_state("residual displacement");
      Teuchos::RCP<const Core::LinAlg::Vector> acc = discretization.get_state("acceleration");
      if (disp == Teuchos::null || res == Teuchos::null)
        FOUR_C_THROW("Cannot get state vectors 'displacement' and/or residual");
      if (acc == Teuchos::null) FOUR_C_THROW("Cannot get state vectors 'acceleration'");

      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);
      std::vector<double> myacc(lm.size());
      Core::FE::extract_my_values(*acc, myacc, lm);

      nlnstiffmass(lm, mydisp, &myacc, myres, &elemat1, &elemat2, &elevec1, &elevec2, nullptr,
          nullptr, nullptr, params, Inpar::Solid::stress_none, Inpar::Solid::strain_none,
          Inpar::Solid::strain_none);

      if (act == Core::Elements::struct_calc_nlnstifflmass) soh8_lumpmass(&elemat2);
    }
    break;
    // recover elementwise stored quantities
    case Core::Elements::struct_calc_recover:
    {
      /* ToDo Probably we have to recover the history information of some special
       * materials.                                           hiermeier 04/2016*/
    }
    break;
    // evaluate stresses and strains at gauss points
    case Core::Elements::struct_calc_stress:
    {
      Teuchos::RCP<const Core::LinAlg::Vector> disp = discretization.get_state("displacement");
      Teuchos::RCP<const Core::LinAlg::Vector> res =
          discretization.get_state("residual displacement");
      Teuchos::RCP<std::vector<char>> stressdata = Teuchos::null;
      Teuchos::RCP<std::vector<char>> straindata = Teuchos::null;
      Teuchos::RCP<std::vector<char>> plstraindata = Teuchos::null;
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
        stressdata = params.get<Teuchos::RCP<std::vector<char>>>("stress", Teuchos::null);
        straindata = params.get<Teuchos::RCP<std::vector<char>>>("strain", Teuchos::null);
        iostress = params.get<Inpar::Solid::StressType>("iostress", Inpar::Solid::stress_none);
        iostrain = params.get<Inpar::Solid::StrainType>("iostrain", Inpar::Solid::strain_none);
        // in case of small strain materials calculate plastic strains for post processing
        plstraindata = params.get<Teuchos::RCP<std::vector<char>>>("plstrain", Teuchos::null);
        ioplstrain = params.get<Inpar::Solid::StrainType>("ioplstrain", Inpar::Solid::strain_none);
      }
      if (disp == Teuchos::null) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      if (stressdata == Teuchos::null) FOUR_C_THROW("Cannot get 'stress' data");
      if (straindata == Teuchos::null) FOUR_C_THROW("Cannot get 'strain' data");
      if (plstraindata == Teuchos::null) FOUR_C_THROW("Cannot get 'plastic strain' data");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      std::vector<double> myres(lm.size());
      Core::FE::extract_my_values(*res, myres, lm);
      Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D> stress;
      Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D> strain;
      Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D> plstrain;

      nlnstiffmass(lm, mydisp, nullptr, myres, nullptr, nullptr, nullptr, nullptr, &stress, &strain,
          &plstrain, params, iostress, iostrain, ioplstrain);
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

    case Core::Elements::struct_calc_eleload:
      FOUR_C_THROW("this method is not supposed to evaluate a load, use evaluate_neumann(...)");
      break;

    case Core::Elements::struct_calc_fsiload:
      FOUR_C_THROW("Case not yet implemented");
      break;

    case Core::Elements::struct_calc_update_istep:
    {
      // Update of history for materials
      Teuchos::RCP<const Core::LinAlg::Vector> disp = discretization.get_state("displacement");
      if (disp == Teuchos::null) FOUR_C_THROW("Cannot get state vectors 'displacement'");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);
      update_element(mydisp, params, material());
    }
    break;

    case Core::Elements::struct_calc_reset_istep:
    {
      // Reset of history (if needed)
      solid_material()->reset_step();
    }
    break;

    //==================================================================================
    case Core::Elements::multi_calc_dens:
    {
      soh8_homog(params);
    }
    break;

    //==================================================================================
    case Core::Elements::struct_update_prestress:
    {
      time_ = params.get<double>("total time");
      Teuchos::RCP<const Core::LinAlg::Vector> disp = discretization.get_state("displacement");
      if (disp == Teuchos::null) FOUR_C_THROW("Cannot get displacement state");
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      // build def gradient for every gauss point
      Core::LinAlg::SerialDenseMatrix gpdefgrd(NUMGPT_SOH8 + 1, 9);
      def_gradient(mydisp, gpdefgrd, *prestress_);

      // update deformation gradient and put back to storage
      Core::LinAlg::Matrix<3, 3> deltaF;
      Core::LinAlg::Matrix<3, 3> Fhist;
      Core::LinAlg::Matrix<3, 3> Fnew;
      for (unsigned gp = 0; gp < NUMGPT_SOH8 + 1; ++gp)
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

    //==================================================================================
    // read restart of microscale
    case Core::Elements::multi_readrestart:
    {
      Teuchos::RCP<Core::Mat::Material> mat = material();

      if (mat->material_type() == Core::Materials::m_struct_multiscale) soh8_read_restart_multi();
    }
    break;

    case Core::Elements::struct_calc_energy:
    {
      // initialization of internal energy
      double intenergy = 0.0;

      // shape functions and Gauss weights
      const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs =
          soh8_derivs();
      const static std::vector<double> weights = soh8_weights();

      // get displacements of this processor
      Teuchos::RCP<const Core::LinAlg::Vector> disp = discretization.get_state("displacement");
      if (disp == Teuchos::null) FOUR_C_THROW("Cannot get state displacement vector");

      // get displacements of this element
      std::vector<double> mydisp(lm.size());
      Core::FE::extract_my_values(*disp, mydisp, lm);

      // update element geometry
      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe;  // material coord. of element
      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xcurr;  // current  coord. of element
      Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xdisp;


      for (int i = 0; i < NUMNOD_SOH8; ++i)
      {
        xrefe(i, 0) = nodes()[i]->x()[0];
        xrefe(i, 1) = nodes()[i]->x()[1];
        xrefe(i, 2) = nodes()[i]->x()[2];

        xcurr(i, 0) = xrefe(i, 0) + mydisp[i * NODDOF_SOH8 + 0];
        xcurr(i, 1) = xrefe(i, 1) + mydisp[i * NODDOF_SOH8 + 1];
        xcurr(i, 2) = xrefe(i, 2) + mydisp[i * NODDOF_SOH8 + 2];

        if (Prestress::is_mulf(pstype_))
        {
          xdisp(i, 0) = mydisp[i * NODDOF_SOH8 + 0];
          xdisp(i, 1) = mydisp[i * NODDOF_SOH8 + 1];
          xdisp(i, 2) = mydisp[i * NODDOF_SOH8 + 2];
        }
      }

      //****************************************************************************
      // deformation gradient at centroid of element
      //****************************************************************************
      double detF_0 = -1.0;
      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invdefgrd_0;
      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_XYZ_0;
      // element coordinate derivatives at centroid
      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_rst_0;
      Core::FE::shape_function_3d_deriv1(N_rst_0, 0.0, 0.0, 0.0, Core::FE::CellType::hex8);
      {
        // inverse jacobian matrix at centroid
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invJ_0;
        invJ_0.multiply(N_rst_0, xrefe);
        invJ_0.invert();
        // material derivatives at centroid
        N_XYZ_0.multiply(invJ_0, N_rst_0);
      }

      if (Prestress::is_mulf(pstype_))
      {
        // get Jacobian mapping wrt to the stored configuration
        // centroid is 9th Gaussian point in storage
        Core::LinAlg::Matrix<3, 3> invJdef_0;
        prestress_->storageto_matrix(NUMGPT_SOH8, invJdef_0, prestress_->j_history());
        // get derivatives wrt to last spatial configuration
        Core::LinAlg::Matrix<3, 8> N_xyz_0;
        N_xyz_0.multiply(invJdef_0, N_rst_0);  // if (!Id()) std::cout << invJdef_0;

        // build multiplicative incremental defgrd
        Core::LinAlg::Matrix<3, 3> defgrd_0(false);
        defgrd_0.multiply_tt(xdisp, N_xyz_0);
        defgrd_0(0, 0) += 1.0;
        defgrd_0(1, 1) += 1.0;
        defgrd_0(2, 2) += 1.0;

        // get stored old incremental F
        Core::LinAlg::Matrix<3, 3> Fhist;
        prestress_->storageto_matrix(NUMGPT_SOH8, Fhist, prestress_->f_history());

        // build total defgrd = delta F * F_old
        Core::LinAlg::Matrix<3, 3> tmp;
        tmp.multiply(defgrd_0, Fhist);
        defgrd_0 = tmp;

        // build inverse and detF
        invdefgrd_0.invert(defgrd_0);
        detF_0 = defgrd_0.determinant();
      }
      else  // no prestressing
      {
        // deformation gradient and its determinant at centroid
        Core::LinAlg::Matrix<3, 3> defgrd_0(false);
        defgrd_0.multiply_tt(xcurr, N_XYZ_0);
        invdefgrd_0.invert(defgrd_0);
        detF_0 = defgrd_0.determinant();
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

          // F_bar deformation gradient =(detF_0/detF)^1/3*F
          double detF = defgrd.determinant();
          Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd_bar(defgrd);
          double f_bar_factor = pow(detF_0 / detF, 1.0 / 3.0);
          defgrd_bar.scale(f_bar_factor);



          // right Cauchy-Green tensor = F^T * F
          Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> cauchygreen;
          cauchygreen.multiply_tn(defgrd_bar, defgrd_bar);

          glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
          glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
          glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
          glstrain(3) = cauchygreen(0, 1);
          glstrain(4) = cauchygreen(1, 2);
          glstrain(5) = cauchygreen(2, 0);
        }
        else  // no prestressing
        {
          // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
          defgrd.multiply_tt(xcurr, N_XYZ);

          // F_bar deformation gradient =(detF_0/detF)^1/3*F
          double detF = defgrd.determinant();
          Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd_bar(defgrd);
          double f_bar_factor = pow(detF_0 / detF, 1.0 / 3.0);
          defgrd_bar.scale(f_bar_factor);

          // right Cauchy-Green tensor = F^T * F
          Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> cauchygreen;
          cauchygreen.multiply_tn(defgrd_bar, defgrd_bar);

          glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
          glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
          glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
          glstrain(3) = cauchygreen(0, 1);
          glstrain(4) = cauchygreen(1, 2);
          glstrain(5) = cauchygreen(2, 0);
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

    case Core::Elements::struct_calc_predict:
    {
      // do nothing here
      break;
    }

    default:
      FOUR_C_THROW("Unknown type of action for So_hex8fbar");
      break;
  }
  return 0;
}

/*----------------------------------------------------------------------*
 |  init the element jacobian mapping (protected)              gee 03/11|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8fbar::init_jacobian_mapping()
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
      if (!(prestress_->is_init()))
        prestress_->matrixto_storage(gp, invJ_[gp], prestress_->j_history());
  }

  // init the centroid invJ
  if (Prestress::is_mulf_active(time_, pstype_, pstime_))
    if (!(prestress_->is_init()))
    {
      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_rst_0;
      Core::FE::shape_function_3d_deriv1(N_rst_0, 0.0, 0.0, 0.0, Core::FE::CellType::hex8);
      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invJ_0;
      invJ_0.multiply(N_rst_0, xrefe);
      invJ_0.invert();
      prestress_->matrixto_storage(NUMGPT_SOH8, invJ_0, prestress_->j_history());
    }


  if (Prestress::is_mulf_active(time_, pstype_, pstime_)) prestress_->is_init() = true;

  return;
}

/*----------------------------------------------------------------------*
 |  Integrate a Volume Neumann boundary condition (public)               |
 *----------------------------------------------------------------------*/
int Discret::ELEMENTS::SoHex8fbar::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  set_params_interface_ptr(params);
  // get values and switches from the condition
  const auto* onoff = &condition.parameters().get<std::vector<int>>("ONOFF");
  const auto* val = &condition.parameters().get<std::vector<double>>("VAL");

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
  if (int(onoff->size()) < NUMDIM_SOH8)
    FOUR_C_THROW("Fewer functions or curves defined than the element has dofs.");

  for (int checkdof = NUMDIM_SOH8; checkdof < int(onoff->size()); ++checkdof)
  {
    if ((*onoff)[checkdof] != 0)
      FOUR_C_THROW(
          "Number of Dimensions in Neumann_Evalutaion is 3. Further DoFs are not considered.");
  }

  // (SPATIAL) FUNCTION BUSINESS
  const auto* funct = &condition.parameters().get<std::vector<int>>("FUNCT");
  Core::LinAlg::Matrix<NUMDIM_SOH8, 1> xrefegp(false);
  bool havefunct = false;
  if (funct)
    for (int dim = 0; dim < NUMDIM_SOH8; dim++)
      if ((*funct)[dim] > 0) havefunct = havefunct or true;

  /* ============================================================================*
  ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for HEX_8 with 8 GAUSS POINTS*
  ** ============================================================================*/
  const static std::vector<Core::LinAlg::Matrix<NUMNOD_SOH8, 1>> shapefcts = soh8_shapefcts();
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs = soh8_derivs();
  const static std::vector<double> gpweights = soh8_weights();
  /* ============================================================================*/

  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe;  // material coord. of element
  for (int i = 0; i < NUMNOD_SOH8; ++i)
  {
    const auto& x = nodes()[i]->x();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];
  }
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
    if (havefunct)
    {
      for (int dim = 0; dim < NUMDIM_SOH8; dim++)
      {
        xrefegp(dim) = 0.0;
        for (int nodid = 0; nodid < NUMNOD_SOH8; ++nodid)
          xrefegp(dim) += shapefcts[gp](nodid) * xrefe(nodid, dim);
      }
    }

    // integration factor
    const double fac = gpweights[gp] * detJ;
    // distribute/add over element load vector
    for (int dim = 0; dim < NUMDIM_SOH8; dim++)
    {
      // function evaluation
      const int functnum = (funct) ? (*funct)[dim] : -1;
      const double functfac =
          (functnum > 0) ? Global::Problem::instance()
                               ->function_by_id<Core::UTILS::FunctionOfSpaceTime>(functnum - 1)
                               .evaluate(xrefegp.data(), time, dim)
                         : 1.0;
      const double dim_fac = (*onoff)[dim] * (*val)[dim] * fac * functfac;
      for (int nodid = 0; nodid < NUMNOD_SOH8; ++nodid)
      {
        elevec1[nodid * NUMDIM_SOH8 + dim] += shapefcts[gp](nodid) * dim_fac;
      }
    }

  } /* ==================================================== end of Loop over GP */

  return 0;
}  // Discret::ELEMENTS::So_hex8fbar::evaluate_neumann

/*----------------------------------------------------------------------*
 |  evaluate the element (private)                                      |
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8fbar::nlnstiffmass(std::vector<int>& lm,  // location matrix
    std::vector<double>& disp,                                          // current displacements
    std::vector<double>* acc,                                           // current accelerations
    std::vector<double>& residual,                                      // current residual displ
    Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8>* stiffmatrix,        // element stiffness matrix
    Core::LinAlg::Matrix<NUMDOF_SOH8, NUMDOF_SOH8>* massmatrix,         // element mass matrix
    Core::LinAlg::Matrix<NUMDOF_SOH8, 1>* force,       // element internal force vector
    Core::LinAlg::Matrix<NUMDOF_SOH8, 1>* forceinert,  // element inertial force vector
    Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D>* elestress,    // stresses at GP
    Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D>* elestrain,    // strains at GP
    Core::LinAlg::Matrix<NUMGPT_SOH8, Mat::NUM_STRESS_3D>* eleplstrain,  // plastic strains at GP
    Teuchos::ParameterList& params,             // algorithmic parameters e.g. time
    const Inpar::Solid::StressType iostress,    // stress output option
    const Inpar::Solid::StrainType iostrain,    // strain output option
    const Inpar::Solid::StrainType ioplstrain)  // strain output option
{
  /* ============================================================================*
  ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for HEX_8 with 8 GAUSS POINTS*
  ** ============================================================================*/
  const static std::vector<Core::LinAlg::Matrix<NUMNOD_SOH8, 1>> shapefcts = soh8_shapefcts();
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs = soh8_derivs();
  const static std::vector<double> gpweights = soh8_weights();
  /* ============================================================================*/

  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe;  // material coord. of element
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xcurr;  // current  coord. of element
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xdisp;
  for (int i = 0; i < NUMNOD_SOH8; ++i)
  {
    const auto& x = nodes()[i]->x();
    xrefe(i, 0) = x[0];
    xrefe(i, 1) = x[1];
    xrefe(i, 2) = x[2];

    xcurr(i, 0) = xrefe(i, 0) + disp[i * NODDOF_SOH8 + 0];
    xcurr(i, 1) = xrefe(i, 1) + disp[i * NODDOF_SOH8 + 1];
    xcurr(i, 2) = xrefe(i, 2) + disp[i * NODDOF_SOH8 + 2];

    if (Prestress::is_mulf(pstype_))
    {
      xdisp(i, 0) = disp[i * NODDOF_SOH8 + 0];
      xdisp(i, 1) = disp[i * NODDOF_SOH8 + 1];
      xdisp(i, 2) = disp[i * NODDOF_SOH8 + 2];
    }
  }

  //****************************************************************************
  // deformation gradient at centroid of element
  //****************************************************************************
  double detF_0 = -1.0;
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invdefgrd_0;
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_XYZ_0;
  // element coordinate derivatives at centroid
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_rst_0;
  Core::FE::shape_function_3d_deriv1(N_rst_0, 0.0, 0.0, 0.0, Core::FE::CellType::hex8);
  {
    // inverse jacobian matrix at centroid
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invJ_0;
    invJ_0.multiply(N_rst_0, xrefe);
    invJ_0.invert();
    // material derivatives at centroid
    N_XYZ_0.multiply(invJ_0, N_rst_0);
  }

  if (Prestress::is_mulf(pstype_))
  {
    // get Jacobian mapping wrt to the stored configuration
    // centroid is 9th Gaussian point in storage
    Core::LinAlg::Matrix<3, 3> invJdef_0;
    prestress_->storageto_matrix(NUMGPT_SOH8, invJdef_0, prestress_->j_history());
    // get derivatives wrt to last spatial configuration
    Core::LinAlg::Matrix<3, 8> N_xyz_0;
    N_xyz_0.multiply(invJdef_0, N_rst_0);

    // build multiplicative incremental defgrd
    Core::LinAlg::Matrix<3, 3> defgrd_0(false);
    defgrd_0.multiply_tt(xdisp, N_xyz_0);
    defgrd_0(0, 0) += 1.0;
    defgrd_0(1, 1) += 1.0;
    defgrd_0(2, 2) += 1.0;

    // get stored old incremental F
    Core::LinAlg::Matrix<3, 3> Fhist;
    prestress_->storageto_matrix(NUMGPT_SOH8, Fhist, prestress_->f_history());

    // build total defgrd = delta F * F_old
    Core::LinAlg::Matrix<3, 3> tmp;
    tmp.multiply(defgrd_0, Fhist);
    defgrd_0 = tmp;

    // build inverse and detF
    invdefgrd_0.invert(defgrd_0);
    detF_0 = defgrd_0.determinant();
  }
  else  // no prestressing
  {
    // deformation gradient and its determinant at centroid
    Core::LinAlg::Matrix<3, 3> defgrd_0(false);
    defgrd_0.multiply_tt(xcurr, N_XYZ_0);
    invdefgrd_0.invert(defgrd_0);
    detF_0 = defgrd_0.determinant();
  }
  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_XYZ;
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd(false);
  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
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
    else  // no prestressing
    {
      // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
      defgrd.multiply_tt(xcurr, N_XYZ);
    }
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invdefgrd;
    invdefgrd.invert(defgrd);
    double detF = defgrd.determinant();

    // Right Cauchy-Green tensor = F^T * F
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> cauchygreen;
    cauchygreen.multiply_tn(defgrd, defgrd);

    // check for negative jacobian
    if (detF_0 < 0. || detF < 0.)
    {
      // check, if errors are tolerated or should throw a FOUR_C_THROW
      if (is_params_interface())
      {
        if (str_params_interface().is_tolerate_errors())
        {
          str_params_interface().set_ele_eval_error_flag(
              Solid::ELEMENTS::ele_error_negative_det_of_def_gradient);
          stiffmatrix->clear();
          force->clear();
          return;
        }
        else
          FOUR_C_THROW("negative defomration gradient determinant");
      }
      // FixMe Deprecated implementation
      else
      {
        bool error_tol = false;
        if (params.isParameter("tolerate_errors")) error_tol = params.get<bool>("tolerate_errors");
        if (error_tol)
        {
          params.set<bool>("eval_error", true);
          stiffmatrix->clear();
          force->clear();
          return;
        }
        else
          FOUR_C_THROW("negative jacobian determinant");
      }
    }
    // F_bar deformation gradient =(detF_0/detF)^1/3*F
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd_bar(defgrd);
    double f_bar_factor = pow(detF_0 / detF, 1.0 / 3.0);
    defgrd_bar.scale(f_bar_factor);

    // Right Cauchy-Green tensor(Fbar) = F_bar^T * F_bar
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> cauchygreen_bar;
    cauchygreen_bar.multiply_tn(defgrd_bar, defgrd_bar);

    // Green-Lagrange strains(F_bar) matrix E = 0.5 * (Cauchygreen(F_bar) - Identity)
    // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
    Core::LinAlg::SerialDenseVector glstrain_bar_epetra(Mat::NUM_STRESS_3D);
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> glstrain_bar(glstrain_bar_epetra.values(), true);
    glstrain_bar(0) = 0.5 * (cauchygreen_bar(0, 0) - 1.0);
    glstrain_bar(1) = 0.5 * (cauchygreen_bar(1, 1) - 1.0);
    glstrain_bar(2) = 0.5 * (cauchygreen_bar(2, 2) - 1.0);
    glstrain_bar(3) = cauchygreen_bar(0, 1);
    glstrain_bar(4) = cauchygreen_bar(1, 2);
    glstrain_bar(5) = cauchygreen_bar(2, 0);

    // return gp strains (only in case of stress/strain output)
    switch (iostrain)
    {
      case Inpar::Solid::strain_gl:
      {
        if (elestrain == nullptr) FOUR_C_THROW("strain data not available");
        for (int i = 0; i < 3; ++i) (*elestrain)(gp, i) = glstrain_bar(i);
        for (int i = 3; i < 6; ++i) (*elestrain)(gp, i) = 0.5 * glstrain_bar(i);
      }
      break;
      case Inpar::Solid::strain_ea:
      {
        if (elestrain == nullptr) FOUR_C_THROW("strain data not available");
        // rewriting Green-Lagrange strains in matrix format
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> gl_bar;
        gl_bar(0, 0) = glstrain_bar(0);
        gl_bar(0, 1) = 0.5 * glstrain_bar(3);
        gl_bar(0, 2) = 0.5 * glstrain_bar(5);
        gl_bar(1, 0) = gl_bar(0, 1);
        gl_bar(1, 1) = glstrain_bar(1);
        gl_bar(1, 2) = 0.5 * glstrain_bar(4);
        gl_bar(2, 0) = gl_bar(0, 2);
        gl_bar(2, 1) = gl_bar(1, 2);
        gl_bar(2, 2) = glstrain_bar(2);

        // inverse of fbar deformation gradient
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invdefgrd_bar;
        invdefgrd_bar.invert(defgrd_bar);

        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> temp;
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> euler_almansi_bar;
        temp.multiply(gl_bar, invdefgrd_bar);
        euler_almansi_bar.multiply_tn(invdefgrd_bar, temp);

        (*elestrain)(gp, 0) = euler_almansi_bar(0, 0);
        (*elestrain)(gp, 1) = euler_almansi_bar(1, 1);
        (*elestrain)(gp, 2) = euler_almansi_bar(2, 2);
        (*elestrain)(gp, 3) = euler_almansi_bar(0, 1);
        (*elestrain)(gp, 4) = euler_almansi_bar(1, 2);
        (*elestrain)(gp, 5) = euler_almansi_bar(0, 2);
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

    // call material law
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D> cmat(true);
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> stress_bar(true);

    // in case of temperature-dependent material parameters, e.g. Young's modulus,
    // i.e. E(T), current element temperature T_{n+1} required for stress and cmat

    UTILS::get_temperature_for_structural_material<Core::FE::CellType::hex8>(shapefcts[gp], params);

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

    solid_material()->evaluate(&defgrd_bar, &glstrain_bar, params, &stress_bar, &cmat, gp, id());
    // end of call material law

    // print plastic strains
    // CAUTION: print plastic strains ONLY in case of small strain regime!
    switch (ioplstrain)
    {
      case Inpar::Solid::strain_gl:
      {
        if (eleplstrain == nullptr) FOUR_C_THROW("plastic strain data not available");
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> plglstrain_bar =
            params.get<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("plglstrain");
        for (int i = 0; i < 3; ++i) (*eleplstrain)(gp, i) = plglstrain_bar(i);
        for (int i = 3; i < 6; ++i) (*eleplstrain)(gp, i) = 0.5 * plglstrain_bar(i);
      }
      break;
      case Inpar::Solid::strain_ea:
      {
        if (eleplstrain == nullptr) FOUR_C_THROW("plastic strain data not available");
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> plglstrain_bar =
            params.get<Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1>>("plglstrain");

        // e = F^{T-1} . E . F^{-1}
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> euler_almansi_bar;
        g_lto_ea(&plglstrain_bar, &defgrd_bar, &euler_almansi_bar);

        (*eleplstrain)(gp, 0) = euler_almansi_bar(0, 0);
        (*eleplstrain)(gp, 1) = euler_almansi_bar(1, 1);
        (*eleplstrain)(gp, 2) = euler_almansi_bar(2, 2);
        (*eleplstrain)(gp, 3) = euler_almansi_bar(0, 1);
        (*eleplstrain)(gp, 4) = euler_almansi_bar(1, 2);
        (*eleplstrain)(gp, 5) = euler_almansi_bar(0, 2);
      }
      break;
      case Inpar::Solid::strain_none:
        break;

      default:
        FOUR_C_THROW("requested plastic strain type not available");
        break;
    }  // switch (ioplstrain)

    // return gp stresses
    switch (iostress)
    {
      case Inpar::Solid::stress_2pk:
      {
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");
        for (int i = 0; i < Mat::NUM_STRESS_3D; ++i) (*elestress)(gp, i) = stress_bar(i);
      }
      break;
      case Inpar::Solid::stress_cauchy:
      {
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");
        const double detF_bar = defgrd_bar.determinant();

        Core::LinAlg::Matrix<3, 3> pkstress_bar;
        pkstress_bar(0, 0) = stress_bar(0);
        pkstress_bar(0, 1) = stress_bar(3);
        pkstress_bar(0, 2) = stress_bar(5);
        pkstress_bar(1, 0) = pkstress_bar(0, 1);
        pkstress_bar(1, 1) = stress_bar(1);
        pkstress_bar(1, 2) = stress_bar(4);
        pkstress_bar(2, 0) = pkstress_bar(0, 2);
        pkstress_bar(2, 1) = pkstress_bar(1, 2);
        pkstress_bar(2, 2) = stress_bar(2);

        Core::LinAlg::Matrix<3, 3> temp;
        Core::LinAlg::Matrix<3, 3> cauchystress_bar;
        temp.multiply(1.0 / detF_bar, defgrd_bar, pkstress_bar);
        cauchystress_bar.multiply_nt(temp, defgrd_bar);

        (*elestress)(gp, 0) = cauchystress_bar(0, 0);
        (*elestress)(gp, 1) = cauchystress_bar(1, 1);
        (*elestress)(gp, 2) = cauchystress_bar(2, 2);
        (*elestress)(gp, 3) = cauchystress_bar(0, 1);
        (*elestress)(gp, 4) = cauchystress_bar(1, 2);
        (*elestress)(gp, 5) = cauchystress_bar(0, 2);
      }
      break;
      case Inpar::Solid::stress_none:
        break;
      default:
        FOUR_C_THROW("requested stress type not available");
        break;
    }

    double detJ_w = detJ * gpweights[gp];

    // update internal force vector
    if (force != nullptr)
    {
      // integrate internal force vector f = f + (B^T . sigma) * detJ * w(gp)
      force->multiply_tn(detJ_w / f_bar_factor, bop, stress_bar, 1.0);
    }

    // update stiffness matrix
    if (stiffmatrix != nullptr)
    {
      // integrate `elastic' and `initial-displacement' stiffness matrix
      // keu = keu + (B^T . C . B) * detJ * w(gp)
      Core::LinAlg::Matrix<6, NUMDOF_SOH8> cb;
      cb.multiply(cmat, bop);
      stiffmatrix->multiply_tn(detJ_w * f_bar_factor, bop, cb, 1.0);

      // integrate `geometric' stiffness matrix and add to keu *****************
      Core::LinAlg::Matrix<6, 1> sfac(stress_bar);  // auxiliary integrated stress
      sfac.scale(detJ_w / f_bar_factor);  // detJ*w(gp)*[S11,S22,S33,S12=S21,S23=S32,S13=S31]
      std::vector<double> SmB_L(3);       // intermediate Sm.B_L
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

      // integrate additional fbar matrix
      Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> cauchygreenvector;
      cauchygreenvector(0) = cauchygreen(0, 0);
      cauchygreenvector(1) = cauchygreen(1, 1);
      cauchygreenvector(2) = cauchygreen(2, 2);
      cauchygreenvector(3) = 2 * cauchygreen(0, 1);
      cauchygreenvector(4) = 2 * cauchygreen(1, 2);
      cauchygreenvector(5) = 2 * cauchygreen(2, 0);

      Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> ccg;
      ccg.multiply(cmat, cauchygreenvector);

      Core::LinAlg::Matrix<NUMDOF_SOH8, 1> bopccg(false);  // auxiliary integrated stress
      bopccg.multiply_tn(detJ_w * f_bar_factor / 3.0, bop, ccg);

      double htensor[NUMDOF_SOH8];
      for (int n = 0; n < NUMDOF_SOH8; n++)
      {
        htensor[n] = 0;
        for (int i = 0; i < NUMDIM_SOH8; i++)
        {
          htensor[n] +=
              invdefgrd_0(i, n % 3) * N_XYZ_0(i, n / 3) - invdefgrd(i, n % 3) * N_XYZ(i, n / 3);
        }
      }

      Core::LinAlg::Matrix<NUMDOF_SOH8, 1> bops(false);  // auxiliary integrated stress
      bops.multiply_tn(-detJ_w / f_bar_factor / 3.0, bop, stress_bar);
      for (int i = 0; i < NUMDOF_SOH8; i++)
      {
        for (int j = 0; j < NUMDOF_SOH8; j++)
        {
          (*stiffmatrix)(i, j) += htensor[j] * (bops(i, 0) + bopccg(i, 0));
        }
      }  // end of integrate additional `fbar' stiffness**********************
    }    // if (stiffmatrix != nullptr)

    if (massmatrix != nullptr)  // evaluate mass matrix +++++++++++++++++++++++++
    {
      double density = material()->density(gp);
      // integrate consistent mass matrix
      const double factor = detJ_w * density;
      double ifactor, massfactor;
      for (int inod = 0; inod < NUMNOD_SOH8; ++inod)
      {
        ifactor = shapefcts[gp](inod) * factor;
        for (int jnod = 0; jnod < NUMNOD_SOH8; ++jnod)
        {
          massfactor = shapefcts[gp](jnod) * ifactor;  // intermediate factor
          (*massmatrix)(NUMDIM_SOH8 * inod + 0, NUMDIM_SOH8 * jnod + 0) += massfactor;
          (*massmatrix)(NUMDIM_SOH8 * inod + 1, NUMDIM_SOH8 * jnod + 1) += massfactor;
          (*massmatrix)(NUMDIM_SOH8 * inod + 2, NUMDIM_SOH8 * jnod + 2) += massfactor;
        }
      }

      // check for non constant mass matrix
      if (solid_material()->varying_density())
      {
        /*
         If the density, i.e. the mass matrix, is not constant, a linearization is neccessary.
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
        // Right Cauchy-Green tensor = F^T * F
        Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> cauchygreen;
        cauchygreen.multiply_tn(defgrd, defgrd);

        // GL strain vector glstrain={E11,E22,E33,2*E12,2*E23,2*E31}
        Core::LinAlg::SerialDenseVector glstrain_epetra(Mat::NUM_STRESS_3D);
        Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, 1> glstrain(glstrain_epetra.values(), true);
        // if (kintype_ == Discret::ELEMENTS::So_hex8::soh8_nonlinear)
        //{
        // Green-Lagrange strains matrix E = 0.5 * (Cauchygreen - Identity)
        glstrain(0) = 0.5 * (cauchygreen(0, 0) - 1.0);
        glstrain(1) = 0.5 * (cauchygreen(1, 1) - 1.0);
        glstrain(2) = 0.5 * (cauchygreen(2, 2) - 1.0);
        glstrain(3) = cauchygreen(0, 1);
        glstrain(4) = cauchygreen(1, 2);
        glstrain(5) = cauchygreen(2, 0);
        //}

        solid_material()->evaluate_non_lin_mass(
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
                  (*massmatrix)(inod * NUMDIM_SOH8 + idim, jnod * NUMDIM_SOH8 + jdim) +=
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
              (*forceinert)(inod * NUMDIM_SOH8 + idim) += forcefactor * density * myacc(idim);
          }
        }
      }
    }  // end of mass matrix +++++++++++++++++++++++++++++++++++++++++++++++++++

  } /* ==================================================== end of Loop over GP */

  return;
}  // Discret::ELEMENTS::So_hex8fbar::nlnstiffmass

/*----------------------------------------------------------------------*
 |  init the element (public)                                           |
 *----------------------------------------------------------------------*/
int Discret::ELEMENTS::SoHex8fbarType::initialize(Core::FE::Discretization& dis)
{
  for (int i = 0; i < dis.num_my_col_elements(); ++i)
  {
    if (dis.l_col_element(i)->element_type() != *this) continue;
    auto* actele = dynamic_cast<Discret::ELEMENTS::SoHex8fbar*>(dis.l_col_element(i));
    if (!actele) FOUR_C_THROW("cast to So_hex8fbar* failed");
    actele->init_jacobian_mapping();
  }
  return 0;
}


/*----------------------------------------------------------------------*
 |  compute def gradient at every gaussian point (protected)   gee 07/08|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8fbar::def_gradient(const std::vector<double>& disp,
    Core::LinAlg::SerialDenseMatrix& gpdefgrd, Discret::ELEMENTS::PreStress& prestress)
{
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs = soh8_derivs();
  // derivatives at centroid point
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_rst_0;
  Core::FE::shape_function_3d_deriv1(N_rst_0, 0.0, 0.0, 0.0, Core::FE::CellType::hex8);

  // update element geometry
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xdisp;  // current  coord. of element
  for (int i = 0; i < NUMNOD_SOH8; ++i)
  {
    xdisp(i, 0) = disp[i * NODDOF_SOH8 + 0];
    xdisp(i, 1) = disp[i * NODDOF_SOH8 + 1];
    xdisp(i, 2) = disp[i * NODDOF_SOH8 + 2];
  }

  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
  {
    // get Jacobian mapping wrt to the stored deformed configuration
    Core::LinAlg::Matrix<3, 3> invJdef;
    prestress.storageto_matrix(gp, invJdef, prestress.j_history());

    // by N_XYZ = J^-1 * N_rst
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_xyz;
    N_xyz.multiply(invJdef, derivs[gp]);

    // build defgrd (independent of xrefe!)
    Core::LinAlg::Matrix<3, 3> defgrd;
    defgrd.multiply_tt(xdisp, N_xyz);
    defgrd(0, 0) += 1.0;
    defgrd(1, 1) += 1.0;
    defgrd(2, 2) += 1.0;

    prestress.matrixto_storage(gp, defgrd, gpdefgrd);
  }

  {
    // get Jacobian mapping wrt to the stored deformed configuration
    Core::LinAlg::Matrix<3, 3> invJdef;
    prestress.storageto_matrix(NUMGPT_SOH8, invJdef, prestress.j_history());

    // by N_XYZ = J^-1 * N_rst
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_xyz;
    N_xyz.multiply(invJdef, N_rst_0);

    // build defgrd (independent of xrefe!)
    Core::LinAlg::Matrix<3, 3> defgrd;
    defgrd.multiply_tt(xdisp, N_xyz);
    defgrd(0, 0) += 1.0;
    defgrd(1, 1) += 1.0;
    defgrd(2, 2) += 1.0;

    prestress.matrixto_storage(NUMGPT_SOH8, defgrd, gpdefgrd);
  }

  return;
}

/*----------------------------------------------------------------------*
 |  compute Jac.mapping wrt deformed configuration (protected) gee 07/08|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8fbar::update_jacobian_mapping(
    const std::vector<double>& disp, Discret::ELEMENTS::PreStress& prestress)
{
  const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs = soh8_derivs();
  // derivatives at centroid
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_rst_0;
  Core::FE::shape_function_3d_deriv1(N_rst_0, 0.0, 0.0, 0.0, Core::FE::CellType::hex8);

  // get incremental disp
  Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xdisp;
  for (int i = 0; i < NUMNOD_SOH8; ++i)
  {
    xdisp(i, 0) = disp[i * NODDOF_SOH8 + 0];
    xdisp(i, 1) = disp[i * NODDOF_SOH8 + 1];
    xdisp(i, 2) = disp[i * NODDOF_SOH8 + 2];
  }

  Core::LinAlg::Matrix<3, 3> invJhist;
  Core::LinAlg::Matrix<3, 3> invJ;
  Core::LinAlg::Matrix<3, 3> defgrd;
  Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_xyz;
  Core::LinAlg::Matrix<3, 3> invJnew;
  for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
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
  }  // for (unsigned gp=0; gp<NUMGPT_SOH8; ++gp)

  {
    // get the invJ old state
    prestress.storageto_matrix(NUMGPT_SOH8, invJhist, prestress.j_history());
    // get derivatives wrt to invJhist
    N_xyz.multiply(invJhist, N_rst_0);
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
    prestress.matrixto_storage(NUMGPT_SOH8, invJnew, prestress.j_history());
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Update inelastic deformation (G&R)                       braeu 07/16|
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::SoHex8fbar::update_element(std::vector<double>& disp,
    Teuchos::ParameterList& params, const Teuchos::RCP<Core::Mat::Material>& mat)
{
  if (solid_material()->uses_extended_update())
  {
    /* ============================================================================*
     ** CONST SHAPE FUNCTIONS, DERIVATIVES and WEIGHTS for HEX_8 with 8 GAUSS POINTS*
     ** ============================================================================*/
    const static std::vector<Core::LinAlg::Matrix<NUMNOD_SOH8, 1>> shapefcts = soh8_shapefcts();
    const static std::vector<Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8>> derivs = soh8_derivs();
    const static std::vector<double> gpweights = soh8_weights();
    /* ============================================================================*/

    // update element geometry
    Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xrefe;  // material coord. of element
    Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xcurr;  // current  coord. of element
    Core::LinAlg::Matrix<NUMNOD_SOH8, NUMDIM_SOH8> xdisp;

    for (int i = 0; i < NUMNOD_SOH8; ++i)
    {
      const auto& x = nodes()[i]->x();
      xrefe(i, 0) = x[0];
      xrefe(i, 1) = x[1];
      xrefe(i, 2) = x[2];

      xcurr(i, 0) = xrefe(i, 0) + disp[i * NODDOF_SOH8 + 0];
      xcurr(i, 1) = xrefe(i, 1) + disp[i * NODDOF_SOH8 + 1];
      xcurr(i, 2) = xrefe(i, 2) + disp[i * NODDOF_SOH8 + 2];

      if (Prestress::is_mulf(pstype_))
      {
        xdisp(i, 0) = disp[i * NODDOF_SOH8 + 0];
        xdisp(i, 1) = disp[i * NODDOF_SOH8 + 1];
        xdisp(i, 2) = disp[i * NODDOF_SOH8 + 2];
      }
    }


    //****************************************************************************
    // deformation gradient at centroid of element
    //****************************************************************************
    double detF_0 = -1.0;
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invdefgrd_0;
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_XYZ_0;
    // element coordinate derivatives at centroid
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_rst_0;
    Core::FE::shape_function_3d_deriv1(N_rst_0, 0.0, 0.0, 0.0, Core::FE::CellType::hex8);
    {
      // inverse jacobian matrix at centroid
      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> invJ_0;
      invJ_0.multiply(N_rst_0, xrefe);
      invJ_0.invert();
      // material derivatives at centroid
      N_XYZ_0.multiply(invJ_0, N_rst_0);
    }

    if (Prestress::is_mulf(pstype_))
    {
      // get Jacobian mapping wrt to the stored configuration
      // centroid is 9th Gaussian point in storage
      Core::LinAlg::Matrix<3, 3> invJdef_0;
      prestress_->storageto_matrix(NUMGPT_SOH8, invJdef_0, prestress_->j_history());
      // get derivatives wrt to last spatial configuration
      Core::LinAlg::Matrix<3, 8> N_xyz_0;
      N_xyz_0.multiply(invJdef_0, N_rst_0);  // if (!Id()) std::cout << invJdef_0;

      // build multiplicative incremental defgrd
      Core::LinAlg::Matrix<3, 3> defgrd_0(false);
      defgrd_0.multiply_tt(xdisp, N_xyz_0);
      defgrd_0(0, 0) += 1.0;
      defgrd_0(1, 1) += 1.0;
      defgrd_0(2, 2) += 1.0;

      // get stored old incremental F
      Core::LinAlg::Matrix<3, 3> Fhist;
      prestress_->storageto_matrix(NUMGPT_SOH8, Fhist, prestress_->f_history());

      // build total defgrd = delta F * F_old
      Core::LinAlg::Matrix<3, 3> tmp;
      tmp.multiply(defgrd_0, Fhist);
      defgrd_0 = tmp;

      // build inverse and detF
      invdefgrd_0.invert(defgrd_0);
      detF_0 = defgrd_0.determinant();
    }
    else  // no prestressing
    {
      // deformation gradient and its determinant at centroid
      Core::LinAlg::Matrix<3, 3> defgrd_0(false);
      defgrd_0.multiply_tt(xcurr, N_XYZ_0);
      invdefgrd_0.invert(defgrd_0);
      detF_0 = defgrd_0.determinant();
    }


    /* =========================================================================*/
    /* ================================================= Loop over Gauss Points */
    /* =========================================================================*/
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> N_XYZ;
    // build deformation gradient wrt to material configuration
    Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd(false);

    // center of element in reference configuration
    Core::LinAlg::Matrix<NUMDIM_SOH8, 1> point(false);
    point.clear();
    soh8_element_center_refe_coords(point, xrefe);
    params.set("elecenter_coords_ref", point);

    for (unsigned gp = 0; gp < NUMGPT_SOH8; ++gp)
    {
      Core::LinAlg::Matrix<NUMDIM_SOH8, 1> point(true);
      soh8_gauss_point_refe_coords(point, xrefe, gp);
      params.set("gp_coords_ref", point);

      /* get the inverse of the Jacobian matrix which looks like:
       **            [ x_,r  y_,r  z_,r ]^-1
       **     J^-1 = [ x_,s  y_,s  z_,s ]
       **            [ x_,t  y_,t  z_,t ]
       */
      // compute derivatives N_XYZ at gp w.r.t. material coordinates
      // by N_XYZ = J^-1 * N_rst
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
      else  // no prestressing
      {
        // (material) deformation gradient F = d xcurr / d xrefe = xcurr^T * N_XYZ^T
        defgrd.multiply_tt(xcurr, N_XYZ);
      }

      double detF = defgrd.determinant();

      // F_bar deformation gradient =(detF_0/detF)^1/3*F
      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMDIM_SOH8> defgrd_bar(defgrd);
      double f_bar_factor = pow(detF_0 / detF, 1.0 / 3.0);
      defgrd_bar.scale(f_bar_factor);

      // This is an additional update call needed for G&R materials (mixture,
      // growthremodel_elasthyper)
      solid_material()->update(defgrd_bar, gp, params, id());
    }
  }
  else
    solid_material()->update();

  return;
}

FOUR_C_NAMESPACE_CLOSE
