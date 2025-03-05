// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_so3_poro.hpp"
#include "4C_so3_poro_eletypes.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

#include <iterator>

FOUR_C_NAMESPACE_OPEN

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::pre_evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
}

template <class So3Ele, Core::FE::CellType distype>
int Discret::Elements::So3Poro<So3Ele, distype>::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  if (not init_) FOUR_C_THROW("internal element data not initialized!");

  // set the pointer to the parameter list in element
  So3Ele::set_params_interface_ptr(params);

  // start with "none"
  Core::Elements::ActionType act = Core::Elements::none;

  if (So3Ele::is_params_interface())
  {
    act = So3Ele::params_interface().get_action_type();
  }
  else
  {
    // get the required action
    std::string action = params.get<std::string>("action", "none");
    if (action == "none")
      FOUR_C_THROW("No action supplied");
    else if (action == "struct_poro_calc_fluidcoupling")
      act = Core::Elements::struct_poro_calc_fluidcoupling;
  }

  // what should the element do
  switch (act)
  {
    //==================================================================================
    // off diagonal terms in stiffness matrix for monolithic coupling
    case Core::Elements::struct_poro_calc_fluidcoupling:
    {
      my_evaluate(params, discretization, la, elemat1_epetra, elemat2_epetra, elevec1_epetra,
          elevec2_epetra, elevec3_epetra);
    }
    break;
    default:
    {
      // in some cases we need to write/change some data before evaluating
      pre_evaluate(params, discretization, la);

      // evaluate parent solid element
      So3Ele::evaluate(params, discretization, la[0].lm_, elemat1_epetra, elemat2_epetra,
          elevec1_epetra, elevec2_epetra, elevec3_epetra);

      // add volume coupling specific terms
      my_evaluate(params, discretization, la, elemat1_epetra, elemat2_epetra, elevec1_epetra,
          elevec2_epetra, elevec3_epetra);
    }
    break;
  }

  return 0;
}

template <class So3Ele, Core::FE::CellType distype>
int Discret::Elements::So3Poro<So3Ele, distype>::my_evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // start with "none"
  // ActionType act = none;
  Core::Elements::ActionType act = Core::Elements::none;

  if (So3Ele::is_params_interface())
  {
    act = So3Ele::params_interface().get_action_type();
  }
  else
  {
    // get the required action
    std::string action = params.get<std::string>("action", "none");
    if (action == "none")
      FOUR_C_THROW("No action supplied");
    else if (action == "calc_struct_internalforce")
      act = Core::Elements::struct_calc_internalforce;
    else if (action == "calc_struct_nlnstiff")
      act = Core::Elements::struct_calc_nlnstiff;
    else if (action == "calc_struct_nlnstiffmass")
      act = Core::Elements::struct_calc_nlnstiffmass;
    else if (action == "struct_poro_calc_fluidcoupling")
      act = Core::Elements::struct_poro_calc_fluidcoupling;
    else if (action == "calc_struct_stress")
      act = Core::Elements::struct_calc_stress;
  }

  // what should the element do
  switch (act)
  {
    //==================================================================================
    // nonlinear stiffness, damping and internal force vector for poroelasticity
    case Core::Elements::struct_calc_nlnstiff:
    {
      // stiffness
      Core::LinAlg::Matrix<numdof_, numdof_> elemat1(elemat1_epetra.values(), true);
      // damping
      Core::LinAlg::Matrix<numdof_, numdof_> elemat2(elemat2_epetra.values(), true);
      // internal force vector
      Core::LinAlg::Matrix<numdof_, 1> elevec1(elevec1_epetra.values(), true);
      // elevec2+3 are not used anyway

      std::vector<int> lm = la[0].lm_;

      Core::LinAlg::Matrix<numdim_, numnod_> mydisp(true);
      extract_values_from_global_vector(discretization, 0, lm, &mydisp, nullptr, "displacement");

      Core::LinAlg::Matrix<numdof_, numdof_>* matptr = nullptr;
      if (elemat1.is_initialized()) matptr = &elemat1;

      enum Inpar::Solid::DampKind damping =
          params.get<enum Inpar::Solid::DampKind>("damping", Inpar::Solid::damp_none);
      Core::LinAlg::Matrix<numdof_, numdof_>* matptr2 = nullptr;
      if (elemat2.is_initialized() and (damping == Inpar::Solid::damp_material)) matptr2 = &elemat2;

      if (la.size() > 1)
      {
        if (discretization.has_state(1, "fluidvel"))
        {
          // need current fluid state,
          // call the fluid discretization: fluid equates 2nd dofset
          // disassemble velocities and pressures
          Core::LinAlg::Matrix<numdim_, numnod_> myvel(true);
          Core::LinAlg::Matrix<numdim_, numnod_> myfluidvel(true);
          Core::LinAlg::Matrix<numnod_, 1> myepreaf(true);

          if (discretization.has_state(0, "velocity"))
            extract_values_from_global_vector(
                discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

          if (discretization.has_state(1, "fluidvel"))
          {
            // extract local values of the global vectors
            extract_values_from_global_vector(
                discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");
          }

          // calculate tangent stiffness matrix
          nonlinear_stiffness_poroelast(
              lm, mydisp, myvel, myfluidvel, myepreaf, matptr, matptr2, &elevec1, params);
        }
        else if (la.size() > 2)
        {
          if (discretization.has_state(1, "porofluid"))
          {
            // get primary variables of multiphase porous medium flow
            std::vector<double> myephi(la[1].size());
            std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
                discretization.get_state(1, "porofluid");
            myephi = Core::FE::extract_values(*matrix_state, la[1].lm_);

            // calculate tangent stiffness matrix
            nonlinear_stiffness_poroelast_pressure_based(
                lm, mydisp, myephi, matptr, &elevec1, params);
          }
        }
      }
    }
    break;

    //==================================================================================
    // nonlinear stiffness, mass matrix and internal force vector for poroelasticity
    case Core::Elements::struct_calc_nlnstiffmass:
    {
      // stiffness
      Core::LinAlg::Matrix<numdof_, numdof_> elemat1(elemat1_epetra.values(), true);
      // internal force vector
      Core::LinAlg::Matrix<numdof_, 1> elevec1(elevec1_epetra.values(), true);
      // elemat2,elevec2+3 are not used anyway

      // build the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      Core::LinAlg::Matrix<numdim_, numnod_> mydisp(true);
      extract_values_from_global_vector(
          discretization, 0, la[0].lm_, &mydisp, nullptr, "displacement");

      Core::LinAlg::Matrix<numdof_, numdof_>* matptr = nullptr;
      if (elemat1.is_initialized()) matptr = &elemat1;

      if (isNurbs_)
      {
        // access knots and weights for this element
        bool zero_size = Core::FE::Nurbs::get_my_nurbs_knots_and_weights(
            discretization, this, myknots_, weights_);

        // if we have a zero sized element due to a interpolated point -> exit here
        if (zero_size) return 0;
      }

      // we skip this evaluation if the coupling is not setup yet, i.e.
      // if the secondary dofset or the secondary material was not set
      // this can happen during setup of the time integrator or restart
      // there might be a better way. For instance do not evaluate
      // before the setup of the multiphysics problem is completed.
      if (la.size() > 1 and So3Ele::num_material() > 1)
      {
        // need current fluid state,
        // call the fluid discretization: fluid equates 2nd dofset
        // disassemble velocities and pressures
        Core::LinAlg::Matrix<numdim_, numnod_> myvel(true);
        Core::LinAlg::Matrix<numdim_, numnod_> myfluidvel(true);
        Core::LinAlg::Matrix<numnod_, 1> myepreaf(true);

        if (discretization.has_state(0, "velocity"))
          extract_values_from_global_vector(
              discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

        // this is kind of a hack. Find a better way! (e.g. move the pressure based variant
        // into own element)
        if (discretization.has_state(1, "fluidvel"))
        {
          // extract local values of the global vectors
          extract_values_from_global_vector(
              discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

          // calculate tangent stiffness matrix
          nonlinear_stiffness_poroelast(
              lm, mydisp, myvel, myfluidvel, myepreaf, matptr, nullptr, &elevec1, params);
        }
        else if (la.size() > 2)
        {
          if (discretization.has_state(1, "porofluid"))
          {
            // get primary variables of multiphase porous medium flow
            std::vector<double> myephi(la[1].size());
            std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
                discretization.get_state(1, "porofluid");
            myephi = Core::FE::extract_values(*matrix_state, la[1].lm_);

            // calculate tangent stiffness matrix
            nonlinear_stiffness_poroelast_pressure_based(
                lm, mydisp, myephi, matptr, &elevec1, params);
          }
        }
      }
    }
    break;

    //==================================================================================
    // coupling terms in force-vector and stiffness matrix for poroelasticity
    case Core::Elements::struct_poro_calc_fluidcoupling:
    {
      // stiffness
      Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_> elemat1(elemat1_epetra.values(), true);

      // build the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>* matptr = nullptr;
      if (elemat1.is_initialized()) matptr = &elemat1;

      if (isNurbs_)
      {
        // access knots and weights for this element
        bool zero_size = Core::FE::Nurbs::get_my_nurbs_knots_and_weights(
            discretization, this, myknots_, weights_);

        // if we have a zero sized element due to a interpolated point -> exit here
        if (zero_size) return 0;
      }

      // need current fluid state,
      // call the fluid discretization: fluid equates 2nd dofset
      // disassemble velocities and pressures
      if (discretization.has_state(1, "fluidvel"))
      {
        Core::LinAlg::Matrix<numdim_, numnod_> myvel(true);
        Core::LinAlg::Matrix<numdim_, numnod_> myfluidvel(true);
        Core::LinAlg::Matrix<numnod_, 1> myepreaf(true);

        Core::LinAlg::Matrix<numdim_, numnod_> mydisp(true);
        extract_values_from_global_vector(
            discretization, 0, la[0].lm_, &mydisp, nullptr, "displacement");

        if (discretization.has_state(0, "velocity"))
          extract_values_from_global_vector(
              discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

        if (discretization.has_state(1, "fluidvel"))
        {
          // extract local values of the global vectors
          extract_values_from_global_vector(
              discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");
        }

        coupling_poroelast(
            lm, mydisp, myvel, myfluidvel, myepreaf, matptr, nullptr, nullptr, params);
      }
      else if (la.size() > 2)
      {
        if (discretization.has_state(1, "porofluid"))
        {
          // get primary variables of multiphase porous medium flow
          std::vector<double> myephi(la[1].size());
          std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
              discretization.get_state(1, "porofluid");
          myephi = Core::FE::extract_values(*matrix_state, la[1].lm_);

          Core::LinAlg::Matrix<numdim_, numnod_> mydisp(true);
          extract_values_from_global_vector(
              discretization, 0, la[0].lm_, &mydisp, nullptr, "displacement");

          // calculate OD-Matrix
          coupling_poroelast_pressure_based(lm, mydisp, myephi, elemat1_epetra, params);
        }
        else
          FOUR_C_THROW("cannot find global states displacement or solidpressure");
      }
    }
    break;

    //==================================================================================
    // nonlinear stiffness and internal force vector for poroelasticity
    case Core::Elements::struct_calc_internalforce:
    {
      // internal force vector
      Core::LinAlg::Matrix<numdof_, 1> elevec1(elevec1_epetra.values(), true);
      // elemat1+2,elevec2+3 are not used anyway

      // build the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      Core::LinAlg::Matrix<numdim_, numnod_> mydisp(true);
      extract_values_from_global_vector(discretization, 0, lm, &mydisp, nullptr, "displacement");

      // need current fluid state,
      // call the fluid discretization: fluid equates 2nd dofset
      // disassemble velocities and pressures
      if (discretization.has_state(1, "fluidvel"))
      {
        // extract local values of the global vectors
        Core::LinAlg::Matrix<numdim_, numnod_> myfluidvel(true);
        Core::LinAlg::Matrix<numnod_, 1> myepreaf(true);
        extract_values_from_global_vector(
            discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

        Core::LinAlg::Matrix<numdim_, numnod_> myvel(true);
        extract_values_from_global_vector(
            discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

        // calculate tangent stiffness matrix
        nonlinear_stiffness_poroelast(
            lm, mydisp, myvel, myfluidvel, myepreaf, nullptr, nullptr, &elevec1, params);
      }
      else if (la.size() > 2)
      {
        if (discretization.has_state(1, "porofluid"))
        {
          // get primary variables of multiphase porous medium flow
          std::vector<double> myephi(la[1].size());
          std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
              discretization.get_state(1, "porofluid");
          myephi = Core::FE::extract_values(*matrix_state, la[1].lm_);

          // calculate tangent stiffness matrix
          nonlinear_stiffness_poroelast_pressure_based(
              lm, mydisp, myephi, nullptr, &elevec1, params);
        }
      }
    }
    break;

    //==================================================================================
    // evaluate stresses and strains at gauss points
    case Core::Elements::struct_calc_stress:
    {
      // get the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      Core::LinAlg::Matrix<numdim_, numnod_> mydisp(true);
      extract_values_from_global_vector(discretization, 0, lm, &mydisp, nullptr, "displacement");

      std::shared_ptr<std::vector<char>> couplingstressdata = nullptr;
      Inpar::Solid::StressType iocouplingstress = Inpar::Solid::stress_none;
      if (this->is_params_interface())
      {
        couplingstressdata = this->str_params_interface().coupling_stress_data_ptr();
        iocouplingstress = this->str_params_interface().get_coupling_stress_output_type();
      }
      else
      {
        iocouplingstress =
            params.get<Inpar::Solid::StressType>("iocouplstress", Inpar::Solid::stress_none);

        // check for output of coupling stress
        if (iocouplingstress == Inpar::Solid::stress_none)
        {
          // nothing to do here for calculation of effective stress
          break;
        }

        couplingstressdata = params.get<std::shared_ptr<std::vector<char>>>("couplstress", nullptr);

        if (couplingstressdata == nullptr) FOUR_C_THROW("Cannot get 'couplstress' data");
      }

      // initialize the coupling stress
      Core::LinAlg::SerialDenseMatrix couplstress(numgpt_, numstr_);
      // need current fluid state,
      // call the fluid discretization: fluid equates 2nd dofset
      // disassemble velocities and pressures
      if (iocouplingstress != Inpar::Solid::stress_none)
      {
        if (discretization.has_state(1, "fluidvel"))
        {
          // extract local values of the global vectors
          Core::LinAlg::Matrix<numdim_, numnod_> myfluidvel(true);
          Core::LinAlg::Matrix<numnod_, 1> myepreaf(true);
          extract_values_from_global_vector(
              discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

          coupling_stress_poroelast(
              mydisp, myfluidvel, myepreaf, &couplstress, nullptr, params, iocouplingstress);
        }
        else if (la.size() > 2)
        {
          if (discretization.has_state(1, "porofluid"))
          {
            FOUR_C_THROW("coupl stress poroelast not yet implemented for pressure-based variant");
          }
        }
      }

      // pack the data for postprocessing
      {
        Core::Communication::PackBuffer data;

        add_to_pack(data, couplstress);
        std::copy(data().begin(), data().end(), std::back_inserter(*couplingstressdata));
      }
    }
    break;
    //==================================================================================
    default:
      // do nothing (no error because there are some actions the poro element is supposed to ignore)
      break;
  }
  return 0;
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::nonlinear_stiffness_poroelast(
    std::vector<int>& lm,                                 // location matrix
    Core::LinAlg::Matrix<numdim_, numnod_>& disp,         // current displacements
    Core::LinAlg::Matrix<numdim_, numnod_>& vel,          // current velocities
    Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,       // current fluid velocities
    Core::LinAlg::Matrix<numnod_, 1>& epreaf,             // current fluid pressure
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,  // element stiffness matrix
    Core::LinAlg::Matrix<numdof_, numdof_>* reamatrix,    // element reactive matrix
    Core::LinAlg::Matrix<numdof_, 1>* force,              // element internal force vector
    // Core::LinAlg::Matrix<numgptpar_, numstr_>* elestress, // stresses at GP
    // Core::LinAlg::Matrix<numgptpar_, numstr_>* elestrain, // strains at GP
    Teuchos::ParameterList&
        params  // algorithmic parameters e.g. time
                //   const Inpar::Solid::StressType       iostress     // stress output option
)
{
  get_materials();

  // update element geometry
  Core::LinAlg::Matrix<numdim_, numnod_> xrefe;  // material coord. of element
  Core::LinAlg::Matrix<numdim_, numnod_> xcurr;  // current  coord. of element


  for (int i = 0; i < numnod_; ++i)
  {
    const auto& x = nodes()[i]->x();
    for (int j = 0; j < numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }

  // initialize element matrizes and vectors
  Core::LinAlg::Matrix<numdof_, numdof_> erea_v(true);

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  gauss_point_loop(
      params, xrefe, xcurr, disp, vel, evelnp, epreaf, nullptr, erea_v, stiffmatrix, force);

  // update stiffness matrix
  if (stiffmatrix != nullptr)
  {
    if (reamatrix != nullptr)
    {
      /* additional "reactive darcy-term"
       detJ * w(gp) * ( J * reacoeff * phi^2  ) * D(v_s)
       */
      reamatrix->update(1.0, erea_v, 1.0);
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::nonlinear_stiffness_poroelast_pressure_based(
    std::vector<int>& lm,                          // location matrix
    Core::LinAlg::Matrix<numdim_, numnod_>& disp,  // current displacements
    const std::vector<double>& ephi,               // primary variable for poro-multiphase flow
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,  // element stiffness matrix
    Core::LinAlg::Matrix<numdof_, 1>* force,              // element internal force vector
    Teuchos::ParameterList& params                        // algorithmic parameters e.g. time
)
{
  get_materials_pressure_based();

  // update element geometry
  Core::LinAlg::Matrix<numdim_, numnod_> xrefe;  // material coord. of element
  Core::LinAlg::Matrix<numdim_, numnod_> xcurr;  // current  coord. of element


  for (int i = 0; i < numnod_; ++i)
  {
    const auto& x = nodes()[i]->x();
    for (int j = 0; j < numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  gauss_point_loop_pressure_based(params, xrefe, xcurr, disp, ephi, stiffmatrix, force);
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::gauss_point_loop(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodalvel,
    const Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,
    const Core::LinAlg::Matrix<numnod_, 1>& epreaf,
    const Core::LinAlg::Matrix<numnod_, 1>* porosity_dof,
    Core::LinAlg::Matrix<numdof_, numdof_>& erea_v,
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix, Core::LinAlg::Matrix<numdof_, 1>* force)
{
  static Core::LinAlg::Matrix<numdim_, numnod_> N_XYZ;
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  static Core::LinAlg::Matrix<numdim_, numdim_> defgrd(true);
  static Core::LinAlg::Matrix<numnod_, 1> shapefct;
  static Core::LinAlg::Matrix<numdim_, numnod_> deriv;

  static Core::LinAlg::Matrix<numstr_, 1> fstress(true);

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    compute_shape_functions_and_derivatives(gp, shapefct, deriv, N_XYZ);

    // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
    compute_def_gradient(defgrd, N_XYZ, xcurr);

    // inverse deformation gradient F^-1
    static Core::LinAlg::Matrix<numdim_, numdim_> defgrd_inv(false);
    defgrd_inv.invert(defgrd);

    // jacobian determinant of transformation between spatial and material space "|dx/dX|"
    double J = 0.0;
    //------linearization of jacobi determinant detF=J w.r.t. structure displacement   dJ/d(us) =
    // dJ/dF : dF/dus = J * F^-T * N,X
    static Core::LinAlg::Matrix<1, numdof_> dJ_dus;
    // volume change (used for porosity law). Same as J in nonlinear theory.
    double volchange = 0.0;
    //------linearization of volume change w.r.t. structure displacement
    static Core::LinAlg::Matrix<1, numdof_> dvolchange_dus;

    // compute J, the volume change and the respctive linearizations w.r.t. structure displacement
    compute_jacobian_determinant_volume_change_and_linearizations(
        J, volchange, dJ_dus, dvolchange_dus, defgrd, defgrd_inv, N_XYZ, nodaldisp);

    // pressure at integration point
    double press = shapefct.dot(epreaf);

    // structure displacement and velocity at integration point
    static Core::LinAlg::Matrix<numdim_, 1> velint;
    velint.multiply(nodalvel, shapefct);

    // fluid velocity at integration point
    static Core::LinAlg::Matrix<numdim_, 1> fvelint;
    fvelint.multiply(evelnp, shapefct);

    // material fluid velocity gradient at integration point
    static Core::LinAlg::Matrix<numdim_, numdim_> fvelder;
    fvelder.multiply_nt(evelnp, N_XYZ);

    // pressure gradient at integration point
    static Core::LinAlg::Matrix<numdim_, 1> Gradp;
    Gradp.multiply(N_XYZ, epreaf);

    // non-linear B-operator
    static Core::LinAlg::Matrix<numstr_, numdof_> bop;
    bop.clear();
    compute_b_operator(bop, defgrd, N_XYZ);

    // Right Cauchy-Green tensor = F^T * F
    static Core::LinAlg::Matrix<numdim_, numdim_> cauchygreen;
    cauchygreen.multiply_tn(defgrd, defgrd);

    // inverse Right Cauchy-Green tensor
    static Core::LinAlg::Matrix<numdim_, numdim_> C_inv(false);
    C_inv.invert(cauchygreen);

    // compute some auxiliary matrixes for computation of linearization
    // dF^-T/dus
    static Core::LinAlg::Matrix<numdim_ * numdim_, numdof_> dFinvTdus(true);
    // F^-T * Grad p
    static Core::LinAlg::Matrix<numdim_, 1> Finvgradp;
    // dF^-T/dus * Grad p
    static Core::LinAlg::Matrix<numdim_, numdof_> dFinvdus_gradp(true);
    // dC^-1/dus * Grad p
    static Core::LinAlg::Matrix<numstr_, numdof_> dCinv_dus(true);

    compute_auxiliary_values(
        N_XYZ, defgrd_inv, C_inv, Gradp, dFinvTdus, Finvgradp, dFinvdus_gradp, dCinv_dus);

    // linearization of porosity w.r.t structure displacement d\phi/d(us) = d\phi/dJ*dJ/d(us)
    static Core::LinAlg::Matrix<1, numdof_> dphi_dus;
    double porosity = 0.0;

    compute_porosity_and_linearization(
        params, press, volchange, gp, shapefct, porosity_dof, dvolchange_dus, porosity, dphi_dus);

    // **********************fill stiffness matrix and force vector+++++++++++++++++++++++++
    if (fluid_mat_->type() == Mat::PAR::darcy_brinkman)
    {
      fill_matrix_and_vectors_brinkman(gp, J, porosity, fvelder, defgrd_inv, bop, C_inv, dphi_dus,
          dJ_dus, dCinv_dus, dFinvTdus, stiffmatrix, force, fstress);
    }

    fill_matrix_and_vectors(gp, shapefct, N_XYZ, J, press, porosity, velint, fvelint, fvelder,
        defgrd_inv, bop, C_inv, Finvgradp, dphi_dus, dJ_dus, dCinv_dus, dFinvdus_gradp, dFinvTdus,
        erea_v, stiffmatrix, force, fstress);
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::gauss_point_loop_pressure_based(
    Teuchos::ParameterList& params, const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp, const std::vector<double>& ephi,
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix, Core::LinAlg::Matrix<numdof_, 1>* force)
{
  /*--------------------------------- get node weights for nurbs elements */
  if (distype == Core::FE::CellType::nurbs4 || distype == Core::FE::CellType::nurbs9)
  {
    for (int inode = 0; inode < numnod_; ++inode)
    {
      auto* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes()[inode]);

      weights_(inode) = cp->w();
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  // first derivatives N_XYZ at gp w.r.t. material coordinates
  Core::LinAlg::Matrix<numdim_, numnod_> N_XYZ;
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  Core::LinAlg::Matrix<numdim_, numdim_> defgrd(true);
  // shape function at gp w.r.t. reference coordinates
  Core::LinAlg::Matrix<numnod_, 1> shapefct;
  // first derivatives at gp w.r.t. reference coordinates
  Core::LinAlg::Matrix<numdim_, numnod_> deriv;

  // Initialize
  const int totalnumdofpernode = fluidmulti_mat_->num_mat();
  const int numfluidphases = fluidmulti_mat_->num_fluid_phases();
  const int numvolfrac = fluidmulti_mat_->num_vol_frac();
  const bool hasvolfracs = (totalnumdofpernode > numfluidphases);
  std::vector<double> phiAtGP(totalnumdofpernode);

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    compute_shape_functions_and_derivatives(gp, shapefct, deriv, N_XYZ);

    // compute deformation gradient
    compute_def_gradient(defgrd, N_XYZ, xcurr);

    // inverse deformation gradient F^-1
    Core::LinAlg::Matrix<numdim_, numdim_> defgrd_inv(false);
    defgrd_inv.invert(defgrd);

    // jacobian determinant of transformation between spatial and material space "|dx/dX|"
    double J = 0.0;
    //------linearization of jacobi determinant detF=J w.r.t. structure displacement   dJ/d(us) =
    // dJ/dF : dF/dus = J * F^-T * N,X
    static Core::LinAlg::Matrix<1, numdof_> dJ_dus;
    // volume change (used for porosity law). Same as J in nonlinear theory.
    double volchange = 0.0;
    //------linearization of volume change w.r.t. structure displacement
    static Core::LinAlg::Matrix<1, numdof_> dvolchange_dus;

    // compute J, the volume change and the respctive linearizations w.r.t. structure displacement
    compute_jacobian_determinant_volume_change_and_linearizations(
        J, volchange, dJ_dus, dvolchange_dus, defgrd, defgrd_inv, N_XYZ, nodaldisp);

    // non-linear B-operator
    Core::LinAlg::Matrix<numstr_, numdof_> bop;
    bop.clear();
    compute_b_operator(bop, defgrd, N_XYZ);

    // derivative of press w.r.t. displacements (only in case of vol fracs)
    Core::LinAlg::Matrix<1, numdof_> dps_dus(true);

    //----------------------------------------------------
    // pressure at integration point
    compute_primary_variable_at_gp(ephi, totalnumdofpernode, shapefct, phiAtGP);
    double press = compute_sol_pressure_at_gp(totalnumdofpernode, numfluidphases, phiAtGP);
    // recalculate for the case of volume fractions
    if (hasvolfracs)
    {
      Core::LinAlg::Matrix<1, numdof_> dphi_dus;
      double porosity = 0.0;

      compute_porosity_and_linearization(
          params, press, volchange, gp, shapefct, nullptr, dvolchange_dus, porosity, dphi_dus);
      // save the pressure coming from the fluid S_i*p_i
      const double fluidpress = press;
      press = recalculate_sol_pressure_at_gp(
          fluidpress, porosity, totalnumdofpernode, numfluidphases, numvolfrac, phiAtGP);
      compute_linearization_of_sol_press_wrt_disp(fluidpress, porosity, totalnumdofpernode,
          numfluidphases, numvolfrac, phiAtGP, dphi_dus, dps_dus);
    }

    // Right Cauchy-Green tensor = F^T * F
    Core::LinAlg::Matrix<numdim_, numdim_> cauchygreen;
    cauchygreen.multiply_tn(defgrd, defgrd);

    // inverse Right Cauchy-Green tensor
    Core::LinAlg::Matrix<numdim_, numdim_> C_inv(false);
    C_inv.invert(cauchygreen);

    // compute some auxiliary matrixes for computation of linearization
    // dC^-1/dus
    Core::LinAlg::Matrix<numstr_, numdof_> dCinv_dus(true);
    for (int n = 0; n < numnod_; ++n)
    {
      for (int k = 0; k < numdim_; ++k)
      {
        const int gid = n * numdim_ + k;
        for (int i = 0; i < numdim_; ++i)
        {
          dCinv_dus(0, gid) += -2 * C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(0, k);
          dCinv_dus(1, gid) += -2 * C_inv(1, i) * N_XYZ(i, n) * defgrd_inv(1, k);
          dCinv_dus(2, gid) += -2 * C_inv(2, i) * N_XYZ(i, n) * defgrd_inv(2, k);
          /* ~~~ */
          dCinv_dus(3, gid) += -C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(1, k) -
                               defgrd_inv(0, k) * N_XYZ(i, n) * C_inv(1, i);
          dCinv_dus(4, gid) += -C_inv(1, i) * N_XYZ(i, n) * defgrd_inv(2, k) -
                               defgrd_inv(1, k) * N_XYZ(i, n) * C_inv(2, i);
          dCinv_dus(5, gid) += -C_inv(2, i) * N_XYZ(i, n) * defgrd_inv(0, k) -
                               defgrd_inv(2, k) * N_XYZ(i, n) * C_inv(0, i);
        }
      }
    }

    // **********************evaluate stiffness matrix and force vector+++++++++++++++++++++++++
    fill_matrix_and_vectors_pressure_based(
        gp, shapefct, N_XYZ, J, press, bop, C_inv, dJ_dus, dCinv_dus, dps_dus, stiffmatrix, force);
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::coupling_poroelast(
    std::vector<int>& lm,                            // location matrix
    Core::LinAlg::Matrix<numdim_, numnod_>& disp,    // current displacements
    Core::LinAlg::Matrix<numdim_, numnod_>& vel,     // current velocities
    Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,  // current fluid velocity
    Core::LinAlg::Matrix<numnod_, 1>& epreaf,        // current fluid pressure
    Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>*
        stiffmatrix,                                                    // element stiffness matrix
    Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>* reamatrix,  // element reactive matrix
    Core::LinAlg::Matrix<numdof_, 1>* force,  // element internal force vector
    Teuchos::ParameterList& params)           // algorithmic parameters e.g. time
{
  //=============================get parameters

  get_materials();

  //=======================================================================

  // update element geometry
  static Core::LinAlg::Matrix<numdim_, numnod_> xrefe;  // material coord. of element
  static Core::LinAlg::Matrix<numdim_, numnod_> xcurr;  // current  coord. of element


  for (int i = 0; i < numnod_; ++i)
  {
    const auto& x = nodes()[i]->x();
    for (int j = 0; j < numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  if (stiffmatrix != nullptr)
    gauss_point_loop_od(params, xrefe, xcurr, disp, vel, evelnp, epreaf, stiffmatrix);
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::coupling_poroelast_pressure_based(
    std::vector<int>& lm,                          // location matrix
    Core::LinAlg::Matrix<numdim_, numnod_>& disp,  // current displacements
    const std::vector<double>& ephi,            // current primary variable for poro-multiphase flow
    Core::LinAlg::SerialDenseMatrix& couplmat,  // element stiffness matrix
    Teuchos::ParameterList& params)             // algorithmic parameters e.g. time
{
  get_materials_pressure_based();

  //=======================================================================

  // update element geometry
  Core::LinAlg::Matrix<numdim_, numnod_> xrefe;  // material coord. of element
  Core::LinAlg::Matrix<numdim_, numnod_> xcurr;  // current  coord. of element


  for (int i = 0; i < numnod_; ++i)
  {
    const auto& x = nodes()[i]->x();
    for (int j = 0; j < numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/

  gauss_point_loop_od_pressure_based(params, xrefe, xcurr, disp, ephi, couplmat);
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::gauss_point_loop_od(
    Teuchos::ParameterList& params, const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodalvel,
    const Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,
    const Core::LinAlg::Matrix<numnod_, 1>& epreaf,
    Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>* stiffmatrix)
{
  static Core::LinAlg::Matrix<numdim_, numnod_>
      N_XYZ;  //  first derivatives at gausspoint w.r.t. X, Y,Z
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  static Core::LinAlg::Matrix<numdim_, numdim_> defgrd(
      true);  //  deformation gradient evaluated at gauss point
  static Core::LinAlg::Matrix<numnod_, 1> shapefct;  //  shape functions evaluated at gauss point
  static Core::LinAlg::Matrix<numdim_, numnod_> deriv(
      true);  //  first derivatives at gausspoint w.r.t. r,s,t

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    compute_shape_functions_and_derivatives(gp, shapefct, deriv, N_XYZ);
    // evaluate second derivatives of shape functions at integration point
    // ComputeSecondDerivativesOfShapeFunctions(gp,xrefe,deriv,deriv2,N_XYZ,N_XYZ2);

    // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
    compute_def_gradient(defgrd, N_XYZ, xcurr);

    // inverse deformation gradient F^-1
    static Core::LinAlg::Matrix<numdim_, numdim_> defgrd_inv(false);
    defgrd_inv.invert(defgrd);

    // jacobian determinant of transformation between spatial and material space "|dx/dX|"
    double J = 0.0;
    // volume change (used for porosity law). Same as J in nonlinear theory.
    double volchange = 0.0;

    // compute J and the volume change
    compute_jacobian_determinant_volume_change(J, volchange, defgrd, N_XYZ, nodaldisp);

    // non-linear B-operator
    static Core::LinAlg::Matrix<numstr_, numdof_> bop;
    compute_b_operator(bop, defgrd, N_XYZ);

    // -----------------Right Cauchy-Green tensor = F^T * F
    static Core::LinAlg::Matrix<numdim_, numdim_> cauchygreen;
    cauchygreen.multiply_tn(defgrd, defgrd);

    //------------------ inverse Right Cauchy-Green tensor
    static Core::LinAlg::Matrix<numdim_, numdim_> C_inv(false);
    C_inv.invert(cauchygreen);

    //---------------- get pressure at integration point
    double press = shapefct.dot(epreaf);

    //------------------ get material pressure gradient at integration point
    static Core::LinAlg::Matrix<numdim_, 1> Gradp;
    Gradp.multiply(N_XYZ, epreaf);

    //--------------------- get fluid velocity at integration point
    static Core::LinAlg::Matrix<numdim_, 1> fvelint;
    fvelint.multiply(evelnp, shapefct);

    // material fluid velocity gradient at integration point
    static Core::LinAlg::Matrix<numdim_, numdim_> fvelder;
    fvelder.multiply_nt(evelnp, N_XYZ);

    //----------------structure velocity at integration point
    static Core::LinAlg::Matrix<numdim_, 1> velint;
    velint.multiply(nodalvel, shapefct);

    //**************************************************+auxiliary variables for computing the
    // porosity and linearization
    double dphi_dp = 0.0;
    double porosity = 0.0;

    compute_porosity_and_linearization_od(
        params, press, volchange, gp, shapefct, nullptr, porosity, dphi_dp);

    // **********************evaluate stiffness matrix and force vector+++++++++++++++++++++++++

    fill_matrix_and_vectors_od(gp, shapefct, N_XYZ, J, porosity, dphi_dp, velint, fvelint,
        defgrd_inv, Gradp, bop, C_inv, stiffmatrix);

    if (fluid_mat_->type() == Mat::PAR::darcy_brinkman)
    {
      fill_matrix_and_vectors_brinkman_od(
          gp, shapefct, N_XYZ, J, porosity, dphi_dp, fvelder, defgrd_inv, bop, C_inv, stiffmatrix);
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::gauss_point_loop_od_pressure_based(
    Teuchos::ParameterList& params, const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp, const std::vector<double>& ephi,
    Core::LinAlg::SerialDenseMatrix& couplmat)
{
  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  Core::LinAlg::Matrix<numdim_, numnod_> N_XYZ;  //  first derivatives at gausspoint w.r.t. X, Y,Z
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  Core::LinAlg::Matrix<numdim_, numdim_> defgrd(
      true);                                  //  deformation gradient evaluated at gauss point
  Core::LinAlg::Matrix<numnod_, 1> shapefct;  //  shape functions evaluated at gauss point
  Core::LinAlg::Matrix<numdim_, numnod_> deriv(
      true);  //  first derivatives at gausspoint w.r.t. r,s,t

  // Initialize
  const int numfluidphases = fluidmulti_mat_->num_fluid_phases();
  const int totalnumdofpernode = fluidmulti_mat_->num_mat();
  const int numvolfrac = fluidmulti_mat_->num_vol_frac();
  const bool hasvolfracs = (totalnumdofpernode - numfluidphases);
  std::vector<double> phiAtGP(totalnumdofpernode);
  std::vector<double> solpressderiv(totalnumdofpernode);

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    compute_shape_functions_and_derivatives(gp, shapefct, deriv, N_XYZ);

    // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
    compute_def_gradient(defgrd, N_XYZ, xcurr);

    // jacobian determinant of transformation between spatial and material space "|dx/dX|"
    double J = 0.0;
    // volume change (used for porosity law). Same as J in nonlinear theory.
    double volchange = 0.0;

    // compute J, the volume change and the respctive linearizations w.r.t. structure displacement
    compute_jacobian_determinant_volume_change(J, volchange, defgrd, N_XYZ, nodaldisp);

    // non-linear B-operator (may so be called, meaning
    Core::LinAlg::Matrix<numstr_, numdof_> bop;
    compute_b_operator(bop, defgrd, N_XYZ);

    // Right Cauchy-Green tensor = F^T * F
    Core::LinAlg::Matrix<numdim_, numdim_> cauchygreen;
    cauchygreen.multiply_tn(defgrd, defgrd);

    // inverse Right Cauchy-Green tensor
    Core::LinAlg::Matrix<numdim_, numdim_> C_inv(false);
    C_inv.invert(cauchygreen);

    // compute derivative of solid pressure w.r.t primary variable phi at node
    compute_primary_variable_at_gp(ephi, totalnumdofpernode, shapefct, phiAtGP);
    compute_sol_pressure_deriv(phiAtGP, numfluidphases, solpressderiv);
    // in case of volume fractions --> recalculate
    if (hasvolfracs)
    {
      double dphi_dp = 0.0;
      double porosity = 0.0;

      double press = compute_sol_pressure_at_gp(totalnumdofpernode, numfluidphases, phiAtGP);

      compute_porosity_and_linearization_od(
          params, press, volchange, gp, shapefct, nullptr, porosity, dphi_dp);

      recalculate_sol_pressure_deriv(
          phiAtGP, totalnumdofpernode, numfluidphases, numvolfrac, press, porosity, solpressderiv);
    }

    // **********************evaluate stiffness matrix and force vector+++++++++++++++++++++++++
    fill_matrix_and_vectors_od_pressure_based(
        gp, shapefct, N_XYZ, J, bop, C_inv, solpressderiv, couplmat);
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::coupling_stress_poroelast(
    Core::LinAlg::Matrix<numdim_, numnod_>& disp,    // current displacements
    Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,  // current fluid velocities
    Core::LinAlg::Matrix<numnod_, 1>& epreaf,        // current fluid pressure
    Core::LinAlg::SerialDenseMatrix* elestress,      // stresses at GP
    Core::LinAlg::SerialDenseMatrix* elestrain,      // strains at GP
    Teuchos::ParameterList& params,                  // algorithmic parameters e.g. time
    const Inpar::Solid::StressType iostress          // stress output option
)
{
  // update element geometry
  Core::LinAlg::Matrix<numdim_, numnod_> xrefe;  // material coord. of element
  Core::LinAlg::Matrix<numdim_, numnod_> xcurr;  // current  coord. of element


  for (int i = 0; i < numnod_; ++i)
  {
    const auto& x = nodes()[i]->x();
    for (int j = 0; j < numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }

  // get structure material
  std::shared_ptr<Mat::StructPoro> structmat =
      std::dynamic_pointer_cast<Mat::StructPoro>(material());
  if (structmat->material_type() != Core::Materials::m_structporo)
    FOUR_C_THROW("invalid structure material for poroelasticity");

  Core::LinAlg::Matrix<numnod_, 1> shapefct;
  Core::LinAlg::Matrix<numdim_, numdim_> defgrd(true);
  Core::LinAlg::Matrix<numdim_, numnod_> N_XYZ;
  Core::LinAlg::Matrix<numdim_, numnod_> deriv;

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    compute_shape_functions_and_derivatives(gp, shapefct, deriv, N_XYZ);

    // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
    compute_def_gradient(defgrd, N_XYZ, xcurr);

    //----------------------------------------------------
    // pressure at integration point
    double press = shapefct.dot(epreaf);

    Core::LinAlg::Matrix<numstr_, 1> couplstress(true);

    structmat->coupl_stress(defgrd, press, couplstress);

    // return gp stresses
    switch (iostress)
    {
      case Inpar::Solid::stress_2pk:
      {
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");
        for (int i = 0; i < numstr_; ++i) (*elestress)(gp, i) = couplstress(i);
      }
      break;
      case Inpar::Solid::stress_cauchy:
      {
        if (elestress == nullptr) FOUR_C_THROW("stress data not available");

        // push forward of material stress to the spatial configuration
        Core::LinAlg::Matrix<numdim_, numdim_> cauchycouplstress;
        p_k2to_cauchy(couplstress, defgrd, cauchycouplstress);

        (*elestress)(gp, 0) = cauchycouplstress(0, 0);
        (*elestress)(gp, 1) = cauchycouplstress(1, 1);
        (*elestress)(gp, 2) = cauchycouplstress(2, 2);
        (*elestress)(gp, 3) = cauchycouplstress(0, 1);
        (*elestress)(gp, 4) = cauchycouplstress(1, 2);
        (*elestress)(gp, 5) = cauchycouplstress(0, 2);
      }
      break;
      case Inpar::Solid::stress_none:
        break;

      default:
        FOUR_C_THROW("requested stress type not available");
        break;
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::init_element()
{
  Core::LinAlg::Matrix<numdim_, numnod_> deriv;
  Core::LinAlg::Matrix<numnod_, numdim_> xrefe;
  for (int i = 0; i < numnod_; ++i)
  {
    xrefe(i, 0) = nodes()[i]->x()[0];
    xrefe(i, 1) = nodes()[i]->x()[1];
    xrefe(i, 2) = nodes()[i]->x()[2];
  }

  if (distype == Core::FE::CellType::nurbs27) isNurbs_ = true;

  invJ_.resize(numgpt_);
  detJ_.resize(numgpt_);
  xsi_.resize(numgpt_);

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    const double* gpcoord = intpoints_.point(gp);
    for (int idim = 0; idim < numdim_; idim++)
    {
      xsi_[gp](idim) = gpcoord[idim];
    }

    if (not isNurbs_)
    {
      Core::FE::shape_function_deriv1<distype>(xsi_[gp], deriv);

      invJ_[gp].multiply(deriv, xrefe);
      detJ_[gp] = invJ_[gp].invert();
      if (detJ_[gp] <= 0.0) FOUR_C_THROW("Element Jacobian mapping %10.5e <= 0.0", detJ_[gp]);
    }
  }

  init_ = true;
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::p_k2to_cauchy(
    Core::LinAlg::Matrix<numstr_, 1>& stress, Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
    Core::LinAlg::Matrix<numdim_, numdim_>& cauchystress)
{
  // calculate the Jacobi-determinant
  const double detF = (defgrd).determinant();

  // sigma = 1/J . F . S . F^T
  Core::LinAlg::Matrix<numdim_, numdim_> pkstress;
  pkstress(0, 0) = (stress)(0);
  pkstress(0, 1) = (stress)(3);
  pkstress(0, 2) = (stress)(5);
  pkstress(1, 0) = pkstress(0, 1);
  pkstress(1, 1) = (stress)(1);
  pkstress(1, 2) = (stress)(4);
  pkstress(2, 0) = pkstress(0, 2);
  pkstress(2, 1) = pkstress(1, 2);
  pkstress(2, 2) = (stress)(2);

  Core::LinAlg::Matrix<numdim_, numdim_> temp;
  temp.multiply((1.0 / detF), (defgrd), pkstress);
  (cauchystress).multiply_nt(temp, (defgrd));
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::compute_porosity_and_linearization(
    Teuchos::ParameterList& params, const double& press, const double& J, const int& gp,
    const Core::LinAlg::Matrix<numnod_, 1>& shapfct,
    const Core::LinAlg::Matrix<numnod_, 1>* myporosity,
    const Core::LinAlg::Matrix<1, numdof_>& dJ_dus, double& porosity,
    Core::LinAlg::Matrix<1, numdof_>& dphi_dus)
{
  double dphi_dJ = 0.0;

  struct_mat_->compute_porosity(params, press, J, gp, porosity,
      nullptr,  // dphi_dp not needed
      &dphi_dJ,
      nullptr,  // dphi_dJdp not needed
      nullptr,  // dphi_dJJ not needed
      nullptr   // dphi_dpp not needed
  );

  dphi_dus.update(dphi_dJ, dJ_dus);
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::compute_porosity_and_linearization_od(
    Teuchos::ParameterList& params, const double& press, const double& J, const int& gp,
    const Core::LinAlg::Matrix<numnod_, 1>& shapfct,
    const Core::LinAlg::Matrix<numnod_, 1>* myporosity, double& porosity, double& dphi_dp)
{
  struct_mat_->compute_porosity(params, press, J, gp, porosity, &dphi_dp,
      nullptr,  // dphi_dJ not needed
      nullptr,  // dphi_dJdp not needed
      nullptr,  // dphi_dJJ not needed
      nullptr   // dphi_dpp not needed
  );
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::extract_values_from_global_vector(
    const Core::FE::Discretization& discretization, const int& dofset, const std::vector<int>& lm,
    Core::LinAlg::Matrix<numdim_, numnod_>* matrixtofill,
    Core::LinAlg::Matrix<numnod_, 1>* vectortofill, const std::string& state)
{
  // get state of the global vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
      discretization.get_state(dofset, state);
  if (matrix_state == nullptr) FOUR_C_THROW("Cannot get state vector %s", state.c_str());

  // ask for the number of dofs of dofset
  const int numdofpernode = discretization.num_dof(dofset, nodes()[0]);

  // extract local values of the global vectors
  std::vector<double> mymatrix = Core::FE::extract_values(*matrix_state, lm);

  if (numdofpernode == numdim_ + 1)
  {
    for (int inode = 0; inode < numnod_; ++inode)  // number of nodes
    {
      // fill a vector field via a pointer
      if (matrixtofill != nullptr)
      {
        for (int idim = 0; idim < numdim_; ++idim)  // number of dimensions
        {
          (*matrixtofill)(idim, inode) = mymatrix[idim + (inode * numdofpernode)];
        }
      }
      // fill a scalar field via a pointer
      if (vectortofill != nullptr)
        (*vectortofill)(inode, 0) = mymatrix[numdim_ + (inode * numdofpernode)];
    }
  }
  else if (numdofpernode == numdim_)
  {
    for (int inode = 0; inode < numnod_; ++inode)  // number of nodes
    {
      // fill a vector field via a pointer
      if (matrixtofill != nullptr)
      {
        for (int idim = 0; idim < numdim_; ++idim)  // number of dimensions
        {
          (*matrixtofill)(idim, inode) = mymatrix[idim + (inode * numdofpernode)];
        }
      }
    }
  }
  else if (numdofpernode == 1)
  {
    for (int inode = 0; inode < numnod_; ++inode)  // number of nodes
    {
      if (vectortofill != nullptr) (*vectortofill)(inode, 0) = mymatrix[inode * numdofpernode];
    }
  }
  else
  {
    for (int inode = 0; inode < numnod_; ++inode)  // number of nodes
    {
      if (vectortofill != nullptr) (*vectortofill)(inode, 0) = mymatrix[inode * numdofpernode];
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::compute_sol_pressure_deriv(
    const std::vector<double>& phiAtGP, const int numfluidphases,
    std::vector<double>& solidpressderiv)
{
  // zero out everything
  std::fill(solidpressderiv.begin(), solidpressderiv.end(), 0.0);

  // initialize auxiliary variables
  std::vector<double> genpress(numfluidphases);
  std::vector<double> press(numfluidphases);
  std::vector<double> sat(numfluidphases);
  Core::LinAlg::SerialDenseMatrix helpderiv(numfluidphases, numfluidphases, true);
  Core::LinAlg::SerialDenseMatrix satderiv(numfluidphases, numfluidphases, true);
  Core::LinAlg::SerialDenseMatrix pressderiv(numfluidphases, numfluidphases, true);
  std::vector<double> fluidphi(phiAtGP.data(), phiAtGP.data() + numfluidphases);

  // evaluate the pressures
  fluidmulti_mat_->evaluate_gen_pressure(genpress, fluidphi);

  // transform generalized pressures to true pressure values
  fluidmulti_mat_->transform_gen_pres_to_true_pres(genpress, press);

  // explicit evaluation of saturation
  fluidmulti_mat_->evaluate_saturation(sat, fluidphi, press);

  // calculate the derivative of the pressure (actually first its inverse)
  fluidmulti_mat_->evaluate_deriv_of_dof_wrt_pressure(pressderiv, fluidphi);

  // now invert the derivatives of the dofs w.r.t. pressure to get the derivatives
  // of the pressure w.r.t. the dofs
  {
    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverse;
    inverse.setMatrix(Teuchos::rcpFromRef(pressderiv));
    int err = inverse.invert();
    if (err != 0)
      FOUR_C_THROW("Inversion of matrix for pressure derivative failed with error code %d.", err);
  }

  // calculate derivatives of saturation w.r.t. pressure
  fluidmulti_mat_->evaluate_deriv_of_saturation_wrt_pressure(helpderiv, press);

  // chain rule: the derivative of saturation w.r.t. dof =
  // (derivative of saturation w.r.t. pressure) * (derivative of pressure w.r.t. dof)
  Core::LinAlg::multiply(satderiv, helpderiv, pressderiv);

  // compute derivative of solid pressure w.r.t. dofs with product rule
  // standard derivative: no volume fractions present
  for (int iphase = 0; iphase < numfluidphases; iphase++)
  {
    for (int jphase = 0; jphase < numfluidphases; jphase++)
      solidpressderiv[iphase] +=
          pressderiv(jphase, iphase) * sat[jphase] + satderiv(jphase, iphase) * press[jphase];
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::compute_linearization_of_sol_press_wrt_disp(
    const double fluidpress, const double porosity, const int totalnumdofpernode,
    const int numfluidphases, const int numvolfrac, const std::vector<double>& phiAtGP,
    const Core::LinAlg::Matrix<1, numdof_>& dphi_dus, Core::LinAlg::Matrix<1, numdof_>& dps_dus)
{
  // get volume fraction primary variables
  std::vector<double> volfracphi(
      phiAtGP.data() + numfluidphases, phiAtGP.data() + numfluidphases + numvolfrac);
  double sumaddvolfrac = 0.0;
  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++) sumaddvolfrac += volfracphi[ivolfrac];

  // get volume fraction pressure at [numfluidphases+numvolfrac...totalnumdofpernode-1]
  std::vector<double> volfracpressure(
      phiAtGP.data() + numfluidphases + numvolfrac, phiAtGP.data() + totalnumdofpernode);

  // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
  //       + 1.0 / porosity sum_i=1^numvolfrac (volfrac_i*pressure_i)
  // d (p_s) / d porosity = + sumaddvolfrac/porosity/porosity * fluidpress
  double dps_dphi = sumaddvolfrac / (porosity * porosity) * fluidpress;

  // ... + 1.0 / porosity / porosity sum_i=1^numvolfrac (volfrac_i*pressure_i)
  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
    dps_dphi -= volfracphi[ivolfrac] * volfracpressure[ivolfrac] / (porosity * porosity);

  // d (p_s) / d u_s = d (p_s) / d porosity * d porosity / d u_s
  dps_dus.update(dps_dphi, dphi_dus);
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::recalculate_sol_pressure_deriv(
    const std::vector<double>& phiAtGP, const int totalnumdofpernode, const int numfluidphases,
    const int numvolfrac, const double press, const double porosity,
    std::vector<double>& solidpressderiv)
{
  // get volume fraction primary variables
  std::vector<double> volfracphi(
      phiAtGP.data() + numfluidphases, phiAtGP.data() + numfluidphases + numvolfrac);
  double sumaddvolfrac = 0.0;
  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++) sumaddvolfrac += volfracphi[ivolfrac];

  // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
  //      + 1.0 / porosity sum_i=1^numvolfrac (volfrac_i*pressure_i)
  const double scale = (porosity - sumaddvolfrac) / porosity;

  // scale original fluid press deriv with (porosity - sumaddvolfrac)/porosity
  for (int iphase = 0; iphase < numfluidphases; iphase++) solidpressderiv[iphase] *= scale;

  // get volfrac pressures at [numfluidphases+numvolfrac...totalnumdofpernode-1]
  std::vector<double> volfracpressure(
      phiAtGP.data() + numfluidphases + numvolfrac, phiAtGP.data() + totalnumdofpernode);

  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
  {
    // d p_s / d volfrac = - fluidpress/porosity + volfracpressure/porosity
    solidpressderiv[ivolfrac + numfluidphases] =
        -1.0 / porosity * press + 1.0 / porosity * volfracpressure[ivolfrac];
    // d p_s / d volfracpress = + volfracphi/porosity
    solidpressderiv[ivolfrac + numfluidphases + numvolfrac] = volfracphi[ivolfrac] / porosity;
  }
}

template <class So3Ele, Core::FE::CellType distype>
double Discret::Elements::So3Poro<So3Ele, distype>::compute_sol_pressure_at_gp(
    const int totalnumdofpernode, const int numfluidphases, const std::vector<double>& phiAtGP)
{
  // initialize auxiliary variables
  std::vector<double> genpress(numfluidphases, 0.0);
  std::vector<double> sat(numfluidphases, 0.0);
  std::vector<double> press(numfluidphases, 0.0);
  std::vector<double> fluidphi(phiAtGP.data(), phiAtGP.data() + numfluidphases);

  // evaluate the pressures
  fluidmulti_mat_->evaluate_gen_pressure(genpress, fluidphi);

  //! transform generalized pressures to true pressure values
  fluidmulti_mat_->transform_gen_pres_to_true_pres(genpress, press);

  // explicit evaluation of saturation
  fluidmulti_mat_->evaluate_saturation(sat, fluidphi, press);

  // solid pressure = sum (S_i*p_i)
  const double solidpressure = std::inner_product(sat.begin(), sat.end(), press.begin(), 0.0);

  return solidpressure;
}

template <class So3Ele, Core::FE::CellType distype>
double Discret::Elements::So3Poro<So3Ele, distype>::recalculate_sol_pressure_at_gp(double press,
    const double porosity, const int totalnumdofpernode, const int numfluidphases,
    const int numvolfrac, const std::vector<double>& phiAtGP)
{
  // get volume fraction primary variables at [numfluidphases-1...numfluidphase-1+numvolfrac]
  std::vector<double> volfracphi(
      phiAtGP.data() + numfluidphases, phiAtGP.data() + numfluidphases + numvolfrac);
  double sumaddvolfrac = 0.0;
  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++) sumaddvolfrac += volfracphi[ivolfrac];

  // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
  //      + 1.0 / porosity * sum_i=1^numvolfrac (volfrac_i*pressure_i)
  // first part
  press *= (porosity - sumaddvolfrac) / porosity;

  // get volfrac pressures at [numfluidphases+numvolfrac...totalnumdofpernode-1]
  std::vector<double> volfracpressure(
      phiAtGP.data() + numfluidphases + numvolfrac, phiAtGP.data() + totalnumdofpernode);

  // second part
  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++)
    press += volfracphi[ivolfrac] / porosity * volfracpressure[ivolfrac];

  // note: in recalculate_solid_pressure in porofluid_phasemanager calculation is performed a bit
  //       differently since we already pass porosity = porosity - sumaddvolfrac, but result is
  //       equivalent

  return press;
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::compute_primary_variable_at_gp(
    const std::vector<double>& ephi, const int totalnumdofpernode,
    const Core::LinAlg::Matrix<numnod_, 1>& shapefct, std::vector<double>& phiAtGP)
{
  // zero out everything
  std::fill(phiAtGP.begin(), phiAtGP.end(), 0.0);
  // compute phi at GP = phi * shapefunction
  for (int i = 0; i < numnod_; i++)
  {
    for (int j = 0; j < totalnumdofpernode; j++)
    {
      phiAtGP[j] += shapefct(i) * ephi[i * totalnumdofpernode + j];
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::get_materials()
{
  // get structure material
  if (struct_mat_ == nullptr)
  {
    struct_mat_ = std::dynamic_pointer_cast<Mat::StructPoro>(material());
    if (struct_mat_->material_type() != Core::Materials::m_structporo and
        struct_mat_->material_type() != Core::Materials::m_structpororeaction and
        struct_mat_->material_type() != Core::Materials::m_structpororeactionECM)
      FOUR_C_THROW("invalid structure material for poroelasticity");
  }

  // get fluid material
  if (fluid_mat_ == nullptr)
  {
    // access second material in structure element
    if (So3Ele::num_material() > 1)
    {
      fluid_mat_ = std::dynamic_pointer_cast<Mat::FluidPoro>(So3Ele::material(1));
      if (fluid_mat_->material_type() != Core::Materials::m_fluidporo)
        FOUR_C_THROW("invalid fluid material for poroelasticity");
    }
    else
      FOUR_C_THROW("no second material defined for element %i", id());
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::get_materials_pressure_based()
{
  // get structure material
  if (struct_mat_ == nullptr)
  {
    struct_mat_ = std::dynamic_pointer_cast<Mat::StructPoro>(material());
    if (struct_mat_ == nullptr) FOUR_C_THROW("cast to poro material failed");

    if (struct_mat_->material_type() != Core::Materials::m_structporo and
        struct_mat_->material_type() != Core::Materials::m_structpororeaction and
        struct_mat_->material_type() != Core::Materials::m_structpororeactionECM)
      FOUR_C_THROW("invalid structure material for poroelasticity");
  }

  // Get Fluid-multiphase-Material
  if (fluidmulti_mat_ == nullptr)
  {
    // access second material in structure element
    if (So3Ele::num_material() > 1)
    {
      fluidmulti_mat_ = std::dynamic_pointer_cast<Mat::FluidPoroMultiPhase>(So3Ele::material(1));
      if (fluidmulti_mat_ == nullptr) FOUR_C_THROW("cast to multiphase fluid poro material failed");
      if (fluidmulti_mat_->material_type() != Core::Materials::m_fluidporo_multiphase and
          fluidmulti_mat_->material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
        FOUR_C_THROW("invalid fluid material for poro-multiphase-elasticity");
      if (fluidmulti_mat_->num_fluid_phases() == 0)
      {
        FOUR_C_THROW(
            "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE = 0 currently not supported since this requires "
            "an adaption of the definition of the solid pressure");
      }
    }
    else
      FOUR_C_THROW("no second material defined for element %i", id());
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::compute_porosity(Teuchos::ParameterList& params,
    double press, double J, int gp, double& porosity, double* dphi_dp, double* dphi_dJ,
    double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, bool save)
{
  struct_mat_->compute_porosity(
      params, press, J, gp, porosity, dphi_dp, dphi_dJ, dphi_dJdp, dphi_dJJ, dphi_dpp, save);
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::compute_surf_porosity(
    Teuchos::ParameterList& params, double press, double J, int surfnum, int gp, double& porosity,
    double* dphi_dp, double* dphi_dJ, double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp,
    bool save)
{
  struct_mat_->compute_surf_porosity(params, press, J, surfnum, gp, porosity, dphi_dp, dphi_dJ,
      dphi_dJdp, dphi_dJJ, dphi_dpp, save);
}

template <class So3Ele, Core::FE::CellType distype>
double Discret::Elements::So3Poro<So3Ele, distype>::ref_porosity_time_deriv()
{
  return struct_mat_->ref_porosity_time_deriv();
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::compute_shape_functions_and_derivatives(
    const int& gp, Core::LinAlg::Matrix<numnod_, 1>& shapefct,
    Core::LinAlg::Matrix<numdim_, numnod_>& deriv, Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ)
{
  if (!isNurbs_)
  {
    Core::FE::shape_function<distype>(xsi_[gp], shapefct);
    Core::FE::shape_function_deriv1<distype>(xsi_[gp], deriv);
  }
  else
  {
    Core::FE::Nurbs::nurbs_get_funct_deriv(shapefct, deriv, xsi_[gp], myknots_, weights_, distype);

    Core::LinAlg::Matrix<numnod_, numdim_> xrefe;
    for (int i = 0; i < numnod_; ++i)
    {
      xrefe(i, 0) = nodes()[i]->x()[0];
      xrefe(i, 1) = nodes()[i]->x()[1];
      xrefe(i, 2) = nodes()[i]->x()[2];
    }

    invJ_[gp].multiply(deriv, xrefe);
    detJ_[gp] = invJ_[gp].invert();
    if (detJ_[gp] <= 0.0) FOUR_C_THROW("Element Jacobian mapping %10.5e <= 0.0", detJ_[gp]);
  }

  /* get the inverse of the Jacobian matrix which looks like:
   **            [ X_,r  Y_,r  Z_,r ]^-1
   **     J^-1 = [ X_,s  Y_,s  Z_,s ]
   **            [ X_,t  Y_,t  Z_,t ]
   */

  // compute derivatives N_XYZ at gp w.r.t. material coordinates
  // by N_XYZ = J^-1 * N_rst
  N_XYZ.multiply(invJ_[gp], deriv);  // (6.21)
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::compute_jacobian_determinant_volume_change(
    double& J, double& volchange, const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp)
{
  // compute J
  J = defgrd.determinant();

  if (So3Ele::kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)  // total lagrange (nonlinear)
  {
    // for nonlinear kinematics the Jacobian of the deformation gradient is the volume change
    volchange = J;
  }
  else if (So3Ele::kintype_ == Inpar::Solid::KinemType::linear)  // linear kinematics
  {
    // for linear kinematics the volume change is the trace of the linearized strains

    // gradient of displacements
    static Core::LinAlg::Matrix<numdim_, numdim_> dispgrad;
    dispgrad.clear();
    // gradient of displacements
    dispgrad.multiply_nt(nodaldisp, N_XYZ);

    volchange = 1.0;
    // volchange = 1 + trace of the linearized strains (= trace of displacement gradient)
    for (int i = 0; i < numdim_; ++i) volchange += dispgrad(i, i);
  }
  else
    FOUR_C_THROW("invalid kinematic type!");
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele,
    distype>::compute_jacobian_determinant_volume_change_and_linearizations(double& J,
    double& volchange, Core::LinAlg::Matrix<1, numdof_>& dJ_dus,
    Core::LinAlg::Matrix<1, numdof_>& dvolchange_dus,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp)
{
  // compute J
  J = defgrd.determinant();
  // compute linearization of J
  compute_linearization_of_jacobian(dJ_dus, J, N_XYZ, defgrd_inv);

  if (So3Ele::kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)  // total lagrange (nonlinear)
  {
    // for nonlinear kinematics the Jacobian of the deformation gradient is the volume change
    volchange = J;
    dvolchange_dus = dJ_dus;
  }
  else if (So3Ele::kintype_ == Inpar::Solid::KinemType::linear)  // linear kinematics
  {
    // for linear kinematics the volume change is the trace of the linearized strains

    // gradient of displacements
    static Core::LinAlg::Matrix<numdim_, numdim_> dispgrad;
    dispgrad.clear();
    // gradient of displacements
    dispgrad.multiply_nt(nodaldisp, N_XYZ);

    volchange = 1.0;
    // volchange = 1 + trace of the linearized strains (= trace of displacement gradient)
    for (int i = 0; i < numdim_; ++i) volchange += dispgrad(i, i);

    for (int i = 0; i < numdim_; ++i)
      for (int j = 0; j < numnod_; ++j) dvolchange_dus(numdim_ * j + i) = N_XYZ(i, j);
  }
  else
    FOUR_C_THROW("invalid kinematic type!");
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::compute_auxiliary_values(
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
    const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
    const Core::LinAlg::Matrix<numdim_, 1>& Gradp,
    Core::LinAlg::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
    Core::LinAlg::Matrix<numdim_, 1>& Finvgradp,
    Core::LinAlg::Matrix<numdim_, numdof_>& dFinvdus_gradp,
    Core::LinAlg::Matrix<numstr_, numdof_>& dCinv_dus)
{
  // F^-T * Grad p
  Finvgradp.multiply_tn(defgrd_inv, Gradp);

  if (So3Ele::kintype_ != Inpar::Solid::KinemType::linear)
  {
    // dF^-T/dus
    dFinvTdus.clear();
    for (int i = 0; i < numdim_; i++)
    {
      for (int n = 0; n < numnod_; n++)
      {
        for (int j = 0; j < numdim_; j++)
        {
          const int gid = numdim_ * n + j;
          for (int k = 0; k < numdim_; k++)
            for (int l = 0; l < numdim_; l++)
              dFinvTdus(i * numdim_ + l, gid) += -defgrd_inv(l, j) * N_XYZ(k, n) * defgrd_inv(k, i);
        }
      }
    }

    // dF^-T/dus * Grad p
    dFinvdus_gradp.clear();
    for (int i = 0; i < numdim_; i++)
    {
      for (int n = 0; n < numnod_; n++)
      {
        for (int j = 0; j < numdim_; j++)
        {
          const int gid = numdim_ * n + j;
          for (int l = 0; l < numdim_; l++)
            dFinvdus_gradp(i, gid) += dFinvTdus(i * numdim_ + l, gid) * Gradp(l);
        }
      }
    }
  }

  dCinv_dus.clear();
  for (int n = 0; n < numnod_; ++n)
  {
    for (int k = 0; k < numdim_; ++k)
    {
      const int gid = n * numdim_ + k;
      for (int i = 0; i < numdim_; ++i)
      {
        dCinv_dus(0, gid) += -2 * C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(0, k);
        dCinv_dus(1, gid) += -2 * C_inv(1, i) * N_XYZ(i, n) * defgrd_inv(1, k);
        dCinv_dus(2, gid) += -2 * C_inv(2, i) * N_XYZ(i, n) * defgrd_inv(2, k);
        /* ~~~ */
        dCinv_dus(3, gid) += -C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(1, k) -
                             defgrd_inv(0, k) * N_XYZ(i, n) * C_inv(1, i);
        dCinv_dus(4, gid) += -C_inv(1, i) * N_XYZ(i, n) * defgrd_inv(2, k) -
                             defgrd_inv(1, k) * N_XYZ(i, n) * C_inv(2, i);
        dCinv_dus(5, gid) += -C_inv(2, i) * N_XYZ(i, n) * defgrd_inv(0, k) -
                             defgrd_inv(2, k) * N_XYZ(i, n) * C_inv(0, i);
      }
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
inline void Discret::Elements::So3Poro<So3Ele, distype>::compute_b_operator(
    Core::LinAlg::Matrix<numstr_, numdof_>& bop,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ)
{
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
  for (int i = 0; i < numnod_; ++i)
  {
    bop(0, noddof_ * i + 0) = defgrd(0, 0) * N_XYZ(0, i);
    bop(0, noddof_ * i + 1) = defgrd(1, 0) * N_XYZ(0, i);
    bop(0, noddof_ * i + 2) = defgrd(2, 0) * N_XYZ(0, i);
    bop(1, noddof_ * i + 0) = defgrd(0, 1) * N_XYZ(1, i);
    bop(1, noddof_ * i + 1) = defgrd(1, 1) * N_XYZ(1, i);
    bop(1, noddof_ * i + 2) = defgrd(2, 1) * N_XYZ(1, i);
    bop(2, noddof_ * i + 0) = defgrd(0, 2) * N_XYZ(2, i);
    bop(2, noddof_ * i + 1) = defgrd(1, 2) * N_XYZ(2, i);
    bop(2, noddof_ * i + 2) = defgrd(2, 2) * N_XYZ(2, i);
    /* ~~~ */
    bop(3, noddof_ * i + 0) = defgrd(0, 0) * N_XYZ(1, i) + defgrd(0, 1) * N_XYZ(0, i);
    bop(3, noddof_ * i + 1) = defgrd(1, 0) * N_XYZ(1, i) + defgrd(1, 1) * N_XYZ(0, i);
    bop(3, noddof_ * i + 2) = defgrd(2, 0) * N_XYZ(1, i) + defgrd(2, 1) * N_XYZ(0, i);
    bop(4, noddof_ * i + 0) = defgrd(0, 1) * N_XYZ(2, i) + defgrd(0, 2) * N_XYZ(1, i);
    bop(4, noddof_ * i + 1) = defgrd(1, 1) * N_XYZ(2, i) + defgrd(1, 2) * N_XYZ(1, i);
    bop(4, noddof_ * i + 2) = defgrd(2, 1) * N_XYZ(2, i) + defgrd(2, 2) * N_XYZ(1, i);
    bop(5, noddof_ * i + 0) = defgrd(0, 2) * N_XYZ(0, i) + defgrd(0, 0) * N_XYZ(2, i);
    bop(5, noddof_ * i + 1) = defgrd(1, 2) * N_XYZ(0, i) + defgrd(1, 0) * N_XYZ(2, i);
    bop(5, noddof_ * i + 2) = defgrd(2, 2) * N_XYZ(0, i) + defgrd(2, 0) * N_XYZ(2, i);
  }
}

template <class So3Ele, Core::FE::CellType distype>
inline void Discret::Elements::So3Poro<So3Ele, distype>::compute_linearization_of_jacobian(
    Core::LinAlg::Matrix<1, numdof_>& dJ_dus, const double& J,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv)
{
  if (So3Ele::kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)  // total lagrange (nonlinear)
  {
    //------------------------------------ build F^-1 as vector 9x1
    Core::LinAlg::Matrix<numdim_ * numdim_, 1> defgrd_inv_vec;
    defgrd_inv_vec(0) = defgrd_inv(0, 0);
    defgrd_inv_vec(1) = defgrd_inv(0, 1);
    defgrd_inv_vec(2) = defgrd_inv(0, 2);
    defgrd_inv_vec(3) = defgrd_inv(1, 0);
    defgrd_inv_vec(4) = defgrd_inv(1, 1);
    defgrd_inv_vec(5) = defgrd_inv(1, 2);
    defgrd_inv_vec(6) = defgrd_inv(2, 0);
    defgrd_inv_vec(7) = defgrd_inv(2, 1);
    defgrd_inv_vec(8) = defgrd_inv(2, 2);

    //--------------------------- build N_X operator (wrt material config)
    Core::LinAlg::Matrix<9, numdof_> N_X(true);  // set to zero
    for (int i = 0; i < numnod_; ++i)
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

    //------linearization of jacobi determinant detF=J w.r.t. structure displacement   dJ/d(us) =
    // dJ/dF : dF/dus = J * F^-T * N,X
    dJ_dus.multiply_tn(J, defgrd_inv_vec, N_X);
  }
  else if (So3Ele::kintype_ == Inpar::Solid::KinemType::linear)  // linear kinematics
  {
    // J=1 -> no linearization
    dJ_dus.clear();
  }
  else
    FOUR_C_THROW("invalid kinematic type!");
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::fill_matrix_and_vectors(const int& gp,
    const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& press,
    const double& porosity, const Core::LinAlg::Matrix<numdim_, 1>& velint,
    const Core::LinAlg::Matrix<numdim_, 1>& fvelint,
    const Core::LinAlg::Matrix<numdim_, numdim_>& fvelder,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
    const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
    const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
    const Core::LinAlg::Matrix<numdim_, 1>& Finvgradp,
    const Core::LinAlg::Matrix<1, numdof_>& dphi_dus,
    const Core::LinAlg::Matrix<1, numdof_>& dJ_dus,
    const Core::LinAlg::Matrix<numstr_, numdof_>& dCinv_dus,
    const Core::LinAlg::Matrix<numdim_, numdof_>& dFinvdus_gradp,
    const Core::LinAlg::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
    Core::LinAlg::Matrix<numdof_, numdof_>& erea_v,
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix, Core::LinAlg::Matrix<numdof_, 1>* force,
    Core::LinAlg::Matrix<numstr_, 1>& fstress)
{
  const double detJ_w = detJ_[gp] * intpoints_.weight(gp);

  {
    static Core::LinAlg::Matrix<numdim_, numdim_> matreatensor(true);
    static Core::LinAlg::Matrix<numdim_, numdim_> reatensor(true);
    static Core::LinAlg::Matrix<numdim_, numdim_> linreac_dphi(true);
    static Core::LinAlg::Matrix<numdim_, numdim_> linreac_dJ(true);
    static Core::LinAlg::Matrix<numdim_, 1> reafvel(true);
    static Core::LinAlg::Matrix<numdim_, 1> reavel(true);
    {
      static Core::LinAlg::Matrix<numdim_, numdim_> temp(true);
      std::vector<double> anisotropic_permeability_coeffs =
          compute_anisotropic_permeability_coeffs_at_gp(shapefct);
      fluid_mat_->compute_reaction_tensor(matreatensor, J, porosity,
          anisotropic_permeability_directions_, anisotropic_permeability_coeffs);
      fluid_mat_->compute_lin_mat_reaction_tensor(linreac_dphi, linreac_dJ, J, porosity);
      temp.multiply(1.0, matreatensor, defgrd_inv);
      reatensor.multiply_tn(defgrd_inv, temp);
      reavel.multiply(reatensor, velint);
      reafvel.multiply(reatensor, fvelint);
    }

    for (int idim = 0; idim < numdim_; idim++)
    {
      const double reafvel_idim = reafvel(idim);
      const double reac_vel_idim = reavel(idim);
      const double Finvgradp_idim = Finvgradp(idim);

      for (int inode = 0; inode < numnod_; inode++)
      {
        const double fac = detJ_w * shapefct(inode);
        const double v = fac * porosity * porosity * J * J;
        const int fk = numdim_ * inode;

        /*-------structure- fluid velocity coupling:  RHS
         "darcy-terms"
         - reacoeff * J^2 *  phi^2 *  v^f
         */
        (*force)(fk + idim) += -v * reafvel_idim;

        /* "reactive darcy-terms"
         reacoeff * J^2 *  phi^2 *  v^s
         */
        (*force)(fk + idim) += v * reac_vel_idim;

        /*-------structure- fluid pressure coupling: RHS
         *                        "pressure gradient terms"
         - J *  F^-T * Grad(p) * phi
         */
        (*force)(fk + idim) += fac * J * Finvgradp_idim * (-porosity);
      }
    }

    for (int idim = 0; idim < numdim_; idim++)
    {
      for (int jdim = 0; jdim < numdim_; jdim++)
      {
        const double reatensor_i_j = reatensor(idim, jdim);

        for (int inode = 0; inode < numnod_; inode++)
        {
          const int fk = numdim_ * inode;
          const double v = detJ_w * shapefct(inode) * porosity * porosity * J * J;

          for (int jnode = 0; jnode < numnod_; jnode++)
          {
            const int fi = numdim_ * jnode;

            /* additional "reactive darcy-term"
             detJ * w(gp) * ( J^2 * reacoeff * phi^2  ) * D(v_s)
             */
            erea_v(fk + idim, fi + jdim) += v * reatensor_i_j * shapefct(jnode);
          }
        }
      }
    }

    for (int idim = 0; idim < numdim_; idim++)
    {
      const double Finvgradp_j = Finvgradp(idim);

      for (int jdim = 0; jdim < numdim_; jdim++)
      {
        for (int jnode = 0; jnode < numnod_; jnode++)
        {
          const int fi = numdim_ * jnode;

          const double val = detJ_w * (-porosity * dJ_dus(fi + jdim) * Finvgradp_j -
                                          porosity * J * dFinvdus_gradp(idim, fi + jdim) -
                                          dphi_dus(fi + jdim) * J * Finvgradp_j);

          for (int inode = 0; inode < numnod_; inode++)
          {
            /* additional "pressure gradient term"
             -  detJ * w(gp) * phi *  ( dJ/d(us) * F^-T * Grad(p) - J * d(F^-T)/d(us) *Grad(p) ) *
             D(us)
             - detJ * w(gp) * d(phi)/d(us) * J * F^-T * Grad(p) * D(us)
             */
            (*stiffmatrix)(numdim_* inode + idim, fi + jdim) += shapefct(inode) * val;
          }
        }
      }
    }

    for (int idim = 0; idim < numdim_; idim++)
    {
      const double reac_vel_j = reavel(idim);
      const double reafvel_j = reafvel(idim);

      for (int jdim = 0; jdim < numdim_; jdim++)
      {
        for (int jnode = 0; jnode < numnod_; jnode++)
        {
          const int fi = numdim_ * jnode;
          const double val = detJ_w * J * porosity * 2 * (reac_vel_j - reafvel_j) *
                             (porosity * dJ_dus(fi + jdim) + J * dphi_dus(fi + jdim));

          for (int inode = 0; inode < numnod_; inode++)
          {
            /* additional "reactive darcy-term"
               detJ * w(gp) * 2 * ( dJ/d(us) * vs * reacoeff * phi^2 + J * reacoeff * phi *
             d(phi)/d(us) * vs ) * D(us)
             - detJ * w(gp) *  2 * ( J * dJ/d(us) * v^f * reacoeff * phi^2 + J * reacoeff * phi *
             d(phi)/d(us) * v^f ) * D(us)
             */
            (*stiffmatrix)(numdim_* inode + idim, fi + jdim) += shapefct(inode) * val;
          }
        }
      }
    }

    // check if derivatives of reaction tensor are zero --> significant speed up
    if (fluid_mat_->permeability_function() == Mat::PAR::constant)
    {
      const double fac = detJ_w * porosity * porosity * J * J;
      for (int idim = 0; idim < numdim_; idim++)
      {
        for (int jdim = 0; jdim < numdim_; jdim++)
        {
          for (int jnode = 0; jnode < numnod_; jnode++)
          {
            const int fi = numdim_ * jnode;

            for (int inode = 0; inode < numnod_; inode++)
            {
              double val = 0.0;
              for (int p = 0; p < numdim_; ++p)
              {
                const double velint_p = velint(p);
                const double fvelint_p = fvelint(p);
                for (int n = 0; n < numdim_; ++n)
                {
                  const double defgrd_inv_n_p = defgrd_inv(n, p);
                  const double dFinvTdus_n_p = dFinvTdus(p * numdim_ + n, fi + jdim);
                  for (int m = 0; m < numdim_; ++m)
                  {
                    val += fac * (velint_p - fvelint_p) *
                           (dFinvTdus(idim * numdim_ + m, fi + jdim) * matreatensor(m, n) *
                                   defgrd_inv_n_p +
                               defgrd_inv(m, idim) * matreatensor(m, n) * dFinvTdus_n_p);
                  }
                }
              }

              (*stiffmatrix)(numdim_* inode + idim, fi + jdim) += shapefct(inode) * val;
            }
          }
        }
      }
    }
    else
    {
      const double fac = detJ_w * porosity * porosity * J * J;
      for (int idim = 0; idim < numdim_; idim++)
      {
        for (int jdim = 0; jdim < numdim_; jdim++)
        {
          for (int jnode = 0; jnode < numnod_; jnode++)
          {
            const int fi = numdim_ * jnode;
            const double dphi_dus_fi_l = dphi_dus(fi + jdim);
            const double dJ_dus_fi_l = dJ_dus(fi + jdim);

            for (int inode = 0; inode < numnod_; inode++)
            {
              double val = 0.0;
              for (int m = 0; m < numdim_; ++m)
              {
                const double dFinvTdus_idim_m_fi_jdim = dFinvTdus(idim * numdim_ + m, fi + jdim);
                const double defgrd_inv_m_idim = defgrd_inv(m, idim);
                for (int n = 0; n < numdim_; ++n)
                {
                  const double matreatensor_m_n = matreatensor(m, n);
                  const double linreac_dphi_m_n = linreac_dphi(m, n);
                  const double linreac_dJ_m_n = linreac_dJ(m, n);

                  for (int p = 0; p < numdim_; ++p)
                  {
                    val +=
                        fac * (velint(p) - fvelint(p)) *
                        (dFinvTdus_idim_m_fi_jdim * matreatensor_m_n * defgrd_inv(n, p) +
                            defgrd_inv_m_idim * matreatensor_m_n *
                                dFinvTdus(p * numdim_ + n, fi + jdim) +
                            defgrd_inv_m_idim *
                                (linreac_dphi_m_n * dphi_dus_fi_l + linreac_dJ_m_n * dJ_dus_fi_l) *
                                defgrd_inv(n, p));
                  }
                }
              }
              (*stiffmatrix)(numdim_* inode + idim, fi + jdim) += val * shapefct(inode);
            }
          }
        }
      }
    }

    // inverse Right Cauchy-Green tensor as vector
    static Core::LinAlg::Matrix<numstr_, 1> C_inv_vec;
    for (int i = 0, k = 0; i < numdim_; i++)
      for (int j = 0; j < numdim_ - i; j++, k++) C_inv_vec(k) = C_inv(i + j, j);

    // B^T . C^-1
    static Core::LinAlg::Matrix<numdof_, 1> cinvb(true);
    cinvb.multiply_tn(bop, C_inv_vec);

    const double fac1 = -detJ_w * press;
    const double fac2 = fac1 * J;

    // additional fluid stress term -(B^T . C^-1 * J * p^f * detJ * w(gp))
    force->update(fac2, cinvb, 1.0);

    static Core::LinAlg::Matrix<numdof_, numdof_> tmp1;
    static Core::LinAlg::Matrix<numdof_, numdof_> tmp2;

    tmp1.multiply(fac1, cinvb, dJ_dus);
    tmp2.multiply_tn(fac2, bop, dCinv_dus);

    // additional fluid stress- stiffness term -(B^T . C^-1 . dJ/d(us) * p^f * detJ * w(gp))
    stiffmatrix->update(1.0, tmp1, 1.0);

    // additional fluid stress- stiffness term -(B^T .  dC^-1/d(us) * J * p^f * detJ * w(gp))
    stiffmatrix->update(1.0, tmp2, 1.0);

    // integrate `geometric' stiffness matrix and add to keu *****************
    Core::LinAlg::Matrix<numstr_, 1> sfac(C_inv_vec);  // auxiliary integrated stress

    // scale and add viscous stress
    sfac.update(detJ_w, fstress, fac2);  // detJ*w(gp)*[S11,S22,S33,S12=S21,S23=S32,S13=S31]

    std::vector<double> SmB_L(3);  // intermediate Sm.B_L
    // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
    for (int inod = 0; inod < numnod_; ++inod)
    {
      SmB_L[0] = sfac(0) * N_XYZ(0, inod) + sfac(3) * N_XYZ(1, inod) + sfac(5) * N_XYZ(2, inod);
      SmB_L[1] = sfac(3) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod) + sfac(4) * N_XYZ(2, inod);
      SmB_L[2] = sfac(5) * N_XYZ(0, inod) + sfac(4) * N_XYZ(1, inod) + sfac(2) * N_XYZ(2, inod);
      for (int jnod = 0; jnod < numnod_; ++jnod)
      {
        double bopstrbop = 0.0;  // intermediate value
        for (int idim = 0; idim < numdim_; ++idim) bopstrbop += N_XYZ(idim, jnod) * SmB_L[idim];
        (*stiffmatrix)(numdim_* inod + 0, numdim_ * jnod + 0) += bopstrbop;
        (*stiffmatrix)(numdim_* inod + 1, numdim_ * jnod + 1) += bopstrbop;
        (*stiffmatrix)(numdim_* inod + 2, numdim_ * jnod + 2) += bopstrbop;
      }
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::fill_matrix_and_vectors_pressure_based(
    const int& gp, const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& press,
    const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
    const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
    const Core::LinAlg::Matrix<1, numdof_>& dJ_dus,
    const Core::LinAlg::Matrix<numstr_, numdof_>& dCinv_dus,
    const Core::LinAlg::Matrix<1, numdof_>& dps_dus,
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix, Core::LinAlg::Matrix<numdof_, 1>* force)
{
  const double detJ_w = detJ_[gp] * intpoints_.weight(gp);

  //-----------inverse Right Cauchy-Green tensor as vector in voigt notation
  static Core::LinAlg::Matrix<numstr_, 1> C_inv_vec(true);
  for (int i = 0, k = 0; i < numdim_; i++)
    for (int j = 0; j < numdim_ - i; j++, k++) C_inv_vec(k) = C_inv(i + j, j);

  // B^T . C^-1
  static Core::LinAlg::Matrix<numdof_, 1> cinvb(true);
  cinvb.multiply_tn(bop, C_inv_vec);

  const double fac1 = -detJ_w * press;
  const double fac2 = fac1 * J;

  // update internal force vector
  if (force != nullptr)
  {
    // additional fluid stress- stiffness term RHS -(B^T .  C^-1  * J * p^f * detJ * w(gp))
    force->update(fac2, cinvb, 1.0);
  }

  // update stiffness matrix
  if (stiffmatrix != nullptr)
  {
    static Core::LinAlg::Matrix<numdof_, numdof_> tmp;

    // additional fluid stress- stiffness term -(B^T . C^-1 . dJ/d(us) * p^f * detJ * w(gp))
    tmp.multiply(fac1, cinvb, dJ_dus);
    stiffmatrix->update(1.0, tmp, 1.0);

    // additional fluid stress- stiffness term -(B^T .  dC^-1/d(us) * J * p^f * detJ * w(gp))
    tmp.multiply_tn(fac2, bop, dCinv_dus);
    stiffmatrix->update(1.0, tmp, 1.0);

    // additional fluid stress- stiffness term -(B^T .  dC^-1 * J * dp^s/d(us) * detJ * w(gp))
    tmp.multiply(-detJ_w * J, cinvb, dps_dus);
    stiffmatrix->update(1.0, tmp, 1.0);

    // integrate `geometric' stiffness matrix and add to keu *****************
    Core::LinAlg::Matrix<numstr_, 1> sfac(C_inv_vec);  // auxiliary integrated stress

    // scale
    sfac.scale(fac2);

    std::vector<double> SmB_L(3);  // intermediate Sm.B_L
    // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
    for (int inod = 0; inod < numnod_; ++inod)
    {
      SmB_L[0] = sfac(0) * N_XYZ(0, inod) + sfac(3) * N_XYZ(1, inod) + sfac(5) * N_XYZ(2, inod);
      SmB_L[1] = sfac(3) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod) + sfac(4) * N_XYZ(2, inod);
      SmB_L[2] = sfac(5) * N_XYZ(0, inod) + sfac(4) * N_XYZ(1, inod) + sfac(2) * N_XYZ(2, inod);
      for (int jnod = 0; jnod < numnod_; ++jnod)
      {
        double bopstrbop = 0.0;  // intermediate value
        for (int idim = 0; idim < numdim_; ++idim) bopstrbop += N_XYZ(idim, jnod) * SmB_L[idim];
        (*stiffmatrix)(numdim_* inod + 0, numdim_ * jnod + 0) += bopstrbop;
        (*stiffmatrix)(numdim_* inod + 1, numdim_ * jnod + 1) += bopstrbop;
        (*stiffmatrix)(numdim_* inod + 2, numdim_ * jnod + 2) += bopstrbop;
      }
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::fill_matrix_and_vectors_brinkman(const int& gp,
    const double& J, const double& porosity, const Core::LinAlg::Matrix<numdim_, numdim_>& fvelder,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
    const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
    const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
    const Core::LinAlg::Matrix<1, numdof_>& dphi_dus,
    const Core::LinAlg::Matrix<1, numdof_>& dJ_dus,
    const Core::LinAlg::Matrix<numstr_, numdof_>& dCinv_dus,
    const Core::LinAlg::Matrix<numdim_ * numdim_, numdof_>& dFinvTdus,
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix, Core::LinAlg::Matrix<numdof_, 1>* force,
    Core::LinAlg::Matrix<numstr_, 1>& fstress)
{
  double detJ_w = detJ_[gp] * intpoints_.weight(gp);

  double visc = fluid_mat_->viscosity();
  Core::LinAlg::Matrix<numdim_, numdim_> CinvFvel;
  Core::LinAlg::Matrix<numdim_, numdim_> visctress1;
  CinvFvel.multiply(C_inv, fvelder);
  visctress1.multiply_nt(CinvFvel, defgrd_inv);
  Core::LinAlg::Matrix<numdim_, numdim_> visctress2(visctress1);
  visctress1.update_t(1.0, visctress2, 1.0);

  fstress(0) = visctress1(0, 0);
  fstress(1) = visctress1(1, 1);
  fstress(2) = visctress1(2, 2);
  fstress(3) = visctress1(0, 1);
  fstress(4) = visctress1(1, 2);
  fstress(5) = visctress1(2, 0);

  fstress.scale(detJ_w * visc * J * porosity);

  // B^T . C^-1
  static Core::LinAlg::Matrix<numdof_, 1> fstressb(true);
  fstressb.multiply_tn(bop, fstress);

  force->update(1.0, fstressb, 1.0);

  // evaluate viscous terms (for darcy-brinkman flow only)
  {
    static Core::LinAlg::Matrix<numdim_, numdim_> tmp;
    tmp.multiply_nt(fvelder, defgrd_inv);

    double fac = detJ_w * visc;

    Core::LinAlg::Matrix<numstr_, numdof_> fstress_dus(true);
    {
      const double tmp_0_0 = tmp(0, 0);
      const double tmp_0_1 = tmp(0, 1);
      const double tmp_0_2 = tmp(0, 2);
      const double tmp_1_0 = tmp(1, 0);
      const double tmp_1_1 = tmp(1, 1);
      const double tmp_1_2 = tmp(1, 2);
      const double tmp_2_0 = tmp(2, 0);
      const double tmp_2_1 = tmp(2, 1);
      const double tmp_2_2 = tmp(2, 2);

      const double CinvFvel_0_0 = CinvFvel(0, 0);
      const double CinvFvel_0_1 = CinvFvel(0, 1);
      const double CinvFvel_0_2 = CinvFvel(0, 2);
      const double CinvFvel_1_0 = CinvFvel(1, 0);
      const double CinvFvel_1_1 = CinvFvel(1, 1);
      const double CinvFvel_1_2 = CinvFvel(1, 2);
      const double CinvFvel_2_0 = CinvFvel(2, 0);
      const double CinvFvel_2_1 = CinvFvel(2, 1);
      const double CinvFvel_2_2 = CinvFvel(2, 2);

      for (int n = 0; n < numnod_; ++n)
      {
        for (int k = 0; k < numdim_; ++k)
        {
          const int gid = n * numdim_ + k;

          fstress_dus(0, gid) += 2 * (dCinv_dus(0, gid) * tmp_0_0 + dCinv_dus(3, gid) * tmp_1_0 +
                                         dCinv_dus(5, gid) * tmp_2_0);
          fstress_dus(1, gid) += 2 * (dCinv_dus(3, gid) * tmp_0_1 + dCinv_dus(1, gid) * tmp_1_1 +
                                         dCinv_dus(4, gid) * tmp_2_1);
          fstress_dus(2, gid) += 2 * (dCinv_dus(5, gid) * tmp_0_2 + dCinv_dus(4, gid) * tmp_1_2 +
                                         dCinv_dus(2, gid) * tmp_2_2);
          /* ~~~ */
          fstress_dus(3, gid) += +dCinv_dus(0, gid) * tmp_0_1 + dCinv_dus(3, gid) * tmp_1_1 +
                                 dCinv_dus(5, gid) * tmp_2_1 + dCinv_dus(3, gid) * tmp_0_0 +
                                 dCinv_dus(1, gid) * tmp_1_0 + dCinv_dus(4, gid) * tmp_2_0;
          fstress_dus(4, gid) += +dCinv_dus(3, gid) * tmp_0_2 + dCinv_dus(1, gid) * tmp_1_2 +
                                 dCinv_dus(4, gid) * tmp_2_2 + dCinv_dus(5, gid) * tmp_0_1 +
                                 dCinv_dus(4, gid) * tmp_1_1 + dCinv_dus(2, gid) * tmp_2_1;
          fstress_dus(5, gid) += +dCinv_dus(5, gid) * tmp_0_0 + dCinv_dus(4, gid) * tmp_1_0 +
                                 dCinv_dus(2, gid) * tmp_2_0 + dCinv_dus(0, gid) * tmp_0_2 +
                                 dCinv_dus(3, gid) * tmp_1_2 + dCinv_dus(5, gid) * tmp_2_2;

          fstress_dus(0, gid) += 2 * CinvFvel_0_0 * dFinvTdus(0 * numdim_, gid) +
                                 2 * CinvFvel_0_1 * dFinvTdus(1 * numdim_, gid) +
                                 2 * CinvFvel_0_2 * dFinvTdus(2 * numdim_, gid);
          fstress_dus(1, gid) += 2 * CinvFvel_1_0 * dFinvTdus(0 * numdim_ + 1, gid) +
                                 2 * CinvFvel_1_1 * dFinvTdus(1 * numdim_ + 1, gid) +
                                 2 * CinvFvel_1_2 * dFinvTdus(2 * numdim_ + 1, gid);
          fstress_dus(2, gid) += 2 * CinvFvel_2_0 * dFinvTdus(0 * numdim_ + 2, gid) +
                                 2 * CinvFvel_2_1 * dFinvTdus(1 * numdim_ + 2, gid) +
                                 2 * CinvFvel_2_2 * dFinvTdus(2 * numdim_ + 2, gid);
          /* ~~~ */
          fstress_dus(3, gid) += +CinvFvel_0_0 * dFinvTdus(0 * numdim_ + 1, gid) +
                                 CinvFvel_1_0 * dFinvTdus(0 * numdim_, gid) +
                                 CinvFvel_0_1 * dFinvTdus(1 * numdim_ + 1, gid) +
                                 CinvFvel_1_1 * dFinvTdus(1 * numdim_, gid) +
                                 CinvFvel_0_2 * dFinvTdus(2 * numdim_ + 1, gid) +
                                 CinvFvel_1_2 * dFinvTdus(2 * numdim_, gid);
          fstress_dus(4, gid) += +CinvFvel_1_0 * dFinvTdus(0 * numdim_ + 2, gid) +
                                 CinvFvel_2_0 * dFinvTdus(0 * numdim_ + 1, gid) +
                                 CinvFvel_1_1 * dFinvTdus(1 * numdim_ + 2, gid) +
                                 CinvFvel_2_1 * dFinvTdus(1 * numdim_ + 1, gid) +
                                 CinvFvel_1_2 * dFinvTdus(2 * numdim_ + 2, gid) +
                                 CinvFvel_2_2 * dFinvTdus(2 * numdim_ + 1, gid);
          fstress_dus(5, gid) += +CinvFvel_2_0 * dFinvTdus(0 * numdim_, gid) +
                                 CinvFvel_0_0 * dFinvTdus(0 * numdim_ + 2, gid) +
                                 CinvFvel_2_1 * dFinvTdus(1 * numdim_, gid) +
                                 CinvFvel_0_1 * dFinvTdus(1 * numdim_ + 2, gid) +
                                 CinvFvel_2_2 * dFinvTdus(2 * numdim_, gid) +
                                 CinvFvel_0_2 * dFinvTdus(2 * numdim_ + 2, gid);
        }
      }
    }

    static Core::LinAlg::Matrix<numdof_, numdof_> fluidstress_part;

    // additional viscous fluid stress- stiffness term (B^T . fstress . dJ/d(us) * porosity * detJ *
    // w(gp))
    fluidstress_part.multiply(fac * porosity, fstressb, dJ_dus);
    stiffmatrix->update(1.0, fluidstress_part, 1.0);

    // additional fluid stress- stiffness term (B^T .  d\phi/d(us) . fstress  * J * w(gp))
    fluidstress_part.multiply(fac * J, fstressb, dphi_dus);
    stiffmatrix->update(1.0, fluidstress_part, 1.0);

    // additional fluid stress- stiffness term (B^T .  phi . dfstress/d(us)  * J * w(gp))
    fluidstress_part.multiply_tn(detJ_w * visc * J * porosity, bop, fstress_dus);
    stiffmatrix->update(1.0, fluidstress_part, 1.0);
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::fill_matrix_and_vectors_od(const int& gp,
    const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& porosity,
    const double& dphi_dp, const Core::LinAlg::Matrix<numdim_, 1>& velint,
    const Core::LinAlg::Matrix<numdim_, 1>& fvelint,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
    const Core::LinAlg::Matrix<numdim_, 1>& Gradp,
    const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
    const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
    Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>* stiffmatrix)
{
  double detJ_w = detJ_[gp] * intpoints_.weight(gp);

  static Core::LinAlg::Matrix<numdim_, numdim_> matreatensor(true);
  static Core::LinAlg::Matrix<numdim_, numdim_> reatensor(true);
  static Core::LinAlg::Matrix<numdim_, numdim_> linreac_dphi(true);
  static Core::LinAlg::Matrix<numdim_, numdim_> linreac_dJ(true);
  static Core::LinAlg::Matrix<numdim_, 1> reafvel(true);
  static Core::LinAlg::Matrix<numdim_, 1> reavel(true);
  {
    Core::LinAlg::Matrix<numdim_, numdim_> temp(true);
    std::vector<double> anisotropic_permeability_coeffs =
        compute_anisotropic_permeability_coeffs_at_gp(shapefct);
    fluid_mat_->compute_reaction_tensor(matreatensor, J, porosity,
        anisotropic_permeability_directions_, anisotropic_permeability_coeffs);
    fluid_mat_->compute_lin_mat_reaction_tensor(linreac_dphi, linreac_dJ, J, porosity);
    temp.multiply(1.0, matreatensor, defgrd_inv);
    reatensor.multiply_tn(defgrd_inv, temp);
    reavel.multiply(reatensor, velint);
    reafvel.multiply(reatensor, fvelint);
  }

  //-----------inverse Right Cauchy-Green tensor as vector in voigt notation
  static Core::LinAlg::Matrix<numstr_, 1> C_inv_vec(true);
  for (int i = 0, k = 0; i < numdim_; i++)
    for (int j = 0; j < numdim_ - i; j++, k++) C_inv_vec(k) = C_inv(i + j, j);

  // B^T . C^-1
  static Core::LinAlg::Matrix<numdof_, 1> cinvb(true);
  cinvb.multiply_tn(bop, C_inv_vec);

  // F^-T * grad p
  static Core::LinAlg::Matrix<numdim_, 1> Finvgradp;
  Finvgradp.multiply_tn(defgrd_inv, Gradp);

  // F^-T * N_XYZ
  static Core::LinAlg::Matrix<numdim_, numnod_> FinvNXYZ;
  FinvNXYZ.multiply_tn(defgrd_inv, N_XYZ);

  {
    const double fac = detJ_w * J * J * 2 * porosity * dphi_dp;
    for (int idim = 0; idim < numdim_; idim++)
    {
      const double reafvel_idim = reafvel(idim);
      const double reac_vel_idim = reavel(idim);

      for (int jnode = 0; jnode < numnod_; jnode++)
      {
        const int fkp1 = (numdim_ + 1) * jnode;

        const double val = fac * shapefct(jnode) * (reac_vel_idim - reafvel_idim);
        for (int inode = 0; inode < numnod_; inode++)
        {
          /*-------structure- fluid pressure coupling:  "dracy-terms" + "reactive darcy-terms"
           - 2 * reacoeff * J * v^f * phi * d(phi)/dp  Dp
           + 2 * reacoeff * J * v^s * phi * d(phi)/dp  Dp
           */
          (*stiffmatrix)(numdim_* inode + idim, fkp1 + numdim_) += shapefct(inode) * val;
        }
      }
    }
  }

  {
    for (int idim = 0; idim < numdim_; idim++)
    {
      const double Finvgradp_idim = Finvgradp(idim);
      for (int jnode = 0; jnode < numnod_; jnode++)
      {
        const int fkp1 = (numdim_ + 1) * jnode;

        const double val1 = detJ_w * (-1.0) * J * shapefct(jnode);
        const double val2 =
            -1.0 * detJ_w * J *
            (Finvgradp_idim * dphi_dp * shapefct(jnode) + porosity * FinvNXYZ(idim, jnode));

        for (int inode = 0; inode < numnod_; inode++)
        {
          /*-------structure- fluid pressure coupling: "stress terms" + "pressure gradient terms"
           -B^T . ( -1*J*C^-1 ) * Dp
           - J * F^-T * dphi/dp * Dp - J * F^-T * d(Grad((p))/(dp) * phi * Dp
           */
          (*stiffmatrix)(numdim_* inode + idim, fkp1 + numdim_) +=
              val1 * cinvb(numdim_ * inode + idim) + val2 * shapefct(inode);
        }
      }
    }
  }

  // check if derivatives of reaction tensor are zero --> significant speed up
  if (fluid_mat_->permeability_function() != Mat::PAR::constant)
  {
    const double fac = detJ_w * J * J * porosity * porosity * dphi_dp;
    for (int idim = 0; idim < numdim_; idim++)
    {
      for (int jnode = 0; jnode < numnod_; jnode++)
      {
        const int fkp1 = (numdim_ + 1) * jnode;
        const double shapefct_jnode = shapefct(jnode);

        for (int inode = 0; inode < numnod_; inode++)
        {
          double val = 0.0;
          for (int p = 0; p < numdim_; ++p)
          {
            const double velint_fvelint_p = velint(p) - fvelint(p);
            for (int n = 0; n < numdim_; ++n)
            {
              const double defgrd_inv_n_p = defgrd_inv(n, p);
              for (int m = 0; m < numdim_; ++m)
              {
                val += fac * defgrd_inv(m, idim) * linreac_dphi(m, n) * defgrd_inv_n_p *
                       velint_fvelint_p;
              }
            }
          }
          val *= shapefct_jnode;

          /*-------structure- fluid pressure coupling:   "reactive darcy-terms"
           + J * J * phi * phi * defgrd_^-T * d(mat_reacoeff)/d(phi) * defgrd_^-1 * (v^s-v^f) *
           d(phi)/dp Dp
           */
          (*stiffmatrix)(numdim_* inode + idim, fkp1 + numdim_) += shapefct(inode) * val;
        }
      }
    }
  }

  {
    const double fac = detJ_w * J * J * porosity * porosity;
    for (int idim = 0; idim < numdim_; idim++)
    {
      for (int jdim = 0; jdim < numdim_; jdim++)
      {
        const double reatensor_idim_jdim = reatensor(idim, jdim);
        for (int jnode = 0; jnode < numnod_; jnode++)
        {
          const double val = -1.0 * fac * shapefct(jnode) * reatensor_idim_jdim;

          /*-------structure- fluid velocity coupling:  "darcy-terms"
           -reacoeff * J * J *  phi^2 *  Dv^f
           */
          for (int inode = 0; inode < numnod_; inode++)
            (*stiffmatrix)(numdim_* inode + idim, (numdim_ + 1) * jnode + jdim) +=
                val * shapefct(inode);
        }
      }
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::fill_matrix_and_vectors_od_pressure_based(
    const int& gp, const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J,
    const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
    const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv, const std::vector<double>& solpressderiv,
    Core::LinAlg::SerialDenseMatrix& couplmat)
{
  const double detJ_w = detJ_[gp] * intpoints_.weight(gp);

  // inverse Right Cauchy-Green tensor as vector
  static Core::LinAlg::Matrix<numstr_, 1> C_inv_vec;
  for (int i = 0, k = 0; i < numdim_; i++)
    for (int j = 0; j < numdim_ - i; j++, k++) C_inv_vec(k) = C_inv(i + j, j);

  // B^T . C^-1
  static Core::LinAlg::Matrix<numdof_, 1> cinvb(true);
  cinvb.multiply_tn(bop, C_inv_vec);

  const int totalnumdofpernode = fluidmulti_mat_->num_mat();

  {
    for (int i = 0; i < numnod_; i++)
    {
      const int fi = numdim_ * i;

      for (int j = 0; j < numdim_; j++)
      {
        for (int k = 0; k < numnod_; k++)
        {
          for (int iphase = 0; iphase < totalnumdofpernode; iphase++)
          {
            int fk_press = k * totalnumdofpernode + iphase;

            /*-------structure- fluid pressure coupling: "stress term"
             -B^T . ( -1*J*C^-1 ) * Dp
             */
            couplmat(fi + j, fk_press) +=
                detJ_w * cinvb(fi + j) * (-1.0) * J * shapefct(k) * solpressderiv[iphase];
          }
        }
      }
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::fill_matrix_and_vectors_brinkman_od(const int& gp,
    const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& porosity,
    const double& dphi_dp, const Core::LinAlg::Matrix<numdim_, numdim_>& fvelder,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
    const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
    const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
    Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>* stiffmatrix)
{
  double detJ_w = detJ_[gp] * intpoints_.weight(gp);  // gpweights[gp];

  static Core::LinAlg::Matrix<numstr_, 1> fstress;

  double visc = fluid_mat_->viscosity();
  static Core::LinAlg::Matrix<numdim_, numdim_> CinvFvel;
  static Core::LinAlg::Matrix<numdim_, numdim_> tmp;
  CinvFvel.multiply(C_inv, fvelder);
  tmp.multiply_nt(CinvFvel, defgrd_inv);
  static Core::LinAlg::Matrix<numdim_, numdim_> tmp2(tmp);
  tmp.update_t(1.0, tmp2, 1.0);

  fstress(0) = tmp(0, 0);
  fstress(1) = tmp(1, 1);
  fstress(2) = tmp(2, 2);
  fstress(3) = tmp(0, 1);
  fstress(4) = tmp(1, 2);
  fstress(5) = tmp(2, 0);

  // B^T . \sigma
  static Core::LinAlg::Matrix<numdof_, 1> fstressb;
  fstressb.multiply_tn(bop, fstress);
  static Core::LinAlg::Matrix<numdim_, numnod_> N_XYZ_Finv;
  N_XYZ_Finv.multiply(defgrd_inv, N_XYZ);

  // dfstress/dv^f
  Core::LinAlg::Matrix<numstr_, numdof_> dfstressb_dv;
  for (int j = 0; j < numdim_; j++)
  {
    const double C_inv_0_j = C_inv(0, j);
    const double C_inv_1_j = C_inv(0, j);
    const double C_inv_2_j = C_inv(0, j);

    for (int i = 0; i < numnod_; i++)
    {
      const int k = numdim_ * i + j;
      const double N_XYZ_Finv_0_i = N_XYZ_Finv(0, i);
      const double N_XYZ_Finv_1_i = N_XYZ_Finv(0, i);
      const double N_XYZ_Finv_2_i = N_XYZ_Finv(0, i);

      dfstressb_dv(0, k) = 2 * N_XYZ_Finv_0_i * C_inv_0_j;
      dfstressb_dv(1, k) = 2 * N_XYZ_Finv_1_i * C_inv_1_j;
      dfstressb_dv(2, k) = 2 * N_XYZ_Finv_2_i * C_inv_2_j;
      //**********************************
      dfstressb_dv(3, k) = N_XYZ_Finv_0_i * C_inv_1_j + N_XYZ_Finv_1_i * C_inv_0_j;
      dfstressb_dv(4, k) = N_XYZ_Finv_1_i * C_inv_2_j + N_XYZ_Finv_2_i * C_inv_1_j;
      dfstressb_dv(5, k) = N_XYZ_Finv_2_i * C_inv_0_j + N_XYZ_Finv_0_i * C_inv_2_j;
    }
  }

  // B^T . dfstress/dv^f
  Core::LinAlg::Matrix<numdof_, numdof_> dfstressb_dv_bop(true);
  dfstressb_dv_bop.multiply_tn(bop, dfstressb_dv);

  for (int i = 0; i < numnod_; i++)
  {
    const int fi = noddof_ * i;

    for (int j = 0; j < numdim_; j++)
    {
      const double fstressb_i_j = fstressb(fi + j);

      for (int k = 0; k < numnod_; k++)
      {
        const int fk = noddof_ * k;
        const int fkp1 = (numdim_ + 1) * k;

        /*-------structure- fluid pressure coupling: "darcy-brinkman stress terms"
         B^T . ( \mu*J - d(phi)/(dp) * fstress ) * Dp
         */
        (*stiffmatrix)(fi + j, fkp1 + numdim_) +=
            detJ_w * fstressb_i_j * dphi_dp * visc * J * shapefct(k);
        for (int l = 0; l < noddof_; l++)
        {
          /*-------structure- fluid velocity coupling: "darcy-brinkman stress terms"
           B^T . ( \mu*J - phi * dfstress/dv^f ) * Dp
           */
          (*stiffmatrix)(fi + j, fkp1 + l) +=
              detJ_w * visc * J * porosity * dfstressb_dv_bop(fi + j, fk + l);
        }
      }
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::compute_def_gradient(
    Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr)
{
  if (So3Ele::kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)  // total lagrange (nonlinear)
  {
    // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
    defgrd.multiply_nt(xcurr, N_XYZ);  //  (6.17)
  }
  else if (So3Ele::kintype_ == Inpar::Solid::KinemType::linear)  // linear kinematics
  {
    defgrd.clear();
    for (int i = 0; i < numdim_; i++) defgrd(i, i) = 1.0;
  }
  else
    FOUR_C_THROW("invalid kinematic type!");
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::get_cauchy_n_dir_and_derivatives_at_xi(
    const Core::LinAlg::Matrix<3, 1>& xi, const std::vector<double>& disp,
    const std::vector<double>& pres, const Core::LinAlg::Matrix<3, 1>& n,
    const Core::LinAlg::Matrix<3, 1>& dir, double& cauchy_n_dir,
    Core::LinAlg::SerialDenseMatrix* d_cauchyndir_dd,
    Core::LinAlg::SerialDenseMatrix* d_cauchyndir_dp, Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dn,
    Core::LinAlg::Matrix<3, 1>* d_cauchyndir_ddir, Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dxi)
{
  if (fluid_mat_->type() != Mat::PAR::darcy)
    FOUR_C_THROW("GetCauchyAtXi just implemented for pure Darcy flow!");

  if (distype != Core::FE::CellType::hex8)
    FOUR_C_THROW("GetCauchyAtXi for Poro just implemented for hex8!");

  So3Ele::get_cauchy_n_dir_and_derivatives_at_xi(xi, disp, n, dir, cauchy_n_dir, d_cauchyndir_dd,
      nullptr, nullptr, nullptr, nullptr, d_cauchyndir_dn, d_cauchyndir_ddir, d_cauchyndir_dxi,
      nullptr, nullptr, nullptr, nullptr, nullptr);

  // Add pressure to sigma_nt
  const double dot = n(0, 0) * dir(0, 0) + n(1, 0) * dir(1, 0) + n(2, 0) * dir(2, 0);
  if (fabs(dot) > 1e-30)
  {
    Core::LinAlg::Matrix<NUMNOD_SOH8, 1> shapefcts;
    Core::FE::shape_function<Core::FE::CellType::hex8>(xi, shapefcts);

    for (unsigned nlid = 0; nlid < NUMNOD_SOH8; ++nlid)
      cauchy_n_dir -= pres[nlid] * shapefcts(nlid, 0) * dot;

    if (d_cauchyndir_dp || d_cauchyndir_dn || d_cauchyndir_ddir || d_cauchyndir_dxi)
    {
      Core::LinAlg::Matrix<NUMDIM_SOH8, NUMNOD_SOH8> deriv;
      Core::FE::shape_function_deriv1<Core::FE::CellType::hex8>(xi, deriv);

      d_cauchyndir_dp->reshape(NUMNOD_SOH8, 1);
      Core::LinAlg::Matrix<NUMNOD_SOH8, 1> dsntdp_m(d_cauchyndir_dp->values(), true);

      for (unsigned nlid = 0; nlid < NUMNOD_SOH8; ++nlid)
      {
        dsntdp_m(nlid, 0) = -dot * shapefcts(nlid, 0);
        for (unsigned dim = 0; dim < 3; ++dim)
        {
          (*d_cauchyndir_dn)(dim, 0) -= pres[nlid] * shapefcts(nlid, 0) * dir(dim, 0);
          (*d_cauchyndir_ddir)(dim, 0) -= pres[nlid] * shapefcts(nlid, 0) * n(dim, 0);
          (*d_cauchyndir_dxi)(dim, 0) -= pres[nlid] * deriv(dim, nlid) * dot;
        }
      }
    }
  }
}

template <class So3Ele, Core::FE::CellType distype>
std::vector<double>
Discret::Elements::So3Poro<So3Ele, distype>::compute_anisotropic_permeability_coeffs_at_gp(
    const Core::LinAlg::Matrix<numnod_, 1>& shapefct) const
{
  std::vector<double> anisotropic_permeability_coeffs(numdim_, 0.0);

  for (int node = 0; node < numnod_; ++node)
  {
    const double shape_val = shapefct(node);
    for (int dim = 0; dim < numdim_; ++dim)
    {
      anisotropic_permeability_coeffs[dim] +=
          shape_val * anisotropic_permeability_nodal_coeffs_[dim][node];
    }
  }

  return anisotropic_permeability_coeffs;
}

FOUR_C_NAMESPACE_CLOSE

#include "4C_so3_poro.inst.hpp"
