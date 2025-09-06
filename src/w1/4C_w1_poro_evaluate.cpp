// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_w1_poro.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

#include <iterator>

FOUR_C_NAMESPACE_OPEN

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::pre_evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la)
{
  if (scatra_coupling_)
  {
    if (la.size() > 2)
    {
      if (discretization.has_state(2, "scalar"))
      {
        // check if you can get the scalar state
        std::shared_ptr<const Core::LinAlg::Vector<double>> scalarnp =
            discretization.get_state(2, "scalar");

        // extract local values of the global vectors
        std::vector<double> myscalar = Core::FE::extract_values(*scalarnp, la[2].lm_);

        if (num_material() < 3) FOUR_C_THROW("no third material defined for Wall poro element!");
        std::shared_ptr<Core::Mat::Material> scatramat = material(2);

        int numscal = 1;
        if (scatramat->material_type() == Core::Materials::m_matlist or
            scatramat->material_type() == Core::Materials::m_matlist_reactions)
        {
          std::shared_ptr<Mat::MatList> matlist =
              std::dynamic_pointer_cast<Mat::MatList>(scatramat);
          numscal = matlist->num_mat();
        }

        std::shared_ptr<std::vector<double>> scalar =
            std::make_shared<std::vector<double>>(numscal, 0.0);
        if ((int)myscalar.size() != numscal * numnod_) FOUR_C_THROW("sizes do not match!");

        for (int i = 0; i < numnod_; i++)
          for (int j = 0; j < numscal; j++) scalar->at(j) += myscalar[numscal * i + j] / numnod_;

        params.set("scalar", scalar);
      }
    }
  }
}

template <Core::FE::CellType distype>
int Discret::Elements::Wall1Poro<distype>::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  if (not init_) FOUR_C_THROW("internal element data not initialized!");

  this->set_params_interface_ptr(params);
  Core::Elements::ActionType act = Core::Elements::none;

  if (this->is_params_interface())
  {
    act = this->params_interface().get_action_type();
  }
  else
  {
    // get the required action
    std::string action = params.get<std::string>("action", "none");
    if (action == "none")
      FOUR_C_THROW("No action supplied");
    else if (action == "struct_poro_calc_fluidcoupling")
      act = Core::Elements::struct_poro_calc_fluidcoupling;
    else if (action == "struct_poro_calc_scatracoupling")
      act = Core::Elements::struct_poro_calc_scatracoupling;
    else if (action == "struct_poro_calc_prescoupling")
      act = Core::Elements::struct_poro_calc_prescoupling;
  }

  // what should the element do
  switch (act)
  {
    //==================================================================================
    // off diagonal terms in stiffness matrix for monolithic coupling
    case Core::Elements::struct_poro_calc_fluidcoupling:
    case Core::Elements::struct_poro_calc_scatracoupling:
    case Core::Elements::struct_poro_calc_prescoupling:
    {
      // in some cases we need to write/change some data before evaluating
      pre_evaluate(params, discretization, la);

      my_evaluate(params, discretization, la, elemat1, elemat2, elevec1, elevec2, elevec3);
    }
    break;
    //==================================================================================
    default:
    {
      // in some cases we need to write/change some data before evaluating
      pre_evaluate(params, discretization, la);

      // evaluate parent solid element
      Discret::Elements::Wall1::evaluate(
          params, discretization, la[0].lm_, elemat1, elemat2, elevec1, elevec2, elevec3);

      // add volume coupling specific terms
      my_evaluate(params, discretization, la, elemat1, elemat2, elevec1, elevec2, elevec3);
    }
    break;
  }

  return 0;
}

template <Core::FE::CellType distype>
int Discret::Elements::Wall1Poro<distype>::my_evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  this->set_params_interface_ptr(params);
  Core::Elements::ActionType act = Core::Elements::none;

  if (this->is_params_interface())
  {
    act = this->params_interface().get_action_type();
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
    else if (action == "struct_poro_calc_prescoupling")
      act = Core::Elements::struct_poro_calc_prescoupling;
  }

  // --------------------------------------------------
  // Now do the nurbs specific stuff
  if (shape() == Core::FE::CellType::nurbs4 || shape() == Core::FE::CellType::nurbs9)
  {
    myknots_.resize(2);

    switch (act)
    {
      case Core::Elements::struct_calc_nlnstiffmass:
      case Core::Elements::struct_calc_nlnstiff:
      case Core::Elements::struct_calc_internalforce:
      case Core::Elements::struct_poro_calc_fluidcoupling:
      case Core::Elements::struct_poro_calc_prescoupling:
      case Core::Elements::struct_calc_stress:
      {
        auto* nurbsdis = dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(discretization));

        bool zero_sized = (*((*nurbsdis).get_knot_vector())).get_ele_knots(myknots_, id());

        // skip zero sized elements in knot span --- they correspond to interpolated nodes
        if (zero_sized)
        {
          return (0);
        }

        break;
      }
      default:
        myknots_.clear();
        break;
    }
  }

  // what should the element do
  switch (act)
  {
    //==================================================================================
    // nonlinear stiffness, damping and internal force vector for poroelasticity
    case Core::Elements::struct_calc_nlnstiff:
    {
      // stiffness
      Core::LinAlg::Matrix<numdof_, numdof_> elemat_1(elemat1.values(), true);
      // damping
      Core::LinAlg::Matrix<numdof_, numdof_> elemat_2(elemat2.values(), true);
      // internal force vector
      Core::LinAlg::Matrix<numdof_, 1> elevec_1(elevec1.values(), true);

      // elevec2+3 are not used anyway

      std::vector<int> lm = la[0].lm_;

      Core::LinAlg::Matrix<numdim_, numnod_> mydisp(Core::LinAlg::Initialization::zero);
      extract_values_from_global_vector(discretization, 0, lm, &mydisp, nullptr, "displacement");

      Core::LinAlg::Matrix<numdof_, numdof_>* matptr = nullptr;
      if (elemat_1.is_initialized()) matptr = &elemat_1;

      enum Inpar::Solid::DampKind damping =
          params.get<enum Inpar::Solid::DampKind>("damping", Inpar::Solid::damp_none);
      Core::LinAlg::Matrix<numdof_, numdof_>* matptr2 = nullptr;
      if (elemat_2.is_initialized() and (damping == Inpar::Solid::damp_material))
        matptr2 = &elemat_2;

      if (la.size() > 1)
      {
        if (discretization.has_state(1, "fluidvel"))
        {
          // need current fluid state,
          // call the fluid discretization: fluid equates 2nd dofset
          // disassemble velocities and pressures
          Core::LinAlg::Matrix<numdim_, numnod_> myvel(Core::LinAlg::Initialization::zero);
          Core::LinAlg::Matrix<numdim_, numnod_> myfluidvel(Core::LinAlg::Initialization::zero);
          Core::LinAlg::Matrix<numnod_, 1> myepreaf(Core::LinAlg::Initialization::zero);

          if (discretization.has_state(0, "velocity"))
            extract_values_from_global_vector(
                discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

          // extract local values of the global vectors
          extract_values_from_global_vector(
              discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

          // calculate tangent stiffness matrix
          nonlinear_stiffness_poroelast(
              lm, mydisp, myvel, myfluidvel, myepreaf, matptr, matptr2, &elevec_1, params);
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
                lm, mydisp, myephi, matptr, &elevec_1, params);
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
      Core::LinAlg::Matrix<numdof_, numdof_> elemat_1(elemat1.values(), true);
      // internal force vector
      Core::LinAlg::Matrix<numdof_, 1> elevec_1(elevec1.values(), true);

      // elemat2,elevec2+3 are not used anyway

      // build the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      Core::LinAlg::Matrix<numdim_, numnod_> mydisp(Core::LinAlg::Initialization::zero);
      extract_values_from_global_vector(
          discretization, 0, la[0].lm_, &mydisp, nullptr, "displacement");

      Core::LinAlg::Matrix<numdof_, numdof_>* matptr = nullptr;
      if (elemat_1.is_initialized()) matptr = &elemat_1;

      // we skip this evaluation if the coupling is not setup yet, i.e.
      // if the secondary dofset or the secondary material was not set
      // this can happen during setup of the time integrator or restart
      // there might be a better way. For instance do not evaluate
      // before the setup of the multiphysics problem is completed.
      if (la.size() > 1 and num_material() > 1)
      {
        // need current fluid state,
        // call the fluid discretization: fluid equates 2nd dofset
        // disassemble velocities and pressures

        Core::LinAlg::Matrix<numdim_, numnod_> myvel(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<numdim_, numnod_> myfluidvel(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<numnod_, 1> myepreaf(Core::LinAlg::Initialization::zero);

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

          nonlinear_stiffness_poroelast(
              lm, mydisp, myvel, myfluidvel, myepreaf, matptr, nullptr, &elevec_1, params);
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
                lm, mydisp, myephi, matptr, &elevec_1, params);
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
      Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_> elemat_1(elemat1.values(), true);

      // elemat2,elevec1-3 are not used anyway

      // build the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>* matptr = nullptr;
      if (elemat_1.is_initialized()) matptr = &elemat_1;

      // need current fluid state,
      // call the fluid discretization: fluid equates 2nd dofset
      // disassemble velocities and pressures
      if (discretization.has_state(1, "fluidvel"))
      {
        Core::LinAlg::Matrix<numdim_, numnod_> myvel(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<numdim_, numnod_> myfluidvel(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<numnod_, 1> myepreaf(Core::LinAlg::Initialization::zero);

        Core::LinAlg::Matrix<numdim_, numnod_> mydisp(Core::LinAlg::Initialization::zero);
        extract_values_from_global_vector(
            discretization, 0, la[0].lm_, &mydisp, nullptr, "displacement");

        if (discretization.has_state(0, "velocity"))
          extract_values_from_global_vector(
              discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

        // extract local values of the global vectors
        extract_values_from_global_vector(
            discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

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

          Core::LinAlg::Matrix<numdim_, numnod_> mydisp(Core::LinAlg::Initialization::zero);
          extract_values_from_global_vector(
              discretization, 0, la[0].lm_, &mydisp, nullptr, "displacement");

          // calculate OD-Matrix
          coupling_poroelast_pressure_based(lm, mydisp, myephi, elemat1, params);
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
      Core::LinAlg::Matrix<numdof_, 1> elevec_1(elevec1.values(), true);

      // elemat1+2,elevec2+3 are not used anyway

      // build the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      Core::LinAlg::Matrix<numdim_, numnod_> mydisp(Core::LinAlg::Initialization::zero);
      extract_values_from_global_vector(discretization, 0, lm, &mydisp, nullptr, "displacement");

      // need current fluid state,
      // call the fluid discretization: fluid equates 2nd dofset
      // disassemble velocities and pressures
      if (discretization.has_state(1, "fluidvel"))
      {
        // extract local values of the global vectors
        Core::LinAlg::Matrix<numdim_, numnod_> myfluidvel(Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<numnod_, 1> myepreaf(Core::LinAlg::Initialization::zero);
        extract_values_from_global_vector(
            discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

        Core::LinAlg::Matrix<numdim_, numnod_> myvel(Core::LinAlg::Initialization::zero);
        extract_values_from_global_vector(
            discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

        // calculate tangent stiffness matrix
        nonlinear_stiffness_poroelast(
            lm, mydisp, myvel, myfluidvel, myepreaf, nullptr, nullptr, &elevec_1, params);
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
              lm, mydisp, myephi, nullptr, &elevec_1, params);
        }
      }
    }
    break;

    //==================================================================================
    default:
      // do nothing
      break;
  }
  return 0;
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::nonlinear_stiffness_poroelast(
    std::vector<int>& lm,                                 // location matrix
    Core::LinAlg::Matrix<numdim_, numnod_>& disp,         // current displacements
    Core::LinAlg::Matrix<numdim_, numnod_>& vel,          // current velocities
    Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,       // current fluid velocities
    Core::LinAlg::Matrix<numnod_, 1>& epreaf,             // current fluid pressure
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,  // element stiffness matrix
    Core::LinAlg::Matrix<numdof_, numdof_>* reamatrix,    // element reactive matrix
    Core::LinAlg::Matrix<numdof_, 1>* force,              // element internal force vector
    Teuchos::ParameterList& params                        // algorithmic parameters e.g. time
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

  Core::LinAlg::Matrix<numdof_, numdof_> erea_v(Core::LinAlg::Initialization::zero);

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  gauss_point_loop(params, xrefe, xcurr, disp, vel, evelnp, epreaf, nullptr, erea_v, stiffmatrix,
      reamatrix, force);

  if (reamatrix != nullptr)
  {
    /* additional "reactive darcy-term"
     detJ * w(gp) * ( J * reacoeff * phi^2  ) * D(v_s)
     */
    reamatrix->update(1.0, erea_v, 1.0);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::nonlinear_stiffness_poroelast_pressure_based(
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

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::gauss_point_loop(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodalvel,
    const Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,
    const Core::LinAlg::Matrix<numnod_, 1>& epreaf,
    const Core::LinAlg::Matrix<numnod_, 1>* porosity_dof,
    Core::LinAlg::Matrix<numdof_, numdof_>& erea_v,
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,
    Core::LinAlg::Matrix<numdof_, numdof_>* reamatrix, Core::LinAlg::Matrix<numdof_, 1>* force)
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
  Core::LinAlg::Matrix<numdim_, numdim_> defgrd(Core::LinAlg::Initialization::zero);
  // shape function at gp w.r.t. reference coordinates
  Core::LinAlg::Matrix<numnod_, 1> shapefct;
  // first derivatives at gp w.r.t. reference coordinates
  Core::LinAlg::Matrix<numdim_, numnod_> deriv;

  Core::LinAlg::Matrix<numstr_, 1> fstress(Core::LinAlg::Initialization::zero);

  for (int gp = 0; gp < numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    compute_shape_functions_and_derivatives(gp, shapefct, deriv, N_XYZ);

    // compute deformation gradient
    compute_def_gradient(defgrd, N_XYZ, xcurr);

    // inverse deformation gradient F^-1
    Core::LinAlg::Matrix<numdim_, numdim_> defgrd_inv(Core::LinAlg::Initialization::uninitialized);
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
    compute_b_operator(bop, defgrd, N_XYZ);

    //----------------------------------------------------
    // pressure at integration point
    double press = shapefct.dot(epreaf);

    // pressure gradient at integration point
    Core::LinAlg::Matrix<numdim_, 1> Gradp;
    Gradp.multiply(N_XYZ, epreaf);

    // fluid velocity at integration point
    Core::LinAlg::Matrix<numdim_, 1> fvelint;
    fvelint.multiply(evelnp, shapefct);

    // material fluid velocity gradient at integration point
    Core::LinAlg::Matrix<numdim_, numdim_> fvelder;
    fvelder.multiply_nt(evelnp, N_XYZ);

    // structure displacement and velocity at integration point
    Core::LinAlg::Matrix<numdim_, 1> velint(Core::LinAlg::Initialization::zero);

    for (int i = 0; i < numnod_; i++)
      for (int j = 0; j < numdim_; j++) velint(j) += nodalvel(j, i) * shapefct(i);

    // Right Cauchy-Green tensor = F^T * F
    Core::LinAlg::Matrix<numdim_, numdim_> cauchygreen;
    cauchygreen.multiply_tn(defgrd, defgrd);

    // inverse Right Cauchy-Green tensor
    Core::LinAlg::Matrix<numdim_, numdim_> C_inv(Core::LinAlg::Initialization::uninitialized);
    C_inv.invert(cauchygreen);

    // compute some auxiliary matrixes for computation of linearization
    // dF^-T/dus
    Core::LinAlg::Matrix<numdim_ * numdim_, numdof_> dFinvTdus(Core::LinAlg::Initialization::zero);
    // F^-T * Grad p
    Core::LinAlg::Matrix<numdim_, 1> Finvgradp;
    // dF^-T/dus * Grad p
    Core::LinAlg::Matrix<numdim_, numdof_> dFinvdus_gradp(Core::LinAlg::Initialization::zero);
    // dC^-1/dus * Grad p
    Core::LinAlg::Matrix<numstr_, numdof_> dCinv_dus(Core::LinAlg::Initialization::zero);

    compute_auxiliary_values(
        N_XYZ, defgrd_inv, C_inv, Gradp, dFinvTdus, Finvgradp, dFinvdus_gradp, dCinv_dus);

    //--------------------------------------------------------------------

    // linearization of porosity w.r.t structure displacement d\phi/d(us) = d\phi/dJ*dJ/d(us)
    Core::LinAlg::Matrix<1, numdof_> dphi_dus;
    double porosity = 0.0;

    compute_porosity_and_linearization(
        params, press, volchange, gp, shapefct, porosity_dof, dvolchange_dus, porosity, dphi_dus);

    // **********************evaluate stiffness matrix and force vector**********************
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

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::gauss_point_loop_pressure_based(
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
  Core::LinAlg::Matrix<numdim_, numdim_> defgrd(Core::LinAlg::Initialization::zero);
  // shape function at gp w.r.t. reference coordinates
  Core::LinAlg::Matrix<numnod_, 1> shapefct;
  // first derivatives at gp w.r.t. reference coordinates
  Core::LinAlg::Matrix<numdim_, numnod_> deriv;

  Core::LinAlg::Matrix<numstr_, 1> fstress(Core::LinAlg::Initialization::zero);

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
    Core::LinAlg::Matrix<numdim_, numdim_> defgrd_inv(Core::LinAlg::Initialization::uninitialized);
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
    compute_b_operator(bop, defgrd, N_XYZ);

    // derivative of press w.r.t. displacements (only in case of vol fracs)
    Core::LinAlg::Matrix<1, numdof_> dps_dus(Core::LinAlg::Initialization::zero);

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
    Core::LinAlg::Matrix<numdim_, numdim_> C_inv(Core::LinAlg::Initialization::uninitialized);
    C_inv.invert(cauchygreen);

    // compute some auxiliary matrixes for computation of linearization
    // dC^-1/dus
    Core::LinAlg::Matrix<numstr_, numdof_> dCinv_dus(Core::LinAlg::Initialization::zero);
    for (int n = 0; n < numnod_; ++n)
    {
      for (int k = 0; k < numdim_; ++k)
      {
        const int gid = n * numdim_ + k;
        for (int i = 0; i < numdim_; ++i)
        {
          dCinv_dus(0, gid) += -2 * C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(0, k);
          dCinv_dus(1, gid) += -2 * C_inv(1, i) * N_XYZ(i, n) * defgrd_inv(1, k);
          /* ~~~ */
          dCinv_dus(2, gid) += -C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(1, k) -
                               defgrd_inv(0, k) * N_XYZ(i, n) * C_inv(1, i);
        }
      }
    }

    // **********************evaluate stiffness matrix and force vector**********************
    fill_matrix_and_vectors_pressure_based(
        gp, shapefct, N_XYZ, J, press, bop, C_inv, dJ_dus, dCinv_dus, dps_dus, stiffmatrix, force);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::fill_matrix_and_vectors(const int& gp,
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
  // const double reacoeff = fluid_mat_->compute_reaction_coeff();

  static Core::LinAlg::Matrix<numdim_, numdim_> matreatensor(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<numdim_, numdim_> reatensor(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<numdim_, numdim_> linreac_dphi(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<numdim_, numdim_> linreac_dJ(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<numdim_, 1> reafvel(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<numdim_, 1> reavel(Core::LinAlg::Initialization::zero);
  {
    static Core::LinAlg::Matrix<numdim_, numdim_> temp(Core::LinAlg::Initialization::uninitialized);
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

  const double detJ_w = detJ_[gp] * intpoints_.weight(gp) * thickness_;

  {
    for (int k = 0; k < numnod_; k++)
    {
      const int fk = numdim_ * k;
      const double fac = detJ_w * shapefct(k);
      const double v = fac * porosity * porosity * J * J;

      for (int j = 0; j < numdim_; j++)
      {
        /*-------structure- velocity coupling:  RHS
         "darcy-terms"
         - reacoeff * J^2 *  phi^2 *  v^f
         */
        (*force)(fk + j) += -v * reafvel(j);

        /* "reactive darcy-terms"
         reacoeff * J^2 *  phi^2 *  v^s
         */
        (*force)(fk + j) += v * reavel(j);

        /*-------structure- fluid pressure coupling: RHS
         *                        "pressure gradient terms"
         - J *  F^-T * Grad(p) * phi
         */
        (*force)(fk + j) += fac * J * Finvgradp(j) * (-porosity);

        for (int i = 0; i < numnod_; i++)
        {
          const int fi = numdim_ * i;

          for (int l = 0; l < numdim_; l++)
          {
            /* additional "reactive darcy-term"
             detJ * w(gp) * ( J^2 * reacoeff * phi^2  ) * D(v_s)
             */
            erea_v(fk + j, fi + l) += v * reatensor(j, l) * shapefct(i);

            /* additional "pressure gradient term"
             -  detJ * w(gp) * phi *  ( dJ/d(us) * F^-T * Grad(p) - J * d(F^-T)/d(us) *Grad(p) ) *
             D(us)
             - detJ * w(gp) * d(phi)/d(us) * J * F^-T * Grad(p) * D(us)
             */
            (*stiffmatrix)(fk + j, fi + l) += fac * (-porosity * dJ_dus(fi + l) * Finvgradp(j) -
                                                        porosity * J * dFinvdus_gradp(j, fi + l) -
                                                        dphi_dus(fi + l) * J * Finvgradp(j));

            /* additional "reactive darcy-term"
               detJ * w(gp) * 2 * ( dJ/d(us) * vs * reacoeff * phi^2 + J * reacoeff * phi *
             d(phi)/d(us) * vs ) * D(us)
             - detJ * w(gp) *  2 * ( J * dJ/d(us) * v^f * reacoeff * phi^2 + J * reacoeff * phi *
             d(phi)/d(us) * v^f ) * D(us)
             */
            (*stiffmatrix)(fk + j, fi + l) += fac * J * porosity * 2.0 * (reavel(j) - reafvel(j)) *
                                              (porosity * dJ_dus(fi + l) + J * dphi_dus(fi + l));

            for (int m = 0; m < numdim_; ++m)
            {
              for (int n = 0; n < numdim_; ++n)
              {
                for (int p = 0; p < numdim_; ++p)
                {
                  (*stiffmatrix)(fk + j, fi + l) +=
                      v * (velint(p) - fvelint(p)) *
                      (dFinvTdus(j * numdim_ + m, fi + l) * matreatensor(m, n) * defgrd_inv(n, p) +
                          defgrd_inv(m, j) * matreatensor(m, n) *
                              dFinvTdus(p * numdim_ + n, fi + l));
                }
              }
            }
            // check if derivatives of reaction tensor are zero --> significant speed up
            if (fluid_mat_->permeability_function() != Mat::PAR::constant)
            {
              for (int m = 0; m < numdim_; ++m)
              {
                for (int n = 0; n < numdim_; ++n)
                {
                  for (int p = 0; p < numdim_; ++p)
                  {
                    (*stiffmatrix)(fk + j, fi + l) += v * (velint(p) - fvelint(p)) *
                                                      (+defgrd_inv(m, j) *
                                                          (linreac_dphi(m, n) * dphi_dus(fi + l) +
                                                              linreac_dJ(m, n) * dJ_dus(fi + l)) *
                                                          defgrd_inv(n, p));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // inverse Right Cauchy-Green tensor as vector
  static Core::LinAlg::Matrix<numstr_, 1> C_inv_vec;
  C_inv_vec(0) = C_inv(0, 0);
  C_inv_vec(1) = C_inv(1, 1);
  C_inv_vec(2) = C_inv(0, 1);

  // B^T . C^-1
  static Core::LinAlg::Matrix<numdof_, 1> cinvb(Core::LinAlg::Initialization::zero);
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

    // integrate `geometric' stiffness matrix and add to keu *****************
    Core::LinAlg::Matrix<numstr_, 1> sfac(C_inv_vec);  // auxiliary integrated stress

    // scale and add viscous stress
    sfac.update(detJ_w, fstress, fac2);  // detJ*w(gp)*[S11,S22,S33,S12=S21,S23=S32,S13=S31]

    std::vector<double> SmB_L(2);  // intermediate Sm.B_L
    // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
    for (int inod = 0; inod < numnod_; ++inod)
    {
      SmB_L[0] = sfac(0) * N_XYZ(0, inod) + sfac(2) * N_XYZ(1, inod);
      SmB_L[1] = sfac(2) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod);
      for (int jnod = 0; jnod < numnod_; ++jnod)
      {
        double bopstrbop = 0.0;  // intermediate value
        for (int idim = 0; idim < numdim_; ++idim) bopstrbop += N_XYZ(idim, jnod) * SmB_L[idim];
        (*stiffmatrix)(numdim_* inod + 0, numdim_ * jnod + 0) += bopstrbop;
        (*stiffmatrix)(numdim_* inod + 1, numdim_ * jnod + 1) += bopstrbop;
      }
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::fill_matrix_and_vectors_pressure_based(const int& gp,
    const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& press,
    const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
    const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
    const Core::LinAlg::Matrix<1, numdof_>& dJ_dus,
    const Core::LinAlg::Matrix<numstr_, numdof_>& dCinv_dus,
    const Core::LinAlg::Matrix<1, numdof_>& dps_dus,
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix, Core::LinAlg::Matrix<numdof_, 1>* force)
{
  const double detJ_w = detJ_[gp] * intpoints_.weight(gp) * thickness_;

  // inverse Right Cauchy-Green tensor as vector
  static Core::LinAlg::Matrix<numstr_, 1> C_inv_vec;
  C_inv_vec(0) = C_inv(0, 0);
  C_inv_vec(1) = C_inv(1, 1);
  C_inv_vec(2) = C_inv(0, 1);

  // B^T . C^-1
  static Core::LinAlg::Matrix<numdof_, 1> cinvb(Core::LinAlg::Initialization::zero);
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
    sfac.scale(fac2);

    std::vector<double> SmB_L(2);  // intermediate Sm.B_L
    // kgeo += (B_L^T . sigma . B_L) * detJ * w(gp)  with B_L = Ni,Xj see NiliFEM-Skript
    for (int inod = 0; inod < numnod_; ++inod)
    {
      SmB_L[0] = sfac(0) * N_XYZ(0, inod) + sfac(2) * N_XYZ(1, inod);
      SmB_L[1] = sfac(2) * N_XYZ(0, inod) + sfac(1) * N_XYZ(1, inod);
      for (int jnod = 0; jnod < numnod_; ++jnod)
      {
        double bopstrbop = 0.0;  // intermediate value
        for (int idim = 0; idim < numdim_; ++idim) bopstrbop += N_XYZ(idim, jnod) * SmB_L[idim];
        (*stiffmatrix)(numdim_* inod + 0, numdim_ * jnod + 0) += bopstrbop;
        (*stiffmatrix)(numdim_* inod + 1, numdim_ * jnod + 1) += bopstrbop;
      }
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::fill_matrix_and_vectors_brinkman(const int& gp,
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
  const double detJ_w = detJ_[gp] * intpoints_.weight(gp) * thickness_;

  const double visc = fluid_mat_->viscosity();
  Core::LinAlg::Matrix<numdim_, numdim_> CinvFvel;
  Core::LinAlg::Matrix<numdim_, numdim_> tmp;
  CinvFvel.multiply(C_inv, fvelder);
  tmp.multiply_nt(CinvFvel, defgrd_inv);
  Core::LinAlg::Matrix<numdim_, numdim_> tmp2(tmp);
  tmp.update_t(1.0, tmp2, 1.0);

  fstress(0) = tmp(0, 0);
  fstress(1) = tmp(1, 1);
  fstress(2) = tmp(0, 1);

  fstress.scale(detJ_w * visc * J * porosity);

  // B^T . C^-1
  Core::LinAlg::Matrix<numdof_, 1> fstressb(Core::LinAlg::Initialization::zero);
  fstressb.multiply_tn(bop, fstress);

  if (force != nullptr) force->update(1.0, fstressb, 1.0);

  // evaluate viscous terms (for darcy-brinkman flow only)
  if (stiffmatrix != nullptr)
  {
    Core::LinAlg::Matrix<numdim_, numdim_> tmp4;
    tmp4.multiply_nt(fvelder, defgrd_inv);

    double fac = detJ_w * visc;

    Core::LinAlg::Matrix<numstr_, numdof_> fstress_dus(Core::LinAlg::Initialization::zero);
    for (int n = 0; n < numnod_; ++n)
    {
      for (int k = 0; k < numdim_; ++k)
      {
        const int gid = n * numdim_ + k;

        fstress_dus(0, gid) +=
            2 * (dCinv_dus(0, gid) * tmp4(0, 0) + dCinv_dus(2, gid) * tmp4(1, 0));
        fstress_dus(1, gid) +=
            2 * (dCinv_dus(2, gid) * tmp4(0, 1) + dCinv_dus(1, gid) * tmp4(1, 1));
        /* ~~~ */
        fstress_dus(2, gid) += +dCinv_dus(0, gid) * tmp4(0, 1) + dCinv_dus(2, gid) * tmp4(1, 1) +
                               dCinv_dus(2, gid) * tmp4(0, 0) + dCinv_dus(1, gid) * tmp4(1, 0);

        for (int j = 0; j < numdim_; j++)
        {
          fstress_dus(0, gid) += 2 * CinvFvel(0, j) * dFinvTdus(j * numdim_, gid);
          fstress_dus(1, gid) += 2 * CinvFvel(1, j) * dFinvTdus(j * numdim_ + 1, gid);
          /* ~~~ */
          fstress_dus(2, gid) += +CinvFvel(0, j) * dFinvTdus(j * numdim_ + 1, gid) +
                                 CinvFvel(1, j) * dFinvTdus(j * numdim_, gid);
        }
      }
    }

    Core::LinAlg::Matrix<numdof_, numdof_> tmp;

    // additional viscous fluid stress- stiffness term (B^T . fstress . dJ/d(us) * porosity * detJ *
    // w(gp))
    tmp.multiply(fac * porosity, fstressb, dJ_dus);
    stiffmatrix->update(1.0, tmp, 1.0);

    // additional fluid stress- stiffness term (B^T .  d\phi/d(us) . fstress  * J * w(gp))
    tmp.multiply(fac * J, fstressb, dphi_dus);
    stiffmatrix->update(1.0, tmp, 1.0);

    // additional fluid stress- stiffness term (B^T .  phi . dfstress/d(us)  * J * w(gp))
    tmp.multiply_tn(detJ_w * visc * J * porosity, bop, fstress_dus);
    stiffmatrix->update(1.0, tmp, 1.0);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::coupling_poroelast(
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
  get_materials();

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
  if (stiffmatrix != nullptr)
    gauss_point_loop_od(params, xrefe, xcurr, disp, vel, evelnp, epreaf, nullptr, *stiffmatrix);
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::coupling_poroelast_pressure_based(
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

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::gauss_point_loop_od(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodalvel,
    const Core::LinAlg::Matrix<numdim_, numnod_>& evelnp,
    const Core::LinAlg::Matrix<numnod_, 1>& epreaf,
    const Core::LinAlg::Matrix<numnod_, 1>* porosity_dof,
    Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>& ecoupl)
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
  Core::LinAlg::Matrix<numdim_, numnod_> N_XYZ;  //  first derivatives at gausspoint w.r.t. X, Y,Z
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  Core::LinAlg::Matrix<numdim_, numdim_> defgrd(
      Core::LinAlg::Initialization::zero);    //  deformation gradient evaluated at gauss point
  Core::LinAlg::Matrix<numnod_, 1> shapefct;  //  shape functions evaluated at gauss point
  Core::LinAlg::Matrix<numdim_, numnod_> deriv(
      Core::LinAlg::Initialization::zero);  //  first derivatives at gausspoint w.r.t. r,s,t
  // Core::LinAlg::Matrix<numdim_,1> xsi;
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
    Core::LinAlg::Matrix<numdim_, numdim_> C_inv(Core::LinAlg::Initialization::uninitialized);
    C_inv.invert(cauchygreen);

    // inverse deformation gradient F^-1
    Core::LinAlg::Matrix<numdim_, numdim_> defgrd_inv(Core::LinAlg::Initialization::uninitialized);
    defgrd_inv.invert(defgrd);

    //---------------- get pressure at integration point
    double press = shapefct.dot(epreaf);

    //------------------ get material pressure gradient at integration point
    Core::LinAlg::Matrix<numdim_, 1> Gradp;
    Gradp.multiply(N_XYZ, epreaf);

    //--------------------- get fluid velocity at integration point
    Core::LinAlg::Matrix<numdim_, 1> fvelint;
    fvelint.multiply(evelnp, shapefct);

    // material fluid velocity gradient at integration point
    Core::LinAlg::Matrix<numdim_, numdim_> fvelder;
    fvelder.multiply_nt(evelnp, N_XYZ);

    //----------------structure displacement and velocity at integration point
    Core::LinAlg::Matrix<numdim_, 1> velint(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < numnod_; i++)
      for (int j = 0; j < numdim_; j++) velint(j) += nodalvel(j, i) * shapefct(i);

    // auxiliary variables for computing the porosity and linearization
    double dphi_dp = 0.0;
    double porosity = 0.0;

    compute_porosity_and_linearization_od(
        params, press, volchange, gp, shapefct, porosity_dof, porosity, dphi_dp);

    // **********************evaluate stiffness matrix and force vector**********************
    fill_matrix_and_vectors_od(gp, shapefct, N_XYZ, J, porosity, dphi_dp, velint, fvelint,
        defgrd_inv, Gradp, bop, C_inv, ecoupl);

    if (fluid_mat_->type() == Mat::PAR::darcy_brinkman)
    {
      fill_matrix_and_vectors_brinkman_od(
          gp, shapefct, N_XYZ, J, porosity, dphi_dp, fvelder, defgrd_inv, bop, C_inv, ecoupl);
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::gauss_point_loop_od_pressure_based(
    Teuchos::ParameterList& params, const Core::LinAlg::Matrix<numdim_, numnod_>& xrefe,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp, const std::vector<double>& ephi,
    Core::LinAlg::SerialDenseMatrix& couplmat)
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
  Core::LinAlg::Matrix<numdim_, numnod_> N_XYZ;  //  first derivatives at gausspoint w.r.t. X, Y,Z
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  Core::LinAlg::Matrix<numdim_, numdim_> defgrd(
      Core::LinAlg::Initialization::zero);    //  deformation gradient evaluated at gauss point
  Core::LinAlg::Matrix<numnod_, 1> shapefct;  //  shape functions evaluated at gauss point
  Core::LinAlg::Matrix<numdim_, numnod_> deriv(
      Core::LinAlg::Initialization::zero);  //  first derivatives at gausspoint w.r.t. r,s,t

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
    Core::LinAlg::Matrix<numdim_, numdim_> C_inv(Core::LinAlg::Initialization::uninitialized);
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

    // **********************evaluate stiffness matrix and force vector**********************
    fill_matrix_and_vectors_od_pressure_based(
        gp, shapefct, N_XYZ, J, bop, C_inv, solpressderiv, couplmat);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::fill_matrix_and_vectors_od(const int& gp,
    const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& porosity,
    const double& dphi_dp, const Core::LinAlg::Matrix<numdim_, 1>& velint,
    const Core::LinAlg::Matrix<numdim_, 1>& fvelint,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
    const Core::LinAlg::Matrix<numdim_, 1>& Gradp,
    const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
    const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
    Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>& ecoupl)
{
  Core::LinAlg::Matrix<numdim_, numdim_> matreatensor(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdim_, numdim_> reatensor(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdim_, numdim_> linreac_dphi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdim_, numdim_> linreac_dJ(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdim_, 1> reafvel(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<numdim_, 1> reavel(Core::LinAlg::Initialization::zero);
  {
    Core::LinAlg::Matrix<numdim_, numdim_> temp(Core::LinAlg::Initialization::zero);
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

  const double detJ_w = detJ_[gp] * intpoints_.weight(gp) * thickness_;

  // inverse Right Cauchy-Green tensor as vector
  Core::LinAlg::Matrix<numstr_, 1> C_inv_vec;
  C_inv_vec(0) = C_inv(0, 0);
  C_inv_vec(1) = C_inv(1, 1);
  C_inv_vec(2) = C_inv(0, 1);

  // B^T . C^-1
  Core::LinAlg::Matrix<numdof_, 1> cinvb(Core::LinAlg::Initialization::zero);
  cinvb.multiply_tn(bop, C_inv_vec);

  // F^-T * Grad p
  Core::LinAlg::Matrix<numdim_, 1> Finvgradp;
  Finvgradp.multiply_tn(defgrd_inv, Gradp);

  // F^-T * N_XYZ
  Core::LinAlg::Matrix<numdim_, numnod_> FinvNXYZ;
  FinvNXYZ.multiply_tn(defgrd_inv, N_XYZ);

  {
    for (int i = 0; i < numnod_; i++)
    {
      const int fi = numdim_ * i;
      const double fac = detJ_w * shapefct(i);

      for (int j = 0; j < numdim_; j++)
      {
        for (int k = 0; k < numnod_; k++)
        {
          const int fk = (numdim_ + 1) * k;
          const int fk_press = fk + numdim_;

          /*-------structure- fluid pressure coupling: "stress terms" + "pressure gradient terms"
           -B^T . ( -1*J*C^-1 ) * Dp
           - J * F^-T * Grad(p) * dphi/dp * Dp - J * F^-T * d(Grad((p))/(dp) * phi * Dp
           */
          ecoupl(fi + j, fk_press) +=
              detJ_w * cinvb(fi + j) * (-1.0) * J * shapefct(k) -
              fac * J * (dphi_dp * Finvgradp(j) * shapefct(k) + porosity * FinvNXYZ(j, k));

          /*-------structure- fluid pressure coupling:  "darcy-terms" + "reactive darcy-terms"
           - 2 * reacoeff * J * v^f * phi * d(phi)/dp  Dp
           + 2 * reacoeff * J * v^s * phi * d(phi)/dp  Dp
           + J * J * phi * phi * defgrd_^-T * d(mat_reacoeff)/d(phi) * defgrd_^-1 * (v^s-v^f) *
           d(phi)/dp Dp
           */
          const double tmp = fac * J * J * 2 * porosity * dphi_dp * shapefct(k);
          ecoupl(fi + j, fk_press) += -tmp * reafvel(j);

          ecoupl(fi + j, fk_press) += tmp * reavel(j);

          // check if derivatives of reaction tensor are zero --> significant speed up
          if (fluid_mat_->permeability_function() != Mat::PAR::constant)
          {
            const double tmp2 = 0.5 * tmp * porosity;
            for (int m = 0; m < numdim_; ++m)
            {
              for (int n = 0; n < numdim_; ++n)
              {
                for (int p = 0; p < numdim_; ++p)
                {
                  ecoupl(fi + j, fk_press) += tmp2 * defgrd_inv(m, j) * linreac_dphi(m, n) *
                                              defgrd_inv(n, p) * (velint(p) - fvelint(p));
                }
              }
            }
          }

          /*-------structure- fluid velocity coupling:  "darcy-terms"
           -reacoeff * J * J *  phi^2 *  Dv^f
           */
          const double v = fac * J * J * porosity * porosity;
          for (int l = 0; l < numdim_; l++)
            ecoupl(fi + j, fk + l) += -v * reatensor(j, l) * shapefct(k);
        }
      }
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::fill_matrix_and_vectors_od_pressure_based(const int& gp,
    const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J,
    const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
    const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv, const std::vector<double>& solpressderiv,
    Core::LinAlg::SerialDenseMatrix& couplmat)
{
  const double detJ_w = detJ_[gp] * intpoints_.weight(gp) * thickness_;

  // inverse Right Cauchy-Green tensor as vector
  Core::LinAlg::Matrix<numstr_, 1> C_inv_vec;
  C_inv_vec(0) = C_inv(0, 0);
  C_inv_vec(1) = C_inv(1, 1);
  C_inv_vec(2) = C_inv(0, 1);

  // B^T . C^-1
  Core::LinAlg::Matrix<numdof_, 1> cinvb(Core::LinAlg::Initialization::zero);
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

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::fill_matrix_and_vectors_brinkman_od(const int& gp,
    const Core::LinAlg::Matrix<numnod_, 1>& shapefct,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ, const double& J, const double& porosity,
    const double& dphi_dp, const Core::LinAlg::Matrix<numdim_, numdim_>& fvelder,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv,
    const Core::LinAlg::Matrix<numstr_, numdof_>& bop,
    const Core::LinAlg::Matrix<numdim_, numdim_>& C_inv,
    Core::LinAlg::Matrix<numdof_, (numdim_ + 1) * numnod_>& ecoupl)
{
  const double detJ_w = detJ_[gp] * intpoints_.weight(gp) * thickness_;
  const double visc = fluid_mat_->viscosity();

  Core::LinAlg::Matrix<numstr_, 1> fstress;

  Core::LinAlg::Matrix<numdim_, numdim_> CinvFvel;
  Core::LinAlg::Matrix<numdim_, numdim_> tmp;
  CinvFvel.multiply(C_inv, fvelder);
  tmp.multiply_nt(CinvFvel, defgrd_inv);
  Core::LinAlg::Matrix<numdim_, numdim_> tmp2(tmp);
  tmp.update_t(1.0, tmp2, 1.0);

  fstress(0) = tmp(0, 0);
  fstress(1) = tmp(1, 1);
  fstress(2) = tmp(0, 1);

  // B^T . \sigma
  Core::LinAlg::Matrix<numdof_, 1> fstressb;
  fstressb.multiply_tn(bop, fstress);
  Core::LinAlg::Matrix<numdim_, numnod_> N_XYZ_Finv;
  N_XYZ_Finv.multiply(defgrd_inv, N_XYZ);

  // dfstress/dv^f
  Core::LinAlg::Matrix<numstr_, numdof_> dfstressb_dv;
  for (int i = 0; i < numnod_; i++)
  {
    const int fi = numdim_ * i;
    for (int j = 0; j < numdim_; j++)
    {
      int k = fi + j;
      dfstressb_dv(0, k) = 2 * N_XYZ_Finv(0, i) * C_inv(0, j);
      dfstressb_dv(1, k) = 2 * N_XYZ_Finv(1, i) * C_inv(1, j);

      dfstressb_dv(2, k) = N_XYZ_Finv(0, i) * C_inv(1, j) + N_XYZ_Finv(1, i) * C_inv(0, j);
    }
  }

  // B^T . dfstress/dv^f
  Core::LinAlg::Matrix<numdof_, numdof_> dfstressb_dv_bop(Core::LinAlg::Initialization::zero);
  dfstressb_dv_bop.multiply_tn(bop, dfstressb_dv);

  for (int i = 0; i < numnod_; i++)
  {
    const int fi = numdim_ * i;

    for (int j = 0; j < numdim_; j++)
    {
      for (int k = 0; k < numnod_; k++)
      {
        const int fk_sub = numdim_ * k;
        const int fk = (numdim_ + 1) * k;
        const int fk_press = fk + numdim_;

        /*-------structure- fluid pressure coupling: "darcy-brinkman stress terms"
         B^T . ( \mu*J - d(phi)/(dp) * fstress ) * Dp
         */
        ecoupl(fi + j, fk_press) += detJ_w * fstressb(fi + j) * dphi_dp * visc * J * shapefct(k);
        for (int l = 0; l < numdim_; l++)
        {
          /*-------structure- fluid velocity coupling: "darcy-brinkman stress terms"
           B^T . ( \mu*J - phi * dfstress/dv^f ) * Dp
           */
          ecoupl(fi + j, fk + l) +=
              detJ_w * visc * J * porosity * dfstressb_dv_bop(fi + j, fk_sub + l);
        }
      }
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::init_element()
{
  Core::LinAlg::Matrix<numdim_, numnod_> deriv;
  Core::LinAlg::Matrix<numnod_, numdim_> xrefe;
  for (int i = 0; i < numnod_; ++i)
  {
    for (int j = 0; j < numdim_; ++j) xrefe(i, j) = nodes()[i]->x()[j];
  }
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
  }

  if (distype != Core::FE::CellType::nurbs4 and distype != Core::FE::CellType::nurbs9)
  {
    for (int gp = 0; gp < numgpt_; ++gp)
    {
      Core::FE::shape_function_deriv1<distype>(xsi_[gp], deriv);

      invJ_[gp].multiply(deriv, xrefe);
      detJ_[gp] = invJ_[gp].invert();
      if (detJ_[gp] <= 0.0) FOUR_C_THROW("Element Jacobian mapping {:10.5e} <= 0.0", detJ_[gp]);
    }
  }

  scatra_coupling_ = false;

  Core::ProblemType probtype = Global::Problem::instance()->get_problem_type();
  if (probtype == Core::ProblemType::poroscatra) scatra_coupling_ = true;

  init_ = true;
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<
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

  if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)  // total lagrange (nonlinear)
  {
    // for nonlinear kinematics the Jacobian of the deformation gradient is the volume change
    volchange = J;
    dvolchange_dus = dJ_dus;
  }
  else if (kintype_ == Inpar::Solid::KinemType::linear)  // linear kinematics
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

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::compute_jacobian_determinant_volume_change(double& J,
    double& volchange, const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
    const Core::LinAlg::Matrix<numdim_, numnod_>& nodaldisp)
{
  // compute J
  J = defgrd.determinant();

  if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)  // total lagrange (nonlinear)
  {
    // for nonlinear kinematics the Jacobian of the deformation gradient is the volume change
    volchange = J;
  }
  else if (kintype_ == Inpar::Solid::KinemType::linear)  // linear kinematics
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

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::p_k2to_cauchy(
    Core::LinAlg::Matrix<Wall1::numstr_, 1>& stress, Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
    Core::LinAlg::Matrix<numdim_, numdim_>& cauchystress)
{
  // calculate the Jacobi-deterinant
  const double detF = (defgrd).determinant();

  // sigma = 1/J . F . S . F^T
  Core::LinAlg::Matrix<numdim_, numdim_> pkstress;
  pkstress(0, 0) = (stress)(0);
  pkstress(0, 1) = (stress)(2);
  pkstress(1, 0) = pkstress(0, 1);
  pkstress(1, 1) = (stress)(1);

  Core::LinAlg::Matrix<numdim_, numdim_> temp;
  temp.multiply((1.0 / detF), (defgrd), pkstress);
  (cauchystress).multiply_nt(temp, (defgrd));
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::compute_def_gradient(
    Core::LinAlg::Matrix<numdim_, numdim_>& defgrd,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr)
{
  if (kintype_ == Inpar::Solid::KinemType::nonlinearTotLag)
  {
    // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
    defgrd.multiply_nt(xcurr, N_XYZ);  //  (6.17)
  }
  else if (kintype_ == Inpar::Solid::KinemType::linear)
  {
    defgrd.clear();
    for (int i = 0; i < numdim_; i++) defgrd(i, i) = 1.0;
  }
  else
    FOUR_C_THROW("invalid kinematic type!");
}

template <Core::FE::CellType distype>
inline void Discret::Elements::Wall1Poro<distype>::compute_b_operator(
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
    bop(1, noddof_ * i + 0) = defgrd(0, 1) * N_XYZ(1, i);
    bop(1, noddof_ * i + 1) = defgrd(1, 1) * N_XYZ(1, i);
    /* ~~~ */
    bop(2, noddof_ * i + 0) = defgrd(0, 0) * N_XYZ(1, i) + defgrd(0, 1) * N_XYZ(0, i);
    bop(2, noddof_ * i + 1) = defgrd(1, 0) * N_XYZ(1, i) + defgrd(1, 1) * N_XYZ(0, i);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::compute_shape_functions_and_derivatives(const int& gp,
    Core::LinAlg::Matrix<numnod_, 1>& shapefct, Core::LinAlg::Matrix<numdim_, numnod_>& deriv,
    Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ)
{
  // get values of shape functions and derivatives in the gausspoint
  if (distype != Core::FE::CellType::nurbs4 and distype != Core::FE::CellType::nurbs9)
  {
    // shape functions and their derivatives for polynomials
    Core::FE::shape_function<distype>(xsi_[gp], shapefct);
    Core::FE::shape_function_deriv1<distype>(xsi_[gp], deriv);
  }
  else
  {
    // nurbs version
    Core::FE::Nurbs::nurbs_get_funct_deriv(shapefct, deriv, xsi_[gp], myknots_, weights_, distype);

    Core::LinAlg::Matrix<numnod_, numdim_> xrefe;
    for (int i = 0; i < numnod_; ++i)
    {
      xrefe(i, 0) = nodes()[i]->x()[0];
      xrefe(i, 1) = nodes()[i]->x()[1];
    }
    invJ_[gp].multiply(deriv, xrefe);
    detJ_[gp] = invJ_[gp].invert();
    if (detJ_[gp] <= 0.0) FOUR_C_THROW("Element Jacobian mapping {:10.5e} <= 0.0", detJ_[gp]);
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

template <Core::FE::CellType distype>
double Discret::Elements::Wall1Poro<distype>::compute_jacobian_determinant(const int& gp,
    const Core::LinAlg::Matrix<numdim_, numnod_>& xcurr,
    const Core::LinAlg::Matrix<numdim_, numnod_>& deriv)
{
  // get Jacobian matrix and determinant w.r.t. spatial configuration
  // transposed jacobian "dx/ds"
  Core::LinAlg::Matrix<numdim_, numdim_> xjm;
  // inverse of transposed jacobian "ds/dx"
  Core::LinAlg::Matrix<numdim_, numdim_> xji;
  xjm.multiply_nt(deriv, xcurr);
  const double det = xji.invert(xjm);

  // determinant of deformationgradient: det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1
  const double J = det / detJ_[gp];

  return J;
}

template <Core::FE::CellType distype>
inline void Discret::Elements::Wall1Poro<distype>::compute_linearization_of_jacobian(
    Core::LinAlg::Matrix<1, numdof_>& dJ_dus, const double& J,
    const Core::LinAlg::Matrix<numdim_, numnod_>& N_XYZ,
    const Core::LinAlg::Matrix<numdim_, numdim_>& defgrd_inv)
{
  //--------------------------- build N_X operator (wrt material config)
  Core::LinAlg::Matrix<numdim_ * numdim_, numdof_> N_X(
      Core::LinAlg::Initialization::zero);  // set to zero
  for (int i = 0; i < numnod_; ++i)
  {
    N_X(0, numdim_ * i + 0) = N_XYZ(0, i);
    N_X(1, numdim_ * i + 1) = N_XYZ(0, i);

    N_X(2, numdim_ * i + 0) = N_XYZ(1, i);
    N_X(3, numdim_ * i + 1) = N_XYZ(1, i);
  }

  //------------------------------------ build F^-1 as vector 4x1
  Core::LinAlg::Matrix<numdim_ * numdim_, 1> defgrd_inv_vec;
  defgrd_inv_vec(0) = defgrd_inv(0, 0);
  defgrd_inv_vec(1) = defgrd_inv(0, 1);
  defgrd_inv_vec(2) = defgrd_inv(1, 0);
  defgrd_inv_vec(3) = defgrd_inv(1, 1);

  //------linearization of jacobi determinant detF=J w.r.t. structure displacement   dJ/d(us) =
  // dJ/dF : dF/dus = J * F^-T * N,X
  dJ_dus.multiply_tn(J, defgrd_inv_vec, N_X);
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::compute_auxiliary_values(
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

  if (kintype_ != Inpar::Solid::KinemType::linear)
  {
    // dF^-T/dus
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

  for (int n = 0; n < numnod_; ++n)
  {
    for (int k = 0; k < numdim_; ++k)
    {
      const int gid = n * numdim_ + k;
      for (int i = 0; i < numdim_; ++i)
      {
        dCinv_dus(0, gid) += -2 * C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(0, k);
        dCinv_dus(1, gid) += -2 * C_inv(1, i) * N_XYZ(i, n) * defgrd_inv(1, k);
        /* ~~~ */
        dCinv_dus(2, gid) += -C_inv(0, i) * N_XYZ(i, n) * defgrd_inv(1, k) -
                             defgrd_inv(0, k) * N_XYZ(i, n) * C_inv(1, i);
      }
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::compute_porosity_and_linearization(
    Teuchos::ParameterList& params, const double& press, const double& J, const int& gp,
    const Core::LinAlg::Matrix<numnod_, 1>& shapfct,
    const Core::LinAlg::Matrix<numnod_, 1>* myporosity,
    const Core::LinAlg::Matrix<1, numdof_>& dJ_dus, double& porosity,
    Core::LinAlg::Matrix<1, numdof_>& dphi_dus)
{
  double dphi_dJ = 0.0;

  struct_mat_->compute_porosity(params, press, J, gp, porosity, nullptr, &dphi_dJ, nullptr, nullptr,
      nullptr  // dphi_dpp not needed
  );

  // linearization of porosity w.r.t structure displacement d\phi/d(us) = d\phi/dJ*dJ/d(us)
  dphi_dus.update(dphi_dJ, dJ_dus);
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::compute_porosity_and_linearization_od(
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

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::compute_sol_pressure_deriv(
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
    inverse.setMatrix(Teuchos::rcpFromRef(pressderiv.base()));
    int err = inverse.invert();
    if (err != 0)
      FOUR_C_THROW("Inversion of matrix for pressure derivative failed with error code {}.", err);
  }

  // calculate derivatives of saturation w.r.t. pressure
  fluidmulti_mat_->evaluate_deriv_of_saturation_wrt_pressure(helpderiv, press);

  // chain rule: the derivative of saturation w.r.t. dof =
  // (derivative of saturation w.r.t. pressure) * (derivative of pressure w.r.t. dof)
  Core::LinAlg::multiply(satderiv, helpderiv, pressderiv);

  // compute derivative of solid pressure w.r.t. dofs with product rule
  for (int iphase = 0; iphase < numfluidphases; iphase++)
  {
    for (int jphase = 0; jphase < numfluidphases; jphase++)
      solidpressderiv[iphase] +=
          pressderiv(jphase, iphase) * sat[jphase] + satderiv(jphase, iphase) * press[jphase];
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::compute_linearization_of_sol_press_wrt_disp(
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

/*----------------------------------------------------------------------*
 * derivative of sol. pres. at GP for multiphase flow   kremheller 10/17|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::recalculate_sol_pressure_deriv(
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

template <Core::FE::CellType distype>
double Discret::Elements::Wall1Poro<distype>::compute_sol_pressure_at_gp(
    const int totalnumdofpernode, const int numfluidphases, const std::vector<double>& phiAtGP)
{
  // initialize auxiliary variables
  std::vector<double> genpress(numfluidphases, 0.0);
  std::vector<double> sat(numfluidphases, 0.0);
  std::vector<double> press(numfluidphases, 0.0);
  std::vector<double> fluidphi(phiAtGP.data(), phiAtGP.data() + numfluidphases);

  // evaluate the pressures
  fluidmulti_mat_->evaluate_gen_pressure(genpress, fluidphi);

  // transform generalized pressures to true pressure values
  fluidmulti_mat_->transform_gen_pres_to_true_pres(genpress, press);

  // explicit evaluation of saturation
  fluidmulti_mat_->evaluate_saturation(sat, fluidphi, press);

  // solid pressure = sum (S_i*p_i)
  const double solidpressure = std::inner_product(sat.begin(), sat.end(), press.begin(), 0.0);

  return solidpressure;
}

template <Core::FE::CellType distype>
double Discret::Elements::Wall1Poro<distype>::recalculate_sol_pressure_at_gp(double press,
    const double porosity, const int totalnumdofpernode, const int numfluidphases,
    const int numvolfrac, const std::vector<double>& phiAtGP)
{
  // get volume fraction primary variables at [numfluidphases-1...numfluidphase-1+numvolfrac]
  std::vector<double> volfracphi(
      phiAtGP.data() + numfluidphases, phiAtGP.data() + numfluidphases + numvolfrac);
  double sumaddvolfrac = 0.0;
  for (int ivolfrac = 0; ivolfrac < numvolfrac; ivolfrac++) sumaddvolfrac += volfracphi[ivolfrac];

  // p_s = (porosity - sumaddvolfrac)/porosity * fluidpress
  //      + 1.0 / porosity sum_i=1^numvolfrac (volfrac_i*pressure_i)
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

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::compute_primary_variable_at_gp(
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

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::extract_values_from_global_vector(
    const Core::FE::Discretization& discretization, const int& dofset, const std::vector<int>& lm,
    Core::LinAlg::Matrix<numdim_, numnod_>* matrixtofill,
    Core::LinAlg::Matrix<numnod_, 1>* vectortofill, const std::string state)
{
  // put on higher level
  // get state of the global vector
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
      discretization.get_state(dofset, state);
  if (matrix_state == nullptr) FOUR_C_THROW("Cannot get state vector {}", state);

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

template <Core::FE::CellType distype>
std::vector<double>
Discret::Elements::Wall1Poro<distype>::compute_anisotropic_permeability_coeffs_at_gp(
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

template class Discret::Elements::Wall1Poro<Core::FE::CellType::tri3>;
template class Discret::Elements::Wall1Poro<Core::FE::CellType::quad4>;
template class Discret::Elements::Wall1Poro<Core::FE::CellType::quad9>;
template class Discret::Elements::Wall1Poro<Core::FE::CellType::nurbs4>;
template class Discret::Elements::Wall1Poro<Core::FE::CellType::nurbs9>;

FOUR_C_NAMESPACE_CLOSE
