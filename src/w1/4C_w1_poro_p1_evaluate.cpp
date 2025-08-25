// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"
#include "4C_utils_function.hpp"
#include "4C_w1_poro_p1.hpp"
#include "4C_w1_poro_p1_eletypes.hpp"

FOUR_C_NAMESPACE_OPEN

template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroP1<distype>::compute_porosity_and_linearization(
    const Core::LinAlg::Matrix<Base::numnod_, 1>& shapfct,
    const Core::LinAlg::Matrix<Base::numnod_, 1>* myporosity, double& porosity,
    Core::LinAlg::Matrix<1, Base::numdof_>& dphi_dus)
{
  if (myporosity == nullptr)
    FOUR_C_THROW("no porosity values given!");
  else
    porosity = shapfct.dot(*myporosity);

  dphi_dus.put_scalar(0.0);
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroP1<distype>::compute_porosity_and_linearization_od(
    Teuchos::ParameterList& params, const double& press, const double& J, const int& gp,
    const Core::LinAlg::Matrix<Base::numnod_, 1>& shapfct,
    const Core::LinAlg::Matrix<Base::numnod_, 1>* myporosity, double& porosity, double& dphi_dp)
{
  dphi_dp = 0.0;

  if (myporosity == nullptr)
    FOUR_C_THROW("no porosity values given!");
  else
    porosity = shapfct.dot(*myporosity);
}

template <Core::FE::CellType distype>
int Discret::Elements::Wall1PoroP1<distype>::evaluate(Teuchos::ParameterList& params,
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
    else if (action == "struct_poro_calc_fluidcoupling")
      act = Core::Elements::struct_poro_calc_fluidcoupling;
    else if (action == "calc_struct_energy")
      act = Core::Elements::struct_calc_energy;
  }

  // what should the element do
  switch (act)
  {
    //==================================================================================
    // off diagonal terms in stiffness matrix for monolithic coupling
    case Core::Elements::struct_poro_calc_fluidcoupling:
    {
      // in some cases we need to write/change some data before evaluating
      Base::pre_evaluate(params, discretization, la);

      my_evaluate(params, discretization, la, elemat1, elemat2, elevec1, elevec2, elevec3);
    }
    break;
    //==================================================================================
    case Core::Elements::struct_calc_energy:
    {
      // in some cases we need to write/change some data before evaluating
      Base::pre_evaluate(params, discretization, la);

      // evaluate parent solid element
      Wall1::evaluate(
          params, discretization, la[0].lm_, elemat1, elemat2, elevec1, elevec2, elevec3);
    }
    break;
    //==================================================================================
    default:
    {
      // in some cases we need to write/change some data before evaluating
      Base::pre_evaluate(params, discretization, la);

      Core::LinAlg::SerialDenseMatrix elemat1_sub;
      Core::LinAlg::SerialDenseMatrix elemat2_sub;
      Core::LinAlg::SerialDenseVector elevec1_sub;
      Core::LinAlg::SerialDenseVector elevec2_sub;
      Core::LinAlg::SerialDenseVector elevec3_sub;

      if (elemat1.values()) elemat1_sub.shape(Base::numdof_, Base::numdof_);
      if (elemat2.values()) elemat2_sub.shape(Base::numdof_, Base::numdof_);
      if (elevec1.values()) elevec1_sub.resize(Base::numdof_);
      if (elevec2.values()) elevec2_sub.resize(Base::numdof_);
      if (elevec3.values()) elevec3_sub.resize(Base::numdof_);

      std::vector<int> lm_sub;
      for (int i = 0; i < Base::numnod_; i++)
        for (int j = 0; j < Base::numdim_; j++) lm_sub.push_back(la[0].lm_[i * noddof_ + j]);

      // evaluate parent solid element
      Wall1::evaluate(params, discretization, lm_sub, elemat1_sub, elemat2_sub, elevec1_sub,
          elevec2_sub, elevec3_sub);


      if (elemat1.values())
      {
        for (int i = 0; i < Base::numnod_; i++)
        {
          for (int j = 0; j < Base::numdim_; j++)
          {
            for (int k = 0; k < Base::numnod_; k++)
            {
              for (int l = 0; l < Base::numdim_; l++)
                elemat1(i * noddof_ + j, k * noddof_ + l) =
                    elemat1_sub(i * Base::noddof_ + j, k * Base::noddof_ + l);
            }
          }
        }
      }

      if (elemat2.values())
      {
        for (int i = 0; i < Base::numnod_; i++)
        {
          for (int j = 0; j < Base::numdim_; j++)
          {
            for (int k = 0; k < Base::numnod_; k++)
            {
              for (int l = 0; l < Base::numdim_; l++)
                elemat2(i * noddof_ + j, k * noddof_ + l) =
                    elemat2_sub(i * Base::noddof_ + j, k * Base::noddof_ + l);
            }
          }
        }
      }

      if (elevec1.values())
      {
        for (int i = 0; i < Base::numnod_; i++)
          for (int j = 0; j < Base::numdim_; j++)
            elevec1(i * noddof_ + j) = elevec1_sub(i * Base::noddof_ + j);
      }

      if (elevec2.values())
      {
        for (int i = 0; i < Base::numnod_; i++)
          for (int j = 0; j < Base::numdim_; j++)
            elevec2(i * noddof_ + j) = elevec2_sub(i * Base::noddof_ + j);
      }

      if (elevec3.values())
      {
        for (int i = 0; i < Base::numnod_; i++)
          for (int j = 0; j < Base::numdim_; j++)
            elevec3(i * noddof_ + j) = elevec3_sub(i * Base::noddof_ + j);
      }

      // add volume coupling specific terms
      my_evaluate(params, discretization, la, elemat1, elemat2, elevec1, elevec2, elevec3);
    }
    break;
  }

  return 0;
}

template <Core::FE::CellType distype>
int Discret::Elements::Wall1PoroP1<distype>::my_evaluate(Teuchos::ParameterList& params,
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
  }

  // what should the element do
  switch (act)
  {
    //==================================================================================
    // nonlinear stiffness, damping and internal force vector for poroelasticity
    case Core::Elements::struct_calc_nlnstiff:
    case Core::Elements::struct_calc_nlnstiffmass:
    {
      if (la.size() > 1)
      {
        // stiffness
        Core::LinAlg::Matrix<numdof_, numdof_> elemat_1(elemat1.values(), true);
        // damping
        Core::LinAlg::Matrix<numdof_, numdof_> elemat_2(elemat2.values(), true);
        // internal force vector
        Core::LinAlg::Matrix<numdof_, 1> elevec_1(elevec1.values(), true);
        // elevec2+3 are not used anyway

        // build the location vector only for the structure field
        std::vector<int> lm = la[0].lm_;

        Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> mydisp(
            Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<Base::numnod_, 1> myporosity(Core::LinAlg::Initialization::zero);
        Base::extract_values_from_global_vector(
            discretization, 0, la[0].lm_, &mydisp, &myporosity, "displacement");

        Core::LinAlg::Matrix<numdof_, numdof_>* matptr = nullptr;
        if (elemat_1.is_initialized()) matptr = &elemat_1;

        enum Inpar::Solid::DampKind damping =
            params.get<enum Inpar::Solid::DampKind>("damping", Inpar::Solid::damp_none);
        Core::LinAlg::Matrix<numdof_, numdof_>* matptr2 = nullptr;
        if (elemat_2.is_initialized() and (damping == Inpar::Solid::damp_material))
          matptr2 = &elemat_2;

        // need current fluid state,
        // call the fluid discretization: fluid equates 2nd dofset
        // disassemble velocities and pressures

        Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> myvel(
            Core::LinAlg::Initialization::zero);

        Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> myfluidvel(
            Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<Base::numnod_, 1> myepreaf(Core::LinAlg::Initialization::zero);

        if (discretization.has_state(0, "velocity"))
          Base::extract_values_from_global_vector(
              discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

        if (discretization.has_state(1, "fluidvel"))
        {
          // extract local values of the global vectors
          Base::extract_values_from_global_vector(
              discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");
        }

        // calculate tangent stiffness matrix
        nonlinear_stiffness_poroelast(lm, mydisp, myvel, &myporosity, myfluidvel, myepreaf, matptr,
            matptr2, &elevec_1, params);
      }
    }
    break;

    //==================================================================================
    // coupling terms in force-vector and stiffness matrix for poroelasticity
    case Core::Elements::struct_poro_calc_fluidcoupling:
    {
      // stiffness
      Core::LinAlg::Matrix<numdof_, (Base::numdim_ + 1) * Base::numnod_> elemat_1(
          elemat1.values(), true);

      // elemat2,elevec1-3 are not used anyway

      // build the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      Core::LinAlg::Matrix<numdof_, (Base::numdim_ + 1) * Base::numnod_>* matptr = nullptr;
      if (elemat_1.is_initialized()) matptr = &elemat_1;

      // need current fluid state,
      // call the fluid discretization: fluid equates 2nd dofset
      // disassemble velocities and pressures
      if (discretization.has_state(1, "fluidvel"))
      {
        Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> myvel(
            Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> myfluidvel(
            Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<Base::numnod_, 1> myepreaf(Core::LinAlg::Initialization::zero);

        Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> mydisp(
            Core::LinAlg::Initialization::zero);
        Core::LinAlg::Matrix<Base::numnod_, 1> myporosity(Core::LinAlg::Initialization::zero);
        Base::extract_values_from_global_vector(
            discretization, 0, la[0].lm_, &mydisp, &myporosity, "displacement");

        if (discretization.has_state(0, "velocity"))
          Base::extract_values_from_global_vector(
              discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

        if (discretization.has_state(1, "fluidvel"))
        {
          // extract local values of the global vectors
          Base::extract_values_from_global_vector(
              discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");
        }

        coupling_poroelast(lm, mydisp, myvel, &myporosity, myfluidvel, myepreaf,
            matptr,  // nullptr,
            nullptr, nullptr, params);
      }
    }
    break;

    //==================================================================================
    // nonlinear stiffness and internal force vector for poroelasticity
    case Core::Elements::struct_calc_internalforce:
    {
      // internal force vector
      Core::LinAlg::Matrix<numdof_, 1> elevec_1(elevec1.values(), true);
      // elemat2,elevec2+3 are not used anyway

      // build the location vector only for the structure field
      std::vector<int> lm = la[0].lm_;

      Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> mydisp(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<Base::numnod_, 1> myporosity(Core::LinAlg::Initialization::zero);
      Base::extract_values_from_global_vector(
          discretization, 0, la[0].lm_, &mydisp, &myporosity, "displacement");

      Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> myvel(Core::LinAlg::Initialization::zero);

      Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> myfluidvel(
          Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<Base::numnod_, 1> myepreaf(Core::LinAlg::Initialization::zero);

      // need current fluid state,
      // call the fluid discretization: fluid equates 2nd dofset
      // disassemble velocities and pressures
      if (discretization.has_state(1, "fluidvel"))
      {
        // extract local values of the global vectors
        Base::extract_values_from_global_vector(
            discretization, 1, la[1].lm_, &myfluidvel, &myepreaf, "fluidvel");

        Base::extract_values_from_global_vector(
            discretization, 0, la[0].lm_, &myvel, nullptr, "velocity");

        nonlinear_stiffness_poroelast(lm, mydisp, myvel, &myporosity, myfluidvel, myepreaf, nullptr,
            nullptr, &elevec_1, params);
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

template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroP1<distype>::init_element()
{
  // initialize base element
  Base::init_element();
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroP1<distype>::nonlinear_stiffness_poroelast(std::vector<int>& lm,
    Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& disp,
    Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& vel,
    Core::LinAlg::Matrix<Base::numnod_, 1>* porosity_dof,
    Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& evelnp,
    Core::LinAlg::Matrix<Base::numnod_, 1>& epreaf,
    Core::LinAlg::Matrix<numdof_, numdof_>* stiffmatrix,
    Core::LinAlg::Matrix<numdof_, numdof_>* reamatrix, Core::LinAlg::Matrix<numdof_, 1>* force,
    Teuchos::ParameterList& params)
{
  Base::get_materials();

  // update element geometry
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> xrefe;  // material coord. of element
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> xcurr;  // current  coord. of element

  Core::Nodes::Node** nodes = Base::nodes();
  for (int i = 0; i < Base::numnod_; ++i)
  {
    const auto& x = nodes[i]->x();
    for (int j = 0; j < Base::numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }

  // initialize element matrizes and vectors
  Core::LinAlg::Matrix<Base::numdof_, Base::numdof_> erea_v(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Base::numdof_, Base::numdof_> sub_stiff(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Base::numdof_, 1> sub_force(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<Base::numdof_, Base::numnod_> ecoupl_p1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Base::numnod_, numdof_> estiff_p1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Base::numnod_, 1> ecoupl_force_p1(Core::LinAlg::Initialization::zero);

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  gauss_point_loop_p1(params, xrefe, xcurr, disp, vel, evelnp, epreaf, porosity_dof, erea_v,
      &sub_stiff, &sub_force, ecoupl_p1, estiff_p1, ecoupl_force_p1);

  // update stiffness matrix
  if (stiffmatrix != nullptr)
  {
    if (reamatrix != nullptr)
    {
      /* additional "reactive darcy-term"
       detJ * w(gp) * ( J * reacoeff * phi^2  ) * D(v_s)
       */
      for (int k = 0; k < Base::numnod_; k++)
      {
        for (int l = 0; l < Base::numdim_; l++)
        {
          for (int i = 0; i < Base::numnod_; i++)
          {
            for (int j = 0; j < Base::numdim_; j++)
              (*reamatrix)(i* noddof_ + j, k * noddof_ + l) +=
                  erea_v(i * Base::numdim_ + j, k * Base::numdim_ + l);
          }
        }
      }
    }

    for (int k = 0; k < Base::numnod_; k++)
    {
      for (int l = 0; l < Base::numdim_; l++)
      {
        for (int i = 0; i < Base::numnod_; i++)
        {
          for (int j = 0; j < Base::numdim_; j++)
            (*stiffmatrix)(i* noddof_ + j, k * noddof_ + l) +=
                sub_stiff(i * Base::numdim_ + j, k * Base::numdim_ + l);
        }
      }
      for (int i = 0; i < Base::numnod_; i++)
      {
        for (int j = 0; j < Base::numdim_; j++)
          (*stiffmatrix)(i* noddof_ + j, k * noddof_ + Base::numdim_) +=
              ecoupl_p1(i * Base::noddof_ + j, k);
      }
    }

    for (int i = 0; i < Base::numnod_; i++)
    {
      for (int j = 0; j < Base::numnod_; j++)
      {
        for (int k = 0; k < noddof_; k++)
          (*stiffmatrix)(i* noddof_ + Base::numdim_, j * noddof_ + k) +=
              estiff_p1(i, j * noddof_ + k);
      }
    }
  }

  // update internal force vector
  if (force != nullptr)
  {
    for (int i = 0; i < Base::numnod_; i++)
    {
      for (int j = 0; j < Base::numdim_; j++)
        (*force)(i* noddof_ + j) += sub_force(i * Base::numdim_ + j);

      (*force)(i* noddof_ + Base::numdim_) += ecoupl_force_p1(i);
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroP1<distype>::gauss_point_loop_p1(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& xrefe,
    const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& xcurr,
    const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& nodaldisp,
    const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& nodalvel,
    const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& evelnp,
    const Core::LinAlg::Matrix<Base::numnod_, 1>& epreaf,
    const Core::LinAlg::Matrix<Base::numnod_, 1>* porosity_dof,
    Core::LinAlg::Matrix<Base::numdof_, Base::numdof_>& erea_v,
    Core::LinAlg::Matrix<Base::numdof_, Base::numdof_>* sub_stiff,
    Core::LinAlg::Matrix<Base::numdof_, 1>* sub_force,
    Core::LinAlg::Matrix<Base::numdof_, Base::numnod_>& ecoupl_p1,
    Core::LinAlg::Matrix<Base::numnod_, numdof_>& estiff_p1,
    Core::LinAlg::Matrix<Base::numnod_, 1>& ecoupl_force_p1)
{
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> N_XYZ;
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> defgrd(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Base::numnod_, 1> shapefct;
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> deriv;

  Core::LinAlg::Matrix<Base::numstr_, 1> fstress(Core::LinAlg::Initialization::zero);

  for (int gp = 0; gp < Base::numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    Base::compute_shape_functions_and_derivatives(gp, shapefct, deriv, N_XYZ);

    // compute deformation gradient
    Base::compute_def_gradient(defgrd, N_XYZ, xcurr);

    // inverse deformation gradient F^-1
    Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> defgrd_inv(
        Core::LinAlg::Initialization::uninitialized);
    defgrd_inv.invert(defgrd);

    // jacobian determinant of transformation between spatial and material space "|dx/dX|"
    double J = 0.0;
    //------linearization of jacobi determinant detF=J w.r.t. structure displacement   dJ/d(us) =
    // dJ/dF : dF/dus = J * F^-T * N,X
    static Core::LinAlg::Matrix<1, Base::numdof_> dJ_dus;
    // volume change (used for porosity law). Same as J in nonlinear theory.
    double volchange = 0.0;
    //------linearization of volume change w.r.t. structure displacement
    static Core::LinAlg::Matrix<1, Base::numdof_> dvolchange_dus;

    // compute J, the volume change and the respctive linearizations w.r.t. structure displacement
    Base::compute_jacobian_determinant_volume_change_and_linearizations(
        J, volchange, dJ_dus, dvolchange_dus, defgrd, defgrd_inv, N_XYZ, nodaldisp);

    //----------------------------------------------------
    // pressure at integration point
    double press = shapefct.dot(epreaf);

    // structure velocity at integration point
    Core::LinAlg::Matrix<Base::numdim_, 1> velint(Core::LinAlg::Initialization::zero);

    for (int i = 0; i < Base::numnod_; i++)
      for (int j = 0; j < Base::numdim_; j++) velint(j) += nodalvel(j, i) * shapefct(i);

    // fluid velocity at integration point
    Core::LinAlg::Matrix<Base::numdim_, 1> fvelint;
    fvelint.multiply(evelnp, shapefct);

    // material fluid velocity gradient at integration point
    Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> fvelder;
    fvelder.multiply_nt(evelnp, N_XYZ);

    // pressure gradient at integration point
    Core::LinAlg::Matrix<Base::numdim_, 1> Gradp;
    Gradp.multiply(N_XYZ, epreaf);

    // non-linear B-operator
    Core::LinAlg::Matrix<Base::numstr_, Base::numdof_> bop;
    Base::compute_b_operator(bop, defgrd, N_XYZ);

    // Right Cauchy-Green tensor = F^T * F
    Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> cauchygreen;
    cauchygreen.multiply_tn(defgrd, defgrd);

    // inverse Right Cauchy-Green tensor
    Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> C_inv(
        Core::LinAlg::Initialization::uninitialized);
    C_inv.invert(cauchygreen);

    //------linearization of material gradient of jacobi determinant GradJ  w.r.t. structure
    // displacement d(GradJ)/d(us)
    //---------------------d(GradJ)/dus =  dJ/dus * F^-T . : dF/dX + J * dF^-T/dus : dF/dX + J *
    // F^-T : N_X_X

    // compute some auxiliary matrixes for computation of linearization
    // dF^-T/dus
    Core::LinAlg::Matrix<Base::numdim_ * Base::numdim_, Base::numdof_> dFinvTdus(
        Core::LinAlg::Initialization::zero);
    // F^-T * Grad p
    Core::LinAlg::Matrix<Base::numdim_, 1> Finvgradp;
    // dF^-T/dus * Grad p
    Core::LinAlg::Matrix<Base::numdim_, Base::numdof_> dFinvdus_gradp(
        Core::LinAlg::Initialization::zero);
    // dC^-1/dus * Grad p
    Core::LinAlg::Matrix<Base::numstr_, Base::numdof_> dCinv_dus(
        Core::LinAlg::Initialization::zero);

    Base::compute_auxiliary_values(
        N_XYZ, defgrd_inv, C_inv, Gradp, dFinvTdus, Finvgradp, dFinvdus_gradp, dCinv_dus);

    //--------------------------------------------------------------------

    // linearization of porosity w.r.t structure displacement d\phi/d(us) = d\phi/dJ*dJ/d(us)
    Core::LinAlg::Matrix<1, Base::numdof_> dphi_dus;
    double porosity = 0.0;

    compute_porosity_and_linearization(shapefct, porosity_dof, porosity, dphi_dus);

    double dW_dphi = 0.0;
    double dW_dJ = 0.0;
    double dW_dp = 0.0;
    double W = 0.0;
    Base::struct_mat_->constitutive_derivatives(params, press, volchange, porosity,
        &dW_dp,  // dW_dp not needed
        &dW_dphi, &dW_dJ, nullptr, &W);

    //--------------------------------------------------------

    // **********************evaluate stiffness matrix and force vector+++++++++++++++++++++++++
    if (Base::fluid_mat_->type() == Mat::PAR::darcy_brinkman)
    {
      Base::fill_matrix_and_vectors_brinkman(gp, J, porosity, fvelder, defgrd_inv, bop, C_inv,
          dphi_dus, dJ_dus, dCinv_dus, dFinvTdus, sub_stiff, sub_force, fstress);
    }

    Base::fill_matrix_and_vectors(gp, shapefct, N_XYZ, J, press, porosity, velint, fvelint, fvelder,
        defgrd_inv, bop, C_inv, Finvgradp, dphi_dus, dJ_dus, dCinv_dus, dFinvdus_gradp, dFinvTdus,
        erea_v, sub_stiff, sub_force, fstress);

    // **********************evaluate stiffness matrix and force vector+++++++++++++++++++++++++
    double detJ_w = Base::detJ_[gp] * Base::intpoints_.weight(gp);  // gpweights[gp];

    const double reacoeff = Base::fluid_mat_->compute_reaction_coeff();
    {
      for (int k = 0; k < Base::numnod_; k++)
      {
        const double fac = detJ_w * shapefct(k);

        ecoupl_force_p1(k) += fac * W;

        for (int i = 0; i < Base::numnod_; i++)
        {
          for (int j = 0; j < Base::numdim_; j++)
          {
            estiff_p1(k, i * noddof_ + j) += fac * dW_dJ * dJ_dus(i * Base::numdim_ + j);

            ecoupl_p1(i * Base::numdim_ + j, k) +=
                fac * (2 * J * reacoeff * porosity * (velint(j) - fvelint(j)) + J * Finvgradp(j)) *
                shapefct(i);
          }
          estiff_p1(k, i * noddof_ + Base::numdim_) += fac * dW_dphi * shapefct(i);
        }
      }
    }

    if (Base::fluid_mat_->type() == Mat::PAR::darcy_brinkman)
    {
      double visc = Base::fluid_mat_->viscosity();
      Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> CinvFvel(
          Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> visctress1(
          Core::LinAlg::Initialization::zero);
      CinvFvel.multiply(C_inv, fvelder);
      visctress1.multiply_nt(CinvFvel, defgrd_inv);
      Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> visctress2(visctress1);
      visctress1.update_t(1.0, visctress2, 1.0);

      fstress(0) = visctress1(0, 0);
      fstress(1) = visctress1(1, 1);
      fstress(2) = visctress1(0, 1);

      fstress.scale(detJ_w * visc * J);

      // B^T . C^-1
      Core::LinAlg::Matrix<Base::numdof_, 1> fstressb(Core::LinAlg::Initialization::zero);
      fstressb.multiply_tn(bop, fstress);

      for (int k = 0; k < Base::numnod_; k++)
      {
        const double fac = detJ_w * shapefct(k);
        for (int i = 0; i < Base::numnod_; i++)
          for (int j = 0; j < Base::numdim_; j++)
            ecoupl_p1(i * Base::numdim_ + j, k) +=
                fac * fstressb(i * Base::numdim_ + j) * shapefct(i);
      }
    }
    /* =========================================================================*/
  } /* ==================================================== end of Loop over GP */
  /* =========================================================================*/
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroP1<distype>::coupling_poroelast(
    std::vector<int>& lm,                                      // location matrix
    Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& disp,  // current displacements
    Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& vel,   // current velocities
    Core::LinAlg::Matrix<Base::numnod_, 1>* porosity,
    Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& evelnp,  // current fluid velocity
    Core::LinAlg::Matrix<Base::numnod_, 1>& epreaf,              // current fluid pressure
    Core::LinAlg::Matrix<numdof_, (Base::numdim_ + 1) * Base::numnod_>*
        stiffmatrix,  // element stiffness matrix
    Core::LinAlg::Matrix<numdof_, (Base::numdim_ + 1) * Base::numnod_>*
        reamatrix,                            // element reactive matrix
    Core::LinAlg::Matrix<numdof_, 1>* force,  // element internal force vector
    Teuchos::ParameterList& params)           // algorithmic parameters e.g. time
{
  Base::get_materials();

  //=======================================================================

  // update element geometry
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> xrefe;  // material coord. of element
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> xcurr;  // current  coord. of element

  Core::Nodes::Node** nodes = Base::nodes();
  for (int i = 0; i < Base::numnod_; ++i)
  {
    const auto& x = nodes[i]->x();
    for (int j = 0; j < Base::numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }
  // initialize element matrizes
  Core::LinAlg::Matrix<Base::numdof_, (Base::numdim_ + 1) * Base::numnod_> ecoupl(
      Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<Base::numnod_, Base::numnod_> ecoupl_p1_p(
      Core::LinAlg::Initialization::zero);

  /* =========================================================================*/
  /* ================================================= Loop over Gauss Points */
  /* =========================================================================*/
  gauss_point_loop_p1_od(
      params, xrefe, xcurr, disp, vel, evelnp, epreaf, porosity, ecoupl_p1_p, ecoupl);

  if (stiffmatrix != nullptr)
  {
    for (int k = 0; k < Base::numnod_; k++)
    {
      for (int l = 0; l < (Base::numdim_ + 1); l++)
      {
        for (int i = 0; i < Base::numnod_; i++)
        {
          for (int j = 0; j < Base::numdim_; j++)
            (*stiffmatrix)(i* noddof_ + j, k * (Base::numdim_ + 1) + l) +=
                ecoupl(i * Base::numdim_ + j, k * (Base::numdim_ + 1) + l);
        }
      }
    }

    for (int ui = 0; ui < Base::numnod_; ++ui)
    {
      for (int ni = 0; ni < Base::numnod_; ++ni)
        (*stiffmatrix)(noddof_* ui + Base::numdim_, (Base::numdim_ + 1) * ni + Base::numdim_) +=
            ecoupl_p1_p(ui, ni);
    }
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1PoroP1<distype>::gauss_point_loop_p1_od(Teuchos::ParameterList& params,
    const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& xrefe,
    const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& xcurr,
    const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& nodaldisp,
    const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& nodalvel,
    const Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>& evelnp,
    const Core::LinAlg::Matrix<Base::numnod_, 1>& epreaf,
    const Core::LinAlg::Matrix<Base::numnod_, 1>* porosity_dof,
    Core::LinAlg::Matrix<Base::numnod_, Base::numnod_>& ecoupl_p1,
    Core::LinAlg::Matrix<Base::numdof_, (Base::numdim_ + 1) * Base::numnod_>& sub_stiff)
{
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_>
      N_XYZ;  //  first derivatives at gausspoint w.r.t. X,Y,Z
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> defgrd(
      Core::LinAlg::Initialization::zero);  //  deformation gradient evaluated at gauss point
  Core::LinAlg::Matrix<Base::numnod_, 1> shapefct;  //  shape functions evaluated at gauss point
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> deriv(
      Core::LinAlg::Initialization::zero);  //  first derivatives at gausspoint w.r.t. r,s,t

  for (int gp = 0; gp < Base::numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    Base::compute_shape_functions_and_derivatives(gp, shapefct, deriv, N_XYZ);
    // evaluate second derivatives of shape functions at integration point
    // ComputeSecondDerivativesOfShapeFunctions(gp,xrefe,deriv,deriv2,N_XYZ,N_XYZ2);

    // (material) deformation gradient F = d xcurr / d xrefe = xcurr * N_XYZ^T
    Base::compute_def_gradient(defgrd, N_XYZ, xcurr);

    // inverse deformation gradient F^-1
    Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> defgrd_inv(
        Core::LinAlg::Initialization::uninitialized);
    defgrd_inv.invert(defgrd);

    // jacobian determinant of transformation between spatial and material space "|dx/dX|"
    double J = 0.0;
    // volume change (used for porosity law). Same as J in nonlinear theory.
    double volchange = 0.0;

    // compute J, the volume change and the respctive linearizations w.r.t. structure displacement
    Base::compute_jacobian_determinant_volume_change(J, volchange, defgrd, N_XYZ, nodaldisp);

    // non-linear B-operator
    Core::LinAlg::Matrix<Base::numstr_, Base::numdof_> bop;
    Base::compute_b_operator(bop, defgrd, N_XYZ);

    // -----------------Right Cauchy-Green tensor = F^T * F
    Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> cauchygreen;
    cauchygreen.multiply_tn(defgrd, defgrd);

    //------------------ inverse Right Cauchy-Green tensor
    Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> C_inv(
        Core::LinAlg::Initialization::uninitialized);
    C_inv.invert(cauchygreen);

    //---------------- get pressure at integration point
    double press = shapefct.dot(epreaf);

    //------------------ get material pressure gradient at integration point
    Core::LinAlg::Matrix<Base::numdim_, 1> Gradp;
    Gradp.multiply(N_XYZ, epreaf);

    //--------------------- get fluid velocity at integration point
    Core::LinAlg::Matrix<Base::numdim_, 1> fvelint;
    fvelint.multiply(evelnp, shapefct);

    //---------------- material fluid velocity gradient at integration point
    Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> fvelder;
    fvelder.multiply_nt(evelnp, N_XYZ);

    //---------------- structure velocity at integration point
    Core::LinAlg::Matrix<Base::numdim_, 1> velint(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < Base::numnod_; i++)
      for (int j = 0; j < Base::numdim_; j++) velint(j) += nodalvel(j, i) * shapefct(i);

    //**************************************************+auxiliary variables for computing the
    // porosity and linearization
    double dphi_dp = 0.0;
    double porosity = 0.0;

    compute_porosity_and_linearization_od(
        params, press, volchange, gp, shapefct, porosity_dof, porosity, dphi_dp);

    // **********************evaluate stiffness matrix and force vector+++++++++++++++++++++++++

    Base::fill_matrix_and_vectors_od(gp, shapefct, N_XYZ, J, porosity, dphi_dp, velint, fvelint,
        defgrd_inv, Gradp, bop, C_inv, sub_stiff);

    if (Base::fluid_mat_->type() == Mat::PAR::darcy_brinkman)
    {
      Base::fill_matrix_and_vectors_brinkman_od(
          gp, shapefct, N_XYZ, J, porosity, dphi_dp, fvelder, defgrd_inv, bop, C_inv, sub_stiff);
    }  // darcy-brinkman

    double dW_dp = 0.0;
    Base::struct_mat_->constitutive_derivatives(params, press, J, porosity, &dW_dp,
        nullptr,  // not needed
        nullptr,  // not needed
        nullptr,  // not needed
        nullptr   // not needed
    );
    //--------------------------------------------------------

    // **********************evaluate stiffness matrix and force vector+++++++++++++++++++++++++
    double detJ_w = Base::detJ_[gp] * Base::intpoints_.weight(gp);

    for (int k = 0; k < Base::numnod_; k++)
    {
      const double fac = detJ_w * shapefct(k);

      for (int i = 0; i < Base::numnod_; i++) ecoupl_p1(k, i) += fac * dW_dp * shapefct(i);
    }

    /* =========================================================================*/
  } /* ==================================================== end of Loop over GP */
  /* =========================================================================*/
}

template <Core::FE::CellType distype>
int Discret::Elements::Wall1PoroP1<distype>::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> disp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Base::numnod_, 1> myporosity(Core::LinAlg::Initialization::zero);
  Base::extract_values_from_global_vector(
      discretization, 0, lm, &disp, &myporosity, "displacement");

  // find out whether we will use a time curve
  const double time = params.get("total time", -1.0);

  /*----------------------------------------------------- geometry update */
  // update element geometry
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> xrefe;  // material coord. of element
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> xcurr;  // current  coord. of element

  Core::Nodes::Node** nodes = Base::nodes();
  for (int i = 0; i < Base::numnod_; ++i)
  {
    const auto& x = nodes[i]->x();
    for (int j = 0; j < Base::numdim_; j++)
    {
      xrefe(j, i) = x[j];
      xcurr(j, i) = xrefe(j, i) + disp(j, i);
    }
  }


  // get values and switches from the condition
  const auto onoff = condition.parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition.parameters().get<std::vector<double>>("VAL");
  const auto funct = condition.parameters().get<std::vector<std::optional<int>>>("FUNCT");


  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> N_XYZ;
  // build deformation gradient wrt to material configuration
  // in case of prestressing, build defgrd wrt to last stored configuration
  // CAUTION: defgrd(true): filled with zeros!
  Core::LinAlg::Matrix<Base::numdim_, Base::numdim_> defgrd(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<Base::numnod_, 1> shapefcts;
  Core::LinAlg::Matrix<Base::numdim_, Base::numnod_> deriv;

  Core::LinAlg::Matrix<Base::numstr_, 1> fstress(Core::LinAlg::Initialization::zero);

  for (int gp = 0; gp < Base::numgpt_; ++gp)
  {
    // evaluate shape functions and derivatives at integration point
    Base::compute_shape_functions_and_derivatives(gp, shapefcts, deriv, N_XYZ);

    // jacobian determinant of transformation between spatial and material space "|dx/dX|"
    Base::compute_jacobian_determinant(gp, xcurr, deriv);

    /*------------------------------------ integration factor  -------*/
    double fac = Base::detJ_[gp] * Base::intpoints_.weight(gp);

    // load vector ar
    std::array<double, Base::numdim_> ar = {0.0, 0.0};
    // loop the dofs of a node
    for (int i = 0; i < Base::numdim_; ++i)
    {
      if (onoff[i])
      {
        double functfac = 1.0;

        // factor given by spatial function
        if (funct[i].has_value() && funct[i].value() > 0)
        {
          // calculate reference position of GP
          Core::LinAlg::Matrix<1, Base::numdim_> gp_coord;
          gp_coord.multiply_tt(shapefcts, xrefe);

          // write coordinates in another datatype
          std::array<double, 3> gp_coord2;  // the position vector has to be given in 3D!!!
          for (int k = 0; k < Base::numdim_; k++) gp_coord2[k] = gp_coord(0, k);
          for (int k = Base::numdim_; k < 3;
              k++)  // set a zero value for the remaining spatial directions
            gp_coord2[k] = 0.0;
          const double* coordgpref = gp_coord2.data();  // needed for function evaluation

          // evaluate function at current gauss point
          functfac = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funct[i].value())
                         .evaluate(coordgpref, time, i);
        }

        ar[i] = fac * val[i] * functfac;
      }
    }

    // add load components
    for (int node = 0; node < Base::numnod_; ++node)
      for (int dim = 0; dim < Base::numdim_; ++dim)
        elevec1[node * noddof_ + dim] += shapefcts(node) * ar[dim];
  }

  return 0;
}

template class Discret::Elements::Wall1PoroP1<Core::FE::CellType::tri3>;
template class Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad4>;
template class Discret::Elements::Wall1PoroP1<Core::FE::CellType::quad9>;

FOUR_C_NAMESPACE_CLOSE
