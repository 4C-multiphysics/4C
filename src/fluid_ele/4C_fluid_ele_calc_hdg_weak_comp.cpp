// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_calc_hdg_weak_comp.hpp"

#include "4C_fem_general_extract_values.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_calc.hpp"
#include "4C_fluid_ele_parameter_std.hpp"
#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_fluid_functions.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mat_fluid_weakly_compressible.hpp"
#include "4C_utils_shared_ptr_from_ref.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN



template <Core::FE::CellType distype>
Discret::Elements::FluidEleCalcHDGWeakComp<distype>::FluidEleCalcHDGWeakComp()
    : usescompletepoly_(true)
{
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::initialize_shapes(
    const Discret::Elements::Fluid* ele)
{
  // Check if this is an HDG weakly compressible element
  if (const Discret::Elements::FluidHDGWeakComp* hdgwkele =
          dynamic_cast<const Discret::Elements::FluidHDGWeakComp*>(ele))
  {
    // use complete polynomial space
    usescompletepoly_ = hdgwkele->uses_complete_polynomial_space();

    // initialize shapes
    if (shapes_ == nullptr)
      shapes_ = std::make_shared<Core::FE::ShapeValues<distype>>(
          hdgwkele->degree(), usescompletepoly_, 2 * ele->degree());
    else if (shapes_->degree_ != unsigned(ele->degree()) ||
             shapes_->usescompletepoly_ != usescompletepoly_)
      shapes_ = std::make_shared<Core::FE::ShapeValues<distype>>(
          hdgwkele->degree(), usescompletepoly_, 2 * ele->degree());

    // initialize shapes on faces
    if (shapesface_ == nullptr)
    {
      Core::FE::ShapeValuesFaceParams svfparams(
          ele->degree(), usescompletepoly_, 2 * ele->degree());
      shapesface_ = std::make_shared<Core::FE::ShapeValuesFace<distype>>(svfparams);
    }

    // initialize local solver
    if (local_solver_ == nullptr)
      local_solver_ = std::make_shared<LocalSolver>(ele, *shapes_, *shapesface_, usescompletepoly_);
  }
  else
    FOUR_C_THROW("Only works for HDG weakly compressible fluid elements");
}



template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcHDGWeakComp<distype>::evaluate(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3, const Core::FE::GaussIntegration&, bool offdiag)
{
  return this->evaluate(
      ele, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2, elevec3, offdiag);
}



template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcHDGWeakComp<distype>::evaluate(Discret::Elements::Fluid* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix&,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector&,
    Core::LinAlg::SerialDenseVector&, bool offdiag)
{
  // read global vectors
  read_global_vectors(*ele, discretization, lm);

  // initialize shapes
  initialize_shapes(ele);
  shapes_->evaluate(*ele, ale_dis_);

  // initialize all
  local_solver_->initialize_all();

  // compute interior residual and matrices
  local_solver_->compute_interior_residual(mat, interior_val_, interior_acc_, ale_vel_);
  local_solver_->compute_interior_matrices(mat);

  // compute face residual and face matrices
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    shapesface_->evaluate_face(*ele, f, ale_dis_);
    local_solver_->compute_face_residual(f, mat, interior_val_, trace_val_, ale_vel_);
    local_solver_->compute_face_matrices(f, mat);
  }

  // group residuals
  local_solver_->compute_local_residual();
  local_solver_->compute_global_residual(*ele);

  // group matrices
  local_solver_->compute_local_local_matrix();
  local_solver_->compute_local_global_matrix(*ele);
  local_solver_->compute_global_local_matrix(*ele);
  local_solver_->compute_global_global_matrix(*ele);

  // condense local part
  local_solver_->invert_local_local_matrix();
  local_solver_->condense_local_residual(elevec1);
  local_solver_->condense_local_matrix(elemat1);

  // divide rhs by alpha_f
  elevec1.scale(1.0 / local_solver_->fldparatimint_->alpha_f());

  return 0;
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::read_global_vectors(
    const Core::Elements::Element& ele, Core::FE::Discretization& discretization,
    const std::vector<int>& lm)
{
  // initialize the vectors
  trace_val_.resize(nfaces_ * (1 + nsd_) * shapesface_->nfdofs_);
  interior_val_.resize((msd_ + 1 + nsd_) * shapes_->ndofs_);
  interior_acc_.resize((msd_ + 1 + nsd_) * shapes_->ndofs_);

  // read the trace values
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
      discretization.get_state(0, "velaf");
  trace_val_ = Core::FE::extract_values(*matrix_state, lm);

  // get local dofs
  std::vector<int> localDofs = discretization.dof(1, &ele);

  // read the interior values
  matrix_state = discretization.get_state(1, "intvelaf");
  interior_val_ = Core::FE::extract_values(*matrix_state, localDofs);

  // read the interior time derivatives
  matrix_state = discretization.get_state(1, "intaccam");
  interior_acc_ = Core::FE::extract_values(*matrix_state, localDofs);

  // read ale vectors
  read_ale_vectors(ele, discretization);
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::read_ale_vectors(
    const Core::Elements::Element& ele, Core::FE::Discretization& discretization)
{
  // initialize ale vectors
  ale_dis_.resize(nsd_ * nen_);
  ale_vel_.resize(nsd_ * nen_);

  // check presence of ale state
  if (discretization.has_state(2, "dispnp"))
  {
    const Discret::Elements::FluidHDGWeakComp* hdgwkele =
        dynamic_cast<const Discret::Elements::FluidHDGWeakComp*>(&ele);
    if (hdgwkele->is_ale())
    {
      // get ale dofs
      const int* nodeids = ele.node_ids();
      std::vector<int> aleDofs;
      for (unsigned int n = 0; n < nen_; ++n)
      {
        std::vector<int> nodeDofs = discretization.dof(2, discretization.g_node(nodeids[n]));
        for (unsigned int d = 0; d < nsd_; ++d) aleDofs.push_back(nodeDofs[d]);
      }

      // initialize matrix state
      std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state;

      // read the ale displacement
      matrix_state = discretization.get_state(2, "dispnp");
      ale_dis_ = Core::FE::extract_values(*matrix_state, aleDofs);

      // read the ale velocity
      matrix_state = discretization.get_state(2, "gridv");
      ale_vel_ = Core::FE::extract_values(*matrix_state, aleDofs);
    }
  }
}



template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcHDGWeakComp<distype>::evaluate_service(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    std::shared_ptr<Core::Mat::Material>& mat, Core::FE::Discretization& discretization,
    std::vector<int>& lm, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3)
{
  // get the action required
  const auto act = Teuchos::getIntegralValue<FLD::Action>(params, "action");

  switch (act)
  {
    case FLD::project_fluid_field:
    {
      return project_field(ele, params, mat, discretization, lm, elevec1, elevec2);
      break;
    }
    case FLD::update_local_solution:
    {
      return update_local_solution(ele, params, mat, discretization, lm, elevec1);
      break;
    }
    case FLD::interpolate_hdg_to_node:
    {
      return interpolate_solution_to_nodes(ele, discretization, elevec1);
      break;
    }
    case FLD::calc_fluid_error:
    {
      return compute_error(ele, params, mat, discretization, lm, elevec1);
      break;
    }
    default:
      FOUR_C_THROW("Unknown type of action for FluidHDG");
      break;
  }

  return 0;
}



template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcHDGWeakComp<distype>::update_local_solution(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    std::shared_ptr<Core::Mat::Material>& mat, Core::FE::Discretization& discretization,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& interiorinc)
{
  // read global vectors
  read_global_vectors(*ele, discretization, lm);

  // initialize shapes
  initialize_shapes(ele);
  shapes_->evaluate(*ele, ale_dis_);

  // initialize all
  local_solver_->initialize_all();

  // compute interior residual and matrices
  local_solver_->compute_interior_residual(mat, interior_val_, interior_acc_, ale_vel_);
  local_solver_->compute_interior_matrices(mat);

  // compute face residual and face matrices
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    shapesface_->evaluate_face(*ele, f, ale_dis_);
    local_solver_->compute_face_residual(f, mat, interior_val_, trace_val_, ale_vel_);
    local_solver_->compute_face_matrices(f, mat);
  }

  // group residuals
  local_solver_->compute_local_residual();

  // group matrices
  local_solver_->compute_local_local_matrix();
  local_solver_->compute_local_global_matrix(*ele);

  // invert local-local matrix
  local_solver_->invert_local_local_matrix();

  // extract local trace increments
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
      discretization.get_state(0, "globaltraceinc");
  std::vector<double> localtraceinc_vec = Core::FE::extract_values(*matrix_state, lm);

  // convert local trace increments to Core::LinAlg::SerialDenseVector
  Core::LinAlg::SerialDenseVector localtraceinc(nfaces_ * (1 + nsd_) * shapesface_->nfdofs_);
  for (unsigned int i = 0; i < nfaces_ * (1 + nsd_) * shapesface_->nfdofs_; ++i)
    localtraceinc(i) = localtraceinc_vec[i];

  // compute local solver vector
  Core::LinAlg::SerialDenseVector LocalSolverVec((msd_ + 1 + nsd_) * shapes_->ndofs_);
  Core::LinAlg::multiply(LocalSolverVec, local_solver_->KlocallocalInv, local_solver_->Rlocal);

  // compute local solver matrix
  Core::LinAlg::SerialDenseMatrix LocalSolverMat(
      (msd_ + 1 + nsd_) * shapes_->ndofs_, nfaces_ * (1 + nsd_) * shapesface_->nfdofs_);
  Core::LinAlg::multiply(
      0.0, LocalSolverMat, -1.0, local_solver_->KlocallocalInv, local_solver_->Klocalglobal);

  // compute local increments
  interiorinc.size((msd_ + 1 + nsd_) * shapes_->ndofs_);
  interiorinc = LocalSolverVec;
  Core::LinAlg::multiply(1.0, interiorinc, 1.0, LocalSolverMat, localtraceinc);

  return 0;
}



template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcHDGWeakComp<distype>::compute_error(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    std::shared_ptr<Core::Mat::Material>& mat, Core::FE::Discretization& discretization,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec)
{
  // read ale vectors
  read_ale_vectors(*ele, discretization);

  // initialize shapes
  initialize_shapes(ele);
  shapes_->evaluate(*ele, ale_dis_);

  // get time
  const double time = local_solver_->fldparatimint_->time();

  // get interior values
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
      discretization.get_state(1, "intvelnp");
  std::vector<int> localDofs = discretization.dof(1, ele);
  std::vector<double> vecValues(localDofs.size());
  for (unsigned int i = 0; i < localDofs.size(); ++i)
  {
    const int lid = matrix_state->get_map().lid(localDofs[i]);
    vecValues[i] = matrix_state->local_values_as_span()[lid];
  }

  // initialize exact solution
  Core::LinAlg::Matrix<msd_, 1> L_ex(Core::LinAlg::Initialization::zero);
  double r_ex = 0.0;
  Core::LinAlg::Matrix<nsd_, 1> w_ex(Core::LinAlg::Initialization::zero);

  // initialize spatial coordinates
  Core::LinAlg::Matrix<nsd_, 1> xyz(Core::LinAlg::Initialization::zero);

  // get function number
  const int calcerrfunctno = params.get<int>("error function number");

  // initialize errors
  double err_L = 0.0;
  double err_r = 0.0;
  double err_w = 0.0;
  double norm_L = 0.0;
  double norm_r = 0.0;
  double norm_w = 0.0;

  // initialize numerical values
  Core::LinAlg::SerialDenseVector Leg;
  Core::LinAlg::SerialDenseVector reg;
  Core::LinAlg::SerialDenseVector weg;

  // ease notation
  Core::LinAlg::SerialDenseMatrix N = shapes_->shfunct;
  Core::LinAlg::SerialDenseVector fac = shapes_->jfac;
  unsigned int nqpoints = shapes_->nqpoints_;
  unsigned int ndofs = shapes_->ndofs_;

  // loop over quadrature points
  for (unsigned int q = 0; q < nqpoints; ++q)
  {
    // clear vectors
    Leg.size(msd_);
    reg.size(1);
    weg.size(nsd_);

    // interpolate values on gauss points
    for (unsigned int i = 0; i < ndofs; ++i)
    {
      for (unsigned int m = 0; m < msd_; ++m) Leg(m) += N(i, q) * vecValues[m * ndofs + i];

      reg(0) += N(i, q) * vecValues[msd_ * ndofs + i];

      for (unsigned int d = 0; d < nsd_; ++d)
        weg(d) += N(i, q) * vecValues[(msd_ + 1 + d) * ndofs + i];
    }

    for (unsigned int d = 0; d < nsd_; ++d) xyz(d) = shapes_->xyzreal(d, q);

    // evaluate analytical solution
    evaluate_all(calcerrfunctno, xyz, time, L_ex, r_ex, w_ex);

    // evaluate errors
    for (unsigned int m = 0; m < msd_; ++m)
      err_L += (L_ex(m) - Leg(m)) * (L_ex(m) - Leg(m)) * fac(q);

    err_r += (r_ex - reg(0)) * (r_ex - reg(0)) * fac(q);

    for (unsigned int d = 0; d < nsd_; ++d)
      err_w += (w_ex(d) - weg(d)) * (w_ex(d) - weg(d)) * fac(q);

    // evaluate norms
    for (unsigned int m = 0; m < msd_; ++m) norm_L += L_ex(m) * L_ex(m) * fac(q);

    norm_r += r_ex * r_ex * fac(q);

    for (unsigned int d = 0; d < nsd_; ++d) norm_w += w_ex(d) * w_ex(d) * fac(q);
  }

  // fill vector
  elevec[0] += err_L;
  elevec[1] += err_r;
  elevec[2] += err_w;
  elevec[3] += norm_L;
  elevec[4] += norm_r;
  elevec[5] += norm_w;

  return 0;
};



template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcHDGWeakComp<distype>::project_field(
    Discret::Elements::Fluid* ele, Teuchos::ParameterList& params,
    std::shared_ptr<Core::Mat::Material>& mat, Core::FE::Discretization& discretization,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2)
{
  // read ale vectors
  read_ale_vectors(*ele, discretization);

  // Create the necessary objects to the solution of the problem as the solver
  // and the shape functions for both the interior, shapes_, and the trace, shapesface_.
  initialize_shapes(ele);

  // Evaluate the element at the gauss points
  shapes_->evaluate(*ele, ale_dis_);

  // reshape elevec2 as matrix
  FOUR_C_ASSERT(elevec2.numRows() == 0 ||
                    elevec2.numRows() == static_cast<int>((msd_ + 1 + nsd_) * shapes_->ndofs_),
      "Wrong size in project vector 2");

  // get initial function and current time
  const auto* initfield = params.getPtr<Inpar::FLUID::InitialField>("initfield");
  const int* startfunc = params.getPtr<int>("startfuncno");
  double* time = params.getPtr<double>("time");

  if (elevec2.numRows() > 0)
  {
    // Create the local matrix from starting at the address where elevec2 is with the right shape
    Core::LinAlg::SerialDenseMatrix localMat(
        Teuchos::View, elevec2.values(), shapes_->ndofs_, shapes_->ndofs_, msd_ + 1 + nsd_);
    // Initialize matrix to zeros
    localMat.putScalar(0.0);

    // create mass matrix for interior by looping over quadrature points
    for (unsigned int q = 0; q < shapes_->nqpoints_; ++q)
    {
      // jfac is a vector containing the jacobian times the weight of the quadrature points
      const double fac = shapes_->jfac(q);
      // xyz is a vector containing the coordinates of the quadrature points in real coordinates
      Core::LinAlg::Matrix<nsd_, 1> xyz(Core::LinAlg::Initialization::uninitialized);
      // Filling xyz with the values take from the element xyzreal matrix
      for (unsigned int d = 0; d < nsd_; ++d) xyz(d) = shapes_->xyzreal(d, q);
      // Declaring vectors for interior variables
      Core::LinAlg::Matrix<msd_, 1> L(Core::LinAlg::Initialization::zero);
      double r = 0.0;
      Core::LinAlg::Matrix<nsd_, 1> w(Core::LinAlg::Initialization::zero);

      FOUR_C_ASSERT(initfield != nullptr && startfunc != nullptr,
          "initfield or startfuncno not set for initial value");

      // This function returns th value of the interior variables from the
      // given initial field that can be a know field or a user-defined one
      evaluate_density_momentum(*startfunc, xyz, 0.0, r, w);

      // now fill the components in the one-sided mass matrix and the right hand side
      for (unsigned int i = 0; i < shapes_->ndofs_; ++i)
      {
        local_solver_->massPart(i, q) = shapes_->shfunct(i, q);
        local_solver_->massPartW(i, q) = shapes_->shfunct(i, q) * fac;

        // RHS for the mixed variable
        for (unsigned int m = 0; m < msd_; ++m)
          localMat(i, m) += shapes_->shfunct(i, q) * L(m) * fac;

        // RHS for the density
        localMat(i, msd_) += shapes_->shfunct(i, q) * r * fac;

        // RHS for the momenutm
        for (unsigned int d = 0; d < nsd_; ++d)
          localMat(i, msd_ + 1 + d) += shapes_->shfunct(i, q) * w(d) * fac;
      }
    }
    // The integration is made by computing the matrix product
    Core::LinAlg::multiply_nt(
        local_solver_->massMat, local_solver_->massPart, local_solver_->massPartW);
    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
    inverseMass.setMatrix(Teuchos::rcpFromRef(local_solver_->massMat.base()));
    inverseMass.setVectors(
        Teuchos::rcpFromRef(localMat.base()), Teuchos::rcpFromRef(localMat.base()));
    inverseMass.solve();
  }

  // Here we have the projection of the field on the trace
  Core::LinAlg::SerialDenseMatrix mass(shapesface_->nfdofs_, shapesface_->nfdofs_);
  // trVec is the vector of the trace values
  Core::LinAlg::SerialDenseMatrix trVec(shapesface_->nfdofs_, 1 + nsd_);
  FOUR_C_ASSERT(
      elevec1.numRows() == static_cast<int>((1 + nsd_) * shapesface_->nfdofs_) ||
          elevec1.numRows() == static_cast<int>(nfaces_ * (1 + nsd_) * shapesface_->nfdofs_),
      "Wrong size in project vector 1");

  const unsigned int* faceConsider = params.getPtr<unsigned int>("faceconsider");
  Teuchos::Array<int>* functno = params.getPtr<Teuchos::Array<int>>("funct");
  Teuchos::Array<int>* onoff = params.getPtr<Teuchos::Array<int>>("onoff");

  // Project the field for all the faces of the element
  for (unsigned int face = 0; face < nfaces_; ++face)
  {
    // check whether we are in the project phase for all faces or for boundary values
    if (initfield == nullptr)
    {
      // We get here only if it is not an initial value but it is a time
      // dependant boundary value. If we are here we only want the function to run
      // for boundary faces specified in the faceConsider variable
      FOUR_C_ASSERT(faceConsider != nullptr, "Unsupported operation");
      if (*faceConsider != face) continue;
    }

    // the same function as before but for the trace elements
    shapesface_->evaluate_face(*ele, face, ale_dis_);

    // Initializing the matrices
    mass.putScalar(0.0);
    trVec.putScalar(0.0);

    // For each quadrature point we evaluate the trace values and the shape functions
    for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
    {
      // shapesface_->jfac contains the jacobian evaluated in the quadrature points
      const double fac = shapesface_->jfac(q);
      // xyz is the vector containing the coordinates of the quadrature points
      Core::LinAlg::Matrix<nsd_, 1> xyz(Core::LinAlg::Initialization::uninitialized);

      // Taking the real coordinates of quadrature points of the current face
      for (unsigned int d = 0; d < nsd_; ++d) xyz(d) = shapesface_->xyzreal(d, q);

      // Creating the vector of traces variables
      double r;
      Core::LinAlg::Matrix<nsd_, 1> w(Core::LinAlg::Initialization::uninitialized);

      // Create dummy variables
      double dummy_r;
      Core::LinAlg::Matrix<nsd_, 1> dummy_w;

      // Deciding if we are initializing a field or if it is a time dependant
      // boundary condition
      if (initfield != nullptr)  // Initial function
        evaluate_density_momentum(*startfunc, xyz, 0.0, r, w);
      else
      {
        // This is used to project a function only on the boundary during the
        // temporal evolution of the simulation
        FOUR_C_ASSERT(functno != nullptr && time != nullptr && onoff != nullptr,
            "No array with functions given");

        // Deciding if to use the function or not for the density
        if ((*onoff)[0] == 1)
        {
          const int funct_num = (*functno)[0];
          if (funct_num > 0) evaluate_density_momentum(funct_num, xyz, *time, r, dummy_w);
        }

        for (unsigned int d = 0; d < nsd_; ++d)
        {
          // Deciding if to use the function or not for the current component of the momentum
          if ((*onoff)[1 + d] == 0) continue;

          // If we are using the function, evaluate it
          const int funct_num = (*functno)[1 + d];
          if (funct_num > 0) evaluate_density_momentum(funct_num, xyz, *time, dummy_r, w);
        }
      }
      // now fill the components in the mass matrix and the right hand side

      // This is a more usual way to compute the mass matrix (double for loop)
      for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      {
        // mass matrix
        for (unsigned int j = 0; j < shapesface_->nfdofs_; ++j)
          mass(i, j) += shapesface_->shfunct(i, q) * shapesface_->shfunct(j, q) * fac;

        // RHS
        trVec(i, 0) += shapesface_->shfunct(i, q) * r * fac;
        for (unsigned int d = 0; d < nsd_; ++d)
          trVec(i, 1 + d) += shapesface_->shfunct(i, q) * w(d) * fac;
      }
    }

    // Solving step, nothing fancy
    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
    inverseMass.setMatrix(Teuchos::rcpFromRef(mass.base()));
    // In this cas trVec is a proper vector and not a matrix used as multiple
    // RHS vectors
    inverseMass.setVectors(Teuchos::rcpFromRef(trVec.base()), Teuchos::rcpFromRef(trVec.base()));
    inverseMass.solve();

    // In this case we fill elevec1 with the values of trVec because we have not
    // defined trVec as a matrix beginning where elevec1 begins
    if (initfield != nullptr)  // This is for initial functions
      for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      {
        elevec1(face * shapesface_->nfdofs_ * (1 + nsd_) + i) = trVec(i, 0);
        for (unsigned int d = 0; d < nsd_; ++d)
          elevec1(face * shapesface_->nfdofs_ * (1 + nsd_) + (1 + d) * shapesface_->nfdofs_ + i) =
              trVec(i, 1 + d);
      }
    else  // This is only for boundary faces during time evolution
      for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      {
        elevec1(i) = trVec(i, 0);
        for (unsigned int d = 0; d < nsd_; ++d)
          elevec1((1 + d) * shapesface_->nfdofs_ + i) = trVec(i, 1 + d);
      }
  }  // for over the faces

  return 0;
}



template <Core::FE::CellType distype>
int Discret::Elements::FluidEleCalcHDGWeakComp<distype>::interpolate_solution_to_nodes(
    Discret::Elements::Fluid* ele, Core::FE::Discretization& discretization,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  // read ale vectors
  read_ale_vectors(*ele, discretization);

  initialize_shapes(ele);
  // Check if the vector has the correct size
  FOUR_C_ASSERT(elevec1.numRows() == (int)nen_ * (msd_ + 1 + nsd_ + 1 + nsd_),
      "Vector does not have correct size");

  // Getting the connectivity matrix
  // Contains the (local) coordinates of the nodes belonging to the element
  Core::LinAlg::SerialDenseMatrix locations =
      Core::FE::get_ele_node_numbering_nodes_paramspace(distype);

  // This vector will contain the values of the shape functions computed in a
  // certain coordinate. In fact the length of the vector is given by the number
  // of shape functions, that is the same of the number of degrees of freedom of
  // an element.
  Core::LinAlg::SerialDenseVector values(shapes_->ndofs_);

  // get local solution values
  // The vector "matrix_state" contains the interior velocity values following
  // the local id numbers
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
      discretization.get_state(1, "intvelnp");
  // Vector of the ids of the DOF for the element
  std::vector<int> localDofs = discretization.dof(1, ele);
  // SOLution VALUES
  std::vector<double> solvalues(localDofs.size());

  // Filling every entry of the solvalue vector obtaining the values from the
  //"matrix_state" vector.
  for (unsigned int i = 0; i < solvalues.size(); ++i)
  {
    // Finding the local id of the current "localDofs"
    const int lid = matrix_state->get_map().lid(localDofs[i]);
    // Saving the value of the "localDofs[i]" in the "solvalues" vector
    solvalues[i] = matrix_state->local_values_as_span()[lid];
  }
  elevec1.putScalar(0.0);

  // EVALUATE SHAPE POLYNOMIALS IN NODE
  // In hdg we can have several more points inside the element than in the
  //"real" discretization and therefore it is necessary to compute the value
  // that the internal solution takes in the node of the discretization.

  // Cycling through all the "real" nodes of the element to get the coordinates
  // Remember that the coordinates are the local ones.
  for (unsigned int i = 0; i < nen_; ++i)
  {
    // Cycling through the spatial dimensions to get the coordinates
    for (unsigned int idim = 0; idim < nsd_; idim++) shapes_->xsi(idim) = locations(idim, i);

    // Evaluating the polinomials in the point given by "shapes_->xsi".
    // The polynomials are the internal ones.
    // The result of the evaluation is given in "values".
    shapes_->polySpace_->evaluate(shapes_->xsi, values);

    // compute values for internal unknowns by summing over all basis functions
    for (unsigned int m = 0; m <= msd_; ++m)
    {
      double sum_mixedvar = 0.0;
      for (unsigned int k = 0; k < shapes_->ndofs_; ++k)
        sum_mixedvar += values(k) * solvalues[m * shapes_->ndofs_ + k];
      elevec1(m * nen_ + i) = sum_mixedvar;
    }
    {
      double sum_density = 0.0;
      for (unsigned int k = 0; k < shapes_->ndofs_; ++k)
        sum_density += values(k) * solvalues[msd_ * shapes_->ndofs_ + k];
      elevec1(msd_ * nen_ + i) = sum_density;
    }
  }

  // get trace solution values
  // Same as before bu this time the dimension is nsd_-1 because we went from
  // the interior to the faces. We have to be careful because we are using a
  // part of the previous vector. The coordinates are still in the local frame.
  locations = Core::FE::get_ele_node_numbering_nodes_paramspace(
      Core::FE::DisTypeToFaceShapeType<distype>::shape);

  // Storing the number of nodes for each face of the element as vector
  // NumberCornerNodes
  std::vector<int> n_corner_nodes = Core::FE::get_number_of_face_element_corner_nodes(distype);
  // NumberInternalNodes
  std::vector<int> n_internal_nodes = Core::FE::get_number_of_face_element_internal_nodes(distype);

  // Now the vector "matrix_state" contains the trace velocity values following
  // the local id numbers
  matrix_state = discretization.get_state(0, "velnp");

  // we have always two dofsets
  Core::Elements::LocationArray la(2);
  ele->location_vector(discretization, la);
  localDofs = la[0].lm_;
  solvalues.resize(localDofs.size());

  for (unsigned int i = 0; i < solvalues.size(); ++i)
  {
    const int lid = matrix_state->get_map().lid(localDofs[i]);
    solvalues[i] = matrix_state->local_values_as_span()[lid];
  }
  for (int i = (msd_ + 1 + nsd_) * nen_; i < elevec1.numRows(); ++i) elevec1(i) = 0.0;

  Core::LinAlg::SerialDenseVector fvalues(shapesface_->nfdofs_);
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // Checking how many nodes the face has
    const int nfn = Core::FE::DisTypeToNumNodePerFace<distype>::numNodePerFace;

    // As already said, the dimension of the coordinate matrix is now nsd_-1
    // times the number of nodes in the face.
    Core::LinAlg::Matrix<nsd_ - 1, nfn> xsishuffle(Core::LinAlg::Initialization::zero);

    // Cycling through the nodes of the face to store the node positions in the
    // correct order using xsishuffle as a temporary vector
    for (int i = 0; i < nfn; ++i)
    {
      // cycling through the spatial dimensions
      for (unsigned int idim = 0; idim < nsd_ - 1; idim++)
      {
        // If the face belongs to the element being considered
        if (ele->faces()[f]->parent_master_element() == ele)
          xsishuffle(idim, i) = locations(idim, i);
        else
          // If the face does not belong to the element being considered it is
          // necessary to change the ordering
          xsishuffle(idim, ele->faces()[f]->get_local_trafo_map()[i]) = locations(idim, i);
      }
    }

    // EVALUATE SHAPE POLYNOMIALS IN NODE
    // Now that we have an ordered coordinates vector we can easily compute the
    // values of the shape functions in the nodes.
    for (int i = 0; i < nfn; ++i)
    {
      // Storing the actual coordinates of the current node
      for (unsigned int idim = 0; idim < nsd_ - 1; idim++)
        shapesface_->xsi(idim) = xsishuffle(idim, i);
      // Actually evaluating shape polynomials in node
      shapesface_->polySpace_->evaluate(shapesface_->xsi, fvalues);

      // compute values for the trace by summing over all basis functions
      {
        double sum_trace_density = 0.0;
        for (unsigned int k = 0; k < shapesface_->nfdofs_; ++k)
          sum_trace_density += fvalues(k) * solvalues[f * shapesface_->nfdofs_ * (1 + nsd_) + k];
        if (i < n_corner_nodes[f])
          elevec1((msd_ + 1 + nsd_) * nen_ + shapesface_->faceNodeOrder[f][i]) +=
              sum_trace_density / nsd_;
        else if (i < nfn - n_internal_nodes[f])
          elevec1((msd_ + 1 + nsd_) * nen_ + shapesface_->faceNodeOrder[f][i]) +=
              sum_trace_density / (nsd_ - 1);
        else
          elevec1((msd_ + 1 + nsd_) * nen_ + shapesface_->faceNodeOrder[f][i]) += sum_trace_density;
      }
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        double sum_trace_momentum = 0.0;
        for (unsigned int k = 0; k < shapesface_->nfdofs_; ++k)
        {
          sum_trace_momentum +=
              fvalues(k) *
              solvalues[f * shapesface_->nfdofs_ * (1 + nsd_) + (1 + d) * shapesface_->nfdofs_ + k];
        }
        if (i < n_corner_nodes[f])
          elevec1((msd_ + 1 + nsd_ + 1 + d) * nen_ + shapesface_->faceNodeOrder[f][i]) +=
              sum_trace_momentum / nsd_;
        else if (i < nfn - n_internal_nodes[f])
          elevec1((msd_ + 1 + nsd_ + 1 + d) * nen_ + shapesface_->faceNodeOrder[f][i]) +=
              sum_trace_momentum / (nsd_ - 1);
        else
          elevec1((msd_ + 1 + nsd_ + 1 + d) * nen_ + shapesface_->faceNodeOrder[f][i]) +=
              sum_trace_momentum;
      }
    }
  }

  return 0;
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::evaluate_all(const int funcnum,
    const Core::LinAlg::Matrix<nsd_, 1>& xyz, const double t, Core::LinAlg::Matrix<msd_, 1>& L,
    double& r, Core::LinAlg::Matrix<nsd_, 1>& w) const
{
  r = Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funcnum)
          .evaluate(xyz.as_span(), t, 0);

  for (unsigned int d = 0; d < nsd_; ++d)
    w(d) = Global::Problem::instance()
               ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funcnum)
               .evaluate(xyz.as_span(), t, 1 + d);

  for (unsigned int m = 0; m < msd_; ++m)
    L(m) = Global::Problem::instance()
               ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funcnum)
               .evaluate(xyz.as_span(), t, 1 + nsd_ + m);
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::evaluate_density_momentum(
    const int funcnum, const Core::LinAlg::Matrix<nsd_, 1>& xyz, const double t, double& r,
    Core::LinAlg::Matrix<nsd_, 1>& w) const
{
  r = Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funcnum)
          .evaluate(xyz.as_span(), t, 0);

  for (unsigned int d = 0; d < nsd_; ++d)
    w(d) = Global::Problem::instance()
               ->function_by_id<Core::Utils::FunctionOfSpaceTime>(funcnum)
               .evaluate(xyz.as_span(), t, 1 + d);
}



template <Core::FE::CellType distype>
Discret::Elements::FluidEleCalcHDGWeakComp<distype>*
Discret::Elements::FluidEleCalcHDGWeakComp<distype>::instance(Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<Discret::Elements::FluidEleCalcHDGWeakComp<distype>>(
            new Discret::Elements::FluidEleCalcHDGWeakComp<distype>());
      });

  return singleton_owner.instance(action);
}



template <Core::FE::CellType distype>
Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::LocalSolver(
    const Discret::Elements::Fluid* ele, const Core::FE::ShapeValues<distype>& shapeValues,
    Core::FE::ShapeValuesFace<distype>& shapeValuesFace, bool completepoly)
    : ndofs_(shapeValues.ndofs_),
      convective(true),
      unsteady(true),
      ale(ele->is_ale()),
      shapes_(shapeValues),
      shapesface_(shapeValuesFace)
{
  // count ndofs on faces
  ndofsfaces_ = 0;
  for (unsigned int i = 0; i < nfaces_; ++i)
  {
    shapesface_.evaluate_face(*ele, i);
    ndofsfaces_ += shapesface_.nfdofs_;
  }

  // initialize stabilization parameters
  tau_r = 0.0;
  tau_w = 0.0;

  // initialize auxiliary matrices
  massPart.shape(ndofs_, shapes_.nqpoints_);
  massPartW.shape(ndofs_, shapes_.nqpoints_);
  massMat.shape(ndofs_, ndofs_);

  // initialize unknowns
  Leg.shape(msd_, shapes_.nqpoints_);
  reg.size(shapes_.nqpoints_);
  weg.shape(nsd_, shapes_.nqpoints_);

  // initialize ALE variables
  aeg.shape(nsd_, shapes_.nqpoints_);
  dadxyzeg.shape(nsd_ * nsd_, shapes_.nqpoints_);

  // initialize matrices
  ALL.shape(msd_ * ndofs_, msd_ * ndofs_);
  ALr.shape(msd_ * ndofs_, 1 * ndofs_);
  ALw.shape(msd_ * ndofs_, nsd_ * ndofs_);
  ALR.shape(msd_ * ndofs_, 1 * ndofsfaces_);
  ALW.shape(msd_ * ndofs_, nsd_ * ndofsfaces_);
  Arr.shape(1 * ndofs_, 1 * ndofs_);
  Arw.shape(1 * ndofs_, nsd_ * ndofs_);
  ArR.shape(1 * ndofs_, 1 * ndofsfaces_);
  ArW.shape(1 * ndofs_, nsd_ * ndofsfaces_);
  AwL.shape(nsd_ * ndofs_, msd_ * ndofs_);
  Awr.shape(nsd_ * ndofs_, 1 * ndofs_);
  Aww.shape(nsd_ * ndofs_, nsd_ * ndofs_);
  AwR.shape(nsd_ * ndofs_, 1 * ndofsfaces_);
  AwW.shape(nsd_ * ndofs_, nsd_ * ndofsfaces_);
  ARr.shape(1 * ndofsfaces_, 1 * ndofs_);
  ARR.shape(1 * ndofsfaces_, 1 * ndofsfaces_);
  AWL.shape(nsd_ * ndofsfaces_, msd_ * ndofs_);
  AWw.shape(nsd_ * ndofsfaces_, nsd_ * ndofs_);
  AWR.shape(nsd_ * ndofsfaces_, 1 * ndofsfaces_);
  AWW.shape(nsd_ * ndofsfaces_, nsd_ * ndofsfaces_);

  // initialize residuals
  RL.size(msd_ * ndofs_);
  Rr.size(1 * ndofs_);
  Rw.size(nsd_ * ndofs_);
  RR.size(1 * ndofsfaces_);
  RW.size(nsd_ * ndofsfaces_);

  // initialize local/global matrices/vectors
  Klocallocal.shape((msd_ + 1 + nsd_) * ndofs_, (msd_ + 1 + nsd_) * ndofs_);
  Klocalglobal.shape((msd_ + 1 + nsd_) * ndofs_, (1 + nsd_) * ndofsfaces_);
  Kgloballocal.shape((1 + nsd_) * ndofsfaces_, (msd_ + 1 + nsd_) * ndofs_);
  Kglobalglobal.shape((1 + nsd_) * ndofsfaces_, (1 + nsd_) * ndofsfaces_);
  Rlocal.size((msd_ + 1 + nsd_) * ndofs_);
  Rglobal.size((1 + nsd_) * ndofsfaces_);
  KlocallocalInv.shape((msd_ + 1 + nsd_) * ndofs_, (msd_ + 1 + nsd_) * ndofs_);

  // pair of indices in Voigt notation
  int s = 0;
  for (unsigned int d = 0; d < nsd_; ++d)
    for (unsigned int e = d + 1; e < nsd_; ++e)
    {
      VoigtP[s][0] = d;
      VoigtP[s][1] = e;
      s = s + 1;
    }

  // pointer to class FluidEleParameter (access to the general parameter)
  fldparatimint_ =
      Core::Utils::shared_ptr_from_ref(*Discret::Elements::FluidEleParameterTimInt::instance());

  // initialize also general parameter list, also it will be overwritten in derived subclasses
  fldpara_ = Core::Utils::shared_ptr_from_ref(*Discret::Elements::FluidEleParameterStd::instance());
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::initialize_all()
{
  // initialize unknowns
  Leg.putScalar(0.0);
  reg.putScalar(0.0);
  weg.putScalar(0.0);

  // initialize ALE variables
  aeg.putScalar(0.0);
  dadxyzeg.putScalar(0.0);

  // initialize matrices
  ALL.putScalar(0.0);
  ALr.putScalar(0.0);
  ALw.putScalar(0.0);
  ALR.putScalar(0.0);
  ALW.putScalar(0.0);
  Arr.putScalar(0.0);
  Arw.putScalar(0.0);
  ArR.putScalar(0.0);
  ArW.putScalar(0.0);
  AwL.putScalar(0.0);
  Awr.putScalar(0.0);
  Aww.putScalar(0.0);
  AwR.putScalar(0.0);
  AwW.putScalar(0.0);
  ARr.putScalar(0.0);
  ARR.putScalar(0.0);
  AWL.putScalar(0.0);
  AWw.putScalar(0.0);
  AWR.putScalar(0.0);
  AWW.putScalar(0.0);

  // initialize residuals
  RL.putScalar(0.0);
  Rr.putScalar(0.0);
  Rw.putScalar(0.0);
  RR.putScalar(0.0);
  RW.putScalar(0.0);

  // initialize local/global matrices/vectors
  Klocallocal.putScalar(0.0);
  Klocalglobal.putScalar(0.0);
  Kgloballocal.putScalar(0.0);
  Kglobalglobal.putScalar(0.0);
  Rlocal.putScalar(0.0);
  Rglobal.putScalar(0.0);
  KlocallocalInv.putScalar(0.0);
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::compute_material_matrix(
    const std::shared_ptr<Core::Mat::Material>& mat, const Core::LinAlg::Matrix<nsd_, 1>& xyz,
    Core::LinAlg::SerialDenseMatrix& DL, Core::LinAlg::SerialDenseMatrix& Dw)
{
  // initialize DL and Dw
  DL.shape(msd_, msd_);
  Dw.shape(msd_, msd_);

  // evaluate D_fac
  Core::LinAlg::SerialDenseMatrix D_fac(msd_, msd_);
  for (unsigned int m = 0; m < msd_; ++m) D_fac(m, m) = 1.0;
  for (unsigned int d = 0; d < nsd_; ++d)
    for (unsigned int e = 0; e < nsd_; ++e) D_fac(d, e) -= 1.0 / 3.0;

  // check for variable viscosity
  const Teuchos::ParameterList& fluidparams = Global::Problem::instance()->fluid_dynamic_params();
  int varviscfuncnum = fluidparams.get<int>("VARVISCFUNCNO");

  // compute material matrix
  if (varviscfuncnum < 0)  // constant viscosity
  {
    // get material
    const Mat::WeaklyCompressibleFluid* actmat =
        static_cast<const Mat::WeaklyCompressibleFluid*>(mat.get());

    // get viscosity
    double mu = actmat->viscosity();

    // evaluate Dw
    for (unsigned int d = 0; d < nsd_; ++d) Dw(d, d) = std::pow(2.0 * mu, 1.0 / 2.0);
    for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
      Dw(nsd_ + s, nsd_ + s) = std::pow(mu, 1.0 / 2.0);

    // evaluate DL
    Core::LinAlg::multiply(DL, Dw, D_fac);
  }
  else  // variable viscosity
  {
    // get time
    const double time = fldparatimint_->time();

    // get viscosity
    double mu = Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(varviscfuncnum)
                    .evaluate(xyz.as_span(), time, 0);

    // evaluate Dw
    for (unsigned int d = 0; d < nsd_; ++d) Dw(d, d) = 2.0 * mu;
    for (unsigned int s = 0; s < (msd_ - nsd_); ++s) Dw(nsd_ + s, nsd_ + s) = mu;

    // evaluate DL
    DL = D_fac;
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::compute_interior_residual(
    const std::shared_ptr<Core::Mat::Material>& mat, const std::vector<double>& val,
    const std::vector<double>& accel, const std::vector<double>& alevel)
{
  // set convective flag
  Inpar::FLUID::PhysicalType physicaltype = fldpara_->physical_type();
  convective = (physicaltype != Inpar::FLUID::weakly_compressible_stokes_dens_mom);

  // set unsteady flag
  const Teuchos::ParameterList& fluidparams = Global::Problem::instance()->fluid_dynamic_params();
  auto timeintegr =
      Teuchos::getIntegralValue<Inpar::FLUID::TimeIntegrationScheme>(fluidparams, "TIMEINTEGR");
  unsteady = (timeintegr != Inpar::FLUID::TimeIntegrationScheme::timeint_stationary);

  // get material properties
  const Mat::WeaklyCompressibleFluid* actmat =
      static_cast<const Mat::WeaklyCompressibleFluid*>(mat.get());

  // get force function
  int forcefuncnum = fluidparams.get<int>("BODYFORCEFUNCNO");

  // get time
  const double time = fldparatimint_->time();

  // ease notation
  Core::LinAlg::SerialDenseMatrix N = shapes_.shfunct;
  Core::LinAlg::SerialDenseMatrix Nn = shapes_.funct;
  Core::LinAlg::SerialDenseMatrix Nxyz = shapes_.shderxy;
  Core::LinAlg::SerialDenseMatrix Nnxyz = shapes_.derxy;
  Core::LinAlg::SerialDenseVector fac = shapes_.jfac;
  unsigned int nqpoints = shapes_.nqpoints_;

  // extract interior nodal values
  Core::LinAlg::Matrix<nsd_, 1> xyze;
  Core::LinAlg::SerialDenseMatrix Le(msd_, ndofs_);
  Core::LinAlg::SerialDenseVector re(ndofs_);
  Core::LinAlg::SerialDenseMatrix we(nsd_, ndofs_);
  Core::LinAlg::SerialDenseVector pe(ndofs_);
  Core::LinAlg::SerialDenseVector drdte(ndofs_);
  Core::LinAlg::SerialDenseMatrix dwdte(nsd_, ndofs_);
  Core::LinAlg::SerialDenseMatrix DLe(msd_, msd_);
  Core::LinAlg::SerialDenseMatrix Dwe(msd_, msd_);
  Core::LinAlg::SerialDenseMatrix DwLe(msd_, ndofs_);

  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int d = 0; d < nsd_; ++d) xyze(d) = shapes_.nodexyzreal(d, i);

    for (unsigned int m = 0; m < msd_; ++m) Le(m, i) = val[m * ndofs_ + i];

    re(i) = val[msd_ * ndofs_ + i];

    for (unsigned int d = 0; d < nsd_; ++d) we(d, i) = val[(msd_ + 1 + d) * ndofs_ + i];

    pe(i) = actmat->compute_pressure(re[i]);

    drdte(i) = accel[msd_ * ndofs_ + i];

    for (unsigned int d = 0; d < nsd_; ++d) dwdte(d, i) = accel[(msd_ + 1 + d) * ndofs_ + i];

    // compute material matrix
    compute_material_matrix(mat, xyze, DLe, Dwe);

    for (unsigned int d = 0; d < nsd_; ++d)
      for (unsigned int e = 0; e < nsd_; ++e) DwLe(d, i) += Dwe(d, e) * Le(e, i);
    for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
      DwLe(nsd_ + s, i) += Dwe(nsd_ + s, nsd_ + s) * Le(nsd_ + s, i);
  }

  // extract ale nodal values
  Core::LinAlg::SerialDenseMatrix ae(nsd_, ndofs_);

  if (ale)
    for (unsigned int n = 0; n < nen_; ++n)
      for (unsigned int d = 0; d < nsd_; ++d) ae(d, n) = alevel[n * nsd_ + d];

  // initialize values interpolated on gauss points
  Core::LinAlg::Matrix<nsd_, 1> xyzeg;
  Core::LinAlg::SerialDenseMatrix feg(1 + nsd_, nqpoints);
  Core::LinAlg::SerialDenseVector drdteg(nqpoints);
  Core::LinAlg::SerialDenseMatrix dwdteg(nsd_, nqpoints);
  Core::LinAlg::SerialDenseMatrix dDwLdxyzeg(msd_ * nsd_, nqpoints);
  Core::LinAlg::SerialDenseMatrix dpdxyzeg(1 * nsd_, nqpoints);
  Core::LinAlg::SerialDenseMatrix DLeg(msd_, msd_);
  Core::LinAlg::SerialDenseMatrix Dweg(msd_, msd_);

  // loop over quadrature points
  for (unsigned int q = 0; q < nqpoints; ++q)
  {
    // interpolate values on gauss points
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int d = 0; d < nsd_; ++d) xyzeg(d) = shapes_.xyzreal(d, q);

      for (unsigned int m = 0; m < msd_; ++m) Leg(m, q) += N(i, q) * Le(m, i);

      reg(q) += N(i, q) * re(i);

      for (unsigned int d = 0; d < nsd_; ++d) weg(d, q) += N(i, q) * we(d, i);

      if (forcefuncnum > 0)
        for (unsigned int dmod = 0; dmod < (1 + nsd_); ++dmod)
          feg(dmod, q) = Global::Problem::instance()
                             ->function_by_id<Core::Utils::FunctionOfSpaceTime>(forcefuncnum)
                             .evaluate(xyzeg.as_span(), time, dmod);

      drdteg(q) += N(i, q) * drdte(i);

      for (unsigned int d = 0; d < nsd_; ++d) dwdteg(d, q) += N(i, q) * dwdte(d, i);

      for (unsigned int m = 0; m < msd_; ++m)
        for (unsigned int d = 0; d < nsd_; ++d)
          dDwLdxyzeg(m * nsd_ + d, q) += Nxyz(i * nsd_ + d, q) * DwLe(m, i);

      for (unsigned int d = 0; d < nsd_; ++d) dpdxyzeg(d, q) += Nxyz(i * nsd_ + d, q) * pe(i);
    }

    // compute material matrix
    compute_material_matrix(mat, xyzeg, DLeg, Dweg);

    // interpolate ale values on gauss points
    if (ale)
      for (unsigned int n = 0; n < nen_; ++n)
      {
        for (unsigned int d = 0; d < nsd_; ++d) aeg(d, q) += Nn(n, q) * ae(d, n);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            dadxyzeg(d * nsd_ + e, q) += Nnxyz(n * nsd_ + e, q) * ae(d, n);
      }

    // compute residuals
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int m = 0; m < msd_; ++m) RL(m * ndofs_ + i) += +N(i, q) * Leg(m, q) * fac(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        for (unsigned int e = 0; e < nsd_; ++e)
          RL(d * ndofs_ + i) += -Nxyz(i * nsd_ + e, q) * DLeg(e, d) * weg(e, q) / reg(q) * fac(q);
      for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        RL((nsd_ + s) * ndofs_ + i) += -Nxyz(i * nsd_ + VoigtP[s][0], q) *
                                           DLeg(nsd_ + s, nsd_ + s) * weg(VoigtP[s][1], q) /
                                           reg(q) * fac(q) -
                                       Nxyz(i * nsd_ + VoigtP[s][1], q) * DLeg(nsd_ + s, nsd_ + s) *
                                           weg(VoigtP[s][0], q) / reg(q) * fac(q);

      if (unsteady) Rr(i) += -N(i, q) * drdteg(q) * fac(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          Rr(i) += -N(i, q) * reg(q) * dadxyzeg(d * nsd_ + d, q) * fac(q);

      for (unsigned int d = 0; d < nsd_; ++d) Rr(i) += +Nxyz(i * nsd_ + d, q) * weg(d, q) * fac(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          Rr(i) += -Nxyz(i * nsd_ + d, q) * reg(q) * aeg(d, q) * fac(q);

      Rr(i) += +N(i, q) * feg(0, q) * fac(q);

      if (unsteady)
        for (unsigned int d = 0; d < nsd_; ++d)
          Rw(d * ndofs_ + i) += -N(i, q) * dwdteg(d, q) * fac(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            Rw(d * ndofs_ + i) += -N(i, q) * weg(d, q) * dadxyzeg(e * nsd_ + e, q) * fac(q);

      if (convective)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            Rw(d * ndofs_ + i) += +Nxyz(i * nsd_ + e, q) * weg(d, q) * weg(e, q) / reg(q) * fac(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            Rw(d * ndofs_ + i) += -Nxyz(i * nsd_ + e, q) * weg(d, q) * aeg(e, q) * fac(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rw(d * ndofs_ + i) += -N(i, q) * dDwLdxyzeg(d * nsd_ + d, q) * fac(q);
      for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
      {
        Rw(VoigtP[s][0] * ndofs_ + i) +=
            -N(i, q) * dDwLdxyzeg((nsd_ + s) * nsd_ + VoigtP[s][1], q) * fac(q);
        Rw(VoigtP[s][1] * ndofs_ + i) +=
            -N(i, q) * dDwLdxyzeg((nsd_ + s) * nsd_ + VoigtP[s][0], q) * fac(q);
      }

      for (unsigned int d = 0; d < nsd_; ++d)
        Rw(d * ndofs_ + i) += -N(i, q) * dpdxyzeg(d, q) * fac(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rw(d * ndofs_ + i) += +N(i, q) * feg(1 + d, q) * fac(q);
    }
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::compute_interior_matrices(
    const std::shared_ptr<Core::Mat::Material>& mat)
{
  // get material properties
  const Mat::WeaklyCompressibleFluid* actmat =
      static_cast<const Mat::WeaklyCompressibleFluid*>(mat.get());
  double eps = actmat->compr_coeff();

  // inverse of time factor
  const double invtimefac = 1.0 / (fldparatimint_->time_fac());

  // ease notation
  Core::LinAlg::SerialDenseMatrix N = shapes_.shfunct;
  Core::LinAlg::SerialDenseMatrix Nxyz = shapes_.shderxy;
  Core::LinAlg::SerialDenseVector fac = shapes_.jfac;
  unsigned int nqpoints = shapes_.nqpoints_;

  // extract interior nodal values
  Core::LinAlg::Matrix<nsd_, 1> xyze;
  Core::LinAlg::SerialDenseMatrix DLe(msd_, msd_);
  Core::LinAlg::SerialDenseMatrix Dwe(msd_, msd_);
  Core::LinAlg::SerialDenseMatrix Dwemod(nsd_ * nsd_ + (msd_ - nsd_), ndofs_);

  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int d = 0; d < nsd_; ++d) xyze(d) = shapes_.nodexyzreal(d, i);

    // compute material matrix
    compute_material_matrix(mat, xyze, DLe, Dwe);

    for (unsigned int d = 0; d < nsd_; ++d)
      for (unsigned int e = 0; e < nsd_; ++e) Dwemod(d * nsd_ + e, i) = Dwe(d, e);
    for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
      Dwemod(nsd_ * nsd_ + s, i) = Dwe(nsd_ + s, nsd_ + s);
  }

  // initialize values interpolated on gauss points
  Core::LinAlg::Matrix<nsd_, 1> xyzeg;
  Core::LinAlg::SerialDenseMatrix DLeg(msd_, msd_);
  Core::LinAlg::SerialDenseMatrix Dweg(msd_, msd_);
  Core::LinAlg::SerialDenseMatrix dDwdxyzeg((nsd_ * nsd_ + (msd_ - nsd_)) * nsd_, nqpoints);

  // loop over quadrature points
  for (unsigned int q = 0; q < nqpoints; ++q)
  {
    // interpolate values on gauss points
    for (unsigned int d = 0; d < nsd_; ++d) xyzeg(d) = shapes_.xyzreal(d, q);

    // compute material matrix
    compute_material_matrix(mat, xyzeg, DLeg, Dweg);

    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int mmod = 0; mmod < (nsd_ * nsd_ + (msd_ - nsd_)); ++mmod)
        for (unsigned int d = 0; d < nsd_; ++d)
          dDwdxyzeg(mmod * nsd_ + d, q) += Nxyz(i * nsd_ + d, q) * Dwemod(mmod, i);
    }

    // compute matrices
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int j = 0; j < ndofs_; ++j)
      {
        for (unsigned int m = 0; m < msd_; ++m)
          ALL(m * ndofs_ + i, m * ndofs_ + j) += -N(i, q) * N(j, q) * fac(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            ALr(d * ndofs_ + i, j) += -Nxyz(i * nsd_ + e, q) * DLeg(e, d) * weg(e, q) /
                                      std::pow(reg(q), 2.0) * N(j, q) * fac(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
          ALr((nsd_ + s) * ndofs_ + i, j) +=
              -Nxyz(i * nsd_ + VoigtP[s][0], q) * DLeg(nsd_ + s, nsd_ + s) * weg(VoigtP[s][1], q) /
                  std::pow(reg(q), 2.0) * N(j, q) * fac(q) -
              Nxyz(i * nsd_ + VoigtP[s][1], q) * DLeg(nsd_ + s, nsd_ + s) * weg(VoigtP[s][0], q) /
                  std::pow(reg(q), 2.0) * N(j, q) * fac(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            ALw(d * ndofs_ + i, e * ndofs_ + j) +=
                +Nxyz(i * nsd_ + e, q) * DLeg(e, d) / reg(q) * N(j, q) * fac(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        {
          ALw((nsd_ + s) * ndofs_ + i, VoigtP[s][0] * ndofs_ + j) +=
              +Nxyz(i * nsd_ + VoigtP[s][1], q) * DLeg(nsd_ + s, nsd_ + s) / reg(q) * N(j, q) *
              fac(q);
          ALw((nsd_ + s) * ndofs_ + i, VoigtP[s][1] * ndofs_ + j) +=
              +Nxyz(i * nsd_ + VoigtP[s][0], q) * DLeg(nsd_ + s, nsd_ + s) / reg(q) * N(j, q) *
              fac(q);
        }

        if (unsteady) Arr(i, j) += +invtimefac * N(i, q) * N(j, q) * fac(q);

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            Arr(i, j) += +N(i, q) * dadxyzeg(d * nsd_ + d, q) * N(j, q) * fac(q);

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            Arr(i, j) += +Nxyz(i * nsd_ + d, q) * aeg(d, q) * N(j, q) * fac(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          Arw(i, d * ndofs_ + j) += -Nxyz(i * nsd_ + d, q) * N(j, q) * fac(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            AwL(d * ndofs_ + i, e * ndofs_ + j) +=
                +N(i, q) * Dweg(d, e) * Nxyz(j * nsd_ + d, q) * fac(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        {
          AwL(VoigtP[s][0] * ndofs_ + i, (nsd_ + s) * ndofs_ + j) +=
              +N(i, q) * Dweg(nsd_ + s, nsd_ + s) * Nxyz(j * nsd_ + VoigtP[s][1], q) * fac(q);
          AwL(VoigtP[s][1] * ndofs_ + i, (nsd_ + s) * ndofs_ + j) +=
              +N(i, q) * Dweg(nsd_ + s, nsd_ + s) * Nxyz(j * nsd_ + VoigtP[s][0], q) * fac(q);
        }

        // TODO to test this term. It arises from the linearization of the variable viscosity term
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            AwL(d * ndofs_ + i, e * ndofs_ + j) +=
                +N(i, q) * dDwdxyzeg((d * nsd_ + e) * nsd_ + d, q) * N(j, q) * fac(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        {
          AwL(VoigtP[s][0] * ndofs_ + i, (nsd_ + s) * ndofs_ + j) +=
              +N(i, q) * dDwdxyzeg((nsd_ * nsd_ + s) * nsd_ + VoigtP[s][1], q) * N(j, q) * fac(q);
          AwL(VoigtP[s][1] * ndofs_ + i, (nsd_ + s) * ndofs_ + j) +=
              +N(i, q) * dDwdxyzeg((nsd_ * nsd_ + s) * nsd_ + VoigtP[s][0], q) * N(j, q) * fac(q);
        }

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              Awr(d * ndofs_ + i, j) += +Nxyz(i * nsd_ + e, q) * weg(d, q) * weg(e, q) /
                                        std::pow(reg(q), 2.0) * N(j, q) * fac(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          Awr(d * ndofs_ + i, j) += +1.0 / eps * N(i, q) * Nxyz(j * nsd_ + d, q) * fac(q);

        if (unsteady)
          for (unsigned int d = 0; d < nsd_; ++d)
            Aww(d * ndofs_ + i, d * ndofs_ + j) += +invtimefac * N(i, q) * N(j, q) * fac(q);

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              Aww(d * ndofs_ + i, d * ndofs_ + j) +=
                  +N(i, q) * dadxyzeg(e * nsd_ + e, q) * N(j, q) * fac(q);

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              Aww(d * ndofs_ + i, d * ndofs_ + j) +=
                  -Nxyz(i * nsd_ + e, q) * weg(e, q) / reg(q) * N(j, q) * fac(q);

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              Aww(d * ndofs_ + i, d * ndofs_ + j) +=
                  +Nxyz(i * nsd_ + e, q) * aeg(e, q) * N(j, q) * fac(q);

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              Aww(d * ndofs_ + i, e * ndofs_ + j) +=
                  -Nxyz(i * nsd_ + e, q) * weg(d, q) / reg(q) * N(j, q) * fac(q);
      }
    }
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::compute_face_residual(
    const int f, const std::shared_ptr<Core::Mat::Material>& mat, const std::vector<double>& val,
    const std::vector<double>& traceval, const std::vector<double>& alevel)
{
  // get material properties
  const Mat::WeaklyCompressibleFluid* actmat =
      static_cast<const Mat::WeaklyCompressibleFluid*>(mat.get());
  double eps = actmat->compr_coeff();

  // get stabilization parameters
  const Teuchos::ParameterList& fluidparams = Global::Problem::instance()->fluid_dynamic_params();
  double tau_r_ref = fluidparams.get<double>("STAB_DEN_REF");
  double tau_w_ref = fluidparams.get<double>("STAB_MOM_REF");
  tau_r = tau_r_ref / eps;
  tau_w = tau_w_ref;

  // ease notation
  Core::FE::ShapeValuesInteriorOnFace Ni = shapesface_.shfunctI;
  Core::LinAlg::SerialDenseMatrix Nf = shapesface_.shfunct;
  Core::LinAlg::SerialDenseMatrix Nn = shapesface_.funct;
  Core::LinAlg::SerialDenseMatrix nxyz = shapesface_.normals;
  Core::LinAlg::SerialDenseVector facf = shapesface_.jfac;
  unsigned int nfqpoints = shapesface_.nqpoints_;
  unsigned int nfdofs = shapesface_.nfdofs_;
  unsigned int nfn = shapesface_.nfn_;
  std::vector<std::vector<int>> faceNodeOrder = shapesface_.faceNodeOrder;

  // extract interior nodal values
  Core::LinAlg::SerialDenseMatrix Le(msd_, ndofs_);
  Core::LinAlg::SerialDenseVector re(ndofs_);
  Core::LinAlg::SerialDenseMatrix we(nsd_, ndofs_);
  Core::LinAlg::SerialDenseVector pe(ndofs_);

  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int m = 0; m < msd_; ++m) Le(m, i) = val[m * ndofs_ + i];

    re(i) = val[msd_ * ndofs_ + i];

    for (unsigned int d = 0; d < nsd_; ++d) we(d, i) = val[(msd_ + 1 + d) * ndofs_ + i];

    pe(i) = actmat->compute_pressure(re[i]);
  }

  // extract ale nodal values
  Core::LinAlg::SerialDenseMatrix aef(nsd_, nfn);

  if (ale)
    for (unsigned int n = 0; n < nfn; ++n)
      for (unsigned int d = 0; d < nsd_; ++d) aef(d, n) = alevel[faceNodeOrder[f][n] * nsd_ + d];

  // extract trace nodal value
  Core::LinAlg::SerialDenseVector rhatef(nfdofs);
  Core::LinAlg::SerialDenseMatrix whatef(nsd_, nfdofs);
  for (unsigned int i = 0; i < nfdofs; ++i)
  {
    rhatef(i) = traceval[f * nfdofs * (1 + nsd_) + i];

    for (unsigned int d = 0; d < nsd_; ++d)
      whatef(d, i) = traceval[f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i];
  }

  // initialize values interpolated on gauss points
  Core::LinAlg::Matrix<nsd_, 1> xyzefg;
  Core::LinAlg::SerialDenseMatrix Lefg(msd_, nfqpoints);
  Core::LinAlg::SerialDenseVector refg(nfqpoints);
  Core::LinAlg::SerialDenseMatrix wefg(nsd_, nfqpoints);
  Core::LinAlg::SerialDenseVector phatefg(nfqpoints);
  Core::LinAlg::SerialDenseMatrix DLefg(msd_, msd_);
  Core::LinAlg::SerialDenseMatrix Dwefg(msd_, msd_);
  rhatefg.size(nfqpoints);
  whatefg.shape(nsd_, nfqpoints);
  aefg.shape(nsd_, nfqpoints);

  // loop over quadrature points
  for (unsigned int q = 0; q < nfqpoints; ++q)
  {
    // interpolate interior values on gauss points
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int m = 0; m < msd_; ++m) Lefg(m, q) += Ni(i, q) * Le(m, i);

      refg(q) += Ni(i, q) * re(i);

      for (unsigned int d = 0; d < nsd_; ++d) wefg(d, q) += Ni(i, q) * we(d, i);
    }

    // interpolate ale values on gauss points
    if (ale)
      for (unsigned int n = 0; n < nfn; ++n)
        for (unsigned int d = 0; d < nsd_; ++d) aefg(d, q) += Nn(n, q) * aef(d, n);

    // interpolate trace values on gauss points
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      rhatefg(q) += Nf(i, q) * rhatef(i);

      for (unsigned int d = 0; d < nsd_; ++d) whatefg(d, q) += Nf(i, q) * whatef(d, i);

      phatefg(q) = actmat->compute_pressure(rhatefg(q));
    }

    // interpolate values on gauss points
    for (unsigned int d = 0; d < nsd_; ++d) xyzefg(d) = shapesface_.xyzreal(d, q);

    // compute material matrix
    compute_material_matrix(mat, xyzefg, DLefg, Dwefg);

    // compute residuals (interior contributions)
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int d = 0; d < nsd_; ++d)
        for (unsigned int e = 0; e < nsd_; ++e)
          RL(d * ndofs_ + i) +=
              +Ni(i, q) * nxyz(e, q) * DLefg(e, d) * whatefg(e, q) / rhatefg(q) * facf(q);
      for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        RL((nsd_ + s) * ndofs_ + i) +=
            +Ni(i, q) * nxyz(VoigtP[s][0], q) * DLefg(nsd_ + s, nsd_ + s) *
                whatefg(VoigtP[s][1], q) / rhatefg(q) * facf(q) +
            Ni(i, q) * nxyz(VoigtP[s][1], q) * DLefg(nsd_ + s, nsd_ + s) *
                whatefg(VoigtP[s][0], q) / rhatefg(q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rr(i) += -Ni(i, q) * whatefg(d, q) * nxyz(d, q) * facf(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          Rr(i) += +Ni(i, q) * rhatefg(q) * aefg(d, q) * nxyz(d, q) * facf(q);

      Rr(i) += -Ni(i, q) * tau_r * refg(q) * facf(q);

      Rr(i) += +Ni(i, q) * tau_r * rhatefg(q) * facf(q);

      if (convective)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            Rw(d * ndofs_ + i) +=
                -Ni(i, q) * whatefg(d, q) * whatefg(e, q) / rhatefg(q) * nxyz(e, q) * facf(q);

      if (ale)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            Rw(d * ndofs_ + i) += +Ni(i, q) * whatefg(d, q) * aefg(e, q) * nxyz(e, q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rw(d * ndofs_ + i) += -Ni(i, q) * tau_w * wefg(d, q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rw(d * ndofs_ + i) += +Ni(i, q) * tau_w * whatefg(d, q) * facf(q);
    }

    // compute residuals (face contributions)
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      RR(f * nfdofs + i) += -Nf(i, q) * tau_r * refg(q) * facf(q);

      RR(f * nfdofs + i) += +Nf(i, q) * tau_r * rhatefg(q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        for (unsigned int e = 0; e < nsd_; ++e)
          RW((f * nsd_ + d) * nfdofs + i) +=
              -Nf(i, q) * Dwefg(d, e) * Lefg(e, q) * nxyz(d, q) * facf(q);
      for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
      {
        RW((f * nsd_ + VoigtP[s][0]) * nfdofs + i) += -Nf(i, q) * Dwefg(nsd_ + s, nsd_ + s) *
                                                      Lefg(nsd_ + s, q) * nxyz(VoigtP[s][1], q) *
                                                      facf(q);
        RW((f * nsd_ + VoigtP[s][1]) * nfdofs + i) += -Nf(i, q) * Dwefg(nsd_ + s, nsd_ + s) *
                                                      Lefg(nsd_ + s, q) * nxyz(VoigtP[s][0], q) *
                                                      facf(q);
      }

      for (unsigned int d = 0; d < nsd_; ++d)
        RW((f * nsd_ + d) * nfdofs + i) += -Nf(i, q) * phatefg(q) * nxyz(d, q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        RW((f * nsd_ + d) * nfdofs + i) += -Nf(i, q) * tau_w * wefg(d, q) * facf(q);

      for (unsigned int d = 0; d < nsd_; ++d)
        RW((f * nsd_ + d) * nfdofs + i) += +Nf(i, q) * tau_w * whatefg(d, q) * facf(q);

      //  TODO Neumann term
    }
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::compute_face_matrices(
    const int f, const std::shared_ptr<Core::Mat::Material>& mat)
{
  // get material properties
  const Mat::WeaklyCompressibleFluid* actmat =
      static_cast<const Mat::WeaklyCompressibleFluid*>(mat.get());
  double eps = actmat->compr_coeff();

  // ease notation
  Core::FE::ShapeValuesInteriorOnFace Ni = shapesface_.shfunctI;
  Core::LinAlg::SerialDenseMatrix Nf = shapesface_.shfunct;
  Core::LinAlg::SerialDenseMatrix nxyz = shapesface_.normals;
  Core::LinAlg::SerialDenseVector facf = shapesface_.jfac;
  unsigned int nfqpoints = shapesface_.nqpoints_;
  unsigned int nfdofs = shapesface_.nfdofs_;

  // initialize values interpolated on gauss points
  Core::LinAlg::Matrix<nsd_, 1> xyzefg;
  Core::LinAlg::SerialDenseMatrix DLefg(msd_, msd_);
  Core::LinAlg::SerialDenseMatrix Dwefg(msd_, msd_);

  // loop over quadrature points
  for (unsigned int q = 0; q < nfqpoints; ++q)
  {
    // interpolate values on gauss points
    for (unsigned int d = 0; d < nsd_; ++d) xyzefg(d) = shapesface_.xyzreal(d, q);

    // compute material matrix
    compute_material_matrix(mat, xyzefg, DLefg, Dwefg);

    // compute matrices (interior - interior contributions)
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int j = 0; j < ndofs_; ++j)
      {
        Arr(i, j) += +Ni(i, q) * tau_r * Ni(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          Aww(d * ndofs_ + i, d * ndofs_ + j) += +Ni(i, q) * tau_w * Ni(j, q) * facf(q);
      }
    }

    // compute matrices (interior - face contributions)
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int j = 0; j < nfdofs; ++j)
      {
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            ALR(d * ndofs_ + i, f * nfdofs + j) += +Ni(i, q) * nxyz(e, q) * DLefg(e, d) *
                                                   whatefg(e, q) / std::pow(rhatefg(q), 2.0) *
                                                   Nf(j, q) * facf(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
          ALR((nsd_ + s) * ndofs_ + i, f * nfdofs + j) +=
              +Ni(i, q) * nxyz(VoigtP[s][0], q) * DLefg(nsd_ + s, nsd_ + s) *
                  whatefg(VoigtP[s][1], q) / std::pow(rhatefg(q), 2.0) * Nf(j, q) * facf(q) +
              Ni(i, q) * nxyz(VoigtP[s][1], q) * DLefg(nsd_ + s, nsd_ + s) *
                  whatefg(VoigtP[s][0], q) / std::pow(rhatefg(q), 2.0) * Nf(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            ALW(d * ndofs_ + i, (f * nsd_ + e) * nfdofs + j) +=
                -Ni(i, q) * nxyz(e, q) * DLefg(e, d) / rhatefg(q) * Nf(j, q) * facf(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        {
          ALW((nsd_ + s) * ndofs_ + i, (f * nsd_ + VoigtP[s][0]) * nfdofs + j) +=
              -Ni(i, q) * nxyz(VoigtP[s][1], q) * DLefg(nsd_ + s, nsd_ + s) / rhatefg(q) *
              Nf(j, q) * facf(q);
          ALW((nsd_ + s) * ndofs_ + i, (f * nsd_ + VoigtP[s][1]) * nfdofs + j) +=
              -Ni(i, q) * nxyz(VoigtP[s][0], q) * DLefg(nsd_ + s, nsd_ + s) / rhatefg(q) *
              Nf(j, q) * facf(q);
        }

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            ArR(i, f * nfdofs + j) += -Ni(i, q) * aefg(d, q) * nxyz(d, q) * Nf(j, q) * facf(q);

        ArR(i, f * nfdofs + j) += -Ni(i, q) * tau_r * Nf(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          ArW(i, (f * nsd_ + d) * nfdofs + j) += +Ni(i, q) * nxyz(d, q) * Nf(j, q) * facf(q);

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              AwR(d * ndofs_ + i, f * nfdofs + j) += -Ni(i, q) * whatefg(d, q) * whatefg(e, q) /
                                                     std::pow(rhatefg(q), 2.0) * nxyz(e, q) *
                                                     Nf(j, q) * facf(q);

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              AwW(d * ndofs_ + i, (f * nsd_ + d) * nfdofs + j) +=
                  +Ni(i, q) * whatefg(e, q) / rhatefg(q) * nxyz(e, q) * Nf(j, q) * facf(q);

        if (ale)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              AwW(d * ndofs_ + i, (f * nsd_ + d) * nfdofs + j) +=
                  -Ni(i, q) * aefg(e, q) * nxyz(e, q) * Nf(j, q) * facf(q);

        if (convective)
          for (unsigned int d = 0; d < nsd_; ++d)
            for (unsigned int e = 0; e < nsd_; ++e)
              AwW(d * ndofs_ + i, (f * nsd_ + e) * nfdofs + j) +=
                  +Ni(i, q) * whatefg(d, q) / rhatefg(q) * nxyz(e, q) * Nf(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          AwW(d * ndofs_ + i, (f * nsd_ + d) * nfdofs + j) +=
              -Ni(i, q) * tau_w * Nf(j, q) * facf(q);
      }
    }

    // compute matrices (face - interior contributions)
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      for (unsigned int j = 0; j < ndofs_; ++j)
      {
        ARr(f * nfdofs + i, j) += +Nf(i, q) * tau_r * Ni(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            AWL((f * nsd_ + d) * nfdofs + i, e * ndofs_ + j) +=
                +Nf(i, q) * nxyz(d, q) * Dwefg(d, e) * Ni(j, q) * facf(q);
        for (unsigned int s = 0; s < (msd_ - nsd_); ++s)
        {
          AWL((f * nsd_ + VoigtP[s][0]) * nfdofs + i, (nsd_ + s) * ndofs_ + j) +=
              +Nf(i, q) * nxyz(VoigtP[s][1], q) * Dwefg(nsd_ + s, nsd_ + s) * Ni(j, q) * facf(q);
          AWL((f * nsd_ + VoigtP[s][1]) * nfdofs + i, (nsd_ + s) * ndofs_ + j) +=
              +Nf(i, q) * nxyz(VoigtP[s][0], q) * Dwefg(nsd_ + s, nsd_ + s) * Ni(j, q) * facf(q);
        }

        for (unsigned int d = 0; d < nsd_; ++d)
          AWw((f * nsd_ + d) * nfdofs + i, d * ndofs_ + j) +=
              +Nf(i, q) * tau_w * Ni(j, q) * facf(q);
      }
    }

    // compute matrices (face - face contributions)
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      for (unsigned int j = 0; j < nfdofs; ++j)
      {
        ARR(f * nfdofs + i, f * nfdofs + j) += -Nf(i, q) * tau_r * Nf(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          AWR((f * nsd_ + d) * nfdofs + i, f * nfdofs + j) +=
              +Nf(i, q) * nxyz(d, q) / eps * Nf(j, q) * facf(q);

        for (unsigned int d = 0; d < nsd_; ++d)
          AWW((f * nsd_ + d) * nfdofs + i, (f * nsd_ + d) * nfdofs + j) +=
              -Nf(i, q) * tau_w * Nf(j, q) * facf(q);
      }
    }
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::compute_local_residual()
{
  // fill vector
  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int m = 0; m < msd_; ++m) Rlocal(m * ndofs_ + i) = RL(m * ndofs_ + i);

    Rlocal(msd_ * ndofs_ + i) = Rr(i);

    for (unsigned int d = 0; d < nsd_; ++d)
      Rlocal((msd_ + 1 + d) * ndofs_ + i) = Rw(d * ndofs_ + i);
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::compute_global_residual(
    Discret::Elements::Fluid& ele)
{
  // loop in faces
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // evaluate face
    shapesface_.evaluate_face(ele, f);

    // ease notation
    unsigned int nfdofs = shapesface_.nfdofs_;

    // fill vector
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      Rglobal(f * nfdofs * (1 + nsd_) + i) = RR(f * nfdofs + i);

      for (unsigned int d = 0; d < nsd_; ++d)
        Rglobal(f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i) = RW((f * nsd_ + d) * nfdofs + i);
    }
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::compute_local_local_matrix()
{
  // fill matrix
  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int j = 0; j < ndofs_; ++j)
    {
      for (unsigned int m = 0; m < msd_; ++m)
      {
        for (unsigned int n = 0; n < msd_; ++n)
          Klocallocal(m * ndofs_ + i, n * ndofs_ + j) = ALL(m * ndofs_ + i, n * ndofs_ + j);

        Klocallocal(m * ndofs_ + i, msd_ * ndofs_ + j) = ALr(m * ndofs_ + i, j);

        for (unsigned int d = 0; d < nsd_; ++d)
          Klocallocal(m * ndofs_ + i, (msd_ + 1 + d) * ndofs_ + j) =
              ALw(m * ndofs_ + i, d * ndofs_ + j);
      }

      Klocallocal(msd_ * ndofs_ + i, msd_ * ndofs_ + j) = Arr(i, j);

      for (unsigned int d = 0; d < nsd_; ++d)
        Klocallocal(msd_ * ndofs_ + i, (msd_ + 1 + d) * ndofs_ + j) = Arw(i, d * ndofs_ + j);

      for (unsigned int d = 0; d < nsd_; ++d)
      {
        for (unsigned int m = 0; m < msd_; ++m)
          Klocallocal((msd_ + 1 + d) * ndofs_ + i, m * ndofs_ + j) =
              AwL(d * ndofs_ + i, m * ndofs_ + j);

        Klocallocal((msd_ + 1 + d) * ndofs_ + i, msd_ * ndofs_ + j) = Awr(d * ndofs_ + i, j);

        for (unsigned int e = 0; e < nsd_; ++e)
          Klocallocal((msd_ + 1 + d) * ndofs_ + i, (msd_ + 1 + e) * ndofs_ + j) =
              Aww(d * ndofs_ + i, e * ndofs_ + j);
      }
    }
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::compute_local_global_matrix(
    Discret::Elements::Fluid& ele)
{
  // loop in faces
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // evaluate face
    shapesface_.evaluate_face(ele, f);

    // ease notation
    unsigned int nfdofs = shapesface_.nfdofs_;

    // fill matrix
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      for (unsigned int j = 0; j < nfdofs; ++j)
      {
        for (unsigned int m = 0; m < msd_; ++m)
        {
          Klocalglobal(m * ndofs_ + i, f * nfdofs * (1 + nsd_) + j) =
              ALR(m * ndofs_ + i, f * nfdofs + j);

          for (unsigned int d = 0; d < nsd_; ++d)
            Klocalglobal(m * ndofs_ + i, f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + j) =
                ALW(m * ndofs_ + i, (f * nsd_ + d) * nfdofs + j);
        }

        Klocalglobal(msd_ * ndofs_ + i, f * nfdofs * (1 + nsd_) + j) = ArR(i, f * nfdofs + j);

        for (unsigned int d = 0; d < nsd_; ++d)
          Klocalglobal(msd_ * ndofs_ + i, f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + j) =
              ArW(i, (f * nsd_ + d) * nfdofs + j);

        for (unsigned int d = 0; d < nsd_; ++d)
        {
          Klocalglobal((msd_ + 1 + d) * ndofs_ + i, f * nfdofs * (1 + nsd_) + j) =
              AwR(d * ndofs_ + i, f * nfdofs + j);

          for (unsigned int e = 0; e < nsd_; ++e)
            Klocalglobal(
                (msd_ + 1 + d) * ndofs_ + i, f * nfdofs * (1 + nsd_) + (1 + e) * nfdofs + j) =
                AwW(d * ndofs_ + i, (f * nsd_ + e) * nfdofs + j);
        }
      }
    }
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::compute_global_local_matrix(
    Discret::Elements::Fluid& ele)
{
  // loop in faces
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // evaluate face
    shapesface_.evaluate_face(ele, f);

    // ease notation
    unsigned int nfdofs = shapesface_.nfdofs_;

    // fill matrix
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      for (unsigned int j = 0; j < ndofs_; ++j)
      {
        Kgloballocal(f * nfdofs * (1 + nsd_) + i, msd_ * ndofs_ + j) = ARr(f * nfdofs + i, j);

        for (unsigned int d = 0; d < nsd_; ++d)
        {
          for (unsigned int m = 0; m < msd_; ++m)
            Kgloballocal(f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i, m * ndofs_ + j) =
                AWL((f * nsd_ + d) * nfdofs + i, m * ndofs_ + j);

          for (unsigned int e = 0; e < nsd_; ++e)
            Kgloballocal(f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i,
                (msd_ + 1 + e) * ndofs_ + j) = AWw((f * nsd_ + d) * nfdofs + i, e * ndofs_ + j);
        }
      }
    }
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::compute_global_global_matrix(
    Discret::Elements::Fluid& ele)
{
  // loop in faces
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // evaluate face
    shapesface_.evaluate_face(ele, f);

    // ease notation
    unsigned int nfdofs = shapesface_.nfdofs_;

    // fill matrix
    for (unsigned int i = 0; i < nfdofs; ++i)
    {
      for (unsigned int j = 0; j < nfdofs; ++j)
      {
        Kglobalglobal(f * nfdofs * (1 + nsd_) + i, f * nfdofs * (1 + nsd_) + j) =
            ARR(f * nfdofs + i, f * nfdofs + j);

        for (unsigned int d = 0; d < nsd_; ++d)
        {
          Kglobalglobal(f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i,
              f * nfdofs * (1 + nsd_) + j) = AWR((f * nsd_ + d) * nfdofs + i, f * nfdofs + j);

          for (unsigned int e = 0; e < nsd_; ++e)
            Kglobalglobal(f * nfdofs * (1 + nsd_) + (1 + d) * nfdofs + i,
                f * nfdofs * (1 + nsd_) + (1 + e) * nfdofs + j) =
                AWW((f * nsd_ + d) * nfdofs + i, (f * nsd_ + e) * nfdofs + j);
        }
      }
    }
  }
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::invert_local_local_matrix()
{
  KlocallocalInv = Klocallocal;
  KlocallocalInvSolver.setMatrix(Teuchos::rcpFromRef(KlocallocalInv.base()));
  int err = KlocallocalInvSolver.invert();
  if (err != 0) FOUR_C_THROW("Inversion of local-local matrix failed with errorcode {}", err);
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::condense_local_residual(
    Core::LinAlg::SerialDenseVector& eleVec)
{
  // initialize element vector
  eleVec.size((1 + nsd_) * ndofsfaces_);

  // create auxiliary vector
  Core::LinAlg::SerialDenseVector eleVecAux;
  eleVecAux.size((msd_ + 1 + nsd_) * ndofs_);

  // compute element vector
  Core::LinAlg::multiply(eleVecAux, KlocallocalInv, Rlocal);
  eleVec = Rglobal;
  Core::LinAlg::multiply(1.0, eleVec, -1.0, Kgloballocal, eleVecAux);
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::condense_local_matrix(
    Core::LinAlg::SerialDenseMatrix& eleMat)
{
  // initialize element matrix
  eleMat.shape((1 + nsd_) * ndofsfaces_, (1 + nsd_) * ndofsfaces_);

  // create auxiliary matrix
  Core::LinAlg::SerialDenseMatrix eleMatAux;
  eleMatAux.shape((msd_ + 1 + nsd_) * ndofs_, (1 + nsd_) * ndofsfaces_);

  // compute element matrix
  Core::LinAlg::multiply(eleMatAux, KlocallocalInv, Klocalglobal);
  eleMat = Kglobalglobal;
  Core::LinAlg::multiply(1.0, eleMat, -1.0, Kgloballocal, eleMatAux);
}



template <Core::FE::CellType distype>
void Discret::Elements::FluidEleCalcHDGWeakComp<distype>::LocalSolver::print_matrices_and_residuals(
    Discret::Elements::Fluid& ele, Core::LinAlg::SerialDenseVector& eleVec,
    Core::LinAlg::SerialDenseMatrix& eleMat)
{
  // element
  std::cout << "\n\n Element number = " << ele.id() + 1;

  // matrices
  std::cout << "\n\n ALL = \n\n";
  std::cout << ALL;
  std::cout << "\n\n ALr = \n\n";
  std::cout << ALr;
  std::cout << "\n\n ALw = \n\n";
  std::cout << ALw;
  std::cout << "\n\n ALR = \n\n";
  std::cout << ALR;
  std::cout << "\n\n ALW = \n\n";
  std::cout << ALW;
  std::cout << "\n\n Arr = \n\n";
  std::cout << Arr;
  std::cout << "\n\n Arw = \n\n";
  std::cout << Arw;
  std::cout << "\n\n ArR = \n\n";
  std::cout << ArR;
  std::cout << "\n\n ArW = \n\n";
  std::cout << ArW;
  std::cout << "\n\n AwL = \n\n";
  std::cout << AwL;
  std::cout << "\n\n Awr = \n\n";
  std::cout << Awr;
  std::cout << "\n\n Aww = \n\n";
  std::cout << Aww;
  std::cout << "\n\n AwR = \n\n";
  std::cout << AwR;
  std::cout << "\n\n AwW = \n\n";
  std::cout << AwW;
  std::cout << "\n\n ARr = \n\n";
  std::cout << ARr;
  std::cout << "\n\n ARR = \n\n";
  std::cout << ARR;
  std::cout << "\n\n AWL = \n\n";
  std::cout << AWL;
  std::cout << "\n\n AWw = \n\n";
  std::cout << AWw;
  std::cout << "\n\n AWR = \n\n";
  std::cout << AWR;
  std::cout << "\n\n AWW = \n\n";
  std::cout << AWW;

  // residuals
  std::cout << "\n\n RL = \n\n";
  RL.print(std::cout);
  std::cout << "\n\n Rr = \n\n";
  Rr.print(std::cout);
  std::cout << "\n\n Rw = \n\n";
  Rw.print(std::cout);
  std::cout << "\n\n RR = \n\n";
  RR.print(std::cout);
  std::cout << "\n\n RW = \n\n";
  RW.print(std::cout);

  // local/global matrices/vectors
  std::cout << "\n\n Klocallocal = \n\n";
  std::cout << Klocallocal;
  std::cout << "\n\n Klocalglobal = \n\n";
  std::cout << Klocalglobal;
  std::cout << "\n\n Kgloballocal = \n\n";
  std::cout << Kgloballocal;
  std::cout << "\n\n Kglobalglobal = \n\n";
  std::cout << Kglobalglobal;
  std::cout << "\n\n Rlocal = \n\n";
  Rlocal.print(std::cout);
  std::cout << "\n\n Rglobal = \n\n";
  Rglobal.print(std::cout);
  std::cout << "\n\n KlocallocalInv = \n\n";
  std::cout << KlocallocalInv;

  // element vector and matrix
  std::cout << "\n\n eleVec = \n\n";
  eleVec.print(std::cout);
  std::cout << "\n\n eleMat = \n\n";
  std::cout << eleMat;
}



// explicit instantiation of template classes
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::hex8>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::hex20>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::hex27>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::tet4>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::tet10>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::wedge6>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::wedge15>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::pyramid5>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::quad4>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::quad8>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::quad9>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::tri3>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::tri6>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::nurbs9>;
template class Discret::Elements::FluidEleCalcHDGWeakComp<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
