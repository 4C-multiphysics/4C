// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_elemag_ele_calc.hpp"

#include "4C_elemag_ele_action.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_electromagnetic.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_SerialDenseSolver.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 * Constructor
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ElemagEleCalc<distype>::ElemagEleCalc()
{
}

/*----------------------------------------------------------------------*
 * Action type: Evaluate
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ElemagEleCalc<distype>::evaluate(Discret::Elements::Elemag* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra, const Core::FE::GaussIntegration&,
    bool offdiag)
{
  return this->evaluate(ele, discretization, lm, params, mat, elemat1_epetra, elemat2_epetra,
      elevec1_epetra, elevec2_epetra, elevec3_epetra, offdiag);
}

/*----------------------------------------------------------------------*
 * Evaluate
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ElemagEleCalc<distype>::evaluate(Discret::Elements::Elemag* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix&,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector&, bool offdiag)
{
  // check if this is an hdg element and init completepoly
  if (const Discret::Elements::Elemag* hdgele = dynamic_cast<const Discret::Elements::Elemag*>(ele))
    usescompletepoly_ = hdgele->uses_complete_polynomial_space();
  else
    FOUR_C_THROW("cannot cast element to elemag element");
  EleMag::Action action;
  if (params.get<bool>("hdg_action", false))
  {
    switch (Teuchos::getIntegralValue<Core::FE::HDGAction>(params, "action"))
    {
      case Core::FE::HDGAction::project_dirich_field:
        action = EleMag::Action::project_dirich_field;
        break;
      default:
        FOUR_C_THROW("HDG Action type not supported");
    }
  }
  else
  {
    action = Teuchos::getIntegralValue<EleMag::Action>(params, "action");
  }

  initialize_shapes(ele);

  bool updateonly = false;
  shapes_->evaluate(*ele);
  switch (action)
  {
    case EleMag::project_field:
    {
      local_solver_->project_field(ele, params, elevec1, elevec2);
      break;
    }
    case EleMag::compute_error:
    {
      local_solver_->compute_error(ele, params, elevec1);
      break;
    }
    case EleMag::project_field_test:
    {
      local_solver_->project_field_test(ele, params, elevec1, elevec2);
      break;
    }
    case EleMag::project_field_test_trace:
    {
      local_solver_->project_field_test_trace(ele, params, elevec1);

      break;
    }
    case EleMag::project_dirich_field:
    {
      // if (mat->material_type() != Core::Materials::m_electromagneticmat)
      //  FOUR_C_THROW("for physical type 'lossless' please supply MAT_Electromagnetic");
      if (params.isParameter("faceconsider"))
      {
        // ElementInit(ele, params);
        local_solver_->project_dirich_field(ele, params, elevec1);
      }
      break;
    }
    case EleMag::ele_init:
    {
      element_init(ele, params);
      break;
    }
    case EleMag::fill_restart_vecs:
    {
      // bool padapty = params.get<bool>("padaptivity");
      read_global_vectors(ele, discretization, lm);
      fill_restart_vectors(ele, discretization);
      break;
    }
    case EleMag::ele_init_from_restart:
    {
      element_init_from_restart(ele, discretization);
      break;
    }
    case EleMag::interpolate_hdg_to_node:
    {
      read_global_vectors(ele, discretization, lm);
      interpolate_solution_to_nodes(ele, discretization, elevec1);
      break;
    }
    case EleMag::calc_abc:
    {
      int face = params.get<int>("face");
      int sumindex = 0;
      for (int i = 0; i < face; ++i)
      {
        Core::FE::PolynomialSpaceParams params(Core::FE::DisTypeToFaceShapeType<distype>::shape,
            ele->faces()[i]->degree(), usescompletepoly_);
        int nfdofs = Core::FE::PolynomialSpaceCache<nsd_ - 1>::instance().create(params)->size();
        sumindex += nfdofs;
      }
      read_global_vectors(ele, discretization, lm);
      if (!params.isParameter("nodeindices"))
        local_solver_->compute_absorbing_bc(
            discretization, ele, params, mat, face, elemat1, sumindex, elevec1);
      else
        FOUR_C_THROW("why would you set an absorbing LINE in THREE dimensions?");

      break;
    }
    /*
    case EleMag::bd_integrate:
    {
      int face = params.get<int>("face");
      localSolver_->compute_boundary_integral(ele, params, face);

      break;
    }
    */
    case EleMag::calc_systemmat_and_residual:
    {
      // const bool resonly = params.get<bool>("resonly");
      // const bool padapty = params.get<bool>("padaptivity");
      const double dt = params.get<double>("dt");
      const double tau = params.get<double>("tau");
      dyna_ = params.get<Inpar::EleMag::DynamicType>("dynamic type");

      read_global_vectors(ele, discretization, lm);
      elevec1.putScalar(0.0);
      local_solver_->compute_matrices(discretization, mat, *ele, dt, dyna_, tau);

      // if (!resonly)
      local_solver_->condense_local_part(elemat1);

      local_solver_->compute_residual(params, elevec1, *ele);

      break;
    }
    case EleMag::update_secondary_solution:
      updateonly = true;  // no break here!!!
      [[fallthrough]];
    case EleMag::update_secondary_solution_and_calc_residual:
    {
      // bool errormaps = params.get<bool>("errormaps");
      bool errormaps = false;
      // const bool allelesequal = params.get<bool>("allelesequal");

      const double dt = params.get<double>("dt");
      const double tau = params.get<double>("tau");
      dyna_ = params.get<Inpar::EleMag::DynamicType>("dynamic type");

      read_global_vectors(ele, discretization, lm);

      elevec1.putScalar(0.0);
      local_solver_->compute_matrices(discretization, mat, *ele, dt, dyna_, tau);
      /* Could be useful for optimization purposes
      if(!allelesequal)
        localSolver_->ComputeMatrices(discretization, mat, *ele, dt, dyna_);
      */

      update_interior_variables_and_compute_residual(
          params, *ele, *mat, elevec1, dt, errormaps, updateonly);

      break;
    }
    case EleMag::get_gauss_points:
    {
      int rows = shapes_->xyzreal.numRows();
      int cols = shapes_->xyzreal.numCols();
      elemat1.shape(rows, cols);

      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) elemat1(r, c) = shapes_->xyzreal(r, c);

      break;
    }
    default:
    {
      std::cout << "Action: " << action << std::endl;
      FOUR_C_THROW("unknown action supplied");
      break;
    }
  }  // switch(action)

  return 0;
}

/*----------------------------------------------------------------------*
 * Print trace
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::print_trace(Core::Elements::Element* ele)
{
  std::cout << "Local trace of element: " << ele->lid() << std::endl;
  std::cout << "Number of entries: " << localtrace_.size() << std::endl;
  std::cout << "Number of spatial dimensions: " << nsd_ << std::endl;
  std::cout << "Numer of faces: " << nfaces_ << std::endl;
  std::cout << "Numer of DOF per face: " << ele->num_dof_per_face(0) << std::endl;
  unsigned int index = 0;
  unsigned int second_index = 0;
  for (std::vector<double>::iterator iter = localtrace_.begin(); iter != localtrace_.end();
       iter++, index++, second_index++)
  {
    if (index % ele->num_dof_per_face(0) == 0)
    {
      std::cout << "Face number: " << index / ele->num_dof_per_face(0) << std::endl;
      second_index = 0;
    }
    if (second_index % shapesface_->nfdofs_ == 0)
      std::cout << "\tField component: " << second_index / shapesface_->nfdofs_ << std::endl;
    std::cout << "\t\t" << *iter << std::endl;
  }
  return;
}

template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::initialize_shapes(
    const Discret::Elements::Elemag* ele)
{
  if (shapes_ == nullptr)
    shapes_ = std::make_shared<Core::FE::ShapeValues<distype>>(
        ele->degree(), usescompletepoly_, 2 * ele->degree());
  else if (shapes_->degree_ != unsigned(ele->degree()) ||
           shapes_->usescompletepoly_ != usescompletepoly_)
    shapes_ = std::make_shared<Core::FE::ShapeValues<distype>>(
        ele->degree(), usescompletepoly_, 2 * ele->degree());

  if (shapesface_ == nullptr)
  {
    Core::FE::ShapeValuesFaceParams svfparams(ele->degree(), usescompletepoly_, 2 * ele->degree());
    shapesface_ = std::make_shared<Core::FE::ShapeValuesFace<distype>>(svfparams);
  }

  if (local_solver_ == nullptr)
    local_solver_ = std::make_shared<LocalSolver>(ele, *shapes_, shapesface_, dyna_);
  else if (local_solver_->ndofs_ != shapes_->ndofs_)
    local_solver_ = std::make_shared<LocalSolver>(ele, *shapes_, shapesface_, dyna_);
}

/*----------------------------------------------------------------------*
 * read_global_vectors
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::read_global_vectors(Core::Elements::Element* ele,
    Core::FE::Discretization& discretization, const std::vector<int>& lm)
{
  TEUCHOS_FUNC_TIME_MONITOR("Discret::Elements::ElemagEleCalc::read_global_vectors");
  Discret::Elements::Elemag* elemagele = dynamic_cast<Discret::Elements::Elemag*>(ele);

  // read vectors from element storage
  interior_electricnp_.size(elemagele->eleinteriorElectric_.numRows());
  interior_magneticnp_.size(elemagele->eleinteriorMagnetic_.numRows());
  if (dyna_ == Inpar::EleMag::elemag_bdf2)
  {
    interior_electricnm_.size(elemagele->eleinteriorElectricnm1_.numRows());
    interior_magneticnm_.size(elemagele->eleinteriorMagneticnm1_.numRows());
    interior_electricnm_ = elemagele->eleinteriorElectricnm1_;
    interior_magneticnm_ = elemagele->eleinteriorMagneticnm1_;
  }

  interior_electricnm_.size(elemagele->eleinteriorElectricnm1_.numRows());
  interior_magneticnm_.size(elemagele->eleinteriorMagneticnm1_.numRows());
  interior_electricnm_ = elemagele->eleinteriorElectricnm1_;
  interior_magneticnm_ = elemagele->eleinteriorMagneticnm1_;

  interior_electricnp_ = elemagele->eleinteriorElectric_;
  interior_magneticnp_ = elemagele->eleinteriorMagnetic_;

  // read vectors from time integrator
  if (discretization.has_state("trace"))  // in case of "update interior variables"
  {
    elemagele->elenodeTrace2d_.size(lm.size());
    std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
        discretization.get_state("trace");
    Core::FE::extract_my_values(*matrix_state, elemagele->elenodeTrace2d_, lm);
  }

  return;
}  // read_global_vectors

/*----------------------------------------------------------------------*
 * fill_restart_vectors
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::fill_restart_vectors(
    Core::Elements::Element* ele, Core::FE::Discretization& discretization)
{
  // sort this back to the interior values vector
  int size = shapes_->ndofs_ * nsd_ * 2;

  std::vector<double> interiorVar(size);
  for (unsigned int i = 0; i < shapes_->ndofs_ * nsd_; ++i)
  {
    interiorVar[i] = interior_magneticnp_(i);
    interiorVar[shapes_->ndofs_ * nsd_ + i] = interior_electricnp_(i);
  }

  // tell this change in the interior variables the discretization
  std::vector<int> localDofs = discretization.dof(1, ele);
  const Epetra_Map* intdofcolmap = discretization.dof_col_map(1);
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
        discretization.get_state(1, "intVar");
    Core::LinAlg::Vector<double>& secondary =
        const_cast<Core::LinAlg::Vector<double>&>(*matrix_state);
    for (unsigned int i = 0; i < localDofs.size(); ++i)
    {
      const int lid = intdofcolmap->LID(localDofs[i]);
      secondary[lid] = interiorVar[i];
    }
  }

  std::vector<double> interiorVarnm(size);
  for (unsigned int i = 0; i < shapes_->ndofs_ * nsd_; ++i)
  {
    interiorVarnm[i] = interior_magneticnm_(i);
    interiorVarnm[shapes_->ndofs_ * nsd_ + i] = interior_electricnm_(i);
  }

  std::shared_ptr<const Core::LinAlg::Vector<double>> intVarnm =
      discretization.get_state(1, "intVarnm");
  Core::LinAlg::Vector<double>& secondary = const_cast<Core::LinAlg::Vector<double>&>(*intVarnm);
  for (unsigned int i = 0; i < localDofs.size(); ++i)
  {
    const int lid = intdofcolmap->LID(localDofs[i]);
    secondary[lid] = interiorVarnm[i];
  }

  return;
}

/*----------------------------------------------------------------------*
 * element_init_from_restart
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::element_init_from_restart(
    Core::Elements::Element* ele, Core::FE::Discretization& discretization)
{
  Discret::Elements::Elemag* elemagele = dynamic_cast<Discret::Elements::Elemag*>(ele);
  int size = shapes_->ndofs_ * nsd_ * 2;

  std::vector<double> interiorVar(size);

  std::shared_ptr<const Core::LinAlg::Vector<double>> intVar =
      discretization.get_state(1, "intVar");
  std::vector<int> localDofs1 = discretization.dof(1, ele);
  Core::FE::extract_my_values(*intVar, interiorVar, localDofs1);
  // now write this in corresponding eleinteriorElectric_ and eleinteriorMagnetic_
  for (unsigned int i = 0; i < shapes_->ndofs_ * nsd_; ++i)
  {
    elemagele->eleinteriorMagnetic_(i) = interiorVar[i];
    elemagele->eleinteriorElectric_(i) = interiorVar[shapes_->ndofs_ * nsd_ + i];
  }

  std::vector<double> interiorVarnm(size);

  std::shared_ptr<const Core::LinAlg::Vector<double>> intVarnm =
      discretization.get_state(1, "intVarnm");
  Core::FE::extract_my_values(*intVarnm, interiorVarnm, localDofs1);
  for (unsigned int i = 0; i < shapes_->ndofs_ * nsd_; ++i)
  {
    elemagele->eleinteriorMagneticnm1_(i) = interiorVarnm[i];
    elemagele->eleinteriorElectricnm1_(i) = interiorVarnm[shapes_->ndofs_ * nsd_ + i];
  }

  return;
}

/*----------------------------------------------------------------------*
 * Element init
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::element_init(
    Discret::Elements::Elemag* ele, Teuchos::ParameterList& params)
{
  // each element has to store the interior vectors by itseld, p-adaptivity or not
  // so, shape it, as you need it
  ele->eleinteriorElectricnm1_.size(shapes_->ndofs_ * nsd_);
  ele->eleinteriorMagneticnm1_.size(shapes_->ndofs_ * nsd_);
  ele->eleinteriorElectric_.size(shapes_->ndofs_ * nsd_);
  ele->eleinteriorMagnetic_.size(shapes_->ndofs_ * nsd_);

  dyna_ = params.get<Inpar::EleMag::DynamicType>("dyna");
  if (dyna_ == Inpar::EleMag::elemag_bdf4)
  {
    ele->eleinteriorElectricnm2_.size(shapes_->ndofs_ * nsd_);
    ele->eleinteriorMagneticnm2_.size(shapes_->ndofs_ * nsd_);
    ele->eleinteriorElectricnm3_.size(shapes_->ndofs_ * nsd_);
    ele->eleinteriorMagneticnm3_.size(shapes_->ndofs_ * nsd_);
  }

  // ele->elenodeTrace_.Size(ele->NumFace() * shapesface_->nfdofs_ * nsd_);
  ele->elenodeTrace2d_.size(ele->num_face() * shapesface_->nfdofs_ * (nsd_ - 1));

  return;
}

/*----------------------------------------------------------------------*
 * ProjectField
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ElemagEleCalc<distype>::LocalSolver::project_field(
    Discret::Elements::Elemag* ele, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2)
{
  shapes_.evaluate(*ele);

  // get function
  const int* start_func = params.getPtr<int>("startfuncno");
  const double time = params.get<double>("time");

  // the RHS matrix has to have the row dimension equal to the number of shape
  // functions(so we have one coefficient for each) and a number of column
  // equal to the overall number of component that we want to solve for.
  // The number is nsd_*2 because we have two fields..
  Core::LinAlg::SerialDenseMatrix localMat(shapes_.ndofs_, nsd_ * 2);
  for (unsigned int q = 0; q < shapes_.nqpoints_; ++q)
  {
    // Storing the values of the coordinates for the current quadrature point
    // and of the jacobian computed in that point
    const double fac = shapes_.jfac(q);
    Core::LinAlg::Matrix<nsd_, 1> xyz;
    for (unsigned int d = 0; d < nsd_; ++d)
      xyz(d) = shapes_.xyzreal(d, q);  // coordinates of quadrature point in real coordinates
    // Creating the temporary electric and magnetic field vector intVal
    // The vector is going to contain first the electric and then the magnetic
    // field such that the field will be initialized as first tree component
    // of the specified function as electric field, last three components as
    // magnetic field. If there is only one component all the components will
    // be initialized to the same value.
    Core::LinAlg::SerialDenseVector intVal(2 * nsd_);
    FOUR_C_ASSERT(start_func != nullptr, "funct not set for initial value");
    evaluate_all(*start_func, time, xyz, intVal);
    // now fill the components in the one-sided mass matrix and the right hand side
    for (unsigned int i = 0; i < shapes_.ndofs_; ++i)
    {
      // Mass matrix
      massPart(i, q) = shapes_.shfunct(i, q);
      massPartW(i, q) = shapes_.shfunct(i, q) * fac;

      // RHS for the electric and magnetic field
      for (int j = 0; j < intVal.numRows(); ++j)
        localMat(i, j) += shapes_.shfunct(i, q) * intVal(j) * fac;
    }
  }
  // The integration is made by computing the matrix product
  Core::LinAlg::multiply_nt(massMat, massPart, massPartW);
  {
    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
    inverseMass.setMatrix(Teuchos::rcpFromRef(massMat));
    inverseMass.setVectors(Teuchos::rcpFromRef(localMat), Teuchos::rcpFromRef(localMat));
    inverseMass.solve();
  }

  // Here we move the values from the temporary variable to the variable
  // contained in the element
  for (unsigned int r = 0; r < shapes_.ndofs_; ++r)
  {
    // Now we are storing the variables by component, meaning that we save for
    // each component the value for each dof and then we move to the next component.
    for (unsigned int d = 0; d < nsd_; ++d)
    {
      ele->eleinteriorElectric_(d * shapes_.ndofs_ + r) = localMat(r, d);         // Electric field
      ele->eleinteriorMagnetic_(d * shapes_.ndofs_ + r) = localMat(r, d + nsd_);  // magnetic
    }
  }

  if (dyna_ == Inpar::EleMag::elemag_bdf4)
    for (int s = 1; s < 4; s++)
    {
      localMat.putScalar(0.0);
      const double dt = params.get<double>("dt");
      for (unsigned int q = 0; q < shapes_.nqpoints_; ++q)
      {
        const double fac = shapes_.jfac(q);
        Core::LinAlg::Matrix<nsd_, 1> xyz;
        for (unsigned int d = 0; d < nsd_; ++d) xyz(d) = shapes_.xyzreal(d, q);

        Core::LinAlg::SerialDenseVector intVal(nsd_ * 2);

        evaluate_all(*start_func, time - s * dt, xyz, intVal);
        for (unsigned int i = 0; i < shapes_.ndofs_; ++i)
        {
          massPart(i, q) = shapes_.shfunct(i, q);
          massPartW(i, q) = shapes_.shfunct(i, q) * fac;
          for (int j = 0; j < intVal.numRows(); ++j)
            localMat(i, j) += shapes_.shfunct(i, q) * intVal(j) * fac;
        }
      }


      // The integration is made by computing the matrix product
      Core::LinAlg::multiply_nt(massMat, massPart, massPartW);
      {
        using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
        using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
        Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
        inverseMass.setMatrix(Teuchos::rcpFromRef(massMat));
        inverseMass.setVectors(Teuchos::rcpFromRef(localMat), Teuchos::rcpFromRef(localMat));
        inverseMass.solve();
      }

      for (unsigned int r = 0; r < shapes_.ndofs_; ++r)
        for (unsigned int d = 0; d < nsd_; ++d) switch (s)
          {
            case 1:
              ele->eleinteriorElectricnm1_(d * shapes_.ndofs_ + r) = localMat(r, d);
              ele->eleinteriorMagneticnm1_(d * shapes_.ndofs_ + r) = localMat(r, d + nsd_);
              break;
            case 2:
              ele->eleinteriorElectricnm2_(d * shapes_.ndofs_ + r) = localMat(r, d);
              ele->eleinteriorMagneticnm2_(d * shapes_.ndofs_ + r) = localMat(r, d + nsd_);
              break;
            case 3:
              ele->eleinteriorElectricnm3_(d * shapes_.ndofs_ + r) = localMat(r, d);
              ele->eleinteriorMagneticnm3_(d * shapes_.ndofs_ + r) = localMat(r, d + nsd_);
              break;
          }
    }

  return 0;
}

/*----------------------------------------------------------------------*
 * compute_error
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::LocalSolver::compute_error(
    Discret::Elements::Elemag* ele, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  double error_ele = 0.0, error_mag = 0.0;
  double exact_ele = 0.0, exact_mag = 0.0;
  shapes_.evaluate(*ele);

  // get function
  const int func = params.get<int>("funcno");
  const double time = params.get<double>("time");
  // for the calculation of the error, we use a higher integration rule
  std::shared_ptr<Core::FE::GaussPoints> highquad =
      Core::FE::create_gauss_points(distype, (ele->degree() + 2) * 2);
  Core::LinAlg::Matrix<nsd_, 1> xsi;
  Core::LinAlg::SerialDenseVector values(shapes_.ndofs_);
  Core::LinAlg::Matrix<nsd_, nen_> deriv;
  Core::LinAlg::Matrix<nsd_, nsd_> xjm;
  Core::LinAlg::SerialDenseVector electric(nsd_);
  Core::LinAlg::SerialDenseVector magnetic(nsd_);
  Core::LinAlg::SerialDenseVector analytical(2 * nsd_);

  for (int q = 0; q < highquad->num_points(); ++q)
  {
    const double* gpcoord = highquad->point(q);
    for (unsigned int idim = 0; idim < nsd_; idim++) xsi(idim) = gpcoord[idim];
    shapes_.polySpace_->evaluate(xsi, values);

    Core::FE::shape_function_deriv1<distype>(xsi, deriv);
    xjm.multiply_nt(deriv, shapes_.xyze);
    double highjfac = xjm.determinant() * highquad->weight(q);

    electric.putScalar(0.0);
    magnetic.putScalar(0.0);
    for (unsigned int i = 0; i < shapes_.ndofs_; ++i)
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        electric(d) += values(i) * ele->eleinteriorElectric_(d * shapes_.ndofs_ + i);
        magnetic(d) += values(i) * ele->eleinteriorMagnetic_(d * shapes_.ndofs_ + i);
      }

    Core::LinAlg::Matrix<nen_, 1> myfunct;
    Core::FE::shape_function<distype>(xsi, myfunct);
    Core::LinAlg::Matrix<nsd_, 1> xyzmat;
    xyzmat.multiply_nn(shapes_.xyze, myfunct);

    // Creating the temporary electric and magnetic field vector intVal
    // The vector is going to contain first the electric and then the magnetic
    // field such that the field will be initialized as first tree component
    // of the specified function as electric field, last three components as
    // magnetic field. If there is only one component all the components will
    // be initialized to the same value.
    analytical.putScalar(0.0);
    evaluate_all(func, time, xyzmat, analytical);

    for (unsigned int d = 0; d < nsd_; ++d)
    {
      error_ele += pow((analytical(d) - electric(d)), 2) * highjfac;
      exact_ele += pow(analytical(d), 2) * highjfac;
      error_mag += pow((analytical(d + nsd_) - magnetic(d)), 2) * highjfac;
      exact_mag += pow(analytical(d + nsd_), 2) * highjfac;
    }
  }

  elevec1[0] = error_ele;
  elevec1[1] = exact_ele;
  elevec1[2] = error_mag;
  elevec1[3] = exact_mag;

  return;
}

/*----------------------------------------------------------------------*
 * ProjectFieldTest
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ElemagEleCalc<distype>::LocalSolver::project_field_test(
    Discret::Elements::Elemag* ele, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2)
{
  shapes_.evaluate(*ele);

  // reshape elevec2 as matrix
  FOUR_C_ASSERT(elevec2.numRows() == 0 || unsigned(elevec2.numRows()) == nsd_ * shapes_.ndofs_,
      "Wrong size in project vector 2");

  // get function
  const int* start_func = params.getPtr<int>("startfuncno");
  const double time = params.get<double>("time");

  // internal variables
  if (elevec2.numRows() > 0)
  {
    // the RHS matrix has to have the row dimension equal to the number of shape
    // functions(so we have one coefficient for each) and a number of column
    // equal to the overall number of component that we want to solve for.
    // The number is nsd_*2 because we have two fields..
    Core::LinAlg::SerialDenseMatrix localMat(shapes_.ndofs_, nsd_ * 2);
    for (unsigned int q = 0; q < shapes_.nqpoints_; ++q)
    {
      // Storing the values of the coordinates for the current quadrature point
      // and of the jacobian computed in that point
      const double fac = shapes_.jfac(q);
      Core::LinAlg::Matrix<nsd_, 1> xyz;
      for (unsigned int d = 0; d < nsd_; ++d)
        xyz(d) = shapes_.xyzreal(d, q);  // coordinates of quadrature point in real coordinates
      // Creating the temporary electric and magnetic field vector intVal
      // The vector is going to contain first the electric and then the magnetic
      // field such that the field will be initialized as first tree component
      // of the specified function as electric field, last three components as
      // magnetic field. If there is only one component all the components will
      // be initialized to the same value.
      Core::LinAlg::SerialDenseVector intVal(2 * nsd_);
      FOUR_C_ASSERT(start_func != nullptr, "funct not set for initial value");
      evaluate_all(*start_func, time, xyz, intVal);
      // now fill the components in the one-sided mass matrix and the right hand side
      for (unsigned int i = 0; i < shapes_.ndofs_; ++i)
      {
        // Mass matrix
        massPart(i, q) = shapes_.shfunct(i, q);
        massPartW(i, q) = shapes_.shfunct(i, q) * fac;

        // RHS for the electric and magnetic field
        for (int j = 0; j < intVal.numRows(); ++j)
          localMat(i, j) += shapes_.shfunct(i, q) * intVal(j) * fac;
      }
    }
    // The integration is made by computing the matrix product
    Core::LinAlg::multiply_nt(massMat, massPart, massPartW);
    {
      using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
      using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
      Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
      inverseMass.setMatrix(Teuchos::rcpFromRef(massMat));
      inverseMass.setVectors(Teuchos::rcpFromRef(localMat), Teuchos::rcpFromRef(localMat));
      inverseMass.solve();
    }

    // Here we move the values from the temporary variable to the variable
    // contained in the element
    for (unsigned int r = 0; r < shapes_.ndofs_; ++r)
    {
      // Now we are storing the variables by component, meaning that we save for
      // each component the value for each dof and then we move to the next component.
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        ele->eleinteriorElectric_(d * shapes_.ndofs_ + r) = localMat(r, d);  // Electric field
        ele->eleinteriorMagnetic_(d * shapes_.ndofs_ + r) = localMat(r, d + nsd_);  // magnetic
      }
    }
  }
  return 0;
}

/*----------------------------------------------------------------------*
 * project_field_test_trace
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ElemagEleCalc<distype>::LocalSolver::project_field_test_trace(
    Discret::Elements::Elemag* ele, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  // Here we have the projection of the field on the trace
  // mass is the mass matrix for the system to be solved
  // the dimension of the mass matrix is given by the number of shape functions
  Core::LinAlg::SerialDenseMatrix mass(shapesface_->nfdofs_, shapesface_->nfdofs_);
  // TRaceVEC is the vector of the trace values
  // instead of being a vector it is a matrix so that we use the same matrix
  // to solve the projection problem on every component of the field
  Core::LinAlg::SerialDenseMatrix trVec(shapesface_->nfdofs_, nsd_);

  const int* start_func = params.getPtr<int>("startfuncno");
  const double time = params.get<double>("time");

  // Cycling through faces
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // Updating face data
    shapesface_->evaluate_face(*ele, f);

    // Initializing the matrices
    // It is necessary to create a matrix and a trVec for each face because the
    // dimensions of each face can differ from the previous one and the jacobian
    // contains the dimension of the face in it.
    mass.putScalar(0.0);
    trVec.putScalar(0.0);

    // Cycling through the quadrature points
    for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
    {
      // For each quadrature point we have a vector containing the field
      // components and a vector containing the spatial coordinates of that point
      Core::LinAlg::SerialDenseVector trace(nsd_);
      Core::LinAlg::Matrix<nsd_, 1> xyz;

      // Temporary variable to store the jacobian of the face (contains the weigth)
      const double fac = shapesface_->jfac(q);
      // Coordinates of quadrature point in real coordinates from the face to
      // the temporary variable. It is just to make the code easier to handle
      for (unsigned int d = 0; d < nsd_; ++d) xyz(d) = shapesface_->xyzreal(d, q);

      // Evaluation of the function in the quadrature point being considered
      evaluate_all(*start_func, time, xyz, trace);

      // Creating the mass matrix and the RHS vector
      for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      {
        // Mass matrix
        for (unsigned int j = 0; j < shapesface_->nfdofs_; ++j)
          mass(i, j) += shapesface_->shfunct(i, q) * shapesface_->shfunct(j, q) * fac;

        // RHS
        for (unsigned int d = 0; d < nsd_; ++d)
          trVec(i, d) += shapesface_->shfunct(i, q) * trace(d) * fac;
      }
    }

    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
    inverseMass.setMatrix(Teuchos::rcpFromRef(mass));
    inverseMass.setVectors(Teuchos::rcpFromRef(trVec), Teuchos::rcpFromRef(trVec));
    inverseMass.solve();

    Core::LinAlg::SerialDenseVector tempVec(shapesface_->nfdofs_ * (nsd_));
    Core::LinAlg::SerialDenseVector faceVec(shapesface_->nfdofs_ * (nsd_ - 1));
    // Filling the vector of trace values
    for (unsigned int d = 0; d < nsd_; ++d)
      for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      {
        // remember that "f" is an iterator index and therefore we are
        // cycling through all the faces and all the entries of elevec1
        // except for the first one where we will put the pressure average
        tempVec(d * shapesface_->nfdofs_ + i) = trVec(i, d);
      }

    Core::LinAlg::SerialDenseMatrix transformatrix(
        (nsd_ - 1) * shapesface_->nfdofs_, nsd_ * shapesface_->nfdofs_);
    for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      for (unsigned int d = 0; d < nsd_; ++d)
        for (unsigned int q = 0; q < nsd_ - 1; ++q)
          transformatrix(shapesface_->nfdofs_ * q + i, shapesface_->nfdofs_ * d + i) =
              shapesface_->tangent(d, q);

    Core::LinAlg::multiply_tn(faceVec, transformatrix, tempVec);

    // Filling the vector of trace values
    for (unsigned int d = 0; d < nsd_ - 1; ++d)
      for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      {
        // remember that "f" is an iterator index and therefore we are
        // cycling through all the faces and all the entries of elevec1
        // except for the first one where we will put the pressure average
        elevec1(f * shapesface_->nfdofs_ * (nsd_ - 1) + d * shapesface_->nfdofs_ + i) =
            faceVec(d * shapesface_->nfdofs_ + i);
      }
  }

  return 0;
}

/*----------------------------------------------------------------------*
 * ProjectDirichField
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ElemagEleCalc<distype>::LocalSolver::project_dirich_field(
    Discret::Elements::Elemag* ele, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  // Updating face data
  const int face = params.get<unsigned int>("faceconsider");
  // std::cout << face << std::endl;
  shapesface_->evaluate_face(*ele, face);

  Teuchos::Array<int>* functno = params.getPtr<Teuchos::Array<int>>("funct");
  const double time = params.get<double>("time");

  // Here we have the projection of the field on the trace
  // mass is the mass matrix for the system to be solved
  // the dimension of the mass matrix is given by the number of shape functions
  Core::LinAlg::SerialDenseMatrix mass(shapesface_->nfdofs_, shapesface_->nfdofs_);
  // TRaceVEC is the vector of the trace values
  // instead of being a vector it is a matrix so that we use the same matrix
  // to solve the projection problem on every component of the field
  Core::LinAlg::SerialDenseMatrix trVec(shapesface_->nfdofs_, nsd_);

  // Cycling through the quadrature points
  for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
  {
    // For each quadrature point we have a vector containing the field
    // components and a vector containing the spatial coordinates of that point
    Core::LinAlg::SerialDenseVector trace(nsd_);
    Core::LinAlg::Matrix<nsd_, 1> xyz;

    // Temporary variable to store the jacobian of the face (contains the weigth)
    const double fac = shapesface_->jfac(q);
    // Coordinates of quadrature point in real coordinates from the face to
    // the temporary variable. It is just to make the code easier to handle
    for (unsigned int d = 0; d < nsd_; ++d) xyz(d) = shapesface_->xyzreal(d, q);

    // Evaluation of the function in the quadrature point being considered
    evaluate_all((*functno)[0], time, xyz, trace);

    // Creating the mass matrix and the RHS vector
    for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
    {
      // Mass matrix
      for (unsigned int j = 0; j < shapesface_->nfdofs_; ++j)
        mass(i, j) += shapesface_->shfunct(i, q) * shapesface_->shfunct(j, q) * fac;

      // RHS
      for (unsigned int d = 0; d < nsd_; ++d)
        trVec(i, d) += shapesface_->shfunct(i, q) * trace(d) * fac;
    }
  }

  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
  inverseMass.setMatrix(Teuchos::rcpFromRef(mass));
  inverseMass.setVectors(Teuchos::rcpFromRef(trVec), Teuchos::rcpFromRef(trVec));
  inverseMass.solve();

  // Filling the vector of trace values
  Core::LinAlg::SerialDenseVector tempVec(shapesface_->nfdofs_ * (nsd_));
  for (unsigned int d = 0; d < nsd_; ++d)
    for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      tempVec(d * shapesface_->nfdofs_ + i) = trVec(i, d);

  Core::LinAlg::SerialDenseMatrix transformatrix(
      (nsd_ - 1) * shapesface_->nfdofs_, nsd_ * shapesface_->nfdofs_);
  for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
    for (unsigned int d = 0; d < nsd_; ++d)
      for (unsigned int q = 0; q < nsd_ - 1; ++q)
        transformatrix(shapesface_->nfdofs_ * q + i, shapesface_->nfdofs_ * d + i) =
            shapesface_->tangent(d, q);

  Core::LinAlg::multiply(elevec1, transformatrix, tempVec);

  return 0;
}

/*----------------------------------------------------------------------*
 * evaluate_all
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::LocalSolver::evaluate_all(const int start_func,
    const double t, const Core::LinAlg::Matrix<nsd_, 1>& xyz,
    Core::LinAlg::SerialDenseVector& v) const
{
  int numComp = Global::Problem::instance()
                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(start_func - 1)
                    .number_components();

  // If there is on component for each entry of the vector use une for each
  if (numComp == v.numRows())
  {
    for (int d = 0; d < v.numRows(); ++d)
      v[d] = Global::Problem::instance()
                 ->function_by_id<Core::Utils::FunctionOfSpaceTime>(start_func - 1)
                 .evaluate(xyz.data(), t, d);
  }
  // If the vector is half the number of the component only use the firt half
  else if (numComp == 2 * v.numRows())
  {
    for (int d = 0; d < v.numRows(); ++d)
      v[d] = Global::Problem::instance()
                 ->function_by_id<Core::Utils::FunctionOfSpaceTime>(start_func - 1)
                 .evaluate(xyz.data(), t, d);
  }
  // If the number of component is half of the vector, repeat the first half twice
  else if (numComp == v.numRows() / 2)
  {
    for (int d = 0; d < v.numRows(); ++d)
      v[d] = Global::Problem::instance()
                 ->function_by_id<Core::Utils::FunctionOfSpaceTime>(start_func - 1)
                 .evaluate(xyz.data(), t, d % numComp);
  }
  // If there is only one component always use it
  else if (numComp == 1)
  {
    for (int d = 0; d < v.numRows(); ++d)
      v[d] = Global::Problem::instance()
                 ->function_by_id<Core::Utils::FunctionOfSpaceTime>(start_func - 1)
                 .evaluate(xyz.data(), t, 0);
  }
  // If the number is not recognised throw an error
  else
    FOUR_C_THROW(
        "Supply ONE component for your start function or NUMDIM, not anything else! With NUMDIM "
        "components the field will be initialized componentwise, if only one component is "
        "provided, every component of the field will be initialized with the same values.");
  return;
}

/*----------------------------------------------------------------------*
 | interpolate_solution_to_nodes                          berardocco 04/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ElemagEleCalc<distype>::interpolate_solution_to_nodes(
    Discret::Elements::Elemag* ele, Core::FE::Discretization& discretization,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  initialize_shapes(ele);

  // Check if the vector has the correct size
  // The last part of the vector is not used so far as the postprocessing is not yet implemented for
  // this type of element
  FOUR_C_ASSERT(elevec1.numRows() == (int)nen_ * (4 * nsd_), "Vector does not have correct size");

  // Getting the connectivity matrix
  // Contains the (local) coordinates of the nodes belonging to the element
  Core::LinAlg::SerialDenseMatrix locations =
      Core::FE::get_ele_node_numbering_nodes_paramspace(distype);

  // This vector will contain the values of the shape functions computed in a
  // certain coordinate. In fact the lenght of the vector is given by the number
  // of shape functions, that is the same of the number of degrees of freedom of
  // an element.
  Core::LinAlg::SerialDenseVector values(shapes_->ndofs_);

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

    // compute values for interior unknown by summing over all basis functions
    for (unsigned int d = 0; d < nsd_; ++d)
    {
      double sum_electric = 0.0;
      double sum_magnetic = 0.0;
      // Cycling through all the shape functions
      for (unsigned int k = 0; k < shapes_->ndofs_; ++k)
      {
        // The overall value in the chosen point is given by the sum of the
        // values of the shape functions multiplied by their coefficients.
        sum_electric += values(k) * interior_electricnp_[d * shapes_->ndofs_ + k];
        sum_magnetic += values(k) * interior_magneticnp_[d * shapes_->ndofs_ + k];
      }
      // sum contains the linear combination of the shape functions times the
      // coefficients and its values are reordered in elevec1 grouped by
      // component: the first component for every node, then the following
      // component for the same nodes and so on for every component.
      elevec1(d * nen_ + i) = sum_electric;
      elevec1(nen_ * nsd_ + d * nen_ + i) = sum_magnetic;
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
  std::vector<int> ncn = Core::FE::get_number_of_face_element_corner_nodes(distype);
  // NumberInternalNodes
  std::vector<int> nin = Core::FE::get_number_of_face_element_internal_nodes(distype);

  // Cycling the faces of the element
  Core::LinAlg::SerialDenseVector fvalues(shapesface_->nfdofs_);
  for (unsigned int f = 0; f < nfaces_; ++f)
  {
    // Checking how many nodes the face has
    const int nfn = Core::FE::DisTypeToNumNodePerFace<distype>::numNodePerFace;

    shapesface_->evaluate_face(*ele, f);

    Core::LinAlg::SerialDenseVector facetrace((nsd_ - 1) * shapesface_->nfdofs_);
    Core::LinAlg::SerialDenseVector temptrace(nsd_ * shapesface_->nfdofs_);

    // As already said, the dimension of the coordinate matrix is now nsd_-1
    // times the number of nodes in the face.
    Core::LinAlg::Matrix<nsd_ - 1, nfn> xsishuffle(true);

    // Cycling throught the nodes of the face to store the node positions in the
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

    // Transformation for the face reference system
    Core::LinAlg::SerialDenseMatrix transformatrix(
        (nsd_ - 1) * shapesface_->nfdofs_, nsd_ * shapesface_->nfdofs_);
    for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      for (unsigned int d = 0; d < nsd_; ++d)
        for (unsigned int q = 0; q < nsd_ - 1; ++q)
          transformatrix(shapesface_->nfdofs_ * q + i, shapesface_->nfdofs_ * d + i) =
              shapesface_->tangent(d, q);

    // Storing the face part of the trace vector
    for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      for (unsigned int d = 0; d < nsd_ - 1; ++d)
        facetrace(shapesface_->nfdofs_ * d + i) =
            ele->elenodeTrace2d_[f * (nsd_ - 1) * shapesface_->nfdofs_ + shapesface_->nfdofs_ * d +
                                 i];

    Core::LinAlg::multiply_tn(temptrace, transformatrix, facetrace);

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

      // compute values for trace vector by summing over the shape functions
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        double sum = 0;
        // Linear combination of the values of the shape functions and
        // relative weighting coefficients. The weighting coefficients are
        // given by the value of the unknowns in the nodes.
        for (unsigned int k = 0; k < shapesface_->nfdofs_; ++k)
          sum += fvalues(k) * temptrace[d * shapesface_->nfdofs_ + k];
        // Ordering the results of the interpolation in the vector being careful
        // about the ordering of the nodes in the faces.
        if (i < ncn[f])
        {
          elevec1((nsd_ * 2 + d) * nen_ + shapesface_->faceNodeOrder[f][i]) += sum / nsd_;
        }
        else if (i < nfn - nin[f])
        {
          elevec1((nsd_ * 2 + d) * nen_ + shapesface_->faceNodeOrder[f][i]) += sum / (nsd_ - 1);
        }
        else
        {
          elevec1((nsd_ * 2 + d) * nen_ + shapesface_->faceNodeOrder[f][i]) += sum;
        }
      }
    }
  }
  return 0;
}

template <Core::FE::CellType distype>
Discret::Elements::ElemagEleCalc<distype>* Discret::Elements::ElemagEleCalc<distype>::instance(
    Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<Discret::Elements::ElemagEleCalc<distype>>(
            new Discret::Elements::ElemagEleCalc<distype>());
      });

  return singleton_owner.instance(action);
}

/*----------------------------------------------------------------------*
 * Constructor LocalSolver
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ElemagEleCalc<distype>::LocalSolver::LocalSolver(
    const Discret::Elements::Elemag* ele, Core::FE::ShapeValues<distype>& shapeValues,
    std::shared_ptr<Core::FE::ShapeValuesFace<distype>>& shapeValuesFace,
    Inpar::EleMag::DynamicType& dyna)
    : ndofs_(shapeValues.ndofs_), shapes_(shapeValues), shapesface_(shapeValuesFace), dyna_(dyna)
{
  // shape all matrices
  // Each one of these matrices is related to one equation of the formulation,
  // therefore ndofs equations in FEM terms) and one variable.
  // The number of entries is then given by ndofs time sthe dimension of the
  // space where the unknown lies. For vectorial field nsd_ gives the dimension.
  Amat.shape(nsd_ * ndofs_, nsd_ * ndofs_);
  invAmat.shape(nsd_ * ndofs_, nsd_ * ndofs_);
  Cmat.shape(nsd_ * ndofs_, nsd_ * ndofs_);
  Emat.shape(nsd_ * ndofs_, nsd_ * ndofs_);
  Fmat.shape(nsd_ * ndofs_, nsd_ * ndofs_);
  Gmat.shape(nsd_ * ndofs_, nsd_ * ndofs_);
  // These matrices have a "strange" shape because to merge them there will be
  // applied a matrix multiplication between the first one and the transposed
  // second one. The shape of the resulting matrix will therefore be ndofs x ndofs.
  massMat.shape(ndofs_, ndofs_);
  massPart.shape(ndofs_, shapeValues.nqpoints_);
  massPartW.shape(ndofs_, shapeValues.nqpoints_);

  // Matrix compriending the hybrid variable or the continuity condition
  // It is necessary to compute the overall number of degrees
  // of freedom by summing the number of degrees of freedom on every face
  // surrounding the volume of the element.
  // ONFaceDegreesOfFreedomS
  int onfdofs = 0;
  for (unsigned int i = 0; i < nfaces_; ++i)
  {
    // Evaluating the dofs number on each face of the element
    shapesface_->evaluate_face(*ele, i);
    // Computing the dimension of the approximation space for the hybrid variable
    onfdofs += shapesface_->nfdofs_;
  }
  // The hybrid variable is vectorial and therefore the dimension of the space
  // has to be multiplied by nsd_.
  onfdofs = onfdofs * (nsd_ - 1);

  // This part is specially dependent on the formulation being used, in fact,
  // when the matrices relative to the surface integrals have to be created
  // those will have different dimensions depending on the variable that appears
  // in the integral itself. The hybrid variable is defined in the trace space
  // and therefore its shape functions belong to the same space,
  // with the consequence of being onfdof shape functions.
  // Dmat and Hmat are the matrix that belongs to the equation for the magnetic
  // and electric field but multiply the hybrid variable, therefore their dimensions are:
  // o) nsd_*ndofs_ x onfdofs
  Dmat.shape(nsd_ * ndofs_, onfdofs);
  Hmat.shape(nsd_ * ndofs_, onfdofs);
  // Matrices Imat and Jmat describe the part of the continuity condition that
  // multiply the electric and magnetic fields and therefore their dimensions are:
  // o) ondofs x nsd_*ndofs_
  Imat.shape(onfdofs, nsd_ * ndofs_);
  Jmat.shape(onfdofs, nsd_ * ndofs_);
  // Finally Jmat is the matrix that belongs to the continuity condition and
  // multiplies the hybrid variable and therefore its dimensions are:
  // o) ondofs x ondofs
  Lmat.shape(onfdofs, onfdofs);
}

/*----------------------------------------------------------------------*
 * update_interior_variables_and_compute_residual
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::update_interior_variables_and_compute_residual(
    Teuchos::ParameterList& params, Discret::Elements::Elemag& ele, Core::Mat::Material& mat,
    Core::LinAlg::SerialDenseVector& elevec, double dt, bool errormaps, bool updateonly)
{
  TEUCHOS_FUNC_TIME_MONITOR(
      "Discret::Elements::ElemagEleCalc::update_interior_variables_and_compute_residual");

  // *****************************************************
  // update interior variables first
  // *****************************************************

  Core::LinAlg::SerialDenseVector tempVec1(shapes_->ndofs_ * nsd_);
  Core::LinAlg::SerialDenseVector tempVec2(shapes_->ndofs_ * nsd_);
  Core::LinAlg::SerialDenseVector tempVec3(shapes_->ndofs_ * nsd_);
  Core::LinAlg::SerialDenseVector xVec(shapes_->ndofs_ * nsd_);
  Core::LinAlg::SerialDenseVector yVec(shapes_->ndofs_ * nsd_);
  Core::LinAlg::SerialDenseMatrix tempMat(shapes_->ndofs_ * nsd_, shapes_->ndofs_ * nsd_);
  Core::LinAlg::SerialDenseMatrix tempMat2(shapes_->ndofs_ * nsd_, shapes_->ndofs_ * nsd_);

  local_solver_->compute_source(params, tempVec2, xVec);

  if (local_solver_->dyna_ == Inpar::EleMag::elemag_bdf2)
  {
    Core::LinAlg::multiply(0.0, tempVec1, -1.0 / 3.0, local_solver_->Amat,
        ele.eleinteriorMagneticnm1_);  //  4/3AH^{n-1} - 1/3AH^{n-2}
    Core::LinAlg::multiply(
        1.0, tempVec1, 4.0 / 3.0, local_solver_->Amat, ele.eleinteriorMagnetic_);  //  4/3AH^{n-1}
    Core::LinAlg::multiply(-1.0, tempVec2, -1.0 / 3.0, local_solver_->Emat,
        ele.eleinteriorElectricnm1_);  // -1/3EE^{n-2} - I_s
    Core::LinAlg::multiply(1.0, tempVec2, 4.0 / 3.0, local_solver_->Emat,
        ele.eleinteriorElectric_);  // 4/3EE^{n-1} - 1/3EE^{n-2} - I_s
  }
  else if (local_solver_->dyna_ == Inpar::EleMag::elemag_bdf4)
  {
    Core::LinAlg::multiply(0.0, tempVec1, -3.0 / 25.0, local_solver_->Amat,
        ele.eleinteriorMagneticnm3_);  // (1/3)E E^{n} + I_s
    Core::LinAlg::multiply(1.0, tempVec1, 16.0 / 25.0, local_solver_->Amat,
        ele.eleinteriorMagneticnm2_);  // ^E = (4/3)EE^{n+1} - (1/3)EE^{n} - I_s
    Core::LinAlg::multiply(
        1.0, tempVec1, -36.0 / 25.0, local_solver_->Amat, ele.eleinteriorMagneticnm1_);
    Core::LinAlg::multiply(
        1.0, tempVec1, 48.0 / 25.0, local_solver_->Amat, ele.eleinteriorMagnetic_);
    Core::LinAlg::multiply(-1.0, tempVec2, -3.0 / 25.0, local_solver_->Emat,
        ele.eleinteriorElectricnm3_);  // (1/3)E E^{n} + I_s
    Core::LinAlg::multiply(1.0, tempVec2, 16.0 / 25.0, local_solver_->Emat,
        ele.eleinteriorElectricnm2_);  // ^E = (4/3)EE^{n+1} - (1/3)EE^{n} - I_s
    Core::LinAlg::multiply(
        1.0, tempVec2, -36.0 / 25.0, local_solver_->Emat, ele.eleinteriorElectricnm1_);
    Core::LinAlg::multiply(
        1.0, tempVec2, 48.0 / 25.0, local_solver_->Emat, ele.eleinteriorElectric_);
  }
  else
  {
    Core::LinAlg::multiply(tempVec1, local_solver_->Amat, ele.eleinteriorMagnetic_);  //  AH^{n-1}
    Core::LinAlg::multiply(
        -1.0, tempVec2, 1.0, local_solver_->Emat, ele.eleinteriorElectric_);  // EE^{n-1} - I_s
  }
  ele.eleinteriorMagneticnm3_ = ele.eleinteriorMagneticnm2_;
  ele.eleinteriorMagneticnm2_ = ele.eleinteriorMagneticnm1_;
  ele.eleinteriorMagneticnm1_ = ele.eleinteriorMagnetic_;
  ele.eleinteriorElectricnm3_ = ele.eleinteriorElectricnm2_;
  ele.eleinteriorElectricnm2_ = ele.eleinteriorElectricnm1_;
  ele.eleinteriorElectricnm1_ = ele.eleinteriorElectric_;

  // Add the trace component
  Core::LinAlg::multiply(1.0, tempVec1, -1.0, local_solver_->Dmat, ele.elenodeTrace2d_);
  Core::LinAlg::multiply(1.0, tempVec2, -1.0, local_solver_->Hmat, ele.elenodeTrace2d_);

  Core::LinAlg::multiply(tempMat, local_solver_->Fmat, local_solver_->invAmat);  // FA^{-1}

  tempMat2 += local_solver_->Emat;
  tempMat2 += local_solver_->Gmat;
  Core::LinAlg::multiply(1.0, tempMat2, -1.0, tempMat, local_solver_->Cmat);  //(E + G) - FA^{-1}C
  {
    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> invert;
    invert.setMatrix(Teuchos::rcpFromRef(tempMat2));
    invert.invert();  //  [(E + G) - FA^{-1}C]^{-1}
  }

  Core::LinAlg::multiply(1.0, tempVec2, -1.0, tempMat, tempVec1);  //  e - FA^{-1}h

  //  E^{n} = [(E + G) - FA^{-1}C]^{-1} (e - FA^{-1}h)
  Core::LinAlg::multiply(interior_electricnp_, tempMat2, tempVec2);


  Core::LinAlg::multiply(
      1.0, tempVec1, -1.0, local_solver_->Cmat, interior_electricnp_);  //  h - CE^{n}

  //  = A^{-1}(h - CE^{n})
  Core::LinAlg::multiply(interior_magneticnp_, local_solver_->invAmat, tempVec1);

  ele.eleinteriorMagnetic_ = interior_magneticnp_;
  ele.eleinteriorElectric_ = interior_electricnp_;

  // Updateresidual

  if (local_solver_->dyna_ == Inpar::EleMag::elemag_bdf2)
  {
    Core::LinAlg::multiply(-1.0, xVec, 4.0 / 3.0, local_solver_->Emat,
        ele.eleinteriorElectric_);  //  = 4/3EE^{n} - I_s
    Core::LinAlg::multiply(1.0, xVec, -1.0 / 3.0, local_solver_->Emat,
        ele.eleinteriorElectricnm1_);  //  = 4/3EE^{n} -1/3EE^{n-1} - I_s
    Core::LinAlg::multiply(1.0, xVec, -4.0 / 3.0, local_solver_->Fmat,
        ele.eleinteriorMagnetic_);  //  = 4/3EE^{n} -1/3EE^{n-1} - I_s - 4/3FH^{n}
    Core::LinAlg::multiply(1.0, xVec, 1.0 / 3.0, local_solver_->Fmat,
        ele.eleinteriorMagneticnm1_);  //  = 4/3EE^{n} -1/3EE^{n-1} - I_s - 4/3FH^{n} + 1/3FH^{n-1}
    Core::LinAlg::multiply(
        0.0, tempVec1, 4.0 / 3.0, local_solver_->Amat, ele.eleinteriorMagnetic_);  //  = 4/3AH^{n}
    Core::LinAlg::multiply(1.0, tempVec1, -1.0 / 3.0, local_solver_->Amat,
        ele.eleinteriorMagneticnm1_);  //  = 4/3AH^{n} - 1/3AH^{n-1}
  }
  else if (local_solver_->dyna_ == Inpar::EleMag::elemag_bdf4)
  {
    Core::LinAlg::multiply(-1.0, xVec, -3.0 / 25.0, local_solver_->Emat,
        ele.eleinteriorElectricnm3_);  // (1/3)E E^{n} + I_s
    Core::LinAlg::multiply(1.0, xVec, 16.0 / 25.0, local_solver_->Emat,
        ele.eleinteriorElectricnm2_);  // ^E = (4/3)EE^{n+1} - (1/3)EE^{n} - I_s
    Core::LinAlg::multiply(
        1.0, xVec, -36.0 / 25.0, local_solver_->Emat, ele.eleinteriorElectricnm1_);
    Core::LinAlg::multiply(1.0, xVec, 48.0 / 25.0, local_solver_->Emat, ele.eleinteriorElectric_);
    Core::LinAlg::multiply(1.0, xVec, 3.0 / 25.0, local_solver_->Fmat,
        ele.eleinteriorMagneticnm3_);  // (1/3)E E^{n} + I_s
    Core::LinAlg::multiply(1.0, xVec, -16.0 / 25.0, local_solver_->Fmat,
        ele.eleinteriorMagneticnm2_);  // ^E = (4/3)EE^{n+1} - (1/3)EE^{n} - I_s
    Core::LinAlg::multiply(
        1.0, xVec, 36.0 / 25.0, local_solver_->Fmat, ele.eleinteriorMagneticnm1_);
    Core::LinAlg::multiply(1.0, xVec, -48.0 / 25.0, local_solver_->Fmat, ele.eleinteriorMagnetic_);
    Core::LinAlg::multiply(0.0, tempVec1, -3.0 / 25.0, local_solver_->Amat,
        ele.eleinteriorMagneticnm3_);  // (1/3)E E^{n} + I_s
    Core::LinAlg::multiply(1.0, tempVec1, 16.0 / 25.0, local_solver_->Amat,
        ele.eleinteriorMagneticnm2_);  // ^E = (4/3)EE^{n+1} - (1/3)EE^{n} - I_s
    Core::LinAlg::multiply(
        1.0, tempVec1, -36.0 / 25.0, local_solver_->Amat, ele.eleinteriorMagneticnm1_);
    Core::LinAlg::multiply(
        1.0, tempVec1, 48.0 / 25.0, local_solver_->Amat, ele.eleinteriorMagnetic_);
    // FOUR_C_THROW("Not implemented");
  }
  else
  {
    Core::LinAlg::multiply(
        -1.0, xVec, -1.0, local_solver_->Fmat, ele.eleinteriorMagnetic_);  //  = -FH^{n} - I_s
    Core::LinAlg::multiply(
        1.0, xVec, 1.0, local_solver_->Emat, ele.eleinteriorElectric_);  //  = EE^{n} -I_s - FH^{n}
    Core::LinAlg::multiply(tempVec1, local_solver_->Amat, ele.eleinteriorMagnetic_);  //  = AH^{n}
  }

  Core::LinAlg::multiply(
      yVec, tempMat2, xVec);  //  = [(E + G) - FA^{-1}C]^{-1}(EE^{n} - I_s^{n} - FH^{n})
  Core::LinAlg::multiply(elevec, local_solver_->Jmat, yVec);
  Core::LinAlg::multiply(1.0, tempVec1, -1.0, local_solver_->Cmat, yVec);  //  = AH^{n} - Cy
  Core::LinAlg::multiply(yVec, local_solver_->invAmat, tempVec1);          //  = A^{-1}(AH^{n} - Cy)

  Core::LinAlg::multiply(-1.0, elevec, -1.0, local_solver_->Imat, yVec);  //  = -Ix -Jy

  return;
}  // update_interior_variables_and_compute_residual

/*----------------------------------------------------------------------*
 * ComputeAbsorbingBC
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::LocalSolver::compute_absorbing_bc(
    Core::FE::Discretization& discretization, Discret::Elements::Elemag* ele,
    Teuchos::ParameterList& params, std::shared_ptr<Core::Mat::Material>& mat, int face,
    Core::LinAlg::SerialDenseMatrix& elemat, int indexstart,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  TEUCHOS_FUNC_TIME_MONITOR("Discret::Elements::ElemagEleCalc::ComputeAbsorbingBC");

  shapesface_->evaluate_face(*ele, face);

  unsigned int newindex = shapesface_->nfdofs_ * (nsd_ - 1) * face;

  auto actmat = static_cast<const Mat::ElectromagneticMat*>(mat.get());
  double impedance = std::sqrt(actmat->epsilon(ele->id()) / actmat->mu(ele->id()));

  bool do_rhs = params.get<bool>("do_rhs");
  if (do_rhs)
  {
    // Get the user defined functions
    auto* cond = params.getPtr<std::shared_ptr<Core::Conditions::Condition>>("condition");
    const auto& funct = (*cond)->parameters().get<std::vector<int>>("FUNCT");
    const double time = params.get<double>("time");

    Core::LinAlg::SerialDenseVector tempVec1(shapesface_->nfdofs_ * nsd_);
    Core::LinAlg::SerialDenseVector tempVec2(shapesface_->nfdofs_ * (nsd_ - 1));
    // the RHS matrix has to have the row dimension equal to the number of shape
    // functions(so we have one coefficient for each) and a number of column
    // equal to the overall number of component that we want to solve for.
    // The number is nsd_*2 because we have two fields..
    Core::LinAlg::SerialDenseMatrix localMat(shapesface_->nfdofs_, nsd_ * 2);
    {
      Core::LinAlg::SerialDenseMatrix tempMassMat(shapesface_->nfdofs_, shapesface_->nfdofs_);
      Core::LinAlg::SerialDenseMatrix tempMat(shapesface_->nfdofs_, shapesface_->nqpoints_);
      Core::LinAlg::SerialDenseMatrix tempMatW(shapesface_->nfdofs_, shapesface_->nqpoints_);
      for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
      {
        // Storing the values of the coordinates for the current quadrature point
        // and of the jacobian computed in that point
        const double fac = shapesface_->jfac(q);
        Core::LinAlg::Matrix<nsd_, 1> xyz;
        for (unsigned int d = 0; d < nsd_; ++d)
          xyz(d) =
              shapesface_->xyzreal(d, q);  // coordinates of quadrature point in real coordinates
        // Creating the temporary electric and magnetic field vector intVal
        // The vector is going to contain first the electric and then the magnetic
        // field such that the field will be initialized as first tree component
        // of the specified function as electric field, last three components as
        // magnetic field. If there is only one component all the components will
        // be initialized to the same value.
        Core::LinAlg::SerialDenseVector intVal(2 * nsd_);
        evaluate_all(funct[0], time, xyz, intVal);
        // now fill the components in the one-sided mass matrix and the right hand side
        for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
        {
          // Mass matrix
          tempMat(i, q) = shapesface_->shfunct(i, q);
          tempMatW(i, q) = shapesface_->shfunct(i, q) * fac;

          // RHS for the electric and magnetic field
          for (int j = 0; j < intVal.numRows(); ++j)
            localMat(i, j) += shapesface_->shfunct(i, q) * intVal(j) * fac;
        }
      }
      // The integration is made by computing the matrix product
      Core::LinAlg::multiply_nt(tempMassMat, tempMat, tempMatW);
      {
        using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
        using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
        Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
        inverseMass.setMatrix(Teuchos::rcpFromRef(tempMassMat));
        inverseMass.setVectors(Teuchos::rcpFromRef(localMat), Teuchos::rcpFromRef(localMat));
        inverseMass.solve();
      }
    }

    for (unsigned int r = 0; r < shapesface_->nfdofs_; ++r)
      for (unsigned int d = 0; d < nsd_; ++d)
        tempVec1(d * shapesface_->nfdofs_ + r) = localMat(r, d);  // Electric field

    // Creating the matrix
    Core::LinAlg::SerialDenseMatrix transformatrix(
        (nsd_ - 1) * shapesface_->nfdofs_, nsd_ * shapesface_->nfdofs_);
    for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      for (unsigned int d = 0; d < nsd_; ++d)
        for (unsigned int q = 0; q < nsd_ - 1; ++q)
          transformatrix(shapesface_->nfdofs_ * q + i, shapesface_->nfdofs_ * d + i) =
              shapesface_->tangent(d, q);

    // MIXED SHAPE FUNCTIONS
    // The matrix that are going to be build here are D,I and J
    // loop over number of internal shape functions
    // Here we need to create only the first part of tghe D and H matrix to be multiplied by the
    // transformation matrices and then put in the real D and H matrices
    Core::LinAlg::SerialDenseMatrix tempI(shapesface_->nfdofs_ * nsd_, shapesface_->nfdofs_ * nsd_);
    Core::LinAlg::SerialDenseMatrix tempJ(shapesface_->nfdofs_ * nsd_, shapesface_->nfdofs_ * nsd_);
    for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
    {
      // If the shape function is zero on the face we can just skip it. Remember
      // that the matrix have already been set to zero and therefore if nothing
      // is done the value ramains zero
      // loop over number of face shape functions
      for (unsigned int j = 0; j < shapesface_->nfdofs_; ++j)
      {
        // Now that the integration has been carried on it is necessary to place
        // the value in the right position inside the matrices
        for (unsigned int d = 0; d < nsd_; ++d)
        {
          // i internal shape functions
          // j boundary shape functions
          for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
          {
            // Storing the value of the integral without the normal components
            const double temp =
                shapesface_->jfac(q) * shapesface_->shfunct(i, q) * shapesface_->shfunct(j, q);

            // Filling the matrices
            // Imat = -[H x n]
            //+1 coordinate
            tempI(shapesface_->nfdofs_ * d + j, ((d + 1) % nsd_) * shapesface_->nfdofs_ + i) -=
                temp * shapesface_->normals((d + 2) % nsd_, q);
            //+2 coordinate
            tempI(shapesface_->nfdofs_ * d + j, ((d + 2) % nsd_) * shapesface_->nfdofs_ + i) +=
                temp * shapesface_->normals((d + 1) % nsd_, q);
            // Jmat
            tempJ(shapesface_->nfdofs_ * d + j, d * shapesface_->nfdofs_ + i) += temp;
          }
        }  // for (unsigned int d = 0; d < nsd_; ++d)
      }    // for (unsigned int j=0; j<shapesface_->nfdofs_; ++j)
    }      // for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)

    // Fill face values into the matrices
    Core::LinAlg::SerialDenseMatrix magneticMat(
        shapesface_->nfdofs_ * (nsd_ - 1), shapesface_->nfdofs_ * nsd_);
    Core::LinAlg::SerialDenseMatrix electricMat(
        shapesface_->nfdofs_ * (nsd_ - 1), shapesface_->nfdofs_ * nsd_);
    Core::LinAlg::multiply(magneticMat, transformatrix, tempI);
    Core::LinAlg::multiply(electricMat, transformatrix, tempJ);

    Core::LinAlg::multiply(0.0, tempVec2, impedance, electricMat, tempVec1);

    for (unsigned int r = 0; r < shapesface_->nfdofs_; ++r)
      for (unsigned int d = 0; d < nsd_; ++d)
        tempVec1(d * shapesface_->nfdofs_ + r) = localMat(r, d + nsd_);  // magnetic

    Core::LinAlg::multiply(1.0, tempVec2, 1.0, magneticMat, tempVec1);

    for (int i = 0; i < tempVec2.numRows(); ++i) elevec1(newindex + i) = tempVec2(i);
  }
  else  // if not(do_rhs) then do the matrix
  {
    for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      for (unsigned int j = 0; j < shapesface_->nfdofs_; ++j)
        for (unsigned int d = 0; d < nsd_ - 1; ++d)
          for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
          {
            const double temp = impedance * shapesface_->jfac(q) * shapesface_->shfunct(i, q) *
                                shapesface_->shfunct(j, q);
            elemat(newindex + shapesface_->nfdofs_ * d + i,
                newindex + shapesface_->nfdofs_ * d + j) += temp;
          }
  }

  return;
}

/*----------------------------------------------------------------------*
 * ComputeSource
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::LocalSolver::compute_source(
    Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector& interiorSourcen,
    Core::LinAlg::SerialDenseVector& interiorSourcenp)
{
  int funcno = params.get<int>("sourcefuncno");
  if (funcno <= 0) return;  // there is no such thing as a volume force

  // the vector to be filled
  Core::LinAlg::SerialDenseVector sourcen(nsd_);
  Core::LinAlg::SerialDenseVector sourcenp(nsd_);

  // what time is it?
  double tn = params.get<double>("time");
  double tp = params.get<double>("timep");

  // double f_value = 0.0;
  for (unsigned int q = 0; q < shapes_.nqpoints_; ++q)
  {
    Core::LinAlg::Matrix<nsd_, 1> xyz;
    for (unsigned int d = 0; d < nsd_; ++d) xyz(d) = shapes_.xyzreal(d, q);

    // calculate right hand side contribution for dp/dt
    evaluate_all(funcno, tn, xyz, sourcen);
    evaluate_all(funcno, tp, xyz, sourcenp);

    // add it all up
    for (unsigned int i = 0; i < shapes_.ndofs_; ++i)
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        interiorSourcen(i + d * shapes_.ndofs_) +=
            shapes_.shfunct(i, q) * sourcen(d) * shapes_.jfac(q);
        interiorSourcenp(i + d * shapes_.ndofs_) +=
            shapes_.shfunct(i, q) * sourcenp(d) * shapes_.jfac(q);
      }
  }

  return;
}

/*----------------------------------------------------------------------*
 * compute_interior_matrices
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::LocalSolver::compute_interior_matrices(
    double dt, double sigma, double mu, double epsilon)
{
  // The definitions of the matrices created here can be found in the internal
  // paper from Gravemeier "A hybridizable discontinous Galerkin method for
  // electromagnetics in subsurface applications".
  // The explicit form of these matrices is reported for convenience?
  TEUCHOS_FUNC_TIME_MONITOR("Discret::Elements::ElemagEleCalc::compute_interior_matrices");
  // Why is this made in this order? Is it faster in this order? Or is it better
  // to have it shape_functions->quadrature_points?
  // loop quadrature points
  for (unsigned int q = 0; q < shapes_.nqpoints_; ++q)
  {
    // loop shape functions
    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      massPart(i, q) = shapes_.shfunct(i, q);
      const double valf = shapes_.shfunct(i, q) * shapes_.jfac(q);
      massPartW(i, q) = valf;
    }
  }

  Core::LinAlg::SerialDenseMatrix tmpMat(ndofs_, ndofs_);
  // this temorary matrix is used to compute the numerical integration and the
  // values are then copied in the right places. Probably it is also possible
  // to have the matrix multiplication to obtain directly the correct matrices
  // but it would mean to compute three time sthe same value for each shape
  // function instead of computing it only omnce and then directly copying it.
  Core::LinAlg::multiply_nt(tmpMat, massPart, massPartW);
  // A, E and part of G
  for (unsigned int j = 0; j < ndofs_; ++j)
    for (unsigned int i = 0; i < ndofs_; ++i)
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        Amat(d * ndofs_ + i, d * ndofs_ + j) = mu * tmpMat(i, j);
        Emat(d * ndofs_ + i, d * ndofs_ + j) = epsilon * tmpMat(i, j);
        // Carefull because G is not yet complete, it is necessary to add the boundary
        Gmat(d * ndofs_ + i, d * ndofs_ + j) = sigma * tmpMat(i, j);
      }

  if (dyna_ == Inpar::EleMag::elemag_bdf2)
  {
    Amat.scale(3.0 / (2.0 * dt));
    Emat.scale(3.0 / (2.0 * dt));
  }
  else if (dyna_ == Inpar::EleMag::elemag_bdf4)
  {
    Amat.scale(25.0 / (12.0 * dt));
    Emat.scale(25.0 / (12.0 * dt));
  }
  else
  {
    Amat.scale(1 / dt);
    Emat.scale(1 / dt);
  }

  {  // We are creating this scope to destroy everything related to the matrix inversion
    // We are going to need both A and its inverse and therefore we are storing both
    invAmat += Amat;
    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> invA;
    invA.setMatrix(Teuchos::rcpFromRef(invAmat));
    int err = invA.invert();
    if (err != 0) FOUR_C_THROW("Inversion for Amat failed with errorcode %d", err);
  }

  for (unsigned int i = 0; i < shapes_.ndofs_; ++i)
    for (unsigned int j = 0; j < shapes_.ndofs_; ++j)
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        for (unsigned int q = 0; q < shapes_.nqpoints_; ++q)
        {
          Cmat(i + d * ndofs_, j + ((d + 1) % nsd_) * ndofs_) +=
              shapes_.shderxy(i * nsd_ + ((d + 2) % nsd_), q) * shapes_.shfunct(j, q) *
              shapes_.jfac(q);
          Cmat(i + d * ndofs_, j + ((d + 2) % nsd_) * ndofs_) -=
              shapes_.shderxy(i * nsd_ + ((d + 1) % nsd_), q) * shapes_.shfunct(j, q) *
              shapes_.jfac(q);
        }
      }

  // The summation over quadrature points is manually made
  for (unsigned int i = 0; i < ndofs_; ++i)
    for (unsigned int j = 0; j < ndofs_; ++j)
      for (unsigned int d = 0; d < nsd_; ++d)
        for (unsigned int q = 0; q < shapes_.nqpoints_; ++q)
        {
          const double temp = shapes_.shfunct(i, q) * shapes_.jfac(q);
          // this can be avoided but the optimization of the code comes later
          Fmat(d * ndofs_ + i, ((d + 1) % nsd_) * ndofs_ + j) +=
              temp * shapes_.shderxy(j * nsd_ + ((d + 2) % nsd_), q);
          Fmat(d * ndofs_ + i, ((d + 2) % nsd_) * ndofs_ + j) -=
              temp * shapes_.shderxy(j * nsd_ + ((d + 1) % nsd_), q);
        }

  return;
}

/*----------------------------------------------------------------------*
 * ComputeResidual
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::LocalSolver::compute_residual(
    Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector& elevec,
    Discret::Elements::Elemag& ele)
{
  TEUCHOS_FUNC_TIME_MONITOR("Discret::Elements::ElemagEleCalc::ComputeResidual");

  // for implicit Euler
  //                                -1
  //                     +---------+    +-----------+
  //                     | A    C  |    | A H^{n-1} |
  // R^{n}  = - [ I  J ] |         |    |           |  =  -Ix - Jy
  //                     | F   E+G |    | E E^{n-1} |
  //                     +---------+    +-----------+
  //
  //  x = A^{-1} (AH^{n-1} - Cy)
  //
  //  y = ((E + G) - F A^{-1} C)^{-1} ((E E^{n-1} - I_s^{n-1}) - F A^{-1} A H^{n-1})

  const unsigned int intdofs = ndofs_ * nsd_;
  // All the vectors are initilized to zero
  Core::LinAlg::SerialDenseVector tempVec1(intdofs);
  Core::LinAlg::SerialDenseVector tempVec2(intdofs);
  Core::LinAlg::SerialDenseVector tempVec3(intdofs);
  // Once the compute source is ready we will need to delete these
  // The ComputeSource is necesessary to include the forcing terms
  compute_source(params, tempVec2, tempVec3);

  if (dyna_ == Inpar::EleMag::elemag_bdf2)
  {
    Core::LinAlg::multiply(
        0.0, tempVec1, 4.0 / 3.0, Amat, ele.eleinteriorMagnetic_);  // 4/3AH^{n-1}
    Core::LinAlg::multiply(
        1.0, tempVec1, -1.0 / 3.0, Amat, ele.eleinteriorMagneticnm1_);  // 4/3AH^{n-1} - 1/3AH^{n-2}
    Core::LinAlg::multiply(
        -1.0, tempVec2, 4.0 / 3.0, Emat, ele.eleinteriorElectric_);  // 4/3E E^{n-1} - I_s^{n-1}
    Core::LinAlg::multiply(1.0, tempVec2, -1.0 / 3.0, Emat,
        ele.eleinteriorElectricnm1_);  // 4/3E E^{n-1} - 1/3EE^{n-2} - I_s^{n-1}
  }
  else if (dyna_ == Inpar::EleMag::elemag_bdf4)
  {
    Core::LinAlg::multiply(
        0.0, tempVec1, -3.0 / 25.0, Amat, ele.eleinteriorMagneticnm3_);  // (1/3)E E^{n} + I_s
    Core::LinAlg::multiply(1.0, tempVec1, 16.0 / 25.0, Amat,
        ele.eleinteriorMagneticnm2_);  // ^E = (4/3)EE^{n+1} - (1/3)EE^{n} - I_s
    Core::LinAlg::multiply(1.0, tempVec1, -36.0 / 25.0, Amat, ele.eleinteriorMagneticnm1_);
    Core::LinAlg::multiply(1.0, tempVec1, 48.0 / 25.0, Amat, ele.eleinteriorMagnetic_);
    Core::LinAlg::multiply(
        -1.0, tempVec2, -3.0 / 25.0, Emat, ele.eleinteriorElectricnm3_);  // (1/3)E E^{n} + I_s
    Core::LinAlg::multiply(1.0, tempVec2, 16.0 / 25.0, Emat,
        ele.eleinteriorElectricnm2_);  // ^E = (4/3)EE^{n+1} - (1/3)EE^{n} - I_s
    Core::LinAlg::multiply(1.0, tempVec2, -36.0 / 25.0, Emat, ele.eleinteriorElectricnm1_);
    Core::LinAlg::multiply(1.0, tempVec2, 48.0 / 25.0, Emat, ele.eleinteriorElectric_);
  }
  else
  {
    Core::LinAlg::multiply(tempVec1, Amat, ele.eleinteriorMagnetic_);  // AH^{n-1}
    Core::LinAlg::multiply(
        -1.0, tempVec2, 1.0, Emat, ele.eleinteriorElectric_);  // E E^{n-1} - I_s^{n-1}
  }

  Core::LinAlg::SerialDenseMatrix tempMat1(intdofs, intdofs);
  Core::LinAlg::multiply(tempMat1, Fmat, invAmat);  // F A^{-1}
  Core::LinAlg::multiply(
      1.0, tempVec2, -1.0, tempMat1, tempVec1);  // ((E E^{n-1} - I_s^{n-1}) - F A^{-1} A H^{n-1})

  Core::LinAlg::SerialDenseMatrix tempMat2(intdofs, intdofs);
  // Gmat already contains Emat in it
  tempMat2 += Emat;
  tempMat2 += Gmat;
  Core::LinAlg::multiply(1.0, tempMat2, -1.0, tempMat1, Cmat);  // = (E + G) - F A^{-1} C
  {
    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseinW;
    inverseinW.setMatrix(Teuchos::rcpFromRef(tempMat2));
    int err = inverseinW.invert();
    if (err != 0)
      FOUR_C_THROW(
          "Inversion of temporary matrix for Schur complement failed with errorcode %d", err);
  }
  // tempMat2 = ((E + G) - F A^{-1} C)^{-1}

  Core::LinAlg::multiply(tempVec3, tempMat2, tempVec2);       // y
  Core::LinAlg::multiply(0.0, elevec, -1.0, Jmat, tempVec3);  //  -Jy

  Core::LinAlg::multiply(1.0, tempVec1, -1.0, Cmat, tempVec3);  // AH^{n-1} - Cy
  Core::LinAlg::multiply(tempVec3, invAmat, tempVec1);          //  x = A^{-1} (AH^{n-1} - Cy)
  Core::LinAlg::multiply(1.0, elevec, -1.0, Imat, tempVec3);    //  -Ix - Jy

  return;
}  // ComputeResidual

/*----------------------------------------------------------------------*
 * ComputeFaceMatrices
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::LocalSolver::compute_face_matrices(const int face,
    double dt, int indexstart, int newindex, double sigma, double mu, const double tau)
{
  TEUCHOS_FUNC_TIME_MONITOR("Discret::Elements::ElemagEleCalc::ComputeFaceMatrices");

  // Tau is defined as (\frac{|sigma|}{\mu t_c})^0.5 where t_c is a
  // characteristic time scale
  // const double stabilization = tau ? tau : mu / dt;
  // const double stabilization = tau ? tau : 2.0;
  // This routine seems complex but it's not (well, it is just as the others)
  // It is divided in three parts:
  //  o   Mixed shape functions integration
  //  o   Interior shaoe function integration
  //  o   Boundary shape function integration
  // The difference lays on the number of dofs per unknown (it depends on the
  // space where we are looking for solutions) and therefore there will be three
  // big groups of nested for loops

  // Be carefull about the fact that this routin is calld once per each face of
  // the element and the convention of grouping the shape functions per spatial
  // dimension (first all those for x, then those for y and so on) is respected
  // on a face basis. Therefore expect to have submatrices of
  // shapesface_->nfdofs_*shapesface_->nfdofs_ or shapesface_->nfdofs_*ndofs_
  //(or vice-versa), divided by nsd_ * "matrix dimension" submatrices with all
  // entries set to zero.
  Core::LinAlg::SerialDenseMatrix transformatrix(
      (nsd_ - 1) * shapesface_->nfdofs_, nsd_ * shapesface_->nfdofs_);
  for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
    for (unsigned int d = 0; d < nsd_; ++d)
      for (unsigned int q = 0; q < nsd_ - 1; ++q)
        transformatrix(shapesface_->nfdofs_ * q + i, shapesface_->nfdofs_ * d + i) =
            shapesface_->tangent(d, q);

  // MIXED SHAPE FUNCTIONS
  // The matrix that are going to be build here are D,I and J
  // loop over number of internal shape functions
  // Here we need to create only the first part of tghe D and H matrix to be multiplied by the
  // transformation matrices and then put in the real D and H matrices
  Core::LinAlg::SerialDenseMatrix tempD(ndofs_ * nsd_, shapesface_->nfdofs_ * nsd_);
  Core::LinAlg::SerialDenseMatrix tempH(ndofs_ * nsd_, shapesface_->nfdofs_ * nsd_);
  Core::LinAlg::SerialDenseMatrix tempI(shapesface_->nfdofs_ * nsd_, ndofs_ * nsd_);
  Core::LinAlg::SerialDenseMatrix tempJ(shapesface_->nfdofs_ * nsd_, ndofs_ * nsd_);
  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    // If the shape function is zero on the face we can just skip it. Remember
    // that the matrix have already been set to zero and therefore if nothing
    // is done the value ramains zero
    if (shapesface_->shfunctI.nonzero_on_face(i))
    {
      // loop over number of face shape functions
      for (unsigned int j = 0; j < shapesface_->nfdofs_; ++j)
      {
        // Now that the integration has been carried on it is necessary to place
        // the value in the right position inside the matrices
        for (unsigned int d = 0; d < nsd_; ++d)
        {
          // i internal shape functions
          // j boundary shape functions
          for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
          {
            // Storing the value of the integral without the normal components
            const double temp = tau * shapesface_->jfac(q) * shapesface_->shfunctI(i, q) *
                                shapesface_->shfunct(j, q);
            const double temp2 =
                shapesface_->jfac(q) * shapesface_->shfunctI(i, q) * shapesface_->shfunct(j, q);
            // Filling the matrices
            // Dmat
            //+1 coordinate
            tempD(d * ndofs_ + i, shapesface_->nfdofs_ * ((d + 2) % nsd_) + j) +=
                temp2 * shapesface_->normals(((d + 1) % nsd_), q);
            //+2 coordinate
            tempD(d * ndofs_ + i, shapesface_->nfdofs_ * ((d + 1) % nsd_) + j) -=
                temp2 * shapesface_->normals(((d + 2) % nsd_), q);
            // Hmat
            // 0 coordinate
            tempH(d * ndofs_ + i, shapesface_->nfdofs_ * d + j) +=
                temp * (pow(shapesface_->normals(((d + 1) % nsd_), q), 2) +
                           pow(shapesface_->normals(((d + 2) % nsd_), q), 2));
            //+1 coordinate
            tempH(d * ndofs_ + i, shapesface_->nfdofs_ * ((d + 1) % nsd_) + j) -=
                temp * shapesface_->normals(d, q) * shapesface_->normals(((d + 1) % nsd_), q);
            //+2 coordinate
            tempH(d * ndofs_ + i, shapesface_->nfdofs_ * ((d + 2) % nsd_) + j) -=
                temp * shapesface_->normals(d, q) * shapesface_->normals(((d + 2) % nsd_), q);
            // Imat
            //+1 coordinate
            tempI(shapesface_->nfdofs_ * d + j, ((d + 1) % nsd_) * ndofs_ + i) -=
                temp2 * shapesface_->normals((d + 2) % nsd_, q);
            //+2 coordinate
            tempI(shapesface_->nfdofs_ * d + j, ((d + 2) % nsd_) * ndofs_ + i) +=
                temp2 * shapesface_->normals((d + 1) % nsd_, q);
            // Jmat
            // Own coordinate
            // Jmat(shapesface_->nfdofs_*d + j, d * ndofs_ + i) += temp2;
            tempJ(shapesface_->nfdofs_ * d + j, d * ndofs_ + i) +=
                temp * (pow(shapesface_->normals((d + 1) % nsd_, q), 2) +
                           pow(shapesface_->normals((d + 2) % nsd_, q), 2));
            //+1 coordinate
            tempJ(shapesface_->nfdofs_ * d + j, ((d + 1) % nsd_) * ndofs_ + i) -=
                temp * shapesface_->normals(d, q) * shapesface_->normals((d + 1) % nsd_, q);
            //+2 coordinate
            tempJ(shapesface_->nfdofs_ * d + j, ((d + 2) % nsd_) * ndofs_ + i) -=
                temp * shapesface_->normals(d, q) * shapesface_->normals((d + 2) % nsd_, q);
          }
        }  // for (unsigned int d = 0; d < nsd_; ++d)
      }    // for (unsigned int j=0; j<ndofs_; ++j)
    }      // if( shapesface_->shfunctI.NonzeroOnFace(i) )
  }        // for (unsigned int i = 0; i < ndofs_; ++i)

  // Fill face values into the matrices
  {
    Core::LinAlg::SerialDenseMatrix tempMat1(ndofs_ * nsd_, shapesface_->nfdofs_ * (nsd_ - 1));
    Core::LinAlg::SerialDenseMatrix tempMat2(ndofs_ * nsd_, shapesface_->nfdofs_ * (nsd_ - 1));
    Core::LinAlg::SerialDenseMatrix tempMat3(shapesface_->nfdofs_ * (nsd_ - 1), ndofs_ * nsd_);
    Core::LinAlg::SerialDenseMatrix tempMat4(shapesface_->nfdofs_ * (nsd_ - 1), ndofs_ * nsd_);
    Core::LinAlg::multiply_nt(tempMat1, tempD, transformatrix);
    Core::LinAlg::multiply_nt(tempMat2, tempH, transformatrix);
    Core::LinAlg::multiply(tempMat3, transformatrix, tempI);
    Core::LinAlg::multiply(tempMat4, transformatrix, tempJ);

    for (unsigned int i = 0; i < ndofs_ * nsd_; ++i)
      for (unsigned int j = 0; j < shapesface_->nfdofs_ * (nsd_ - 1); ++j)
      {
        Dmat(i, newindex + j) = tempMat1(i, j);
        Hmat(i, newindex + j) = tempMat2(i, j);
        Imat(newindex + j, i) = tempMat3(j, i);
        Jmat(newindex + j, i) = tempMat4(j, i);
      }
  }


  // BOUNDARY SHAPE FUNCTIONS
  // Core::LinAlg::SerialDenseMatrix tempL(shapesface_->nfdofs_ * (nsd_-1), shapesface_->nfdofs_ *
  // (nsd_-1));
  // loop over number of shape functions
  for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
  {
    // loop over number of shape functions
    for (unsigned int j = 0; j < shapesface_->nfdofs_; ++j)
    {
      for (unsigned int d = 0; d < nsd_ - 1; ++d)
      {
        // If the face is perpendicular to the d direction it is necessary to
        // enforce the component of the hybrid variable to be zero because we
        // know that the hybrid variable is defined as the perpendicular
        // component of the elctric field.
        for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
        {
          const double temp =
              tau * shapesface_->jfac(q) * shapesface_->shfunct(i, q) * shapesface_->shfunct(j, q);
          Lmat(newindex + shapesface_->nfdofs_ * d + i, newindex + shapesface_->nfdofs_ * d + j) -=
              temp;  //* (pow(shapesface_->normals(((d+1)%nsd_), q),2) +
                     // pow(shapesface_->normals(((d+2)%nsd_), q),2));
        }
      }
    }  // for (unsigned int j = 0; j < shapesface_->nfdofs_; ++j)
  }    // for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)

  // INTERIOR SHAPE FUNCTIONS
  // Some terms are still missing in G!!
  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int j = 0; j < ndofs_; ++j)
    {
      if (shapesface_->shfunctI.nonzero_on_face(i) && shapesface_->shfunctI.nonzero_on_face(j))
      {
        for (unsigned int d = 0; d < nsd_; ++d)
        {
          for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
          {
            const double temp = tau * shapesface_->jfac(q) * shapesface_->shfunctI(i, q) *
                                shapesface_->shfunctI(j, q);
            // Gmat
            // 0 coordinate
            Gmat(d * ndofs_ + i, d * ndofs_ + j) -=
                temp * (pow(shapesface_->normals(((d + 1) % nsd_), q), 2) +
                           pow(shapesface_->normals(((d + 2) % nsd_), q), 2));
            //+1 coordinate
            Gmat(d * ndofs_ + i, ((d + 1) % nsd_) * ndofs_ + j) +=
                temp * shapesface_->normals(d, q) * shapesface_->normals(((d + 1) % nsd_), q);
            //+2 coordinate
            Gmat(d * ndofs_ + i, ((d + 2) % nsd_) * ndofs_ + j) +=
                temp * shapesface_->normals(d, q) * shapesface_->normals(((d + 2) % nsd_), q);
          }
        }
      }
    }
  }

  return;
}  // ComputeFaceMatrices


/*----------------------------------------------------------------------*
 * CondenseLocalPart
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::LocalSolver::condense_local_part(
    Core::LinAlg::SerialDenseMatrix& eleMat)
{
  TEUCHOS_FUNC_TIME_MONITOR("Discret::Elements::ElemagEleCalc::CondenseLocalPart");

  // THE MATRIX
  //                             -1
  //                   +--------+    +-----+
  //                   |        |    |     |
  //                   | A   C  |    |  D  |
  //  K = L - [ I  J ] |        |    |     |  = L - I X - J Y
  //                   | F  E+G |    |  H  |
  //                   +--------+    +-----+

  //   Y = [ (E+G) - F A^{-1} C ]^{-1} [ H - F A^{-1} D]

  //   X = A^{-1} [ D - C Y ]

  const unsigned int onfdofs = eleMat.numRows();
  const unsigned int intdofs = ndofs_ * nsd_;

  // Thi can be useful to remember when coding
  // int 	Multiply (char TransA, char TransB, double ScalarAB, Matrix &A, Matrix &B, double
  // ScalarThis) this = ScalarThis*this + ScalarAB*A*B
  Core::LinAlg::SerialDenseMatrix tempMat1(intdofs, intdofs);
  Core::LinAlg::multiply(tempMat1, Fmat, invAmat);  // =  F A^{-1}

  Core::LinAlg::SerialDenseMatrix tempMat2(intdofs, intdofs);

  // This is E+G
  tempMat2 += Emat;  // = E
  tempMat2 += Gmat;  // = E + G

  Core::LinAlg::multiply(1.0, tempMat2, -1.0, tempMat1, Cmat);  // = (E+G) - F A^{-1} C

  Core::LinAlg::SerialDenseMatrix tempMat3(intdofs, onfdofs);
  tempMat3 += Hmat;                                             // = H
  Core::LinAlg::multiply(1.0, tempMat3, -1.0, tempMat1, Dmat);  // = H - F A^{-1} D

  // Inverting the first part of the Y matrix
  {
    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseinW;
    inverseinW.setMatrix(Teuchos::rcpFromRef(tempMat2));
    int err = inverseinW.invert();
    if (err != 0)
      FOUR_C_THROW(
          "Inversion of temporary matrix for Schur complement failed with errorcode %d", err);
  }
  // tempMat2 = [(E+G) - F A^{-1} C]^{-1}

  eleMat = Lmat;  // = L
  // reusing matrix that are not needed
  tempMat1.shape(intdofs, onfdofs);
  Core::LinAlg::multiply(
      tempMat1, tempMat2, tempMat3);  //  Y = [(E+G) - F A^{-1} C]^{-1}(H - F A^{-1} D)
  Core::LinAlg::multiply(1.0, eleMat, -1.0, Jmat, tempMat1);  // = L - J Y

  tempMat2.shape(intdofs, onfdofs);
  tempMat2 = Dmat;
  Core::LinAlg::multiply(1.0, tempMat2, -1.0, Cmat, tempMat1);  // = D - C Y

  tempMat3.shape(intdofs, onfdofs);
  Core::LinAlg::multiply(tempMat3, invAmat, tempMat2);  // = X = A^{-1} ( D - C Y )

  Core::LinAlg::multiply(1.0, eleMat, -1.0, Imat, tempMat3);  // = K = L - I X - J y

  return;
}  // CondenseLocalPart

/*----------------------------------------------------------------------*
 * Compute internal and face matrices
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ElemagEleCalc<distype>::LocalSolver::compute_matrices(
    Core::FE::Discretization& discretization, const std::shared_ptr<Core::Mat::Material>& mat,
    Discret::Elements::Elemag& ele, double dt, Inpar::EleMag::DynamicType dyna, const double tau)
{
  // The material properties change elementwise or can also be computed pointwise?
  // Check current_informations, \chapter{Elements and materials for electromagnetics},
  // \section{Remarks}
  const Mat::ElectromagneticMat* elemagmat = static_cast<const Mat::ElectromagneticMat*>(mat.get());
  double sigma = elemagmat->sigma(ele.id());
  double epsilon = elemagmat->epsilon(ele.id());
  double mu = elemagmat->mu(ele.id());

  // Why this? Why do we need to make these matrices zero here? Why not all of them?
  // init face matrices
  invAmat.putScalar(0.0);
  Amat.putScalar(0.0);
  Cmat.putScalar(0.0);
  Dmat.putScalar(0.0);
  Emat.putScalar(0.0);
  Fmat.putScalar(0.0);
  Gmat.putScalar(0.0);
  Hmat.putScalar(0.0);
  Imat.putScalar(0.0);
  Jmat.putScalar(0.0);
  Lmat.putScalar(0.0);

  // Here is the computation for the matrices of volume integrals
  compute_interior_matrices(dt, sigma, mu, epsilon);

  // sumindex is going to be used to decide where we are inside the face matrix
  // because for every face we move to different dofs
  int sumindex = 0;
  int newindex = 0;
  for (unsigned int face = 0; face < nfaces_; ++face)
  {
    /* This part is to be used for efficiency reasons, at the beginning the
    //standard procedure is used
    Core::FE::ShapeValuesFaceParams svfparams(
        ele.Faces()[face]->Degree(),
        shapes_.usescompletepoly_, 2 * ele.Faces()[face]->Degree());
    shapesface_ = Core::FE::ShapeValuesFaceCache<distype>::instance().Create(svfparams);
    */

    // Updating face data
    shapesface_->evaluate_face(ele, face);

    // Here are the matrices for the boundary integrals
    compute_face_matrices(face, dt, sumindex, newindex, sigma, mu, tau);
    sumindex += nsd_ * shapesface_->nfdofs_;
    newindex += (nsd_ - 1) * shapesface_->nfdofs_;
  }

  return;
}

// template classes
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::hex8>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::hex20>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::hex27>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::tet4>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::tet10>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::wedge6>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::pyramid5>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::quad4>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::quad8>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::quad9>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::tri3>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::tri6>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::nurbs9>;
template class Discret::Elements::ElemagEleCalc<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
