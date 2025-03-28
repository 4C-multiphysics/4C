// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_hdg.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_hdg.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_fem_general_utils_polynomial.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_global_data.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_scatra.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_calc.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_function.hpp"

#include <Teuchos_SerialDenseSolver.hpp>
#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN



/*----------------------------------------------------------------------*
 * Constructor
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::ScaTraEleCalcHDG(
    const int numdofpernode, const int numscal, const std::string& disname)
    : numdofpernode_(numdofpernode),
      numscal_(numscal),
      usescompletepoly_(),
      scatrapara_(Discret::Elements::ScaTraEleParameterStd::instance(disname))

{
}


/*----------------------------------------------------------------------*
 | singleton access method                               hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcHDG<distype, probdim>*
Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname, bool create)
{
  static std::map<std::string, ScaTraEleCalcHDG<distype, probdim>*> instances;

  if (create)
  {
    if (instances.find(disname) == instances.end())
      instances[disname] = new ScaTraEleCalcHDG<distype, probdim>(numdofpernode, numscal, disname);
  }

  else if (instances.find(disname) != instances.end())
  {
    for (typename std::map<std::string, ScaTraEleCalcHDG<distype, probdim>*>::iterator i =
             instances.begin();
        i != instances.end(); ++i)
    {
      delete i->second;
      i->second = nullptr;
    }

    instances.clear();
    return nullptr;
  }

  return instances[disname];
}

/*----------------------------------------------------------------------*
 | Initialize Shapes                                     hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::initialize_shapes(
    const Core::Elements::Element* ele, const std::string& disname)
{
  //  Discret::Elements::ScaTraHDG * hdgele =
  //  dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele));
  // Check if this is an HDG element, if yes, can initialize...
  if (Discret::Elements::ScaTraHDG* hdgele =
          dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele)))
  {
    usescompletepoly_ = hdgele->uses_complete_polynomial_space();

    if (shapes_ == nullptr)
      shapes_ = std::make_shared<Core::FE::ShapeValues<distype>>(
          hdgele->degree(), usescompletepoly_, 2 * hdgele->degree());
    else if (shapes_->degree_ != unsigned(hdgele->degree()) ||
             shapes_->usescompletepoly_ != usescompletepoly_)
      shapes_ = std::make_shared<Core::FE::ShapeValues<distype>>(
          hdgele->degree(), usescompletepoly_, 2 * hdgele->degree());

    int onfdofs = 0;
    for (unsigned int i = 0; i < nfaces_; ++i)
    {
      Core::FE::ShapeValuesFaceParams svfparams(
          ele->faces()[i]->degree(), shapes_->usescompletepoly_, 2 * ele->faces()[i]->degree());

      shapesface_ = Core::FE::ShapeValuesFaceCache<distype>::instance().create(svfparams);
      onfdofs += shapesface_->nfdofs_;
    }
    hdgele->set_dofs(shapes_->ndofs_);
    hdgele->set_onf_dofs(onfdofs);

    // check if only one scalar is defined
    if (numscal_ > 1) FOUR_C_THROW("Not implemented for multiple scalars");

    if (local_solver_ == nullptr)
      local_solver_ =
          std::make_shared<LocalSolver>(ele, *shapes_, *shapesface_, usescompletepoly_, disname, 1);
  }
  else
    FOUR_C_THROW("Only works for HDG transport elements");
}

/*----------------------------------------------------------------------*
 | Evaluate                                              hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::evaluate(Core::Elements::Element* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix&, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector&, Core::LinAlg::SerialDenseVector&)
{
  // check if this is an hdg element
  const Discret::Elements::ScaTraHDG* hdgele =
      dynamic_cast<const Discret::Elements::ScaTraHDG*>(ele);
  if (!hdgele) FOUR_C_THROW("Cannot cast element to scatra hdg element");

  initialize_shapes(ele, discretization.name());

  shapes_->evaluate(*ele);

  read_global_vectors(ele, discretization, la);
  get_material_params(ele);

  elevec1.putScalar(0.0);
  if (!local_solver_->scatrapara_->semi_implicit())
  {
    elemat1.putScalar(0.0);
    local_solver_->add_diff_mat(elemat1, hdgele);
    local_solver_->add_reac_mat(elemat1, hdgele);
  }
  local_solver_->compute_residual(
      params, elevec1, elemat1, interiorPhin_, tracenm_, tracen_, hdgele);

  return 0;
}


/*----------------------------------------------------------------------*
 * Evaluate Service                                      hoermann  09/15|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::evaluate_service(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // check if this is an hdg element
  Discret::Elements::ScaTraHDG* hdgele = dynamic_cast<Discret::Elements::ScaTraHDG*>(ele);
  if (!hdgele) FOUR_C_THROW("cannot cast element to scatrahdg element");

  // get the action required

  ScaTra::Action act;
  if (params.get<bool>("hdg_action", false))
  {
    switch (Teuchos::getIntegralValue<Core::FE::HDGAction>(params, "action"))
    {
      case Core::FE::HDGAction::project_dirich_field:
        act = ScaTra::Action::project_dirich_field;
        break;
      default:
        FOUR_C_THROW("HDG Action type not supported");
    }
  }
  else
  {
    act = Teuchos::getIntegralValue<ScaTra::Action>(params, "action");
  }

  initialize_shapes(ele, discretization.name());

  switch (act)
  {
    case ScaTra::Action::update_interior_variables:
    {
      shapes_->evaluate(*ele);
      read_global_vectors(ele, discretization, la);

      return update_interior_variables(hdgele, params, elevec1_epetra);
      break;
    }

    case ScaTra::Action::interpolate_hdg_to_node:
    {
      shapes_->evaluate(*ele);
      read_global_vectors(ele, discretization, la);
      return node_based_values(ele, discretization, elevec1_epetra);
      break;
    }

    case ScaTra::Action::set_initial_field:
    {
      element_init(ele);
      prepare_material_params(ele);
      // set initial field
      return set_initial_field(ele, params, elevec1_epetra, elevec2_epetra);

      break;
    }
    case ScaTra::Action::calc_mat_initial:
    {
      if (hdgele->padapt_ele() || !hdgele->mat_init())
      {
        shapes_->evaluate(*ele);
        element_init(ele);
        read_global_vectors(ele, discretization, la);
        prepare_material_params(ele);
        local_solver_->compute_matrices(ele);
        local_solver_->condense_local_part(hdgele);
      }
      elemat1_epetra.putScalar(0.0);
      local_solver_->add_diff_mat(elemat1_epetra, hdgele);

      break;
    }
    case ScaTra::Action::project_material_field:
    {
      project_material_field(ele);
      break;
    }
    case ScaTra::Action::project_field:
    {
      shapes_->evaluate(*ele);
      return project_field(ele, discretization, params, elevec1_epetra, elevec2_epetra, la);
      break;
    }
    case ScaTra::Action::time_update_material:
    {
      time_update_material(ele);
      break;
    }
    case ScaTra::Action::get_material_internal_state:
    {
      get_material_internal_state(ele, params, discretization);
      break;
    }
    case ScaTra::Action::set_material_internal_state:
    {
      set_material_internal_state(ele, params, discretization);
      break;
    }
    case ScaTra::Action::project_dirich_field:
    {
      if (params.isParameter("faceconsider"))
      {
        return project_dirich_field(ele, params, discretization, la, elevec1_epetra);
      }
      break;
    }
    case ScaTra::Action::project_neumann_field:
    {
      int face = params.get<int>("face");
      int sumindex = 0;
      for (int i = 0; i < face; ++i)
      {
        Core::FE::PolynomialSpaceParams parameter(Core::FE::DisTypeToFaceShapeType<distype>::shape,
            ele->faces()[i]->degree(), shapes_->usescompletepoly_);
        int nfdofs = Core::FE::PolynomialSpaceCache<nsd_ - 1>::instance().create(parameter)->size();
        sumindex += nfdofs;
      }
      local_solver_->compute_neumann_bc(ele, params, face, elevec1_epetra, sumindex);
      break;
    }
    case ScaTra::Action::calc_padaptivity:
    {
      shapes_->evaluate(*ele);
      read_global_vectors(ele, discretization, la);
      return calc_p_adaptivity(ele, discretization, params);
      break;
    }
    case ScaTra::Action::calc_error:
    {
      shapes_->evaluate(*ele);
      read_global_vectors(ele, discretization, la);
      return calc_error(ele, params, elevec1_epetra);
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown type of action for ScaTraHDG");
      break;
    }
  }  // end of switch(act)

  return 0;
}

/*----------------------------------------------------------------------*
 | Calculate node based values                           hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::node_based_values(
    Core::Elements::Element* ele, Core::FE::Discretization& discretization,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  FOUR_C_ASSERT(elevec1.numRows() == (int)nen_ * (2 + nsd_), "Vector does not have correct size");
  Core::LinAlg::SerialDenseMatrix locations =
      Core::FE::get_ele_node_numbering_nodes_paramspace(distype);
  Core::LinAlg::SerialDenseVector values(shapes_->ndofs_);

  Discret::Elements::ScaTraHDG* hdgele = dynamic_cast<Discret::Elements::ScaTraHDG*>(ele);

  for (unsigned int i = 0; i < nen_; ++i)
  {
    // evaluate shape polynomials in node
    for (unsigned int idim = 0; idim < nsd_; idim++) shapes_->xsi(idim) = locations(idim, i);
    shapes_->polySpace_->evaluate(shapes_->xsi, values);

    // compute values for concentrations (and gradients) by summing over all basis functions
    double sum = 0;
    std::vector<double> sumgrad(nsd_, 0.0);
    for (unsigned int k = 0; k < hdgele->ndofs_; ++k)
    {
      sum += values(k) * interiorPhinp_(k);
      for (unsigned int d = 0; d < nsd_; ++d)
        sumgrad[d] += values(k) * interiorPhinp_(k + (d + 1) * hdgele->ndofs_);
    }
    // store node value for concentrations and gradient in element vector
    elevec1(i) = sum;
    for (unsigned int d = 0; d < nsd_; ++d) elevec1(i + (2 + d) * nen_) = sumgrad[d];
  }

  // get trace solution values
  locations = Core::FE::get_ele_node_numbering_nodes_paramspace(
      Core::FE::DisTypeToFaceShapeType<distype>::shape);


  Core::LinAlg::SerialDenseVector touchcount(nen_);
  Core::LinAlg::SerialDenseVector fvalues(1);
  int sumindex = 0;
  for (unsigned int face = 0; face < nfaces_; ++face)
  {
    Core::FE::ShapeValuesFaceParams svfparams(
        ele->faces()[face]->degree(), shapes_->usescompletepoly_, 2 * ele->faces()[face]->degree());
    shapesface_ = Core::FE::ShapeValuesFaceCache<distype>::instance().create(svfparams);
    shapesface_->evaluate_face(*ele, face);

    fvalues.resize(shapesface_->nfdofs_);

    for (int i = 0; i < Core::FE::DisTypeToNumNodePerFace<distype>::numNodePerFace; ++i)
    {
      // evaluate shape polynomials in node
      for (unsigned int idim = 0; idim < nsd_ - 1; idim++)
        shapesface_->xsi(idim) = locations(idim, i);

      shapesface_->polySpace_->evaluate(shapesface_->xsi, fvalues);

      double sum = 0.0;
      for (unsigned int k = 0; k < shapesface_->nfdofs_; ++k)
        sum += fvalues(k) * tracen_(sumindex + k);

      elevec1(nen_ + shapesface_->faceNodeOrder[face][i]) += sum;
      touchcount(shapesface_->faceNodeOrder[face][i])++;
    }
    sumindex += shapesface_->nfdofs_;
  }

  for (unsigned int i = 0; i < nen_; ++i) elevec1(nen_ + i) /= touchcount(i);

  return 0;
}  // NodeBasedValues

/*----------------------------------------------------------------------*
 * ProjectDirichField                                  berardocco 05/20 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::project_dirich_field(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Elements::LocationArray& la,
    Core::LinAlg::SerialDenseVector& elevec1)
{
  // get actual time
  const double time = params.get<double>("time");

  Teuchos::Array<int>* func = params.getPtr<Teuchos::Array<int>>("funct");

  const int face = params.get<unsigned int>("faceconsider");
  Core::FE::ShapeValuesFaceParams svfparams(
      ele->faces()[face]->degree(), shapes_->usescompletepoly_, 2 * ele->faces()[face]->degree());

  shapesface_ = Core::FE::ShapeValuesFaceCache<distype>::instance().create(svfparams);

  shapesface_->evaluate_face(*ele, face);

  Core::LinAlg::SerialDenseMatrix mass(shapesface_->nfdofs_, shapesface_->nfdofs_);
  Core::LinAlg::SerialDenseVector trVec(shapesface_->nfdofs_);

  // integration loop
  for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
  {
    // global coordinates of current Gauss point
    double coordgp[3];  // we always need three coordinates for function evaluation!
    for (int i = 0; i < 3; ++i) coordgp[i] = shapesface_->xyzreal(i, q);

    const double fac = shapesface_->jfac(q);
    // evaluate function at current Gauss point (provide always 3D coordinates!)
    const double functfac = Global::Problem::instance()
                                ->function_by_id<Core::Utils::FunctionOfSpaceTime>((*func)[0])
                                .evaluate(coordgp, time, 0);

    // Creating the mass matrix and the RHS vector
    for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
    {
      // Mass matrix
      for (unsigned int j = 0; j < shapesface_->nfdofs_; ++j)
        mass(i, j) += shapesface_->shfunct(i, q) * shapesface_->shfunct(j, q) * fac;

      // RHS
      trVec(i) += shapesface_->shfunct(i, q) * functfac * fac;
    }

  }  // loop over integration points

  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
  inverseMass.setMatrix(Teuchos::rcpFromRef(mass));
  inverseMass.setVectors(Teuchos::rcpFromRef(trVec), Teuchos::rcpFromRef(trVec));
  inverseMass.solve();

  for (unsigned int node = 0; node < shapesface_->nfdofs_; node++) elevec1[node] = trVec(node);

  return 0;
}  // ProjectDirichField


/*----------------------------------------------------------------------*
 * read_global_vectors                                     hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::read_global_vectors(
    Core::Elements::Element* ele, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la)
{
  Discret::Elements::ScaTraHDG* hdgele =
      dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele));

  // read the HDG solution vector (for traces)

  tracen_.size(hdgele->onfdofs_);
  interiorPhin_.size(shapes_->ndofs_ * (nsd_ + 1));
  interiorPhinp_.size(shapes_->ndofs_ * (nsd_ + 1));
  tracenm_.size(hdgele->onfdofs_);
  std::shared_ptr<const Core::LinAlg::Vector<double>> phiaf = discretization.get_state("phiaf");
  if (phiaf == nullptr) FOUR_C_THROW("Cannot get state vector phiaf");
  Core::FE::extract_my_values(*phiaf, tracen_, la[0].lm_);


  if (discretization.has_state("phin"))
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> phin = discretization.get_state("phin");
    Core::FE::extract_my_values(*phin, tracenm_, la[0].lm_);
  }

  std::shared_ptr<const Core::LinAlg::Vector<double>> intphinp =
      discretization.get_state(2, "intphinp");
  if (intphinp == nullptr) FOUR_C_THROW("Cannot get state vector intphinp");
  std::vector<int> localDofs = discretization.dof(2, ele);
  Core::FE::extract_my_values(*intphinp, interiorPhinp_, localDofs);

  if (discretization.has_state(2, "intphin"))
  {
    std::shared_ptr<const Core::LinAlg::Vector<double>> intphin =
        discretization.get_state(2, "intphin");
    Core::FE::extract_my_values(*intphin, interiorPhin_, localDofs);
  }

  return;
}  // read_global_vectors


/*----------------------------------------------------------------------*
 * LocalSolver                                           hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::LocalSolver::LocalSolver(
    const Core::Elements::Element* ele, Core::FE::ShapeValues<distype>& shapeValues,
    Core::FE::ShapeValuesFace<distype>& shapeValuesFace, bool completepoly,
    const std::string& disname,
    int numscal)
    :  // ndofs_ (shapeValues.ndofs_),
       //    shapes_(&shapeValues),
       //    shapesface_(&shapeValuesFace),
       //    Amat(ndofs_,ndofs_),
       //    Bmat(ndofs_,nsd_*ndofs_),
       //    Cmat(),
       //    Dmat(nsd_*ndofs_,nsd_*ndofs_),
       //    Emat(),
       //    Gmat(),
       //    Hmat(),
       //    Mmat(ndofs_,ndofs_),
       //    EmatT(),
       //    BmatMT(nsd_*ndofs_,ndofs_),
       //    Kmat(),
       //    Ivecn_(hdgele->ndofs_),
       //    Ivecnp_(hdgele->ndofs_),
       //    Imatnpderiv_(hdgele->ndofs_,hdgele->ndofs_),
       //    invAmat(ndofs_,ndofs_),
       //    invAMmat(ndofs_,ndofs_),
       //    massPart(ndofs_,shapeValues.nqpoints_),
       //    massPartW(ndofs_,shapeValues.nqpoints_),
       //    BTAMmat(ndofs_*nsd_, ndofs_),
       //    invCondmat(ndofs_*nsd_, ndofs_*nsd_),
       //    Xmat(),
      onfdofs_(0),
      scatrapara_(Discret::Elements::ScaTraEleParameterStd::instance(disname)),
      scatraparatimint_(Core::Utils::shared_ptr_from_ref(
          *Discret::Elements::ScaTraEleParameterTimInt::instance(disname)))
//    diff_(nsd_,nsd_),                        // diffusion coefficient
//    invdiff_(nsd_,nsd_),                     // inverse diffusion coefficient
//    reacoeff_(numscal),                     // reaction coefficient
//    tau_()

{
}  // LocalSolver


/*----------------------------------------------------------------------*
 * Compute internal and face matrices
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::LocalSolver::compute_matrices(
    Core::Elements::Element* ele)
{
  Discret::Elements::ScaTraHDG* hdgele =
      dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele));

  // init matrices
  hdgele->Amat_.putScalar(0.0);
  hdgele->Bmat_.putScalar(0.0);
  hdgele->Cmat_.putScalar(0.0);
  hdgele->Dmat_.putScalar(0.0);
  hdgele->Emat_.putScalar(0.0);
  hdgele->Gmat_.putScalar(0.0);
  hdgele->Hmat_.putScalar(0.0);
  hdgele->Mmat_.putScalar(0.0);
  hdgele->EmatT_.putScalar(0.0);
  hdgele->BmatMT_.putScalar(0.0);
  hdgele->Kmat_.putScalar(0.0);
  hdgele->invAMmat_.putScalar(0.0);
  hdgele->BTAMmat_.putScalar(0.0);
  hdgele->invCondmat_.putScalar(0.0);
  hdgele->Xmat_.putScalar(0.0);


  bool usescompletepoly = hdgele->uses_complete_polynomial_space();

  shapes_ = std::make_shared<Core::FE::ShapeValues<distype>>(
      hdgele->degree(), usescompletepoly, 2 * ele->degree());
  shapes_->evaluate(*ele);
  compute_interior_matrices(hdgele);

  int sumindex = 0;
  for (unsigned int nface = 0; nface < nfaces_; ++nface)
  {
    Core::FE::ShapeValuesFaceParams svfparams(ele->faces()[nface]->degree(),
        shapes_->usescompletepoly_, 2 * ele->faces()[nface]->degree());

    shapesface_ = Core::FE::ShapeValuesFaceCache<distype>::instance().create(svfparams);

    shapesface_->evaluate_face(*ele, nface);

    compute_face_matrices(nface, sumindex, hdgele);
    sumindex += shapesface_->nfdofs_;
  }

  // calculate AMmat = A + (1/(dt*theta))*M
  double dt = scatraparatimint_->dt();
  double theta = scatraparatimint_->time_fac() * (1 / dt);

  hdgele->invAMmat_ = hdgele->Mmat_;
  hdgele->invAMmat_.scale(1.0 / (dt * theta));
  hdgele->invAMmat_ += hdgele->Amat_;
  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseAMmat;
  inverseAMmat.setMatrix(Teuchos::rcpFromRef(hdgele->invAMmat_));
  int err = inverseAMmat.invert();
  if (err != 0)
  {
    if (scatraparatimint_->is_stationary())
      FOUR_C_THROW(
          "Inversion for AMmat failed with errorcode {}. This might be due to the fact that in "
          "stationary problems Mmat_ is a zero matrix and AMat_ (if there is no convection) only "
          "has boundary integrals. Therefore, if you are using elements with internal degrees of "
          "freedom (high degree?), invAMmat_ matrix will be singular. If none of this is the case, "
          "you'll need to find the problem yourself.",
          err);
    else
      FOUR_C_THROW("Inversion for AMmat failed with errorcode {}", err);
  }
}

/*----------------------------------------------------------------------*
 * ComputeFaceMatrices                                   hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::LocalSolver::compute_face_matrices(
    const int face, int indexstart, Discret::Elements::ScaTraHDG* hdgele)
{
  TEUCHOS_FUNC_TIME_MONITOR("Discret::Elements::ScaTraEleCalcHDG::ComputeFaceMatrices");

  // Compute the matrices C, E and H
  // Here, we don't consider material properties! Don't forget this during condensation

  // stabilization parameter tau is used as defined in the inputfile, i.e. it is NOT multiplied by
  // diffusion, timestep etc. Thus in the input file one has to define it and already multiply with
  // the diffusion, timstep etc. Two possibilities are to use tau=diff/dx or tau=diff/dx+dx/dt.
  double tau = 0.0;

  // set the correct stabilization parameter tau depending on the stabilization method
  switch (scatrapara_->stab_type())
  {
    case (Inpar::ScaTra::stabtype_hdg_centered):
      tau = scatrapara_->tau_value();
      break;
    case (Inpar::ScaTra::stabtype_hdg_upwind):
      if (shapesface_->normal(0, 0) + shapesface_->normal(1, 0) < 0.0)
        tau = 0.0;
      else
        tau = scatrapara_->tau_value();
      break;
    case (Inpar::ScaTra::stabtype_no_stabilization):
      tau = 0.0;
      break;
    default:
      FOUR_C_THROW("Unknown definition for stabilization parameter for HDG");
      break;
  }

  // Add convection term (velocity at quadrature points on face)
  // at the moment it is set to zero
  Core::LinAlg::SerialDenseMatrix velface(nsd_, shapesface_->nqpoints_);

  // loop over number of shape functions (scalar dofs per element)
  for (unsigned int q = 0; q < hdgele->ndofs_; ++q)
  {
    if (shapesface_->shfunctI.nonzero_on_face(q))
    {
      // loop over number of shape functions (scalar dofs per face)
      for (unsigned int p = 0; p < shapesface_->nfdofs_; ++p)
      {
        // C and E
        // numerical integration: sum over quadrature points
        double tempE = 0.0;
        double tempC = 0.0;
        std::array<double, nsd_> temp_d = {0.0};
        // loop over number of quadrature points on face
        for (unsigned int i = 0; i < shapesface_->nqpoints_; ++i)
        {
          double temp =
              shapesface_->jfac(i) * shapesface_->shfunct(p, i) * shapesface_->shfunctI(q, i);
          tempE += temp;
          for (unsigned int j = 0; j < nsd_; ++j)
          {
            velface(j, i) = 0.0;
            temp_d[j] += temp * shapesface_->normals(j, i);
            tempC += temp * velface(j, i) * shapesface_->normals(j, i);
          }
          for (unsigned int j = 0; j < nsd_; ++j)
          {
            hdgele->Emat_(j * hdgele->ndofs_ + q, indexstart + p) = -temp_d[j];
            hdgele->EmatT_(indexstart + p, j * hdgele->ndofs_ + q) = -temp_d[j];
          }
        }
        hdgele->Cmat_(q, indexstart + p) = tempC - tau * tempE;
        hdgele->Gmat_(indexstart + p, q) = tau * tempE;
      }
    }  // for (unsigned int q=0; q<hdgele->ndofs_; ++q)
  }  // for (unsigned int p=0; p<nfdofs_; ++p)

  // H
  // loop over number of shape functions (scalar dofs per face)
  for (unsigned int p = 0; p < shapesface_->nfdofs_; ++p)
  {
    // loop over number of shape functions (scalar dofs per face)
    for (unsigned int q = 0; q < shapesface_->nfdofs_; ++q)
    {
      double tempG = 0.0;
      double tempH = 0.0;
      // loop over number of quadrature points on face
      for (unsigned int i = 0; i < shapesface_->nqpoints_; ++i)
      {
        double temp =
            shapesface_->jfac(i) * shapesface_->shfunct(p, i) * shapesface_->shfunct(q, i);
        tempG += temp;
        for (unsigned int j = 0; j < nsd_; ++j)
        {
          velface(j, i) = 0.0;  // convection
          tempH += temp * velface(j, i) * shapesface_->normals(j, i);
        }
      }
      hdgele->Hmat_(indexstart + p, indexstart + q) = tempH - tau * tempG;
    }  // for (unsigned int q=0; q<nfdofs_; ++q)
  }  // for (unsigned int p=0; p<nfdofs_; ++p)


  // one term is still missing in A
  // loop over number of shape functions (scalar dofs per element)
  for (unsigned int p = 0; p < hdgele->ndofs_; ++p)
  {
    for (unsigned int q = 0; q <= p; ++q)
    {
      double tempA = 0.0;
      if (shapesface_->shfunctI.nonzero_on_face(p) && shapesface_->shfunctI.nonzero_on_face(q))
      {
        // loop over number of quadrature points on face
        for (unsigned int i = 0; i < shapesface_->nqpoints_; ++i)
          tempA += shapesface_->jfac(i) * shapesface_->shfunctI(p, i) * shapesface_->shfunctI(q, i);
        hdgele->Amat_(p, q) += tau * tempA;
        if (p != q) hdgele->Amat_(q, p) += tau * tempA;
      }
    }
  }

  return;
}  // ComputeFaceMatrices

/*----------------------------------------------------------------------*
 * compute_interior_matrices_tet                             hoermann 01/17|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype,
    probdim>::LocalSolver::compute_interior_matrices_tet(Discret::Elements::ScaTraHDG* hdgele)
{
  Core::LinAlg::SerialDenseMatrix vel(nsd_, shapes_->nqpoints_);
  Core::LinAlg::SerialDenseMatrix gradPart(hdgele->ndofs_ * nsd_, shapes_->nqpoints_);
  Core::LinAlg::SerialDenseMatrix gradPartVel(hdgele->ndofs_, shapes_->nqpoints_);

  Core::LinAlg::SerialDenseMatrix massPart(hdgele->ndofs_, shapes_->nqpoints_);
  Core::LinAlg::SerialDenseMatrix massPartW(hdgele->ndofs_, shapes_->nqpoints_);
  std::vector<Core::LinAlg::SerialDenseMatrix> DW(
      nsd_ * nsd_, Core::LinAlg::SerialDenseMatrix(hdgele->ndofs_, hdgele->ndofs_));


  // polynomial space to get the value of the shape function at the material gauss points
  Core::FE::PolynomialSpaceParams params(distype, hdgele->degree(), shapes_->usescompletepoly_);
  std::shared_ptr<Core::FE::PolynomialSpace<probdim>> polySpace =
      Core::FE::PolynomialSpaceCache<probdim>::instance().create(params);

  const Core::FE::IntPointsAndWeights<Core::FE::dim<distype>> intpoints(
      ScaTra::DisTypeToMatGaussRule<distype>::get_gauss_rule(2 * hdgele->degree()));

  std::vector<Core::LinAlg::SerialDenseVector> shape_gp(intpoints.ip().nquad);
  std::vector<Core::LinAlg::SerialDenseMatrix> massPartDW(
      nsd_ * nsd_, Core::LinAlg::SerialDenseMatrix(hdgele->ndofs_, intpoints.ip().nquad));


  // coordinate of gauss points
  Core::LinAlg::Matrix<probdim, 1> gp_coord(Core::LinAlg::Initialization::zero);

  for (int q = 0; q < intpoints.ip().nquad; ++q)
  {
    shape_gp[q].size(polySpace->size());

    // gaussian points coordinates
    for (int idim = 0; idim < Core::FE::dim<distype>; ++idim)
      gp_coord(idim) = intpoints.ip().qxg[q][idim];
    polySpace->evaluate(gp_coord, shape_gp[q]);
  }

  double jacdet = shapes_->xjm.determinant();

  Core::LinAlg::SerialDenseMatrix massPartD(hdgele->ndofs_, shape_gp.size());

  // loop over quadrature points
  for (unsigned int q = 0; q < shape_gp.size(); ++q)
  {
    // loop over shape functions
    for (unsigned int i = 0; i < hdgele->ndofs_; ++i)
    {
      massPartD(i, q) = shape_gp[q](i);

      if (hdgele->invdiff_.size() == 1)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            massPartDW[d * nsd_ + e](i, q) =
                shape_gp[q](i) * jacdet * intpoints.ip().qwgt[q] * hdgele->invdiff_[0](d, e);
      else if (hdgele->invdiff_.size() == shape_gp.size())
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            massPartDW[d * nsd_ + e](i, q) =
                shape_gp[q](i) * jacdet * intpoints.ip().qwgt[q] * hdgele->invdiff_[q](d, e);
      else
        FOUR_C_THROW("Diffusion tensor not defined properly");
    }
  }

  // loop over quadrature points
  for (unsigned int q = 0; q < shapes_->nqpoints_; ++q)
  {
    // loop over shape functions
    for (unsigned int i = 0; i < hdgele->ndofs_; ++i)
    {
      massPart(i, q) = shapes_->shfunct(i, q);
      massPartW(i, q) = shapes_->shfunct(i, q) * shapes_->jfac(q);

      for (unsigned int d = 0; d < nsd_; ++d)
      {
        vel(d, q) = 0.0;
        gradPart(d * hdgele->ndofs_ + i, q) = shapes_->shderxy(i * nsd_ + d, q);

        gradPartVel(i, q) += shapes_->shderxy(i * nsd_ + d, q) * vel(d, q);
      }
    }
  }

  for (unsigned int d = 0; d < nsd_; ++d)
    for (unsigned int e = 0; e < nsd_; ++e)
      Core::LinAlg::multiply_nt(DW[d * nsd_ + e], massPartD, massPartDW[d * nsd_ + e]);


  // multiply matrices to perform summation over quadrature points
  if (not scatraparatimint_->is_stationary())
  {
    Core::LinAlg::multiply_nt(hdgele->Mmat_, massPart, massPartW);
  }
  Core::LinAlg::multiply_nt(0.0, hdgele->Amat_, -1.0, gradPartVel,
      massPartW);  // first part of A matrix (only if velocity field not zero)
  Core::LinAlg::multiply_nt(0.0, hdgele->Bmat_, -1.0, massPartW, gradPart);

  for (unsigned int j = 0; j < hdgele->ndofs_; ++j)
    for (unsigned int i = 0; i < hdgele->ndofs_; ++i)
    {
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        for (unsigned int e = 0; e < nsd_; e++)
          hdgele->Dmat_(d * hdgele->ndofs_ + i, e * hdgele->ndofs_ + j) = DW[d * nsd_ + e](i, j);
        hdgele->BmatMT_(d * hdgele->ndofs_ + i, j) =
            -1.0 * hdgele->Bmat_(j, d * hdgele->ndofs_ + i);
      }
    }

  return;
}  // compute_interior_matrices_tet

/*----------------------------------------------------------------------*
 * compute_interior_matrices                                hoermann 01/17|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::LocalSolver::compute_interior_matrices(
    Discret::Elements::ScaTraHDG* hdgele)
{
  if (distype == Core::FE::CellType::tet4 or distype == Core::FE::CellType::tet10)
    compute_interior_matrices_tet(hdgele);
  else
    compute_interior_matrices_all(hdgele);

  return;
}

/*----------------------------------------------------------------------*
 * compute_interior_matrices                                hoermann 09/15|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype,
    probdim>::LocalSolver::compute_interior_matrices_all(Discret::Elements::ScaTraHDG* hdgele)
{
  Core::LinAlg::SerialDenseMatrix vel(nsd_, shapes_->nqpoints_);
  Core::LinAlg::SerialDenseMatrix gradPart(hdgele->ndofs_ * nsd_, shapes_->nqpoints_);
  Core::LinAlg::SerialDenseMatrix gradPartVel(hdgele->ndofs_, shapes_->nqpoints_);

  Core::LinAlg::SerialDenseMatrix massPart(hdgele->ndofs_, shapes_->nqpoints_);
  Core::LinAlg::SerialDenseMatrix massPartW(hdgele->ndofs_, shapes_->nqpoints_);
  std::vector<Core::LinAlg::SerialDenseMatrix> massPartDW(
      nsd_ * nsd_, Core::LinAlg::SerialDenseMatrix(hdgele->ndofs_, shapes_->nqpoints_));
  std::vector<Core::LinAlg::SerialDenseMatrix> DW(
      nsd_ * nsd_, Core::LinAlg::SerialDenseMatrix(hdgele->ndofs_, hdgele->ndofs_));

  // loop over quadrature points
  for (unsigned int q = 0; q < shapes_->nqpoints_; ++q)
  {
    // loop over shape functions
    for (unsigned int i = 0; i < hdgele->ndofs_; ++i)
    {
      massPart(i, q) = shapes_->shfunct(i, q);
      massPartW(i, q) = shapes_->shfunct(i, q) * shapes_->jfac(q);

      if (hdgele->invdiff_.size() == 1)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            massPartDW[d * nsd_ + e](i, q) =
                shapes_->shfunct(i, q) * shapes_->jfac(q) * hdgele->invdiff_[0](d, e);
      else if (hdgele->invdiff_.size() == shapes_->nqpoints_)
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            massPartDW[d * nsd_ + e](i, q) =
                shapes_->shfunct(i, q) * shapes_->jfac(q) * hdgele->invdiff_[q](d, e);
      else
        FOUR_C_THROW("Diffusion tensor not defined properly");


      for (unsigned int d = 0; d < nsd_; ++d)
      {
        vel(d, q) = 0.0;
        gradPart(d * hdgele->ndofs_ + i, q) = shapes_->shderxy(i * nsd_ + d, q);

        gradPartVel(i, q) += shapes_->shderxy(i * nsd_ + d, q) * vel(d, q);
      }
    }
  }



  for (unsigned int d = 0; d < nsd_; ++d)
    for (unsigned int e = 0; e < nsd_; ++e)
      Core::LinAlg::multiply_nt(DW[d * nsd_ + e], massPart, massPartDW[d * nsd_ + e]);


  // multiply matrices to perform summation over quadrature points
  if (not scatraparatimint_->is_stationary())
  {
    Core::LinAlg::multiply_nt(hdgele->Mmat_, massPart, massPartW);
  }
  Core::LinAlg::multiply_nt(0.0, hdgele->Amat_, -1.0, gradPartVel,
      massPartW);  // first part of A matrix (only if velocity field not zero)
  Core::LinAlg::multiply_nt(0.0, hdgele->Bmat_, -1.0, massPartW, gradPart);

  for (unsigned int j = 0; j < hdgele->ndofs_; ++j)
    for (unsigned int i = 0; i < hdgele->ndofs_; ++i)
    {
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        for (unsigned int e = 0; e < nsd_; e++)
          hdgele->Dmat_(d * hdgele->ndofs_ + i, e * hdgele->ndofs_ + j) = DW[d * nsd_ + e](i, j);
        hdgele->BmatMT_(d * hdgele->ndofs_ + i, j) =
            -1.0 * hdgele->Bmat_(j, d * hdgele->ndofs_ + i);
      }
    }

  return;
}  // compute_interior_matrices

/*----------------------------------------------------------------------*
 * ComputeResidual
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::LocalSolver::compute_residual(
    Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector& elevec,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseVector& interiorPhin,
    Core::LinAlg::SerialDenseVector& tracen, Core::LinAlg::SerialDenseVector& tracenp,
    const Discret::Elements::ScaTraHDG* hdgele)
{
  TEUCHOS_FUNC_TIME_MONITOR("Discret::Elements::ScaTraEleCalcHDG::ComputeResidual");

  /*
   for trapezoidal rule
                                                   -1
                        +--------------------------+
   +----------------------------------------------------------------+ |                          |
   |      n                          n       n            n         |  }=:s n+1                 |
   (1/dt*theta)M + A    B  |   | M Phi   + dt (1 - theta) ( A Phi   + B Q   + C Lambda   )      | n
   n            n R     = - [ G  E^T ] |                          |   | n       n            n     |
   + dt (1 - theta) ( G Phi    + E^T Q  + H Lambda   )  - |          -B^T         D  |   |   0     +
   dt (1 - theta) ( - B^T Phi   + D Q   + E Lambda   )  |  }=:t
                        +--------------------------+
   +----------------------------------------------------------------+ -1
                                   +-------------------------+
   +---------------------------------------------+ |                         |    | n n+1   | |
   (1/dt*theta)M + A    B  |    |  - dt (1 - theta) I    -   dt theta  I      |
                      - [ G  E^T ] |                         |    | | |      -B^T            D  | |
   0                           |
                                   +-------------------------+
   +---------------------------------------------+ I..reactive term


    With

     s =   -M * U^n + dt*(1-theta) * (  A U^n + B Q^n + C L^n )   - dt*theta I^n+1 -dt*(1-theta) I^n

     t =             dt*(1-theta) * (-B^T U^n  + D Q^n + E L^n )



                              +--------------------------+^-1   +-----+
    n+1                       |  1/(dt*theta)M + A    B  |      |  s  |                           n
   n            n R     =  - [ G  E^T ]      |                          |      |     |     +
   dt(1-theta) ( G Phi    + E^T Q  + H Lambda )  = |        -B^T           D  |      |  t  |
                              +--------------------------+      +-----+

                                                                    n          n      n
                         =  - (G x + E^T y ) +   dt(1-theta) (  G  U    + E^T Q  + H L  )

               with
                                            -1    -1                                    -1
       y = ( D - (-B^T)   (1/(dt*theta)M + A)    B)     (t - (-B^T)   (1/(dt*theta)M + A)    s)

       x = (1/(dt*theta)M + A)^-1 ( s - B y)


       AM = (1/(dt*theta)M + A)
   */

  Core::LinAlg::SerialDenseVector tempinteriorphin(hdgele->ndofs_);
  for (unsigned int i = 0; i < hdgele->ndofs_; i++) tempinteriorphin(i) = interiorPhin(i);

  Core::LinAlg::SerialDenseVector tempinteriorgradphin(hdgele->ndofs_ * nsd_);
  for (unsigned int i = 0; i < hdgele->ndofs_ * nsd_; i++)
    tempinteriorgradphin(i) = interiorPhin(hdgele->ndofs_ + i);

  double dt = scatraparatimint_->dt();
  double theta = scatraparatimint_->time_fac() * (1 / dt);

  Core::LinAlg::SerialDenseVector tempVec1(hdgele->ndofs_);
  Core::LinAlg::SerialDenseVector tempVec2(hdgele->ndofs_ * nsd_);

  if (theta != 1.0)
  {
    Core::LinAlg::multiply(tempVec1, hdgele->Amat_, tempinteriorphin);
    Core::LinAlg::multiply(1.0, tempVec1, 1.0, hdgele->Bmat_, tempinteriorgradphin);
    Core::LinAlg::multiply(
        1.0, tempVec1, 1.0, hdgele->Cmat_, tracen);  // = (  A U^n + B Q^n + C L^n )
    tempVec1.scale(dt * (1.0 - theta));
  }
  Core::LinAlg::multiply(1.0, tempVec1, -1.0, hdgele->Mmat_,
      tempinteriorphin);  // = s = -M * U^n + dt*(1-theta) * (  A U^n + B Q^n + C L^n )

  Core::LinAlg::SerialDenseVector tempVecI(hdgele->ndofs_);
  if (!scatrapara_->semi_implicit())
  {
    // Reaction term
    tempVecI = hdgele->Ivecnp_;
    tempVecI.scale(dt * theta);
    tempVec1 += tempVecI;
    if (theta != 1.0)
    {
      tempVecI = hdgele->Ivecn_;
      tempVecI.scale(dt * (1.0 - theta));
      tempVec1 += tempVecI;
    }
  }
  //= M * U^n + dt*(1-theta) * (  A U^n + B Q^n + C L^n )   - dt*theta I^n+1 -dt*(1-theta) I^n
  else
  {
    tempVecI = hdgele->Ivecn_;
    tempVecI.scale(dt);
    tempVec1 += tempVecI;
  }

  if (theta != 1.0)
  {
    Core::LinAlg::multiply(tempVec2, hdgele->BmatMT_, tempinteriorphin);
    Core::LinAlg::multiply(1.0, tempVec2, 1.0, hdgele->Dmat_, tempinteriorgradphin);
    Core::LinAlg::multiply(
        1.0, tempVec2, 1.0, hdgele->Emat_, tracen);  // = (-B^T U^n  + D Q^n + E L^n )
    tempVec2.scale(dt * (1.0 - theta));  //= t = dt*(1-theta) * (-B^T U^n  + D Q^n + E L^n )
  }

  Core::LinAlg::multiply(
      1.0, tempVec2, -1.0, hdgele->BTAMmat_, tempVec1);  // = t - (-B^T) AM^{-1} s

  Core::LinAlg::SerialDenseVector tempVec3(hdgele->ndofs_ * nsd_);
  Core::LinAlg::multiply(tempVec3, hdgele->invCondmat_,
      tempVec2);  // = y= ( D - (-B^T)   (AM)^-1    B)^-1     (t - (-B^T)   (AM^-1)    s)

  Core::LinAlg::multiply(1.0, tempVec1, -1.0, hdgele->Bmat_, tempVec3);  // = ( s - B y)
  Core::LinAlg::SerialDenseVector tempVec4(hdgele->ndofs_);
  Core::LinAlg::multiply(
      tempVec4, hdgele->invAMmat_, tempVec1);  // = x= (1/(dt*theta)M + A)^-1 ( s - B y)


  if (theta != 1.0)
  {
    Core::LinAlg::multiply(elevec, hdgele->Gmat_, tempinteriorphin);
    Core::LinAlg::multiply(1.0, elevec, 1.0, hdgele->EmatT_, tempinteriorgradphin);
    Core::LinAlg::multiply(
        1.0, elevec, 1.0, hdgele->Hmat_, tracen);  // = (  G  U^n    + E^T Q^n  + H L^n  )
    elevec.scale(dt * (1.0 - theta));  // = dt*(1-theta) * (  G  U^n    + E^T Q^n  + H L^n  )
  }

  Core::LinAlg::multiply(1.0, elevec, -1.0, hdgele->Gmat_, tempVec4);
  Core::LinAlg::multiply(1.0, elevec, -1.0, hdgele->EmatT_,
      tempVec3);  // =  - (G x + E^T y ) +   dt(1-theta) (  G  U    + E^T Q  + H L  )

  Core::LinAlg::multiply(1.0, elevec, 1.0, hdgele->Kmat_, tracenp);

  return;
}  // ComputeResidual


/*----------------------------------------------------------------------*
 * CondenseLocalPart
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::LocalSolver::condense_local_part(
    Discret::Elements::ScaTraHDG* hdgele)
{
  TEUCHOS_FUNC_TIME_MONITOR("Discret::Elements::ScaTraEleCalcHDG::CondenseLocalPart");

  /*
   THE MATRIX

      +-------------------------------------+  +----+
      | 1/(dt*theta)M + A      B         C  |  | U  |
      |                                     |  |    |
      |         -B^T           D         E  |  | Q  |
      |                                     |  |    |
      |           G           E^T        H  |  | L  |
      +-------------------------------------+  +----+


                     (               +----------+^-1  +-----+      )
                     (               |  AM   B  |     |  C  |      )
    K = (dt*theta)   ( - [ G  E^T ]  |          |     |     |  + H )   = (dt*theta) ( - G x - E^T y
   + H )  , (               | -B^T  D  |     |  E  |      ) (               +----------+     +-----+
   )

       with
                            -1   -1                    -1
       y = ( D - (-B^T)   AM    B)     (E - (-B^T)   AM    C)

       x = AM^-1 ( C - B y)


       AM = (1/(dt*theta)M + A)

   */

  int onfdofs = hdgele->Kmat_.numRows();

  double dt = scatraparatimint_->dt();
  double theta = scatraparatimint_->time_fac() * (1 / dt);

  Core::LinAlg::SerialDenseMatrix tempMat1(hdgele->ndofs_ * nsd_, hdgele->ndofs_);
  Core::LinAlg::multiply(tempMat1, hdgele->BmatMT_, hdgele->invAMmat_);  // =  (-B^T) AM^{-1}

  hdgele->BTAMmat_ = tempMat1;

  Core::LinAlg::SerialDenseMatrix tempMat2(hdgele->ndofs_ * nsd_, hdgele->ndofs_ * nsd_);
  tempMat2 = hdgele->Dmat_;

  Core::LinAlg::multiply(1.0, tempMat2, -1.0, tempMat1, hdgele->Bmat_);  // = D - (-B^T) AM^{-1} B
  Core::LinAlg::SerialDenseMatrix tempMat3(hdgele->ndofs_ * nsd_, onfdofs);
  tempMat3 = hdgele->Emat_;
  Core::LinAlg::multiply(1.0, tempMat3, -1.0, tempMat1, hdgele->Cmat_);  // = E - (-B^T) AM^{-1} C

  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseinW;
  inverseinW.setMatrix(Teuchos::rcpFromRef(tempMat2));
  int err = inverseinW.invert();
  if (err != 0)
    FOUR_C_THROW(
        "Inversion of temporary matrix for Schur complement failed with errorcode {}", err);
  // tempMat2 = (  D - H A^{-1} B )^{-1}

  hdgele->invCondmat_ = tempMat2;

  hdgele->Kmat_ = hdgele->Hmat_;  // = H

  Core::LinAlg::SerialDenseMatrix tempMat4(hdgele->ndofs_ * nsd_, onfdofs);
  Core::LinAlg::multiply(tempMat4, tempMat2, tempMat3);                        // = y
  Core::LinAlg::multiply(1.0, hdgele->Kmat_, -1.0, hdgele->EmatT_, tempMat4);  // = - E^T y + H

  Core::LinAlg::SerialDenseMatrix tempMat5(hdgele->ndofs_, onfdofs);
  tempMat5 = hdgele->Cmat_;
  Core::LinAlg::multiply(1.0, tempMat5, -1.0, hdgele->Bmat_, tempMat4);  // = C -B y

  Core::LinAlg::SerialDenseMatrix tempMat6(hdgele->ndofs_, onfdofs);
  Core::LinAlg::multiply(tempMat6, hdgele->invAMmat_, tempMat5);  // = x = AM^{-1} ( C - B y )

  // save for later use
  hdgele->Xmat_ = tempMat6;

  Core::LinAlg::multiply(
      1.0, hdgele->Kmat_, -1.0, hdgele->Gmat_, tempMat6);  // = K = H - G x - E^T y

  hdgele->Kmat_.scale(dt * theta);

  return;
}  // CondenseLocalPart

/*----------------------------------------------------------------------*
 * Add Diffusive Part to Matrix
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::LocalSolver::add_diff_mat(
    Core::LinAlg::SerialDenseMatrix& eleMat, const Discret::Elements::ScaTraHDG* hdgele)
{
  eleMat = hdgele->Kmat_;
  eleMat.scale(-1.0);

  return;
}  // AddDiffMat


/*----------------------------------------------------------------------*
 * Add Reactive Part to Matrix
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::LocalSolver::add_reac_mat(
    Core::LinAlg::SerialDenseMatrix& eleMat, const Discret::Elements::ScaTraHDG* hdgele)
{
  double dt = scatraparatimint_->dt();
  double theta = scatraparatimint_->time_fac() * (1 / dt);

  // add derivative of reaction term
  Core::LinAlg::SerialDenseMatrix tempMat1(hdgele->ndofs_, hdgele->ndofs_);
  tempMat1 = hdgele->Imatnpderiv_;  // = I'

  Core::LinAlg::SerialDenseMatrix tempMat2(hdgele->ndofs_, hdgele->onfdofs_);
  Core::LinAlg::multiply(0.0, tempMat2, -1.0, tempMat1, hdgele->Xmat_);  // = I' * (-x1)

  Core::LinAlg::SerialDenseMatrix tempMat3(hdgele->ndofs_ * nsd_, hdgele->onfdofs_);
  Core::LinAlg::multiply(
      0.0, tempMat3, -1.0, hdgele->BTAMmat_, tempMat2);  // = nullptr*y1 - (-B^T) AM^{-1} I'* (-x1)
  Core::LinAlg::SerialDenseMatrix tempMat4(hdgele->ndofs_ * nsd_, hdgele->onfdofs_);
  Core::LinAlg::multiply(tempMat4, hdgele->invCondmat_,
      tempMat3);  // = y2 = ( D - (-B^T) AM^{-1} B)^-1  (nullptr*y1 - (-B^T) AM^{-1} I'* (-x1))

  Core::LinAlg::multiply(1.0, tempMat2, -1.0, hdgele->Bmat_, tempMat4);  // = I'*(-x1) -B y2

  Core::LinAlg::SerialDenseMatrix tempMat5(hdgele->ndofs_, hdgele->onfdofs_);
  Core::LinAlg::multiply(
      tempMat5, hdgele->invAMmat_, tempMat2);  // = x2 = AM^{-1} ( I'*(-x1) - B y2 )

  Core::LinAlg::multiply(1.0, eleMat, dt * theta, hdgele->EmatT_, tempMat4);  // = K - E^T y2
  Core::LinAlg::multiply(1.0, eleMat, dt * theta, hdgele->Gmat_, tempMat5);   // = K- G x2 - E^T y2

}  // AddReacMat


/*----------------------------------------------------------------------*
 |  Compute Neumann BC                                    hoermann 09/15|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::LocalSolver::compute_neumann_bc(
    Core::Elements::Element* ele, Teuchos::ParameterList& params, int face,
    Core::LinAlg::SerialDenseVector& elevec, int indexstart)
{
  TEUCHOS_FUNC_TIME_MONITOR("Discret::Elements::ScaTraHDGEleCalc::ComputeNeumannBC");

  Core::Conditions::Condition* condition = params.get<Core::Conditions::Condition*>("condition");
  if (condition == nullptr) FOUR_C_THROW("Cannot access Neumann boundary condition!");

  // get actual time
  const double time = scatraparatimint_->time();

  // get values, switches and spatial functions from the condition
  // (assumed to be constant on element boundary)
  const auto onoff = condition->parameters().get<std::vector<int>>("ONOFF");
  const auto val = condition->parameters().get<std::vector<double>>("VAL");
  const auto func = condition->parameters().get<std::vector<std::optional<int>>>("FUNCT");


  Core::FE::ShapeValuesFaceParams svfparams(
      ele->faces()[face]->degree(), shapes_->usescompletepoly_, 2 * ele->faces()[face]->degree());

  shapesface_ = Core::FE::ShapeValuesFaceCache<distype>::instance().create(svfparams);

  shapesface_->evaluate_face(*ele, face);
  shapes_->evaluate(*ele);

  // integration loop
  for (unsigned int iquad = 0; iquad < shapesface_->nqpoints_; ++iquad)
  {
    // factor given by spatial function
    double functfac = 1.0;

    // global coordinates of current Gauss point
    double coordgp[3];  // we always need three coordinates for function evaluation!
    for (int i = 0; i < 3; ++i) coordgp[i] = shapesface_->xyzreal(i, iquad);

    const double* coordgpref = coordgp;  // needed for function evaluation

    if (onoff[0])  // is this dof activated?
    {
      // factor given by spatial function
      if (func[0].has_value() && func[0].value() > 0)
      {
        // evaluate function at current Gauss point (provide always 3D coordinates!)
        functfac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfSpaceTime>(func[0].value())
                       .evaluate(coordgpref, time, 0);
      }
      else
        functfac = 1.;

      const double val_fac_funct_fac = val[0] * shapesface_->jfac(iquad) * functfac;

      for (unsigned int node = 0; node < shapesface_->nfdofs_; node++)
        elevec[indexstart + node] += shapesface_->shfunct(node, iquad) * val_fac_funct_fac;
    }  // if (onoff[dof])
  }  // loop over integration points
}  // ComputeNeumannBC

/*----------------------------------------------------------------------*
 |  prepare material parameter                            hoermann 11/16|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::prepare_material_params(
    Core::Elements::Element* ele  //!< the element we are dealing with
)
{
  std::shared_ptr<std::vector<Core::LinAlg::SerialDenseMatrix>> difftensor =
      std::make_shared<std::vector<Core::LinAlg::SerialDenseMatrix>>();

  // get the material
  std::shared_ptr<Core::Mat::Material> material = ele->material();

  if (material->material_type() == Core::Materials::m_matlist)
  {
    const std::shared_ptr<const Mat::MatList>& actmat =
        std::dynamic_pointer_cast<const Mat::MatList>(material);
    if (actmat->num_mat() < numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < numscal_; ++k)
    {
      int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      for (unsigned int q = 0; q < shapes_->nqpoints_; ++q)
        prepare_materials(ele, singlemat, k, difftensor);
    }
  }
  else
    prepare_materials(ele, material, 0, difftensor);

  Discret::Elements::ScaTraHDG* hdgele = dynamic_cast<Discret::Elements::ScaTraHDG*>(ele);
  for (unsigned int i = 0; i < (*difftensor).size(); ++i)
    local_solver_->prepare_material_parameter(hdgele, (*difftensor)[i]);



  return;
}  // ScaTraEleCalcHDG::get_material_params


/*----------------------------------------------------------------------*
 |  get the material parameter                            hoermann 09/15|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::get_material_params(
    Core::Elements::Element* ele  //!< the element we are dealing with
)
{
  Core::LinAlg::SerialDenseMatrix difftensor(nsd_, nsd_);
  Core::LinAlg::SerialDenseVector ivecn(shapes_->ndofs_);
  Core::LinAlg::SerialDenseVector ivecnp(shapes_->ndofs_);
  Core::LinAlg::SerialDenseMatrix ivecnpderiv(shapes_->ndofs_, shapes_->ndofs_);

  // get the material
  std::shared_ptr<Core::Mat::Material> material = ele->material();

  if (material->material_type() == Core::Materials::m_matlist)
  {
    const std::shared_ptr<const Mat::MatList>& actmat =
        std::dynamic_pointer_cast<const Mat::MatList>(material);
    if (actmat->num_mat() < numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < numscal_; ++k)
    {
      int matid = actmat->mat_id(k);
      std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);

      materials(singlemat, k, difftensor, ivecn, ivecnp, ivecnpderiv);
    }
  }
  else
    materials(material, 0, difftensor, ivecn, ivecnp, ivecnpderiv);

  Discret::Elements::ScaTraHDG* hdgele = dynamic_cast<Discret::Elements::ScaTraHDG*>(ele);
  local_solver_->set_material_parameter(hdgele, ivecn, ivecnp, ivecnpderiv);



  return;
}  // ScaTraEleCalcHDG::get_material_params


/*----------------------------------------------------------------------*
 * update_interior_variables
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::update_interior_variables(
    Discret::Elements::ScaTraHDG* hdgele, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& elevec
    //    double dt
)
{
  /*
   THE MATRIX

      +-------------------------------------+  +----+
      | 1/(dt*theta)M + A      B         C  |  | U  |
      |                                     |  |    |
      |         -B^T           D         E  |  | Q  |
      |                                     |  |    |
      |           G           E^T        H  |  | L  |
      +-------------------------------------+  +----+


    +---------+                       +--------------------------+^-1
  +--------------------------------------------------------------------------------------------+ |
  U^n+1  |                       |  1/(dt*theta)M + A    B  |     |   M * U^n - dt*(1-theta) * (  A
  U^n + B Q^n + C L^n )   - dt*theta I^n+1 -dt*(1-theta) I^n | |         |    = 1/(dt * theta)   |
  |     | | - |  Q^n+1  |                       |        -B^T           D  |     |  - dt*(1-theta) *
  (   -B^T U^n  + D Q^n + E L^n )                                          |
    +---------+                       +--------------------------+
  +--------------------------------------------------------------------------------------------+

                              +--------------------------+^-1    +------+
                              |  1/(dt*theta)M + A    B  |       |   C  |
                        -     |                          |       |      |  L^n+1
                              |        -B^T           D  |       |   E  |
                              +--------------------------+       +------+


  s =   M * U^n - dt*(1-theta) * (  A U^n + B Q^n + C L^n )   - dt*theta I^n+1 -dt*(1-theta) I^n -
  dt* theta C L^n+1

  t =            - dt*(1-theta) * (-B^T U^n  + D Q^n + E L^n ) -     dt*theta E L^n+1



   +---------+                        +--------------------------+^-1   +-----+ +-----+ |  U^n+1  |
  |  1/(dt*theta)M + A    B  |      |  s  |                          |  x  | |         |   =   1/(dt
  * theta)   |                          |      |     |      =   1/(dt * theta)  |     |    , | Q^n+1
  |                        |        -B^T           D  |      |  t  |                          |  y |
   +---------+                        +--------------------------+      +-----+ +-----+

       with
                                             -1    -1                                    -1
       y = ( D - (-B^T)   (1/(dt*theta)M + A)    B)     (t - (-B^T)   (1/(dt*theta)M + A)    s)

       x = (1/(dt*theta)M + A)^-1 ( s - B y)

       AM = (1/(dt*theta)M + A)

   */

  Core::LinAlg::SerialDenseVector tempinteriorphin(hdgele->ndofs_);
  for (unsigned int i = 0; i < hdgele->ndofs_; ++i) tempinteriorphin(i) = interiorPhin_(i);

  Core::LinAlg::SerialDenseVector tempinteriorgradphin(hdgele->ndofs_ * nsd_);
  for (unsigned int i = 0; i < hdgele->ndofs_ * nsd_; ++i)
    tempinteriorgradphin(i) = interiorPhin_(hdgele->ndofs_ + i);

  double dt = local_solver_->scatraparatimint_->dt();
  double theta = local_solver_->scatraparatimint_->time_fac() * (1 / dt);

  Core::LinAlg::SerialDenseVector tempVec1(hdgele->ndofs_);
  if (theta != 1.0)
  {
    Core::LinAlg::multiply(tempVec1, hdgele->Amat_, tempinteriorphin);
    Core::LinAlg::multiply(1.0, tempVec1, 1.0, hdgele->Bmat_, tempinteriorgradphin);
    Core::LinAlg::multiply(
        1.0, tempVec1, 1.0, hdgele->Cmat_, tracenm_);  // = ( A U^n + B Q^n + C L^n )
    tempVec1.scale(-dt * (1. - theta));
  }
  Core::LinAlg::multiply(1.0, tempVec1, 1.0, hdgele->Mmat_,
      tempinteriorphin);  // = -M * U^n + dt*(1-theta) * (  A U^n + B Q^n + C L^n )


  // Reaction term
  Core::LinAlg::SerialDenseVector tempVecI(hdgele->ndofs_);
  if (!local_solver_->scatrapara_->semi_implicit())
  {
    tempVecI = hdgele->Ivecnp_;
    tempVecI.scale(-dt * theta);
    tempVec1 += tempVecI;
    if (theta != 1.0)
    {
      tempVecI = hdgele->Ivecn_;
      tempVecI.scale(-dt * (1.0 - theta));
      tempVec1 += tempVecI;
    }
  }
  else
  {
    tempVecI = hdgele->Ivecn_;
    tempVecI.scale(-dt);
    tempVec1 += tempVecI;
  }


  Core::LinAlg::multiply(1.0, tempVec1, -dt * theta, hdgele->Cmat_,
      tracen_);  //= s = -M * U^n + dt*(1-theta) * (A U^n + B Q^n + C L^n) - dt*theta I^n+1
                 //-dt*(1-theta) I^n - dt* theta C L^n+1

  Core::LinAlg::SerialDenseVector tempVec2(hdgele->ndofs_ * nsd_);
  if (theta != 1.0)
  {
    Core::LinAlg::multiply(tempVec2, hdgele->BmatMT_, tempinteriorphin);
    Core::LinAlg::multiply(1.0, tempVec2, 1.0, hdgele->Dmat_, tempinteriorgradphin);
    Core::LinAlg::multiply(
        1.0, tempVec2, 1.0, hdgele->Emat_, tracenm_);  // = (-B^T U^n  + D Q^n + E L^n )
    tempVec2.scale(-dt * (1. - theta));
  }
  Core::LinAlg::multiply(1.0, tempVec2, -dt * theta, hdgele->Emat_,
      tracen_);  //= t = -dt*(1-theta) * (-B^T U^n  + D Q^n + E L^n )  - dt*theta E L^n+1

  // y = ( D - (-B^T)   (AM)^-1    B)^-1     (t - (-B^T)   (AM^-1)    s)
  // x = (1/(dt*theta)M + A)^-1 ( s - B y)


  Core::LinAlg::multiply(
      1.0, tempVec2, -1.0, hdgele->BTAMmat_, tempVec1);  // = t - (-B^T) AM^{-1} s

  Core::LinAlg::SerialDenseVector tempVec3(hdgele->ndofs_ * nsd_);
  Core::LinAlg::multiply(tempVec3, hdgele->invCondmat_,
      tempVec2);  // = y= ( D - (-B^T)   (AM)^-1    B)^-1     (t - (-B^T)   (AM^-1)    s)

  Core::LinAlg::multiply(1.0, tempVec1, -1.0, hdgele->Bmat_, tempVec3);  // = ( s - B y)
  Core::LinAlg::SerialDenseVector tempVec4(hdgele->ndofs_);
  Core::LinAlg::multiply(
      tempVec4, hdgele->invAMmat_, tempVec1);  // = x= (1/(dt*theta)M + A)^-1 ( s - B y)

  tempVec3.scale(1. / (dt * theta));
  tempVec4.scale(1. / (dt * theta));

  for (unsigned int i = 0; i < hdgele->ndofs_; ++i) elevec(i) = tempVec4(i);
  for (unsigned int i = 0; i < nsd_ * hdgele->ndofs_; ++i) elevec(hdgele->ndofs_ + i) = tempVec3(i);


  return 0;
}  // update_interior_variables


/*----------------------------------------------------------------------*
 * SetInitialField
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::set_initial_field(
    const Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2)
{
  shapes_->evaluate(*ele);

  Core::LinAlg::SerialDenseMatrix Mmat(shapes_->ndofs_, shapes_->ndofs_);
  Core::LinAlg::SerialDenseMatrix massPart(shapes_->ndofs_, shapes_->nqpoints_);
  Core::LinAlg::SerialDenseMatrix massPartW(shapes_->ndofs_, shapes_->nqpoints_);

  // reshape elevec2 as matrix
  FOUR_C_ASSERT(
      elevec2.numRows() == 0 || unsigned(elevec2.numRows()) == shapes_->ndofs_ * (nsd_ + 1),
      "Wrong size in project vector 2");

  // get function
  const int* start_func = params.getPtr<int>("funct");

  // internal variables
  if (elevec2.numRows() > 0)
  {
    Core::LinAlg::SerialDenseMatrix localMat(
        Teuchos::View, elevec2.values(), shapes_->ndofs_, shapes_->ndofs_, nsd_ + 1);
    localMat.putScalar(0.0);

    // create mass matrix for interior by looping over quadrature points
    for (unsigned int q = 0; q < shapes_->nqpoints_; ++q)
    {
      const double fac = shapes_->jfac(q);
      std::array<double, 3> xyz{};
      for (unsigned int d = 0; d < nsd_; ++d)
        xyz[d] = shapes_->xyzreal(d, q);  // coordinates of quadrature point in real coordinates
      double phi;
      double gradphi[nsd_];

      FOUR_C_ASSERT(start_func != nullptr, "funct not set for initial value");
      if (Global::Problem::instance()
                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(*start_func)
                  .number_components() != 1 &&
          Global::Problem::instance()
                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(*start_func)
                  .number_components() != nsd_ + 2)
        FOUR_C_THROW(
            "Impossible to initialize the field with the given number of components of the initial "
            "field. Set the number of components to either 1 or nsd_ + 2.\nThe fields are ordered "
            "as:\n- phi\n- gradphi\n- tracephi");

      phi = Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfSpaceTime>(*start_func)
                .evaluate(xyz.data(), 0, 0);
      for (unsigned int i = 0; i < nsd_; ++i)
        gradphi[i] = Global::Problem::instance()
                         ->function_by_id<Core::Utils::FunctionOfSpaceTime>(*start_func)
                         .evaluate(xyz.data(), 0, 1 + i);

      // now fill the components in the one-sided mass matrix and the right hand side
      for (unsigned int i = 0; i < shapes_->ndofs_; ++i)
      {
        massPart(i, q) = shapes_->shfunct(i, q);
        massPartW(i, q) = shapes_->shfunct(i, q) * fac;
        localMat(i, 0) += shapes_->shfunct(i, q) * phi * fac;
        for (unsigned int j = 0; j < nsd_; ++j)
          localMat(i, 1 + j) += shapes_->shfunct(i, q) * gradphi[j] * fac;
      }
    }

    Core::LinAlg::multiply_nt(Mmat, massPart, massPartW);
    {
      using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
      using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
      Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
      inverseMass.setMatrix(Teuchos::rcpFromRef(Mmat));
      inverseMass.setVectors(Teuchos::rcpFromRef(localMat), Teuchos::rcpFromRef(localMat));
      inverseMass.factorWithEquilibration(true);
      int err2 = inverseMass.factor();
      int err = inverseMass.solve();
      if (err != 0 || err2 != 0) FOUR_C_THROW("Inversion of matrix failed with errorcode {}", err);
    }
  }

  // We have to set the initial trace field and gradient field,
  // because the calculation of the residual in the first time step should have a correct trace
  // field and also a correct gradient!

  // trace variable
  int nfdofs = 0;
  for (unsigned int face = 0; face < nfaces_; ++face)
  {
    Core::FE::ShapeValuesFaceParams svfparams(
        ele->faces()[face]->degree(), shapes_->usescompletepoly_, 2 * ele->faces()[face]->degree());
    shapesface_ = Core::FE::ShapeValuesFaceCache<distype>::instance().create(svfparams);
    shapesface_->evaluate_face(*ele, face);

    Core::LinAlg::SerialDenseMatrix mass(shapesface_->nfdofs_, shapesface_->nfdofs_);
    Core::LinAlg::SerialDenseMatrix trVec(shapesface_->nfdofs_, 1);

    // loop over quadrature points
    for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
    {
      const double fac = shapesface_->jfac(q);
      std::array<double, 3> xyz{};
      for (unsigned int d = 0; d < nsd_; ++d) xyz[d] = shapesface_->xyzreal(d, q);

      double trphi;
      trphi = Global::Problem::instance()
                  ->function_by_id<Core::Utils::FunctionOfSpaceTime>(*start_func)
                  .evaluate(xyz.data(), 0, nsd_ + 1);

      // now fill the components in the mass matrix and the right hand side
      for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      {
        // mass matrix
        for (unsigned int j = 0; j < shapesface_->nfdofs_; ++j)
          mass(i, j) += shapesface_->shfunct(i, q) * shapesface_->shfunct(j, q) * fac;
        trVec(i, 0) += shapesface_->shfunct(i, q) * trphi * fac;
      }
    }

    using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
    using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
    Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseMass;
    inverseMass.setMatrix(Teuchos::rcpFromRef(mass));
    inverseMass.setVectors(Teuchos::rcpFromRef(trVec), Teuchos::rcpFromRef(trVec));
    inverseMass.factorWithEquilibration(true);
    int err2 = inverseMass.factor();
    int err = inverseMass.solve();
    if (err != 0 || err2 != 0) FOUR_C_THROW("Inversion of matrix failed with errorcode {}", err);
    for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i) elevec1(nfdofs + i) = trVec(i, 0);

    nfdofs += shapesface_->nfdofs_;
  }

  return 0;

}  // SetInitialField


/*----------------------------------------------------------------------*
 |  evaluate single material  (protected)                hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::prepare_materials(
    Core::Elements::Element* ele,                               //!< the element we are dealing with
    const std::shared_ptr<const Core::Mat::Material> material,  //!< pointer to current material
    const int k,                                                //!< id of current scalar
    std::shared_ptr<std::vector<Core::LinAlg::SerialDenseMatrix>> difftensor  //!< diffusion tensor
)
{
  const std::shared_ptr<const Mat::ScatraMat>& actmat =
      std::dynamic_pointer_cast<const Mat::ScatraMat>(material);

  double diffscalar;

  // get constant diffusivity
  diffscalar = actmat->diffusivity();

  Core::LinAlg::SerialDenseMatrix difftensortmp(nsd_, nsd_);

  for (unsigned int i = 0; i < nsd_; ++i) difftensortmp(i, i) = diffscalar;

  (*difftensor).push_back(difftensortmp);

  return;
}  // ScaTraEleCalcHDG::Materials


/*----------------------------------------------------------------------*
 |  Set material parameter                               hoermann 09/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::LocalSolver::set_material_parameter(
    Discret::Elements::ScaTraHDG* hdgele,         //!< hdg element
    Core::LinAlg::SerialDenseVector& ivecn,       //!< reaction term at time n
    Core::LinAlg::SerialDenseVector& ivecnp,      //!< reaction term at time n+1
    Core::LinAlg::SerialDenseMatrix& ivecnpderiv  //!< reaction term derivaitve
)
{
  // Initialize reaction and diffusion matrices
  hdgele->Ivecn_.size(hdgele->ndofs_);
  hdgele->Ivecnp_.size(hdgele->ndofs_);
  hdgele->Imatnpderiv_.shape(hdgele->ndofs_, hdgele->ndofs_);


  hdgele->Ivecn_ = ivecn;
  hdgele->Ivecnp_ = ivecnp;
  hdgele->Imatnpderiv_ = ivecnpderiv;
}

/*----------------------------------------------------------------------*
 |  Prepare material parameter                           hoermann 11/16 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::LocalSolver::prepare_material_parameter(
    Discret::Elements::ScaTraHDG* hdgele,        //!< hdg element
    Core::LinAlg::SerialDenseMatrix& difftensor  //!< diffusion tensor
)
{
  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  Teuchos::SerialDenseSolver<ordinalType, scalarType> inverseindifftensor;
  inverseindifftensor.setMatrix(Teuchos::rcpFromRef(difftensor));
  int err = inverseindifftensor.invert();
  if (err != 0) FOUR_C_THROW("Inversion of diffusion tensor failed with errorcode {}", err);

  hdgele->invdiff_.push_back(difftensor);
}

/*----------------------------------------------------------------------*
 * Element Init
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::element_init(
    Core::Elements::Element* ele)
{
  Discret::Elements::ScaTraHDG* hdgele =
      dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele));

  hdgele->Amat_.shape(hdgele->ndofs_, hdgele->ndofs_);
  hdgele->Bmat_.shape(hdgele->ndofs_, nsd_ * hdgele->ndofs_);
  hdgele->Cmat_.shape(hdgele->ndofs_, hdgele->onfdofs_);
  hdgele->Dmat_.shape(nsd_ * hdgele->ndofs_, nsd_ * hdgele->ndofs_);
  hdgele->Emat_.shape(hdgele->ndofs_ * nsd_, hdgele->onfdofs_);
  hdgele->Gmat_.shape(hdgele->onfdofs_, hdgele->ndofs_);
  hdgele->EmatT_.shape(hdgele->onfdofs_, nsd_ * hdgele->ndofs_);
  hdgele->Hmat_.shape(hdgele->onfdofs_, hdgele->onfdofs_);
  hdgele->Mmat_.shape(hdgele->ndofs_, hdgele->ndofs_);
  hdgele->Kmat_.shape(hdgele->onfdofs_, hdgele->onfdofs_);
  hdgele->Xmat_.shape(hdgele->ndofs_, hdgele->onfdofs_);
  hdgele->BmatMT_.shape(nsd_ * hdgele->ndofs_, hdgele->ndofs_);
  hdgele->invAMmat_.shape(hdgele->ndofs_, hdgele->ndofs_);
  hdgele->BTAMmat_.shape(hdgele->ndofs_ * nsd_, hdgele->ndofs_);
  hdgele->invCondmat_.shape(hdgele->ndofs_ * nsd_, hdgele->ndofs_ * nsd_);
  hdgele->diff_.shape(nsd_, nsd_);
  hdgele->invdiff_.clear();
  hdgele->Ivecn_.size(hdgele->ndofs_);
  hdgele->Ivecnp_.size(hdgele->ndofs_);
  hdgele->Imatnpderiv_.shape(hdgele->ndofs_, hdgele->ndofs_);

  hdgele->set_mat_init(true);

  return;
}  // ElementInit

/*----------------------------------------------------------------------*
 * ProjectField
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::project_field(
    const Core::Elements::Element* ele, Core::FE::Discretization& discretization,
    Teuchos::ParameterList& params, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::Elements::LocationArray& la)
{
  int nds_var_old = params.get<int>("nds_var_old");
  int nds_intvar_old = params.get<int>("nds_intvar_old");

  Discret::Elements::ScaTraHDG* hdgele =
      dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele));

  // set change of element degree to false
  hdgele->set_padapt_ele(false);

  std::shared_ptr<Core::FE::ShapeValues<distype>> shapes_old =
      std::make_shared<Core::FE::ShapeValues<distype>>(
          hdgele->degree_old(), usescompletepoly_, 2 * hdgele->degree_old());

  FOUR_C_ASSERT(
      elevec2.numRows() == 0 || unsigned(elevec2.numRows()) == shapes_->ndofs_ * (nsd_ + 1),
      "Wrong size in project vector 2");

  // polynomial space to get the value of the shape function at the new points
  Core::FE::PolynomialSpaceParams params_old(distype, shapes_old->degree_, usescompletepoly_);
  std::shared_ptr<Core::FE::PolynomialSpace<probdim>> polySpace_old =
      Core::FE::PolynomialSpaceCache<probdim>::instance().create(params_old);

  Core::LinAlg::SerialDenseVector interiorPhi_old(shapes_old->ndofs_ * (nsd_ + 1));

  // get node based values!
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state =
      params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("phi");

  std::vector<double> tracephi = Core::FE::extract_values(*matrix_state, la[nds_var_old].lm_);

  // get node based values!
  matrix_state = params.get<std::shared_ptr<Core::LinAlg::Vector<double>>>("intphi");
  std::vector<double> intphi = Core::FE::extract_values(*matrix_state, la[nds_intvar_old].lm_);
  if (intphi.size() != shapes_old->ndofs_ * (nsd_ + 1))
    FOUR_C_THROW(
        "node number not matching: {} vs. {}", intphi.size(), shapes_old->ndofs_ * (nsd_ + 1));

  for (unsigned int i = 0; i < shapes_old->ndofs_ * (nsd_ + 1); ++i) interiorPhi_old(i) = intphi[i];

  if (!usescompletepoly_)
  {
    // copy values if degree stays the same instead of projecting
    if (hdgele->degree_old() == hdgele->degree())
      for (unsigned int i = 0; i < shapes_old->ndofs_ * (nsd_ + 1); ++i) elevec2(i) = intphi[i];
    else
    {
      // set change of element degree to true
      hdgele->set_padapt_ele(true);
      Core::LinAlg::SerialDenseMatrix tempMat(
          shapes_->ndofs_ * (nsd_ + 1), shapes_old->ndofs_ * (nsd_ + 1));

      for (unsigned int i = 0; i < shapes_->ndofs_; i++)
      {
        Core::LinAlg::SerialDenseVector tempVec(shapes_old->ndofs_);
        Core::LinAlg::Matrix<nsd_, 1> point(shapes_->nodexyzunit[i]);
        polySpace_old->evaluate(point, tempVec);
        for (unsigned int j = 0; j < nsd_ + 1; j++)
          for (unsigned int k = 0; k < shapes_old->ndofs_; k++)
            tempMat(j * shapes_->ndofs_ + i, j * shapes_old->ndofs_ + k) = tempVec(k);
      }
      Core::LinAlg::multiply(elevec2, tempMat, interiorPhi_old);
    }

    // project trace field
    int nfdofs = 0;
    int nfdofs_old = 0;
    for (unsigned int face = 0; face < nfaces_; face++)
    {
      // shape values of new element degree
      Core::FE::ShapeValuesFaceParams svfparams(
          ele->faces()[face]->degree(), usescompletepoly_, 2 * ele->faces()[face]->degree());
      shapesface_ = Core::FE::ShapeValuesFaceCache<distype>::instance().create(svfparams);
      shapesface_->evaluate_face(*ele, face);


      // shape values of old element degree
      Discret::Elements::ScaTraHDGIntFace* hdgeleface =
          dynamic_cast<Discret::Elements::ScaTraHDGIntFace*>(
              const_cast<Core::Elements::FaceElement*>(ele->faces()[face].get()));
      Core::FE::ShapeValuesFaceParams svfparams_old(
          hdgeleface->degree_old(), usescompletepoly_, 2 * hdgeleface->degree_old());

      std::shared_ptr<Core::FE::ShapeValuesFace<distype>> shapesface_old =
          Core::FE::ShapeValuesFaceCache<distype>::instance().create(svfparams_old);


      Core::FE::PolynomialSpaceParams polyparams(Core::FE::DisTypeToFaceShapeType<distype>::shape,
          hdgeleface->degree_old(), usescompletepoly_);
      std::shared_ptr<Core::FE::PolynomialSpace<nsd_ - 1>> polySpaceFace_old =
          Core::FE::PolynomialSpaceCache<nsd_ - 1>::instance().create(polyparams);

      Core::LinAlg::SerialDenseVector tracePhi_face_old(shapesface_old->nfdofs_);

      for (unsigned int i = 0; i < shapesface_old->nfdofs_; i++)
        tracePhi_face_old(i) = tracephi[nfdofs_old + i];

      if (ele->faces()[face]->degree() == hdgeleface->degree_old())
        for (unsigned int i = 0; i < shapesface_old->nfdofs_; i++)
          elevec1(nfdofs + i) = tracephi[nfdofs_old + i];
      else
      {
        // set change of element degree to true
        hdgele->set_padapt_ele(true);

        Core::LinAlg::SerialDenseMatrix tempMat1(shapesface_->nfdofs_, shapesface_old->nfdofs_);

        Core::LinAlg::SerialDenseVector tempVec2(shapesface_->nfdofs_);

        for (unsigned int i = 0; i < shapesface_->nfdofs_; i++)
        {
          Core::LinAlg::SerialDenseVector tempVec(shapesface_old->nfdofs_);
          Core::LinAlg::Matrix<nsd_ - 1, 1> point(shapesface_->nodexyzunit[i]);

          polySpaceFace_old->evaluate(point, tempVec);
          for (unsigned int k = 0; k < shapesface_old->nfdofs_; k++) tempMat1(i, k) = tempVec(k);
        }

        Core::LinAlg::multiply(tempVec2, tempMat1, tracePhi_face_old);

        for (unsigned int i = 0; i < shapesface_->nfdofs_; i++) elevec1(nfdofs + i) = tempVec2(i);
      }

      nfdofs += shapesface_->nfdofs_;
      nfdofs_old += shapesface_old->nfdofs_;
    }
  }
  else
  {
    if (hdgele->degree_old() != hdgele->degree())
      hdgele->set_padapt_ele(true);  // set change of element degree to true

    unsigned int size_ndofs = std::min(shapes_old->ndofs_, shapes_->ndofs_);
    for (unsigned int i = 0; i < nsd_ + 1; i++)
      for (unsigned int j = 0; j < size_ndofs; j++)
        elevec2(i * shapes_->ndofs_ + j) = interiorPhi_old(i * shapes_old->ndofs_ + j);

    int nfdofs = 0;
    int nfdofs_old = 0;
    for (unsigned int face = 0; face < nfaces_; face++)
    {
      // shape values of new element degree
      Core::FE::ShapeValuesFaceParams svfparams(
          ele->faces()[face]->degree(), usescompletepoly_, 2 * ele->faces()[face]->degree());
      shapesface_ = Core::FE::ShapeValuesFaceCache<distype>::instance().create(svfparams);
      shapesface_->evaluate_face(*ele, face);

      // shape values of old element degree
      Discret::Elements::ScaTraHDGIntFace* hdgeleface =
          dynamic_cast<Discret::Elements::ScaTraHDGIntFace*>(
              const_cast<Core::Elements::FaceElement*>(ele->faces()[face].get()));
      Core::FE::ShapeValuesFaceParams svfparams_old(
          hdgeleface->degree_old(), usescompletepoly_, 2 * hdgeleface->degree_old());

      std::shared_ptr<Core::FE::ShapeValuesFace<distype>> shapesface_old =
          Core::FE::ShapeValuesFaceCache<distype>::instance().create(svfparams_old);


      Core::FE::PolynomialSpaceParams polyparams(Core::FE::DisTypeToFaceShapeType<distype>::shape,
          hdgeleface->degree_old(), usescompletepoly_);
      std::shared_ptr<Core::FE::PolynomialSpace<nsd_ - 1>> polySpaceFace_old =
          Core::FE::PolynomialSpaceCache<nsd_ - 1>::instance().create(polyparams);

      //      Core::LinAlg::SerialDenseVector tracePhi_face_old(shapesface_old->nfdofs_);

      if (ele->faces()[face]->degree() != hdgeleface->degree_old())
        hdgele->set_padapt_ele(true);  // set change of element degree to true

      unsigned int size_nfdofs = std::min(shapesface_->nfdofs_, shapesface_old->nfdofs_);
      for (unsigned int i = 0; i < size_nfdofs; i++) elevec1(nfdofs + i) = tracephi[nfdofs_old + i];

      nfdofs += shapesface_->nfdofs_;
      nfdofs_old += shapesface_old->nfdofs_;
    }
  }

  return 0;

}  // ProjectField


/*----------------------------------------------------------------------*
 * Calc P-Adaptivity
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::calc_p_adaptivity(
    const Core::Elements::Element* ele, Core::FE::Discretization& discretization,
    Teuchos::ParameterList& params)
{
  Discret::Elements::ScaTraHDG* hdgele =
      dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele));

  Core::LinAlg::SerialDenseVector tempinteriorgradphinp(hdgele->ndofs_ * nsd_);
  for (unsigned int i = 0; i < hdgele->ndofs_ * nsd_; ++i)
    tempinteriorgradphinp(i) = interiorPhinp_(hdgele->ndofs_ + i);

  double error = 0;
  int sumindex = 0;
  for (unsigned int nface = 0; nface < nfaces_; ++nface)
  {
    Core::FE::ShapeValuesFaceParams svfparams(ele->faces()[nface]->degree(),
        shapes_->usescompletepoly_, 2 * ele->faces()[nface]->degree());

    shapesface_ = Core::FE::ShapeValuesFaceCache<distype>::instance().create(svfparams);
    shapesface_->evaluate_face(*ele, nface);

    Core::LinAlg::SerialDenseMatrix QMat(shapesface_->nqpoints_, hdgele->ndofs_ * nsd_);
    Core::LinAlg::SerialDenseMatrix QMatW(shapesface_->nqpoints_, hdgele->ndofs_ * nsd_);
    Core::LinAlg::SerialDenseMatrix UMat(
        shapesface_->nqpoints_, hdgele->ndofs_ + shapesface_->nfdofs_);
    Core::LinAlg::SerialDenseMatrix UMatW(
        shapesface_->nqpoints_, hdgele->ndofs_ + shapesface_->nfdofs_);

    Core::LinAlg::SerialDenseVector tempinteriorphinp(hdgele->ndofs_ + shapesface_->nfdofs_);
    for (unsigned int i = 0; i < hdgele->ndofs_; i++) tempinteriorphinp(i) = interiorPhinp_(i);
    for (unsigned int i = 0; i < shapesface_->nfdofs_; i++)
      tempinteriorphinp(hdgele->ndofs_ + i) = tracen_(sumindex + i);

    // loop over quadrature points
    for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
    {
      // loop over shape functions
      for (unsigned int i = 0; i < hdgele->ndofs_; ++i)
      {
        UMat(q, i) = shapesface_->shfunctI(i, q);
        UMatW(q, i) = shapesface_->shfunctI(i, q) * shapesface_->jfac(q);
        for (unsigned int k = 0; k < nsd_; ++k)
        {
          QMat(q, hdgele->ndofs_ * k + i) = shapesface_->shfunctI(i, q) * shapesface_->normal(k);
          QMatW(q, hdgele->ndofs_ * k + i) =
              shapesface_->shfunctI(i, q) * shapesface_->jfac(q) * shapesface_->normal(k);
        }
      }
      // loop over face shape functions
      for (unsigned int i = 0; i < shapesface_->nfdofs_; ++i)
      {
        UMat(q, i + hdgele->ndofs_) = -shapesface_->shfunct(i, q);
        UMatW(q, i + hdgele->ndofs_) = -shapesface_->shfunct(i, q) * shapesface_->jfac(q);
      }
    }
    sumindex += shapesface_->nfdofs_;

    Core::LinAlg::SerialDenseVector tempVec1(shapesface_->nqpoints_);
    Core::LinAlg::SerialDenseVector tempVec2(shapesface_->nqpoints_);
    Core::LinAlg::SerialDenseVector tempVec3(shapesface_->nqpoints_);
    Core::LinAlg::SerialDenseVector tempVec4(shapesface_->nqpoints_);


    Core::LinAlg::multiply(tempVec1, QMatW, tempinteriorgradphinp);
    Core::LinAlg::multiply(tempVec2, QMat, tempinteriorgradphinp);
    Core::LinAlg::multiply(tempVec3, UMatW, tempinteriorphinp);
    Core::LinAlg::multiply(tempVec4, UMat, tempinteriorphinp);

    double errorface = 0;
    double facearea = 0;

    for (unsigned int q = 0; q < shapesface_->nqpoints_; ++q)
    {
      errorface +=
          tempVec1(q) * tempVec2(q) + tempVec3(q) * tempVec4(q) - 2 * tempVec1(q) * tempVec4(q);
      facearea += shapesface_->jfac(q);
    }

    // normalize error with surface area of face
    error += errorface / facearea;
  }

  params.set<double>("error", error);


  return 0;
}  // CalcPAdaptivity

/*----------------------------------------------------------------------*
 * Calc Error
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcHDG<distype, probdim>::calc_error(
    const Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& elevec)
{
  Discret::Elements::ScaTraHDG* hdgele =
      dynamic_cast<Discret::Elements::ScaTraHDG*>(const_cast<Core::Elements::Element*>(ele));

  // For the calculation of the error we use a higher integration rule
  Core::FE::ShapeValues<distype> highshapes(
      ele->degree(), shapes_->usescompletepoly_, (ele->degree() + 2) * 2);
  highshapes.evaluate(*ele);

  double error_phi = 0.0, error_grad_phi = 0.0;
  double exact_phi = 0.0, exact_grad_phi = 0.0;

  // get function
  const int func = params.get<int>("error function number");
  const double time = params.get<double>("time");

  if (Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfSpaceTime>(func)
          .number_components() != 1)
    FOUR_C_THROW(
        "The number of component must be one. The gradient is computed with forward auomatic "
        "differentiation.");

  Core::LinAlg::Matrix<3, 1> xsi;
  double phi(nsd_);
  Core::LinAlg::SerialDenseVector gradPhi(nsd_);

  for (unsigned int q = 0; q < highshapes.nqpoints_; ++q)
  {
    phi = 0;
    gradPhi.putScalar(0.0);
    if (hdgele->invdiff_.size() == 1)
      for (unsigned int i = 0; i < shapes_->ndofs_; ++i)
      {
        phi += highshapes.shfunct(i, q) * interiorPhinp_(i);
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            gradPhi(d) += highshapes.shfunct(i, q) * interiorPhinp_(i + (e + 1) * shapes_->ndofs_) *
                          hdgele->invdiff_[0](d, e);
      }
    else if (hdgele->invdiff_.size() == highshapes.nqpoints_)
      for (unsigned int i = 0; i < shapes_->ndofs_; ++i)
      {
        phi += highshapes.shfunct(i, q) * interiorPhinp_(i);
        for (unsigned int d = 0; d < nsd_; ++d)
          for (unsigned int e = 0; e < nsd_; ++e)
            gradPhi(d) += highshapes.shfunct(i, q) * interiorPhinp_(i + (e + 1) * shapes_->ndofs_) *
                          hdgele->invdiff_[q](d, e);
      }
    else
      FOUR_C_THROW("Diffusion tensor not defined properly. Impossible to compute error.");


    // Analytical function evaluation
    // Evaluate error function and its derivatives in the integration point (real) coordinates
    for (unsigned int idim = 0; idim < nsd_; idim++) xsi(idim) = highshapes.xyzreal(idim, q);
    double funct = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfSpaceTime>(func)
                       .evaluate(xsi.data(), time, 0);
    std::vector<double> deriv = Global::Problem::instance()
                                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(func)
                                    .evaluate_spatial_derivative(xsi.data(), time, 0);

    error_phi += std::pow((funct - phi), 2) * highshapes.jfac(q);
    exact_phi += std::pow(funct, 2) * highshapes.jfac(q);
    for (unsigned int d = 0; d < nsd_; ++d)
    {
      error_grad_phi += std::pow((deriv[d] - gradPhi(d)), 2) * highshapes.jfac(q);
      exact_grad_phi += std::pow(deriv[d], 2) * highshapes.jfac(q);
    }
  }

  elevec[0] = error_phi;
  elevec[1] = exact_phi;
  elevec[2] = error_grad_phi;
  elevec[3] = exact_grad_phi;

  return 0;
}

// template classes
// 1D elements
// template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::line2,1>;
// template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::line2,2>;
// template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::line2,3>;
// template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::line3,1>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::tri3>;
// template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::quad4, 3>;
// template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::hex8, 3>;
// template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::hex27, 3>;
template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::tet4, 3>;
template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::tet10, 3>;
// template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::pyramid5, 3>;
// template class Discret::Elements::ScaTraEleCalcHDG<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
