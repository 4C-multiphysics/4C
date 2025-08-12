// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_nurbs_discretization.hpp"

#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_io_control.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_manager.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                            gammi 05/08|
 *----------------------------------------------------------------------*/
Core::FE::Nurbs::NurbsDiscretization::NurbsDiscretization(
    const std::string name, MPI_Comm comm, const unsigned int n_dim)
    : Core::FE::Discretization::Discretization(name, comm, n_dim), knots_(nullptr)
{
  return;
}


/*----------------------------------------------------------------------*
 |  add a knotvector to the discretization (public)          gammi 05/08|
 *----------------------------------------------------------------------*/
void Core::FE::Nurbs::NurbsDiscretization::set_knot_vector(
    std::shared_ptr<Core::FE::Nurbs::Knotvector> knots)
{
  if (knots == nullptr)
  {
    FOUR_C_THROW(
        "You're trying to set an invalid knotvector in the "
        "Core::FE::Nurbs::NurbsDiscretization "
        "'{}'. The given know vector is a null vector and can't be set as such.",
        (this->name()).c_str());
  }

  knots_ = knots;
  filled_ = false;
}

/*----------------------------------------------------------------------*
 |  get a pointer to knotvector from the discretization         (public)|
 |                                                           gammi 05/08|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::FE::Nurbs::Knotvector> Core::FE::Nurbs::NurbsDiscretization::get_knot_vector()
{
  if (knots_ == nullptr)
  {
    FOUR_C_THROW(
        "You're trying to access the NURBS knot vector in the "
        "Core::FE::Nurbs::NurbsDiscretization "
        "'{}'. The required knot vector is a null vector and can't be accessed as such.",
        (this->name()).c_str());
  }
  return knots_;
}


/*----------------------------------------------------------------------*
 |  get a pointer to knotvector from the discretization         (public)|
 |  (const version, read-only)                               gammi 05/08|
 *----------------------------------------------------------------------*/
std::shared_ptr<const Core::FE::Nurbs::Knotvector>
Core::FE::Nurbs::NurbsDiscretization::get_knot_vector() const
{
  if (knots_ == nullptr)
  {
    FOUR_C_THROW(
        "You're trying to access the NURBS knot vector in the "
        "Core::FE::Nurbs::NurbsDiscretization "
        "'{}'. The required knot vector is a null vector and can't be accessed as such.",
        (this->name()).c_str());
  }
  return knots_;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::DbcNurbs::evaluate(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, double time,
    const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
    Core::FE::Dbc::DbcInfo& info, std::shared_ptr<std::set<int>>* dbcgids) const
{
  // --------------------------- Step 1 ---------------------------------------
  Core::FE::Dbc::evaluate(params, discret, time, systemvectors, info, dbcgids);

  // --------------------------- Step 2 ---------------------------------------
  std::vector<std::string> dbc_cond_names(2, "");
  dbc_cond_names[0] = "Dirichlet";
  dbc_cond_names[1] = "NurbsLSDirichlet";

  std::vector<const Core::Conditions::Condition*> conds;
  std::vector<const Core::Conditions::Condition*> curr_conds;
  for (std::vector<std::string>::const_iterator cit_name = dbc_cond_names.begin();
      cit_name != dbc_cond_names.end(); ++cit_name)
  {
    discret.get_condition(*cit_name, curr_conds);

    conds.reserve(conds.size() + curr_conds.size());
    std::copy(curr_conds.begin(), curr_conds.end(), std::back_inserter(conds));
  }

  Core::FE::Dbc::DbcInfo info2(info.toggle.get_map());
  read_dirichlet_condition(params, discret, conds, time, info2, dbcgids);

  // --------------------------- Step 3 ---------------------------------------
  conds.clear();
  discret.get_condition("NurbsLSDirichlet", conds);

  std::shared_ptr<std::set<int>> dbcgids_nurbs[2] = {nullptr, nullptr};
  dbcgids_nurbs[set_row] = std::make_shared<std::set<int>>();
  dbcgids_nurbs[set_col] = std::make_shared<std::set<int>>();

  // create a new toggle vector with column layout
  const Core::FE::Nurbs::NurbsDiscretization* discret_nurbs =
      dynamic_cast<const Core::FE::Nurbs::NurbsDiscretization*>(&discret);
  if (not discret_nurbs) FOUR_C_THROW("Dynamic cast failed!");

  // build dummy column toggle vector and auxiliary vectors
  Core::FE::Dbc::DbcInfo info_col(*discret_nurbs->dof_col_map());
  read_dirichlet_condition(params, discret, conds, time, info_col, dbcgids_nurbs);

  // --------------------------- Step 4 ---------------------------------------
  do_dirichlet_condition(
      params, discret, conds, time, systemvectors, info_col.toggle, dbcgids_nurbs);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Core::FE::DbcNurbs::do_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond, double time,
    const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
    const Core::LinAlg::Vector<int>& toggle, const std::shared_ptr<std::set<int>>* dbcgids) const
{
  // default call
  if (dbcgids[set_col] == nullptr)
  {
    Core::FE::Dbc::do_dirichlet_condition(
        params, discret, cond, time, systemvectors, toggle, dbcgids);
    return;
  }

  Teuchos::Time timer("", true);

  // get the processor ID from the communicator
  const int myrank = Core::Communication::my_mpi_rank(discret.get_comm());
  if (myrank == 0) std::cout << "calculating least squares Dirichlet condition in ... ";

  const Core::FE::Nurbs::NurbsDiscretization& nurbs_dis =
      static_cast<const Core::FE::Nurbs::NurbsDiscretization&>(discret);

  // create map extractor to always (re)build dbcmapextractor which is needed later
  Core::LinAlg::MapExtractor auxdbcmapextractor;
  {
    // build map of Dirichlet DOFs
    int nummyelements = 0;
    int* myglobalelements = nullptr;
    std::vector<int> dbcgidsv;
    if (dbcgids[set_row]->size() > 0)
    {
      dbcgidsv.reserve(dbcgids[set_row]->size());
      dbcgidsv.assign(dbcgids[set_row]->begin(), dbcgids[set_row]->end());
      nummyelements = dbcgidsv.size();
      myglobalelements = dbcgidsv.data();
    }
    std::shared_ptr<Core::LinAlg::Map> dbcmap =
        std::make_shared<Core::LinAlg::Map>(-1, nummyelements, myglobalelements,
            discret.dof_row_map()->index_base(), discret.dof_row_map()->get_comm());
    // build the map extractor of Dirichlet-conditioned and free DOFs
    auxdbcmapextractor = Core::LinAlg::MapExtractor(*(discret.dof_row_map()), dbcmap);
  }

  // column map of all DOFs subjected to a least squares Dirichlet condition
  std::shared_ptr<Core::LinAlg::Map> dbccolmap = nullptr;
  {
    // build map of Dirichlet DOFs
    int nummyelements = 0;
    int* myglobalelements = nullptr;
    std::vector<int> dbcgidsv;
    if (dbcgids[set_col]->size() > 0)
    {
      dbcgidsv.reserve(dbcgids[set_col]->size());
      dbcgidsv.assign(dbcgids[set_col]->begin(), dbcgids[set_col]->end());
      nummyelements = dbcgidsv.size();
      myglobalelements = dbcgidsv.data();
    }
    dbccolmap = std::make_shared<Core::LinAlg::Map>(-1, nummyelements, myglobalelements,
        nurbs_dis.dof_col_map()->index_base(), discret.dof_row_map()->get_comm());
  }

  // -------------------------------------------------------------------
  // get a vector layout from the discretization to construct matching
  // vectors and matrices
  //                 local <-> global dof numbering
  // -------------------------------------------------------------------
  const std::shared_ptr<const Core::LinAlg::Map> dofrowmap = auxdbcmapextractor.cond_map();

  if (dofrowmap->num_global_elements() == 0) return;  // no dbc gids ->leave

  // read information from condition
  const std::vector<int>* nodeids = cond.get_nodes();
  if (!nodeids) FOUR_C_THROW("Dirichlet condition does not have nodal cloud");

  const auto funct = cond.parameters().get<std::vector<std::optional<int>>>("FUNCT");
  const auto val = cond.parameters().get<std::vector<double>>("VAL");


  // determine highest degree of time derivative
  // and first existent system vector to apply DBC to
  unsigned deg = 0;  // highest degree of requested time derivative
  std::shared_ptr<Core::LinAlg::Vector<double>> systemvectoraux =
      nullptr;  // auxiliary system vector
  if (systemvectors[0] != nullptr)
  {
    deg = 0;
    systemvectoraux = systemvectors[0];
  }
  if (systemvectors[1] != nullptr)
  {
    deg = 1;
    if (systemvectoraux == nullptr) systemvectoraux = systemvectors[1];
  }
  if (systemvectors[2] != nullptr)
  {
    deg = 2;
    if (systemvectoraux == nullptr) systemvectoraux = systemvectors[2];
  }
  FOUR_C_ASSERT(systemvectoraux != nullptr, "At least one vector must be unequal to null");


  // -------------------------------------------------------------------
  // create empty mass matrix
  // -------------------------------------------------------------------
  std::shared_ptr<Core::LinAlg::SparseMatrix> massmatrix =
      std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 108, false, true);

  // -------------------------------------------------------------------
  // create empty right hand side vector
  // -------------------------------------------------------------------
  std::shared_ptr<Core::LinAlg::Vector<double>> rhs = Core::LinAlg::create_vector(*dofrowmap, true);
  std::shared_ptr<Core::LinAlg::Vector<double>> dbcvector =
      Core::LinAlg::create_vector(*dofrowmap, true);

  std::shared_ptr<Core::LinAlg::Vector<double>> rhsd = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> dbcvectord = nullptr;
  if (systemvectors[1] != nullptr)
  {
    rhsd = Core::LinAlg::create_vector(*dofrowmap, true);
    dbcvectord = Core::LinAlg::create_vector(*dofrowmap, true);
  }

  std::shared_ptr<Core::LinAlg::Vector<double>> rhsdd = nullptr;
  std::shared_ptr<Core::LinAlg::Vector<double>> dbcvectordd = nullptr;
  if (systemvectors[2] != nullptr)
  {
    rhsdd = Core::LinAlg::create_vector(*dofrowmap, true);
    dbcvectordd = Core::LinAlg::create_vector(*dofrowmap, true);
  }

  const bool assemblevecd = rhsd != nullptr;
  const bool assemblevecdd = rhsdd != nullptr;

  // -------------------------------------------------------------------
  // call elements to calculate massmatrix and righthandside
  // -------------------------------------------------------------------
  {
    // call elements and assemble
    if (!discret.filled()) FOUR_C_THROW("fill_complete() was not called");
    if (!discret.have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

    // see what we have for input
    bool assemblemat = massmatrix != nullptr;
    bool assemblevec = rhs != nullptr;

    // define element matrices and vectors
    Core::LinAlg::SerialDenseMatrix elemass;
    std::vector<Core::LinAlg::SerialDenseVector> elerhs(deg + 1);

    std::vector<int> lm;
    std::vector<int> lmowner;

    std::vector<int> lm_full;
    std::vector<int> lmowner_full;
    std::vector<int> lmstride_full;

    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = cond.geometry();
    std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator curr;
    for (curr = geom.begin(); curr != geom.end(); ++curr)
    {
      std::shared_ptr<Core::Elements::Element> actele = curr->second;

      static const int probdim = discret.n_dim();
      const Core::FE::CellType distype = actele->shape();
      const int dim = Core::FE::get_dimension(distype);
      const bool isboundary = (dim != probdim);
      const int nen = Core::FE::get_number_of_element_nodes(distype);

      // access elements knot span
      std::vector<Core::LinAlg::SerialDenseVector> eleknots(dim);
      Core::LinAlg::SerialDenseVector weights(nen);

      bool zero_size = false;
      if (isboundary)
      {
        std::shared_ptr<Core::Elements::FaceElement> faceele =
            std::dynamic_pointer_cast<Core::Elements::FaceElement>(actele);
        double normalfac = 0.0;
        std::vector<Core::LinAlg::SerialDenseVector> pknots(probdim);
        zero_size = Core::FE::Nurbs::get_knot_vector_and_weights_for_nurbs_boundary(actele.get(),
            faceele->face_master_number(), faceele->parent_element()->id(), discret, pknots,
            eleknots, weights, normalfac);
      }
      else
        zero_size = Core::FE::Nurbs::get_my_nurbs_knots_and_weights(
            discret, actele.get(), eleknots, weights);

      // nothing to be done for a zero sized element
      if (zero_size)
      {
        continue;
      }

      // get element full location vector, dirichlet flags and ownerships
      lm_full.clear();
      lmowner_full.clear();
      lmstride_full.clear();
      actele->location_vector(nurbs_dis, lm_full, lmowner_full, lmstride_full);

      // we are only interested in DOFs with dirichlet condition, hence we compare the location
      // vector with the
      // drichlet condition map
      lm.clear();
      lmowner.clear();
      for (unsigned j = 0; j < lm_full.size(); ++j)
      {
        int gid = lm_full[j];
        if (dbccolmap->my_gid(gid))
        {
          lm.push_back(gid);
          lmowner.push_back(lmowner_full[j]);
        }
      }

      // get dimension of element matrices and vectors
      // Reshape element matrices and vectors and init to zero
      const int eledim = (int)lm.size();

      if (assemblemat)
      {
        if (elemass.numRows() != eledim or elemass.numCols() != eledim)
          elemass.shape(eledim, eledim);
        else
          elemass.putScalar(0.0);
      }
      if (assemblevec)
      {
        for (unsigned i = 0; i < deg + 1; ++i)
        {
          if (elerhs[i].length() != eledim)
            elerhs[i].size(eledim);
          else
            elerhs[i].putScalar(0.0);
        }
      }

      if (isboundary) switch (distype)
        {
          case Core::FE::CellType::nurbs2:
            fill_matrix_and_rhs_for_ls_dirichlet_boundary<Core::FE::CellType::nurbs2>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          case Core::FE::CellType::nurbs3:
            fill_matrix_and_rhs_for_ls_dirichlet_boundary<Core::FE::CellType::nurbs3>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          case Core::FE::CellType::nurbs4:
            fill_matrix_and_rhs_for_ls_dirichlet_boundary<Core::FE::CellType::nurbs4>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          case Core::FE::CellType::nurbs9:
            fill_matrix_and_rhs_for_ls_dirichlet_boundary<Core::FE::CellType::nurbs9>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          case Core::FE::CellType::nurbs8:
            fill_matrix_and_rhs_for_ls_dirichlet_boundary<Core::FE::CellType::nurbs8>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          case Core::FE::CellType::nurbs27:
            fill_matrix_and_rhs_for_ls_dirichlet_boundary<Core::FE::CellType::nurbs27>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          default:
            FOUR_C_THROW("invalid element shape for least squares dirichlet evaluation: {}",
                Core::FE::cell_type_to_string(distype));
            break;
        }
      else
        switch (distype)
        {
          case Core::FE::CellType::nurbs2:
            fill_matrix_and_rhs_for_ls_dirichlet_domain<Core::FE::CellType::nurbs2>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          case Core::FE::CellType::nurbs3:
            fill_matrix_and_rhs_for_ls_dirichlet_domain<Core::FE::CellType::nurbs3>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          case Core::FE::CellType::nurbs4:
            fill_matrix_and_rhs_for_ls_dirichlet_domain<Core::FE::CellType::nurbs4>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          case Core::FE::CellType::nurbs9:
            fill_matrix_and_rhs_for_ls_dirichlet_domain<Core::FE::CellType::nurbs9>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          case Core::FE::CellType::nurbs8:
            fill_matrix_and_rhs_for_ls_dirichlet_domain<Core::FE::CellType::nurbs8>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          case Core::FE::CellType::nurbs27:
            fill_matrix_and_rhs_for_ls_dirichlet_domain<Core::FE::CellType::nurbs27>(*actele,
                &eleknots, lm, funct, val, deg, time, elemass, elerhs,
                *params.get<const Core::Utils::FunctionManager*>("function_manager"));
            break;
          default:
            FOUR_C_THROW("invalid element shape for least squares dirichlet evaluation: {}",
                Core::FE::cell_type_to_string(distype));
            break;
        }

      int eid = actele->id();
      if (assemblemat) massmatrix->assemble(eid, elemass, lm, lmowner);
      if (assemblevec) Core::LinAlg::assemble(*rhs, elerhs[0], lm, lmowner);
      if (assemblevecd) Core::LinAlg::assemble(*rhsd, elerhs[1], lm, lmowner);
      if (assemblevecdd) Core::LinAlg::assemble(*rhsdd, elerhs[2], lm, lmowner);
    }
  }
  // -------------------------------------------------------------------
  // finalize the system matrix
  // -------------------------------------------------------------------
  massmatrix->complete();

  // -------------------------------------------------------------------
  // solve system
  // -------------------------------------------------------------------
  // Get the solver parameters for the least squares problem
  const Teuchos::ParameterList& ls_dbc_solver_params = params.sublist("ls_dbc_solver_params");

  Core::LinAlg::Solver solver(
      ls_dbc_solver_params, discret.get_comm(), nullptr, Core::IO::Verbositylevel::standard);
  // FixMe actually the const qualifier could stay, if someone adds to each single
  // related ComputeNullSpace routine a "const"....
  const_cast<Core::FE::Discretization&>(discret).compute_null_space_if_necessary(solver.params());

  // solve for control point values
  // always refactor and reset the matrix before a single new solver call
  Core::LinAlg::SolverParams solver_params;
  solver_params.refactor = true;
  solver_params.reset = true;
  solver.solve(massmatrix, dbcvector, rhs, solver_params);

  // solve for first derivatives in time
  if (assemblevecd) solver.solve(massmatrix, dbcvectord, rhsd, solver_params);

  // solve for second derivatives in time
  if (assemblevecdd) solver.solve(massmatrix, dbcvectordd, rhsdd, solver_params);

  // perform resets for solver and matrix
  solver.reset();
  massmatrix->reset();

  // insert nodal values to sysvec
  auxdbcmapextractor.insert_cond_vector(*dbcvector, *systemvectors[0]);
  if (assemblevecd) auxdbcmapextractor.insert_cond_vector(*dbcvectord, *systemvectors[1]);
  if (assemblevecdd) auxdbcmapextractor.insert_cond_vector(*dbcvectordd, *systemvectors[2]);

  if (myrank == 0) std::cout << timer.totalElapsedTime(true) << " seconds \n\n";

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate Dirichlet conditions (public)                   vuong 08/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Core::FE::DbcNurbs::fill_matrix_and_rhs_for_ls_dirichlet_boundary(
    Core::Elements::Element& actele, const std::vector<Core::LinAlg::SerialDenseVector>* knots,
    const std::vector<int>& lm, const std::vector<std::optional<int>>& funct,
    const std::vector<double>& val, const unsigned deg, const double time,
    Core::LinAlg::SerialDenseMatrix& elemass, std::vector<Core::LinAlg::SerialDenseVector>& elerhs,
    const Core::Utils::FunctionManager& function_manager) const
{
  if (deg + 1 != elerhs.size())
    FOUR_C_THROW("given degree of time derivative does not match number or rhs vectors!");

  static const int dim = Core::FE::dim<distype>;

  const int ndbcdofs = (int)lm.size();

  // set element data
  static const int nen = Core::FE::num_nodes(distype);

  // dofblocks (number of DOFs with Dirichlet condition per node)
  const int dofblock = ndbcdofs / nen;

  // get node coordinates of element
  Core::LinAlg::Matrix<dim + 1, nen> xyze;
  Core::Nodes::Node** nodes = actele.nodes();

  for (int inode = 0; inode < nen; inode++)
  {
    const auto& x = nodes[inode]->x();
    for (int idim = 0; idim < dim + 1; ++idim)
    {
      xyze(idim, inode) = x[idim];
    }
  }

  // acquire weights from nodes
  Core::LinAlg::SerialDenseVector weights(nen);

  for (int inode = 0; inode < nen; ++inode)
  {
    Core::FE::Nurbs::ControlPoint* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes[inode]);

    weights(inode) = cp->w();
  }

  // shape functions
  Core::LinAlg::Matrix<nen, 1> shpfunct;
  // coordinates of integration points in parameter space
  Core::LinAlg::Matrix<dim, 1> xsi;
  // first derivative of shape functions
  Core::LinAlg::Matrix<dim, nen> deriv;
  // coordinates of integration point in physical space
  Core::LinAlg::SerialDenseVector position(
      3);  // always three-dimensional coordinates for function evaluation!
  // auxiliary date container for dirichlet evaluation
  std::vector<Core::LinAlg::SerialDenseVector> value(deg + 1, dofblock);
  // unit normal on boundary element
  Core::LinAlg::Matrix<dim + 1, 1> unitnormal;

  // gaussian points
  const Core::FE::IntPointsAndWeights<dim> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // integration factor
    double fac = 0.0;
    double drs = 0.0;

    Core::FE::eval_shape_func_at_bou_int_point<distype>(
        shpfunct, deriv, fac, unitnormal, drs, xsi, xyze, intpoints, iquad, knots, &weights, true);

    // get real physical coordinates of integration point
    /*
    //              +-----
    //               \
    //    pos (x) =   +      N (x) * x
    //               /        j       j
    //              +-----
    //              node j
    */
    for (int rr = 0; rr < dim + 1; ++rr)
    {
      position(rr) = shpfunct(0) * xyze(rr, 0);
      for (int mm = 1; mm < nen; ++mm)
      {
        position(rr) += shpfunct(mm) * xyze(rr, mm);
      }
    }
    // if dim < 3, ensure we define a valid z-coordinate!
    for (int rr = dim + 1; rr < 3; ++rr) position(rr) = 0.0;

    for (int rr = 0; rr < dofblock; ++rr)
    {
      // factor given by FUNCTS
      std::vector<double> functimederivfac(deg + 1, 1.0);
      for (unsigned i = 1; i < (deg + 1); ++i) functimederivfac[i] = 0.0;

      if (funct[rr].has_value() && funct[rr].value() > 0)
      {
        // important: position has to have always three components!!
        functimederivfac =
            function_manager.function_by_id<Core::Utils::FunctionOfSpaceTime>(funct[rr].value())
                .evaluate_time_derivative(position.values(), time, deg, rr);
      }

      // apply factors to Dirichlet value
      for (unsigned i = 0; i < deg + 1; ++i)
      {
        value[i](rr) = val[rr] * functimederivfac[i];
      }
    }

    for (int vi = 0; vi < nen; ++vi)  // loop rows  (test functions)
    {
      const int fvi = dofblock * vi;

      for (int ui = 0; ui < nen; ++ui)  // loop columns  (test functions)
      {
        const int fui = dofblock * ui;

        const double diag = fac * shpfunct(ui) * shpfunct(vi);

        for (int rr = 0; rr < dofblock; ++rr)
        {
          elemass(fvi + rr, fui + rr) += diag;
        }
      }
      for (int rr = 0; rr < dofblock; ++rr)
      {
        for (unsigned i = 0; i < deg + 1; ++i)
          elerhs[i](fvi + rr) += fac * shpfunct(vi) * value[i](rr);
      }
    }
  }  // end gaussloop

  return;
}

/*----------------------------------------------------------------------*
 |  evaluate Dirichlet conditions (public)                   vuong 08/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Core::FE::DbcNurbs::fill_matrix_and_rhs_for_ls_dirichlet_domain(
    Core::Elements::Element& actele, const std::vector<Core::LinAlg::SerialDenseVector>* knots,
    const std::vector<int>& lm, const std::vector<std::optional<int>>& funct,
    const std::vector<double>& val, const unsigned deg, const double time,
    Core::LinAlg::SerialDenseMatrix& elemass, std::vector<Core::LinAlg::SerialDenseVector>& elerhs,
    const Core::Utils::FunctionManager& function_manager) const
{
  if (deg + 1 != elerhs.size())
    FOUR_C_THROW("given degree of time derivative does not match number or rhs vectors!");

  static const int dim = Core::FE::dim<distype>;
  const int ndbcdofs = (int)lm.size();

  // set element data
  static const int nen = Core::FE::num_nodes(distype);

  // dofblocks (number of DOFs with Dirichlet condition per node)
  const int dofblock = ndbcdofs / nen;

  // get node coordinates of element
  Core::LinAlg::Matrix<dim, nen> xyze;
  Core::Nodes::Node** nodes = actele.nodes();

  for (int inode = 0; inode < nen; inode++)
  {
    const auto& x = nodes[inode]->x();
    for (int idim = 0; idim < dim; ++idim)
    {
      xyze(idim, inode) = x[idim];
    }
  }

  // acquire weights from nodes
  Core::LinAlg::SerialDenseVector weights(nen);

  for (int inode = 0; inode < nen; ++inode)
  {
    Core::FE::Nurbs::ControlPoint* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes[inode]);

    weights(inode) = cp->w();
  }

  // shape functions
  Core::LinAlg::Matrix<nen, 1> shpfunct;
  // coordinates of integration points in parameter space
  Core::LinAlg::Matrix<dim, 1> xsi;
  // transposed jacobian "dx/ds"
  Core::LinAlg::Matrix<dim, dim> xjm;
  // first derivative of shape functions
  Core::LinAlg::Matrix<dim, nen> deriv;
  // coordinates of integration point in physical space
  Core::LinAlg::SerialDenseVector position(
      3);  // always three-dimensional coordinates for function evaluation!
  // auxiliary date container for dirichlet evaluation
  std::vector<Core::LinAlg::SerialDenseVector> value(deg + 1, dofblock);

  // gaussian points
  const Core::FE::IntPointsAndWeights<dim> intpoints(
      Discret::Elements::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    for (int idim = 0; idim < dim; idim++)
    {
      xsi(idim) = intpoints.ip().point(iquad)[idim];
    }

    // evaluate shape function and derivatevs at integration point
    Core::FE::Nurbs::nurbs_get_funct_deriv(shpfunct, deriv, xsi, *knots, weights, distype);

    xjm.multiply_nt(deriv, xyze);
    double det = xjm.determinant();

    if (det < 1E-16)
      FOUR_C_THROW(
          "GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", actele.id(), det);

    // compute integration factor
    double fac = intpoints.ip().qwgt[iquad] * det;

    // get real physical coordinates of integration point
    /*
    //              +-----
    //               \
    //    pos (x) =   +      N (x) * x
    //               /        j       j
    //              +-----
    //              node j
    */
    for (int rr = 0; rr < dim; ++rr)
    {
      position(rr) = shpfunct(0) * xyze(rr, 0);
      for (int mm = 1; mm < nen; ++mm)
      {
        position(rr) += shpfunct(mm) * xyze(rr, mm);
      }
    }
    // if dim < 3, ensure we define a valid z-coordinate!
    for (int rr = dim; rr < 3; ++rr) position(rr) = 0.0;

    for (int rr = 0; rr < dofblock; ++rr)
    {
      // factor given by FUNCTS
      std::vector<double> functimederivfac(deg + 1, 1.0);
      double functfac = 1.0;
      if (funct[rr].has_value() && funct[rr].value() > 0)
      {
        // important: position has to have always three components!!
        functimederivfac =
            function_manager.function_by_id<Core::Utils::FunctionOfSpaceTime>(funct[rr].value())
                .evaluate_time_derivative(position.values(), time, deg, rr);

        functfac =
            function_manager.function_by_id<Core::Utils::FunctionOfSpaceTime>(funct[rr].value())
                .evaluate(position.values(), time, rr);
      }

      // apply factors to Dirichlet value
      for (unsigned i = 0; i < deg + 1; ++i)
      {
        value[i](rr) = val[rr] * functfac * functimederivfac[i];
      }
    }

    for (int vi = 0; vi < nen; ++vi)  // loop rows  (test functions)
    {
      const int fvi = dofblock * vi;

      for (int ui = 0; ui < nen; ++ui)  // loop columns  (test functions)
      {
        const int fui = dofblock * ui;

        const double diag = fac * shpfunct(ui) * shpfunct(vi);

        for (int rr = 0; rr < dofblock; ++rr)
        {
          elemass(fvi + rr, fui + rr) += diag;
        }
      }
      for (int rr = 0; rr < dofblock; ++rr)
      {
        for (unsigned i = 0; i < deg + 1; ++i)
          elerhs[i](fvi + rr) += fac * shpfunct(vi) * value[i](rr);
      }
    }
  }  // end gaussloop

  return;
}

FOUR_C_NAMESPACE_CLOSE
