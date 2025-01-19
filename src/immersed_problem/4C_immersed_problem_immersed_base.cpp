// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_immersed_problem_immersed_base.hpp"

#include "4C_adapter_fld_wrapper.hpp"
#include "4C_adapter_str_fsiwrapper_immersed.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_utils_function_of_time.hpp"

FOUR_C_NAMESPACE_OPEN


Immersed::ImmersedBase::ImmersedBase() : issetup_(false), isinit_(false)
{
  // empty
  return;
}  // ImmersedBase



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::build_condition_dof_map(
    const std::shared_ptr<const Core::FE::Discretization>& dis, const std::string condname,
    const Epetra_Map& cond_dofmap_orig, const int numdof, std::shared_ptr<Epetra_Map>& cond_dofmap)
{
  // declare dof vector
  std::vector<int> mydirichdofs(0);

  // get condition and conditioned nodes
  Core::Conditions::Condition* condition = dis->get_condition(condname);
  const std::vector<int>* cond_nodes = condition->get_nodes();
  int cond_nodes_size = cond_nodes->size();

  if (cond_nodes_size == 0)
    FOUR_C_THROW("No nodes in nodal cloud of condition %s", condname.c_str());

  // loop over all conditioned nodes
  for (int node = 0; node < cond_nodes_size; node++)
  {
    // get node id
    int nodeid = cond_nodes->at(node);
    // get node pointer
    Core::Nodes::Node* node_ptr = dis->g_node(nodeid);
    if (node_ptr == nullptr) FOUR_C_THROW("Could not get node with id %d", nodeid);

    if (dis->node_row_map()->LID(nodeid) != -1)
    {
      // get dofs
      std::vector<int> dofs = dis->dof(0, node_ptr);

      for (int dim = 0; dim < numdof; ++dim)
      {
        // if not already in original dirich map
        if (cond_dofmap_orig.LID(dofs[dim]) == -1) mydirichdofs.push_back(dofs[dim]);
      }
    }  // if my node
  }  // loop over all conditioned nodes

  int nummydirichvals = mydirichdofs.size();
  cond_dofmap = std::make_shared<Epetra_Map>(-1, nummydirichvals, mydirichdofs.data(), 0,
      Core::Communication::as_epetra_comm(dis->get_comm()));

  return;
}  // build_condition_dof_row_map


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::do_dirichlet_cond(Core::LinAlg::Vector<double>& statevector,
    const Core::LinAlg::Vector<double>& dirichvals, const Epetra_Map& dbcmap_new)
{
  int mynumvals = dbcmap_new.NumMyElements();
  double* myvals = dirichvals.Values();

  for (int i = 0; i < mynumvals; ++i)
  {
    int gid = dbcmap_new.GID(i);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (mynumvals == 0) FOUR_C_THROW("dbcmap empty!");
    int err = -2;
    int lid = dirichvals.Map().LID(gid);
    err = statevector.ReplaceGlobalValue(gid, 0, myvals[lid]);
    if (err == -1)
      FOUR_C_THROW("VectorIndex >= NumVectors()");
    else if (err == 1)
      FOUR_C_THROW("GlobalRow not associated with calling processor");
    else if (err != -1 and err != 1 and err != 0)
      FOUR_C_THROW("Trouble using ReplaceGlobalValue on fluid state vector. ErrorCode = %d", err);
#else
    int lid = dirichvals.Map().LID(gid);
    statevector.ReplaceGlobalValue(gid, 0, myvals[lid]);
#endif
  }
  return;
}  // DoDirichletCond


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::do_dirichlet_cond(Core::LinAlg::Vector<double>& statevector,
    const Core::LinAlg::Vector<double>& dirichvals, const Epetra_Map& dbcmap_new,
    const Epetra_Map& dbcmap_orig)
{
  int mynumvals = dbcmap_new.NumMyElements();
  double* myvals = dirichvals.Values();

  for (int i = 0; i < mynumvals; ++i)
  {
    int gid = dbcmap_new.GID(i);

#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (mynumvals == 0) FOUR_C_THROW("dbcmap empty!");
    int err = -2;
    int lid = dirichvals.Map().LID(gid);
    err = statevector.ReplaceGlobalValue(gid, 0, myvals[lid]);
    if (err == -1)
      FOUR_C_THROW("VectorIndex >= NumVectors()");
    else if (err == 1)
      FOUR_C_THROW("GlobalRow not associated with calling processor");
    else if (err != -1 and err != 1 and err != 0)
      FOUR_C_THROW("Trouble using ReplaceGlobalValue on fluid state vector. ErrorCode = %d", err);
#else
    int lid = dirichvals.Map().LID(gid);

    // we do not want to overwrite original values
    if (dbcmap_orig.LID(gid) == -1) statevector.ReplaceGlobalValue(gid, 0, myvals[lid]);
#endif
  }
  return;
}  // DoDirichletCond


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::apply_dirichlet(
    const std::shared_ptr<Adapter::StructureWrapper>& field_wrapper,
    const std::shared_ptr<Core::FE::Discretization>& dis, const std::string condname,
    std::shared_ptr<Epetra_Map>& cond_dofrowmap, const int numdof,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& dirichvals)
{
  const std::shared_ptr<const Epetra_Map> condmap_orig =
      field_wrapper->get_dbc_map_extractor()->cond_map();

  // build map of dofs subjected to Dirichlet condition
  build_condition_dof_map(dis, condname, *condmap_orig, numdof, cond_dofrowmap);

  // add adhesion dofs to dbc map
  field_wrapper->add_dirich_dofs(cond_dofrowmap);

  // write Dirichlet values to systemvector
  do_dirichlet_cond(*field_wrapper->write_access_dispnp(), *dirichvals,
      *field_wrapper->get_dbc_map_extractor()->cond_map());

  return;
}  // ApplyDirichlet


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::apply_dirichlet_to_fluid(
    const std::shared_ptr<Adapter::FluidWrapper>& field_wrapper,
    const std::shared_ptr<Core::FE::Discretization>& dis, const std::string condname,
    std::shared_ptr<Epetra_Map>& cond_dofrowmap, const int numdof,
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& dirichvals)
{
  // save the original condition map
  const std::shared_ptr<const Epetra_Map> condmap_orig =
      field_wrapper->get_dbc_map_extractor()->cond_map();

  // build map of dofs subjected to Dirichlet condition
  build_condition_dof_map(dis, condname, *condmap_orig, numdof, cond_dofrowmap);

  // add adhesion dofs to dbc map
  field_wrapper->add_dirich_cond(cond_dofrowmap);

  // write Dirichlet values to systemvector
  do_dirichlet_cond(*field_wrapper->write_access_velnp(), *dirichvals,
      *field_wrapper->get_dbc_map_extractor()->cond_map());

  return;
}  // ApplyDirichlet

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::remove_dirichlet(const std::shared_ptr<const Epetra_Map>& cond_dofmap,
    const std::shared_ptr<Adapter::StructureWrapper>& field_wrapper)
{
  field_wrapper->remove_dirich_dofs(cond_dofmap);
  return;
}  // RemoveDirichlet


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::remove_dirichlet_from_fluid(
    const std::shared_ptr<const Epetra_Map>& cond_dofmap,
    const std::shared_ptr<Adapter::FluidWrapper>& field_wrapper)
{
  field_wrapper->remove_dirich_cond(cond_dofmap);
  return;
}  // RemoveDirichlet


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::evaluate_immersed(Teuchos::ParameterList& params,
    Core::FE::Discretization& dis, Core::FE::AssembleStrategy* strategy,
    std::map<int, std::set<int>>* elementstoeval,
    std::shared_ptr<Core::Geo::SearchTree> structsearchtree,
    std::map<int, Core::LinAlg::Matrix<3, 1>>* currpositions_struct, const FLD::Action action,
    bool evaluateonlyboundary)
{
  // pointer to element
  Core::Elements::Element* ele;

  for (std::map<int, std::set<int>>::const_iterator closele = elementstoeval->begin();
      closele != elementstoeval->end(); closele++)
  {
    for (std::set<int>::const_iterator eleIter = (closele->second).begin();
        eleIter != (closele->second).end(); eleIter++)
    {
      ele = dis.g_element(*eleIter);

      Discret::Elements::FluidImmersedBase* immersedelebase =
          dynamic_cast<Discret::Elements::FluidImmersedBase*>(ele);
      if (immersedelebase == nullptr)
        FOUR_C_THROW(
            "dynamic cast from Core::Elements::Element* to Discret::Elements::FluidImmersedBase* "
            "failed");

      // evaluate this element and fill vector with immersed dirichlets
      int row = strategy->first_dof_set();
      int col = strategy->second_dof_set();

      params.set<FLD::Action>("action", action);
      params.set<std::shared_ptr<Core::Geo::SearchTree>>("structsearchtree_rcp", structsearchtree);
      params.set<std::map<int, Core::LinAlg::Matrix<3, 1>>*>(
          "currpositions_struct", currpositions_struct);
      params.set<Inpar::FLUID::PhysicalType>("Physical Type", Inpar::FLUID::poro_p1);

      Core::Elements::LocationArray la(1);
      immersedelebase->location_vector(dis, la, false);
      strategy->clear_element_storage(la[row].size(), la[col].size());

      if (!evaluateonlyboundary)
        immersedelebase->evaluate(params, dis, la[0].lm_, strategy->elematrix1(),
            strategy->elematrix2(), strategy->elevector1(), strategy->elevector2(),
            strategy->elevector3());
      else
      {
        if (immersedelebase->is_boundary_immersed())
          immersedelebase->evaluate(params, dis, la[0].lm_, strategy->elematrix1(),
              strategy->elematrix2(), strategy->elevector1(), strategy->elevector2(),
              strategy->elevector3());
      }

      strategy->assemble_vector1(la[row].lm_, la[row].lmowner_);
    }
  }
  return;
}  // EvaluateImmersed

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::evaluate_immersed_no_assembly(Teuchos::ParameterList& params,
    Core::FE::Discretization& dis, std::map<int, std::set<int>>* elementstoeval,
    std::shared_ptr<Core::Geo::SearchTree> structsearchtree,
    std::map<int, Core::LinAlg::Matrix<3, 1>>* currpositions_struct, const FLD::Action action)
{
  // pointer to element
  Core::Elements::Element* ele;

  for (std::map<int, std::set<int>>::const_iterator closele = elementstoeval->begin();
      closele != elementstoeval->end(); closele++)
  {
    for (std::set<int>::const_iterator eleIter = (closele->second).begin();
        eleIter != (closele->second).end(); eleIter++)
    {
      ele = dis.g_element(*eleIter);

      Discret::Elements::FluidImmersedBase* immersedelebase =
          dynamic_cast<Discret::Elements::FluidImmersedBase*>(ele);
      if (immersedelebase == nullptr)
        FOUR_C_THROW(
            "dynamic cast from Core::Elements::Element* to Discret::Elements::FluidImmersedBase* "
            "failed");

      // provide important objects to ParameterList
      params.set<FLD::Action>("action", action);
      params.set<std::shared_ptr<Core::Geo::SearchTree>>("structsearchtree_rcp", structsearchtree);
      params.set<std::map<int, Core::LinAlg::Matrix<3, 1>>*>(
          "currpositions_struct", currpositions_struct);
      params.set<Inpar::FLUID::PhysicalType>("Physical Type", Inpar::FLUID::poro_p1);
      if (dis.name() == "fluid")
        params.set<std::string>("immerseddisname", "structure");
      else if (dis.name() == "porofluid")
        params.set<std::string>("immerseddisname", "cell");
      else
        FOUR_C_THROW("no corresponding immerseddisname set for this type of backgrounddis!");

      // evaluate the element
      Core::LinAlg::SerialDenseMatrix dummymat;
      Core::LinAlg::SerialDenseVector dummyvec;

      Core::Elements::LocationArray la(1);
      immersedelebase->location_vector(dis, la, false);

      immersedelebase->evaluate(
          params, dis, la[0].lm_, dummymat, dummymat, dummyvec, dummyvec, dummyvec);
    }
  }
  return;
}  // evaluate_immersed_no_assembly

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::evaluate_scatra_with_internal_communication(
    Core::FE::Discretization& dis, const Core::FE::Discretization& idis,
    Core::FE::AssembleStrategy* strategy, std::map<int, std::set<int>>* elementstoeval,
    std::shared_ptr<Core::Geo::SearchTree> structsearchtree,
    std::map<int, Core::LinAlg::Matrix<3, 1>>* currpositions_struct, Teuchos::ParameterList& params,
    bool evaluateonlyboundary)
{
  // pointer to element
  Core::Elements::Element* ele;
  Core::Elements::Element* iele;

  for (std::map<int, std::set<int>>::const_iterator closele = elementstoeval->begin();
      closele != elementstoeval->end(); closele++)
  {
    for (std::set<int>::const_iterator eleIter = (closele->second).begin();
        eleIter != (closele->second).end(); eleIter++)
    {
      ele = dis.g_element(*eleIter);
      iele = idis.g_element(*eleIter);

      Discret::Elements::FluidImmersedBase* immersedelebase =
          dynamic_cast<Discret::Elements::FluidImmersedBase*>(iele);
      if (immersedelebase == nullptr)
        FOUR_C_THROW(
            "dynamic cast from Core::Elements::Element* to Discret::Elements::FluidImmersedBase* "
            "failed");

      // evaluate this element and fill vector with immersed dirichlets
      int row = strategy->first_dof_set();
      int col = strategy->second_dof_set();

      params.set<std::shared_ptr<Core::Geo::SearchTree>>("structsearchtree_rcp", structsearchtree);
      params.set<std::map<int, Core::LinAlg::Matrix<3, 1>>*>(
          "currpositions_struct", currpositions_struct);
      params.set<Inpar::FLUID::PhysicalType>("Physical Type", Inpar::FLUID::poro_p1);

      Core::Elements::LocationArray la(dis.num_dof_sets());
      ele->location_vector(dis, la, false);
      strategy->clear_element_storage(la[row].size(), la[col].size());

      if (!evaluateonlyboundary)
        ele->evaluate(params, dis, la, strategy->elematrix1(), strategy->elematrix2(),
            strategy->elevector1(), strategy->elevector2(), strategy->elevector3());
      else
      {
        if (immersedelebase->is_boundary_immersed())
          ele->evaluate(params, dis, la, strategy->elematrix1(), strategy->elematrix2(),
              strategy->elevector1(), strategy->elevector2(), strategy->elevector3());
      }

      strategy->assemble_vector1(la[row].lm_, la[row].lmowner_);
    }
  }
}  // EvaluateWithInternalCommunication

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/// Reduces to standard evaluate_condition on one proc.
/// Evaluate a specific condition using assemble strategy allowing communication at element level
/// until every conditioned element is evaluated. Needed especially during interpolation from an
/// other discretization to the conditioned elements (e.g. in immersed method).
/// The integration point of a conditioned element requesting a quantity may be owned by another
/// proc as the interpolating element providing this quantity.  rauch 05/14
void Immersed::ImmersedBase::evaluate_interpolation_condition(Core::FE::Discretization& evaldis,
    Teuchos::ParameterList& params, Core::FE::AssembleStrategy& strategy,
    const std::string& condstring)
{
#ifdef FOUR_C_ENABLE_ASSERTIONS
  if (!(evaldis.filled())) FOUR_C_THROW("fill_complete() was not called");
  if (!(evaldis.have_dofs())) FOUR_C_THROW("assign_degrees_of_freedom() was not called");
#endif

  int row = strategy.first_dof_set();
  int col = strategy.second_dof_set();

  // get the current time
  const double time = params.get("total time", -1.0);

  params.set<int>("dummy_call", 0);

  Core::Elements::LocationArray la(evaldis.num_dof_sets());

  std::multimap<std::string, std::shared_ptr<Core::Conditions::Condition>>::iterator fool;

  //----------------------------------------------------------------------
  // loop through conditions and evaluate them if they match the criterion
  //----------------------------------------------------------------------
  for (fool = evaldis.get_all_conditions().begin(); fool != evaldis.get_all_conditions().end();
      ++fool)
  {
    if (fool->first == condstring)
    {
      Core::Conditions::Condition& cond = *(fool->second);

      std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = cond.geometry();
      if (geom.empty())
        FOUR_C_THROW("evaluation of condition with empty geometry on proc %d",
            Core::Communication::my_mpi_rank(evaldis.get_comm()));

      std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator curr;

      // Evaluate Loadcurve if defined. Put current load factor in parameterlist
      const auto curvenum = cond.parameters().get<Core::IO::Noneable<int>>("curve");
      double curvefac = 1.0;
      if (curvenum.has_value() && curvenum.value() > 0 && time >= 0)
      {
        curvefac = Global::Problem::instance()
                       ->function_by_id<Core::Utils::FunctionOfTime>(curvenum.value() - 1)
                       .evaluate(time);
      }

      constexpr unsigned character_length = 30;
      char factorname[character_length];
      snprintf(factorname, character_length, "LoadCurveFactor");
      params.set(factorname, curvefac);
      params.set<std::shared_ptr<Core::Conditions::Condition>>("condition", fool->second);

      int mygeometrysize = -1234;
      if (geom.empty())
        mygeometrysize = 0;
      else
        mygeometrysize = geom.size();
      int maxgeometrysize = -1234;
      Core::Communication::max_all(&mygeometrysize, &maxgeometrysize, 1, evaldis.get_comm());
      curr = geom.begin();

#ifdef FOUR_C_ENABLE_ASSERTIONS
      std::cout << "PROC " << Core::Communication::my_mpi_rank(evaldis.get_comm())
                << ": mygeometrysize = " << mygeometrysize
                << " maxgeometrysize = " << maxgeometrysize << std::endl;
#endif


      // enter loop on every proc until the last proc evaluated his last geometry element
      // because there is communication happening inside
      for (int i = 0; i < maxgeometrysize; ++i)
      {
        if (i >= mygeometrysize) params.set<int>("dummy_call", 1);

        // get element location vector and ownerships
        // the LocationVector method will return the the location vector
        // of the dofs this condition is meant to assemble into.
        // These dofs do not need to be the same as the dofs of the element
        // (this is the standard case, though). Special boundary conditions,
        // like weak dirichlet conditions, assemble into the dofs of the parent element.
        curr->second->location_vector(evaldis, la, false, condstring, params);

        // get dimension of element matrices and vectors
        // Reshape element matrices and vectors and init to zero

        strategy.clear_element_storage(la[row].size(), la[col].size());

        // call the element specific evaluate method
        int err = curr->second->evaluate(params, evaldis, la[0].lm_, strategy.elematrix1(),
            strategy.elematrix2(), strategy.elevector1(), strategy.elevector2(),
            strategy.elevector3());
        if (err) FOUR_C_THROW("error while evaluating elements");

        // assemble every element contribution only once
        // do not assemble after dummy call for internal communication
        if (i < mygeometrysize)
        {
          // assembly
          int eid = curr->second->id();
          strategy.assemble_matrix1(
              eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
          strategy.assemble_matrix2(
              eid, la[row].lm_, la[col].lm_, la[row].lmowner_, la[col].stride_);
          strategy.assemble_vector1(la[row].lm_, la[row].lmowner_);
          strategy.assemble_vector2(la[row].lm_, la[row].lmowner_);
          strategy.assemble_vector3(la[row].lm_, la[row].lmowner_);
        }

        // go to next element
        if (i < (mygeometrysize - 1)) ++curr;

      }  // for 0 to max. geometrysize over all procs
    }  // if condstring found
  }  // for (fool=condition_.begin(); fool!=condition_.end(); ++fool)
}  // evaluate_interpolation_condition

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::search_potentially_covered_backgrd_elements(
    std::map<int, std::set<int>>* current_subset_tofill,
    std::shared_ptr<Core::Geo::SearchTree> backgrd_SearchTree, const Core::FE::Discretization& dis,
    const std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions,
    const Core::LinAlg::Matrix<3, 1>& point, const double radius, const int label)
{
  *current_subset_tofill =
      backgrd_SearchTree->search_elements_in_radius(dis, currentpositions, point, radius, label);
  return;
}  // search_potentially_covered_backgrd_elements


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::evaluate_subset_elements(Teuchos::ParameterList& params,
    Core::FE::Discretization& dis, std::map<int, std::set<int>>& elementstoeval, int action)
{
  // pointer to element
  Core::Elements::Element* ele;

  // initialize location array
  Core::Elements::LocationArray la(1);

  for (std::map<int, std::set<int>>::const_iterator closele = elementstoeval.begin();
      closele != elementstoeval.end(); closele++)
  {
    for (std::set<int>::const_iterator eleIter = (closele->second).begin();
        eleIter != (closele->second).end(); eleIter++)
    {
      ele = dis.g_element(*eleIter);

      Core::LinAlg::SerialDenseMatrix dummymatrix;
      Core::LinAlg::SerialDenseVector dummyvector;
      ele->evaluate(
          params, dis, la, dummymatrix, dummymatrix, dummyvector, dummyvector, dummyvector);
    }
  }

  return;
}  // evaluate_subset_elements


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Immersed::ImmersedBase::write_extra_output(MPI_Comm comm, const double time,
    const std::string filenameending, const std::vector<double> valuetowrite,
    const std::vector<double> valuetowrite2, const std::vector<double> valuetowrite3)
{
  // append values to output file
  if (Core::Communication::my_mpi_rank(comm) == 0)
  {
    const std::string fname1 =
        Global::Problem::instance()->output_control_file()->file_name() + "." + filenameending;

    std::ofstream f1;
    f1.open(fname1.c_str(), std::fstream::ate | std::fstream::app);

    f1 << time << " " << valuetowrite[0] << " " << valuetowrite[1] << " " << valuetowrite[2] << " "
       << valuetowrite[3] << " " << valuetowrite2[0] << " " << valuetowrite2[1] << " "
       << valuetowrite2[2] << " " << valuetowrite2[3] << " " << valuetowrite3[0] << " "
       << valuetowrite3[1] << " " << valuetowrite3[2] << " " << valuetowrite3[3] << "   ";

    f1 << "\n";
    f1.flush();
    f1.close();

  }  // only proc 0 writes
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<double> Immersed::ImmersedBase::calc_global_resultantfrom_epetra_vector(MPI_Comm comm,
    const Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& vec_epetra)
{
  double summyrowentriesx = 0.0;
  double summyrowentriesy = 0.0;
  double summyrowentriesz = 0.0;
  double result_globalx = 0.0;
  double result_globally = 0.0;
  double result_globalz = 0.0;
  double result_L2norm = 0.0;

  std::vector<double> result;

  const int nummyrownodes = dis.num_my_row_nodes();
  const int myveclength = vec_epetra.MyLength();

  if (myveclength != nummyrownodes * 3) FOUR_C_THROW("local vector length is invalid!");

  for (int i = 0; i < nummyrownodes; i++)
  {
    summyrowentriesx += vec_epetra.Values()[i * 3 + 0];
    summyrowentriesy += vec_epetra.Values()[i * 3 + 1];
    summyrowentriesz += vec_epetra.Values()[i * 3 + 2];
  }

  Core::Communication::barrier(comm);
  Core::Communication::sum_all(&summyrowentriesx, &result_globalx, 1, comm);
  Core::Communication::sum_all(&summyrowentriesy, &result_globally, 1, comm);
  Core::Communication::sum_all(&summyrowentriesz, &result_globalz, 1, comm);

  result_L2norm = sqrt(pow(result_globalx, 2) + pow(result_globally, 2) + pow(result_globalz, 2));

  result.push_back(result_globalx);
  result.push_back(result_globally);
  result.push_back(result_globalz);
  result.push_back(result_L2norm);

  return result;
}

FOUR_C_NAMESPACE_CLOSE
