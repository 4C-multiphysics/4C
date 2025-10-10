// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_DbcHDG.hpp"

#include "4C_fem_discretization_hdg.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_calc.hpp"
#include "4C_fluid_ele_calc_hdg.hpp"
#include "4C_fluid_ele_hdg.hpp"
#include "4C_global_data.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::Utils::DbcHdgFluid::read_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond, double time,
    Core::FE::Dbc::DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids,
    int hierarchical_order) const
{
  // no need to check the cast, because it has been done during
  // the build process (see build_dbc())
  const Core::FE::DiscretizationFaces& face_discret =
      static_cast<const Core::FE::DiscretizationFaces&>(discret);

  read_dirichlet_condition(params, face_discret, cond, time, info, dbcgids, hierarchical_order);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::Utils::DbcHdgFluid::read_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::DiscretizationFaces& discret, const Core::Conditions::Condition& cond,
    double time, Core::FE::Dbc::DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids,
    int hierarchical_order) const

{
  // call to corresponding method in base class; safety checks inside
  Core::FE::Dbc::read_dirichlet_condition(
      params, discret, cond, time, info, dbcgids, hierarchical_order);

  // say good bye if there are no face elements
  if (discret.face_row_map() == nullptr) return;

  // get onoff toggles
  const auto& onoff = cond.parameters().get<std::vector<int>>("ONOFF");

  if (discret.num_my_row_faces() > 0)
  {
    // initialize with true on each proc except proc 0
    bool pressureDone = Core::Communication::my_mpi_rank(discret.get_comm()) != 0;

    // loop over all faces
    for (int i = 0; i < discret.num_my_row_faces(); ++i)
    {
      const Core::Elements::FaceElement* faceele =
          dynamic_cast<const Core::Elements::FaceElement*>(discret.l_row_face(i));
      const unsigned int dofperface =
          faceele->parent_master_element()->num_dof_per_face(faceele->face_master_number());
      const unsigned int dofpercomponent =
          faceele->parent_master_element()->num_dof_per_component(faceele->face_master_number());
      const unsigned int component = dofperface / dofpercomponent;

      if (onoff.size() <= component || onoff[component] == 0 ||
          Global::Problem::instance(0)->get_problem_type() != Core::ProblemType::fluid)
        pressureDone = true;
      if (!pressureDone)
      {
        if (discret.num_my_row_elements() > 0 &&
            Core::Communication::my_mpi_rank(discret.get_comm()) == 0)
        {
          std::vector<int> predof = discret.dof(0, discret.l_row_element(0));
          const int gid = predof[0];
          const int lid = discret.dof_row_map(0)->lid(gid);

          // set toggle vector
          info.toggle.get_local_values()[lid] = 1;
          // amend vector of DOF-IDs which are Dirichlet BCs
          if (dbcgids[set_row] != nullptr) (*dbcgids[set_row]).insert(gid);
          pressureDone = true;
        }
      }

      // do only faces where all nodes are present in the node list
      bool faceRelevant = true;
      int nummynodes = discret.l_row_face(i)->num_node();
      const int* mynodes = discret.l_row_face(i)->node_ids();
      for (int j = 0; j < nummynodes; ++j)
        if (!cond.contains_node(mynodes[j]))
        {
          faceRelevant = false;
          break;
        }
      if (!faceRelevant) continue;

      // get dofs of current face element
      std::vector<int> dofs = discret.dof(0, discret.l_row_face(i));

      // loop over dofs
      for (unsigned int j = 0; j < dofperface; ++j)
      {
        // get global id
        const int gid = dofs[j];
        // get corresponding local id
        const int lid = info.toggle.get_map().lid(gid);
        if (lid < 0)
          FOUR_C_THROW("Global id {} not on this proc {} in system vector", dofs[j],
              Core::Communication::my_mpi_rank(discret.get_comm()));
        // get position of label for this dof in condition line
        int onesetj = j / dofpercomponent;

        if (onoff[onesetj] == 0)
        {
          // no DBC on this dof, set toggle zero
          info.toggle.get_local_values()[lid] = 0;
          // get rid of entry in DBC map - if it exists
          if (dbcgids[set_row] != nullptr) (*dbcgids[set_row]).erase(gid);
          continue;
        }
        else  // if (onoff[onesetj]==1)
        {
          // dof has DBC, set toggle vector one
          info.toggle.get_local_values()[lid] = 1;
          // amend vector of DOF-IDs which are dirichlet BCs
          if (dbcgids[set_row] != nullptr) (*dbcgids[set_row]).insert(gid);
        }

      }  // loop over DOFs of face
    }  // loop over all faces
  }  // if there are faces

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::Utils::DbcHdgFluid::do_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond, double time,
    const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
    const Core::LinAlg::Vector<int>& toggle, const std::shared_ptr<std::set<int>>* dbcgids) const
{
  // no need to check the cast, because it has been done during
  // the build process (see build_dbc())
  const Core::FE::DiscretizationFaces& face_discret =
      static_cast<const Core::FE::DiscretizationFaces&>(discret);

  do_dirichlet_condition(params, face_discret, cond, time, systemvectors, toggle);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void FLD::Utils::DbcHdgFluid::do_dirichlet_condition(const Teuchos::ParameterList& params,
    const Core::FE::DiscretizationFaces& discret, const Core::Conditions::Condition& cond,
    double time, const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
    const Core::LinAlg::Vector<int>& toggle) const
{
  // call corresponding method from base class; safety checks inside
  Core::FE::Dbc::do_dirichlet_condition(
      params, discret, cond, time, systemvectors, toggle, nullptr);

  // say good bye if there are no face elements
  if (discret.face_row_map() == nullptr) return;

  // get ids of conditioned nodes
  const std::vector<int>* nodeids = cond.get_nodes();
  if (!nodeids) FOUR_C_THROW("Dirichlet condition does not have nodal cloud");

  // get curves, functs, vals, and onoff toggles from the condition
  const auto funct = cond.parameters().get<std::vector<std::optional<int>>>("FUNCT");
  const auto val = cond.parameters().get<std::vector<double>>("VAL");
  const auto onoff = cond.parameters().get<std::vector<int>>("ONOFF");

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

  // do we have faces?
  if (discret.num_my_row_faces() > 0)
  {
    Core::LinAlg::SerialDenseVector elevec1, elevec2, elevec3;
    Core::LinAlg::SerialDenseMatrix elemat1, elemat2;
    Core::Elements::LocationArray dummy(1);
    Teuchos::ParameterList initParams;
    if (Global::Problem::instance(0)->get_problem_type() == Core::ProblemType::scatra)
      Core::Utils::add_enum_class_to_parameter_list<Core::FE::HDGAction>(
          "action", Core::FE::HDGAction::project_dirich_field, initParams);
    else
      // TODO: Introduce a general action type that is valid for all problems
      initParams.set<FLD::Action>("action", FLD::project_fluid_field);

    // valid for all problems
    std::vector<int> funct_without_nones(funct.size());
    for (unsigned int i = 0; i < funct.size(); ++i)
    {
      if (funct[i].has_value() && funct[i].value() > 0)
        funct_without_nones[i] = funct[i].value();
      else
        funct_without_nones[i] = -1;
    }

    Teuchos::Array<int> functarray(funct_without_nones);
    initParams.set("funct", functarray);

    Teuchos::Array<int> onoffarray(onoff);
    initParams.set("onoff", onoffarray);
    initParams.set("time", time);

    // initialize with true if proc is not proc 0
    bool pressureDone = Core::Communication::my_mpi_rank(discret.get_comm()) != 0;

    // loop over all faces
    for (int i = 0; i < discret.num_my_row_faces(); ++i)
    {
      const Core::Elements::FaceElement* faceele =
          dynamic_cast<const Core::Elements::FaceElement*>(discret.l_row_face(i));
      const unsigned int dofperface =
          faceele->parent_master_element()->num_dof_per_face(faceele->face_master_number());
      const unsigned int dofpercomponent =
          faceele->parent_master_element()->num_dof_per_component(faceele->face_master_number());
      const unsigned int component = dofperface / dofpercomponent;

      if (onoff.size() <= component || onoff[component] == 0 ||
          Global::Problem::instance(0)->get_problem_type() != Core::ProblemType::fluid)
        pressureDone = true;
      if (!pressureDone)
      {
        if (discret.num_my_row_elements() > 0 &&
            Core::Communication::my_mpi_rank(discret.get_comm()) == 0)
        {
          std::vector<int> predof = discret.dof(0, discret.l_row_element(0));
          const int gid = predof[0];
          const int lid = discret.dof_row_map(0)->lid(gid);

          // amend vector of DOF-IDs which are Dirichlet BCs
          if (systemvectors[0] != nullptr) (*systemvectors[0]).get_values()[lid] = 0.0;
          if (systemvectors[1] != nullptr) (*systemvectors[1]).get_values()[lid] = 0.0;
          if (systemvectors[2] != nullptr) (*systemvectors[2]).get_values()[lid] = 0.0;

          // --------------------------------------------------------------------------------------
          // get parameters
          Teuchos::ParameterList params = Global::Problem::instance()->fluid_dynamic_params();

          // check whether the imposition of the average pressure is requested
          const auto dopressavgbc = Teuchos::get<bool>(params, "PRESSAVGBC");

          if (dopressavgbc)
          {
            double pressureavgBC = 0.0;

            // get 1st element
            Core::Elements::Element* ele = discret.l_row_element(0);
            Discret::Elements::Fluid* fluidele = dynamic_cast<Discret::Elements::Fluid*>(ele);

            // get material
            std::shared_ptr<Core::Mat::Material> mat = ele->material();

            // get discretization type
            const Core::FE::CellType distype = ele->shape();

            // evaluate pressure average     //TODO also make it valid for every discretization type
            Core::LinAlg::SerialDenseVector elevec = Core::LinAlg::SerialDenseVector(1);
            if (distype == Core::FE::CellType::quad4)
              Discret::Elements::FluidEleCalcHDG<Core::FE::CellType::quad4>::instance()
                  ->evaluate_pressure_average(fluidele, params, mat, elevec);
            else if (distype == Core::FE::CellType::quad8)
              Discret::Elements::FluidEleCalcHDG<Core::FE::CellType::quad8>::instance()
                  ->evaluate_pressure_average(fluidele, params, mat, elevec);
            else if (distype == Core::FE::CellType::quad9)
              Discret::Elements::FluidEleCalcHDG<Core::FE::CellType::quad9>::instance()
                  ->evaluate_pressure_average(fluidele, params, mat, elevec);
            else if (distype == Core::FE::CellType::tri3)
              Discret::Elements::FluidEleCalcHDG<Core::FE::CellType::tri3>::instance()
                  ->evaluate_pressure_average(fluidele, params, mat, elevec);
            else if (distype == Core::FE::CellType::tri6)
              Discret::Elements::FluidEleCalcHDG<Core::FE::CellType::tri6>::instance()
                  ->evaluate_pressure_average(fluidele, params, mat, elevec);
            else
              FOUR_C_THROW("Given distype currently not implemented.");
            pressureavgBC = elevec[0];

            (*systemvectors[0]).get_values()[lid] = pressureavgBC;

            std::cout << "\n-----------------------------------------------------------------------"
                         "-------------------"
                      << std::endl;
            std::cout << "| Warning: Imposing the analytical average pressure in the first element "
                         "as Dirichlet BC |"
                      << std::endl;
            std::cout << "-------------------------------------------------------------------------"
                         "-----------------\n"
                      << std::endl;
          }
          // --------------------------------------------------------------------------------------
          pressureDone = true;
        }
      }
      int nummynodes = discret.l_row_face(i)->num_node();
      const int* mynodes = discret.l_row_face(i)->node_ids();

      // do only faces where all nodes are present in the node list
      bool faceRelevant = true;
      for (int j = 0; j < nummynodes; ++j)
        if (!cond.contains_node(mynodes[j]))
        {
          faceRelevant = false;
          break;
        }
      if (!faceRelevant) continue;

      initParams.set<unsigned int>(
          "faceconsider", static_cast<unsigned int>(faceele->face_master_number()));
      if (static_cast<unsigned int>(elevec1.numRows()) != dofperface) elevec1.shape(dofperface, 1);
      std::vector<int> dofs = discret.dof(0, discret.l_row_face(i));

      bool do_evaluate = false;
      for (unsigned int i = 0; i < component; ++i)
        if (funct[i].has_value() && funct[i].value() > 0) do_evaluate = true;

      if (do_evaluate)
      {
        // cast the const qualifier away, thus the Evaluate routine can be called.
        Core::FE::DiscretizationFaces& non_const_dis =
            const_cast<Core::FE::DiscretizationFaces&>(discret);
        faceele->parent_master_element()->evaluate(
            initParams, non_const_dis, dummy, elemat1, elemat2, elevec1, elevec2, elevec3);
      }
      else
        for (unsigned int i = 0; i < dofperface; ++i) elevec1(i) = 1.;

      // loop over face dofs
      for (unsigned int j = 0; j < dofperface; ++j)
      {
        // get global id
        const int gid = dofs[j];
        // get corresponding local id
        const int lid = toggle.get_map().lid(gid);
        if (lid < 0)
          FOUR_C_THROW("Global id {} not on this proc {} in system vector", dofs[j],
              Core::Communication::my_mpi_rank(discret.get_comm()));
        // get position of label for this dof in condition line
        int onesetj = j / dofpercomponent;

        // check whether dof gid is a dbc gid
        if (toggle.get_local_values()[lid] == 0) continue;

        std::vector<double> value(deg + 1, val[onesetj]);

        // assign value
        if (systemvectors[0] != nullptr)
          (*systemvectors[0]).get_values()[lid] = value[0] * elevec1(j);
        if (systemvectors[1] != nullptr)
          (*systemvectors[1]).get_values()[lid] = value[1] * elevec1(j);
        if (systemvectors[2] != nullptr)
          (*systemvectors[2]).get_values()[lid] = value[2] * elevec1(j);

      }  // loop over all DOFs
    }  // loop over all faces

  }  // if there are faces

  return;
}

FOUR_C_NAMESPACE_CLOSE
