// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_springdashpot.hpp"

#include "4C_adapter_coupling_nonlin_mortar.hpp"
#include "4C_contact_interface.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_pstream.hpp"  // has to go before io.hpp
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_truss3.hpp"
#include "4C_utils_function.hpp"
#include "4C_utils_function_of_time.hpp"

#include <iostream>
#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                         pfaller Apr15|
 *----------------------------------------------------------------------*/
CONSTRAINTS::SpringDashpot::SpringDashpot(std::shared_ptr<Core::FE::Discretization> dis,
    std::shared_ptr<Core::Conditions::Condition> cond)
    : actdisc_(std::move(dis)),
      spring_(std::move(cond)),
      stiff_tens_((spring_->parameters().get<std::vector<double>>("STIFF"))[0]),
      stiff_comp_((spring_->parameters().get<std::vector<double>>("STIFF"))[0]),
      offset_((spring_->parameters().get<std::vector<double>>("DISPLOFFSET"))[0]),
      viscosity_((spring_->parameters().get<std::vector<double>>("VISCO"))[0]),
      coupling_(spring_->parameters().get<int>("COUPLING")),
      nodes_(spring_->get_nodes()),
      area_(),
      gap0_(),
      gap_(),
      gapdt_(),
      dgap_(),
      normals_(),
      dnormals_(),
      offset_prestr_(),
      offset_prestr_new_(nullptr)
{
  offset_prestr_new_ = std::make_shared<Core::LinAlg::Vector<double>>(*actdisc_->dof_row_map());
  offset_prestr_new_->PutScalar(0.0);

  // set type of this spring
  set_spring_type();

  if (springtype_ != cursurfnormal && coupling_ >= 0)
  {
    FOUR_C_THROW(
        "Coupling of spring dashpot to reference surface only possible for DIRECTION "
        "cursurfnormal.");
  }

  if (springtype_ == cursurfnormal && coupling_ == -1)
    FOUR_C_THROW("Coupling id necessary for DIRECTION cursurfnormal.");

  // safety checks of input
  const auto* springstiff = &spring_->parameters().get<std::vector<double>>("STIFF");
  const auto* numfuncstiff = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTSTIFF");
  const auto* numfuncnonlinstiff = &spring_->parameters().get<std::vector<int>>("FUNCTNONLINSTIFF");

  for (unsigned i = 0; i < (*numfuncnonlinstiff).size(); ++i)
  {
    if ((*numfuncnonlinstiff)[i] != 0 and ((*springstiff)[i] != 0 or (*numfuncstiff)[i] != 0))
      FOUR_C_THROW("Must not apply nonlinear stiffness and linear stiffness");
  }

  // ToDo: delete rest until return statement!

  // get normal vectors if necessary
  if (springtype_ == cursurfnormal)
  {
    // get geometry
    std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = spring_->geometry();
    // calculate nodal area
    if (!Core::Communication::my_mpi_rank(actdisc_->get_comm()))
      Core::IO::cout << "Computing area for spring dashpot condition...\n";
    get_area(geom);
    initialize_cur_surf_normal();
  }

  // ToDo: do we really need this??
  // initialize prestressing offset
  initialize_prestr_offset();
}

// NEW version, consistently integrated over element surface!!
/*----------------------------------------------------------------------*
 * Integrate a Surface Robin boundary condition (public)       mhv 08/16|
 * ---------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::evaluate_robin(std::shared_ptr<Core::LinAlg::SparseMatrix> stiff,
    std::shared_ptr<Core::LinAlg::Vector<double>> fint,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> velo, Teuchos::ParameterList p)
{
  // reset last Newton step
  springstress_.clear();

  const bool assvec = fint != nullptr;
  const bool assmat = stiff != nullptr;

  actdisc_->clear_state();
  actdisc_->set_state("displacement", disp);
  actdisc_->set_state("velocity", velo);
  actdisc_->set_state("offset_prestress", offset_prestr_new_);

  // get values and switches from the condition
  const auto* onoff = &spring_->parameters().get<std::vector<int>>("ONOFF");
  const auto* springstiff = &spring_->parameters().get<std::vector<double>>("STIFF");
  const auto* numfuncstiff = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTSTIFF");
  const auto* dashpotvisc = &spring_->parameters().get<std::vector<double>>("VISCO");
  const auto* numfuncvisco = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTVISCO");
  const auto* disploffset = &spring_->parameters().get<std::vector<double>>("DISPLOFFSET");
  const auto* numfuncdisploffset =
      &spring_->parameters().get<std::vector<int>>("TIMEFUNCTDISPLOFFSET");
  const auto* numfuncnonlinstiff = &spring_->parameters().get<std::vector<int>>("FUNCTNONLINSTIFF");
  const auto* direction = &spring_->parameters().get<std::string>("DIRECTION");

  // time-integration factor for stiffness contribution of dashpot, d(v_{n+1})/d(d_{n+1})
  const double time_fac = p.get("time_fac", 0.0);
  const double total_time = p.get("total time", 0.0);

  Teuchos::ParameterList params;
  params.set("action", "calc_struct_robinforcestiff");
  params.set("ONOFF", onoff);
  params.set("springstiff", springstiff);
  params.set("dashpotvisc", dashpotvisc);
  params.set("DISPLOFFSET", disploffset);
  params.set("time_fac", time_fac);
  params.set("DIRECTION", direction);
  params.set("TIMEFUNCTSTIFF", numfuncstiff);
  params.set("TIMEFUNCTVISCO", numfuncvisco);
  params.set("TIMEFUNCTDISPLOFFSET", numfuncdisploffset);
  params.set("FUNCTNONLINSTIFF", numfuncnonlinstiff);
  params.set("total time", total_time);

  switch (spring_->g_type())
  {
    case Core::Conditions::geometry_type_surface:
    {
      std::map<int, std::shared_ptr<Core::Elements::Element>>& geom = spring_->geometry();

      // no check for empty geometry here since in parallel computations
      // can exist processors which do not own a portion of the elements belonging
      // to the condition geometry
      for (auto& curr : geom)
      {
        // get element location vector and ownerships
        std::vector<int> lm;
        std::vector<int> lmowner;
        std::vector<int> lmstride;

        curr.second->location_vector(*actdisc_, lm, lmowner, lmstride);

        const int eledim = (int)lm.size();

        // define element matrices and vectors
        Core::LinAlg::SerialDenseMatrix elematrix1;
        Core::LinAlg::SerialDenseMatrix elematrix2;
        Core::LinAlg::SerialDenseVector elevector1;
        Core::LinAlg::SerialDenseVector elevector2;
        Core::LinAlg::SerialDenseVector elevector3;

        elevector1.size(eledim);
        elevector2.size(eledim);
        elevector3.size(eledim);
        elematrix1.shape(eledim, eledim);

        int err = curr.second->evaluate(
            params, *actdisc_, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
        if (err) FOUR_C_THROW("error while evaluating elements");

        if (assvec) Core::LinAlg::assemble(*fint, elevector1, lm, lmowner);
        if (assmat) stiff->assemble(lmstride, elematrix1, lm, lmowner);

        // save spring stress for postprocessing
        const int numdim = 3;
        const int numdf = 3;
        std::vector<double> stress(numdim, 0.0);

        for (int node = 0; node < curr.second->num_node(); ++node)
        {
          for (int dim = 0; dim < numdim; dim++) stress[dim] = elevector3[node * numdf + dim];
          springstress_.insert(
              std::pair<int, std::vector<double>>(curr.second->node_ids()[node], stress));
        }
      } /* end of loop over geometry */
      break;
    }
    case Core::Conditions::geometry_type_point:
    {
      if (*direction == "xyz")
      {
        // get all nodes of this condition and check, if it's just one -> get this node
        const auto* nodes_cond = spring_->get_nodes();
        if (nodes_cond->size() != 1)
          FOUR_C_THROW("Point Robin condition must be defined on one node.");
        const int node_gid = nodes_cond->at(0);
        auto* node = actdisc_->g_node(node_gid);

        // get adjacent element of this node and check if it's just one -> get this element and cast
        // it to truss element
        if (node->num_element() != 1) FOUR_C_THROW("Node may only have one element");
        auto* ele = node->elements();
        auto* truss_ele = dynamic_cast<Discret::Elements::Truss3*>(ele[0]);
        if (truss_ele == nullptr)
        {
          FOUR_C_THROW(
              "Currently, only Truss Elements are allowed to evaluate point Robin Conditon. Cast "
              "to "
              "Truss Element failed.");
        }

        // dofs of this node
        auto dofs_gid = actdisc_->dof(0, node);

        // get cross section for integration of this element
        const double cross_section = truss_ele->cross_section();

        for (size_t dof = 0; dof < onoff->size(); ++dof)
        {
          const int dof_onoff = (*onoff)[dof];
          if (dof_onoff == 0) continue;

          const int dof_gid = dofs_gid[dof];
          const int dof_lid = actdisc_->dof_row_map()->LID(dof_gid);

          const double dof_disp = (*disp)[dof_lid];
          const double dof_vel = (*velo)[dof_lid];

          // compute stiffness, viscosity, and initial offset from functions
          const double dof_stiffness =
              (*numfuncstiff)[dof] != 0
                  ? (*springstiff)[dof] *
                        Global::Problem::instance()
                            ->function_by_id<Core::Utils::FunctionOfTime>((*numfuncstiff)[dof] - 1)
                            .evaluate(total_time)
                  : (*springstiff)[dof];
          const double dof_viscosity =
              (*numfuncvisco)[dof] != 0
                  ? (*dashpotvisc)[dof] *
                        Global::Problem::instance()
                            ->function_by_id<Core::Utils::FunctionOfTime>((*numfuncvisco)[dof] - 1)
                            .evaluate(total_time)
                  : (*dashpotvisc)[dof];
          const double dof_disploffset =
              (*numfuncdisploffset)[dof] != 0
                  ? (*disploffset)[dof] * Global::Problem::instance()
                                              ->function_by_id<Core::Utils::FunctionOfTime>(
                                                  (*numfuncdisploffset)[dof] - 1)
                                              .evaluate(total_time)
                  : (*disploffset)[dof];

          // displacement related forces and derivatives
          double force_disp = 0.0;
          double force_disp_deriv = 0.0;
          if ((*numfuncnonlinstiff)[dof] == 0)
          {
            force_disp = dof_stiffness * (dof_disp - dof_disploffset);
            force_disp_deriv = dof_stiffness;
          }
          else
          {
            std::array<double, 3> displ = {(*disp)[0], (*disp)[1], (*disp)[2]};
            force_disp = Global::Problem::instance()
                             ->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                                 (*numfuncnonlinstiff)[dof] - 1)
                             .evaluate(displ.data(), total_time, 0);

            force_disp_deriv = (Global::Problem::instance()
                                    ->function_by_id<Core::Utils::FunctionOfSpaceTime>(
                                        (*numfuncnonlinstiff)[dof] - 1)
                                    .evaluate_spatial_derivative(displ.data(), total_time, 0))[dof];
          }

          // velocity related forces and derivatives
          const double force_vel = dof_viscosity * dof_vel;
          const double force_vel_deriv = dof_viscosity;

          const double force = force_disp + force_vel;
          const double stiffness = force_disp_deriv + force_vel_deriv * time_fac;

          // assemble contributions into force vector and stiffness matrix
          (*fint)[dof_lid] += force * cross_section;
          if (stiff != nullptr) stiff->assemble(-stiffness * cross_section, dof_gid, dof_gid);
        }
      }
      else
      {
        FOUR_C_THROW(
            "Only 'xyz' for 'DIRECTION' supported in 'DESIGN POINT ROBIN SPRING DASHPOT "
            "CONDITIONS'");
      }
      break;
    }
    default:
      FOUR_C_THROW("Geometry type for spring dashpot must either be either 'Surface' or 'Point'.");
  }
}


// old version, NOT consistently integrated over element surface!!
/*----------------------------------------------------------------------*
 |                                                         pfaller Mar16|
 *----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::evaluate_force(Core::LinAlg::Vector<double>& fint,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
    const Core::LinAlg::Vector<double>& vel, const Teuchos::ParameterList& p)
{
  if (disp == nullptr) FOUR_C_THROW("Cannot find displacement state in discretization");

  if (springtype_ == cursurfnormal) get_cur_normals(disp, p);

  // loop nodes of current condition
  for (int node_gid : *nodes_)
  {
    // nodes owned by processor
    if (actdisc_->node_row_map()->MyGID(node_gid))
    {
      int gid = node_gid;
      Core::Nodes::Node* node = actdisc_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find global node %d", gid);

      // get nodal values
      const double nodalarea = area_[gid];               // nodal area
      const std::vector<double> normal = normals_[gid];  // normalized nodal normal
      const std::vector<double> offsetprestr =
          offset_prestr_[gid];  // get nodal displacement values of last time step for MULF offset

      const int numdof = actdisc_->num_dof(0, node);
      assert(numdof == 3);
      std::vector<int> dofs = actdisc_->dof(0, node);

      // initialize
      double gap = 0.;          // displacement
      double gapdt = 0.;        // velocity
      double springstiff = 0.;  // spring stiffness

      // calculation of normals and displacements differs for each spring variant
      switch (springtype_)
      {
        case xyz:  // spring dashpot acts in every surface dof direction
          FOUR_C_THROW("You should not be here! Use the new consistent EvaluateRobin routine!!!");
          break;

        case refsurfnormal:  // spring dashpot acts in refnormal direction
          FOUR_C_THROW("You should not be here! Use the new consistent EvaluateRobin routine!!!");
          break;

        case cursurfnormal:  // spring dashpot acts in curnormal direction

          // safety checks
          const auto* numfuncstiff = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTSTIFF");
          const auto* numfuncvisco = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTVISCO");
          const auto* numfuncdisploffset =
              &spring_->parameters().get<std::vector<int>>("TIMEFUNCTDISPLOFFSET");
          const auto* numfuncnonlinstiff =
              &spring_->parameters().get<std::vector<int>>("FUNCTNONLINSTIFF");
          for (int dof_numfuncstiff : *numfuncstiff)
          {
            if (dof_numfuncstiff != 0)
            {
              FOUR_C_THROW(
                  "temporal dependence of stiffness not implemented for current surface "
                  "evaluation");
            }
          }
          for (int dof_numfuncvisco : *numfuncvisco)
          {
            if (dof_numfuncvisco != 0)
              FOUR_C_THROW(
                  "temporal dependence of damping not implemented for current surface evaluation");
          }
          for (int dof_numfuncdisploffset : *numfuncdisploffset)
          {
            if (dof_numfuncdisploffset != 0)
              FOUR_C_THROW(
                  "temporal dependence of offset not implemented for current surface evaluation");
          }
          for (int dof_numfuncnonlinstiff : *numfuncnonlinstiff)
          {
            if (dof_numfuncnonlinstiff != 0)
              FOUR_C_THROW("Nonlinear spring not implemented for current surface evaluation");
          }

          // spring displacement
          gap = gap_[gid];
          //        gapdt = gapdt_[gid]; // unused ?!?

          // select spring stiffnes
          springstiff = select_stiffness(gap);

          // assemble into residual vector
          std::vector<double> out_vec(numdof, 0.);
          for (int k = 0; k < numdof; ++k)
          {
            // force
            const double val =
                -nodalarea *
                (springstiff * (gap - offsetprestr[k] - offset_) + viscosity_ * gapdt) * normal[k];
            const int err = fint.SumIntoGlobalValues(1, &val, &dofs[k]);
            if (err) FOUR_C_THROW("SumIntoGlobalValues failed!");

            // store spring stress for output
            out_vec[k] =
                (springstiff * (gap - offsetprestr[k] - offset_) + viscosity_ * gapdt) * normal[k];
          }
          // add to output
          springstress_.insert(std::pair<int, std::vector<double>>(gid, out_vec));
          break;
      }
    }  // node owned by processor
  }    // loop over nodes
}


// old version, NOT consistently integrated over element surface!!
/*----------------------------------------------------------------------*
 |                                                         pfaller mar16|
 *----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::evaluate_force_stiff(Core::LinAlg::SparseMatrix& stiff,
    Core::LinAlg::Vector<double>& fint,
    const std::shared_ptr<const Core::LinAlg::Vector<double>> disp,
    const Core::LinAlg::Vector<double>& vel, Teuchos::ParameterList p)
{
  if (disp == nullptr) FOUR_C_THROW("Cannot find displacement state in discretization");

  if (springtype_ == cursurfnormal)
  {
    get_cur_normals(disp, p);
    stiff.un_complete();  // sparsity pattern might change
  }

  // time-integration factor for stiffness contribution of dashpot, d(v_{n+1})/d(d_{n+1})
  const double dt = p.get("dt", 1.0);

  // loop nodes of current condition
  for (int node_gid : *nodes_)
  {
    // nodes owned by processor
    if (actdisc_->node_row_map()->MyGID(node_gid))
    {
      Core::Nodes::Node* node = actdisc_->g_node(node_gid);
      if (!node) FOUR_C_THROW("Cannot find global node %d", node_gid);

      // get nodal values
      const double nodalarea = area_[node_gid];               // nodal area
      const std::vector<double> normal = normals_[node_gid];  // normalized nodal normal
      const std::vector<double> offsetprestr =
          offset_prestr_[node_gid];  // get nodal displacement values of last time step for MULF
                                     // offset

      const int numdof = actdisc_->num_dof(0, node);
      assert(numdof == 3);
      std::vector<int> dofs = actdisc_->dof(0, node);

      // initialize
      double gap = 0.;          // displacement
      double gapdt = 0.;        // velocity
      double springstiff = 0.;  // spring stiffness

      // calculation of normals and displacements differs for each spring variant
      switch (springtype_)
      {
        case xyz:  // spring dashpot acts in every surface dof direction
          FOUR_C_THROW("You should not be here! Use the new consistent EvaluateRobin routine!!!");
          break;

        case refsurfnormal:  // spring dashpot acts in refnormal direction
          FOUR_C_THROW("You should not be here! Use the new consistent EvaluateRobin routine!!!");
          break;

        case cursurfnormal:  // spring dashpot acts in curnormal direction

          // safety checks
          const auto* numfuncstiff = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTSTIFF");
          const auto* numfuncvisco = &spring_->parameters().get<std::vector<int>>("TIMEFUNCTVISCO");
          const auto* numfuncdisploffset =
              &spring_->parameters().get<std::vector<int>>("TIMEFUNCTDISPLOFFSET");
          const auto* numfuncnonlinstiff =
              &spring_->parameters().get<std::vector<int>>("FUNCTNONLINSTIFF");
          for (int dof_numfuncstiff : *numfuncstiff)
          {
            if (dof_numfuncstiff != 0)
            {
              FOUR_C_THROW(
                  "temporal dependence of stiffness not implemented for current surface "
                  "evaluation");
            }
          }
          for (int dof_numfuncvisco : *numfuncvisco)
          {
            if (dof_numfuncvisco != 0)
              FOUR_C_THROW(
                  "temporal dependence of damping not implemented for current surface evaluation");
          }
          for (int dof_numfuncdisploffset : *numfuncdisploffset)
          {
            if (dof_numfuncdisploffset != 0)
              FOUR_C_THROW(
                  "temporal dependence of offset not implemented for current surface evaluation");
          }
          for (int dof_numfuncnonlinstiff : *numfuncnonlinstiff)
          {
            if (dof_numfuncnonlinstiff != 0)
              FOUR_C_THROW("Nonlinear spring not implemented for current surface evaluation");
          }

          // spring displacement
          gap = gap_[node_gid];
          gapdt = gapdt_[node_gid];

          // select spring stiffnes
          springstiff = select_stiffness(gap);

          // assemble into residual vector and stiffness matrix
          std::vector<double> out_vec(numdof, 0.);
          for (int k = 0; k < numdof; ++k)
          {
            // force
            const double val =
                -nodalarea *
                (springstiff * (gap - offsetprestr[k] - offset_) + viscosity_ * gapdt) * normal[k];
            const int err = fint.SumIntoGlobalValues(1, &val, &dofs[k]);
            if (err) FOUR_C_THROW("SumIntoGlobalValues failed!");

            // stiffness
            std::map<int, double> dgap = dgap_[node_gid];
            std::vector<Core::Gen::Pairedvector<int, double>> dnormal = dnormals_[node_gid];

            // check if projection exists
            if (!dnormal.empty() && !dgap.empty())
            {
              // linearize gap
              for (auto& i : dgap)
              {
                const double dval = -nodalarea *
                                    (springstiff * (i.second) + viscosity_ * (i.second) / dt) *
                                    normal[k];
                stiff.assemble(dval, dofs[k], i.first);
              }

              // linearize normal
              for (auto& i : dnormal[k])
              {
                const double dval =
                    -nodalarea *
                    (springstiff * (gap - offsetprestr[k] - offset_) + viscosity_ * gapdt) *
                    (i.second);
                stiff.assemble(dval, dofs[k], i.first);
              }
            }
            // store negative value of internal force for output (=reaction force)
            out_vec[k] = -val;
          }
          // add to output
          springstress_.insert(std::pair<int, std::vector<double>>(node_gid, out_vec));
          break;
      }
    }  // node owned by processor
  }    // loop over nodes

  if (springtype_ == cursurfnormal) stiff.complete();  // sparsity pattern might have changed
}

/*----------------------------------------------------------------------*
 |                                                         pfaller Mar16|
 *----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::reset_newton()
{
  // all springs
  gap_.clear();
  gapdt_.clear();
  springstress_.clear();

  // only curnormal
  if (springtype_ == cursurfnormal)
  {
    dgap_.clear();
    normals_.clear();
    dnormals_.clear();
  }
}

/*----------------------------------------------------------------------*
 |                                                             mhv 12/15|
 *----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::reset_prestress(const Core::LinAlg::Vector<double>& dis)
{
  // this should be sufficient, no need to loop over nodes anymore
  offset_prestr_new_->Update(1.0, dis, 1.0);

  // loop over all nodes only necessary for cursurfnormal which does not use consistent integration
  if (springtype_ == cursurfnormal)
  {
    for (int node_gid : *nodes_)
    {
      // nodes owned by processor
      if (actdisc_->node_row_map()->MyGID(node_gid))
      {
        Core::Nodes::Node* node = actdisc_->g_node(node_gid);
        if (!node) FOUR_C_THROW("Cannot find global node %d", node_gid);

        const int numdof = actdisc_->num_dof(0, node);
        assert(numdof == 3);
        std::vector<int> dofs = actdisc_->dof(0, node);

        // initialize. calculation of displacements differs for each spring variant
        std::vector<double> uoff(numdof, 0.0);  // displacement vector of condition nodes
        offset_prestr_.insert(std::pair<int, std::vector<double>>(node_gid, uoff));

      }  // node owned by processor
    }    // loop over nodes
  }
}

/*----------------------------------------------------------------------*
 |                                                             mhv 12/15|
 *----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::set_restart(Core::LinAlg::Vector<double>& vec)
{
  offset_prestr_new_->Update(1.0, vec, 0.0);
}

/*----------------------------------------------------------------------*
 |                                                             mhv 12/15|
 *----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::set_restart_old(Core::LinAlg::MultiVector<double>& vec)
{
  // loop nodes of current condition
  for (int node_gid : *nodes_)
  {
    // nodes owned by processor
    if (actdisc_->node_row_map()->MyGID(node_gid))
    {
      Core::Nodes::Node* node = actdisc_->g_node(node_gid);
      if (!node) FOUR_C_THROW("Cannot find global node %d", node_gid);

      [[maybe_unused]] const int numdof = actdisc_->num_dof(0, node);
      assert(numdof == 3);
      std::vector<int> dofs = actdisc_->dof(0, node);


      if (springtype_ == refsurfnormal || springtype_ == xyz)
      {
        // import spring offset length
        for (auto& i : offset_prestr_)
        {
          // global id -> local id
          const int lid = vec.Map().LID(i.first);
          // local id on processor
          if (lid >= 0)
          {
            // copy all components of spring offset length vector
            (i.second)[0] = ((vec)(0))[lid];
            (i.second)[1] = ((vec)(1))[lid];
            (i.second)[2] = ((vec)(2))[lid];
          }
        }
      }

    }  // node owned by processor
  }    // loop over nodes
}

/*----------------------------------------------------------------------*
 |                                                         pfaller Jan14|
 *----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::output_gap_normal(Core::LinAlg::Vector<double>& gap,
    Core::LinAlg::MultiVector<double>& normals, Core::LinAlg::MultiVector<double>& stress) const
{
  // export gap function
  for (const auto& i : gap_)
  {
    // global id -> local id
    const int lid = gap.Map().LID(i.first);
    // local id on processor
    if (lid >= 0) (gap)[lid] += i.second;
  }

  // export normal
  for (const auto& normal : normals_)
  {
    // global id -> local id
    const int lid = normals.Map().LID(normal.first);
    // local id on processor
    if (lid >= 0)
    {
      // copy all components of normal vector
      ((normals)(0))[lid] += (normal.second).at(0);
      ((normals)(1))[lid] += (normal.second).at(1);
      ((normals)(2))[lid] += (normal.second).at(2);
    }
  }

  // export spring stress
  for (const auto& i : springstress_)
  {
    // global id -> local id
    const int lid = stress.Map().LID(i.first);
    // local id on processor
    if (lid >= 0)
    {
      // copy all components of normal vector
      ((stress)(0))[lid] += (i.second).at(0);
      ((stress)(1))[lid] += (i.second).at(1);
      ((stress)(2))[lid] += (i.second).at(2);
    }
  }
}

/*----------------------------------------------------------------------*
 |                                                             mhv Dec15|
 *----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::output_prestr_offset(
    Core::LinAlg::Vector<double>& springprestroffset) const
{
  springprestroffset.Update(1.0, *offset_prestr_new_, 0.0);
}

/*----------------------------------------------------------------------*
 |                                                             mhv Dec15|
 *----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::output_prestr_offset_old(
    Core::LinAlg::MultiVector<double>& springprestroffset) const
{
  // export spring offset length
  for (const auto& i : offset_prestr_)
  {
    // global id -> local id
    const int lid = springprestroffset.Map().LID(i.first);
    // local id on processor
    if (lid >= 0)
    {
      // copy all components of spring offset length vector
      ((springprestroffset)(0))[lid] = (i.second)[0];
      ((springprestroffset)(1))[lid] = (i.second)[1];
      ((springprestroffset)(2))[lid] = (i.second)[2];
    }
  }
}

/*-----------------------------------------------------------------------*
|(private)                                                  pfaller Apr15|
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::initialize_cur_surf_normal()
{
  // create MORTAR interface
  mortar_ = std::make_shared<Adapter::CouplingNonLinMortar>(Global::Problem::instance()->n_dim(),
      Global::Problem::instance()->mortar_coupling_params(),
      Global::Problem::instance()->contact_dynamic_params(),
      Global::Problem::instance()->spatial_approximation_type());

  // create CONTACT elements at interface for normal and gap calculation
  mortar_->setup_spring_dashpot(actdisc_, actdisc_, spring_, coupling_, actdisc_->get_comm());

  // create temp vectors for gap initialization
  std::map<int, std::map<int, double>> tmpdgap_;
  std::map<int, std::vector<double>> tmpnormals_;
  std::map<int, std::vector<Core::Gen::Pairedvector<int, double>>> tmpdnormals_;

  // empty displacement vector
  std::shared_ptr<Core::LinAlg::Vector<double>> disp;
  disp = Core::LinAlg::create_vector(*(actdisc_->dof_row_map()), true);

  // initialize gap in reference configuration
  mortar_->interface()->evaluate_distances(disp, tmpnormals_, tmpdnormals_, gap0_, tmpdgap_);
}

// ToDo: this function should vanish completely
// obsolete when using new EvaluateRobin function!
/*-----------------------------------------------------------------------*
|(private) adapted from mhv 01/14                           pfaller Apr15|
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::get_area(
    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geom)
{
  for (const auto& ele : geom)
  {
    Core::Elements::Element* element = ele.second.get();

    Teuchos::ParameterList eparams;

    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;
    element->location_vector(*(actdisc_), lm, lmowner, lmstride);
    Core::LinAlg::SerialDenseMatrix dummat;
    Core::LinAlg::SerialDenseVector dumvec;
    Core::LinAlg::SerialDenseVector elevector;
    const int eledim = (int)lm.size();
    elevector.size(eledim);

    eparams.set("action", "calc_struct_area");
    eparams.set("area", 0.0);
    element->evaluate(eparams, *(actdisc_), lm, dummat, dummat, dumvec, dumvec, dumvec);

    Core::FE::CellType shape = element->shape();

    double a = eparams.get("area", -1.0);

    // loop over all nodes of the element that share the area
    // do only contribute to my own row nodes
    double apernode = 0.;
    for (int i = 0; i < element->num_node(); ++i)
    {
      /* here we have to take care to assemble the right stiffness to the nodes!!! (mhv 05/2014):
          we do some sort of "manual" gauss integration here since we have to pay attention to
         assemble the correct stiffness in case of quadratic surface elements*/

      switch (shape)
      {
        case Core::FE::CellType::tri3:
          apernode = a / element->num_node();
          break;
        case Core::FE::CellType::tri6:
        {
          // integration of shape functions over parameter element surface
          double int_N_cornernode = 0.;
          double int_N_edgemidnode = 1. / 6.;

          int numcornernode = 3;
          int numedgemidnode = 3;

          double weight = numcornernode * int_N_cornernode + numedgemidnode * int_N_edgemidnode;
          double a_inv_weight = a / weight;

          if (i < 3)  // corner nodes
            apernode = int_N_cornernode * a_inv_weight;
          else  // edge mid nodes
            apernode = int_N_edgemidnode * a_inv_weight;
        }
        break;
        case Core::FE::CellType::quad4:
          apernode = a / element->num_node();
          break;
        case Core::FE::CellType::quad8:
        {
          // integration of shape functions over parameter element surface
          double int_N_cornernode = -1. / 3.;
          double int_N_edgemidnode = 4. / 3.;

          int numcornernode = 4;
          int numedgemidnode = 4;

          double weight = numcornernode * int_N_cornernode + numedgemidnode * int_N_edgemidnode;
          double a_inv_weight = a / weight;

          if (i < 4)  // corner nodes
            apernode = int_N_cornernode * a_inv_weight;
          else  // edge mid nodes
            apernode = int_N_edgemidnode * a_inv_weight;
        }
        break;
        case Core::FE::CellType::quad9:
        {
          // integration of shape functions over parameter element surface
          double int_N_cornernode = 1. / 9.;
          double int_N_edgemidnode = 4. / 9.;
          double int_N_centermidnode = 16. / 9.;

          int numcornernode = 4;
          int numedgemidnode = 4;
          int numcentermidnode = 1;

          double weight = numcornernode * int_N_cornernode + numedgemidnode * int_N_edgemidnode +
                          numcentermidnode * int_N_centermidnode;
          double a_inv_weight = a / weight;

          if (i < 4)  // corner nodes
            apernode = int_N_cornernode * a_inv_weight;
          else if (i == 8)  // center mid node
            apernode = int_N_centermidnode * a / weight;
          else  // edge mid nodes
            apernode = int_N_edgemidnode * a_inv_weight;
        }
        break;
        case Core::FE::CellType::nurbs9:
          FOUR_C_THROW(
              "Not yet implemented for Nurbs! To do: Apply the correct weighting of the area per "
              "node!");
          break;
        default:
          FOUR_C_THROW("shape type unknown!\n");
          break;
      }

      const int gid = element->nodes()[i]->id();
      if (!actdisc_->node_row_map()->MyGID(gid)) continue;

      // store area in map (gid, area). erase old value before adding new one
      const double newarea = area_[gid] + apernode;
      area_.erase(gid);
      area_.insert(std::pair<int, double>(gid, newarea));
    }
  }  // for (ele=geom.begin(); ele != geom.end(); ++ele)
}


/*-----------------------------------------------------------------------*
|(private)                                                    mhv 12/2015|
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::initialize_prestr_offset()
{
  offset_prestr_.clear();

  for (int node_gid : *nodes_)
  {
    if (actdisc_->node_row_map()->MyGID(node_gid))
    {
      Core::Nodes::Node* node = actdisc_->g_node(node_gid);
      if (!node) FOUR_C_THROW("Cannot find global node %d", node_gid);

      int numdof = actdisc_->num_dof(0, node);
      std::vector<int> dofs = actdisc_->dof(0, node);

      assert(numdof == 3);

      std::vector<double> temp(numdof, 0.0);

      // insert to map
      offset_prestr_.insert(std::pair<int, std::vector<double>>(node_gid, temp));
    }
  }
}


/*-----------------------------------------------------------------------*
|(private)                                                  pfaller Apr15|
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::get_cur_normals(
    const std::shared_ptr<const Core::LinAlg::Vector<double>>& disp, Teuchos::ParameterList p)
{
  // get current time step size
  const double dt = p.get("dt", 1.0);

  // temp nodal gap
  std::map<int, double> tmpgap;

  // calculate normals and gap using CONTACT elements
  mortar_->interface()->evaluate_distances(disp, normals_, dnormals_, tmpgap, dgap_);

  // subtract reference gap from current gap (gap in reference configuration is zero everywhere)
  for (auto& i : tmpgap)
  {
    auto j = gap0_.find(i.first);
    if (j == gap0_.end()) gap_[i.first] = i.second;
    //      FOUR_C_THROW("The maps of reference gap and current gap are inconsistent.");
    else
      gap_[i.first] = i.second - j->second;

    // calculate gap velocity via local finite difference (not the best way but also not the worst)
    gapdt_[i.first] = (gap_[i.first] - gapn_[i.first]) / dt;
  }
}

/*-----------------------------------------------------------------------*
|(private)                                                  pfaller Apr15|
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::set_spring_type()
{
  // get spring direction from condition
  const auto dir = spring_->parameters().get<std::string>("DIRECTION");

  if (dir == "xyz")
    springtype_ = xyz;
  else if (dir == "refsurfnormal")
    springtype_ = refsurfnormal;
  else if (dir == "cursurfnormal")
    springtype_ = cursurfnormal;
  else
  {
    FOUR_C_THROW(
        "Invalid direction option! Choose DIRECTION xyz, DIRECTION refsurfnormal or DIRECTION "
        "cursurfnormal!");
  }
}

/*-----------------------------------------------------------------------*
 *-----------------------------------------------------------------------*/
void CONSTRAINTS::SpringDashpot::update()
{
  // store current time step
  gapn_ = gap_;
}

void CONSTRAINTS::SpringDashpot::reset_step_state() { gap_ = gapn_; }

FOUR_C_NAMESPACE_CLOSE
