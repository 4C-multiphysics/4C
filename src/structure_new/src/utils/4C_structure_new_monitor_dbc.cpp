// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_monitor_dbc.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_geometry_element_volume.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_pstream.hpp"
#include "4C_io_runtime_csv_writer.hpp"
#include "4C_io_yaml.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_monitor_dbc_input.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_timint_basedataio.hpp"
#include "4C_structure_new_timint_basedataio_monitor_dbc.hpp"

#include <algorithm>
#include <ranges>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::MonitorDbc::init(const std::shared_ptr<Solid::TimeInt::BaseDataIO>& io_ptr,
    Core::FE::Discretization& discret, Solid::TimeInt::BaseDataGlobalState& gstate, Solid::Dbc& dbc)
{
  issetup_ = false;
  isinit_ = false;

  os_precision_ = io_ptr->get_monitor_dbc_params()->screen_precision();

  std::vector<const Core::Conditions::Condition*> tagged_conds;
  get_tagged_condition(tagged_conds, "Dirichlet", "monitor_reaction", discret);

  // There are no tagged conditions. This indicates that the reaction forces
  // shall not be monitored thus we can leave.
  isempty_ = (tagged_conds.size() == 0);
  if (isempty_)
  {
    isinit_ = true;
    return;
  }

  // copy the information of the tagged Dirichlet condition into a new
  // auxiliary "ReactionForce" condition and build the related geometry
  for (const Core::Conditions::Condition* tagged_cond : tagged_conds)
    create_reaction_force_condition(*tagged_cond, discret);

  // build geometry
  discret.fill_complete({
      .assign_degrees_of_freedom = false,
      .init_elements = false,
      .do_boundary_conditions = true,
  });

  discret_ptr_ = &discret;
  gstate_ptr_ = &gstate;
  dbc_ptr_ = &dbc;

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::MonitorDbc::get_tagged_condition(
    std::vector<const Core::Conditions::Condition*>& tagged_conds, const std::string& cond_name,
    const std::string& tag_name, const Core::FE::Discretization& discret) const
{
  tagged_conds.clear();

  std::vector<std::string> cond_names;
  std::vector<const Core::Conditions::Condition*> cond_vec;
  discret.get_condition(cond_name, cond_vec);

  for (auto& cond_ptr : cond_vec)
  {
    const auto& cptr = cond_ptr->parameters().get<std::string>("TAG");

    if (cptr == tag_name) tagged_conds.push_back(cond_ptr);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int Solid::MonitorDbc::get_unique_id(int tagged_id, Core::Conditions::GeometryType gtype) const
{
  switch (gtype)
  {
    case Core::Conditions::geometry_type_point:
      return tagged_id + 100;
    case Core::Conditions::geometry_type_line:
      return tagged_id + 1000;
    case Core::Conditions::geometry_type_surface:
      return tagged_id + 10000;
    default:
      FOUR_C_THROW("Unsupported geometry type! (enum={})", gtype);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::MonitorDbc::create_reaction_force_condition(
    const Core::Conditions::Condition& tagged_cond, Core::FE::Discretization& discret) const
{
  const int new_id = get_unique_id(tagged_cond.id(), tagged_cond.g_type());

  auto rcond_ptr =
      std::make_shared<Core::Conditions::Condition>(new_id, Core::Conditions::ElementTag, true,
          tagged_cond.g_type(), Core::Conditions::EntityType::legacy_id);

  rcond_ptr->parameters().add("ONOFF", (tagged_cond.parameters().get<std::vector<int>>("ONOFF")));
  rcond_ptr->set_nodes(*tagged_cond.get_nodes());

  discret.set_condition("ReactionForce", rcond_ptr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::MonitorDbc::setup()
{
  throw_if_not_init();

  if (isempty_)
  {
    issetup_ = true;
    return;
  }

  const Teuchos::ParameterList& sublist_IO_monitor_structure_dbc =
      Global::Problem::instance()->io_params().sublist("MONITOR STRUCTURE DBC");
  file_type_ = sublist_IO_monitor_structure_dbc.get<IOMonitorStructureDBC::FileType>("FILE_TYPE");

  std::vector<const Core::Conditions::Condition*> rconds;
  discret_ptr_->get_condition("ReactionForce", rconds);
  for (const auto& rcond_ptr : rconds)
  {
    const Core::Conditions::Condition& rcond = *rcond_ptr;
    auto ipair = react_maps_.insert(
        std::make_pair(rcond.id(), std::vector<std::shared_ptr<Core::LinAlg::Map>>(3, nullptr)));

    if (not ipair.second)
      FOUR_C_THROW("The reaction condition id #{} seems to be non-unique!", rcond.id());

    create_reaction_maps(*discret_ptr_, rcond, ipair.first->second.data());

    switch (file_type_)
    {
      case IOMonitorStructureDBC::FileType::csv:
      {
        dbc_monitor_csvwriter_.emplace_back(std::make_unique<Core::IO::RuntimeCsvWriter>(
            Core::Communication::my_mpi_rank(get_comm()),
            *Global::Problem::instance()->output_control_file(),
            std::to_string(rcond.id() + 1) + "_monitor_dbc"));
        const int csv_precision = 16;
        dbc_monitor_csvwriter_.back()->register_data_vector("ref_area", 1, csv_precision);
        dbc_monitor_csvwriter_.back()->register_data_vector("curr_area", 1, csv_precision);
        dbc_monitor_csvwriter_.back()->register_data_vector("f", DIM, csv_precision);
        dbc_monitor_csvwriter_.back()->register_data_vector("m", DIM, csv_precision);
        break;
      }
      case IOMonitorStructureDBC::FileType::yaml:
      {
        // handle restart
        if (Global::Problem::instance()->restart() and
            Core::Communication::my_mpi_rank(get_comm()) == 0)
        {
          const std::string full_restart_dirpath(
              Global::Problem::instance()->output_control_file()->restart_name());

          const std::string filename =
              full_restart_dirpath + "-" + std::to_string(rcond.id() + 1) + "_monitor_dbc.yaml";

          std::ifstream in(filename, std::ios::binary);
          if (!in)
          {
            FOUR_C_THROW("Could not open file {}", filename);
          }
          std::stringstream ss;
          ss << in.rdbuf();
          std::string file_contents = ss.str();

          c4::csubstr yaml_view{file_contents.c_str(), file_contents.size()};

          dbc_monitor_yaml_file_trees_.emplace_back(ryml::parse_in_arena(yaml_view));

          ryml::NodeRef root = dbc_monitor_yaml_file_trees_.back().rootref();
          auto data_node = root["dbc monitor condition data"];

          std::vector<c4::yml::NodeRef> to_remove;

          for (auto child : data_node.children())
          {
            auto candidate = child["step"];
            if (!candidate.invalid())
            {
              // ensure scalar
              if (!candidate.is_map() && !candidate.is_seq())
              {
                int val = 0;
                candidate >> val;

                if (val > Global::Problem::instance()->restart())
                {
                  to_remove.push_back(child);
                }
              }
            }
          }

          for (auto id : std::ranges::reverse_view(to_remove))
          {
            data_node.remove_child(id);
          }
        }
        break;
      }
    }

    if (sublist_IO_monitor_structure_dbc.get<bool>("WRITE_CONDITION_INFORMATION"))
    {
      FOUR_C_ASSERT_ALWAYS(file_type_ == IOMonitorStructureDBC::FileType::yaml,
          "DBC monitor including condition information ('WRITE_CONDITION_INFORMATION: true') can "
          "only be written to file for the 'yaml' file format!");

      // only needs to be done if simulation is not restarted because in this case the file from
      // the simulation that is restarted is read and on proc 0
      if (!Global::Problem::instance()->restart() and
          Core::Communication::my_mpi_rank(get_comm()) == 0)
      {
        dbc_monitor_yaml_file_trees_.emplace_back(Core::IO::init_yaml_tree_with_exceptions());
        ryml::NodeRef root = dbc_monitor_yaml_file_trees_.back().rootref();

        root |= ryml::MAP;
        auto information_node = root.append_child();
        information_node << ryml::key("dbc monitor condition");
        information_node |= ryml::MAP;
        {
          auto node_gids = information_node.append_child();
          node_gids << ryml::key("node gids");
          node_gids |= ryml::SEQ;
          const auto* node_ids = rcond_ptr->get_nodes();
          for (const auto node_id : *node_ids)
          {
            node_gids.append_child() << node_id;
          }
        }

        // add the data node to be appended later
        root |= ryml::MAP;
        auto data_node = root.append_child();
        data_node << ryml::key("dbc monitor condition data");
        data_node |= ryml::SEQ;
      }
    }
  }

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::MonitorDbc::create_reaction_maps(const Core::FE::Discretization& discret,
    const Core::Conditions::Condition& rcond, std::shared_ptr<Core::LinAlg::Map>* react_maps) const
{
  const auto onoff = rcond.parameters().get<std::vector<int>>("ONOFF");
  const auto* nids = rcond.get_nodes();
  std::vector<int> my_dofs[DIM];
  int ndof = 0;
  for (int i : onoff) ndof += i;

  for (auto& my_dof : my_dofs) my_dof.reserve(nids->size() * ndof);

  MPI_Comm comm = discret.get_comm();
  for (int nid : *nids)
  {
    const int rlid = discret.node_row_map()->lid(nid);
    if (rlid == -1) continue;

    const Core::Nodes::Node* node = discret.l_row_node(rlid);

    for (unsigned i = 0; i < DIM; ++i)
      if (onoff[i] == 1) my_dofs[i].push_back(discret.dof(node, i));
  }

  for (unsigned i = 0; i < DIM; ++i)
    react_maps[i] =
        std::make_shared<Core::LinAlg::Map>(-1, my_dofs[i].size(), my_dofs[i].data(), 0, comm);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::MonitorDbc::execute(Core::IO::DiscretizationWriter& writer)
{
  throw_if_not_init();
  throw_if_not_setup();

  if (isempty_) return;

  std::vector<const Core::Conditions::Condition*> rconds;
  discret_ptr_->get_condition("ReactionForce", rconds);

  std::array<double, 2> area = {0.0, 0.0};
  double& area_ref = area[0];
  double& area_curr = area[1];
  Core::LinAlg::Matrix<DIM, 1> rforce_xyz(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Matrix<DIM, 1> rmoment_xyz(Core::LinAlg::Initialization::uninitialized);

  for (std::size_t condition_counter = 0; condition_counter < rconds.size(); ++condition_counter)
  {
    std::fill_n(area.data(), 2, 0.0);
    std::fill_n(rforce_xyz.data(), DIM, 0.0);
    std::fill_n(rmoment_xyz.data(), DIM, 0.0);

    const auto* const rcond_ptr = rconds[condition_counter];
    const int rid = rcond_ptr->id();
    get_area(area.data(), rcond_ptr);

    get_reaction_force(rforce_xyz, react_maps_[rid].data());
    get_reaction_moment(rmoment_xyz, react_maps_[rid].data(), rcond_ptr);

    std::vector<double> rforce_vec(DIM);
    std::vector<double> rmoment_vec(DIM);
    rforce_vec.assign(rforce_xyz.data(), rforce_xyz.data() + DIM);
    rmoment_vec.assign(rmoment_xyz.data(), rmoment_xyz.data() + DIM);

    switch (file_type_)
    {
      case IOMonitorStructureDBC::FileType::csv:
      {
        std::map<std::string, std::vector<double>> output_data;
        output_data["ref_area"] = {area_ref};
        output_data["curr_area"] = {area_curr};
        output_data["f"] = rforce_vec;
        output_data["m"] = rmoment_vec;
        dbc_monitor_csvwriter_[condition_counter]->write_data_to_file(
            gstate_ptr_->get_time_n(), gstate_ptr_->get_step_n(), output_data);

        break;
      }
      case IOMonitorStructureDBC::FileType::yaml:
      {
        if (Core::Communication::my_mpi_rank(get_comm()) != 0) continue;

        ryml::NodeRef root = dbc_monitor_yaml_file_trees_[condition_counter].rootref();

        auto data_node = root["dbc monitor condition data"];

        auto current_entry = data_node.append_child();
        current_entry |= ryml::MAP;

        current_entry["step"] << gstate_ptr_->get_step_n();
        current_entry["time"] << gstate_ptr_->get_time_n();
        current_entry["f"] << rforce_vec;
        current_entry["m"] << rmoment_vec;
        current_entry["curr_area"] << area_curr;
        current_entry["ref_area"] << area_ref;

        std::string file_name = Global::Problem::instance()->output_control_file()->file_name() +
                                "-" + std::to_string(rid + 1) + "_monitor_dbc.yaml";

        std::ofstream output_filestream(file_name, std::ios::out);
        if (!output_filestream.is_open())
        {
          FOUR_C_THROW("Failed to open file for writing");
        }

        output_filestream << dbc_monitor_yaml_file_trees_[condition_counter];

        break;
      }
    }

    write_results_to_screen(*rcond_ptr, rforce_xyz, rmoment_xyz, area_ref, area_curr);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::MonitorDbc::write_results_to_screen(const Core::Conditions::Condition& rcond,
    const Core::LinAlg::Matrix<DIM, 1>& rforce, const Core::LinAlg::Matrix<DIM, 1>& rmoment,
    const double& area_ref, const double& area_curr) const
{
  if (Core::Communication::my_mpi_rank(get_comm()) != 0) return;

  Core::IO::cout << "\n\n--- Monitor Dirichlet boundary condition " << rcond.id() + 1 << " \n";
  write_condition_header(Core::IO::cout.os(), OS_WIDTH);
  write_column_header(Core::IO::cout.os(), OS_WIDTH);
  write_results(Core::IO::cout.os(), OS_WIDTH, os_precision_, gstate_ptr_->get_step_n(),
      gstate_ptr_->get_time_n(), rforce, rmoment, area_ref, area_curr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::MonitorDbc::write_condition_header(
    std::ostream& os, const int col_width, const Core::Conditions::Condition* cond) const
{
  if (cond)
  {
    cond->print(os);
    os << "\n\n";
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::MonitorDbc::write_column_header(std::ostream& os, const int col_width) const
{
  os << std::setw(col_width) << "step" << std::setw(col_width) << "time" << std::setw(col_width)
     << "ref_area" << std::setw(col_width) << "curr_area" << std::setw(col_width) << "f_x"
     << std::setw(col_width) << "f_y" << std::setw(col_width) << "f_z" << std::setw(col_width)
     << "m_x" << std::setw(col_width) << "m_y" << std::setw(col_width) << "m_z\n";
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::MonitorDbc::write_results(std::ostream& os, const int col_width, const int precision,
    const unsigned step, const double time, const Core::LinAlg::Matrix<DIM, 1>& rforce,
    const Core::LinAlg::Matrix<DIM, 1>& rmoment, const double& area_ref,
    const double& area_curr) const
{
  os << std::setw(col_width) << step << std::setprecision(precision);
  os << std::setw(col_width) << std::scientific << time << std::setw(col_width) << std::scientific
     << area_ref << std::setw(col_width) << std::scientific << area_curr;

  for (unsigned i = 0; i < DIM; ++i) os << std::setw(col_width) << std::scientific << rforce(i, 0);
  for (unsigned i = 0; i < DIM; ++i) os << std::setw(col_width) << std::scientific << rmoment(i, 0);

  os << "\n";
  os << std::flush;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
MPI_Comm Solid::MonitorDbc::get_comm() const { return discret_ptr_->get_comm(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::MonitorDbc::get_area(double area[], const Core::Conditions::Condition* rcond) const
{
  // no area for point DBCs
  if (rcond->g_type() == Core::Conditions::geometry_type_point)
  {
    std::fill(area, area + 2, 0.0);
    return;
  }

  const Core::FE::Discretization& discret =
      dynamic_cast<const Core::FE::Discretization&>(*discret_ptr_);

  enum AreaType : int
  {
    ref = 0,
    curr = 1
  };
  std::array<double, 2> larea = {0.0, 0.0};
  Core::LinAlg::SerialDenseMatrix xyze_ref;
  Core::LinAlg::SerialDenseMatrix xyze_curr;

  const std::map<int, std::shared_ptr<Core::Elements::Element>>& celes = rcond->geometry();
  std::shared_ptr<const Core::LinAlg::Vector<double>> dispn = gstate_ptr_->get_dis_np();
  Core::LinAlg::Vector<double> dispn_col(*discret.dof_col_map(), true);
  Core::LinAlg::export_to(*dispn, dispn_col);

  for (auto& cele_pair : celes)
  {
    const Core::Elements::Element* cele = cele_pair.second.get();
    const Core::Elements::FaceElement* fele =
        dynamic_cast<const Core::Elements::FaceElement*>(cele);
    if (!fele) FOUR_C_THROW("No face element!");

    if (!fele->parent_element() or
        fele->parent_element()->owner() != Core::Communication::my_mpi_rank(discret.get_comm()))
      continue;

    const Core::Nodes::Node* const* fnodes = fele->nodes();
    const unsigned num_fnodes = fele->num_node();
    std::vector<int> fele_dofs;
    fele_dofs.reserve(num_fnodes * DIM);

    for (unsigned i = 0; i < num_fnodes; ++i) discret.dof(fele, fnodes[i], fele_dofs);

    std::vector<double> mydispn = Core::FE::extract_values(dispn_col, fele_dofs);

    xyze_ref.reshape(DIM, num_fnodes);
    xyze_curr.reshape(DIM, num_fnodes);

    for (unsigned i = 0; i < num_fnodes; ++i)
    {
      const Core::Nodes::Node& fnode = *fnodes[i];
      std::copy(fnode.x().data(), fnode.x().data() + DIM, &xyze_ref(0, i));
      std::copy(fnode.x().data(), fnode.x().data() + DIM, &xyze_curr(0, i));

      std::vector<int> ndofs;
      discret.dof(&fnode, ndofs);

      for (unsigned d = 0; d < ndofs.size(); ++d)
      {
        const int ndof = ndofs[d];

        size_t fedof_count = 0;
        for (auto cit = fele_dofs.cbegin(); cit != fele_dofs.cend(); ++cit, ++fedof_count)
        {
          if (*cit == ndof) break;
        }

        if (fedof_count == fele_dofs.size())
          FOUR_C_THROW(
              "Couln't find the face element dof corresponding to the "
              "current node!");

        xyze_curr(d, i) += mydispn[fedof_count];
      }
    }

    larea[AreaType::ref] += Core::Geo::element_area(fele->shape(), xyze_ref);
    larea[AreaType::curr] += Core::Geo::element_area(fele->shape(), xyze_curr);
  }

  std::array<double, 2> garea = Core::Communication::sum_all(larea, discret.get_comm());
  std::ranges::copy(garea, area);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::MonitorDbc::get_reaction_force(Core::LinAlg::Matrix<DIM, 1>& rforce_xyz,
    const std::shared_ptr<Core::LinAlg::Map>* react_maps) const
{
  Core::LinAlg::Vector<double> complete_freact(*gstate_ptr_->get_freact_np());
  dbc_ptr_->rotate_global_to_local(complete_freact);

  Core::LinAlg::Matrix<DIM, 1> lrforce_xyz(Core::LinAlg::Initialization::zero);
  for (unsigned d = 0; d < DIM; ++d)
  {
    std::shared_ptr<Core::LinAlg::Vector<double>> partial_freact_ptr =
        Core::LinAlg::extract_my_vector(complete_freact, *(react_maps[d]));

    double& lrforce_comp = lrforce_xyz(d, 0);
    const double* vals = partial_freact_ptr->get_values();
    for (int i = 0; i < react_maps[d]->num_my_elements(); ++i) lrforce_comp += vals[i];
  }

  rforce_xyz = Core::Communication::sum_all(lrforce_xyz, discret_ptr_->get_comm());
  return rforce_xyz.norm2();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Solid::MonitorDbc::get_reaction_moment(Core::LinAlg::Matrix<DIM, 1>& rmoment_xyz,
    const std::shared_ptr<Core::LinAlg::Map>* react_maps,
    const Core::Conditions::Condition* rcond) const
{
  std::shared_ptr<const Core::LinAlg::Vector<double>> dispn = gstate_ptr_->get_dis_np();

  Core::LinAlg::Vector<double> complete_freact(*gstate_ptr_->get_freact_np());
  dbc_ptr_->rotate_global_to_local(complete_freact);

  Core::LinAlg::Matrix<DIM, 1> lrmoment_xyz(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<DIM, 1> node_reaction_force(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<DIM, 1> node_position(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<DIM, 1> node_reaction_moment(Core::LinAlg::Initialization::zero);
  std::vector<int> node_gid(3);

  const auto onoff = rcond->parameters().get<std::vector<int>>("ONOFF");
  const std::vector<int>* nids = rcond->get_nodes();
  std::vector<int> my_dofs[DIM];
  int ndof = 0;
  for (int i : onoff) ndof += i;

  for (unsigned i = 0; i < DIM; ++i) my_dofs[i].reserve(nids->size() * ndof);

  for (int nid : *nids)
  {
    // Check if the node of the boundary condition is owned by this rank.
    const int rlid = discret_ptr_->node_row_map()->lid(nid);
    if (rlid == -1) continue;

    const Core::Nodes::Node* node = discret_ptr_->l_row_node(rlid);

    for (unsigned i = 0; i < DIM; ++i) node_gid[i] = discret_ptr_->dof(node, i);

    std::vector<double> mydisp = Core::FE::extract_values(*dispn, node_gid);
    for (unsigned i = 0; i < DIM; ++i) node_position(i) = node->x()[i] + mydisp[i];

    // Get the reaction force at this node. This force will only contain non-zero values at the DOFs
    // where the DBC is active.
    node_reaction_force.put_scalar(0.0);
    for (unsigned i = 0; i < DIM; ++i)
    {
      if (onoff[i] == 1)
      {
        const int lid = complete_freact.get_map().lid(node_gid[i]);
        if (lid < 0)
          FOUR_C_THROW("Proc {}: Cannot find gid={} in Core::LinAlg::Vector<double>",
              Core::Communication::my_mpi_rank(complete_freact.get_comm()), node_gid[i]);
        node_reaction_force(i) = complete_freact.local_values_as_span()[lid];
      }
    }

    // Add the moment contribution w.r.t the origin of this reaction force.
    node_reaction_moment.cross_product(node_position, node_reaction_force);
    lrmoment_xyz += node_reaction_moment;
  }

  rmoment_xyz = Core::Communication::sum_all(lrmoment_xyz, discret_ptr_->get_comm());
  return rmoment_xyz.norm2();
}

FOUR_C_NAMESPACE_CLOSE
