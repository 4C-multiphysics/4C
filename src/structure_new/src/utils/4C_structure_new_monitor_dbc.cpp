/*----------------------------------------------------------------------------*/
/*! \file

\brief Monitor tagged Dirichlet boundary conditions

\level 3


*/
/*----------------------------------------------------------------------------*/

#include "4C_structure_new_monitor_dbc.hpp"

#include "4C_discretization_fem_general_extract_values.hpp"
#include "4C_discretization_geometry_element_volume.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_io_every_iteration_writer.hpp"
#include "4C_io_pstream.hpp"
#include "4C_lib_discret.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_structure_new_dbc.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"
#include "4C_structure_new_timint_basedataio.hpp"
#include "4C_structure_new_timint_basedataio_monitor_dbc.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::Init(const Teuchos::RCP<STR::TIMINT::BaseDataIO>& io_ptr,
    DRT::Discretization& discret, STR::TIMINT::BaseDataGlobalState& gstate, STR::Dbc& dbc)
{
  issetup_ = false;
  isinit_ = false;


  of_precision_ = io_ptr->GetMonitorDBCParams()->FilePrecision();
  os_precision_ = io_ptr->GetMonitorDBCParams()->ScreenPrecision();

  std::vector<const CORE::Conditions::Condition*> tagged_conds;
  GetTaggedCondition(tagged_conds, "Dirichlet", "monitor_reaction", discret);

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
  for (const CORE::Conditions::Condition* tagged_cond : tagged_conds)
    create_reaction_force_condition(*tagged_cond, discret);

  // build geometry
  discret.FillComplete(false, false, true);

  discret_ptr_ = &discret;
  gstate_ptr_ = &gstate;
  dbc_ptr_ = &dbc;

  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::GetTaggedCondition(
    std::vector<const CORE::Conditions::Condition*>& tagged_conds, const std::string& cond_name,
    const std::string& tag_name, const DRT::Discretization& discret) const
{
  tagged_conds.clear();

  std::vector<std::string> cond_names;
  std::vector<Teuchos::RCP<CORE::Conditions::Condition>> cond_vec;
  discret.GetCondition(cond_name, cond_vec);

  for (auto& cond_ptr : cond_vec)
  {
    const std::string& cptr = cond_ptr->parameters().Get<std::string>("tag");

    if (cptr == tag_name) tagged_conds.push_back(cond_ptr.get());
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int STR::MonitorDbc::GetUniqueId(int tagged_id, CORE::Conditions::GeometryType gtype) const
{
  switch (gtype)
  {
    case CORE::Conditions::geometry_type_point:
      return tagged_id + 100;
    case CORE::Conditions::geometry_type_line:
      return tagged_id + 1000;
    case CORE::Conditions::geometry_type_surface:
      return tagged_id + 10000;
    default:
      FOUR_C_THROW("Unsupported geometry type! (enum=%d)", gtype);
      exit(EXIT_FAILURE);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::create_reaction_force_condition(
    const CORE::Conditions::Condition& tagged_cond, DRT::Discretization& discret) const
{
  const int new_id = GetUniqueId(tagged_cond.Id(), tagged_cond.GType());

  Teuchos::RCP<CORE::Conditions::Condition> rcond_ptr =
      Teuchos::rcp(new CORE::Conditions::Condition(
          new_id, CORE::Conditions::ElementTag, true, tagged_cond.GType()));

  rcond_ptr->parameters().Add("onoff", (tagged_cond.parameters().Get<std::vector<int>>("onoff")));
  rcond_ptr->SetNodes(*tagged_cond.GetNodes());

  dynamic_cast<DRT::Discretization&>(discret).SetCondition("ReactionForce", rcond_ptr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::Setup()
{
  ThrowIfNotInit();

  const Teuchos::ParameterList& sublist_IO_monitor_structure_dbc =
      GLOBAL::Problem::Instance()->IOParams().sublist("MONITOR STRUCTURE DBC");

  std::string filetype = sublist_IO_monitor_structure_dbc.get<std::string>("FILE_TYPE");

  if (isempty_)
  {
    issetup_ = true;
    return;
  }

  std::vector<Teuchos::RCP<CORE::Conditions::Condition>> rconds;
  discret_ptr_->GetCondition("ReactionForce", rconds);
  for (const Teuchos::RCP<CORE::Conditions::Condition>& rcond_ptr : rconds)
  {
    CORE::Conditions::Condition& rcond = *rcond_ptr;
    auto ipair = react_maps_.insert(
        std::make_pair(rcond.Id(), std::vector<Teuchos::RCP<Epetra_Map>>(3, Teuchos::null)));

    if (not ipair.second)
      FOUR_C_THROW("The reaction condition id #%d seems to be non-unique!", rcond.Id());

    CreateReactionMaps(*discret_ptr_, rcond, ipair.first->second.data());
  }

  // create directory ...
  const std::string full_dirpath(
      GLOBAL::Problem::Instance()->OutputControlFile()->FileName() + "_monitor_dbc");
  const std::string filename_only_prefix(
      GLOBAL::Problem::Instance()->OutputControlFile()->FileNameOnlyPrefix());
  IO::CreateDirectory(full_dirpath, Comm().MyPID());
  // ... create files paths ...
  full_filepaths_ = CreateFilePaths(rconds, full_dirpath, filename_only_prefix, filetype);
  // ... clear them and write header
  clear_files_and_write_header(rconds, full_filepaths_,
      CORE::UTILS::IntegralValue<int>(sublist_IO_monitor_structure_dbc, "WRITE_HEADER"));

  // handle restart
  if (GLOBAL::Problem::Instance()->Restart())
  {
    const std::string full_restart_dirpath(
        GLOBAL::Problem::Instance()->OutputControlFile()->RestartName() + "_monitor_dbc");
    const std::string filename_restart_only_prefix(
        IO::ExtractFileName(GLOBAL::Problem::Instance()->OutputControlFile()->RestartName()));

    std::vector<std::string> full_restart_filepaths =
        CreateFilePaths(rconds, full_restart_dirpath, filename_restart_only_prefix, filetype);

    read_results_prior_restart_step_and_write_to_file(
        full_restart_filepaths, gstate_ptr_->GetStepN());
  }

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::CreateReactionMaps(const DRT::Discretization& discret,
    const CORE::Conditions::Condition& rcond, Teuchos::RCP<Epetra_Map>* react_maps) const
{
  const auto* onoff = &rcond.parameters().Get<std::vector<int>>("onoff");
  const auto* nids = rcond.GetNodes();
  std::vector<int> my_dofs[DIM];
  int ndof = 0;
  for (int i : *onoff) ndof += i;

  for (auto& my_dof : my_dofs) my_dof.reserve(nids->size() * ndof);

  const Epetra_Comm& comm = discret.Comm();
  for (int nid : *nids)
  {
    const int rlid = discret.NodeRowMap()->LID(nid);
    if (rlid == -1) continue;

    const DRT::Node* node = discret.lRowNode(rlid);

    for (unsigned i = 0; i < DIM; ++i)
      if ((*onoff)[i] == 1) my_dofs[i].push_back(discret.Dof(node, i));
  }

  for (unsigned i = 0; i < DIM; ++i)
    react_maps[i] = Teuchos::rcp(new Epetra_Map(-1, my_dofs[i].size(), my_dofs[i].data(), 0, comm));
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::read_results_prior_restart_step_and_write_to_file(
    const std::vector<std::string>& full_restart_filepaths, int restart_step) const
{
  if (Comm().MyPID() != 0) return;

  if (full_restart_filepaths.size() != full_filepaths_.size())
    FOUR_C_THROW(
        " Your monitoring of dbc's has changed after restart, this is not supported right now");

  for (unsigned int i = 0; i < full_restart_filepaths.size(); ++i)
  {
    std::stringstream section_prior_restart;
    std::ifstream restart_file;
    restart_file.open(full_restart_filepaths[i].c_str(), std::ios_base::out);

    // check if file was found
    if (not restart_file)
      FOUR_C_THROW(" restart file for monitoring structure dbcs could not be found");

    // loop over lines of restarted collection file
    std::string line;
    bool at_numerics = false;
    while (std::getline(restart_file, line))
    {
      if ((not at_numerics) and (line.find("step", 0) != std::string::npos))
      {
        at_numerics = true;
        continue;
      }

      // found line with timestep
      if (at_numerics)
      {
        // get time step of current line
        int readtime = std::atof(line.substr(0, OF_WIDTH).c_str());

        if (readtime <= restart_step)
          section_prior_restart << line << "\n";
        else
          break;
      }
    }

    // write to file
    std::ofstream of(full_filepaths_[i], std::ios_base::out | std::ios_base::app);
    of << section_prior_restart.str();
    of.close();

    restart_file.close();
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::Execute(IO::DiscretizationWriter& writer)
{
  ThrowIfNotInit();
  ThrowIfNotSetup();

  if (isempty_) return;

  std::vector<Teuchos::RCP<CORE::Conditions::Condition>> rconds;
  discret_ptr_->GetCondition("ReactionForce", rconds);

  std::array<double, 2> area = {0.0, 0.0};
  double& area_ref = area[0];
  double& area_curr = area[1];
  CORE::LINALG::Matrix<DIM, 1> rforce_xyz(false);
  CORE::LINALG::Matrix<DIM, 1> rmoment_xyz(false);

  auto filepath = full_filepaths_.cbegin();
  for (const Teuchos::RCP<CORE::Conditions::Condition>& rcond_ptr : rconds)
  {
    std::fill(area.data(), area.data() + 2, 0.0);
    std::fill(rforce_xyz.A(), rforce_xyz.A() + DIM, 0.0);
    std::fill(rmoment_xyz.A(), rmoment_xyz.A() + DIM, 0.0);

    const int rid = rcond_ptr->Id();
    GetArea(area.data(), rcond_ptr.get());

    GetReactionForce(rforce_xyz, react_maps_[rid].data());
    GetReactionMoment(rmoment_xyz, react_maps_[rid].data(), rcond_ptr.get());

    WriteResultsToFile(*(filepath++), rforce_xyz, rmoment_xyz, area_ref, area_curr);
    write_results_to_screen(rcond_ptr, rforce_xyz, rmoment_xyz, area_ref, area_curr);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::WriteResultsToFile(const std::string& full_filepath,
    const CORE::LINALG::Matrix<DIM, 1>& rforce, const CORE::LINALG::Matrix<DIM, 1>& rmoment,
    const double& area_ref, const double& area_curr) const
{
  if (Comm().MyPID() != 0) return;

  std::ofstream of(full_filepath, std::ios_base::out | std::ios_base::app);

  WriteResults(of, OF_WIDTH, of_precision_, gstate_ptr_->GetStepN(), gstate_ptr_->GetTimeN(),
      rforce, rmoment, area_ref, area_curr);

  of.close();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::write_results_to_screen(
    const Teuchos::RCP<CORE::Conditions::Condition>& rcond_ptr,
    const CORE::LINALG::Matrix<DIM, 1>& rforce, const CORE::LINALG::Matrix<DIM, 1>& rmoment,
    const double& area_ref, const double& area_curr) const
{
  if (Comm().MyPID() != 0) return;

  IO::cout << "\n\n--- Monitor Dirichlet boundary condition " << rcond_ptr->Id() + 1 << " \n";
  write_condition_header(IO::cout.os(), OS_WIDTH);
  WriteColumnHeader(IO::cout.os(), OS_WIDTH);
  WriteResults(IO::cout.os(), OS_WIDTH, os_precision_, gstate_ptr_->GetStepN(),
      gstate_ptr_->GetTimeN(), rforce, rmoment, area_ref, area_curr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::vector<std::string> STR::MonitorDbc::CreateFilePaths(
    const std::vector<Teuchos::RCP<CORE::Conditions::Condition>>& rconds,
    const std::string& full_dirpath, const std::string& filename_only_prefix,
    const std::string& file_type) const
{
  std::vector<std::string> full_filepaths(rconds.size());

  if (Comm().MyPID() != 0) return full_filepaths;

  size_t i = 0;
  for (const Teuchos::RCP<CORE::Conditions::Condition>& rcond : rconds)
    full_filepaths[i++] = full_dirpath + "/" + filename_only_prefix + "_" +
                          std::to_string(rcond->Id() + 1) + "_monitor_dbc." + file_type;

  return full_filepaths;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::clear_files_and_write_header(
    const std::vector<Teuchos::RCP<CORE::Conditions::Condition>>& rconds,
    std::vector<std::string>& full_filepaths, bool do_write_condition_header)
{
  if (Comm().MyPID() != 0) return;

  size_t i = 0;
  for (const Teuchos::RCP<CORE::Conditions::Condition>& rcond : rconds)
  {
    // clear old content
    std::ofstream of(full_filepaths[i], std::ios_base::out);
    if (do_write_condition_header) write_condition_header(of, OF_WIDTH, rcond.get());
    WriteColumnHeader(of, OF_WIDTH);
    of.close();
    ++i;
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::write_condition_header(
    std::ostream& os, const int col_width, const CORE::Conditions::Condition* cond) const
{
  if (cond)
  {
    cond->Print(os);
    os << "\n\n";
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::WriteColumnHeader(std::ostream& os, const int col_width) const
{
  os << std::setw(col_width) << "step" << std::setw(col_width) << "time" << std::setw(col_width)
     << "ref_area" << std::setw(col_width) << "curr_area" << std::setw(col_width) << "f_x"
     << std::setw(col_width) << "f_y" << std::setw(col_width) << "f_z" << std::setw(col_width)
     << "m_x" << std::setw(col_width) << "m_y" << std::setw(col_width) << "m_z\n";
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::WriteResults(std::ostream& os, const int col_width, const int precision,
    const unsigned step, const double time, const CORE::LINALG::Matrix<DIM, 1>& rforce,
    const CORE::LINALG::Matrix<DIM, 1>& rmoment, const double& area_ref,
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
const Epetra_Comm& STR::MonitorDbc::Comm() const { return discret_ptr_->Comm(); }

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void STR::MonitorDbc::GetArea(double area[], const CORE::Conditions::Condition* rcond) const
{
  // no area for point DBCs
  if (rcond->GType() == CORE::Conditions::geometry_type_point)
  {
    std::fill(area, area + 2, 0.0);
    return;
  }

  const DRT::Discretization& discret = dynamic_cast<const DRT::Discretization&>(*discret_ptr_);

  enum AreaType : int
  {
    ref = 0,
    curr = 1
  };
  std::array<double, 2> larea = {0.0, 0.0};
  CORE::LINALG::SerialDenseMatrix xyze_ref;
  CORE::LINALG::SerialDenseMatrix xyze_curr;

  const std::map<int, Teuchos::RCP<DRT::Element>>& celes = rcond->Geometry();
  Teuchos::RCP<const Epetra_Vector> dispn = gstate_ptr_->GetDisNp();
  Epetra_Vector dispn_col(*discret.DofColMap(), true);
  CORE::LINALG::Export(*dispn, dispn_col);

  for (auto& cele_pair : celes)
  {
    const DRT::Element* cele = cele_pair.second.get();
    const DRT::FaceElement* fele = dynamic_cast<const DRT::FaceElement*>(cele);
    if (!fele) FOUR_C_THROW("No face element!");

    if (!fele->ParentElement() or fele->ParentElement()->Owner() != discret.Comm().MyPID())
      continue;

    const DRT::Node* const* fnodes = fele->Nodes();
    const unsigned num_fnodes = fele->NumNode();
    std::vector<int> fele_dofs;
    fele_dofs.reserve(num_fnodes * DIM);

    for (unsigned i = 0; i < num_fnodes; ++i) discret.Dof(fele, fnodes[i], fele_dofs);

    std::vector<double> mydispn;
    CORE::FE::ExtractMyValues(dispn_col, mydispn, fele_dofs);

    xyze_ref.reshape(DIM, num_fnodes);
    xyze_curr.reshape(DIM, num_fnodes);

    for (unsigned i = 0; i < num_fnodes; ++i)
    {
      const DRT::Node& fnode = *fnodes[i];
      std::copy(fnode.X().data(), fnode.X().data() + DIM, &xyze_ref(0, i));
      std::copy(fnode.X().data(), fnode.X().data() + DIM, &xyze_curr(0, i));

      std::vector<int> ndofs;
      discret.Dof(&fnode, ndofs);

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

    larea[AreaType::ref] += CORE::GEO::ElementArea(fele->Shape(), xyze_ref);
    larea[AreaType::curr] += CORE::GEO::ElementArea(fele->Shape(), xyze_curr);
  }

  discret.Comm().SumAll(larea.data(), area, 2);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double STR::MonitorDbc::GetReactionForce(
    CORE::LINALG::Matrix<DIM, 1>& rforce_xyz, const Teuchos::RCP<Epetra_Map>* react_maps) const
{
  Epetra_Vector complete_freact(*gstate_ptr_->GetFreactNp());
  dbc_ptr_->RotateGlobalToLocal(Teuchos::rcpFromRef(complete_freact));

  CORE::LINALG::Matrix<DIM, 1> lrforce_xyz(true);
  for (unsigned d = 0; d < DIM; ++d)
  {
    Teuchos::RCP<Epetra_Vector> partial_freact_ptr =
        CORE::LINALG::ExtractMyVector(complete_freact, *(react_maps[d]));

    double& lrforce_comp = lrforce_xyz(d, 0);
    const double* vals = partial_freact_ptr->Values();
    for (int i = 0; i < react_maps[d]->NumMyElements(); ++i) lrforce_comp += vals[i];
  }

  discret_ptr_->Comm().SumAll(lrforce_xyz.A(), rforce_xyz.A(), DIM);
  return rforce_xyz.Norm2();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double STR::MonitorDbc::GetReactionMoment(CORE::LINALG::Matrix<DIM, 1>& rmoment_xyz,
    const Teuchos::RCP<Epetra_Map>* react_maps, const CORE::Conditions::Condition* rcond) const
{
  Teuchos::RCP<const Epetra_Vector> dispn = gstate_ptr_->GetDisNp();

  Epetra_Vector complete_freact(*gstate_ptr_->GetFreactNp());
  dbc_ptr_->RotateGlobalToLocal(Teuchos::rcpFromRef(complete_freact));

  CORE::LINALG::Matrix<DIM, 1> lrmoment_xyz(true);
  CORE::LINALG::Matrix<DIM, 1> node_reaction_force(true);
  CORE::LINALG::Matrix<DIM, 1> node_position(true);
  CORE::LINALG::Matrix<DIM, 1> node_reaction_moment(true);
  std::vector<int> node_gid(3);

  const auto* onoff = &rcond->parameters().Get<std::vector<int>>("onoff");
  const std::vector<int>* nids = rcond->GetNodes();
  std::vector<int> my_dofs[DIM];
  int ndof = 0;
  for (int i : *onoff) ndof += i;

  for (unsigned i = 0; i < DIM; ++i) my_dofs[i].reserve(nids->size() * ndof);

  for (int nid : *nids)
  {
    // Check if the node of the boundary condition is owned by this rank.
    const int rlid = discret_ptr_->NodeRowMap()->LID(nid);
    if (rlid == -1) continue;

    const DRT::Node* node = discret_ptr_->lRowNode(rlid);

    for (unsigned i = 0; i < DIM; ++i) node_gid[i] = discret_ptr_->Dof(node, i);

    std::vector<double> mydisp;
    CORE::FE::ExtractMyValues(*dispn, mydisp, node_gid);
    for (unsigned i = 0; i < DIM; ++i) node_position(i) = node->X()[i] + mydisp[i];

    // Get the reaction force at this node. This force will only contain non-zero values at the DOFs
    // where the DBC is active.
    node_reaction_force.PutScalar(0.0);
    for (unsigned i = 0; i < DIM; ++i)
    {
      if ((*onoff)[i] == 1)
      {
        const int lid = complete_freact.Map().LID(node_gid[i]);
        if (lid < 0)
          FOUR_C_THROW("Proc %d: Cannot find gid=%d in Epetra_Vector",
              complete_freact.Comm().MyPID(), node_gid[i]);
        node_reaction_force(i) = complete_freact[lid];
      }
    }

    // Add the moment contribution w.r.t the origin of this reaction force.
    node_reaction_moment.CrossProduct(node_position, node_reaction_force);
    lrmoment_xyz += node_reaction_moment;
  }

  discret_ptr_->Comm().SumAll(lrmoment_xyz.A(), rmoment_xyz.A(), DIM);
  return rmoment_xyz.Norm2();
}

FOUR_C_NAMESPACE_CLOSE
