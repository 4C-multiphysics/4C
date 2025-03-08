// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_utils.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_io_every_iteration_writer.hpp"
#include "4C_linalg_serialdensematrix.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <fstream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::string CONTACT::vec_block_type_to_str(const CONTACT::VecBlockType bt)
{
  switch (bt)
  {
    case VecBlockType::displ:
      return "displ";
    case VecBlockType::temp:
      return "temp";
    case VecBlockType::scatra:
      return "scatra";
    case VecBlockType::constraint:
      return "constraint";
    case VecBlockType::elch:
      return "elch";
    default:
      FOUR_C_THROW("Unknown block type %d", bt);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int CONTACT::Utils::get_contact_conditions(
    std::vector<Core::Conditions::Condition*>& contact_conditions,
    const std::vector<Core::Conditions::Condition*>& beamandsolidcontactconditions,
    const bool& throw_error)
{
  /* Sort out beam-to-solid contact pairs, since these are treated in the
   * beam3contact framework */
  for (auto* beamandsolidcontactcondition : beamandsolidcontactconditions)
  {
    if ((beamandsolidcontactcondition->parameters().get<std::string>("Application")) !=
        "Beamtosolidcontact")
    {
      contact_conditions.push_back(beamandsolidcontactcondition);
    }
  }

  /* There must be more than one contact condition unless we have a self
   * contact problem! */
  if (contact_conditions.size() < 1)
  {
    if (throw_error) FOUR_C_THROW("Not enough contact conditions in discretization");
    return -1;
  }
  if (contact_conditions.size() == 1)
  {
    const auto& side = contact_conditions[0]->parameters().get<std::string>("Side");
    if (side != "Selfcontact")
    {
      if (throw_error) FOUR_C_THROW("Not enough contact conditions in discretization");
      return -2;
    }
  }
  // everything worked fine
  return 0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
int CONTACT::Utils::get_contact_condition_groups(
    std::vector<std::vector<Core::Conditions::Condition*>>& ccond_grps,
    const Core::FE::Discretization& discret, const bool& throw_error)
{
  // vector that contains solid-to-solid and beam-to-solid contact pairs
  std::vector<Core::Conditions::Condition*> beamandsolidcontactconditions(0);
  discret.get_condition("Contact", beamandsolidcontactconditions);

  std::vector<Core::Conditions::Condition*> cconds(0);
  int err =
      CONTACT::Utils::get_contact_conditions(cconds, beamandsolidcontactconditions, throw_error);
  // direct return, if an error occurred
  if (err) return err;
  CONTACT::Utils::get_contact_condition_groups(ccond_grps, cconds);
  return 0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Utils::get_contact_condition_groups(
    std::vector<std::vector<Core::Conditions::Condition*>>& ccond_grps,
    const std::vector<Core::Conditions::Condition*>& cconds)
{
  ccond_grps.clear();
  /* find all pairs of matching contact conditions
   * there is a maximum of (conditions / 2) groups */
  std::vector<int> found_grps(0);

  for (std::size_t i = 0; i < cconds.size(); ++i)
  {
    std::vector<Core::Conditions::Condition*> current_grp(0);
    Core::Conditions::Condition* tempcond = nullptr;

    // try to build contact group around this condition
    current_grp.push_back(cconds[i]);
    const auto groupid1 = current_grp[0]->parameters().get<int>("InterfaceID");
    bool foundit = false;

    // only one surface per group is ok for self contact
    const auto& side = cconds[i]->parameters().get<std::string>("Side");
    if (side == "Selfcontact") foundit = true;

    for (std::size_t j = 0; j < cconds.size(); ++j)
    {
      // do not compare ids of one and the same contact condition
      if (j == i) continue;
      tempcond = cconds[j];
      const auto groupid2 = tempcond->parameters().get<int>("InterfaceID");

      // Do the IDs coincide?
      if (groupid1 != groupid2) continue;  // not in the group
      foundit = true;                      // found a group entry
      current_grp.push_back(tempcond);     // store it in the current group
    }

    // now we should have found a group of conditions
    if (!foundit) FOUR_C_THROW("Cannot find matching contact condition for id %i", groupid1);

    // see whether we found this group before
    bool foundbefore = false;
    for (int found_grp : found_grps)
    {
      if (groupid1 == found_grp)
      {
        foundbefore = true;
        break;
      }
    }

    // if we have processed this group before, do nothing
    if (foundbefore) continue;

    // we have not found this group before, store it
    found_grps.push_back(groupid1);

    // store the new unique group of contact conditions
    ccond_grps.push_back(current_grp);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Utils::get_master_slave_side_info(std::vector<bool>& isslave,
    std::vector<bool>& isself, const std::vector<Core::Conditions::Condition*>& cond_grp)
{
  bool hasslave = false;
  bool hasmaster = false;
  bool hasself = false;
  std::vector<const std::string*> sides(cond_grp.size());
  // safety...
  isslave.clear();
  isslave.resize(cond_grp.size(), false);
  isself.clear();
  isself.resize(cond_grp.size(), false);

  for (int j = 0; j < (int)sides.size(); ++j)
  {
    sides[j] = &cond_grp[j]->parameters().get<std::string>("Side");
    if (*sides[j] == "Slave")
    {
      hasslave = true;
      isslave[j] = true;
      isself[j] = false;
    }
    else if (*sides[j] == "Master")
    {
      hasmaster = true;
      isslave[j] = false;
      isself[j] = false;
    }
    else if (*sides[j] == "Selfcontact")
    {
      hasmaster = true;
      hasslave = true;
      hasself = true;
      isslave[j] = false;
      isself[j] = true;
    }
    else
    {
      FOUR_C_THROW("Unknown contact side qualifier!");
    }
  }

  if (!hasslave) FOUR_C_THROW("Slave side missing in contact condition group!");
  if (!hasmaster) FOUR_C_THROW("Master side missing in contact condition group!");

  // check for self contact group
  if (hasself)
  {
    for (auto&& j : isself)
    {
      if (!j)
      {
        FOUR_C_THROW(
            "ERROR: Inconsistent definition of self contact condition group! You defined one "
            "condition as 'Selfcontact' condition. So all other contact conditions with same ID "
            "need to be defined as 'Selfcontact' as well!");
      }
    }
  }
}

/*----------------------------------------------------------------------------*
 | gather initialization information                            schmidt 11/18 |
 *----------------------------------------------------------------------------*/
void CONTACT::Utils::get_initialization_info(bool& Two_half_pass,
    bool& Check_nonsmooth_selfcontactsurface, bool& Searchele_AllProc, std::vector<bool>& isactive,
    std::vector<bool>& isslave, std::vector<bool>& isself,
    const std::vector<Core::Conditions::Condition*>& cond_grp)
{
  std::vector<const std::string*> active(cond_grp.size());
  std::vector<int> two_half_pass(cond_grp.size());
  std::vector<int> check_nonsmooth_selfcontactsurface(cond_grp.size());

  for (std::size_t j = 0; j < cond_grp.size(); ++j)
  {
    active[j] = &cond_grp[j]->parameters().get<std::string>("Initialization");
    if (isslave[j])
    {
      // slave sides may be initialized as "Active" or as "Inactive"
      if (*active[j] == "Active")
        isactive[j] = true;
      else if (*active[j] == "Inactive")
        isactive[j] = false;
      else
        FOUR_C_THROW("Unknown contact init qualifier!");
    }
    else if (isself[j])
    {
      // self contact surf must NOT be initialized as "Active" as this makes no sense
      if (*active[j] == "Active")
        FOUR_C_THROW("Selfcontact surface cannot be active!");
      else if (*active[j] == "Inactive")
        isactive[j] = false;
      else
        FOUR_C_THROW("Unknown contact init qualifier!");
    }
    else
    {
      // master sides must NOT be initialized as "Active" as this makes no sense
      if (*active[j] == "Active")
        FOUR_C_THROW("Master side cannot be active!");
      else if (*active[j] == "Inactive")
        isactive[j] = false;
      else
        FOUR_C_THROW("Unknown contact init qualifier!");
    }

    // check for two half pass approach
    two_half_pass[j] = cond_grp[j]->parameters().get<double>("TwoHalfPass");
    if (two_half_pass[j]) Two_half_pass = true;

    // check for reference configuration check for non-smooth self contact surfaces
    check_nonsmooth_selfcontactsurface[j] =
        cond_grp[j]->parameters().get<double>("RefConfCheckNonSmoothSelfContactSurface");
    if (check_nonsmooth_selfcontactsurface[j]) Check_nonsmooth_selfcontactsurface = true;
  }

  // SAFETY CHECKS
  // read parameter list and problem type
  const Core::ProblemType problemtype = Global::Problem::instance()->get_problem_type();
  const Teuchos::ParameterList& contact = Global::Problem::instance()->contact_dynamic_params();
  const Teuchos::ParameterList& mortar = Global::Problem::instance()->mortar_coupling_params();

  // XFSI is the only reason why you want this option (as the xfluid redistribution is different)
  if (problemtype == Core::ProblemType::fsi_xfem || problemtype == Core::ProblemType::fpsi_xfem)
    Searchele_AllProc = true;
  else
    Searchele_AllProc = false;

  // all definitions of one interface need to be consistent
  if (Two_half_pass)
  {
    for (int is_two_half_pass : two_half_pass)
    {
      if (!is_two_half_pass)
      {
        FOUR_C_THROW(
            "ERROR: Inconsistent definition of contact condition group! You set the 'TwoHalfPass' "
            "to true for at least one condition. So all other contact conditions with same ID need "
            "to be defined accordingly!");
      }
    }

    for (unsigned j = 0; j < cond_grp.size(); ++j)
    {
      if (!isself[j])
      {
        FOUR_C_THROW(
            "Setting 'TwoHalfPass' to true is only reasonable in combination with self contact so "
            "far!");
      }

      if (Check_nonsmooth_selfcontactsurface && (!check_nonsmooth_selfcontactsurface[j]))
      {
        FOUR_C_THROW(
            "ERROR: Inconsistent definition of contact condition group! You set the "
            "'RefConfCheckNonSmoothSelfContactSurface' to true for at least one condition. So all "
            "other contact conditions with same ID need to be defined accordingly!");
      }
    }

    if ((problemtype != Core::ProblemType::structure) and
        (problemtype != Core::ProblemType::fsi_xfem) and
        (problemtype != Core::ProblemType::fpsi_xfem) and (problemtype != Core::ProblemType::ssi))
      FOUR_C_THROW(
          "two half pass algorithm only implemented in structural, fsi/fpsi and ssi problems");
    if (Teuchos::getIntegralValue<Inpar::CONTACT::SolvingStrategy>(contact, "STRATEGY") !=
        Inpar::CONTACT::solution_nitsche)
      FOUR_C_THROW("two half pass algorithm only with nitsche contact formulation");
    if (Teuchos::getIntegralValue<Inpar::CONTACT::NitscheWeighting>(contact, "NITSCHE_WEIGHTING") !=
        Inpar::CONTACT::NitWgt_harmonic)
      FOUR_C_THROW("two half pass algorithm only with harmonic weighting");
  }

  if (!Two_half_pass && problemtype == Core::ProblemType::fsi_xfem)
    FOUR_C_THROW("Nitsche FSI with Contact requires Two_half_pass which is not set!");

  if ((!Two_half_pass) && Check_nonsmooth_selfcontactsurface)
  {
    FOUR_C_THROW(
        "ERROR: 'RefConfCheckNonSmoothSelfContactSurface' is activated, which is only reasonable "
        "for non-smooth self contact surfaces in combination with the two half pass 'TwoHalfPass' "
        "approach so far!");
  }

  if (Two_half_pass && (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(
                            mortar, "ALGORITHM") != Inpar::Mortar::algorithm_gpts))
  {
    FOUR_C_THROW(
        "ERROR: You activated the two half pass 'TwoHalfPass' approach, but the 'MORTAR COUPLING' "
        "Algorithm is NOT 'GPTS'!");
  }


  if (Check_nonsmooth_selfcontactsurface && (!contact.get<bool>("NONSMOOTH_CONTACT_SURFACE")))
  {
    FOUR_C_THROW(
        "ERROR: You activated the self contact condition reference configuration check for "
        "non-smooth contact surfaces, but flag 'NONSMOOTH_CONTACT_SURFACE' in the 'CONTACT "
        "DYNAMIC' section is not true!");
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Utils::write_conservation_data_to_file(const int mypid, const int interface_id,
    const int nln_iter, const Core::LinAlg::SerialDenseMatrix& conservation_data,
    const std::string& ofile_path, const std::string& prefix)
{
  if (mypid != 0) return;

  static std::vector<std::string> done_prefixes;

  const std::string path(Core::IO::extract_path(ofile_path));
  const std::string dir_name(
      Core::IO::remove_restart_step_from_file_name(
          Core::IO::extract_file_name(ofile_path), Global::Problem::instance()->restart()) +
      "_conservation");

  std::string full_filepath(path + dir_name);
  Core::IO::create_directory(full_filepath, mypid);
  full_filepath += "/" + prefix + "_" + "conservation.data";

  bool is_done = false;
  for (const std::string& done_prefix : done_prefixes)
  {
    if (done_prefix == prefix)
    {
      is_done = true;
      break;
    }
  }

  // first attempt: clear file content and write header
  if (not is_done)
  {
    done_prefixes.push_back(prefix);

    std::ofstream of(full_filepath, std::ios_base::out);
    of << std::setw(24) << "it" << std::setw(24) << "interface" << std::setw(24) << "Fsl_X"
       << std::setw(24) << "Fsl_Y" << std::setw(24) << "Fsl_Z" << std::setw(24) << "Fma_X"
       << std::setw(24) << "Fma_Y" << std::setw(24) << "Fma_Z" << std::setw(24) << "Fb_X"
       << std::setw(24) << "Fb_Y" << std::setw(24) << "Fb_Z" << std::setw(24) << "Mosl_X"
       << std::setw(24) << "Mosl_Y" << std::setw(24) << "Mosl_Z" << std::setw(24) << "Moma_X"
       << std::setw(24) << "Moma_Y" << std::setw(24) << "Moma_Z" << std::setw(24) << "Mob_X"
       << std::setw(24) << "Mob_Y" << std::setw(24) << "Mob_Z\n";
    of.close();
  }

  std::ofstream of(full_filepath, std::ios_base::out | std::ios_base::app);

  if (conservation_data.numRows() < 18)
    FOUR_C_THROW("The conservation_data has insufficient size!");

  of << std::setw(24) << nln_iter << std::setw(24) << interface_id;
  of << std::setprecision(16);
  for (int i = 0; i < conservation_data.numRows(); ++i)
  {
    of << std::setw(24) << std::setw(24) << std::scientific << conservation_data(i, 0);
  }
  of << "\n";
  of.close();
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Utils::DbcHandler::detect_dbc_slave_nodes_and_elements(
    const Core::FE::Discretization& str_discret,
    const std::vector<std::vector<Core::Conditions::Condition*>>& ccond_grps,
    std::set<const Core::Nodes::Node*>& dbc_slave_nodes,
    std::set<const Core::Elements::Element*>& dbc_slave_eles)
{
  dbc_slave_nodes.clear();
  dbc_slave_eles.clear();

  std::map<const Core::Nodes::Node*, int> dbc_slave_node_map;

  std::vector<const Core::Conditions::Condition*> sl_conds;

  for (const auto& ccond_grp : ccond_grps)
  {
    std::vector<bool> isslave;
    std::vector<bool> isself;
    CONTACT::Utils::get_master_slave_side_info(isslave, isself, ccond_grp);

    for (unsigned i = 0; i < ccond_grp.size(); ++i)
    {
      if (not isslave[i]) continue;

      const Core::Conditions::Condition* sl_cond = ccond_grp[i];

      const auto dbc_handling =
          sl_cond->parameters().get<Inpar::Mortar::DBCHandling>("DbcHandling");
      switch (dbc_handling)
      {
        case Inpar::Mortar::DBCHandling::RemoveDBCSlaveNodes:
        {
          sl_conds.push_back(sl_cond);
          break;
        }
        case Inpar::Mortar::DBCHandling::DoNothing:
        {
          break;
        }
        default:
          FOUR_C_THROW("Unknown dbc_handlin enum %d", dbc_handling);
      }
    }
  }

  detect_dbc_slave_nodes(dbc_slave_node_map, str_discret, sl_conds);

  for (auto& dbc_slave_node_pair : dbc_slave_node_map)
    dbc_slave_nodes.insert(dbc_slave_node_pair.first);

  detect_dbc_slave_elements(dbc_slave_eles, dbc_slave_node_map, sl_conds);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Utils::DbcHandler::detect_dbc_slave_nodes(
    std::map<const Core::Nodes::Node*, int>& dbc_slave_node_map,
    const Core::FE::Discretization& str_discret,
    const std::vector<const Core::Conditions::Condition*>& sl_conds)
{
  std::vector<Core::Conditions::Condition*> dconds;
  str_discret.get_condition("Dirichlet", dconds);

  // collect all slave node ids
  std::vector<std::pair<int, int>> slnodeids;
  for (const auto& sl_cond : sl_conds)
  {
    const auto* sl_nids = sl_cond->get_nodes();
    slnodeids.reserve(slnodeids.size() + sl_nids->size());
    for (int sl_nid : *sl_nids) slnodeids.emplace_back(sl_nid, sl_cond->id());
  }

  for (std::pair<int, int> slpair : slnodeids)
  {
    const int snid = slpair.first;

    bool found = false;

    for (Core::Conditions::Condition* dcond : dconds)
    {
      const auto* dnids = dcond->get_nodes();
      for (int dnid : *dnids)
      {
        if (snid == dnid)
        {
          found = true;
          break;
        }
      }
      if (found) break;
    }

    // skip non dbc nodes
    if (not found) continue;

    // skip nullptr ptrs
    if (str_discret.have_global_node(snid))
    {
      const Core::Nodes::Node* node = str_discret.g_node(snid);
      dbc_slave_node_map.insert(std::make_pair(node, slpair.second));
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void CONTACT::Utils::DbcHandler::detect_dbc_slave_elements(
    std::set<const Core::Elements::Element*>& dbc_slave_eles,
    const std::map<const Core::Nodes::Node*, int>& dbc_slave_nodes,
    const std::vector<const Core::Conditions::Condition*>& sl_conds)
{
  for (const auto& dbc_sl_node : dbc_slave_nodes)
  {
    const int slnid = dbc_sl_node.first->id();
    const int slcond_id = dbc_sl_node.second;

    auto sl_citer = sl_conds.cbegin();
    while (sl_citer != sl_conds.cend())
    {
      if ((*sl_citer)->id() == slcond_id) break;

      ++sl_citer;
    }
    const Core::Conditions::Condition& slcond = **sl_citer;

    const std::map<int, std::shared_ptr<Core::Elements::Element>>& geometry = slcond.geometry();
    for (const auto& iele_pair : geometry)
    {
      const Core::Elements::Element* ele = iele_pair.second.get();

      const int* ele_nids = ele->node_ids();
      for (int i = 0; i < ele->num_node(); ++i)
      {
        const int ele_nid = ele_nids[i];

        if (ele_nid == slnid) dbc_slave_eles.insert(ele);
      }
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
