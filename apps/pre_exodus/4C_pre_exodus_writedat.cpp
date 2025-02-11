// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_pre_exodus_writedat.hpp"

#include "4C_fem_condition_definition.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_inpar_validconditions.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_pre_exodus_reader.hpp"

#include <fstream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
int EXODUS::write_dat_file(const std::string& datfile, const EXODUS::Mesh& mymesh,
    const std::string& headfile, const std::vector<EXODUS::ElemDef>& eledefs,
    const std::vector<EXODUS::CondDef>& condefs)
{
  // open datfile
  std::ofstream dat(datfile.c_str());
  if (!dat) FOUR_C_THROW("failed to open file: %s", datfile.c_str());

  // write dat-file intro
  EXODUS::write_dat_intro(headfile, mymesh, dat);

  // write "header"
  EXODUS::write_dat_head(headfile, dat);

  // write conditions
  EXODUS::write_dat_conditions(condefs, mymesh, dat);

  // write design-topology
  EXODUS::write_dat_design_topology(condefs, mymesh, dat);

  // write nodal coordinates
  EXODUS::write_dat_nodes(mymesh, dat);

  // write elements
  EXODUS::write_dat_eles(eledefs, mymesh, dat);

  // close datfile
  if (dat.is_open()) dat.close();

  return 0;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EXODUS::write_dat_intro(
    const std::string& headfile, const EXODUS::Mesh& mymesh, std::ostream& dat)
{
  dat << "==================================================================\n"
         "                   General Data File 4C\n"
         "==================================================================\n"
         "-------------------------------------------------------------TITLE\n"
         "created by pre_exodus\n"
         "------------------------------------------------------PROBLEM SIZE\n";
  // print number of elements and nodes just as an comment instead of
  // a valid parameter (prevents possible misuse of these parameters in 4C)
  dat << "//ELEMENTS    " << mymesh.get_num_ele() << std::endl;
  dat << "//NODES       " << mymesh.get_num_nodes() << std::endl;
  // parameter for the number of spatial dimensions
  dat << "DIM           " << mymesh.get_four_c_dim() << std::endl;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EXODUS::write_dat_head(const std::string& headfile, std::ostream& dat)
{
  std::stringstream head;
  const char* headfilechar = headfile.c_str();
  std::ifstream header(headfilechar, std::ifstream::in);
  if (not header.good())
  {
    std::cout << std::endl << "Unable to open file: " << headfilechar << std::endl;
    FOUR_C_THROW("Unable to open head-file");
  }
  while (header.good()) head << (char)header.get();
  // while (!header.eof()) head << (char) header.get();
  header.close();
  std::string headstring = head.str();

  // delete sections which has been written by WriteDatIntro already
  remove_dat_section("PROBLEM SIZE", headstring);

  // remove eof character
  headstring.erase(headstring.end() - 1);

  // now put everything to the input file
  dat << headstring << std::endl;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EXODUS::remove_dat_section(const std::string& secname, std::string& headstring)
{
  const size_t secpos = headstring.find(secname);
  if (secpos != std::string::npos)
  {
    // where does this section line actually start?
    const size_t endoflastline = headstring.substr(0, secpos).rfind('\n');
    // want to keep the newline character of line before
    const size_t sectionbegin = endoflastline + 1;
    // now we remove the whole section
    const size_t sectionend = headstring.find("---", secpos);
    headstring.erase(sectionbegin, sectionend - sectionbegin);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EXODUS::write_dat_conditions(
    const std::vector<EXODUS::CondDef>& condefs, const EXODUS::Mesh& mymesh, std::ostream& dat)
{
  using namespace FourC;

  std::vector<Core::Conditions::ConditionDefinition> condlist = Input::valid_conditions();

  // count how often we have one specific condition
  std::map<std::string, std::vector<int>> count_cond;
  std::map<std::string, std::vector<int>>::const_iterator count;
  std::vector<int>::const_iterator i_c;
  for (int i_cond = 0; i_cond < static_cast<int>(condefs.size()); ++i_cond)
    (count_cond[condefs.at(i_cond).sec]).push_back(i_cond);

  // loop all valid conditions that 4C knows
  for (auto& condition : condlist)
  {
    const std::string& sectionname = condition.section_name();

    // ignore conditions occurring zero times
    count = count_cond.find(sectionname);
    if (count == count_cond.end()) continue;

    Core::IO::print_section_header(dat, sectionname);

    for (i_c = (count->second).begin(); i_c != (count->second).end(); ++i_c)
    {
      EXODUS::CondDef actcon = condefs[*i_c];
      std::string name;
      std::string pname;
      if (actcon.me == EXODUS::bcns)
      {
        name = (mymesh.get_node_set(actcon.id).get_name());
        pname = (mymesh.get_node_set(actcon.id).get_prop_name());
      }
      else if (actcon.me == EXODUS::bceb)
      {
        name = (mymesh.get_element_block(actcon.id)->get_name());
      }
      else if (actcon.me == EXODUS::bcss)
      {
        name = (mymesh.get_side_set(actcon.id).get_name());
      }
      else
        FOUR_C_THROW("Unidentified Actcon");
      if ((name != ""))
      {
        dat << "// " << name;
        if (pname != "none")
        {
          dat << " " << pname;
        }
        dat << std::endl;
      }
      else if (pname != "none")
        dat << "// " << pname << std::endl;

      // write the condition
      if (actcon.desc == "" and actcon.sec == "DESIGN SURF LOCSYS CONDITIONS" and
          actcon.me == EXODUS::bcns)
      {
        // special case for locsys conditions: calculate normal
        std::vector<double> normal_tangent = EXODUS::calc_normal_surf_locsys(actcon.id, mymesh);
        dat << "E " << actcon.e_id << " ";
        for (double normtang : normal_tangent)
          dat << std::setprecision(10) << std::fixed << normtang << " ";
        dat << std::endl;
      }
      else
        dat << "E " << actcon.e_id << " " << actcon.desc << std::endl;
    }
    // remove sectionname from map, since writing is done
    count_cond.erase(sectionname);
  }
  if (count_cond.size() > 0)  // there are conditions left that were not recognized!!
  {
    std::cout << std::endl << std::endl;
    for (count = count_cond.begin(); count != count_cond.end(); ++count)
      std::cout << "Section name  " << count->first << "  is not valid. Typo?" << std::endl;

    FOUR_C_THROW("There are invalid condition names in your bc file (see list above)");
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<double> EXODUS::calc_normal_surf_locsys(const int ns_id, const EXODUS::Mesh& m)
{
  std::vector<double> normaltangent;
  EXODUS::NodeSet ns = m.get_node_set(ns_id);

  std::set<int> nodes_from_nodeset = ns.get_node_set();
  std::set<int>::iterator it;

  // compute normal
  auto surfit = nodes_from_nodeset.begin();
  int origin = *surfit;  // get first set node
  ++surfit;
  int head1 = *surfit;  // get second set node
  ++surfit;

  std::set<int>::iterator thirdnode;

  const auto compute_normal = [](int head1, int origin, int head2, const EXODUS::Mesh& basemesh)
  {
    std::vector<double> normal(3);
    std::vector<double> h1 = basemesh.get_node(head1);
    std::vector<double> h2 = basemesh.get_node(head2);
    std::vector<double> o = basemesh.get_node(origin);

    normal[0] = ((h1[1] - o[1]) * (h2[2] - o[2]) - (h1[2] - o[2]) * (h2[1] - o[1]));
    normal[1] = -((h1[0] - o[0]) * (h2[2] - o[2]) - (h1[2] - o[2]) * (h2[0] - o[0]));
    normal[2] = ((h1[0] - o[0]) * (h2[1] - o[1]) - (h1[1] - o[1]) * (h2[0] - o[0]));

    double length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    const double epsilon = 1.E-4;
    if (length > epsilon)
    {
      normal[0] = normal[0] / length;
      normal[1] = normal[1] / length;
      normal[2] = normal[2] / length;
    }
    else
    {  // normal is undefined, vectors seem collinear
      normal.resize(1);
      normal[0] = 0.0;
    }

    return normal;
  };

  // find third node such that a proper normal can be computed
  for (it = surfit; it != nodes_from_nodeset.end(); ++it)
  {
    thirdnode = it;
    normaltangent = compute_normal(head1, origin, *thirdnode, m);
    if (normaltangent.size() != 1) break;
  }
  if (normaltangent.size() == 1)
  {
    FOUR_C_THROW(
        "Warning! No normal defined for SurfLocsys within nodeset '%s'!", (ns.get_name()).c_str());
  }

  // find tangent by Gram-Schmidt
  std::vector<double> t(3);
  t.at(0) = 1.0;
  t.at(1) = 0.0;
  t.at(2) = 0.0;  // try this one
  double sp = t[0] * normaltangent[0] + t[1] * normaltangent[1] +
              t[2] * normaltangent[2];  // scalar product
  // subtract projection
  t.at(0) -= normaltangent[0] * sp;
  t.at(1) -= normaltangent[1] * sp;
  t.at(2) -= normaltangent[2] * sp;

  // very unlucky case
  if (t.at(0) < 1.0E-14)
  {
    t.at(0) = 0.0;
    t.at(1) = 1.0;
    t.at(2) = 0.0;  // rather use this
  }

  normaltangent.push_back(t.at(0));
  normaltangent.push_back(t.at(1));
  normaltangent.push_back(t.at(2));

  return normaltangent;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EXODUS::write_dat_design_topology(
    const std::vector<EXODUS::CondDef>& condefs, const EXODUS::Mesh& mymesh, std::ostream& dat)
{
  using namespace FourC;

  // sort 4C conditions w.r.t. underlying topology
  std::map<int, EXODUS::CondDef> dpoints;
  std::map<int, EXODUS::CondDef> dlines;
  std::map<int, EXODUS::CondDef> dsurfs;
  std::map<int, EXODUS::CondDef> dvols;

  for (const auto& conditiondefinition : condefs)
  {
    switch (conditiondefinition.gtype)
    {
      case Core::Conditions::geometry_type_point:
        dpoints.insert(std::make_pair(conditiondefinition.e_id, conditiondefinition));
        break;
      case Core::Conditions::geometry_type_line:
        dlines.insert(std::make_pair(conditiondefinition.e_id, conditiondefinition));
        break;
      case Core::Conditions::geometry_type_surface:
        dsurfs.insert(std::make_pair(conditiondefinition.e_id, conditiondefinition));
        break;
      case Core::Conditions::geometry_type_volume:
        dvols.insert(std::make_pair(conditiondefinition.e_id, conditiondefinition));
        break;
      case Core::Conditions::geometry_type_no_geom:
        // do nothing
        break;
      default:
        FOUR_C_THROW("Cannot identify Condition GeometryType");
        break;
    }
  }

  dat << "-----------------------------------------------DNODE-NODE TOPOLOGY" << std::endl;
  for (const auto& dpoint : dpoints)
  {
    const auto nodes = EXODUS::get_ns_from_bc_entity(dpoint.second, mymesh);
    for (auto node : nodes)
    {
      dat << "NODE    " << node << " "
          << "DNODE " << dpoint.second.e_id << std::endl;
    }
  }
  dat << "-----------------------------------------------DLINE-NODE TOPOLOGY" << std::endl;
  for (const auto& dline : dlines)
  {
    const auto nodes = EXODUS::get_ns_from_bc_entity(dline.second, mymesh);
    for (auto node : nodes)
    {
      dat << "NODE    " << node << " "
          << "DLINE " << dline.second.e_id << std::endl;
    }
  }
  dat << "-----------------------------------------------DSURF-NODE TOPOLOGY" << std::endl;
  for (const auto& dsurf : dsurfs)
  {
    const auto nodes = EXODUS::get_ns_from_bc_entity(dsurf.second, mymesh);
    for (auto node : nodes)
    {
      dat << "NODE    " << node << " "
          << "DSURFACE " << dsurf.second.e_id << std::endl;
    }
  }
  dat << "------------------------------------------------DVOL-NODE TOPOLOGY" << std::endl;
  for (const auto& dvol : dvols)
  {
    const auto nodes = EXODUS::get_ns_from_bc_entity(dvol.second, mymesh);
    for (auto node : nodes)
    {
      dat << "NODE    " << node << " "
          << "DVOL " << dvol.second.e_id << std::endl;
    }
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::set<int> EXODUS::get_ns_from_bc_entity(const EXODUS::CondDef& e, const EXODUS::Mesh& m)
{
  if (e.me == EXODUS::bcns)
  {
    EXODUS::NodeSet ns = m.get_node_set(e.id);
    return ns.get_node_set();
  }
  else if (e.me == EXODUS::bceb)
  {
    std::set<int> allnodes;
    std::shared_ptr<EXODUS::ElementBlock> eb = m.get_element_block(e.id);
    std::shared_ptr<const std::map<int, std::vector<int>>> eles = eb->get_ele_conn();
    for (const auto& ele : *eles)
    {
      const std::vector<int> nodes = ele.second;
      for (auto node : nodes) allnodes.insert(node);
    }
    return allnodes;
  }
  else if (e.me == EXODUS::bcss)
  {
    std::set<int> allnodes;
    EXODUS::SideSet ss = m.get_side_set(e.id);
    const std::map<int, std::vector<int>> eles = ss.get_side_set();
    for (const auto& ele : eles)
    {
      const std::vector<int> nodes = ele.second;
      for (auto node : nodes) allnodes.insert(node);
    }
    return allnodes;
  }
  else
    FOUR_C_THROW("Cannot identify mesh_entity");
  std::set<int> n;
  return n;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EXODUS::write_dat_nodes(const EXODUS::Mesh& mymesh, std::ostream& dat)
{
  dat << "-------------------------------------------------------NODE COORDS" << std::endl;
  dat.precision(16);
  std::shared_ptr<std::map<int, std::vector<double>>> nodes = mymesh.get_nodes();

  for (const auto& node : *nodes)
  {
    std::vector<double> coords = node.second;
    dat << "NODE " << std::setw(9) << node.first << " COORD";
    for (double coord : coords) dat << " " << std::setw(23) << std::scientific << coord;
    dat << std::endl;
  }
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EXODUS::write_dat_eles(
    const std::vector<ElemDef>& eledefs, const EXODUS::Mesh& mymesh, std::ostream& dat)
{
  // sort elements w.r.t. structure, fluid, ale, scalar transport, thermo, etc.
  std::vector<EXODUS::ElemDef> structure_elements;
  std::vector<EXODUS::ElemDef> fluid_elements;
  std::vector<EXODUS::ElemDef> ale_elements;
  std::vector<EXODUS::ElemDef> lubrication_elements;
  std::vector<EXODUS::ElemDef> transport_elements;
  std::vector<EXODUS::ElemDef> transport2_elements;
  std::vector<EXODUS::ElemDef> thermo_elements;
  std::vector<EXODUS::ElemDef> cell_elements;
  std::vector<EXODUS::ElemDef> cellscatra_elements;
  std::vector<EXODUS::ElemDef> artery_elements;

  for (const auto& element_definition : eledefs)
  {
    if (element_definition.sec == "STRUCTURE")
      structure_elements.push_back(element_definition);
    else if (element_definition.sec == "FLUID")
      fluid_elements.push_back(element_definition);
    else if (element_definition.sec == "ALE")
      ale_elements.push_back(element_definition);
    else if (element_definition.sec == "LUBRICATION")
      lubrication_elements.push_back(element_definition);
    else if (element_definition.sec == "TRANSPORT")
      transport_elements.push_back(element_definition);
    else if (element_definition.sec == "TRANSPORT2")
      transport2_elements.push_back(element_definition);
    else if (element_definition.sec == "THERMO")
      thermo_elements.push_back(element_definition);
    else if (element_definition.sec == "CELL")
      cell_elements.push_back(element_definition);
    else if (element_definition.sec == "CELLSCATRA")
      cellscatra_elements.push_back(element_definition);
    else if (element_definition.sec == "ARTERY")
      artery_elements.push_back(element_definition);
    else if (element_definition.sec == "")
      ;
    else
    {
      std::cout << "Unknown ELEMENT sectionname in eb" << element_definition.id << ": '"
                << element_definition.sec << "'!" << std::endl;
      FOUR_C_THROW("Unknown ELEMENT sectionname");
    }
  }

  // element ids in 4C dat files start with 1, this int is adapted for more than one element section
  int startele = 1;

  const auto printElementSection =
      [&](const std::vector<ElemDef>& ele_vector, const std::string& section_name)
  {
    const unsigned padding_length = 66;
    // we need at least 2 dashes at the beginning to be recognizable to the dat file reader
    const unsigned min_num_preceding_dashes = 2;

    if (section_name.length() > padding_length - min_num_preceding_dashes)
      FOUR_C_THROW("The section name you chose exceeds padding length");

    std::string padded_section_name(section_name);
    padded_section_name.insert(
        padded_section_name.begin(), padding_length - padded_section_name.length(), '-');

    dat << padded_section_name << std::endl;

    for (const auto& ele : ele_vector)
    {
      std::shared_ptr<EXODUS::ElementBlock> eb = mymesh.get_element_block(ele.id);
      EXODUS::dat_eles(*eb, ele, startele, dat, ele.id);
    }
  };

  // print structure elements
  if (!structure_elements.empty()) printElementSection(structure_elements, "STRUCTURE ELEMENTS");

  // print fluid elements
  if (!fluid_elements.empty()) printElementSection(fluid_elements, "FLUID ELEMENTS");

  // print ale elements
  if (!ale_elements.empty()) printElementSection(ale_elements, "ALE ELEMENTS");

  // print Lubrication elements
  if (!lubrication_elements.empty())
    printElementSection(lubrication_elements, "LUBRICATION ELEMENTS");

  // print transport elements
  if (!transport_elements.empty()) printElementSection(transport_elements, "TRANSPORT ELEMENTS");

  // print transport2 elements
  if (!transport2_elements.empty()) printElementSection(transport2_elements, "TRANSPORT2 ELEMENTS");

  // print thermo elements
  if (!thermo_elements.empty()) printElementSection(thermo_elements, "THERMO ELEMENTS");

  // print cell elements
  if (!cell_elements.empty()) printElementSection(cell_elements, "CELL ELEMENTS");

  // print cellscatra elements
  if (!cellscatra_elements.empty()) printElementSection(cellscatra_elements, "CELLSCATRA ELEMENTS");

  // print artery elements
  if (!artery_elements.empty()) printElementSection(artery_elements, "ARTERY ELEMENTS");
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void EXODUS::dat_eles(const EXODUS::ElementBlock& eb, const EXODUS::ElemDef& acte, int& startele,
    std::ostream& datfile, const int eb_id)
{
  auto eles = eb.get_ele_conn();
  for (const auto& ele : *eles)
  {
    std::stringstream dat;  // first build up the std::string for actual element line
    const std::vector<int> nodes = ele.second;
    dat << "   " << startele;
    dat << " " << acte.ename;  // e.g. "SOLID"
    dat << " " << Core::FE::cell_type_to_string(pre_shape_to_drt(eb.get_shape()));
    dat << "  ";
    for (auto node : nodes) dat << node << " ";
    dat << "   " << acte.desc;  // e.g. "MAT 1"
    dat << std::endl;           // finish this element line

    startele++;
    datfile << dat.str();  // only one access to the outfile (saves system time)
  }
}

FOUR_C_NAMESPACE_CLOSE
