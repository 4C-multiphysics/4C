// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_post_monitor.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_gauss_point_postprocess.hpp"
#include "4C_global_data.hpp"
#include "4C_io_legacy_table.hpp"
#include "4C_post_common.hpp"
#include "4C_thermo_ele_action.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Teuchos_CommandLineProcessor.hpp>

#include <fstream>
#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
MonWriter::MonWriter(PostProblem& problem, std::string& infieldtype,
    int node)
    : myrank_(Core::Communication::my_mpi_rank(problem.get_comm()))  // get my processor id
{
  using namespace FourC;

  // determine the owner of the node
  nodeowner_ = false;

  int numdis = problem.num_discr();
  // std::string fieldtype = "";
  // loop over all available discretizations
  for (int i = 0; i < numdis; ++i)
  {
    PostField* field = problem.get_discretization(i);
    if (field->name() == infieldtype)
    {
      // pointer (rcp) to actual discretisation
      std::shared_ptr<Core::FE::Discretization> mydiscrete = field->discretization();
      // store, if this node belongs to me
      if (mydiscrete->have_global_node(node))
      {
        nodeowner_ = mydiscrete->have_global_node(node);
      }
    }
  }  // end loop over dis

  // ensure that we really found exactly one node owner
  {
    int localnodeowner = (int)nodeowner_;
    int numnodeowner = 0;
    numnodeowner = Core::Communication::sum_all(localnodeowner, (problem.get_comm()));
    if ((myrank_ == 0) and (numnodeowner == 0)) FOUR_C_THROW("Could not find node {}", node);
    if ((myrank_ == 0) and (numnodeowner > 1))
      FOUR_C_THROW("Found more than one owner of node {}: {}", node, numnodeowner);
  }

  return;
}


/*----------------------------------------------------------------------*/
void MonWriter::write_mon_file(PostProblem& problem, std::string& infieldtype, int node)
{
  using namespace FourC;

  // create my output file
  std::string filename = problem.outname() + ".mon";
  std::ofstream outfile;
  if (nodeowner_)
  {
    outfile.open(filename.c_str());
  }
  // int numdis = problem.num_discr();

  // get pointer to discretisation of actual field
  PostField* field = get_field_ptr(problem);
  if (field == nullptr) FOUR_C_THROW("Could not obtain field");

  check_infield_type(infieldtype);

  // pointer (rcp) to actual discretisation
  std::shared_ptr<Core::FE::Discretization> mydiscrete = field->discretization();
  // space dimension of the problem
  int dim = problem.num_dim();

  // get actual results of total problem
  PostResult result = PostResult(field);

  // compute offset = datamap.MinAllGID() - field->discretization()->dof_row_map()->MinAllGID().
  // Note that datamap can only be computed in WriteResult(...), which is pure virtual on
  // this level. Hence offset is split up into two parts!
  // First part:
  const int offset1 = -field->discretization()->dof_row_map()->min_all_gid();

  // global nodal dof numbers
  std::vector<int> gdof;

  if (nodeowner_)
  {
    // test, if this node belongs to me
    bool ismynode = mydiscrete->have_global_node(node);
    if (!ismynode)  // if this node does not belong to this field ( or proc, but we should be
                    // serial)
      field_error(node);

    // pointer to my actual node
    const Core::Nodes::Node* mynode = mydiscrete->g_node(node);

    // global nodal dof numbers
    gdof = mydiscrete->dof(mynode);
    // set some dummy values
    for (unsigned i = 0; i < gdof.size(); i++)
    {
      gdof[i] += offset1;
    }

    // write header
    write_header(outfile);
    outfile << node << "\n";
    outfile << "# control information: nodal coordinates   ";
    outfile << "x = " << mynode->x()[0] << "    ";
    outfile << "y = " << mynode->x()[1] << "    ";
    if (dim > 2) outfile << "z = " << mynode->x()[2];
    outfile << "\n";
    outfile << "#\n";

    write_table_head(outfile, dim);
  }
  else  // this proc is not the node owner
  {
    // set some dummy values
    for (int i = 0; i < dim + 1; ++i)
    {
      gdof.push_back(-1);
    }
  }

  // this is a loop over all time steps that should be written
  // writing step size is considered
  if (nodeowner_)
  {
    while (result.next_result()) write_result(outfile, result, gdof, dim);
  }

  // close file
  if (outfile.is_open()) outfile.close();
}

/*----------------------------------------------------------------------*/
void MonWriter::write_mon_stress_file(
    PostProblem& problem, std::string& infieldtype, std::string stresstype, int node)
{
  // stop it now
  if ((stresstype != "none") and (stresstype != "ndxyz"))
    FOUR_C_THROW("Cannot deal with requested stress output type: {}", stresstype);

  // write stress
  if (stresstype != "none")
  {
    // file name
    const std::string filename = problem.outname() + ".stress.mon";
    // define kind of stresses
    std::vector<std::string> groupnames;
    groupnames.push_back("gauss_cauchy_stresses_xyz");
    groupnames.push_back("gauss_2PK_stresses_xyz");
    // write it, now
    write_mon_str_file(filename, problem, infieldtype, "stress", stresstype, groupnames, node);
  }

  return;
}

/*----------------------------------------------------------------------*/
void MonWriter::write_mon_strain_file(
    PostProblem& problem, std::string& infieldtype, std::string straintype, int node)
{
  // stop it now
  if ((straintype != "none") and (straintype != "ndxyz"))
    FOUR_C_THROW("Cannot deal with requested strain output type: {}", straintype);

  if (straintype != "none")
  {
    // output file name
    const std::string filename = problem.outname() + ".strain.mon";

    // define kind of strains
    std::vector<std::string> groupnames;
    groupnames.push_back("gauss_GL_strains_xyz");
    groupnames.push_back("gauss_EA_strains_xyz");
    groupnames.push_back("gauss_LOG_strains_xyz");

    // write, now
    write_mon_str_file(filename, problem, infieldtype, "strain", straintype, groupnames, node);
  }

  return;
}

/*----------------------------------------------------------------------*/
void MonWriter::write_mon_pl_strain_file(
    PostProblem& problem, std::string& infieldtype, std::string straintype, int node)
{
  // stop it now
  if ((straintype != "none") and (straintype != "ndxyz"))
    FOUR_C_THROW("Cannot deal with requested plastic strain output type: {}", straintype);

  if (straintype != "none")
  {
    // output file name
    const std::string filename = problem.outname() + ".plasticstrain.mon";

    // define kind of strains
    std::vector<std::string> groupnames;
    groupnames.push_back("gauss_pl_GL_strains_xyz");
    groupnames.push_back("gauss_pl_EA_strains_xyz");

    // write, now
    write_mon_str_file(filename, problem, infieldtype, "strain", straintype, groupnames, node);
  }

  return;
}


/*----------------------------------------------------------------------*/
void MonWriter::write_mon_str_file(const std::string& filename, PostProblem& problem,
    std::string& infieldtype, const std::string strname, const std::string strtype,
    std::vector<std::string> groupnames, int node)
{
  using namespace FourC;

  // create my output file
  std::ofstream outfile;
  if (nodeowner_)
  {
    outfile.open(filename.c_str());
  }
  // int numdis = problem.num_discr();

  // get pointer to discretisation of actual field
  PostField* field = get_field_ptr(problem);
  if (field == nullptr) FOUR_C_THROW("Could not obtain field");

  check_infield_type(infieldtype);

  // pointer (rcp) to actual discretisation
  std::shared_ptr<Core::FE::Discretization> mydiscrete = field->discretization();
  // space dimension of the problem
  const int dim = problem.num_dim();

  // get actual results of total problem
  PostResult result = PostResult(field);

  // global nodal dof numbers
  std::vector<int> gdof;

  // compute offset = datamap.MinAllGID() - field->discretization()->dof_row_map()->MinAllGID().
  // Note that datamap can only be compute in WriteResult(...), which is pure virtual on
  // this level. Hence offset is split up into two parts!
  // First part:
  const int offset1 = -field->discretization()->dof_row_map()->min_all_gid();

  if (nodeowner_)
  {
    // test, if this node belongs to me
    bool ismynode = mydiscrete->have_global_node(node);
    if (!ismynode)  // if this node does not belong to this field ( or proc, but we should be
                    // seriell)
      field_error(node);

    // pointer to my actual node
    const Core::Nodes::Node* mynode = mydiscrete->g_node(node);

    // global nodal dof numbers
    gdof = mydiscrete->dof(mynode);
    // set some dummy values
    for (unsigned i = 0; i < gdof.size(); i++)
    {
      gdof[i] += offset1;
    }
    // write header
    write_header(outfile);
    outfile << node << "\n";
    outfile << "# control information: nodal coordinates   ";
    outfile << "x = " << mynode->x()[0] << "    ";
    outfile << "y = " << mynode->x()[1] << "    ";
    if (dim > 2) outfile << "z = " << mynode->x()[2];
    outfile << "\n";
    outfile << "#\n";

    write_str_table_head(outfile, strname, strtype, dim);
  }
  else  // this proc is not the node owner
  {
    // set some dummy values
    for (int i = 0; i < dim + 1; ++i)
    {
      gdof.push_back(-1);
    }
  }

  // This is a loop over all possible stress or strain modes (called groupnames).
  // The call is handed to _all_ processors, because the extrapolation of the
  // stresses/strains from Gauss points to nodes is done by Core::FE::Discretization
  // utilising an assembly call. The assembly is parallel and thus all processors
  // have to be incorporated --- at least I think so.
  // (culpit: bborn, 07/09)
  for (std::vector<std::string>::iterator gn = groupnames.begin(); gn != groupnames.end(); ++gn)
    write_str_results(outfile, problem, result, gdof, dim, strtype, *gn, node);

  if (outfile.is_open()) outfile.close();
}


/*----------------------------------------------------------------------*/
void MonWriter::write_mon_heatflux_file(
    PostProblem& problem, std::string& infieldtype, std::string heatfluxtype, int node)
{
  // stop it now
  if ((heatfluxtype != "none") and (heatfluxtype != "ndxyz"))
    FOUR_C_THROW("Cannot deal with requested heatflux output type: {}", heatfluxtype);

  // write heatflux
  if (heatfluxtype != "none")
  {
    // file name
    const std::string filename = problem.outname() + ".heatflux.mon";

    // define kind of heatfluxes
    std::vector<std::string> groupnames;
    groupnames.push_back("gauss_current_heatfluxes_xyz");
    groupnames.push_back("gauss_initial_heatfluxes_xyz");

    // write it, now
    write_mon_thermo_file(
        filename, problem, infieldtype, "heatflux", heatfluxtype, groupnames, node);
  }

  return;
}


/*----------------------------------------------------------------------*/
void MonWriter::write_mon_tempgrad_file(
    PostProblem& problem, std::string& infieldtype, std::string tempgradtype, int node)
{
  // stop it now
  if ((tempgradtype != "none") and (tempgradtype != "ndxyz"))
    FOUR_C_THROW("Cannot deal with requested temperature gradient output type: {}", tempgradtype);

  if (tempgradtype != "none")
  {
    // output file name
    const std::string filename = problem.outname() + ".tempgrad.mon";

    // define kind of temperature gradient
    std::vector<std::string> groupnames;
    groupnames.push_back("gauss_initial_tempgrad_xyz");
    groupnames.push_back("gauss_current_tempgrad_xyz");

    // write, now
    write_mon_thermo_file(
        filename, problem, infieldtype, "tempgrad", tempgradtype, groupnames, node);
  }

  return;
}


/*----------------------------------------------------------------------*/
void MonWriter::write_mon_thermo_file(const std::string& filename, PostProblem& problem,
    std::string& infieldtype, const std::string thrname, const std::string thrtype,
    std::vector<std::string> groupnames, int node)
{
  using namespace FourC;

  // create my output file
  std::ofstream outfile;
  if (nodeowner_)
  {
    outfile.open(filename.c_str());
  }

  //  int numdis = problem.num_discr();

  // get pointer to discretisation of actual field
  PostField* field = get_field_ptr(problem);
  if (field == nullptr) FOUR_C_THROW("Could not obtain field");

  check_infield_type(infieldtype);

  // pointer (rcp) to actual discretisation
  std::shared_ptr<Core::FE::Discretization> mydiscrete = field->discretization();
  // space dimension of the problem
  const int dim = problem.num_dim();

  // get actual results of total problem
  PostResult result = PostResult(field);

  // global nodal dof numbers
  std::vector<int> gdof;

  // compute offset = datamap.MinAllGID() - field->discretization()->dof_row_map()->MinAllGID().
  // Note that datamap can only be compute in WriteResult(...), which is pure virtual on
  // this level. Hence offset is split up into two parts!
  // First part:
  const int offset1 = -field->discretization()->dof_row_map()->min_all_gid();

  if (nodeowner_)
  {
    // test, if this node belongs to me
    bool ismynode = mydiscrete->have_global_node(node);
    if (!ismynode)  // if this node does not belong to this field ( or proc, but we should be
                    // seriell)
      field_error(node);

    // pointer to my actual node
    const Core::Nodes::Node* mynode = mydiscrete->g_node(node);

    // global nodal dof numbers
    gdof = mydiscrete->dof(mynode);
    // set some dummy values
    for (unsigned i = 0; i < gdof.size(); i++)
    {
      gdof[i] += offset1;
    }

    // write header
    write_header(outfile);
    outfile << node << "\n";
    outfile << "# control information: nodal coordinates   ";
    outfile << "x = " << mynode->x()[0] << "    ";
    outfile << "y = " << mynode->x()[1] << "    ";
    if (dim > 2) outfile << "z = " << mynode->x()[2];
    outfile << "\n";
    outfile << "#\n";

    write_thermo_table_head(outfile, thrname, thrtype, dim);
  }
  else  // this proc is not the node owner
  {
    // set some dummy values
    for (int i = 0; i < dim + 1; ++i)
    {
      gdof.push_back(-1);
    }
  }

  // This is a loop over all possible heatflux or temperature gradient modes
  // (called groupnames).The call is handed to _all_ processors, because the
  // extrapolation of the heatfluxes/temperature gradients from Gauss points to
  // nodes is done by Core::FE::Discretization utilising an assembly call. The
  // assembly is parallel and thus all processors have to be incorporated
  // --- at least I think so. (culpit: bborn, 07/09)
  for (std::vector<std::string>::iterator gn = groupnames.begin(); gn != groupnames.end(); ++gn)
    write_thermo_results(outfile, problem, result, gdof, dim, thrtype, *gn, node);

  if (outfile.is_open()) outfile.close();
}  // write_mon_thermo_file()


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
PostField* FieldMonWriter::get_field_ptr(PostProblem& problem)
{
  return problem.get_discretization(0);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FluidMonWriter::check_infield_type(std::string& infieldtype)
{
  if (infieldtype != "fluid")
    std::cout << "\nPure fluid problem, field option other than fluid has been ignored!\n\n";
}

/*----------------------------------------------------------------------*/
void FluidMonWriter::field_error(int node)
{
  FOUR_C_THROW("Node {} does not belong to fluid field!", node);
}

/*----------------------------------------------------------------------*/
void FluidMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# fluid problem, writing nodal data of node ";
}

/*----------------------------------------------------------------------*/
void FluidMonWriter::write_table_head(std::ofstream& outfile, int dim)
{
  switch (dim)
  {
    case 2:
      outfile << "# step   time     u_x      u_y      p\n";
      break;
    case 3:
      outfile << "# step   time     u_x      u_y      u_z      p\n";
      break;
    default:
      FOUR_C_THROW("Number of dimensions in space differs from 2 and 3!");
      break;
  }
}

/*----------------------------------------------------------------------*/
void FluidMonWriter::write_result(
    std::ofstream& outfile, PostResult& result, std::vector<int>& gdof, int dim)
{
  // get actual result vector
  auto resvec = result.read_result("velnp");
  const Core::LinAlg::Map& velmap = resvec->get_map();
  // do output of general time step data
  outfile << std::right << std::setw(20) << result.step();
  outfile << std::right << std::setw(20) << std::scientific << result.time();

  // compute second part of offset
  int offset2 = velmap.min_all_gid();

  // do output for velocity and pressure
  for (unsigned i = 0; i < gdof.size(); ++i)
  {
    const int lid = velmap.lid(gdof[i] + offset2);
    outfile << std::right << std::setw(20) << std::setprecision(10) << std::scientific
            << resvec->local_values_as_span()[lid];
  }
  outfile << "\n";
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void RedAirwayMonWriter::check_infield_type(std::string& infieldtype)
{
  if (infieldtype != "red_airway")
    std::cout
        << "\nPure red_airway problem, field option other than red_airway has been ignored!\n\n";
}

/*----------------------------------------------------------------------*/
void RedAirwayMonWriter::field_error(int node)
{
  FOUR_C_THROW("Node {} does not belong to red_airway field!", node);
}

/*----------------------------------------------------------------------*/
void RedAirwayMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# red_airway problem, writing nodal data of node ";
}

/*----------------------------------------------------------------------*/
void RedAirwayMonWriter::write_table_head(std::ofstream& outfile, int dim)
{
  outfile << "# step   time     P\n";
}

/*----------------------------------------------------------------------*/
void RedAirwayMonWriter::write_result(
    std::ofstream& outfile, PostResult& result, std::vector<int>& gdof, int dim)
{
  // get actual result vector
  auto resvec = result.read_result("PO2");
  const Core::LinAlg::Map& pmap = resvec->get_map();
  // do output of general time step data
  outfile << std::right << std::setw(20) << result.step();
  outfile << std::right << std::setw(20) << std::scientific << result.time();

  // compute second part of offset
  //  int offset2 = pmap.MinAllGID();

  // do output for velocity and pressure
  for (unsigned i = 0; i < gdof.size(); ++i)
  {
    const int lid = pmap.lid(gdof[i]);
    outfile << std::right << std::setw(20) << resvec->local_values_as_span()[lid];
  }
  outfile << "\n";
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructMonWriter::check_infield_type(std::string& infieldtype)
{
  if (infieldtype != "structure")
    std::cout
        << "\nPure structural problem, field option other than structure has been ignored!\n\n";
}

/*----------------------------------------------------------------------*/
void StructMonWriter::field_error(int node)
{
  FOUR_C_THROW("Node {} does not belong to structure field!", node);
}

/*----------------------------------------------------------------------*/
void StructMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# structure problem, writing nodal data of node ";
}

/*----------------------------------------------------------------------*/
void StructMonWriter::write_table_head(std::ofstream& outfile, int dim)
{
  switch (dim)
  {
    case 2:
      outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16)
              << "time" << std::right << std::setw(16) << "d_x" << std::right << std::setw(16)
              << "d_y" << std::right << std::setw(16) << "v_x" << std::right << std::setw(16)
              << "v_y" << std::right << std::setw(16) << "a_x" << std::right << std::setw(16)
              << "a_y" << std::endl;
      break;
    case 3:
      outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16)
              << "time" << std::right << std::setw(16) << "d_x" << std::right << std::setw(16)
              << "d_y" << std::right << std::setw(16) << "d_z" << std::right << std::setw(16)
              << "v_x" << std::right << std::setw(16) << "v_y" << std::right << std::setw(16)
              << "v_z" << std::right << std::setw(16) << "a_x" << std::right << std::setw(16)
              << "a_y" << std::right << std::setw(16) << "a_z" << std::right << std::setw(16) << "p"
              << std::endl;
      break;
    default:
      FOUR_C_THROW("Number of dimensions in space differs from 2 and 3!");
      break;
  }
}

/*----------------------------------------------------------------------*/
void StructMonWriter::write_result(
    std::ofstream& outfile, PostResult& result, std::vector<int>& gdof, int dim)
{
  // write front

  // do output of general time step data
  outfile << std::right << std::setw(10) << result.step();
  outfile << std::right << std::setw(16) << std::scientific << result.time();

  // check dimensions
  unsigned noddof = 0;
  if (dim == 2)
  {
    noddof = gdof.size();
  }
  else if (dim == 3)
  {
    if (gdof.size() == (unsigned)dim)  // ordinary case: 3 displ DOFs
      noddof = (unsigned)dim;
    else if (gdof.size() == (unsigned)dim + 1)  // displacement+pressure: 3+1 DOFs
      noddof = (unsigned)dim;
    else  // eg. shell with displacement+rotation: 3+3 DOFs
      noddof = gdof.size();
  }

  // displacement

  // get actual result vector displacement
  auto resvec = result.read_result("displacement");
  const Core::LinAlg::Map& dispmap = resvec->get_map();

  // compute second part of offset
  int offset2 = dispmap.min_all_gid();

  // do output of displacement
  for (unsigned i = 0; i < noddof; ++i)
  {
    const int lid = dispmap.lid(gdof[i] + offset2);
    if (lid == -1) FOUR_C_THROW("illegal gid {} at {}!", gdof[i], i);
    outfile << std::right << std::setw(16) << std::scientific
            << resvec->local_values_as_span()[lid];
  }

  // pressure
  if (gdof.size() == (unsigned)dim + 1)
  {
    // get actual result vector displacement/pressure
    resvec = result.read_result("displacement");
    const Core::LinAlg::Map& pressmap = resvec->get_map();

    // compute second part of offset
    offset2 = pressmap.min_all_gid();

    // do output of pressure
    {
      const unsigned i = (unsigned)dim;
      const int lid = pressmap.lid(gdof[i] + offset2);
      if (lid == -1) FOUR_C_THROW("illegal gid {} at {}!", gdof[i], i);
      outfile << std::right << std::setw(16) << std::scientific
              << resvec->local_values_as_span()[lid];
    }
  }

  outfile << "\n";
}

/*----------------------------------------------------------------------*/
void StructMonWriter::write_str_table_head(
    std::ofstream& outfile, const std::string strname, const std::string strtype, const int dim)
{
  switch (dim)
  {
    case 2:
      outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16)
              << "time" << std::right << std::setw(16) << strname + "_xx" << std::right
              << std::setw(16) << strname + "_yy" << std::right << std::setw(16) << strname + "_xy"
              << std::endl;
      break;
    case 3:
      outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16)
              << "time" << std::right << std::setw(16) << strname + "_xx" << std::right
              << std::setw(16) << strname + "_yy" << std::right << std::setw(16) << strname + "_zz"
              << std::right << std::setw(16) << strname + "_xy" << std::right << std::setw(16)
              << strname + "_yz" << std::right << std::setw(16) << strname + "_zx" << std::endl;
      break;
    default:
      FOUR_C_THROW("Number of dimensions in space differs from 2 and 3!");
      break;
  }

  return;
}

/*----------------------------------------------------------------------*/
void StructMonWriter::write_str_results(std::ofstream& outfile, PostProblem& problem,
    PostResult& result, std::vector<int>& gdof, int dim, std::string strtype, std::string groupname,
    const int node)
{
  result.next_result();  // needed
  if (map_has_map(result.group(), groupname.c_str()))
  {
    // strings
    std::string name;
    std::string out;
    if (groupname == "gauss_2PK_stresses_xyz")
    {
      name = "nodal_2PK_stresses_xyz";
      out = "2nd Piola-Kirchhoff stresses";
    }
    else if (groupname == "gauss_cauchy_stresses_xyz")
    {
      name = "nodal_cauchy_stresses_xyz";
      out = "Cauchy stresses";
    }
    else if (groupname == "gauss_GL_strains_xyz")
    {
      name = "nodal_GL_strains_xyz";
      out = "Green-Lagrange strains";
    }
    else if (groupname == "gauss_EA_strains_xyz")
    {
      name = "nodal_EA_strains_xyz";
      out = "Euler-Almansi strains";
    }
    else if (groupname == "gauss_LOG_strains_xyz")
    {
      name = "nodal_LOG_strains_xyz";
      out = "Logarithmic strains";
    }
    // the same for plastic strains
    else if (groupname == "gauss_pl_GL_strains_xyz")
    {
      name = "nodal_pl_GL_strains_xyz";
      out = "Plastic Green-Lagrange strains";
    }
    else if (groupname == "gauss_pl_EA_strains_xyz")
    {
      name = "nodal_pl_EA_strains_xyz";
      out = "Plastic Euler-Almansi strains";
    }
    else
    {
      FOUR_C_THROW("trying to write something that is not a stress or a strain");
    }

    // get pointer to discretisation of actual field
    PostField* field = get_field_ptr(problem);

    // inform (eagerly waiting) user
    if (myrank_ == 0) std::cout << "writing node-based " << out << std::endl;

    // DOFs at node
    int numdf = 0;
    if (dim == 3)
      numdf = 3;
    else if (dim == 2)
      numdf = 2;
    else
      FOUR_C_THROW("Cannot handle dimension {}", dim);

    // this is a loop over all time steps that should be written
    // bottom control here, because first set has been read already
    do
    {
      write_str_result(outfile, field, result, groupname, name, numdf, node);
    } while (result.next_result());
  }

  return;
}

/*----------------------------------------------------------------------*/
void StructMonWriter::write_str_result(std::ofstream& outfile, PostField*& field,
    PostResult& result, const std::string groupname, const std::string name, const int numdf,
    const int node) const
{
  using namespace FourC;

  // get stresses/strains at Gauss points
  const std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> data =
      result.read_result_serialdensematrix(groupname);
  // discretisation (once more)
  const std::shared_ptr<Core::FE::Discretization> dis = field->discretization();

  Core::LinAlg::MultiVector<double> nodal_stress(*dis->node_row_map(), 6, true);

  dis->evaluate(
      [&](Core::Elements::Element& ele)
      {
        Core::FE::extrapolate_gauss_point_quantity_to_nodes(
            ele, *data->at(ele.id()), *dis, nodal_stress);
      });

  if (nodeowner_)
  {
    outfile << std::right << std::setw(10) << result.step();
    outfile << std::right << std::setw(16) << std::scientific << result.time();
    for (int i = 0; i < 6; i++)
      outfile << std::right << std::setw(16) << std::scientific
              << nodal_stress(i).local_values_as_span()[node];
    outfile << std::endl;
  }

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void AleMonWriter::check_infield_type(std::string& infieldtype)
{
  if (infieldtype != "ale")
    std::cout << "\nPure ALE problem, field option other than ale has been ignored!\n\n";
}

/*----------------------------------------------------------------------*/
void AleMonWriter::field_error(int node)
{
  FOUR_C_THROW("Node {} does not belong to ALE field!", node);
}

/*----------------------------------------------------------------------*/
void AleMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# ALE problem, writing nodal data of node ";
}

/*----------------------------------------------------------------------*/
void AleMonWriter::write_table_head(std::ofstream& outfile, int dim)
{
  switch (dim)
  {
    case 2:
      outfile << "# step   time     d_x      d_y\n";
      break;
    case 3:
      outfile << "# step   time     d_x      d_y      d_z\n";
      break;
    default:
      FOUR_C_THROW("Number of dimensions in space differs from 2 and 3!");
      break;
  }
}

/*----------------------------------------------------------------------*/
void AleMonWriter::write_result(
    std::ofstream& outfile, PostResult& result, std::vector<int>& gdof, int dim)
{
  // get actual result vector for displacement
  auto resvec = result.read_result("displacement");
  const Core::LinAlg::Map& dispmap = resvec->get_map();
  // do output of general time step data
  outfile << std::right << std::setw(10) << result.step();
  outfile << std::right << std::setw(16) << std::scientific << result.time();

  // compute second part of offset
  int offset2 = dispmap.min_all_gid();

  // do output for velocity and pressure
  for (unsigned i = 0; i < gdof.size() - 1; ++i)
  {
    const int lid = dispmap.lid(gdof[i] + offset2);
    outfile << std::right << std::setw(16) << std::scientific
            << resvec->local_values_as_span()[lid];
  }
  outfile << "\n";
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
PostField* FsiFluidMonWriter::get_field_ptr(PostProblem& problem)
{
  // get pointer to discretisation of actual field
  PostField* myfield = problem.get_discretization(1);
  if (myfield->name() != "fluid") FOUR_C_THROW("Fieldtype of field 1 is not fluid.");
  return myfield;
}

/*----------------------------------------------------------------------*/
void FsiFluidMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# FSI problem, writing nodal data of fluid node ";
}

/*----------------------------------------------------------------------*/
void FsiFluidMonWriter::write_table_head(std::ofstream& outfile, int dim)
{
  switch (dim)
  {
    case 2:
    {
      outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16)
              << "time" << std::right << std::setw(16) << "d_x" << std::right << std::setw(16)
              << "d_y" << std::right << std::setw(16) << "u_x" << std::right << std::setw(16)
              << "u_y" << std::right << std::setw(16) << "p" << std::right << std::setw(16)
              << "lambda_x" << std::right << std::setw(16) << "lambda_y" << std::endl;

      break;
    }
    case 3:
    {
      outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16)
              << "time" << std::right << std::setw(16) << "d_x" << std::right << std::setw(16)
              << "d_y" << std::right << std::setw(16) << "d_z" << std::right << std::setw(16)
              << "u_x" << std::right << std::setw(16) << "u_y" << std::right << std::setw(16)
              << "u_z" << std::right << std::setw(16) << "p" << std::right << std::setw(16)
              << "lambda_x" << std::right << std::setw(16) << "lambda_y" << std::right
              << std::setw(16) << "lambda_z" << std::endl;

      break;
    }
    default:
    {
      FOUR_C_THROW("Number of dimensions in space differs from 2 and 3!");
      break;
    }
  }
}

/*----------------------------------------------------------------------*/
void FsiFluidMonWriter::write_result(
    std::ofstream& outfile, PostResult& result, std::vector<int>& gdof, int dim)
{
  // get actual result vector for displacement
  std::shared_ptr<Core::LinAlg::Vector<double>> resvec = result.read_result("dispnp");
  const Core::LinAlg::Map& dispmap = resvec->get_map();
  // do output of general time step data
  outfile << std::right << std::setw(10) << result.step();
  outfile << std::right << std::setw(16) << std::scientific << result.time();

  // compute second part of offset
  int offset2 = dispmap.min_all_gid();

  for (unsigned i = 0; i < gdof.size() - 1; ++i)
  {
    const int lid = dispmap.lid(gdof[i] + offset2);
    outfile << std::right << std::setw(16) << std::scientific
            << resvec->local_values_as_span()[lid];
  }


  // get actual result vector for velocity
  resvec = result.read_result("velnp");
  const Core::LinAlg::Map& velmap = resvec->get_map();

  // compute second part of offset
  offset2 = velmap.min_all_gid();

  // do output for velocity and pressure
  for (unsigned i = 0; i < gdof.size(); ++i)
  {
    const int lid = velmap.lid(gdof[i] + offset2);
    outfile << std::right << std::setw(16) << std::scientific
            << resvec->local_values_as_span()[lid];
  }

  // check if fsilambda is available
  MAP* dummydir;
  if (map_find_map(result.group(), "fsilambda", &dummydir))
  {
    // get actual result vector for fsilambda
    resvec = result.read_result("fsilambda");
    const Core::LinAlg::Map& lambdamap = resvec->get_map();

    // compute second part of offset
    offset2 = lambdamap.min_all_gid();

    for (unsigned i = 0; i < gdof.size() - 1; ++i)
    {
      const int lid = lambdamap.lid(gdof[i] + offset2);
      outfile << std::right << std::setw(16) << std::scientific
              << resvec->local_values_as_span()[lid];
    }
  }
  else
  {
    for (unsigned i = 0; i < gdof.size() - 1; ++i)
    {
      outfile << std::right << std::setw(16) << "not avail.";
    }
  }

  outfile << "\n";
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
PostField* FsiStructMonWriter::get_field_ptr(PostProblem& problem)
{
  // get pointer to discretisation of actual field
  PostField* myfield = problem.get_discretization(0);
  if (myfield->name() != "structure") FOUR_C_THROW("Fieldtype of field 1 is not structure.");
  return myfield;
}

/*----------------------------------------------------------------------*/
void FsiStructMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# FSI problem, writing nodal data of structure node ";
}

/*----------------------------------------------------------------------*/
void FsiStructMonWriter::write_table_head(std::ofstream& outfile, int dim)
{
  switch (dim)
  {
    case 2:
      outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16)
              << "time" << std::right << std::setw(16) << "d_x" << std::right << std::setw(16)
              << "d_y" << std::right << std::setw(16) << "v_x" << std::right << std::setw(16)
              << "v_y" << std::right << std::setw(16) << "a_x" << std::right << std::setw(16)
              << "a_y" << std::right << std::setw(16) << "p" << std::right << std::setw(16)
              << "lambda_x" << std::right << std::setw(16) << "lambda_y" << std::endl;
      break;
    case 3:
      outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16)
              << "time" << std::right << std::setw(16) << "d_x" << std::right << std::setw(16)
              << "d_y" << std::right << std::setw(16) << "d_z" << std::right << std::setw(16)
              << "v_x" << std::right << std::setw(16) << "v_y" << std::right << std::setw(16)
              << "v_z" << std::right << std::setw(16) << "a_x" << std::right << std::setw(16)
              << "a_y" << std::right << std::setw(16) << "a_z" << std::right << std::setw(16) << "p"
              << std::right << std::setw(16) << "lambda_x" << std::right << std::setw(16)
              << "lambda_y" << std::right << std::setw(16) << "lambda_z" << std::endl;
      break;
    default:
      FOUR_C_THROW("Number of dimensions in space differs from 2 and 3!");
      break;
  }
}

/*----------------------------------------------------------------------*/
void FsiStructMonWriter::write_result(
    std::ofstream& outfile, PostResult& result, std::vector<int>& gdof, int dim)
{
  // write front

  // do output of general time step data
  outfile << std::right << std::setw(10) << result.step();
  outfile << std::right << std::setw(16) << std::scientific << result.time();

  // check dimensions
  unsigned noddof = 0;
  if (dim == 2)
  {
    noddof = gdof.size();
  }
  else if (dim == 3)
  {
    if (gdof.size() == (unsigned)dim)  // ordinary case: 3 displ DOFs
      noddof = (unsigned)dim;
    else if (gdof.size() == (unsigned)dim + 1)  // displacement+pressure: 3+1 DOFs
      noddof = (unsigned)dim;
    else  // eg. shell with displacement+rotation: 3+3 DOFs
      noddof = gdof.size();
  }

  // displacement

  // get actual result vector displacement
  std::shared_ptr<Core::LinAlg::Vector<double>> resvec = result.read_result("displacement");
  const Core::LinAlg::Map& dispmap = resvec->get_map();

  // compute second part of offset
  int offset2 = dispmap.min_all_gid();

  // do output of displacement
  for (unsigned i = 0; i < noddof; ++i)
  {
    const int lid = dispmap.lid(gdof[i] + offset2);
    if (lid == -1) FOUR_C_THROW("illegal gid {} at {}!", gdof[i], i);
    outfile << std::right << std::setw(16) << std::scientific
            << resvec->local_values_as_span()[lid];
  }

  MAP* dummydir;
  // pressure
  if (gdof.size() == (unsigned)dim + 1)
  {
    // get actual result vector displacement/pressure
    resvec = result.read_result("displacement");
    const Core::LinAlg::Map& pressmap = resvec->get_map();

    // compute second part of offset
    offset2 = pressmap.min_all_gid();

    // do output of pressure
    {
      const unsigned i = (unsigned)dim;
      const int lid = pressmap.lid(gdof[i] + offset2);
      if (lid == -1) FOUR_C_THROW("illegal gid {} at {}!", gdof[i], i);
      outfile << std::right << std::setw(16) << std::scientific
              << resvec->local_values_as_span()[lid];
    }
  }
  else
  {
    outfile << std::right << std::setw(16) << "not avail.";
  }

  // check if fsilambda is available
  if (map_find_map(result.group(), "fsilambda", &dummydir))
  {
    // get actual result vector for fsilambda
    resvec = result.read_result("fsilambda");
    const Core::LinAlg::Map& lambdamap = resvec->get_map();

    // compute second part of offset
    offset2 = lambdamap.min_all_gid();

    for (unsigned i = 0; i < noddof; ++i)
    {
      const int lid = lambdamap.lid(gdof[i] + offset2);
      if (lid == -1) FOUR_C_THROW("illegal gid {} at {}!", gdof[i], i);
      outfile << std::right << std::setw(16) << std::scientific
              << resvec->local_values_as_span()[lid];
    }
  }
  else
  {
    for (unsigned i = 0; i < noddof; ++i)
    {
      outfile << std::right << std::setw(16) << "not avail.";
    }
  }

  outfile << "\n";
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
PostField* FsiAleMonWriter::get_field_ptr(PostProblem& problem)
{
  // get pointer to discretisation of actual field
  PostField* myfield = problem.get_discretization(1);
  if (myfield->name() != "fluid") FOUR_C_THROW("Fieldtype of field 1 is not fluid.");
  return myfield;
}

/*----------------------------------------------------------------------*/
void FsiAleMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# FSI problem, writing nodal data of ALE node ";
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ScatraMonWriter::check_infield_type(std::string& infieldtype)
{
  std::cout << "\nscatra something\n\n";
}

/*----------------------------------------------------------------------*/
void ScatraMonWriter::field_error(int node)
{
  FOUR_C_THROW("Node {} does not belong to scatra field!", node);
}

/*----------------------------------------------------------------------*/
void ScatraMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# SCATRA problem, writing nodal data of node ";
}

/*----------------------------------------------------------------------*/
PostField* ScatraMonWriter::get_field_ptr(PostProblem& problem)
{
  // get pointer to discretisation of actual field
  PostField* myfield = problem.get_discretization(1);
  if (myfield->name() != "scatra") FOUR_C_THROW("Fieldtype of field 1 is not scatra.");
  return myfield;
}

/*----------------------------------------------------------------------*/
void ScatraMonWriter::write_table_head(std::ofstream& outfile, int dim)
{
  switch (dim)
  {
    case 2:
      outfile << "# step   time     phi\n";
      break;
    case 3:
      outfile << "# step   time     phi\n";
      break;
    default:
      FOUR_C_THROW("Number of dimensions in space differs from 2 and 3!");
      break;
  }
}

/*----------------------------------------------------------------------*/
void ScatraMonWriter::write_result(
    std::ofstream& outfile, PostResult& result, std::vector<int>& gdof, int dim)
{
  // get actual result vector for displacement
  std::shared_ptr<Core::LinAlg::Vector<double>> resvec = result.read_result("phinp");

  const Core::LinAlg::Map& dispmap = resvec->get_map();
  // do output of general time step data
  outfile << std::right << std::setw(10) << result.step();
  outfile << std::right << std::setw(20) << std::scientific << result.time();

  // compute second part of offset
  int offset2 = dispmap.min_all_gid();

  // do output for velocity
  for (unsigned i = 0; i < gdof.size(); ++i)
  {
    const int lid = dispmap.lid(gdof[i] + offset2);
    outfile << std::right << std::setw(20) << std::setprecision(10) << std::scientific
            << resvec->local_values_as_span()[lid];
  }
  outfile << "\n";
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ThermoMonWriter::check_infield_type(std::string& infieldtype)
{
  if (infieldtype != "thermo")
    std::cout << "\nPure thermal problem, field option other than thermo has been ignored!\n\n";
}

/*----------------------------------------------------------------------*/
void ThermoMonWriter::field_error(int node)
{
  FOUR_C_THROW("Node {} does not belong to thermal field!", node);
}

/*----------------------------------------------------------------------*/
void ThermoMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# thermo problem, writing nodal data of node ";
}

/*----------------------------------------------------------------------*/
void ThermoMonWriter::write_table_head(std::ofstream& outfile, int dim)
{
  outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16) << "time"
          << std::right << std::setw(16) << "theta" << std::right << std::setw(16) << "thetarate"
          << std::endl;
}

/*----------------------------------------------------------------------*/
void ThermoMonWriter::write_result(
    std::ofstream& outfile, PostResult& result, std::vector<int>& gdof, int dim)
{
  // write front

  // do output of general time step data
  outfile << std::right << std::setw(10) << result.step();
  outfile << std::right << std::setw(16) << std::scientific << result.time();

  // temperature

  // get actual result vector temperature
  std::shared_ptr<Core::LinAlg::Vector<double>> resvec = result.read_result("temperature");
  const Core::LinAlg::Map& dispmap = resvec->get_map();

  // compute second part of offset
  int offset2 = dispmap.min_all_gid();

  // do output of temperature (always one DOF)
  for (unsigned i = 0; i < gdof.size(); ++i)
  {
    const int lid = dispmap.lid(gdof[i] + offset2);
    if (lid == -1) FOUR_C_THROW("illegal gid {} at {}!", gdof[i], i);
    outfile << std::right << std::setw(16) << std::scientific
            << resvec->local_values_as_span()[lid];
  }

  // temperature rate

  // get actual result vector temperature rate
  resvec = result.read_result("rate");
  const Core::LinAlg::Map& ratemap = resvec->get_map();

  // compute second part of offset
  offset2 = ratemap.min_all_gid();

  // do output of temperature rate
  for (unsigned i = 0; i < gdof.size(); ++i)
  {
    const int lid = ratemap.lid(gdof[i] + offset2);
    if (lid == -1) FOUR_C_THROW("illegal gid {} at {}!", gdof[i], i);
    outfile << std::right << std::setw(16) << std::scientific
            << resvec->local_values_as_span()[lid];
  }

  outfile << "\n";
}

/*----------------------------------------------------------------------*/
void ThermoMonWriter::write_thermo_table_head(
    std::ofstream& outfile, const std::string thrname, const std::string thrtype, const int dim)
{
  switch (dim)
  {
    case 1:
      outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16)
              << "time" << std::right << std::setw(16) << thrname + "_x" << std::endl;
      break;
    case 2:
      outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16)
              << "time" << std::right << std::setw(16) << thrname + "_x" << std::right
              << std::setw(16) << thrname + "_y" << std::endl;
      break;
    case 3:
      outfile << "#" << std::right << std::setw(9) << "step" << std::right << std::setw(16)
              << "time" << std::right << std::setw(16) << thrname + "_x" << std::right
              << std::setw(16) << thrname + "_y" << std::right << std::setw(16) << thrname + "_z"
              << std::endl;
      break;
    default:
      FOUR_C_THROW("Number of dimensions in space differs from 2 and 3!");
      break;
  }

  return;
}

/*----------------------------------------------------------------------*/
void ThermoMonWriter::write_thermo_results(std::ofstream& outfile, PostProblem& problem,
    PostResult& result, std::vector<int>& gdof, int dim, std::string thrtype, std::string groupname,
    const int node)
{
  result.next_result();  // needed
  if (map_has_map(result.group(), groupname.c_str()))
  {
    // strings
    std::string name;
    std::string out;
    if (groupname == "gauss_initial_heatfluxes_xyz")
    {
      name = "nodal_initial_heatfluxes_xyz";
      out = "Initial heatfluxes";
    }
    else if (groupname == "gauss_current_heatfluxes_xyz")
    {
      name = "nodal_current_heatfluxes_xyz";
      out = "Current heatfluxes";
    }
    else if (groupname == "gauss_initial_tempgrad_xyz")
    {
      name = "nodal_initial_tempgrad_xyz";
      out = "Initial temperature gradients";
    }
    else if (groupname == "gauss_current_tempgrad_xyz")
    {
      name = "nodal_current_tempgrad_xyz";
      out = "Current temperature gradients";
    }
    else
    {
      FOUR_C_THROW("trying to write something that is not a heatflux or a temperature gradient");
    }

    // get pointer to discretisation of actual field
    PostField* field = get_field_ptr(problem);

    // inform (eagerly waiting) user
    if (myrank_ == 0) std::cout << "writing node-based " << out << std::endl;

    // this is a loop over all time steps that should be written
    // bottom control here, because first set has been read already
    do
    {
      write_thermo_result(outfile, field, result, groupname, name, dim, node);
    } while (result.next_result());
  }

  return;
}

/*----------------------------------------------------------------------*/
void ThermoMonWriter::write_thermo_result(std::ofstream& outfile, PostField*& field,
    PostResult& result, const std::string groupname, const std::string name, const int dim,
    const int node) const
{
  using namespace FourC;

  // get heatfluxes/temperature gradients at Gauss points
  const std::shared_ptr<std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>>> data =
      result.read_result_serialdensematrix(groupname);
  // discretisation (once more)
  const std::shared_ptr<Core::FE::Discretization> dis = field->discretization();

  // extrapolate heatfluxes/temperature gradients to nodes
  // and assemble them in two global vectors
  Teuchos::ParameterList p;
  p.set("action", Thermo::postproc_thermo_heatflux);
  p.set("heatfluxtype", "ndxyz");
  p.set("gpheatfluxmap", data);
  p.set("total time", -1.0);

  std::shared_ptr<Core::LinAlg::Vector<double>> heatfluxx =
      std::make_shared<Core::LinAlg::Vector<double>>(*(dis->dof_row_map()));
  std::shared_ptr<Core::LinAlg::Vector<double>> heatfluxy =
      std::make_shared<Core::LinAlg::Vector<double>>(*(dis->dof_row_map()));
  std::shared_ptr<Core::LinAlg::Vector<double>> heatfluxz =
      std::make_shared<Core::LinAlg::Vector<double>>(*(dis->dof_row_map()));
  dis->evaluate(p, nullptr, nullptr, heatfluxx, heatfluxy, heatfluxz);

  const unsigned numdofpernode = 1;

  // average heatfluxes/temperature gradients and print to file
  if (nodeowner_)
  {
    const Core::Nodes::Node* lnode = dis->g_node(node);
    const std::vector<int> lnodedofs = dis->dof(lnode);
    const int adjele = lnode->num_element();

    std::vector<double> nodal_heatfluxes;
    if (dim == 3)
    {
      if (lnodedofs.size() < numdofpernode) FOUR_C_THROW("Too few DOFs at node of interest");
      nodal_heatfluxes.push_back(heatfluxx->local_values_as_span()[lnodedofs[0]] / adjele);
      nodal_heatfluxes.push_back(heatfluxy->local_values_as_span()[lnodedofs[0]] / adjele);
      nodal_heatfluxes.push_back(heatfluxz->local_values_as_span()[lnodedofs[0]] / adjele);
    }
    else if (dim == 2)
    {
      if (lnodedofs.size() < numdofpernode) FOUR_C_THROW("Too few DOFs at node of interest");
      nodal_heatfluxes.push_back(heatfluxx->local_values_as_span()[lnodedofs[0]] / adjele);
      nodal_heatfluxes.push_back(heatfluxy->local_values_as_span()[lnodedofs[0]] / adjele);
    }
    else if (dim == 1)
    {
      if (lnodedofs.size() < numdofpernode) FOUR_C_THROW("Too few DOFs at node of interest");
      nodal_heatfluxes.push_back(heatfluxx->local_values_as_span()[lnodedofs[0]] / adjele);
    }
    else
    {
      FOUR_C_THROW("Don't know what to do with {} dimensions", dim);
    }

    // print to file
    // step number
    outfile << std::right << std::setw(10) << result.step();
    // current time step
    outfile << std::right << std::setw(16) << std::scientific << result.time();
    // current heatfluxes
    for (std::vector<double>::iterator ns = nodal_heatfluxes.begin(); ns != nodal_heatfluxes.end();
        ++ns)
      outfile << std::right << std::setw(16) << std::scientific << *ns;
    outfile << std::endl;
  }

  return;
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
PostField* TsiStructMonWriter::get_field_ptr(PostProblem& problem)
{
  // get pointer to discretisation of actual field
  PostField* myfield = problem.get_discretization(1);
  if (myfield->name() != "structure") FOUR_C_THROW("Fieldtype of field 1 is not structure.");
  return myfield;
}

/*----------------------------------------------------------------------*/
void TsiStructMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# TSI problem, writing nodal data of structure node ";
}



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
PostField* TsiThermoMonWriter::get_field_ptr(PostProblem& problem)
{
  // get pointer to discretisation of actual field
  PostField* myfield = problem.get_discretization(0);
  if (myfield->name() != "thermo") FOUR_C_THROW("Fieldtype of field 2 is not thermo.");
  return myfield;
}

/*----------------------------------------------------------------------*/
void TsiThermoMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# TSI problem, writing nodal data of thermal node ";
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void PoroFluidMultiMonWriter::check_infield_type(std::string& infieldtype)
{
  if (infieldtype != "porofluid")
    std::cout << "\nPure fluid problem, field option other than porofluid has been ignored!\n\n";
}

/*----------------------------------------------------------------------*/
void PoroFluidMultiMonWriter::field_error(int node)
{
  FOUR_C_THROW("Node {} does not belong to porofluid field!", node);
}

/*----------------------------------------------------------------------*/
void PoroFluidMultiMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# pressure-based porofluid problem, writing nodal data of node ";
}

/*----------------------------------------------------------------------*/
void PoroFluidMultiMonWriter::write_table_head(std::ofstream& outfile, int dim)
{
  switch (dim)
  {
    case 2:
    case 3:
      outfile << "# step   time     phi_1 ... phi_n     saturation_1 ... saturation_n     "
                 "pressure_1 ... pressure_n ... porosity (with --optquantity=porosity)\n";
      break;
    default:
      FOUR_C_THROW("Number of dimensions in space differs from 2 and 3!");
      break;
  }
}

/*----------------------------------------------------------------------*/
void PoroFluidMultiMonWriter::write_result(
    std::ofstream& outfile, PostResult& result, std::vector<int>& gdof, int dim)
{
  // get actual result vector for displacement
  std::shared_ptr<Core::LinAlg::Vector<double>> resvec = result.read_result("phinp_fluid");
  std::shared_ptr<Core::LinAlg::Vector<double>> resvec_sat = result.read_result("saturation");
  std::shared_ptr<Core::LinAlg::Vector<double>> resvec_press = result.read_result("pressure");

  const Core::LinAlg::Map& phimap = resvec->get_map();
  const Core::LinAlg::Map& satmap = resvec_sat->get_map();
  const Core::LinAlg::Map& pressmap = resvec_press->get_map();

  // do output of general time step data
  outfile << std::right << std::setw(10) << result.step();
  outfile << std::right << std::setw(20) << std::scientific << result.time();

  // compute second part of offset
  int offset2 = phimap.min_all_gid();

  // do output for primary variable
  for (unsigned i = 0; i < gdof.size(); ++i)
  {
    const int lid = phimap.lid(gdof[i] + offset2);
    outfile << std::right << std::setw(20) << std::setprecision(10) << std::scientific
            << resvec->local_values_as_span()[lid];
  }
  // do output for saturations
  for (unsigned i = 0; i < gdof.size(); ++i)
  {
    const int lid = satmap.lid(gdof[i] + offset2);
    outfile << std::right << std::setw(20) << std::setprecision(10) << std::scientific
            << resvec_sat->local_values_as_span()[lid];
  }
  // do output for pressures
  for (unsigned i = 0; i < gdof.size(); ++i)
  {
    const int lid = pressmap.lid(gdof[i] + offset2);
    outfile << std::right << std::setw(20) << std::setprecision(10) << std::scientific
            << resvec_press->local_values_as_span()[lid];
  }
  outfile << "\n";
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
PostField* PoroMultiElastScatraFluidMonWriter::get_field_ptr(PostProblem& problem)
{
  // get pointer to discretisation of actual field
  PostField* myfield = problem.get_discretization(1);
  if (myfield->name() != "porofluid") FOUR_C_THROW("Fieldtype of field 2 is not porofluid.");
  return myfield;
}

/*----------------------------------------------------------------------*/
void PoroMultiElastScatraFluidMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# PoroMultiElast(Scatra) problem, writing nodal data of porofluid node ";
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
PostField* PoroMultiElastScatraScatraMonWriter::get_field_ptr(PostProblem& problem)
{
  // get pointer to discretisation of actual field
  PostField* myfield = problem.get_discretization(2);

  if (myfield->name() != "scatra")
  {
    std::cout
        << "Scatra-dis is not the third field. Artery coupling active? Trying field 4 for scatra"
        << std::endl;
    myfield = problem.get_discretization(3);
    if (myfield->name() != "scatra") FOUR_C_THROW("Fieldtype of field 3 or 4 is not scatra.");
  }
  return myfield;
}

/*----------------------------------------------------------------------*/
void PoroMultiElastScatraScatraMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# PoroMultiElastScatra problem, writing nodal data of scatra node ";
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
PostField* PoroMultiElastScatraArteryScatraMonWriter::get_field_ptr(PostProblem& problem)
{
  // get pointer to discretisation of actual field
  PostField* myfield = problem.get_discretization(4);
  if (myfield->name() != "artery_scatra")
    FOUR_C_THROW("Fieldtype of field 5 is not artery_scatra.");
  return myfield;
}

/*----------------------------------------------------------------------*/
void PoroMultiElastScatraArteryScatraMonWriter::write_header(std::ofstream& outfile)
{
  outfile << "# PoroMultiElastScatra problem, writing nodal data of artery_scatra node ";
}

FOUR_C_NAMESPACE_CLOSE

/*!
 * \brief filter main routine for monitoring filter
 *
 * Write ASCII file of one nodes history
 *
 * Note: Works in seriell version only! Requires to read one instance of the discretisation!!
 *

 */
int main(int argc, char** argv)
{
  using namespace FourC;

  Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;

  // command line processor to deal with arguments
  Teuchos::CommandLineProcessor my_comlinproc;
  my_comlinproc.setDocString(
      "Post monitoring filter\n\nwrite nodal result data of specified node into outfile.mon");

  bool required = true;
  /* Set an additional integer command line option which is the global node Id
     of the node you're interested in */
  int node = 0;
  my_comlinproc.setOption("node", &node, "Global node number", required);
  /* Set a string command line option */
  std::string infieldtype = "fluid";
  my_comlinproc.setOption(
      "field", &infieldtype, "Field to which output node belongs (fluid, structure, ale, scatra)");

  // my post processing problem itself
  PostProblem problem(my_comlinproc, argc, argv);


  switch (problem.problemtype())
  {
    case Core::ProblemType::fsi:
    case Core::ProblemType::fsi_redmodels:
    {
      if (infieldtype == "fluid")
      {
        FsiFluidMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else if (infieldtype == "structure")
      {
        FsiStructMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else if (infieldtype == "ale")
      {
        FOUR_C_THROW("There is no ALE output. Displacements of fluid nodes can be printed.");
        FsiAleMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else
      {
        FOUR_C_THROW("handling for monitoring of this fieldtype not yet implemented");
      }
      break;
    }
    case Core::ProblemType::structure:
    case Core::ProblemType::loma:
    case Core::ProblemType::fluid:
    case Core::ProblemType::fluid_redmodels:
    case Core::ProblemType::fps3i:
    {
      if (infieldtype == "scatra")
      {
        ScatraMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else if (infieldtype == "fluid")
      {
        FluidMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else if (infieldtype == "structure")
      {
        StructMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      break;
    }
    case Core::ProblemType::ale:
    {
      AleMonWriter mymonwriter(problem, infieldtype, node);
      mymonwriter.write_mon_file(problem, infieldtype, node);
      break;
    }
    case Core::ProblemType::thermo:
    {
      ThermoMonWriter mymonwriter(problem, infieldtype, node);
      mymonwriter.write_mon_file(problem, infieldtype, node);
      mymonwriter.write_mon_heatflux_file(problem, infieldtype, problem.heatfluxtype(), node);
      mymonwriter.write_mon_tempgrad_file(problem, infieldtype, problem.tempgradtype(), node);
      break;
    }
    case Core::ProblemType::tsi:
    {
      if (infieldtype == "structure")
      {
        TsiStructMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
        mymonwriter.write_mon_stress_file(problem, infieldtype, problem.stresstype(), node);
        mymonwriter.write_mon_strain_file(problem, infieldtype, problem.straintype(), node);
        mymonwriter.write_mon_pl_strain_file(problem, infieldtype, problem.straintype(), node);
      }
      else if (infieldtype == "thermo")
      {
        TsiThermoMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
        mymonwriter.write_mon_stress_file(problem, infieldtype, problem.stresstype(), node);
        mymonwriter.write_mon_strain_file(problem, infieldtype, problem.straintype(), node);
        // TODO: bugfix in case of coupled tsi
        //        mymonwriter.write_mon_heatflux_file(problem,infieldtype,problem.heatfluxtype(),node);
        //        mymonwriter.write_mon_tempgrad_file(problem,infieldtype,problem.tempgradtype(),node);
      }
      else
      {
        FOUR_C_THROW("handling for monitoring of this fieldtype not yet implemented");
      }
      break;
    }
    case Core::ProblemType::gas_fsi:
    case Core::ProblemType::biofilm_fsi:
    case Core::ProblemType::thermo_fsi:
    {
      FOUR_C_THROW("not implemented yet");
      break;
    }
    case Core::ProblemType::red_airways:
    {
      if (infieldtype == "red_airway")
      {
        RedAirwayMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      break;
    }
    case Core::ProblemType::poroelast:
    {
      if (infieldtype == "fluid")
      {
        FluidMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else if (infieldtype == "structure")
      {
        StructMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      break;
    }
    case Core::ProblemType::porofluid_pressure_based:
    {
      PoroFluidMultiMonWriter mymonwriter(problem, infieldtype, node);
      mymonwriter.write_mon_file(problem, infieldtype, node);
      break;
    }
    case Core::ProblemType::porofluid_pressure_based_elast:
    {
      if (infieldtype == "structure")
      {
        StructMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else if (infieldtype == "porofluid")
      {
        PoroMultiElastScatraFluidMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else
        FOUR_C_THROW(
            "Unsupported field type {} for problem-type Multiphase_Poroelasticity", infieldtype);
      break;
    }
    case Core::ProblemType::porofluid_pressure_based_elast_scatra:
    {
      if (infieldtype == "structure")
      {
        StructMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else if (infieldtype == "porofluid")
      {
        PoroMultiElastScatraFluidMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else if (infieldtype == "scatra")
      {
        PoroMultiElastScatraScatraMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else if (infieldtype == "artery_scatra")
      {
        PoroMultiElastScatraArteryScatraMonWriter mymonwriter(problem, infieldtype, node);
        mymonwriter.write_mon_file(problem, infieldtype, node);
      }
      else
        FOUR_C_THROW("Unsupported field type {} for problem-type Multiphase_Poroelasticity_ScaTra",
            infieldtype);
      break;
    }
    default:
    {
      FOUR_C_THROW("problem type {} not yet supported", problem.problemtype());
    }
    break;
  }

  return 0;
}
/*! @} (documentation module close)*/
