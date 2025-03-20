// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_post_processor_single_field_writers.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_post_common.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureFilter::write_all_results(PostField* field)
{
  writer_->write_result("displacement", "displacement", dofbased, field->problem()->num_dim());
  writer_->write_result(
      "prolongated_gauss_2PK_stresses_xyz", "prolongated_gauss_2PK_stresses_xyz", nodebased, 6);
  writer_->write_result(
      "prolongated_gauss_GL_strains_xyz", "prolongated_gauss_GL_strains_xyz", nodebased, 6);
  if (field->problem()->struct_mat_disp() == "yes")
    writer_->write_result(
        "material_displacement", "material_displacement", dofbased, field->problem()->num_dim());

  // contact and meshtying results
  writer_->write_result("activeset", "activeset", nodebased, 1);
  writer_->write_result("contactowner", "contactowner", nodebased, 1);
  writer_->write_result(
      "norcontactstress", "norcontactstress", dofbased, field->problem()->num_dim());
  writer_->write_result(
      "tancontactstress", "tancontactstress", dofbased, field->problem()->num_dim());
  writer_->write_result(
      "interfacetraction", "interfacetraction", dofbased, field->problem()->num_dim());
  writer_->write_result("slaveforces", "slaveforces", dofbased, field->problem()->num_dim());
  writer_->write_result("masterforces", "masterforces", dofbased, field->problem()->num_dim());
  writer_->write_result("norslaveforce", "norslaveforce", dofbased, field->problem()->num_dim());
  writer_->write_result("tanslaveforce", "tanslaveforce", dofbased, field->problem()->num_dim());
  writer_->write_result("normasterforce", "normasterforce", dofbased, field->problem()->num_dim());
  writer_->write_result("tanmasterforce", "tanmasterforce", dofbased, field->problem()->num_dim());
  writer_->write_result("wear", "wear", dofbased, field->problem()->num_dim());
  writer_->write_result(
      "norslaveforcelm", "norslaveforcelm", dofbased, field->problem()->num_dim());
  writer_->write_result("norslaveforceg", "norslaveforceg", dofbased, field->problem()->num_dim());
  writer_->write_result(
      "normasterforcelm", "normasterforcelm", dofbased, field->problem()->num_dim());
  writer_->write_result(
      "normasterforceg", "normasterforceg", dofbased, field->problem()->num_dim());
  writer_->write_result(
      "poronopen_lambda", "poronopen_lambda", dofbased, field->problem()->num_dim());

  // spring dashpot
  writer_->write_result("gap", "gap", nodebased, 1);
  writer_->write_result("curnormals", "curnormals", nodebased, 3);
  writer_->write_result("springstress", "springstress", nodebased, 3);

  // error norms
  writer_->write_result("L2_norm", "L2_norm", elementbased, 1);
  writer_->write_result("H1_norm", "H1_norm", elementbased, 1);
  writer_->write_result("Energy_norm", "Energy_norm", elementbased, 1);

  // one-dimensional artery
  writer_->write_result("one_d_artery_pressure", "pressure", nodebased, 1);
  writer_->write_result("one_d_artery_flow", "flow", nodebased, 1);
  writer_->write_result("one_d_artery_area", "area", nodebased, 1);
  writer_->write_result("forward_speed", "forward_speed", nodebased, 1);
  writer_->write_result("forward_speed0", "forward_speed0", nodebased, 1);
  writer_->write_result("backward_speed", "backward_speed", nodebased, 1);
  writer_->write_result("backward_speed0", "backward_speed0", nodebased, 1);

  // reduced dimensional airway
  writer_->write_result("pnp", "pressure", dofbased, 1);
  writer_->write_result("p_nonlin", "pressure_error", dofbased, 1);
  writer_->write_result("NodeIDs", "NodeIDs", dofbased, 1);
  writer_->write_result("radii", "radii", dofbased, 1);
  //  writer_->WriteResult("juncVolMix", "juncVolMix", dofbased, 1);
  //  writer_->WriteResult("forwardVolume","forwardVolume", dofbased, 1);
  writer_->write_result("scatraO2np", "scatraO2", dofbased, 1);
  writer_->write_result("PO2", "PO2", dofbased, 1);
  writer_->write_result("dVO2", "dVO2", dofbased, 1);
  writer_->write_result("AcinarPO2", "AcinarPO2", dofbased, 1);
  //  writer_->WriteResult("e1scatranp","e1scatranp",elementbased, 1);
  //  writer_->WriteResult("e2scatranp","e2scatranp",elementbased, 1);
  writer_->write_result("juncVolMix", "juncVolMix", dofbased, 1);

  writer_->write_result("acini_vnp", "acini_volume", elementbased, 1);
  writer_->write_result("acini_volumetric_strain", "acini_volume_strain", elementbased, 1);
  writer_->write_result("acini_max_strain_location", "acini_max_strain_location", elementbased, 1);
  writer_->write_result("acin_bc", "acini_bc", elementbased, 1);
  writer_->write_result("acini_v0", "acini_volume0", elementbased, 1);

  writer_->write_result("qin_np", "flow_in", elementbased, 1);
  writer_->write_result("qout_np", "flow_out", elementbased, 1);
  writer_->write_result("x_np", "x_np", elementbased, 1);
  writer_->write_result("open", "open", elementbased, 1);
  writer_->write_result("p_extnp", "p_extnp", elementbased, 1);
  writer_->write_result("p_extn", "p_extn", elementbased, 1);
  writer_->write_result("airway_acinus_dep", "airway_acinus_dep", elementbased, 1);
  writer_->write_result("generations", "generations", elementbased, 1);
  writer_->write_result("elemVolumenp", "airway_volume", elementbased, 1);
  writer_->write_result("elemVolume0", "airway_volume0", elementbased, 1);
  writer_->write_result("elemRadius_current", "radius_current", elementbased, 1);


  // additional forces due to lung fsi (volume constraint)
  writer_->write_result("Add_Forces", "Add_Forces", dofbased, field->problem()->num_dim());

  // Lagrange multiplier at the interface in monolithic fsi
  writer_->write_result("fsilambda", "fsilambda", dofbased, field->problem()->num_dim());

  // Lagrange multiplier at the interface in monolithic fsi
  writer_->write_result("fpilambda_ps", "fpilambda_ps", dofbased, field->problem()->num_dim());
  writer_->write_result("fpilambda_pf", "fpilambda_pf", dofbased, field->problem()->num_dim());

  // additional output for biofilm problems
  writer_->write_result(
      "str_growth_displ", "str_growth_displ", dofbased, field->problem()->num_dim());

  // additional output for poro problems
  writer_->write_result("porosity_p1", "porosity_p1", dofbased, 1);
  writer_->write_result(
      "poronopencond_lambda", "poronopencond_lambda", dofbased, field->problem()->num_dim());

  // additional output for tsi (non-matching grid)
  writer_->write_result("struct_temperature", "struct_temperature", nodebased, 1);

  // monolithic scalar-structure interaction involving scatra-scatra interface coupling outputs
  // nodal Cauchy stresses instead of Gauss-point ones
  writer_->write_result("nodal_stresses_xyz", "nodal_stresses_xyz", nodebased, 6);

  // elastohydrodynamic lubrication
  writer_->write_result("fluid_force", "fluid_force", dofbased, field->problem()->num_dim());
  writer_->write_result("normal_contact", "normal_contact", dofbased, field->problem()->num_dim());
  writer_->write_result(
      "tangential_contact", "tangential_contact", dofbased, field->problem()->num_dim());
  writer_->write_result("active", "active", nodebased, 1);
  writer_->write_result("slip", "slip", nodebased, 1);

  if (stresstype_ != "none")
  {
    // although appearing here twice, only one function call to PostStress
    // is really postprocessing Gauss point stresses, since only _either_
    // Cauchy _or_ 2nd Piola-Kirchhoff stresses are written during simulation!
    post_stress("gauss_cauchy_stresses_xyz", stresstype_);
    post_stress("gauss_2PK_stresses_xyz", stresstype_);
  }
  if (straintype_ != "none")
  {
    // although appearing here twice, only one function call to PostStress
    // is really postprocessing Gauss point strains, since only _either_
    // Green-Lagrange _or_ Euler-Almansi strains are written during simulation!
    post_stress("gauss_GL_strains_xyz", straintype_);
    post_stress("gauss_EA_strains_xyz", straintype_);
    post_stress("gauss_LOG_strains_xyz", straintype_);
    // the same for plastic strains
    post_stress("gauss_pl_GL_strains_xyz", straintype_);
    post_stress("gauss_pl_EA_strains_xyz", straintype_);
  }

  // write structural rotation tensor R
  if (field->problem()->struct_rot() == "yes") post_stress("rotation", "cxyz");

  // Write element and node owners
  write_element_results(field);
  //  WriteNodeResults(field); // Uncomment to visualize owner proc on a nodal basis

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureFilter::write_all_results_one_time_step(
    PostResult& result, bool firststep, bool laststep)
{
  writer_->write_result_one_time_step(result, "displacement", "displacement", dofbased,
      result.field()->problem()->num_dim(), firststep, laststep);
  write_node_results(result.field());
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void MortarFilter::write_all_results(PostField* field)
{
  writer_->write_result("displacement", "displacement", dofbased, field->problem()->num_dim());

  writer_->write_result(
      "norcontactstress", "norcontactstress", dofbased, field->problem()->num_dim());
  writer_->write_result(
      "tancontactstress", "tancontactstress", dofbased, field->problem()->num_dim());

  writer_->write_result(
      "interfacetraction", "interfacetraction", dofbased, field->problem()->num_dim());

  writer_->write_result("slaveforces", "slaveforces", dofbased, field->problem()->num_dim());
  writer_->write_result("masterforces", "masterforces", dofbased, field->problem()->num_dim());

  writer_->write_result("norslaveforce", "norslaveforce", dofbased, field->problem()->num_dim());
  writer_->write_result("tanslaveforce", "tanslaveforce", dofbased, field->problem()->num_dim());

  writer_->write_result("normasterforce", "normasterforce", dofbased, field->problem()->num_dim());
  writer_->write_result("tanmasterforce", "tanmasterforce", dofbased, field->problem()->num_dim());

  write_node_results(field);
  write_element_results(field);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FluidFilter::write_all_results(PostField* field)
{
  writer_->write_result("averaged_pressure", "averaged_pressure", dofbased, 1);
  writer_->write_result(
      "averaged_velnp", "averaged_velocity", dofbased, field->problem()->num_dim());
  writer_->write_result(
      "averaged_scanp", "averaged_scalar_field", dofbased, 1, field->problem()->num_dim());
  writer_->write_result("filteredvel", "filteredvel", dofbased, field->problem()->num_dim());
  writer_->write_result("fsvelaf", "fsvel", dofbased, field->problem()->num_dim());
  writer_->write_result("velnp", "velocity", dofbased, field->problem()->num_dim());
  writer_->write_result("pressure", "pressure", dofbased, 1);
  writer_->write_result("scalar_field", "scalar_field", dofbased, 1);
  writer_->write_result("residual", "residual", dofbased, field->problem()->num_dim());
  writer_->write_result("dispnp", "ale_displacement", dofbased, field->problem()->num_dim());
  writer_->write_result("idispnfull", "ale_idisp", dofbased, field->problem()->num_dim());
  writer_->write_result("traction", "traction", dofbased, field->problem()->num_dim());
  writer_->write_result("wss", "wss", dofbased, field->problem()->num_dim());
  writer_->write_result("wss_mean", "wss_mean", dofbased, field->problem()->num_dim());

  // for xwall
  writer_->write_result("xwall_enrvelnp", "xwall_enrvelnp", dofbased, field->problem()->num_dim());
  writer_->write_result("xwall_tauw", "xwall_tauw", nodebased, 1);
  //  writer_->WriteResult("radii", "radii", nodebased, 1);
  writer_->write_result("par_vel", "par_vel", dofbased, field->problem()->num_dim());

  // additional forces due to lung fsi (volume constraint)
  writer_->write_result("Add_Forces", "Add_Forces", dofbased, field->problem()->num_dim());

  // additional output for poro problems
  writer_->write_result("convel", "convective_velocity", dofbased, field->problem()->num_dim());
  writer_->write_result("gridv", "grid_velocity", dofbased, field->problem()->num_dim());

  // additional fields due to adjoint equations
  writer_->write_result("adjoint_velnp", "adjoint_velocity", dofbased, field->problem()->num_dim());
  writer_->write_result("adjoint_pressure", "adjoint_pressure", dofbased, 1);

  // additional fields for meshfree problems
  writer_->write_result(
      "velatmeshfreenodes", "velocity_atnodes", dofbased, field->problem()->num_dim());
  writer_->write_result("pressureatmeshfreenodes", "pressure_atnodes", dofbased, 1);

  // Lagrange multiplier at the interface in monolithic fsi
  writer_->write_result("fsilambda", "fsilambda", dofbased, field->problem()->num_dim());

  // additional output for biofilm problems
  writer_->write_result(
      "fld_growth_displ", "fld_growth_displ", dofbased, field->problem()->num_dim());

  // additional output for HDG
  writer_->write_result("velnp_hdg", "velocity", nodebased, field->problem()->num_dim());
  writer_->write_result("pressure_hdg", "pressure", nodebased, 1);
  writer_->write_result("tracevel_hdg", "trace_velocity", nodebased, field->problem()->num_dim());


  // additional output for XFluid level-set boundaries: write combined level-set field created from
  // all level-set coupling objects
  writer_->write_result("fluid_levelset_boundary", "combined_ls_bound", nodebased, 1);

  // additional output for XFluid level-set boundaries: write single level-set fields for all
  // level-set coupling objects
  int num_levelsets = 20;

  for (int k = 0; k < num_levelsets; k++)
  {
    std::string name = "phinp_" + std::to_string(k);
    writer_->write_result(name, name, nodebased, 1);
  }

  write_element_results(field);
  write_node_results(field);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void XFluidFilter::write_all_results(PostField* field)
{
  // XFEM has changing number of degrees of freedoms
  // - restart vectors are of changing size
  // - output vectors (e.g. Paraview) have fixed size with 4 DOF per node)
  //   and are named "*_smoothed" for this reason (no integration cells)
  // calling both at the same time will crash, since restart vectors do not fit
  // the 4 DOF per node pattern.

  writer_->write_result(
      "velocity_smoothed", "velocity_smoothed", dofbased, field->problem()->num_dim());
  writer_->write_result("pressure_smoothed", "pressure_smoothed", dofbased, 1);
  writer_->write_result(
      "averaged_velnp", "averaged_velocity", dofbased, field->problem()->num_dim());
  writer_->write_result("averaged_pressure", "averaged_pressure", dofbased, 1);
  writer_->write_result("fsvelocity", "fsvelocity", dofbased, field->problem()->num_dim());

  write_element_results(field);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void InterfaceFilter::write_all_results(PostField* field)
{
  writer_->write_result("idispnp", "idispnp", dofbased, field->problem()->num_dim());
  writer_->write_result("idispn", "idispn", dofbased, field->problem()->num_dim());
  writer_->write_result("ivelnp", "ivelnp", dofbased, field->problem()->num_dim());
  writer_->write_result("iveln", "iveln", dofbased, field->problem()->num_dim());
  writer_->write_result("ivelnm", "ivelnm", dofbased, field->problem()->num_dim());
  writer_->write_result("iaccn", "iaccn", dofbased, field->problem()->num_dim());
  writer_->write_result("itrueresnp", "itrueresnp", dofbased, field->problem()->num_dim());
  write_element_results(field);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void AleFilter::write_all_results(PostField* field)
{
  writer_->write_result("dispnp", "displacement", dofbased, field->problem()->num_dim());
  writer_->write_result("det_j", "det_j", elementbased, 1);
  writer_->write_result("element_quality", "element_quality", elementbased, 1);
  write_element_results(field);
}


/*----------------------------------------------------------------------*
|                                                           wirtz 11/15 |
\*----------------------------------------------------------------------*/
void LubricationFilter::write_all_results(PostField* field)
{
  writer_->write_result("prenp", "pressure", dofbased, 1);
  writer_->write_result("height", "height", dofbased, 1);
  writer_->write_result("no_gap_DBC", "no_gap_DBC", dofbased, 1);
  writer_->write_result("dispnp", "displacement", nodebased, 3);

  writer_->write_result("viscosity", "viscosity", dofbased, 1);

  // write element results (e.g. element owner)
  write_element_results(field);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ThermoFilter::write_all_results(PostField* field)
{
  // number of dofs per node in thermal problems is always 1
  const int numdofpernode = 1;

  // write temperature
  writer_->write_result("temperature", "temperature", dofbased, numdofpernode);

  // write temperature rate
  // writer_->WriteResult("rate", "rate", dofbased, numdofpernode);

  if (heatfluxtype_ != "none")
  {
    // although appearing here twice, only one function call to post_heatflux
    // is really postprocessing Gauss point heatfluxes, since only _either_
    // Current _or_ Initial heatfluxes are written during simulation!
    post_heatflux("gauss_current_heatfluxes_xyz", heatfluxtype_);
    post_heatflux("gauss_initial_heatfluxes_xyz", heatfluxtype_);
    writer_->write_result("heatflux", "heatflux", nodebased, field->problem()->num_dim());
  }
  if (tempgradtype_ != "none")
  {
    // although appearing here twice, only one function call to post_heatflux
    // is really postprocessing Gauss point temperature gradients, since only _either_
    // Initial _or_ Current temperature gradients are written during simulation!
    post_heatflux("gauss_current_tempgrad_xyz", tempgradtype_);
    post_heatflux("gauss_initial_tempgrad_xyz", tempgradtype_);
    writer_->write_result("tempgrad", "tempgrad", nodebased, field->problem()->num_dim());
  }

  // write displacement field
  writer_->write_result("displacement", "displacement", nodebased, field->problem()->num_dim());

  // special information for SLM
  writer_->write_result("phase", "phase", dofbased, numdofpernode);
  writer_->write_result("conductivity", "conductivity", dofbased, numdofpernode);
  writer_->write_result("capacity", "capacity", dofbased, numdofpernode);

  // write element results (e.g. element owner)
  write_element_results(field);

}  // ThermoFilter::WriteAllResults

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void AnyFilter::write_all_results(PostField* field)
{
  write_dof_results(field);
  write_node_results(field);
  write_element_results(field);
}

FOUR_C_NAMESPACE_CLOSE
