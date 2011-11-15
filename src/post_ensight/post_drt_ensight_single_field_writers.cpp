/*!
  \file post_drt_ensight_single_field_writers.cpp

  \brief main routine of the Ensight filter

  <pre>
  Maintainer: Axel Gerstenberger
  gerstenberger@lnm.mw.tum.de
  http://www.lnm.mw.tum.de/Members/gerstenberger
  089 - 289-15236
  </pre>

*/

#ifdef CCADISCRET

#include "post_drt_ensight_single_field_writers.H"
#include <string>



/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureEnsightWriter::WriteAllResults(PostField* field)
{
  EnsightWriter::WriteResult("displacement", "displacement", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("prolongated_gauss_2PK_stresses_xyz", "prolongated_gauss_2PK_stresses_xyz", nodebased,6);
  EnsightWriter::WriteResult("prolongated_gauss_GL_strains_xyz", "prolongated_gauss_GL_strains_xyz", nodebased,6);
  //EnsightWriter::WriteResult("velocity", "velocity", dofbased, field->problem()->num_dim());
  //EnsightWriter::WriteResult("acceleration", "acceleration", dofbased, field->problem()->num_dim());
  // Statistical Output from MLMC
  EnsightWriter::WriteResult("mean_displacements", "mean_displacement", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("variance_displacements", "variance_displacement", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("mean_gauss_2PK_stresses_xyz", "mean_gauss_2PK_stresses_xyz", nodebased,6);
  EnsightWriter::WriteResult("variance_gauss_2PK_stresses_xyz", "variance_gauss_2PK_stresses_xyz", nodebased,6);
  EnsightWriter::WriteResult("mean_gauss_GL_strain_xyz", "mean_gauss_GL_strain_xyz", nodebased,6);
  EnsightWriter::WriteResult("variance_gauss_GL_strain_xyz", "variance_gauss_GL_strain_xyz", nodebased,6);

  EnsightWriter::WriteResult("diff_to_ll_displacement", "diff_to_ll_displacement", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("diff_to_ll_prolongated_gauss_2PK_stresses_xyz", "diff_to_ll_prolongated_gauss_2PK_stresses_xyz", nodebased,6);
  EnsightWriter::WriteResult("diff_to_ll_prolongated_gauss_GL_strains_xyz", "diff_to_ll_prolongated_gauss_GL_strains_xyz", nodebased,6);

  EnsightWriter::WriteResult("diff_mean_displacements", "diff_mean_displacement", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("diff_variance_displacements", "diff_variance_displacement", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("diff_mean_gauss_2PK_stresses_xyz", "diff_mean_gauss_2PK_stresses_xyz", nodebased,6);
  EnsightWriter::WriteResult("diff_variance_gauss_2PK_stresses_xyz", "diff_variance_gauss_2PK_stresses_xyz", nodebased,6);
  EnsightWriter::WriteResult("diff_mean_gauss_GL_strain_xyz", "diff_mean_gauss_GL_strain_xyz", nodebased,6);
  EnsightWriter::WriteResult("diff_variance_gauss_GL_strain_xyz", "diff_variance_gauss_GL_strain_xyz", nodebased,6);


   // contact and meshtying results
  EnsightWriter::WriteResult("activeset", "activeset", nodebased,1);
  EnsightWriter::WriteResult("norcontactstress", "norcontactstress", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("tancontactstress", "tancontactstress", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("interfacetraction", "interfacetraction", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("slaveforces", "slaveforces", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("masterforces", "masterforces", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("norslaveforce", "norslaveforce", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("tanslaveforce", "tanslaveforce", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("normasterforce", "normasterforce", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("tanmasterforce", "tanmasterforce", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("wear", "wear", dofbased, field->problem()->num_dim());

  // thermo results
  EnsightWriter::WriteResult("temperature", "temperature", nodebased, 1);
  
  // one-dimensional artery
  EnsightWriter::WriteResult("one_d_artery_pressure", "pressure", nodebased, 1);
  EnsightWriter::WriteResult("one_d_artery_flow", "flow", nodebased, 1);

  // reduced dimensional airway
  EnsightWriter::WriteResult("pnp", "pressure", dofbased, 1);
  EnsightWriter::WriteResult("NodeIDs", "NodeIDs", dofbased, 1);
  EnsightWriter::WriteResult("radii", "radii", dofbased, 1);
  EnsightWriter::WriteResult("acini_volume", "acini_volume", dofbased, 1);
  EnsightWriter::WriteResult("acin_bc", "acini_bc", elementbased, 1);
  EnsightWriter::WriteResult("qin_np", "flow_in", elementbased, 1);
  EnsightWriter::WriteResult("qout_np", "flow_out", elementbased, 1);
  EnsightWriter::WriteResult("generations", "generations", elementbased, 1);

  // additional forces due to lung fsi (volume constraint)
  EnsightWriter::WriteResult("Add_Forces", "Add_Forces", dofbased, field->problem()->num_dim());

  EnsightWriter::WriteElementResults(field); //To comment
  if (stresstype_!="none")
  {
    // although appearing here twice, only one function call to PostStress
    // is really postprocessing Gauss point stresses, since only _either_
    // Cauchy _or_ 2nd Piola-Kirchhoff stresses are written during simulation!
    PostStress("gauss_cauchy_stresses_xyz", stresstype_);
    PostStress("gauss_2PK_stresses_xyz", stresstype_);
  }
  if (straintype_!="none")
  {
    // although appearing here twice, only one function call to PostStress
    // is really postprocessing Gauss point strains, since only _either_
    // Green-Lagrange _or_ Euler-Almansi strains are written during simulation!
    PostStress("gauss_GL_strains_xyz", straintype_);
    PostStress("gauss_EA_strains_xyz", straintype_);
    // the same for plastic strains
    PostStress("gauss_pl_GL_strains_xyz", straintype_);
    PostStress("gauss_pl_EA_strains_xyz", straintype_);
  }
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FluidEnsightWriter::WriteAllResults(PostField* field)
{
  EnsightWriter::WriteResult("averaged_pressure", "averaged_pressure", dofbased, 1);
  EnsightWriter::WriteResult("averaged_velnp", "averaged_velocity", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("averaged_scanp", "averaged_scalar_field", dofbased, 1,field->problem()->num_dim());
  EnsightWriter::WriteResult("filteredvel", "filteredvel", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("fsvelaf", "fsvel" , dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("velnp", "velocity", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("pressure", "pressure", dofbased, 1);
  EnsightWriter::WriteResult("scalar_field", "scalar_field", dofbased, 1);
  EnsightWriter::WriteResult("residual", "residual", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("dispnp", "ale_displacement", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("idispnfull", "ale_idisp", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("traction", "traction", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("wss", "wss", dofbased, field->problem()->num_dim());
  //  EnsightWriter::WriteResult("radii", "radii", nodebased, 1);
  EnsightWriter::WriteResult("par_vel", "par_vel", dofbased, field->problem()->num_dim());

  //additional output for turbulent flows (subfilter/-gridstress)
  EnsightWriter::WriteResult("sfs11", "sfs11", nodebased, 1);
  EnsightWriter::WriteResult("sfs12", "sfs12", nodebased, 1);
  EnsightWriter::WriteResult("sfs13", "sfs13", nodebased, 1);
  EnsightWriter::WriteResult("sfs22", "sfs22", nodebased, 1);
  EnsightWriter::WriteResult("sfs23", "sfs23", nodebased, 1);
  EnsightWriter::WriteResult("sfs33", "sfs33", nodebased, 1);

  // additional forces due to lung fsi (volume constraint)
  EnsightWriter::WriteResult("Add_Forces", "Add_Forces", dofbased, field->problem()->num_dim());

  WriteElementResults(field);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void XFluidEnsightWriter::WriteAllResults(PostField* field)
{
  // XFEM has changing number of degrees of freedoms
  // - restart vectors are of changing size
  // - output vectors (e.g. Paraview) have fixed size with 4 DOF per node)
  //   and are named "*_smoothed" for this reason (no integration cells)
  // calling both at the same time will crash, since restart vectors do not fit
  // the 4 DOF per node pattern. BACI will produce consistent naming now, but for old
  // data you can switch the naming convention here (old naming will be removed soon)

  const bool consistent_naming = true;

  if (consistent_naming)
  {
    EnsightWriter::WriteResult("velocity_smoothed", "velocity_smoothed", dofbased, field->problem()->num_dim());
    EnsightWriter::WriteResult("pressure_smoothed", "pressure_smoothed", dofbased, 1);

    // for diffusion problem, if ever needed
    EnsightWriter::WriteResult("temperature_smoothed", "temperature_smoothed", dofbased, 1);

  }
  else
  {
    cout << "Depreciated naming convention!!!" << endl;
    // note old output files might still use the name names velnp and pressure
    // just turn the following lines on
    EnsightWriter::WriteResult("velnp", "velocity", dofbased, field->problem()->num_dim());
    EnsightWriter::WriteResult("pressure", "pressure", dofbased, 1);
    EnsightWriter::WriteResult("tract_resid", "tract_residual", dofbased, field->problem()->num_dim());
  }

  WriteElementResults(field);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void InterfaceEnsightWriter::WriteAllResults(PostField* field)
{
  EnsightWriter::WriteResult("idispnp", "idispnp", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("idispn", "idispn", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("ivelnp", "ivelnp", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("iveln", "iveln", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("ivelnm", "ivelnm", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("iaccn", "iaccn", dofbased, field->problem()->num_dim());
  EnsightWriter::WriteResult("itrueresnp", "itrueresnp", dofbased, field->problem()->num_dim());
  WriteElementResults(field);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void AleEnsightWriter::WriteAllResults(PostField* field)
{
  EnsightWriter::WriteResult("dispnp", "displacement", dofbased, field->problem()->num_dim());
  WriteElementResults(field);
}


/*----------------------------------------------------------------------*
|                                                           gjb 12/07   |
\*----------------------------------------------------------------------*/
void ScaTraEnsightWriter::WriteAllResults(PostField* field)
{
  //compute number of dofs per node (ask the first node)
  int numdofpernode = field->discretization()->NumDof(field->discretization()->lRowNode(0));

  // write results for each transported scalar
  if (numdofpernode == 1)
  {
    EnsightWriter::WriteResult("phinp","phi",dofbased,1);
    EnsightWriter::WriteResult("averaged_phinp","averaged_phi",dofbased,1);
    EnsightWriter::WriteResult("normalflux","normalflux",dofbased,1);
    // write flux vectors (always 3D)
    EnsightWriter::WriteResult("flux", "flux", nodebased, 3);
  }
  else
  {
    for(int k = 1; k <= numdofpernode; k++)
    {
      ostringstream temp;
      temp << k;
      string name = "phi_"+temp.str();
      EnsightWriter::WriteResult("phinp", name, dofbased, 1,k-1);
      EnsightWriter::WriteResult("averaged_phinp", "averaged_"+name, dofbased, 1,k-1);
      // intermediate work-around for nurbs discretizations (no normal vectors applied)
      EnsightWriter::WriteResult("normalflux","normalflux"+name,dofbased,1,k-1);
      // write flux vectors (always 3D)
      EnsightWriter::WriteResult("flux_"+name, "flux_"+name, nodebased, 3);
    }
  }

  // write velocity field (always 3D)
  EnsightWriter::WriteResult("convec_velocity", "velocity", nodebased, 3);

  // write displacement field (always 3D)
  EnsightWriter::WriteResult("dispnp", "ale-displacement", nodebased, 3);

  // write element results (e.g. element owner)
  WriteElementResults(field);
}


/*----------------------------------------------------------------------*
|                                                             gjb 09/08 |
\*----------------------------------------------------------------------*/
void ElchEnsightWriter::WriteAllResults(PostField* field)
{
  //compute number of dofs per node (ask the first node)
  int numdofpernode = field->discretization()->NumDof(field->discretization()->lRowNode(0));

  // write results for each transported scalar
  if (numdofpernode == 1)
  {
    // do the single ion concentration
      string name = "c_1";
      EnsightWriter::WriteResult("phinp", name, dofbased, 1, 0);
      // write flux vectors (always 3D)
      EnsightWriter::WriteResult("flux", "flux", nodebased, 3);

      // there is no electric potential in this special case

      // temporal mean field from turbulent statistics (if present)
      EnsightWriter::WriteResult("averaged_phinp", "averaged_"+name, dofbased, 1, 0);
  }
  else
  {
    // do the ion concentrations first
    for(int k = 1; k < numdofpernode; k++)
    {
      ostringstream temp;
      temp << k;
      string name = "c_"+temp.str();
      EnsightWriter::WriteResult("phinp", name, dofbased, 1,k-1);
      // write flux vectors (always 3D)
      EnsightWriter::WriteResult("flux_phi_"+temp.str(), "flux_"+name, nodebased, 3);

      // temporal mean field from turbulent statistics (if present)
      EnsightWriter::WriteResult("averaged_phinp", "averaged_"+name, dofbased, 1, k-1);
    }
    // finally, handle the electric potential
    EnsightWriter::WriteResult("phinp", "phi", dofbased, 1,numdofpernode-1);
    // temporal mean field from turbulent statistics (if present)
    EnsightWriter::WriteResult("averaged_phinp", "averaged_phi", dofbased, 1,numdofpernode-1);
  }

  // write velocity field (always 3D)
  EnsightWriter::WriteResult("convec_velocity", "velocity", nodebased, 3);

  // write displacement field (always 3D)
  EnsightWriter::WriteResult("dispnp", "ale-displacement", nodebased, 3);

  // write magnetic field (always 3D)
  EnsightWriter::WriteResult("magnetic_field", "B", nodebased, 3);

  // write element results (e.g. element owner)
  WriteElementResults(field);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void ThermoEnsightWriter::WriteAllResults(PostField* field)
{
  // number of dofs per node in thermal problems is always 1
  const int numdofpernode = 1;

  // write temperature
  EnsightWriter::WriteResult("temperature", "temperature", dofbased, numdofpernode);

  // write temperature rate
  //EnsightWriter::WriteResult("rate", "rate", dofbased, numdofpernode);

  // write element results (e.g. element owner)
  EnsightWriter::WriteElementResults(field);

  if (heatfluxtype_ != "none")
  {
    // although appearing here twice, only one function call to PostHeatflux
    // is really postprocessing Gauss point heatfluxes, since only _either_
    // Current _or_ Initial heatfluxes are written during simulation!
    PostHeatflux("gauss_current_heatfluxes_xyz", heatfluxtype_);
    PostHeatflux("gauss_initial_heatfluxes_xyz", heatfluxtype_);
    EnsightWriter::WriteResult("heatflux", "heatflux", nodebased, field->problem()->num_dim());
  }
  if (tempgradtype_ != "none")
  {
    // although appearing here twice, only one function call to PostHeatflux
    // is really postprocessing Gauss point temperature gradients, since only _either_
    // Initial _or_ Current temperature gradients are written during simulation!
    PostHeatflux("gauss_current_tempgrad_xyz", tempgradtype_);
    PostHeatflux("gauss_initial_tempgrad_xyz", tempgradtype_);
    EnsightWriter::WriteResult("tempgrad", "tempgrad", nodebased, field->problem()->num_dim());
  }

} // ThermoEnsightWriter::WriteAllResults

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void AnyEnsightWriter::WriteAllResults(PostField* field)
{
  WriteDofResults(field);
  WriteNodeResults(field);
  WriteElementResults(field);
}


#endif
