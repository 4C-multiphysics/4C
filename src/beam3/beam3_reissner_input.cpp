/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief input related methods of 3D nonlinear Reissner beam element

\level 2

*/
/*-----------------------------------------------------------------------------------------------*/

#include "beam3_reissner.H"

#include "mat_material.H"
#include "mat_par_parameter.H"

#include "lib_linedefinition.H"

#include "fem_general_largerotations.H"

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
bool DRT::ELEMENTS::Beam3r::ReadElement(
    const std::string& eletype, const std::string& distype, DRT::INPUT::LineDefinition* linedef)
{
  /* the triad field is discretized with Lagrange polynomials of order NumNode()-1;
   * the centerline is either discretized in the same way or with 3rd order Hermite polynomials;
   * we thus make a difference between nnodetriad and nnodecl;
   * assumptions: nnodecl<=nnodetriad
   * first nodes with local ID 0...nnodecl-1 are used for interpolation of centerline AND triad
   * field*/
  const int nnodetriad = NumNode();


  // read number of material model and cross-section specs
  int material = 0;
  linedef->ExtractInt("MAT", material);
  SetMaterial(material);

  if (Material()->Parameter()->Name() != "MAT_BeamReissnerElastHyper" and
      Material()->Parameter()->Name() != "MAT_BeamReissnerElastHyper_ByModes" and
      Material()->Parameter()->Name() != "MAT_BeamReissnerElastPlastic")
  {
    dserror(
        "The material parameter definition '%s' is not supported by Beam3r element! "
        "Choose MAT_BeamReissnerElastHyper, MAT_BeamReissnerElastHyper_ByModes or "
        "MAT_BeamReissnerElastPlastic!",
        Material()->Parameter()->Name().c_str());
  }

  if (linedef->HaveNamed("HERM2LINE2") or linedef->HaveNamed("HERM2LINE3") or
      linedef->HaveNamed("HERM2LINE4") or linedef->HaveNamed("HERM2LINE5"))
    centerline_hermite_ = true;
  else
    centerline_hermite_ = false;

  // read whether automatic differentiation via Sacado::Fad package shall be used
  useFAD_ = linedef->HaveNamed("FAD") ? true : false;


  // store nodal triads according to input file
  theta0node_.resize(nnodetriad);

  /* Attention! expression "TRIADS" in input file is misleading.
   * The 3 specified values per node define a rotational pseudovector, which
   * parameterizes the orientation of the triad at this node
   * (relative to the global reference coordinate system)*/
  /* extract rotational pseudovectors at element nodes in reference configuration
   *  and save them as quaternions at each node, respectively*/
  std::vector<double> nodal_rotvecs;
  linedef->ExtractDoubleVector("TRIADS", nodal_rotvecs);

  for (int node = 0; node < nnodetriad; node++)
    for (int dim = 0; dim < 3; dim++) theta0node_[node](dim) = nodal_rotvecs[3 * node + dim];

  return true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
void DRT::ELEMENTS::Beam3r::SetCenterlineHermite(const bool yesno) { centerline_hermite_ = yesno; }
