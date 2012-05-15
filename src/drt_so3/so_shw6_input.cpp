/*!----------------------------------------------------------------------
\file so_shw6_input.cpp
\brief

<pre>
Maintainer: Moritz Frenzel
            frenzel@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15240
</pre>

*----------------------------------------------------------------------*/


#include "so_shw6.H" //**
#include "../drt_mat/artwallremod.H"
#include "../drt_mat/viscoanisotropic.H"
#include "../drt_mat/elasthyper.H"
#include "../drt_lib/drt_linedefinition.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::ELEMENTS::So_shw6::ReadElement(const std::string& eletype,
                                         const std::string& distype,
                                         DRT::INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT",material);
  SetMaterial(material);

  // special element-dependent input of material parameters
  if (Material()->MaterialType() == INPAR::MAT::m_artwallremod){
    MAT::ArtWallRemod* remo = static_cast <MAT::ArtWallRemod*>(Material().get());
    remo->Setup(NUMGPT_WEG6, this->Id(), linedef);
  } else if (Material()->MaterialType() == INPAR::MAT::m_viscoanisotropic){
    MAT::ViscoAnisotropic* visco = static_cast <MAT::ViscoAnisotropic*>(Material().get());
    visco->Setup(NUMGPT_WEG6, linedef);
  } else if (Material()->MaterialType() == INPAR::MAT::m_elasthyper){
    MAT::ElastHyper* elahy = static_cast <MAT::ElastHyper*>(Material().get());
    elahy->Setup(linedef);
  }

  std::string buffer;
  linedef->ExtractString("KINEM",buffer);


  // geometrically non-linear with Total Lagrangean approach
  if (buffer=="nonlinear")
    {
      kintype_ = sow6_nonlinear;

    }
  // geometrically linear
  else if (buffer=="linear")
    {
    kintype_ = sow6_linear;
    dserror("Reading of SOLIDSHW6 element failed onlz nonlinear kinetmatics implemented");
    }

  // geometrically non-linear with Updated Lagrangean approach
  else dserror("Reading of SOLIDSHW6 element failed KINEM unknown");

  linedef->ExtractString("EAS",buffer);

  // full sohw6 EAS technology
  if      (buffer=="soshw6")
  {
    eastype_ = soshw6_easpoisthick; // EAS to allow linear thickness strain
    neas_ = soshw6_easpoisthick;    // number of eas parameters
    soshw6_easinit();
  }
  // no EAS technology
  else if (buffer=="none")
  {
    eastype_ = soshw6_easnone;
    neas_ = 0;    // number of eas parameters
    cout << "Warning: Solid-Shell Wegde6 without EAS" << endl;
  }
  else dserror("Reading of SOLIDSHW6 EAS technology failed");

  // check for automatically align material space optimally with parameter space
  optimal_parameterspace_map_ = false;
  nodes_rearranged_ = false;
  if (linedef->HaveNamed("OPTORDER"))
    optimal_parameterspace_map_ = true;

  return true;
}
