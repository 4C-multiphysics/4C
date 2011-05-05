/*----------------------------------------------------------------------*/
/*!
\file stru_ale_utils.cpp

\brief utility functions for structure with ale problems

<pre>
Maintainer: Markus Gitterle
            gitterle@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15251
</pre>
*/
/*----------------------------------------------------------------------*
 | definitions                                               mgit 04/11 |
 *----------------------------------------------------------------------*/
#ifdef CCADISCRET

#ifdef PARALLEL
#include <mpi.h>
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

/*----------------------------------------------------------------------*
 | headers                                                   mgit 04/11 |
 *----------------------------------------------------------------------*/
#include "stru_ale_utils.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_mat/material.H"
#include "../drt_mat/matpar_material.H"
#include "../drt_mat/matpar_bundle.H"
#include "../drt_mat/matpar_parameter.H"
#include "../drt_ale2/ale2.H"

#ifdef PARALLEL
#include <mpi.h>
#endif

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::map<std::string,std::string> STRU_ALE::UTILS::AleStructureCloneStrategy::ConditionsToCopy()
{
  std::map<std::string,std::string> conditions_to_copy;

  // special Thermo conditions
  conditions_to_copy.insert(pair<std::string,std::string>("AleDirichlet","Dirichlet"));
  conditions_to_copy.insert(pair<std::string,std::string>("AlePointNeumann","PointNeumann"));
  conditions_to_copy.insert(pair<std::string,std::string>("AleLineNeumann","LineNeumann"));
  conditions_to_copy.insert(pair<std::string,std::string>("AleSurfaceNeumann","SurfaceNeumann"));
  conditions_to_copy.insert(pair<std::string,std::string>("AleVolumeNeumann","VolumeNeumann"));

  return conditions_to_copy;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void STRU_ALE::UTILS::AleStructureCloneStrategy::CheckMaterialType(const int matid)
{
//  //// We take the material with the ID specified by the user
//  //// Here we check first, whether this material is of admissible type
//  INPAR::MAT::MaterialType mtype = DRT::Problem::Instance()->Materials()->ById(matid)->Type();
//  if ((mtype != INPAR::MAT::m_th_fourier_iso))
//  dserror("Material with ID %d is not admissible for thermo elements",matid);
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void STRU_ALE::UTILS::AleStructureCloneStrategy::SetElementData(
  Teuchos::RCP<DRT::Element> newele,
  DRT::Element* oldele,
  const int matid,
  const bool isnurbs
  )
{
//  // We must not add a new material type here because that might move
//  // the internal material vector. And each element material might
//  // have a pointer to that vector. Too bad.
//  // So we search for a Fourier material and take the first one we find.
//  // => matid from outside remains unused!
//  //const int matnr =
//  //  DRT::Problem::Instance()->Materials()->FirstIdByType(INPAR::MAT::m_th_fourier_iso);
//  //if (matnr==-1)
//  //  dserror("No isotropic Fourier material defined. Cannot generate thermo mesh.");
//
//  // We need to set material and possibly other things to complete element setup.
//  // This is again really ugly as we have to extract the actual
//  // element type in order to access the material property
//
//  RCP<MAT::Material > mat = oldele->Material();
//  const int matnr = (mat->Parameter()->Id())+1;
//  
//  // note: SetMaterial() was reimplemented by the thermo element!
//#if defined(D_THERMO)
//      DRT::ELEMENTS::Ale2* ale2 = dynamic_cast<DRT::ELEMENTS::Ale2*>(newele.get());
//      //DRT::ELEMENTS::Ale2* ale2 = dynamic_cast<DRT::ELEMENTS::Ale2*>(newele.get());
//      if (ale2!=NULL)
//      {
//        ale2->SetMaterial(matnr);
//        //ale2->SetDisType(oldele->Shape()); // set distype as well!
//      }
//      else
//#endif
//    {
//      dserror("unsupported element type '%s'", typeid(*newele).name());
//    }
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool STRU_ALE::UTILS::AleStructureCloneStrategy::DetermineEleType(
  DRT::Element* actele,
  const bool ismyele,
  vector<string>& eletype
  )
{
//  // we only support ale2 elements here
//  eletype.push_back("ALE2");
  
  return true; // yes, we copy EVERY element (no submeshes)
}

/*----------------------------------------------------------------------*/
#endif // CCADISCRET
