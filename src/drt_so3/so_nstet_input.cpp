/*!----------------------------------------------------------------------**###
\file so_nstet_input.cpp

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "so_nstet.H"
#include "../drt_lib/drt_linedefinition.H"
#include "../drt_mat/elasthyper.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::ELEMENTS::NStet::ReadElement(const std::string& eletype,
                                      const std::string& distype,
                                      DRT::INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT",material);
  SetMaterial(material);

  if (Material()->MaterialType() == INPAR::MAT::m_elasthyper){
    MAT::ElastHyper* elahy = static_cast <MAT::ElastHyper*>(Material().get());
    elahy->Setup(linedef);
  }

  std::string buffer;
   linedef->ExtractString("KINEM",buffer);
   if (buffer=="linear")
   {
     // kintype_ not yet implemented for nstet5
     //kintype_ = sonstet5_linear;
     dserror("Reading of SO_NSTET element failed only nonlinear kinematics implemented");
   }
   else if (buffer=="nonlinear")
   {
     // kintype_ not yet implemented for nstet5
     //kintype_ = sonstet5_nonlinear;
   }
   else dserror ("Reading SO_NSTET element failed KINEM unknown");

  return true;
}


#endif  // #ifdef CCADISCRET
