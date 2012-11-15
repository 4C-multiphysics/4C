/*!----------------------------------------------------------------------
\file so_hex27_multiscale.cpp
\brief

<pre>
Maintainer: Lena Yoshihara
            yoshihara@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15303
</pre>

*----------------------------------------------------------------------*/


#include "so_hex27.H"
#include "../drt_mat/micromaterial.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_comm/comm_utils.H"
#include "../drt_lib/drt_discret.H"



/*----------------------------------------------------------------------*
 |  homogenize material density (public)                                |
 *----------------------------------------------------------------------*/
// this routine is intended to determine a homogenized material
// density for multi-scale analyses by averaging over the initial volume

void DRT::ELEMENTS::So_hex27::soh27_homog(ParameterList&  params)
{
  if(DRT::Problem::Instance(0)->GetNPGroup()->SubComm()->MyPID() == Owner())
  {
    double homogdens = 0.;
    const static std::vector<double> weights = soh27_weights();
    const double density = Material()->Density();

    for (int gp=0; gp<NUMGPT_SOH27; ++gp)
    {
      homogdens += detJ_[gp] * weights[gp] * density;
    }

    double homogdensity = params.get<double>("homogdens", 0.0);
    params.set("homogdens", homogdensity+homogdens);
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Read restart on the microscale                                      |
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex27::soh27_read_restart_multi()
{
  RCP<MAT::Material> mat = Material();

  if (mat->MaterialType() == INPAR::MAT::m_struct_multiscale)
  {
    MAT::MicroMaterial* micro = static_cast <MAT::MicroMaterial*>(mat.get());
    int eleID = Id();
    bool eleowner = false;
    if (DRT::Problem::Instance()->GetDis("structure")->Comm().MyPID()==Owner()) eleowner = true;

    for (int gp=0; gp<NUMGPT_SOH27; ++gp)
      micro->ReadRestart(gp, eleID, eleowner);
  }

  return;
}



