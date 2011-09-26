/*!----------------------------------------------------------------------
\file so_hex8_multiscale.cpp
\brief

<pre>
Maintainer: Lena Wiechert
            wiechert@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15303
</pre>

*----------------------------------------------------------------------*/
#ifdef D_SOLID3
#ifdef CCADISCRET
#include "so_hex8.H"
#include "../drt_lib/drt_utils.H"
#include "Epetra_SerialDenseSolver.h"
#include "../drt_mat/micromaterial.H"
#include "../drt_lib/drt_globalproblem.H"

using namespace std; // cout etc.

extern struct _GENPROB     genprob;


/*----------------------------------------------------------------------*
 |  homogenize material density (public)                        lw 07/07|
 *----------------------------------------------------------------------*/
// this routine is intended to determine a homogenized material
// density for multi-scale analyses by averaging over the initial volume

void DRT::ELEMENTS::So_hex8::soh8_homog(ParameterList&  params)
{
  double homogdens = 0.;
  const static std::vector<double> weights = soh8_weights();
  const double density = Material()->Density();

  for (int gp=0; gp<NUMGPT_SOH8; ++gp)
  {
    homogdens += detJ_[gp] * weights[gp] * density;
  }

  double homogdensity = params.get<double>("homogdens", 0.0);
  params.set("homogdens", homogdensity+homogdens);

  return;
}


/*----------------------------------------------------------------------*
 |  Set EAS internal variables on the microscale (public)       lw 04/08|
 *----------------------------------------------------------------------*/
// the microscale internal EAS data have to be saved separately for every
// macroscopic Gauss point and set before the determination of microscale
// stiffness etc.

void DRT::ELEMENTS::So_hex8::soh8_set_eas_multi(ParameterList&  params)
{
  if (eastype_ != soh8_easnone)
  {
    RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > oldalpha =
      params.get<RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > >("oldalpha", null);
    RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > oldfeas =
      params.get<RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > >("oldfeas", null);
    RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > oldKaainv =
      params.get<RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > >("oldKaainv", null);
    RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > oldKda =
      params.get<RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > >("oldKda", null);

    if (oldalpha==null || oldfeas==null || oldKaainv==null || oldKda==null)
      dserror("Cannot get EAS internal data from parameter list for multi-scale problems");

    data_.Add("alpha", (*oldalpha)[Id()]);
    data_.Add("feas", (*oldfeas)[Id()]);
    data_.Add("invKaa", (*oldKaainv)[Id()]);
    data_.Add("Kda", (*oldKda)[Id()]);
  }
  return;
}


/*----------------------------------------------------------------------*
 |  Initialize EAS internal variables on the microscale         lw 03/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex8::soh8_eas_init_multi(ParameterList&  params)
{
  if (eastype_ != soh8_easnone)
  {
    RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > lastalpha =
      params.get<RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > >("lastalpha", null);
    RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > oldalpha =
      params.get<RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > >("oldalpha", null);
    RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > oldfeas =
      params.get<RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > >("oldfeas", null);
    RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > oldKaainv =
      params.get<RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > >("oldKaainv", null);
    RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > oldKda =
      params.get<RCP<std::map<int, RefCountPtr<Epetra_SerialDenseMatrix> > > >("oldKda", null);

    (*lastalpha)[Id()] = rcp(new Epetra_SerialDenseMatrix(neas_, 1));
    (*oldalpha)[Id()]  = rcp(new Epetra_SerialDenseMatrix(neas_, 1));
    (*oldfeas)[Id()]   = rcp(new Epetra_SerialDenseMatrix(neas_, 1));
    (*oldKaainv)[Id()] = rcp(new Epetra_SerialDenseMatrix(neas_, neas_));
    (*oldKda)[Id()]    = rcp(new Epetra_SerialDenseMatrix(neas_, NUMDOF_SOH8));
  }
  return;
}


/*----------------------------------------------------------------------*
 |  Read restart on the microscale                              lw 05/08|
 *----------------------------------------------------------------------*/
void DRT::ELEMENTS::So_hex8::soh8_read_restart_multi()
{
  RefCountPtr<MAT::Material> mat = Material();

  if (mat->MaterialType() == INPAR::MAT::m_struct_multiscale)
  {
    MAT::MicroMaterial* micro = static_cast <MAT::MicroMaterial*>(mat.get());
    int eleID = Id();
    bool eleowner = false;
    if (DRT::Problem::Instance()->Dis(genprob.numsf,0)->Comm().MyPID()==Owner()) eleowner = true;

    for (int gp=0; gp<NUMGPT_SOH8; ++gp)
      micro->ReadRestart(gp, eleID, eleowner);
  }

  return;
}

#endif
#endif
