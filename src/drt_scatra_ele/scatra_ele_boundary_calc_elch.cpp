/*----------------------------------------------------------------------*/
/*!
\file scatra_ele_boundary_calc_elch.cpp

\brief evaluation of scatra boundary terms at integration points

<pre>
Maintainer: Andreas Ehrl
            ehrl@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15252
</pre>
 */
/*----------------------------------------------------------------------*/
#include "scatra_ele_boundary_calc_elch.H"
#include "scatra_ele_parameter_elch.H"
#include "scatra_ele_utils_elch.H"

#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/standardtypes_cpp.H"

/*----------------------------------------------------------------------*
 | protected constructor for singletons                      fang 01/15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<distype>::ScaTraEleBoundaryCalcElch(const int numdofpernode, const int numscal) :
  // constructor of base class
  my::ScaTraEleBoundaryCalc(numdofpernode,numscal),

  // instance of utility class supporting element evaluation
  utils_(ScaTraEleUtilsElch<distype>::Instance(numdofpernode,numscal))
{
  // replace parameter class for standard scalar transport by parameter class for electrochemistry problems
  my::scatraparams_ = DRT::ELEMENTS::ScaTraEleParameterElch::Instance();

  return;
}


/*----------------------------------------------------------------------*
 | evaluate action                                           fang 02/15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<distype>::EvaluateAction(
    DRT::FaceElement*                   ele,
    Teuchos::ParameterList&             params,
    DRT::Discretization&                discretization,
    SCATRA::BoundaryAction              action,
    std::vector<int>&                   lm,
    Epetra_SerialDenseMatrix&           elemat1_epetra,
    Epetra_SerialDenseMatrix&           elemat2_epetra,
    Epetra_SerialDenseVector&           elevec1_epetra,
    Epetra_SerialDenseVector&           elevec2_epetra,
    Epetra_SerialDenseVector&           elevec3_epetra
)
{
  // determine and evaluate action
  switch(action)
  {
  case SCATRA::bd_calc_elch_boundary_kinetics:
  {
    CalcElchBoundaryKinetics(
        ele,
        params,
        discretization,
        lm,
        elemat1_epetra,
        elevec1_epetra,
        1.
        );

    break;
  }

  case SCATRA::bd_calc_elch_linearize_nernst:
  {
    CalcNernstLinearization(ele,
        params,
        discretization,
        lm,
        elemat1_epetra,
        elevec1_epetra);

    break;
  }

  case SCATRA::bd_calc_elch_cell_voltage:
  {
    CalcCellVoltage(ele,params,discretization,lm,elevec1_epetra);

    break;
  }

  default:
  {
    my::EvaluateAction(
        ele,
        params,
        discretization,
        action,
        lm,
        elemat1_epetra,
        elemat2_epetra,
        elevec1_epetra,
        elevec2_epetra,
        elevec3_epetra
        );

   break;
  }
  } // switch action

  return 0;
}


/*----------------------------------------------------------------------*
 | evaluate electrode kinetics boundary condition                 ehrl  |
 *----------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<distype>::CalcElchBoundaryKinetics(
    DRT::FaceElement*                 ele,
    Teuchos::ParameterList&           params,
    DRT::Discretization&              discretization,
    std::vector<int>&                 lm,
    Epetra_SerialDenseMatrix&         elemat1_epetra,
    Epetra_SerialDenseVector&         elevec1_epetra,
    const double                      scalar
    )
{
  // get actual values of transported scalars
  Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
  if(phinp == Teuchos::null)
    dserror("Cannot get state vector 'phinp'");

  // extract local values from the global vector
  std::vector<double> ephinp(lm.size());
  DRT::UTILS::ExtractMyValues(*phinp,ephinp,lm);

  // get history variable (needed for double layer modeling)
  Teuchos::RCP<const Epetra_Vector> hist = discretization.GetState("hist");
  if(phinp == Teuchos::null)
    dserror("Cannot get state vector 'hist'");

  // extract local values from the global vector
  std::vector<double> ehist(lm.size());
  DRT::UTILS::ExtractMyValues(*hist,ehist,lm);

  // get current condition
  Teuchos::RCP<DRT::Condition> cond = params.get<Teuchos::RCP<DRT::Condition> >("condition");
  if(cond == Teuchos::null)
    dserror("Cannot access condition 'ElchBoundaryKinetics'");

  // access parameters of the condition
  const int                 kinetics = cond->GetInt("kinetic model");
  double                    pot0 = cond->GetDouble("pot");
  const int                 curvenum = cond->GetInt("curve");
  const int                 nume = cond->GetInt("e-");
  // if zero=1=true, the current flow across the electrode is zero (comparable to do-nothing Neuman condition)
  // but the electrode status is evaluated
  const int                 zerocur = cond->GetInt("zero_cur");
  if(nume < 0)
    dserror("The convention for electrochemical reactions at the electrodes does not allow \n"
        "a negative number of transferred electrons");

  // convention for stoichiometric coefficients s_i:
  // Sum_i (s_i  M_i^(z_i)) -> n e- (n needs to be positive)
  const std::vector<int>*   stoich = cond->GetMutable<std::vector<int> >("stoich");
  if((unsigned int)my::numscal_ != (*stoich).size())
    dserror("Electrode kinetics: number of stoichiometry coefficients %u does not match"
            " the number of ionic species %d", (*stoich).size(), my::numscal_);

  // the classical implementations of kinetic electrode models does not support
  // more than one reagent or product!! There are alternative formulations
  // as e.g. Newman (2004), pp. 205, eq. 8.6 with 8.10
  {
    int reactspecies = 0;
    for(int kk=0; kk<my::numscal_; ++kk)
      reactspecies += abs((*stoich)[kk]);

    if(reactspecies>1 and (kinetics==INPAR::SCATRA::butler_volmer or kinetics == INPAR::SCATRA::butler_volmer_yang1997 or
        kinetics == INPAR::SCATRA::tafel or kinetics == INPAR::SCATRA::linear))
      dserror("Kinetic model Butler-Volmer / Butler-Volmer-Yang / Tafel and Linear: \n"
          "Only one educt and no product is allowed in the implemented version");
  }

  // access input parameter
  const double frt = ElchParams()->FRT();
  if (frt<=0.0)
    dserror("A negative factor frt is not possible by definition");

  // get control parameter from parameter list
  const bool   is_stationary = my::scatraparamstimint_->IsStationary();
  const double time = my::scatraparamstimint_->Time();
  double       timefac = 1.0;
  double       rhsfac  = 1.0;
  // find out whether we shell use a time curve and get the factor
  // this feature can be also used for stationary "pseudo time loops"
  if (curvenum>=0)
  {
    const double curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
    // adjust potential at metal side accordingly
    pot0 *= curvefac;
  }

  //TODO: Ist es besser über parameter?
  if(!(params.get<bool>("calc_status",false)))
  {
    if(not is_stationary)
    {
      // One-step-Theta:    timefac = theta*dt
      // BDF2:              timefac = 2/3 * dt
      // generalized-alpha: timefac = (gamma*alpha_F/alpha_M) * dt
      timefac = my::scatraparamstimint_->TimeFac();
      if (timefac < 0.0) dserror("time factor is negative.");
      // for correct scaling of rhs contribution (see below)
      rhsfac =  1/my::scatraparamstimint_->AlphaF();
    }

    if(zerocur == 0)
    {
      EvaluateElchBoundaryKinetics(
          ele,
          elemat1_epetra,
          elevec1_epetra,
          ephinp,
          ehist,
          timefac,
          ele->ParentElement()->Material(),
          cond,
          nume,
          *stoich,
          kinetics,
          pot0,
          frt,
          scalar
      );
    }

    // realize correct scaling of rhs contribution for gen.alpha case
    // with dt*(gamma/alpha_M) = timefac/alpha_F
    // matrix contributions are already scaled correctly with
    // timefac=dt*(gamma*alpha_F/alpha_M)
    elevec1_epetra.Scale(rhsfac);
  }
  else
  {
    // get actual values of transported scalars
    Teuchos::RCP<const Epetra_Vector> phidtnp = discretization.GetState("phidtnp");
    if(phidtnp == Teuchos::null)
      dserror("Cannot get state vector 'ephidtnp'");
    // extract local values from the global vector
    std::vector<double> ephidtnp(lm.size());
    DRT::UTILS::ExtractMyValues(*phidtnp,ephidtnp,lm);

    if(not is_stationary)
    {
      // One-step-Theta:    timefacrhs = theta*dt
      // BDF2:              timefacrhs = 2/3 * dt
      // generalized-alpha: timefacrhs = (gamma/alpha_M) * dt
      timefac = my::scatraparamstimint_->TimeFacRhs();
      if(timefac < 0.)
        dserror("time factor is negative.");
    }

    ElectrodeStatus(
        ele,
        elevec1_epetra,
        params,
        cond,
        ephinp,
        ephidtnp,
        kinetics,
        *stoich,
        nume,
        pot0,
        frt,
        timefac,
        scalar
        );
  }
}


/*----------------------------------------------------------------------*
 | calculate linearization of nernst equation                     ehrl  |
 *----------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<distype>::CalcNernstLinearization(
    DRT::FaceElement*                 ele,
    Teuchos::ParameterList&           params,
    DRT::Discretization&              discretization,
    std::vector<int>&                 lm,
    Epetra_SerialDenseMatrix&         elemat1_epetra,
    Epetra_SerialDenseVector&         elevec1_epetra
    )
{
  Teuchos::RCP<DRT::Condition> cond = params.get<Teuchos::RCP<DRT::Condition> >("condition");
    if (cond == Teuchos::null) dserror("Cannot access condition 'ElchBoundaryKinetics'");

  const int  kinetics = cond->GetInt("kinetic model");

  // Nernst-BC
  if(kinetics == INPAR::SCATRA::nernst)
  {
    // get actual values of transported scalars
    Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (phinp==Teuchos::null) dserror("Cannot get state vector 'phinp'");

    // extract local values from the global vector
    std::vector<double> ephinp(lm.size());
    DRT::UTILS::ExtractMyValues(*phinp,ephinp,lm);

    // access parameters of the condition
    double       pot0 = cond->GetDouble("pot");
    const int    curvenum = cond->GetInt("curve");
    const int    nume = cond->GetInt("e-");
    const double e0 = cond->GetDouble("e0");
    const double c0 = cond->GetDouble("c0");

    if(nume < 0)
      dserror("The convention for electrochemical reactions at the electrodes does not allow \n"
          "a negative number of transferred electrons");

    const std::vector<int>*   stoich = cond->GetMutable<std::vector<int> >("stoich");
    if((unsigned int)my::numscal_ != (*stoich).size())
      dserror("Electrode kinetics: number of stoichiometry coefficients %u does not match"
              " the number of ionic species %d", (*stoich).size(), my::numscal_);

    // access input parameter
    const double frt = ElchParams()->FRT();
    if (frt<0.0)
      dserror("A negative factor frt is not possible by definition");

    const double time = my::scatraparamstimint_->Time();

    if (curvenum>=0)
    {
      const double curvefac = DRT::Problem::Instance()->Curve(curvenum).f(time);
      // adjust potential at metal side accordingly
      pot0 *= curvefac;
    }

    // concentration values of reactive species at element nodes
    std::vector<LINALG::Matrix<my::nen_,1> > conreact(my::numscal_);

    // el. potential values at element nodes
    LINALG::Matrix<my::nen_,1> pot(true);

    for (int inode=0; inode< my::nen_;++inode)
    {
      for(int kk=0; kk<my::numscal_; kk++)
        conreact[kk](inode) = ephinp[inode*my::numdofpernode_+kk];

      pot(inode) = ephinp[inode*my::numdofpernode_+my::numscal_];
    }

    // concentration of active species at integration point
    std::vector<double> conint(my::numscal_,0.0);

    // index of reactive species (starting from zero)
    // loop over all scalars
    for(int k=0; k<my::numscal_; ++k)
    {
      if((*stoich)[k]==0)
      continue;

     /*----------------------------------------------------------------------*
      |               start loop over integration points                     |
      *----------------------------------------------------------------------*/
      const DRT::UTILS::IntPointsAndWeights<my::nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);
      for (int gpid=0; gpid<intpoints.IP().nquad; gpid++)
      {
        const double fac = my::EvalShapeFuncAndIntFac(intpoints,gpid);

        // elch-specific values at integration point:
        // concentration is evaluated at all GP since some reaction models depend on all concentrations
        for(int kk=0;kk<my::numscal_;++kk)
          conint[kk] = my::funct_.Dot(conreact[kk]);

        // el. potential at integration point
        const double potint = my::funct_.Dot(pot);

        if (c0 < EPS12) dserror("reference concentration is too small (c0 < 1.0E-12) : %f",c0);

         for (int vi=0; vi<my::nen_; ++vi)
         {
           for (int ui=0; ui<my::nen_; ++ui)
           {
             elemat1_epetra(vi*my::numdofpernode_+my::numscal_,ui*my::numdofpernode_+k) += fac*my::funct_(vi)/(frt*conint[k]*nume)*my::funct_(ui);
             elemat1_epetra(vi*my::numdofpernode_+my::numscal_,ui*my::numdofpernode_+my::numscal_) += fac*my::funct_(vi)*my::funct_(ui);
           }

           // -----right-hand-side
           elevec1_epetra[vi*my::numdofpernode_+my::numscal_] += fac*my::funct_(vi)*(pot0-e0-potint-log(conint[k]/c0)/(frt*nume));
         }
      } // end of loop over integration points gpid
    } // end loop over scalars
  }  // end if(kinetics == INPAR::SCATRA::nernst)
}


/*----------------------------------------------------------------------*
 | calculate cell voltage                                    fang 01/15 |
 *----------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<distype>::CalcCellVoltage(
    const DRT::Element*               ele,              //!< the element we are dealing with
    Teuchos::ParameterList&           params,           //!< parameter list
    DRT::Discretization&              discretization,   //!< discretization
    const std::vector<int>&           lm,               //!< location vector
    Epetra_SerialDenseVector&         scalars           //!< result vector for scalar integrals to be computed
    )
{
  // get global state vector
  Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
  if(phinp == Teuchos::null)
    dserror("Cannot get state vector \"phinp\"!");

  // extract local nodal values of electric potential from global state vector
  std::vector<double> ephinpvec(lm.size());
  DRT::UTILS::ExtractMyValues(*phinp,ephinpvec,lm);
  LINALG::Matrix<my::nen_,1> epotnp(true);
  for(int inode=0; inode<my::nen_; ++inode)
    epotnp(inode,0) = ephinpvec[inode*my::numdofpernode_+my::numscal_];

  // initialize variables for electric potential and domain integrals
  double intpotential(0.);
  double intdomain(0.);

  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<my::nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    // evaluate values of shape functions and domain integration factor at current integration point
    const double fac = my::EvalShapeFuncAndIntFac(intpoints,iquad);

    // calculate potential and domain integrals
    for (int vi=0; vi<my::nen_; ++vi)
    {
      const double funct_vi_fac = my::funct_(vi)*fac;

      // potential integral
      intpotential += funct_vi_fac*epotnp(vi,0);

      // domain integral
      intdomain += funct_vi_fac;
    }
  } // loop over integration points

  // safety check
  if(scalars.Length() != 2)
    dserror("Result vector for cell voltage computation has invalid length!");

  // write results for electric potential and domain integrals into result vector
  scalars(0) = intpotential;
  scalars(1) = intdomain;

  return;
}


/*----------------------------------------------------------------------*
 | evaluate an electrode kinetics boundary condition (private) gjb 01/09|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<distype>::EvaluateElchBoundaryKinetics(
    const DRT::Element*                 ele,        ///< the actual boundary element
    Epetra_SerialDenseMatrix&           emat,       ///< element-matrix
    Epetra_SerialDenseVector&           erhs,       ///< element-rhs
    const std::vector<double>&          ephinp,     ///< actual conc. and pot. values
    const std::vector<double>&          ehist,      ///< element history vector
    double                              timefac,    ///< time factor
    Teuchos::RCP<const MAT::Material>   material,   ///< the material
    Teuchos::RCP<DRT::Condition>        cond,       ///< the condition
    const int                           nume,       ///< number of transferred electrons
    const std::vector<int>              stoich,     ///< stoichiometry of the reaction
    const int                           kinetics,   ///< desired electrode kinetics model
    const double                        pot0,       ///< actual electrode potential on metal side
    const double                        frt,        ///< factor F/RT
    const double                        scalar      ///< scaling factor for element matrix and residual contributions
)
{
  // for pre-multiplication of i0 with 1/(F z_k)
  const double faraday = INPAR::ELCH::faraday_const;    // unit of F: C/mol or mC/mmol or µC/µmol

  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<my::nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // concentration values of reactive species at element nodes
  std::vector<LINALG::Matrix<my::nen_,1> > conreact(my::numscal_);

  // el. potential values at element nodes
  LINALG::Matrix<my::nen_,1> pot(true);

  // element history vector for potential at electrode
  LINALG::Matrix<my::nen_,1> phihist(true);

  for (int inode=0; inode< my::nen_;++inode)
  {
    for(int kk=0; kk<my::numscal_; kk++)
      conreact[kk](inode) = ephinp[inode*my::numdofpernode_+kk];

    pot(inode) = ephinp[inode*my::numdofpernode_+my::numscal_];
    phihist(inode) = ehist[inode*my::numdofpernode_+my::numscal_];
  }

  // loop over all scalars
  for(int k=0; k<my::numscal_; ++k)
  {
    if(stoich[k]==0)
      continue;

    //(- N^(d+m)*n) = j = s_k / (nume * faraday * z_e-) * i
    //                  = s_k / (nume * faraday * (-1)) * i
    //                    |_______fns_________________|
    // see, e.g. in Ehrl et al., "A computational approach for the simulation of natural convection in
    // electrochemical cells", JCP, 2012
    double fns = -1.0/faraday/nume;
    // stoichiometry as a consequence of the reaction convention
    fns*=stoich[k];

    // get valence of the single reactant
    const double valence_k = GetValence(material,k);

   /*----------------------------------------------------------------------*
    |               start loop over integration points                     |
    *----------------------------------------------------------------------*/
    for(int gpid=0; gpid<intpoints.IP().nquad; gpid++)
    {
      const double fac = my::EvalShapeFuncAndIntFac(intpoints,gpid);

      // get boundary porosity from condition if available, or set equal to volume porosity otherwise
      double epsilon = cond->GetDouble("epsilon");
      if(epsilon == -1)
        epsilon = scalar;
      else if(epsilon <= 0 or epsilon > 1)
        dserror("Boundary porosity has to be between 0 and 1, or -1 by default!");

      // call utility class for element evaluation
      utils_->EvaluateElchKineticsAtIntegrationPoint(
          ele,
          emat,
          erhs,
          conreact,
          pot,
          phihist,
          timefac,
          fac,
          my::funct_,
          material,
          cond,
          nume,
          stoich,
          valence_k,
          kinetics,
          pot0,
          frt,
          fns,
          epsilon,
          k
      );
    } // loop over integration points
  } // loop over all scalars

  return;
} // DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<distype>::EvaluateElchBoundaryKinetics


/*----------------------------------------------------------------------*
 | calculate electrode kinetics status information             gjb 01/09|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<distype>::ElectrodeStatus(
    const DRT::Element*           ele,        ///< the actual boundary element
    Epetra_SerialDenseVector&     scalars,    ///< scalars to be computed
    Teuchos::ParameterList&       params,     ///< the parameter list
    Teuchos::RCP<DRT::Condition>  cond,       ///< the condition
    const std::vector<double>&    ephinp,     ///< current conc. and potential values
    const std::vector<double>&    ephidtnp,   ///< time derivative vector evaluated at t_{n+1}
    const int                     kinetics,   ///< desired electrode kinetics model
    const std::vector<int>        stoich,     ///< stoichiometry of the reaction
    const int                     nume,       ///<  number of transferred electrons
    const double                  pot0,       ///< actual electrode potential on metal side at t_{n+1}
    const double                  frt,        ///< factor F/RT
    const double                  timefac,    ///< factor due to time discretization
    const double                  scalar      ///< scaling factor for current related quantities
)
{
  // Warning:
  // Specific time integration parameter are set in the following function.
  // In the case of a genalpha-time integration scheme the solution vector phiaf_ at time n+af
  // is passed to the element evaluation routine. Therefore, the electrode status is evaluate at a
  // different time (n+af) than our output routine (n+1), resulting in slightly different values at the electrode.
  // A different approach is not possible (without major hacks) since the time-integration scheme is
  // necessary to perform galvanostatic simulations, for instance.
  // Think about: double layer effects for genalpha time-integration scheme

  // if zero=1=true, the current flow across the electrode is zero (comparable to do-nothing Neuman condition)
  // but the electrode status is evaluated
  const int zerocur = cond->GetInt("zero_cur");

  // get boundary porosity from condition if available, or set equal to volume porosity otherwise
  double epsilon = cond->GetDouble("epsilon");
  if(epsilon == -1)
    epsilon = scalar;
  else if(epsilon <= 0 or epsilon > 1)
    dserror("Boundary porosity has to be between 0 and 1, or -1 by default!");

  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<my::nsd_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // concentration values of reactive species at element nodes
  std::vector<LINALG::Matrix<my::nen_,1> > conreact(my::numscal_);

  // el. potential values at element nodes
  LINALG::Matrix<my::nen_,1> pot(true);
  LINALG::Matrix<my::nen_,1> potdtnp(true);
  for (int inode=0; inode< my::nen_;++inode)
  {
    for (int kk=0; kk<my::numscal_; ++kk)
      conreact[kk](inode) = ephinp[inode*my::numdofpernode_+kk];

    pot(inode) = ephinp[inode*my::numdofpernode_+my::numscal_];
    potdtnp(inode) = ephidtnp[inode*my::numdofpernode_+my::numscal_];
  }

  bool statistics = false;

  // index of reactive species (starting from zero)
  for(int k=0; k<my::numscal_; ++k)
  {
    // only the first oxidized species O is considered for statistics
    // statistics of other species result directly from the oxidized species (current density, ...)
    // or need to be implemented (surface concentration, OCV, ...)
    if(stoich[k]>=0)
      continue;

    statistics = true;

    // loop over integration points
    for (int gpid=0; gpid<intpoints.IP().nquad; gpid++)
    {
      const double fac = my::EvalShapeFuncAndIntFac(intpoints,gpid);

      // call utility class for element evaluation
      utils_->EvaluateElectrodeStatusAtIntegrationPoint(
        ele,
        scalars,
        params,
        cond,
        conreact,
        pot,
        potdtnp,
        my::funct_,
        zerocur,
        kinetics,
        stoich,
        nume,
        pot0,
        frt,
        timefac,
        fac,
        epsilon,
        k
      );
    } // loop over integration points

    // stop loop over ionic species after one evaluation (see also comment above)
    break;
  } // loop over scalars

  // safety check
  if(statistics == false)
    dserror("There is no oxidized species O (stoich<0) defined in your input file!! \n"
            " Statistics could not be evaluated");

  return;
} // DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<distype>::ElectrodeStatus


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes
template class DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<DRT::Element::quad4>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<DRT::Element::quad8>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<DRT::Element::quad9>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<DRT::Element::tri3>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<DRT::Element::tri6>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<DRT::Element::line2>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<DRT::Element::line3>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<DRT::Element::nurbs3>;
template class DRT::ELEMENTS::ScaTraEleBoundaryCalcElch<DRT::Element::nurbs9>;
