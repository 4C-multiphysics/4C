/*----------------------------------------------------------------------*/
/*!
 \file scatra_ele_calc_poro.cpp

 \brief

 <pre>
   Maintainer: Anh-Tu Vuong
               vuong@lnm.mw.tum.de
               http://www.lnm.mw.tum.de
               089 - 289-15264
 </pre>
 *----------------------------------------------------------------------*/

#include "scatra_ele_calc_poro.H"

#include "../drt_lib/drt_globalproblem.H"
#include "../drt_lib/drt_discret.H"
#include "../drt_lib/drt_element.H"
#include "../drt_lib/drt_utils.H"

#include "../drt_mat/structporo.H"
#include "../drt_mat/scatra_mat.H"
#include "../drt_mat/matlist.H"

#include "scatra_ele.H"
#include "scatra_ele_parameter_timint.H"

#include "../drt_lib/standardtypes_cpp.H"  // for EPS13 and so on

/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template<DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraEleCalcPoro<distype> * DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::Instance(
  const int numdofpernode,
  const int numscal,
  const std::string& disname,
  bool create )
{
  static std::map<std::string,ScaTraEleCalcPoro<distype>* >  instances;

  if(create)
  {
    if(instances.find(disname) == instances.end())
      instances[disname] = new ScaTraEleCalcPoro<distype>(numdofpernode,numscal,disname);
  }

  else
  {
    for( typename std::map<std::string,ScaTraEleCalcPoro<distype>* >::iterator i=instances.begin(); i!=instances.end(); ++i )
     {
      delete i->second;
      i->second = NULL;
     }

    instances.clear();
    return NULL;
  }

  return instances[disname];
}


/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::Done()
{
  // delete this pointer! Afterwards we have to go! But since this is a
  // cleanup call, we can do it this way.
  Instance( 0, 0, "", false );
}


/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::ScaTraEleCalcPoro(
    const int numdofpernode,
    const int numscal,
    const std::string& disname)
  : DRT::ELEMENTS::ScaTraEleCalc<distype>::ScaTraEleCalc(numdofpernode,numscal,disname),
    xyze0_(true),
    eporosity_(true),
    isnodalporosity_(false)
{
  // initialization of diffusion manager (override initialization in base class)
  my::diffmanager_ = Teuchos::rcp(new ScaTraEleDiffManagerPoro(my::numscal_));

  return;
}

// /*----------------------------------------------------------------------*
// * Action type: Evaluate                                    vuong 07/14 |
// *----------------------------------------------------------------------*/
//template <DRT::Element::DiscretizationType distype>
//int DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::Evaluate(
//  DRT::Element*              ele,
//  Teuchos::ParameterList&    params,
//  DRT::Discretization&       discretization,
//  const std::vector<int>&    lm,
//  Epetra_SerialDenseMatrix&  elemat1_epetra,
//  Epetra_SerialDenseMatrix&  elemat2_epetra,
//  Epetra_SerialDenseVector&  elevec1_epetra,
//  Epetra_SerialDenseVector&  elevec2_epetra,
//  Epetra_SerialDenseVector&  elevec3_epetra
//  )
//{
//  // check for the action parameter
//  const SCATRA::Action action = DRT::INPUT::get<SCATRA::Action>(params,"action");
//  switch(action)
//  {
//    case SCATRA::calc_scatra_mono_odblock_mesh:
//    {
//      return EvaluateODMesh(
//          ele,
//          params,
//          discretization,
//          lm,
//          elemat1_epetra,
//          elemat2_epetra,
//          elevec1_epetra,
//          elevec2_epetra,
//          elevec3_epetra
//          );
//      break;
//    }
//    case SCATRA::calc_scatra_mono_odblock_fluid:
//    {
//      return EvaluateODFluid(
//          ele,
//          params,
//          discretization,
//          lm,
//          elemat1_epetra,
//          elemat2_epetra,
//          elevec1_epetra,
//          elevec2_epetra,
//          elevec3_epetra
//          );
//      break;
//    }
//    default:
//    {
//      return my::Evaluate(
//          ele,
//          params,
//          discretization,
//          lm,
//          elemat1_epetra,
//          elemat2_epetra,
//          elevec1_epetra,
//          elevec2_epetra,
//          elevec3_epetra
//          );
//      break;
//    }
//  }
//
//  //you should no turn up here -> return error code
//  return -1;
//}

/*----------------------------------------------------------------------*
 | evaluate action                                          vuong 07/15 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
int DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::EvaluateAction(
    DRT::Element*                 ele,
    Teuchos::ParameterList&       params,
    DRT::Discretization&          discretization,
    const SCATRA::Action&         action,
    DRT::Element::LocationArray&  la,
    Epetra_SerialDenseMatrix&     elemat1_epetra,
    Epetra_SerialDenseMatrix&     elemat2_epetra,
    Epetra_SerialDenseVector&     elevec1_epetra,
    Epetra_SerialDenseVector&     elevec2_epetra,
    Epetra_SerialDenseVector&     elevec3_epetra
    )
{
  // determine and evaluate action
  switch(action)
  {
  case SCATRA::calc_mean_scalars:
  {
    // get flag for inverting
    bool inverting = params.get<bool>("inverting");

    // need current scalar vector
    // -> extract local values from the global vectors
    Teuchos::RCP<const Epetra_Vector> phinp = discretization.GetState("phinp");
    if (phinp==Teuchos::null) dserror("Cannot get state vector 'phinp'");
    DRT::UTILS::ExtractMyValues<LINALG::Matrix<my::nen_,1> >(*phinp,my::ephinp_,la[0].lm_);

    ExtractElementAndNodeValuesPoro(ele,params,discretization,la);

    // calculate scalars and domain integral
    CalculateScalars(ele,elevec1_epetra,inverting);

    break;
  }
  default:
    return my::EvaluateAction(
              ele,
              params,
              discretization,
              action,
              la,
              elemat1_epetra,
              elemat2_epetra,
              elevec1_epetra,
              elevec2_epetra,
              elevec3_epetra
              );
    break;
  }

  return 0;
}

/*----------------------------------------------------------------------*
 | read element coordinates                                 vuong 10/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::ReadElementCoordinates(
    const DRT::Element*                 ele
    )
{
  //call base class
  my::ReadElementCoordinates(ele);

  //copy initial node position
  xyze0_= my::xyze_;

  return;
}

/*----------------------------------------------------------------------*
 | extract element based or nodal values                     ehrl 12/13 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::ExtractElementAndNodeValues(
    DRT::Element*                 ele,
    Teuchos::ParameterList&       params,
    DRT::Discretization&          discretization,
    DRT::Element::LocationArray&  la
)
{
  ExtractElementAndNodeValuesPoro(ele,params,discretization,la);

  my::ExtractElementAndNodeValues(ele,params,discretization,la);

  return;
}

/*----------------------------------------------------------------------*
 | extract element based or nodal values                     ehrl 12/13 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::ExtractElementAndNodeValuesPoro(
    DRT::Element*                 ele,
    Teuchos::ParameterList&       params,
    DRT::Discretization&          discretization,
    DRT::Element::LocationArray&  la
)
{
  const Teuchos::RCP<Epetra_MultiVector> pre = params.get< Teuchos::RCP<Epetra_MultiVector> >("pressure field");
  LINALG::Matrix<1,my::nen_> eprenp;
  DRT::UTILS::ExtractMyNodeBasedValues(ele,eprenp,pre,1);

  //pressure values
  for (int i=0;i<my::nen_;++i)
  {
    my::eprenp_(i) = eprenp(0,i);
  }

  // this is a hack. Check if the structure (assumed to be the dofset 1) has more DOFs than dimension. If so,
  // we assume that this is the porosity
  if( discretization.NumDof(1,ele->Nodes()[0])==my::nsd_+1 )
  {
    isnodalporosity_=true;

    Teuchos::RCP<const Epetra_Vector> disp= discretization.GetState(1,"displacement");

    if(disp!=Teuchos::null)
    {
      std::vector<double> mydisp(la[1].lm_.size());
      DRT::UTILS::ExtractMyValues(*disp,mydisp,la[1].lm_);

      for (int inode=0; inode<my::nen_; ++inode)  // number of nodes
        eporosity_(inode,0) = mydisp[my::nsd_+(inode*(my::nsd_+1))];
    }
    else
      dserror("Cannot get state vector displacement");
  }
  else
    isnodalporosity_=false;

  return;
}

/*----------------------------------------------------------------------*
 |  get the material constants  (protected)                  vuong 10/14|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::GetMaterialParams(
  const DRT::Element* ele,       //!< the element we are dealing with
  double&             densn,     //!< density at t_(n)
  double&             densnp,    //!< density at t_(n+1) or t_(n+alpha_F)
  double&             densam,    //!< density at t_(n+alpha_M)
  double&             visc,      //!< fluid viscosity
  const int           iquad      //!< id of current gauss point
  )
{
  //calculate gauss point porosity from fluid and solid and (potentially) scatra solution
  ComputePorosity(ele);

  // get the material
  Teuchos::RCP<MAT::Material> material = ele->Material();

  // get diffusivity / diffusivities
  if (material->MaterialType() == INPAR::MAT::m_matlist)
  {
    const Teuchos::RCP<const MAT::MatList>& actmat
      = Teuchos::rcp_dynamic_cast<const MAT::MatList>(material);
    if (actmat->NumMat() < my::numscal_) dserror("Not enough materials in MatList.");

    for (int k = 0;k<my::numscal_;++k)
    {
      int matid = actmat->MatID(k);
      Teuchos::RCP< MAT::Material> singlemat = actmat->MaterialById(matid);

      my::Materials(singlemat,k,densn,densnp,densam,visc,iquad);
    }
  }
  else
    my::Materials(material,0,densn,densnp,densam,visc,iquad);

  return;
} //ScaTraEleCalcPoro::GetMaterialParams

/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::MatScaTra(
  const Teuchos::RCP<const MAT::Material> material, //!< pointer to current material
  const int                               k,        //!< id of current scalar
  double&                                 densn,    //!< density at t_(n)
  double&                                 densnp,   //!< density at t_(n+1) or t_(n+alpha_F)
  double&                                 densam,   //!< density at t_(n+alpha_M)
  double&                                 visc,     //!< fluid viscosity
  const int                               iquad     //!< id of current gauss point
  )
{
  if(iquad==-1)
    dserror("no gauss point given for evaluation of scatra material. Check your input file.");

  //read the porosity from the diffusion manager
  const double porosity = DiffManager()->GetPorosity(k);

  const Teuchos::RCP<const MAT::ScatraMat>& actmat
    = Teuchos::rcp_dynamic_cast<const MAT::ScatraMat>(material);

  // set diffusivity (scaled with porosity)
  SetDiffusivity(actmat,k,porosity);

  // set densities (scaled with porosity)
  SetDensities(porosity,densn,densnp,densam);

  return;
} // ScaTraEleCalcPoro<distype>::MatScaTra

/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
inline void DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::SetDiffusivity(
    const Teuchos::RCP<const MAT::ScatraMat>& material,
    const int                                 k,
    const double                              scale)
{
  my::diffmanager_->SetIsotropicDiff(material->Diffusivity()*scale,k);

  return;
}

/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
inline void DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::SetReaCoefficient(
    const Teuchos::RCP<const MAT::ScatraMat>& material,
    const int                                k,
    const double                             scale)
{
  //set reaction coefficient (no scaling with porosity)
  my::reamanager_->SetReaCoeff(material->ReaCoeff()*scale,k);

  return;
}

/*----------------------------------------------------------------------*
 |                                                           vuong 07/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
inline void DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::SetDensities(
    double  porosity,
    double& densn,
    double& densnp,
    double& densam
    )
{
  //all densities are set to the porosity
  densn = porosity;
  densnp = porosity;
  densam = porosity;

  return;
}

/*----------------------------------------------------------------------*
 |  get the material constants  (protected)                  vuong 10/14|
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::ComputePorosity(
    const DRT::Element* ele       //!< the element we are dealing with
  )
{
  double porosity=0.0;

  if(isnodalporosity_)
  {
    porosity = eporosity_.Dot(my::funct_);
  }
  else
  {
    //gauss point displacements
    LINALG::Matrix<my::nsd_,1> dispint(false);
    dispint.Multiply(my::edispnp_,my::funct_);

    //------------------------get determinant of Jacobian dX / ds
    // transposed jacobian "dX/ds"
    LINALG::Matrix<my::nsd_,my::nsd_> xjm0;
    xjm0.MultiplyNT(my::deriv_,xyze0_);

    // inverse of transposed jacobian "ds/dX"
    const double det0= xjm0.Determinant();

    my::xjm_.MultiplyNT(my::deriv_,my::xyze_);
    const double det = my::xjm_.Determinant();

    // determinant of deformationgradient det F = det ( d x / d X ) = det (dx/ds) * ( det(dX/ds) )^-1
    const double J = det/det0;

    //fluid pressure at gauss point
    const double pres = my::eprenp_.Dot(my::funct_);

    //empty parameter list
    Teuchos::ParameterList             params;

    if(ele->NumMaterial()<2)
      dserror("no secondary material available");

    //here we rely that the structure material has been added as second material
    Teuchos::RCP< MAT::StructPoro > structmat = Teuchos::rcp_dynamic_cast<MAT::StructPoro>(ele->Material(1));
    if(structmat==Teuchos::null)
      dserror("cast to MAT::StructPoro failed!");

    //just evaluate the first scalar (used only in case of reactive porosity)
    Teuchos::RCP<std::vector<double> > scalars = Teuchos::rcp(new std::vector<double>(0));
    for(int k=0;k<my::numscal_;++k)
    {
      const double phinp = my::ephinp_[k].Dot(my::funct_);
      scalars->push_back(phinp);
    }
    params.set<Teuchos::RCP<std::vector<double> > >("scalar",scalars);

    params.set<double>("delta time",my::scatraparatimint_->Dt());

    //use structure material to evaluate porosity
    structmat->ComputePorosity( params,
                                pres,
                                J,
                                -1,
                                porosity,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                NULL,
                                false);
  }

  //save porosity in diffusion manager for later access
  DiffManager()->SetPorosity(porosity);

  return;
}

/*----------------------------------------------------------------------*
|  calculate scalar(s) and domain integral                  vuong 07/15|
| (overwrites method in ScaTraEleCalc)                                 |
*----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::ELEMENTS::ScaTraEleCalcPoro<distype>::CalculateScalars(
const DRT::Element*             ele,
Epetra_SerialDenseVector&       scalars,
const bool                      inverting
  )
{
  // integration points and weights
  const DRT::UTILS::IntPointsAndWeights<my::nsd_ele_> intpoints(SCATRA::DisTypeToOptGaussRule<distype>::rule);

  // integration loop
  for (int iquad=0; iquad<intpoints.IP().nquad; ++iquad)
  {
    const double fac = my::EvalShapeFuncAndDerivsAtIntPoint(intpoints,iquad);

    //calculate gauss point porosity from fluid and solid and (potentially) scatra solution
    ComputePorosity(ele);

    // calculate integrals of (inverted) scalar(s) and domain
    if (inverting)
    {
      for (int i=0; i<my::nen_; i++)
      {
        const double fac_funct_i = fac*my::funct_(i);
        for (int k = 0; k < my::numscal_; k++)
        {
          const double porosity = DiffManager()->GetPorosity(k);
          if (std::abs(my::ephinp_[k](i,0))> EPS14)
            scalars[k] += fac_funct_i/(my::ephinp_[k](i,0)*porosity);
          else
            dserror("Division by zero");
        }
        // for domain volume
        scalars[my::numscal_] += fac_funct_i;
      }
    }
    else
    {
      for (int i=0; i<my::nen_; i++)
      {
        const double fac_funct_i = fac*my::funct_(i);
        for (int k = 0; k < my::numscal_; k++)
        {
          const double porosity = DiffManager()->GetPorosity(k);
          scalars[k] += fac_funct_i*my::ephinp_[k](i,0)*porosity;;
        }
        // for domain volume
        scalars[my::numscal_] += fac_funct_i;
      }
    }
  } // loop over integration points

  return;
} // ScaTraEleCalc::CalculateScalars

// template classes

// 1D elements
template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::line2>;
template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::line3>;

// 2D elements
//template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::tri3>;
//template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::tri6>;
template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::quad4>;
//template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::quad8>;
template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::quad9>;

// 3D elements
template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::hex8>;
//template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::hex20>;
template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::hex27>;
template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::tet4>;
template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::tet10>;
//template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::wedge6>;
template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::pyramid5>;
template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::nurbs9>;
//template class DRT::ELEMENTS::ScaTraEleCalcPoro<DRT::Element::nurbs27>;

