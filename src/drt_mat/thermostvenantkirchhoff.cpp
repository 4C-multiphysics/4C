/*----------------------------------------------------------------------*/
/*!
\file thermostvenantkirchhoff.cpp
\brief St.Venant Kirchhoff with an additional temperature dependent term
       describing heat expansion

       example input line:
       MAT 1   MAT_Struct_ThrStVenantK YOUNGNUM 2 YOUNG 1.48e8 1.48e5 NUE 0.3 DENS
       9.130e-6 THEXPANS 1.72e-5 INITTEMP 293.15

<pre>
Maintainer: Caroline Danowski
            danowski@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15253
</pre>
*/
/*----------------------------------------------------------------------*
 | headers                                                   dano 02/10 |
 *----------------------------------------------------------------------*/
#include "thermostvenantkirchhoff.H"
#include "matpar_bundle.H"

#include "../drt_lib/drt_globalproblem.H"



/*----------------------------------------------------------------------*
 |                                                           dano 02/10 |
 *----------------------------------------------------------------------*/
MAT::PAR::ThermoStVenantKirchhoff::ThermoStVenantKirchhoff(
  Teuchos::RCP<MAT::PAR::Material> matdata
  )
: Parameter(matdata),
  youngs_(*(matdata->Get<std::vector<double> >("YOUNG"))),
  poissonratio_(matdata->GetDouble("NUE")),
  density_(matdata->GetDouble("DENS")),
  thermexpans_(matdata->GetDouble("THEXPANS")),
  capa_(matdata->GetDouble("CAPA")),
  conduct_(matdata->GetDouble("CONDUCT")),
  thetainit_(matdata->GetDouble("INITTEMP"))
{
  bool young_temp = (DRT::INPUT::IntegralValue<int>(DRT::Problem::Instance()->StructuralDynamicParams(),"YOUNG_IS_TEMP_DEPENDENT") == 1);
  if ( (youngs_.size() > 1) and (young_temp == false) )
    dserror("in case of temperature-dependent Young's modulus you have to specifiy YOUNG_IS_TEMP_DEPENDENT in the input file");
}


/*----------------------------------------------------------------------*
 | is called in Material::Factory from ReadMaterials()       dano 02/12 |
 *----------------------------------------------------------------------*/
Teuchos::RCP<MAT::Material> MAT::PAR::ThermoStVenantKirchhoff::CreateMaterial()
{
  return Teuchos::rcp(new MAT::ThermoStVenantKirchhoff(this));
}


MAT::ThermoStVenantKirchhoffType MAT::ThermoStVenantKirchhoffType::instance_;


/*----------------------------------------------------------------------*
 | is called in Material::Factory from ReadMaterials()       dano 02/12 |
 *----------------------------------------------------------------------*/
DRT::ParObject* MAT::ThermoStVenantKirchhoffType::Create(
  const std::vector<char> & data
  )
{
  MAT::ThermoStVenantKirchhoff* thrstvenantk = new MAT::ThermoStVenantKirchhoff();
  thrstvenantk->Unpack(data);
  return thrstvenantk;
}


/*----------------------------------------------------------------------*
 |  constructor (public)                                     dano 02/10 |
 *----------------------------------------------------------------------*/
MAT::ThermoStVenantKirchhoff::ThermoStVenantKirchhoff()
: params_(NULL)
{
}


/*----------------------------------------------------------------------*
 | constructor (public)                                      dano 02/10 |
 *----------------------------------------------------------------------*/
MAT::ThermoStVenantKirchhoff::ThermoStVenantKirchhoff(
  MAT::PAR::ThermoStVenantKirchhoff* params
  )
: params_(params)
{
}


/*----------------------------------------------------------------------*
 |  Pack (public)                                            dano 02/10 |
 *----------------------------------------------------------------------*/
void MAT::ThermoStVenantKirchhoff::Pack(DRT::PackBuffer& data) const
{
  DRT::PackBuffer::SizeMarker sm( data );
  sm.Insert();

  // pack type of this instance of ParObject
  int type = UniqueParObjectId();
  AddtoPack(data,type);

  // matid
  int matid = -1;
  if (params_ != NULL) matid = params_->Id();  // in case we are in post-process mode
  AddtoPack(data,matid);
}  // Pack()


/*----------------------------------------------------------------------*
 |  Unpack (public)                                          dano 02/10 |
 *----------------------------------------------------------------------*/
void MAT::ThermoStVenantKirchhoff::Unpack(const std::vector<char>& data)
{
  std::vector<char>::size_type position = 0;
  // extract type
  int type = 0;
  ExtractfromPack(position,data,type);
  if (type != UniqueParObjectId()) dserror("wrong instance type data");

  // matid and recover params_
  int matid;
  ExtractfromPack(position,data,matid);
  params_ = NULL;
  if (DRT::Problem::Instance()->Materials() != Teuchos::null)
    if (DRT::Problem::Instance()->Materials()->Num() != 0)
    {
      const int probinst = DRT::Problem::Instance()->Materials()->GetReadFromProblem();
      MAT::PAR::Parameter* mat = DRT::Problem::Instance(probinst)->Materials()->ParameterById(matid);
      if (mat->Type() == MaterialType())
        params_ = static_cast<MAT::PAR::ThermoStVenantKirchhoff*>(mat);
      else
        dserror("Type of parameter material %d does not fit to calling type %d", mat->Type(), MaterialType());
    }

  if (position != data.size())
    dserror("Mismatch in size of data %d <-> %d",data.size(),position);
}  // Unpack()


/*----------------------------------------------------------------------*
 | calculates stresses using one of the above method to      dano 02/10 |
 | evaluate the elasticity tensor                                       |
 *----------------------------------------------------------------------*/
void MAT::ThermoStVenantKirchhoff::Evaluate(
  const LINALG::Matrix<3,3>* defgrd,
  const LINALG::Matrix<6,1>* glstrain,
  Teuchos::ParameterList& params,
  LINALG::Matrix<6,1>* stress,
  LINALG::Matrix<6,6>* cmat
  )
{
  SetupCmat(*cmat,params);
  // evaluate stresses
  // \f \sigma = {\mathbf C} \,\varepsilon_{\rm GL} \f
  stress->MultiplyNN(*cmat,*glstrain);

}  // STR_Evaluate()


/*----------------------------------------------------------------------*
 | computes isotropic elasticity tensor in matrix notion     dano 02/10 |
 | for 3d                                                               |
 *----------------------------------------------------------------------*/
void MAT::ThermoStVenantKirchhoff::SetupCmat(
  LINALG::Matrix<6,6>& cmat,
  Teuchos::ParameterList params
  )
{
  // get material parameters
  // Young's modulus (modulus of elasticity)
  double Emod = 0.0;
  // young's modulus is temperature-dependent, E(T)
  bool young_temp = (params.get<int>("young_temp") == 1);
  if (young_temp == true)
  {
    double tempnp = params.get<double>("scalartemp");
    Emod = GetMatParameterAtTempnp(&(params_->youngs_), tempnp);
  }
  // young's modulus is constant
  else
    Emod = params_->youngs_[0];

  // Poisson's ratio (Querdehnzahl)
  const double nu = params_->poissonratio_;

/*
  if (nu == 0.5) {
    // linearly isochoric. i.e. deviatoric, isotropic elasticity tensor C in
    // Voigt matrix notation
    //
    //             [  2/3   -1/3   -1/3 |   0    0    0 ]
    //             [         2/3   -1/3 |   0    0    0 ]
    //         E   [                2/3 |   0    0    0 ]
    // C = ------- [ ~~~~   ~~~~   ~~~~   ~~~  ~~~  ~~~ ]
    //     (1+nu)  [                    | 1/2    0    0 ]
    //             [                    |      1/2    0 ]
    //             [ symmetric          |           1/2 ]
    //
    const double mfac = Emod/(1.0+nu);  // 2x shear modulus
    cmat(0,0) = mfac*2.0/3.0;
    cmat(0,1) = -mfac*1.0/3.0;
    cmat(0,2) = -mfac*1.0/3.0;
    cmat(1,0) = -mfac*1.0/3.0;
    cmat(1,1) = mfac*2.0/3.0;
    cmat(1,2) = -mfac*1.0/3.0;
    cmat(2,0) = -mfac*1.0/3.0;
    cmat(2,1) = -mfac*1.0/3.0;
    cmat(2,2) = mfac*2.0/3.0;
    // ~~~
    cmat(3,3) = mfac*0.5;
    cmat(4,4) = mfac*0.5;
    cmat(5,5) = mfac*0.5;
  }
  else */
  {
    // isotropic elasticity tensor C in Voigt matrix notation
    //                     [ 1-nu     nu     nu |       0       0       0    ]
    //                     [        1-nu     nu |       0       0       0    ]
    //         E           [               1-nu |       0       0       0    ]
    // C = --------------- [ ~~~~   ~~~~   ~~~~   ~~~~~~~~~~  ~~~~~~  ~~~~~~ ]
    //     (1+nu)*(1-2*nu) [                    | (1-2*nu)/2    0       0    ]
    //                     [                    |         (1-2*nu)/2    0    ]
    //                     [ symmetric          |                 (1-2*nu)/2 ]
    //
    const double mfac = Emod / ( (1.0 + nu) * (1.0 - 2.0 * nu) );  // factor

    // clear the material tangent
    cmat.Clear();
    // write non-zero components
    cmat(0,0) = mfac * (1.0 - nu);
    cmat(0,1) = mfac * nu;
    cmat(0,2) = mfac * nu;
    cmat(1,0) = mfac * nu;
    cmat(1,1) = mfac * (1.0 - nu);
    cmat(1,2) = mfac * nu;
    cmat(2,0) = mfac * nu;
    cmat(2,1) = mfac * nu;
    cmat(2,2) = mfac * (1.0 - nu);
    // ~~~
    cmat(3,3) = mfac * 0.5 * (1.0 - 2.0 * nu);
    cmat(4,4) = mfac * 0.5 * (1.0 - 2.0 * nu);
    cmat(5,5) = mfac * 0.5 * (1.0 - 2.0 * nu);
  }
}  // SetupCmat()


/*----------------------------------------------------------------------*
 | calculates stress-temperature modulus                     dano 04/10 |
 *----------------------------------------------------------------------*/
double MAT::ThermoStVenantKirchhoff::STModulus(
  Teuchos::ParameterList& params
  ) const
{
  double Emod = 0.0;
  // is young's modulus temperature-dependent, i.e. E(T)?
  bool young_temp = (params.get<int>("young_temp") == 1);
  if (young_temp == true)
  {
    double tempnp = params.get<double>("scalartemp");
    Emod = GetMatParameterAtTempnp(&(params_->youngs_), tempnp);
  }
  else
    Emod = params_->youngs_[0];

  // initialise the parameters for the lame constants
  const double pv  = params_->poissonratio_;

  // initialise the thermal expansion coefficient
  const double thermexpans = params_->thermexpans_;

  // plane strain, rotational symmetry
  // E / (1+nu)
  const double c1 = Emod / (1.0 + pv);
  // (E*nu) / ((1+nu)(1-2nu))
  const double b1 = c1 * pv / (1.0 - 2.0 * pv);

  // build the lame constants
  //            E
  //   mu = --------
  //        2*(1+nu)
  //                  E*nu
  //   lambda = ----------------
  //            (1+nu)*(1-2*nu)
  //
  //  \f \mu =  \frac{E}{2(1+\nu)} \f
  const double mu = 0.5 * c1;
  // lambda
  // \f \frac{E\,\nu}{(1-2\nu)(1+\nu)} \f
  const double lambda = b1;

  // stress-temperature modulus
  // \f m\, = \, -(2\,\cdot \mu \, +\, 3\cdot\lambda)\cdot\varalpha_T \f
  const double stmodulus = (-1) * (2 * mu + 3 * lambda) * thermexpans;

  return stmodulus;

} // STModulus()


/*----------------------------------------------------------------------*
 | computes temperature dependent isotropic                  dano 05/10 |
 | elasticity tensor in matrix notion for 3d, second(!) order tensor    |
 *----------------------------------------------------------------------*/
void MAT::ThermoStVenantKirchhoff::SetupCthermo(
  LINALG::Matrix<6,1>& ctemp,
  Teuchos::ParameterList& params
  )
{
  double m = STModulus(params);

  // isotropic elasticity tensor C_temp in Voigt matrix notation C_temp = m I
  //
  // Matrix-notation for 3D case
  //       [ m      0      0 ]
  // C_T = [ 0      m      0 ]
  //       [ 0      0      m ]
  //
  // in Vector notation
  // C_T = [m, m, m, 0, 0, 0]^T
  //
  // write non-zero components

  // clear the material tangent, equal to PutScalar(0.0), but faster
  ctemp.Clear();

  // loop over the element nodes, non-zero entries only in main directions
  for (int i=0; i<3; ++i)
    ctemp(i,0) = m;

}  // SetupCthermo()


/*----------------------------------------------------------------------*
 | calculates stresses by evaluating the temperature tangent dano 05/10 |
 *----------------------------------------------------------------------*/
void MAT::ThermoStVenantKirchhoff::Evaluate(
  const LINALG::Matrix<1,1>& Ntemp,  // shapefcts . temperatures
  LINALG::Matrix<6,1>& ctemp,
  LINALG::Matrix<6,1>& stresstemp,
  Teuchos::ParameterList& params
  )
{
  LINALG::Matrix<1,1> init(true);
  const double inittemp = -1.0 * (params_->thetainit_);
  // loop over the element nodes
  init(0,0) = inittemp;
  // Delta T = T - T_0
  LINALG::Matrix<1,1> deltaT(true);
  deltaT.Update(Ntemp,init);

  SetupCthermo(ctemp,params);

  // temperature dependent stress
  // sigma = C_T . Delta T = m . I . Delta T
  stresstemp.MultiplyNN(ctemp,deltaT);

} // THR_Evaluate()


/*----------------------------------------------------------------------*
 | return temperature-dependent material parameter           dano 01/13 |
 | at current temperature --> polynomial type, cf. robinson material    |
 *----------------------------------------------------------------------*/
double MAT::ThermoStVenantKirchhoff::GetMatParameterAtTempnp(
  const std::vector<double>* paramvector,  // (i) given parameter is a vector
  const double& tempnp  // tmpr (i) current temperature
  ) const
{
  // polynomial type

  // initialise the temperature dependent material parameter
  double parambytempnp = 0.0;
  double tempnp_pow = 1.0;

  // Param = a + b . T + c . T^2 + d . T^3 + ...
  // with T: current temperature
  for (unsigned i=0; i<(*paramvector).size(); ++i)
  {
    // calculate coefficient of variable T^i
    parambytempnp += (*paramvector)[i] * tempnp_pow;
    // for the higher polynom increase the exponent of the temperature
    tempnp_pow *= tempnp;
  }

  // return temperature-dependent material parameter
  return parambytempnp;

}  // GetMatParameterAtTempnp()


/*----------------------------------------------------------------------*
 | calculate derivative of material parameter with respect   dano 01/13 |
 | to the current temperature --> polynomial type                       |
 *----------------------------------------------------------------------*/
double MAT::ThermoStVenantKirchhoff::GetMatParameterAtTempnp_T(
  const std::vector<double>* paramvector,  // (i) given parameter is a vector
  const double& tempnp  // tmpr (i) current temperature
  ) const
{
  // polynomial type

  // initialise the temperature dependent material parameter
  double parambytempnp = 0.0;
  double tempnp_pow = 1.0;

  // Param = a + b . T + c . T^2 + d . T^3 + ...
  //       = a + b . N_T . T_{n+1} + c . (N_T . T_{n+1})^2 + d . (N_T . T_{n+1})^3 + ...
  // with T: current scalar-valued temperature, T = N_T . T_{n+1}

  // calculate derivative of E(T_{n+1}) w.r.t. T_{n+1}
  // d(Param)/dT . Delta T = b . N_T + 2 . c . T . N_T + 3 . d . T^2 . N_T + ...
  //                       = ( b + 2 . c . T + 3 . d . T^2 + ...) . N_T
  //                       = parambytempnp . N_T

  // the first, constant term has no influence for the derivative
  // --> start with i=1(!): neglect first term of paramvector
  for (unsigned i=1; i<(*paramvector).size(); ++i)
  {
    // calculate coefficient of variable T^i
    parambytempnp += i * (*paramvector)[i] * tempnp_pow;
    // for the higher polynom increase the exponent of the temperature
    tempnp_pow *= tempnp;
  }

  // return derivative of temperature-dependent material parameter w.r.t. T_{n+1}
  return parambytempnp;

}  // GetMatParameterAtTempnp_T()


/*----------------------------------------------------------------------*
 | calculate linearisation of stress-temperature modulus     dano 04/10 |
 | w.r.t. T_{n+1} for k_dT, k_TT                                        |
 *----------------------------------------------------------------------*/
double MAT::ThermoStVenantKirchhoff::GetSTModulus_T(
  Teuchos::ParameterList& params
  ) const
{
  // build the derivative of the stress-temperature modulus w.r.t. T_{n+1}
  // m = - (2 . mu + 3 . lambda) . varalpha_T
  //   = - (2 . nu / ((1+nu)(1-2nu)) + 3 / (2 . (1+nu))) . varalpha_T . E(T)
  double stmodulus_T = 0.0;

  // is young's modulus temperature-dependent, i.e. E(T)?
  bool young_temp = (params.get<int>("young_temp") == 1);
  if (young_temp == true)
  {
    const double tempnp = params.get<double>("scalartemp");
    const double Ederiv = GetMatParameterAtTempnp_T(&(params_->youngs_), tempnp);

    // initialise the parameters for the lame constants
    const double pv  = params_->poissonratio_;

    // initialise the thermal expansion coefficient
    const double thermexpans = params_->thermexpans_;

    // plane strain, rotational symmetry
    // E / (1+nu)
    const double c1 = Ederiv/(1.0+pv);
    // (E . nu) / ((1+nu)(1-2nu))
    const double b1 = c1*pv/(1.0-2.0*pv);

    // build the lame constants
    //         E
    // mu = --------  --> \f \mu = \frac{E}{2(1+\nu)} \f
    //      2*(1+nu)
    const double mu = 0.5 * c1;
    //              E*nu
    // lambda = --------------- --> \f \frac{E\,\nu}{(1-2\nu)(1+\nu)} \f
    //          (1+nu)*(1-2*nu)
    const double lambda = b1;

    // build the derivative of the stress-temperature modulus w.r.t. T_{n+1}
    // m = -(2 . mu + 3 . lambda) . varalpha_T
    stmodulus_T = (-1) * (2 * mu + 3 * lambda) * thermexpans;
  }
  // else (young_temp == false)
  // constant young's modulus, i.e. independent of T, no linearisation, return 0;

  return stmodulus_T;

} // GetSTModulus_T()


/*----------------------------------------------------------------------*
 | calculates stresses with derived cmat w.r.t. T_{n+1}       dano 05/13 |
 *----------------------------------------------------------------------*/
void MAT::ThermoStVenantKirchhoff::GetMechStress_T(
  const LINALG::Matrix<6,1>* glstrain,
  Teuchos::ParameterList& params,
  LINALG::Matrix<6,1>* stress
  )
{
  // calculate derivative of cmat w.r.t. T_{n+1}
  LINALG::Matrix<6,6> cmat_T(false);
  GetCmatAtTempnp_T(cmat_T,params);

  // evaluate stresses
  // \f \sigma = {\mathbf C}_{,T} \,\varepsilon_{\rm GL} \f
  stress->MultiplyNN(cmat_T,*glstrain);

}  // GetMechStress_T


/*----------------------------------------------------------------------*
 | calculates stresses using one of the above method to      dano 05/13 |
 | evaluate the elasticity tensor                                       |
 *----------------------------------------------------------------------*/
void MAT::ThermoStVenantKirchhoff::GetThermalStress_T(
  const LINALG::Matrix<1,1>* Ntemp,
  Teuchos::ParameterList& params,
  LINALG::Matrix<6,1>* stresstemp_T
  )
{
  LINALG::Matrix<1,1> init(true);
  const double inittemp = -1.0 * (params_->thetainit_);
  // loop over the element nodes
  init(0,0) = inittemp;
  // Delta T = T - T_0
  LINALG::Matrix<1,1> deltaT(true);
  deltaT.Update(*Ntemp,init);

  // calculate derivative of ctemp w.r.t. T_{n+1}
  LINALG::Matrix<6,1> ctemp_T(false);
  GetCthermoAtTempnp_T(ctemp_T,params);

  // temperature dependent stress
  // sigma = C_T . Delta T = m . I . Delta T
  stresstemp_T->MultiplyNN(ctemp_T,deltaT);

}  // GetThermalStress_T


/*----------------------------------------------------------------------*
 | computes thermal derivative of the isotropic elasticity   dano 01/13 |
 | tensor in matrix notion for 3d for k_dT                              |
 *----------------------------------------------------------------------*/
void MAT::ThermoStVenantKirchhoff::GetCmatAtTempnp_T(
  LINALG::Matrix<6,6>& derivcmat,
  Teuchos::ParameterList params
  )
{
  // clear the material tangent, identical to PutScalar(0.0)
  derivcmat.Clear();

  // is young's modulus temperature-dependent, i.e. E(T)?
  bool young_temp = (params.get<int>("young_temp") == 1);
  if (young_temp == true)
  {
    const double tempnp = params.get<double>("scalartemp");
    const double Ederiv = GetMatParameterAtTempnp_T(&(params_->youngs_), tempnp);
    // Poisson's ratio (Querdehnzahl)
    const double nu = params_->poissonratio_;

  /*
    if (nu == 0.5) {
      // linearly isochoric. i.e. deviatoric, isotropic elasticity tensor C in
      // Voigt matrix notation
      //                [  2/3   -1/3   -1/3 |   0    0    0 ]
      //                [         2/3   -1/3 |   0    0    0 ]
      //           1    [                2/3 |   0    0    0 ]
      // dC/dT = -------[ ~~~~   ~~~~   ~~~~   ~~~  ~~~  ~~~ ] . d[E(T)]/dT
      //         (1+nu) [                    | 1/2    0    0 ]
      //                [                    |      1/2    0 ]
      //                [ symmetric          |           1/2 ]
      //
      const double mfac = Ederiv/(1.0+nu);  // 2x shear modulus
      cmat(0,0) = mfac*2.0/3.0;
      cmat(0,1) = -mfac*1.0/3.0;
      cmat(0,2) = -mfac*1.0/3.0;
      cmat(1,0) = -mfac*1.0/3.0;
      cmat(1,1) = mfac*2.0/3.0;
      cmat(1,2) = -mfac*1.0/3.0;
      cmat(2,0) = -mfac*1.0/3.0;
      cmat(2,1) = -mfac*1.0/3.0;
      cmat(2,2) = mfac*2.0/3.0;
      // ~~~
      cmat(3,3) = mfac*0.5;
      cmat(4,4) = mfac*0.5;
      cmat(5,5) = mfac*0.5;
    }
    else */
    {
      // isotropic elasticity tensor C in Voigt matrix notation
      //                          [ 1-nu     nu     nu |       0       0       0    ]
      //                          [        1-nu     nu |       0       0       0    ]
      //                1         [               1-nu |       0       0       0    ]
      // C_{,T} = --------------- [ ~~~~   ~~~~   ~~~~   ~~~~~~~~~~  ~~~~~~  ~~~~~~ ] . d[E(T)]/dT
      //          (1+nu)*(1-2*nu) [                    | (1-2*nu)/2    0       0    ]
      //                          [                    |         (1-2*nu)/2    0    ]
      //                          [ symmetric          |                 (1-2*nu)/2 ]
      //
      const double mfac = Ederiv / ( (1.0 + nu) * (1.0 - 2.0 * nu) );  // factor

      // write non-zero components
      derivcmat(0,0) = mfac * (1.0 - nu);
      derivcmat(0,1) = mfac * nu;
      derivcmat(0,2) = mfac * nu;
      derivcmat(1,0) = mfac * nu;
      derivcmat(1,1) = mfac * (1.0 - nu);
      derivcmat(1,2) = mfac * nu;
      derivcmat(2,0) = mfac * nu;
      derivcmat(2,1) = mfac * nu;
      derivcmat(2,2) = mfac * (1.0 - nu);
      // ~~~
      derivcmat(3,3) = mfac * 0.5 * (1.0 - 2.0 * nu);
      derivcmat(4,4) = mfac * 0.5 * (1.0 - 2.0 * nu);
      derivcmat(5,5) = mfac * 0.5 * (1.0 - 2.0 * nu);
    }
  }
  // else (young_temp == false)
  // constant young's modulus, i.e. independent of T, no linearisation, do nothing
  // return derivcmat == zero;

}  // GetCmatAtTempnp_T()


/*----------------------------------------------------------------------*
 | computes linearisation of temperature-dependent isotropic dano 01/13 |
 | elasticity tensor in matrix notion for 3d, 2nd order tensor          |
 *----------------------------------------------------------------------*/
void MAT::ThermoStVenantKirchhoff::GetCthermoAtTempnp_T(
  LINALG::Matrix<6,1>& derivctemp,
  Teuchos::ParameterList& params
  )
{
  double m_T = GetSTModulus_T(params);

  // isotropic elasticity tensor C_T in Voigt matrix notation C_T = m I
  //
  // Matrix-notation for 3D case
  //                      [ m_T      0      0 ]
  // (C_T)/dT = C_{T,T} = [ 0      m_T      0 ]
  //                      [ 0      0      m_T ]
  //
  // in Vector notation
  // C_T = [m_{,T}, m_{,T}, m_{,T}, 0, 0, 0]^T
  //
  // write non-zero components

  // clear the material tangent, identical to PutScalar(0.0), but faster
  derivctemp.Clear();

  // loop over the element nodes, non-zero entries only on main directions
  for (int i=0; i<3; ++i)
    derivctemp(i,0) = m_T;

}  // GetCthermoAtTempnp_T()


/*----------------------------------------------------------------------*/


