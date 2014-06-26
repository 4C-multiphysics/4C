/*----------------------------------------------------------------------*/
/*!
\file fluid_ele_parameter.cpp

\brief Setting of general fluid parameter for internal faces evaluation

This file has to contain all parameters called in fluid_ele_intfaces.cpp.
Additional parameters required in derived classes of FluidIntFaceImpl have to
be set in problem specific parameter lists derived from this class.
<pre>
Maintainers: Benedikt Schott
             schott@lnm.mw.tum.de
             http://www.lnm.mw.tum.de
             089 - 289-15241
</pre>
*/
/*----------------------------------------------------------------------*/
#include <Teuchos_TimeMonitor.hpp>

#include "fluid_ele_parameter_intface.H"
#include "../drt_inpar/inpar_xfem.H"

#include <string>
#include <iostream>
#include "../drt_lib/drt_dserror.H"
#include "../drt_io/io_pstream.H"


//----------------------------------------------------------------------*/
//    definition of the instance
//----------------------------------------------------------------------*/
DRT::ELEMENTS::FluidEleParameterIntFace* DRT::ELEMENTS::FluidEleParameterIntFace::Instance( bool create )
{
  static FluidEleParameterIntFace* instance;
  if ( create )
  {
    if ( instance==NULL )
    {
      instance = new FluidEleParameterIntFace();
    }
  }
  else
  {
    if ( instance!=NULL )
      delete instance;
    instance = NULL;
  }
  return instance;
}

//----------------------------------------------------------------------*/
//    destruction method
//----------------------------------------------------------------------*/
void DRT::ELEMENTS::FluidEleParameterIntFace::Done()
{
  // delete this pointer! Afterwards we have to go! But since this is a
  // cleanup call, we can do it this way.
  Instance( false );
}

//----------------------------------------------------------------------*/
//    constructor
//----------------------------------------------------------------------*/
DRT::ELEMENTS::FluidEleParameterIntFace::FluidEleParameterIntFace()
:   set_face_general_fluid_parameter_(false),
    set_face_general_XFEM_parameter_(false),
    stabtype_(INPAR::FLUID::stabtype_edgebased),
    EOS_pres_(INPAR::FLUID::EOS_PRES_none),
    EOS_conv_stream_(INPAR::FLUID::EOS_CONV_STREAM_none),
    EOS_conv_cross_(INPAR::FLUID::EOS_CONV_CROSS_none),
    EOS_div_(INPAR::FLUID::EOS_DIV_none),
    EOS_whichtau_(INPAR::FLUID::EOS_tau_burman_fernandez),
    EOS_element_length_(INPAR::FLUID::EOS_he_max_dist_to_opp_surf),
    ghost_penalty_visc_fac(0.0),
    ghost_penalty_trans_fac(0.0),
    ghost_penalty_visc_(false),
    ghost_penalty_trans_(false),
    ghost_penalty_u_p_2nd_(false),
    is_face_EOS_Pres_(false),
    is_face_EOS_Conv_Stream_(false),
    is_face_EOS_Conv_Cross_(false),
    is_face_EOS_Div_vel_jump_(false),
    is_face_EOS_Div_div_jump_(false),
    is_face_GP_visc_(false),
    is_face_GP_trans_(false),
    is_face_GP_u_p_2nd_(false),
    face_eos_gp_pattern_(INPAR::FLUID::EOS_GP_Pattern_up)
{
  // we have to know the time parameters here to check for illegal combinations
  fldparatimint_ = DRT::ELEMENTS::FluidEleParameterTimInt::Instance();
}

//----------------------------------------------------------------------*
//  set general parameters                                 schott Jun14 |
//---------------------------------------------------------------------*/
void DRT::ELEMENTS::FluidEleParameterIntFace::SetFaceGeneralFluidParameter(
  Teuchos::ParameterList& params,
  int myrank )
{
  if(set_face_general_fluid_parameter_ == false)
    set_face_general_fluid_parameter_ = true;
  // For turbulent inflow generation,
  // this function is indeed two times called.
  // In this sepcial case, calling this function twice
  // is ok!
  else
  {
    if (myrank == 0)
      std::cout << std::endl <<
      (" Warning: general face fluid XFEM parameters should be set only once!!\n "
       " If you run a turbulent inflow generation, calling this function twice is ok!\n ")
      << std::endl << std::endl;
//    dserror(" general face fluid XFEM parameters should be set only once!! -> Check this?!");
  }

  //---------------------------------
  // which basic stabilization type?
  // residual-based: for residualbased standard fluid or residual-based XFEM fluid in combination with edge-based ghost penalty stabilization
  // edge-based:     for pure edge-based/ghost-penalty stabilization
  stabtype_ = DRT::INPUT::get<INPAR::FLUID::StabType>(params, "STABTYPE");

  // --------------------------------
  // edge-based fluid stabilization can be used as standard fluid stabilization or
  // as ghost-penalty stabilization in addition to residual-based stabilizations in the XFEM

  // set parameters if single stabilization terms are switched on/off or which type of stabilization is chosen

  Teuchos::ParameterList& stablist_edgebased = params.sublist("EDGE-BASED STABILIZATION");

  EOS_pres_         = DRT::INPUT::IntegralValue<INPAR::FLUID::EOS_Pres>(stablist_edgebased,"EOS_PRES");
  EOS_conv_stream_  = DRT::INPUT::IntegralValue<INPAR::FLUID::EOS_Conv_Stream>(stablist_edgebased,"EOS_CONV_STREAM");
  EOS_conv_cross_   = DRT::INPUT::IntegralValue<INPAR::FLUID::EOS_Conv_Cross>(stablist_edgebased,"EOS_CONV_CROSS");
  EOS_div_          = DRT::INPUT::IntegralValue<INPAR::FLUID::EOS_Div>(stablist_edgebased,"EOS_DIV");

  // check for reasonable combinations of non-edgebased fluid stabilizations with edge-based stabilizations
  if(stabtype_ != INPAR::FLUID::stabtype_edgebased)
  {
    // case of xfem check whether additional xfem-stabilization terms in the form of
    // edge-based terms are activated (i.e., ghost penalties)

    if(EOS_pres_ != INPAR::FLUID::EOS_PRES_xfem_gp and EOS_pres_ != INPAR::FLUID::EOS_PRES_none)
      dserror("check the combination of edge-based pressure-EOS and non-edge-based stabtype! Do you really want to use this combination?");
    if(EOS_conv_stream_ != INPAR::FLUID::EOS_CONV_STREAM_xfem_gp and EOS_conv_stream_ != INPAR::FLUID::EOS_CONV_STREAM_none)
       dserror("check the combination of edge-based pressure-EOS and non-edge-based stabtype! Do you really want to use this combination?");
    if(EOS_conv_cross_ != INPAR::FLUID::EOS_CONV_CROSS_xfem_gp and EOS_conv_cross_ != INPAR::FLUID::EOS_CONV_CROSS_none)
       dserror("check the combination of edge-based pressure-EOS and non-edge-based stabtype! Do you really want to use this combination?");
    if(EOS_div_ != INPAR::FLUID::EOS_DIV_div_jump_xfem_gp and EOS_div_ != INPAR::FLUID::EOS_DIV_vel_jump_xfem_gp
        and EOS_div_ != INPAR::FLUID::EOS_DIV_none)
       dserror("check the combination of edge-based pressure-EOS and non-edge-based stabtype! Do you really want to use this combination?");

  }

  EOS_element_length_ = DRT::INPUT::IntegralValue<INPAR::FLUID::EOS_ElementLength>(stablist_edgebased, "EOS_H_DEFINITION");
  EOS_whichtau_       = DRT::INPUT::IntegralValue<INPAR::FLUID::EOS_TauType>(stablist_edgebased, "EOS_DEFINITION_TAU");


  // set correct stationary definition of stabilization parameter automatically
  if (fldparatimint_->IsStationary())
  {
    if (EOS_whichtau_ == INPAR::FLUID::EOS_tau_burman_fernandez_hansbo)
      EOS_whichtau_ = INPAR::FLUID::EOS_tau_burman_fernandez_hansbo_wo_dt;
    if (EOS_whichtau_ == INPAR::FLUID::EOS_tau_burman_hansbo_dangelo_zunino)
      EOS_whichtau_ = INPAR::FLUID::EOS_tau_burman_hansbo_dangelo_zunino_wo_dt;
  }

  return;
}


//----------------------------------------------------------------------*
//  set general parameters                                 schott Jun14 |
//---------------------------------------------------------------------*/
void DRT::ELEMENTS::FluidEleParameterIntFace::SetFaceGeneralXFEMParameter(
  Teuchos::ParameterList& params,
  int myrank )
{
  if(set_face_general_XFEM_parameter_ == false)
    set_face_general_XFEM_parameter_ = true;
  else
  {
    dserror(" general face fluid XFEM parameters should be set only once!! -> Check this?!");
  }

  // --------------------------------
  // XFEM-specific ghost-penalty stabilizations

  Teuchos::ParameterList& stablist_xfem = params.sublist("XFLUID DYNAMIC/STABILIZATION");

  ghost_penalty_visc_fac  = stablist_xfem.get<double>("GHOST_PENALTY_FAC", 0.0);
  ghost_penalty_trans_fac = stablist_xfem.get<double>("GHOST_PENALTY_TRANSIENT_FAC", 0.0);

  ghost_penalty_visc_     = (bool)DRT::INPUT::IntegralValue<int>(stablist_xfem, "GHOST_PENALTY_STAB");
  ghost_penalty_trans_    = (bool)DRT::INPUT::IntegralValue<int>(stablist_xfem, "GHOST_PENALTY_TRANSIENT_STAB");

  // safety check
  if(fldparatimint_->IsStationary() and ghost_penalty_trans_) dserror("Do not use transient ghost penalties for stationary problems");

  ghost_penalty_u_p_2nd_  = (bool)DRT::INPUT::IntegralValue<int>(stablist_xfem, "GHOST_PENALTY_2nd_STAB");

  return;
}

//----------------------------------------------------------------------*
// specific fluid xfem (ghost-penalty) parameters are set for the face  |
// and return if stabilization for current face is required             |
//                                                         schott Jun14 |
//---------------------------------------------------------------------*/
bool DRT::ELEMENTS::FluidEleParameterIntFace::SetFaceSpecificFluidXFEMParameter(
    const INPAR::XFEM::FaceType &    face_type,        ///< which type of face std, ghost, ghost-penalty
    Teuchos::ParameterList&          params            ///< parameter list
)
{

  TEUCHOS_FUNC_TIME_MONITOR( "XFEM::Edgestab EOS: SetFaceSpecificFluidXFEMParameter" );

  //----------------------------------------------------------------------------
  // decide which terms have to be assembled and decide the assembly pattern

  // final decision which terms are assembled for the current face
  // set the values into the fldpara_intface_-list

  if(face_type == INPAR::XFEM::face_type_std)
  {
    Set_Face_EOS_Pres        ((EOS_Pres()        == INPAR::FLUID::EOS_PRES_std_eos));
    Set_Face_EOS_Conv_Stream ((EOS_Conv_Stream() == INPAR::FLUID::EOS_CONV_STREAM_std_eos));
    Set_Face_EOS_Conv_Cross  ((EOS_Conv_Cross()  == INPAR::FLUID::EOS_CONV_CROSS_std_eos));
    Set_Face_EOS_Div_vel_jump((EOS_Div()         == INPAR::FLUID::EOS_DIV_vel_jump_std_eos));
    Set_Face_EOS_Div_div_jump((EOS_Div()         == INPAR::FLUID::EOS_DIV_div_jump_std_eos));

    Set_Face_GP_visc    (false);
    Set_Face_GP_trans   (false);
    Set_Face_GP_u_p_2nd (false);
  }
  else if(face_type == INPAR::XFEM::face_type_ghost_penalty)
  {
    Set_Face_EOS_Pres        ((EOS_Pres()        != INPAR::FLUID::EOS_PRES_none));
    Set_Face_EOS_Conv_Stream ((EOS_Conv_Stream() != INPAR::FLUID::EOS_CONV_STREAM_none));
    Set_Face_EOS_Conv_Cross  ((EOS_Conv_Cross()  != INPAR::FLUID::EOS_CONV_CROSS_none));
    Set_Face_EOS_Div_vel_jump((EOS_Div()         == INPAR::FLUID::EOS_DIV_vel_jump_std_eos
                            or EOS_Div()         == INPAR::FLUID::EOS_DIV_vel_jump_xfem_gp));
    Set_Face_EOS_Div_div_jump((EOS_Div()         == INPAR::FLUID::EOS_DIV_div_jump_std_eos
                            or EOS_Div()         == INPAR::FLUID::EOS_DIV_div_jump_xfem_gp));

    Set_Face_GP_visc    (Is_General_Ghost_Penalty_visc());
    Set_Face_GP_trans   (Is_General_Ghost_Penalty_trans());
    Set_Face_GP_u_p_2nd (Is_General_Ghost_Penalty_u_p_2nd());
  }
  else if(face_type == INPAR::XFEM::face_type_ghost)
  {
    Set_Face_EOS_Pres        (false);
    Set_Face_EOS_Conv_Stream (false);
    Set_Face_EOS_Conv_Cross  (false);
    Set_Face_EOS_Div_vel_jump(false);
    Set_Face_EOS_Div_div_jump(false);
    Set_Face_GP_visc         (false);
    Set_Face_GP_trans        (false);
    Set_Face_GP_u_p_2nd      (false);
  }
  else dserror("unknown face_type!!!");

  // which pattern has to be activated?
  // TODO: this can be improved if only pressure is assembled and so on!

  if(Face_EOS_Div_div_jump())
  {
    Set_Face_EOS_GP_Pattern(INPAR::FLUID::EOS_GP_Pattern_up);
  }
  else
  {
    Set_Face_EOS_GP_Pattern(INPAR::FLUID::EOS_GP_Pattern_uvwp);
  }

  // return false if no stabilization is required
  if( !Face_EOS_Pres() and
      !Face_EOS_Conv_Stream() and
      !Face_EOS_Conv_Cross() and
      !Face_EOS_Div_vel_jump() and
      !Face_EOS_Div_div_jump() and
      !Face_GP_visc() and
      !Face_GP_trans() and
      !Face_GP_u_p_2nd()
      )
  {
    return false;
  }

  return true;
}
