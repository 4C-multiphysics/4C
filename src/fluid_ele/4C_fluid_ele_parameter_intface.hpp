// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_ELE_PARAMETER_INTFACE_HPP
#define FOUR_C_FLUID_ELE_PARAMETER_INTFACE_HPP

#include "4C_config.hpp"

#include "4C_fluid_ele_parameter_timint.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_utils_singleton_owner.hpp"
#include "4C_xfem_input.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    class FluidEleParameterIntFace
    {
     public:
      //! Singleton access method
      static FluidEleParameterIntFace* instance(
          Core::Utils::SingletonAction action = Core::Utils::SingletonAction::create);

      virtual ~FluidEleParameterIntFace() = default;


      /*========================================================================*/
      //! @name set-routines, individually for each face
      /*========================================================================*/

      //! set the EOS pressure stabilization flag
      void set_face_eos_pres(const bool is_face_EOS_Pres) { is_face_EOS_Pres_ = is_face_EOS_Pres; };

      //! set the EOS convective streamline stabilization flag
      void set_face_eos_conv_stream(const bool is_face_EOS_Conv_Stream)
      {
        is_face_EOS_Conv_Stream_ = is_face_EOS_Conv_Stream;
      };

      //! set the EOS convective cross wind stabilization flag
      void set_face_eos_conv_cross(const bool is_face_EOS_Conv_Cross)
      {
        is_face_EOS_Conv_Cross_ = is_face_EOS_Conv_Cross;
      };

      //! set the EOS divergence vel-jump  flag
      void set_face_eos_div_vel_jump(const bool is_face_EOS_Div_vel_jump)
      {
        is_face_EOS_Div_vel_jump_ = is_face_EOS_Div_vel_jump;
      };

      //! set the EOS divergence div-jump flag
      void set_face_eos_div_div_jump(const bool is_face_EOS_Div_div_jump)
      {
        is_face_EOS_Div_div_jump_ = is_face_EOS_Div_div_jump;
      };

      //! set the viscous ghost-penalty stabilization flag
      void set_face_gp_visc(const bool is_face_GP_visc) { is_face_GP_visc_ = is_face_GP_visc; };

      //! set the transient ghost-penalty stabilization flag
      void set_face_gp_trans(const bool is_face_GP_trans) { is_face_GP_trans_ = is_face_GP_trans; };

      //! set the 2nd order derivatives ghost-penalty stabilization flag
      void set_face_gp_u_p_2nd(const bool is_face_GP_u_p_2nd)
      {
        is_face_GP_u_p_2nd_ = is_face_GP_u_p_2nd;
      };

      //! set the EOS pattern for the assembly of the current face
      void set_face_eos_gp_pattern(const Inpar::FLUID::EosGpPattern face_eos_gp_pattern)
      {
        face_eos_gp_pattern_ = face_eos_gp_pattern;
      };

      //! general fluid parameters are set
      void set_face_general_fluid_parameter(Teuchos::ParameterList& params,  //> parameter list
          int myrank);                                                       //> proc id

      //! general xfem (ghost-penalty) parameters are set
      void set_face_general_xfem_parameter(Teuchos::ParameterList& params,  //> parameter list
          int myrank);                                                      //> proc id


      //! specific fluid xfem (ghost-penalty) parameters are set for the face and return if
      //! stabilization for current face is required
      bool set_face_specific_fluid_xfem_parameter(
          const XFEM::FaceType& face_type,  ///< which type of face std, ghost, ghost-penalty
          Teuchos::ParameterList& params    ///< parameter list
      );

      //! set flag if step is a ghost-penalty reconstruction step for xfluid time integration
      void set_ghost_penalty_reconstruction(const bool is_ghost_penalty_reconstruction_step)
      {
        is_ghost_penalty_reconstruction_step_ = is_ghost_penalty_reconstruction_step;
      };



      /*========================================================================*/
      //! @name access-routines
      /*========================================================================*/

      /*----------------------------------------------------*/
      //! @name general parameters
      /*----------------------------------------------------*/

      //! Flag for physical type of the fluid flow (incompressible, loma, varying_density,
      //! Boussinesq, poro)
      Inpar::FLUID::PhysicalType physical_type() const { return physicaltype_; };

      //! Return function number of Oseen advective field
      int oseen_field_func_no() const { return oseenfieldfuncno_; };


      /*----------------------------------------------------*/
      //! @name general stabilization parameters (non-individually for each face, see below)
      /*----------------------------------------------------*/

      /// parameter for edge-based (EOS,CIP) stabilizations
      //! Flag to (de)activate pressure stabilization
      Inpar::FLUID::EosPres eos_pres() const { return EOS_pres_; };
      //! Flag to (de)activate convective streamline stabilization
      Inpar::FLUID::EosConvStream eos_conv_stream() const { return EOS_conv_stream_; };
      //! Flag to (de)activate convective crosswind stabilization
      Inpar::FLUID::EosConvCross eos_conv_cross() const { return EOS_conv_cross_; };
      //! Flag to (de)activate divergence stabilization
      Inpar::FLUID::EosDiv eos_div() const { return EOS_div_; };
      //! Flag to define element length
      Inpar::FLUID::EosElementLength eos_element_length() const { return EOS_element_length_; };
      //! Flag to define tau for edge-based stabilization
      Inpar::FLUID::EosTauType eos_which_tau() const { return EOS_whichtau_actual_; };

      bool is_eos_pres() const { return EOS_pres_ != Inpar::FLUID::EOS_PRES_none; };
      //! Flag to (de)activate convective streamline stabilization
      bool is_conv_stream() const { return EOS_conv_stream_; };
      //! Flag to (de)activate convective crosswind stabilization
      bool is_conv_cross() const { return EOS_conv_cross_; };
      //! Flag to (de)activate divergence stabilization
      bool is_div() const { return EOS_div_; };
      //! Flag to define element length
      bool is_element_length() const { return EOS_element_length_; };
      //! Flag to define tau for edge-based stabilization
      bool is_which_tau() const { return EOS_whichtau_actual_; };

      //! Flag to activate special least-squares condition for pseudo 2D examples where pressure
      //! level is determined via Krylov-projection
      bool pres_krylov_2dz() const { return presKrylov2Dz_; };

      //! get the viscous ghost-penalty stabilization flag
      bool is_general_ghost_penalty_visc() { return ghost_penalty_visc_; };

      //! get the transient ghost-penalty stabilization flag
      bool is_general_ghost_penalty_trans() { return ghost_penalty_trans_; };

      //! get the 2nd order derivatives ghost-penalty stabilization flag
      bool is_general_ghost_penalty_u_p_2nd() { return ghost_penalty_u_p_2nd_; };

      //! get the 2nd order derivatives ghost-penalty stabilization flag - normal derivatives or
      //! full derivatives
      bool is_general_ghost_penalty_u_p_2nd_normal() { return ghost_penalty_u_p_2nd_normal_; };

      //! get the viscous ghost-penalty stabilization factor
      double ghost_penalty_visc_fac() { return ghost_penalty_visc_fac_; };

      //! get the transient ghost-penalty stabilization factor
      double ghost_penalty_trans_fac() { return ghost_penalty_trans_fac_; };

      //! get the viscous ghost-penalty stabilization factor
      double ghost_penalty_visc_2nd_fac() { return ghost_penalty_visc_2nd_fac_; };

      //! get the viscous ghost-penalty stabilization factor
      double ghost_penalty_press_2nd_fac() { return ghost_penalty_press_2nd_fac_; };
      //
      /*----------------------------------------------------*/
      //! @name stabilization parameters, individually set for each face
      /*----------------------------------------------------*/

      //! get the EOS pressure stabilization flag
      bool face_eos_pres() { return is_face_EOS_Pres_; };

      //! get the EOS convective streamline stabilization flag
      bool face_eos_conv_stream() { return is_face_EOS_Conv_Stream_; };

      //! get the EOS convective cross wind stabilization flag
      bool face_eos_conv_cross() { return is_face_EOS_Conv_Cross_; };

      //! get the EOS divergence vel-jump  flag
      bool face_eos_div_vel_jump() { return is_face_EOS_Div_vel_jump_; };

      //! get the EOS divergence div-jump flag
      bool face_eos_div_div_jump() { return is_face_EOS_Div_div_jump_; };

      //! get the viscous ghost-penalty stabilization flag
      bool face_gp_visc() { return is_face_GP_visc_; };

      //! get the transient ghost-penalty stabilization flag
      bool face_gp_trans() { return is_face_GP_trans_; };

      //! get the 2nd order derivatives ghost-penalty stabilization flag
      bool face_gp_u_p_2nd() { return is_face_GP_u_p_2nd_; };

      //! get the EOS pattern for the assembly of the current face
      Inpar::FLUID::EosGpPattern face_eos_gp_pattern() { return face_eos_gp_pattern_; };

      //! get flag if step is a ghost-penalty reconstruction step for xfluid time integration
      bool is_ghost_penalty_reconstruction() { return is_ghost_penalty_reconstruction_step_; };


     protected:
      /*----------------------------------------------------*/
      //! @name general parameters
      /*----------------------------------------------------*/

      //! Flag set_face_general_fluid_parameter was called
      bool set_face_general_fluid_parameter_;

      //! Flag set_face_general_xfem_parameter was called
      bool set_face_general_XFEM_parameter_;

      //! Flag for physical type of the fluid flow (incompressible, loma, varying_density,
      //! Boussinesq, Poro)
      Inpar::FLUID::PhysicalType physicaltype_;

      //! Which stabilization type
      Inpar::FLUID::StabType stabtype_;

      //! function number for advective velocity for Oseen problems
      int oseenfieldfuncno_;


      /*----------------------------------------------------*/
      //! @name general edge-based/face-oriented fluid stabilization parameters (not individually
      //! set for each face, see below)
      /*----------------------------------------------------*/

      //! which EOS pressure stabilization
      Inpar::FLUID::EosPres EOS_pres_;

      //! which EOS convective streamline stabilization
      Inpar::FLUID::EosConvStream EOS_conv_stream_;

      //! which EOS convective crosswind stabilization
      Inpar::FLUID::EosConvCross EOS_conv_cross_;

      //! which EOS divergence stabilization
      Inpar::FLUID::EosDiv EOS_div_;


      //! which EOS stabilization parameter definition
      Inpar::FLUID::EosTauType EOS_whichtau_;
      Inpar::FLUID::EosTauType EOS_whichtau_actual_;

      //! which EOS characteristic element length definition
      Inpar::FLUID::EosElementLength EOS_element_length_;

      //! flag to active special least-squares condition for pseudo 2D examples where pressure level
      //! is determined via Krylov-projection
      bool presKrylov2Dz_;


      /*----------------------------------------------------*/
      //! @name general XFEM ghost penalty stabilization parameters (not individually set for each
      //! face, see below)
      /*----------------------------------------------------*/

      //! factor for viscous ghost penalty XFEM stabilization terms
      double ghost_penalty_visc_fac_;

      //! factor for transient ghost penalty XFEM stabilization terms
      double ghost_penalty_trans_fac_;

      //! general flag for viscous ghost-penalty stabilization terms
      bool ghost_penalty_visc_;

      //! general flag for transient ghost-penalty stabilization terms
      bool ghost_penalty_trans_;

      //! general flag for 2nd order derivatives in u- and p- ghost-penalty stabilization terms
      bool ghost_penalty_u_p_2nd_;

      //! general flag for 2nd order derivatives in u- and p- ghost-penalty stabilization terms
      bool ghost_penalty_u_p_2nd_normal_;

      //! factor for viscous ghost penalty XFEM stabilization terms
      double ghost_penalty_visc_2nd_fac_;

      //! factor for pressure ghost penalty XFEM stabilization terms
      double ghost_penalty_press_2nd_fac_;

      /*----------------------------------------------------*/
      //! @name EOS stabilization parameter, individually set for each face
      /*----------------------------------------------------*/

      bool is_face_EOS_Pres_;
      bool is_face_EOS_Conv_Stream_;
      bool is_face_EOS_Conv_Cross_;
      bool is_face_EOS_Div_vel_jump_;
      bool is_face_EOS_Div_div_jump_;

      /*----------------------------------------------------*/
      //! @name Ghost-Penalty parameter, individually set for each face
      /*----------------------------------------------------*/
      bool is_face_GP_visc_;
      bool is_face_GP_trans_;
      bool is_face_GP_u_p_2nd_;


      /*----------------------------------------------------*/
      //! @name Combined EOS/Ghost-Penalty parameters, individually set for each face
      /*----------------------------------------------------*/

      Inpar::FLUID::EosGpPattern face_eos_gp_pattern_;

      /*----------------------------------------------------*/
      //! @name Flag for ghost-penalty reconstruction steps
      /*----------------------------------------------------*/
      bool is_ghost_penalty_reconstruction_step_;

      /// private Constructor since we are a Singleton.
      FluidEleParameterIntFace();

     private:
      //! access time-integration parameters
      Discret::Elements::FluidEleParameterTimInt* fldparatimint_;
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
