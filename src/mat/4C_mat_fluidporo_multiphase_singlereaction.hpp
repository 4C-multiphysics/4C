// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_FLUIDPORO_MULTIPHASE_SINGLEREACTION_HPP
#define FOUR_C_MAT_FLUIDPORO_MULTIPHASE_SINGLEREACTION_HPP


#include "4C_config.hpp"

#include "4C_mat_fluidporo_singlephase.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------*
 | forward declarations                                                |
 *---------------------------------------------------------------------*/
namespace Discret
{
  namespace Utils
  {
    class FunctionOfAnything;
  }  // namespace Utils
}  // namespace Discret

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for a single phase of porous multiphase fluid
    ///
    /// This object exists only once for each read fluid.
    class FluidPoroSingleReaction : public Core::Mat::PAR::Parameter
    {
      enum PorofluidReactionCoupling
      {
        porofluid_reac_coup_none,              ///< no coupling, initialization value
        porofluid_reac_coup_scalarsbyfunction  ///< reaction depending on scalars defined by
                                               ///< function
      };

     public:
      /// standard constructor
      FluidPoroSingleReaction(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// initialize
      void initialize();

      /// evaluate reaction for by-function definition
      void evaluate_function(std::vector<double>& reacval,
          std::vector<std::vector<double>>& reacderivspressure,
          std::vector<std::vector<double>>& reacderivssaturation,
          std::vector<double>& reacderivsporosity,
          std::vector<std::vector<double>>& reacderivsvolfrac,
          std::vector<std::vector<double>>& reacderivsvolfracpressure,
          std::vector<std::vector<double>>& reacderivsscalar, const std::vector<double>& pressure,
          const std::vector<double>& saturation, const double& porosity,
          const std::vector<double>& volfracs, const std::vector<double>& volfracpressures,
          const std::vector<double>& scalar);

      /// Check sizes of vectors
      void check_sizes(std::vector<double>& reacval,
          std::vector<std::vector<double>>& reacderivspressure,
          std::vector<std::vector<double>>& reacderivssaturation,
          std::vector<double>& reacderivsporosity,
          std::vector<std::vector<double>>& reacderivsvolfrac,
          std::vector<std::vector<double>>& reacderivsvolfracpressure,
          std::vector<std::vector<double>>& reacderivsscalar, const std::vector<double>& pressure,
          const std::vector<double>& saturation, const double& porosity,
          const std::vector<double>& volfracs, const std::vector<double>& volfracpressures,
          const std::vector<double>& scalar);

      /// @name material parameters
      //@{
      /// number of scalars in this problem (not only in this reaction but total problem!)
      const int numscal_;

      /// number of additional volume fractions of this problem
      const int numvolfrac_;

      /// total number of multiphase-dofs = number of fluid phases + number of additional volume
      /// fractions
      const int totalnummultiphasedof_;

      /// number of fluid phases in this problem
      const int numfluidphases_;

      /// the list of material IDs
      const std::vector<int> scale_;

      /// type of coupling
      const Mat::PAR::FluidPoroSingleReaction::PorofluidReactionCoupling coupling_;

      /// ID of the function defining the reaction
      const int functID_;
      //@}

     private:
      /// returns the enum of the current coupling type
      Mat::PAR::FluidPoroSingleReaction::PorofluidReactionCoupling set_coupling_type(
          const Core::Mat::PAR::Parameter::Data& matdata);

      // get number of fluid phases in multiphase porespace from closing relation type of volfrac
      // material
      int get_numfluidphases_from_closing_relation_of_additional_porous_network_type(
          const Core::Mat::PAR::Parameter::Data& matdata);

      //! templated internal Initialize implementation
      template <int dim>
      void initialize_internal();

      //! templated internal evaluate_function implementation
      template <int dim>
      void evaluate_function_internal(std::vector<double>& reacval,
          std::vector<std::vector<double>>& reacderivspressure,
          std::vector<std::vector<double>>& reacderivssaturation,
          std::vector<double>& reacderivsporosity,
          std::vector<std::vector<double>>& reacderivsvolfrac,
          std::vector<std::vector<double>>& reacderivsvolfracpressure,
          std::vector<std::vector<double>>& reacderivsscalar, const std::vector<double>& pressure,
          const std::vector<double>& saturation, const double& porosity,
          const std::vector<double>& volfracs, const std::vector<double>& volfracpressures,
          const std::vector<double>& scalar);

      /// flag if initialize() has been called
      bool isinit_;

      /// string name used for scalars in function parser
      std::vector<std::string> scalarnames_;
      /// string name used for pressure in function parser
      std::vector<std::string> pressurenames_;
      /// string name used for saturation in function parser
      std::vector<std::string> saturationnames_;
      /// string name used for porosity in function parser
      const std::string porosityname_;
      /// string name used for volume fractions in function parser
      std::vector<std::string> volfracnames_;
      /// string name used for volume fraction pressures in function parser
      std::vector<std::string> volfracpressurenames_;

    };  // class FluidPoroSinglePhaseReaction

  }  // namespace PAR

  /*----------------------------------------------------------------------*
   | instance access method                                   vuong 08/16 |
   *----------------------------------------------------------------------*/
  class FluidPoroSingleReactionType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "FluidPoroSingleReactionType"; }

    static FluidPoroSingleReactionType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static FluidPoroSingleReactionType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for a single porous fluid phase within multiphase porous flow
  ///
  /// This object exists (several times) at every element
  class FluidPoroSingleReaction : public FluidPoroSinglePhaseBase
  {
   public:
    /// construct empty material object
    FluidPoroSingleReaction();

    /// construct the material object given material parameters
    explicit FluidPoroSingleReaction(Mat::PAR::FluidPoroSingleReaction* params);

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_fluidporo_singlereaction;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<FluidPoroSingleReaction>(*this);
    }

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return FluidPoroSinglePhaseType::instance().unique_par_object_id();
    }

    /*!
     \brief Pack this class so it can be communicated

     Resizes the vector data and stores all information of a class in it.
     The first information to be stored in data has to be the
     unique parobject id delivered by unique_par_object_id() which will then
     identify the exact class on the receiving processor.

     \param data (in/out): char vector to store class information
     */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
     \brief Unpack data from a char vector into this class

     The vector data contains all information to rebuild the
     exact copy of an instance of a class on a different processor.
     The first entry in data has to be an integer which is the unique
     parobject id defined at the top of this file and delivered by
     unique_par_object_id().

     \param data (in) : vector storing all data to be unpacked into this
     instance.
     */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    /// initialize
    void initialize() override;

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// evaluate reaction
    void evaluate_reaction(std::vector<double>& reacval,
        std::vector<std::vector<double>>& reacderivspressure,
        std::vector<std::vector<double>>& reacderivssaturation,
        std::vector<double>& reacderivsporosity,
        std::vector<std::vector<double>>& reacderivsvolfrac,
        std::vector<std::vector<double>>& reacderivsvolfracpressure,
        std::vector<std::vector<double>>& reacderivsscalar, const std::vector<double>& pressure,
        const std::vector<double>& saturation, const double& porosity,
        const std::vector<double>& volfracs, const std::vector<double>& volfracpressures,
        const std::vector<double>& scalar);

    /// return whether phase 'phasenum' is involved in this reaction
    bool is_reactive(int phasenum) const { return params_->scale_.at(phasenum) != 0; };

    /// Check sizes of vectors
    int total_num_dof() { return params_->totalnummultiphasedof_; }

   private:
    /// my material parameters
    Mat::PAR::FluidPoroSingleReaction* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
