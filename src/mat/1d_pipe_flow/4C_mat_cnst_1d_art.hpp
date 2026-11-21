// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_CNST_1D_ART_HPP
#define FOUR_C_MAT_CNST_1D_ART_HPP



#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    enum ArteryViscosityLaw
    {
      viscositylaw_undefined,
      viscositylaw_constant,
      viscositylaw_blood
    };
    enum ArteryDiameterLaw
    {
      diameterlaw_undefined,
      diameterlaw_constant,
      diameterlaw_by_function
    };
    /*----------------------------------------------------------------------*/
    /// material parameters for constant 1D_Artery
    ///
    // This object exists only once for each read Newton fluid. ???
    class Cnst1dArt : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      Cnst1dArt(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      /// Newtonian viscosity of blood
      const double viscosity_;
      /// density of blood
      const double density_;
      /// Artery Youngs modulus of elasticity */
      const double young_;
      /// Artery Poisson's ratio
      const double nue_;
      /// Artery wall thickness
      const double th_;
      /// Fixed external pressure at node 1
      const double pext1_;
      /// Fixed external pressure at node 2
      const double pext2_;
      /// viscosity law
      ArteryViscosityLaw viscositylaw_;
      /// viscosity law
      ArteryDiameterLaw diameterlaw_;
      //! used to scale the diameter for blood viscosity law to microns if your problem is not
      //! given in microns, e.g., if you use mms, set this parameter to 1.0e3
      const double blood_visc_scale_diam_to_microns_;
      //! function used for calculating the diameter
      const int diameter_law_funct_;
      //! collapse threshold (below this diameter, element is assumed to be collapsed with zero
      //! diameter and is not evaluated)
      const double collapse_threshold_;
      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class NewtonianFluid

  }  // namespace PAR

  class Cnst1dArtType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "Cnst_1d_artType"; }

    static Cnst1dArtType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static Cnst1dArtType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for constant 1D_Artery material
  ///
  /// This object exists (several times) at every element
  class Cnst1dArt : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    Cnst1dArt();

    /// construct the material object given material parameters
    explicit Cnst1dArt(Mat::PAR::Cnst1dArt* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return Cnst1dArtType::instance().unique_par_object_id();
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

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_cnst_art;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<Cnst1dArt>(*this);
    }

    /// return viscosity
    double viscosity() const;

    /// return density
    double density() const override { return params_->density_; }

    /// return DiameterLaw
    virtual Mat::PAR::ArteryDiameterLaw diameter_law() const { return params_->diameterlaw_; }

    /// return DiameterFunction
    virtual int diameter_function() const { return params_->diameter_law_funct_; }

    /// return Youngs modulus
    double young() const { return params_->young_; }

    /// return Poisson's ratio
    double nue() const { return params_->nue_; }

    /// set the artery diameter
    void set_diam(const double diam) { diam_ = diam; }

    /// set the initial artery diameter
    void set_diam_initial(const double diam) { diam_init_ = diam; }

    /// return artery diameter
    double diam() const { return diam_; }

    /// return initial artery diameter
    double diam_initial() const { return diam_init_; }

    /// return artery diameter of previous time step
    double diam_previous_time_step() const { return diam_previous_time_step_; }

    /// set the artery diameter of previous time step
    void set_diam_previous_time_step(const double diam_previous_time_step)
    {
      diam_previous_time_step_ = diam_previous_time_step;
    }

    /// check if element is collapsed
    bool is_collapsed() const { return diam_ < params_->collapse_threshold_; }

    /// return threshold for collapse
    double collapse_threshold() const { return params_->collapse_threshold_; }

    /// return artery wall thickness
    double th() const { return params_->th_; }

    /// return artery external pressure
    double pext(int i) const
    {
      if (i == 0)
        return params_->pext1_;
      else if (i == 1)
        return params_->pext2_;
      else
        FOUR_C_THROW("There is no pressure with id {}", i);
      return 0.0;
    }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /*! \brief Calculate blood viscosity based on empirical law for blood fully saturated with
     *         oxygen, i.e., hematocrit 0.45
     *
     *  Calculate blood viscosity based on empirical law by
     *  Pries AR, Secomb TW. 2005. Microvascular blood viscosity in vivo and the endothelial surface
     *  layer. Am. J. Physiol. Heart Circ. Physiol. 289:H2657-64
     *  https://doi.org/10.1152/ajpheart.00297.2005
     *  hematocrit of 0.45 (blood fully saturated with oxygen) is assumed
     *
     *  \note In the aforementioned paper, everything is given in a micrometer scaling, hence, this
     *        function assumes micro-meters. If spatial units are not given in micro-meters, caller
     *        of this function has to take care of passing diameter in units of micro-meter. If your
     *        problem is given in different units, consider the parameter
     *        BLOOD_VISC_SCALE_DIAM_TO_MICRONS, which may be used to scale your diameter to the
     *        appropriate units, i.e., for a problem with length of mm use
     *        BLOOD_VISC_SCALE_DIAM_TO_MICRONS = 1.0e3 to get diameter in microns
     *
     *  \param[in] diam        diameter of 1D element (in micro-meters)
     *  \param[in] plasmavisc  viscosity of blood plasma (should be approx. 1.05e-3 Pa s)
     *  \return    blood viscosity
     */
    double calculate_blood_viscosity(const double diam, const double plasmavisc) const;

    /// my material parameters
    Mat::PAR::Cnst1dArt* params_;
    /// Artery initial diameter
    double diam_init_;
    /// Artery current diameter
    double diam_;
    /// Artery diameter of previous time step
    double diam_previous_time_step_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
