// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCATRA_MULTIPORO_HPP
#define FOUR_C_MAT_SCATRA_MULTIPORO_HPP


#include "4C_config.hpp"

#include "4C_mat_scatra.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace ScaTraMatMultiPoro
  {
    /*--------------------------------------------------------------------------*/
    /*!
     * \brief enum that provides all possible MatScatraMultiPoro species types
     *///                                                        kremheller 03/18
    /*--------------------------------------------------------------------------*/
    enum SpeciesType
    {
      // species type
      species_in_fluid,    /*!< species in fluid */
      species_in_volfrac,  /*!< species in volume fraction */
      species_in_solid,    /*!< species in solid */
      species_temperature, /*!< temperature */
      species_undefined
    };  // enum SpeciesType

    /**
     *  \brief a structure for the mapping of scalars to the transporting phase
     */
    struct ScalarToPhaseMap
    {
      int phaseID;              /*!< ID of the fluid phase containing the scalar*/
      SpeciesType species_type; /*!< species type */
    };

  }  // namespace ScaTraMatMultiPoro


  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// parameters for scalar transport material (species in fluid)
    class ScatraMatMultiPoroFluid : public ScatraMat
    {
     public:
      /// standard constructor
      ScatraMatMultiPoroFluid(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// ID of fluid phase the scalar is associated with
      const int phaseID_;
      /// delta used for modelling dependency of diffusivity on (saturation*porosity)^delta
      //  as in G. Sciume, William G. Gray, F. Hussain, M. Ferrari, P. Decuzzi, and B. A. Schrefler.
      //  Three phase flow dynamics in tumor growth. Computational Mechanics, 53:465-484,
      //  2014.
      const double delta_;
      // minimum saturation under which also corresponding mass fraction is equal to zero
      const double min_sat_;
      /// function ID of relative mobility function
      const int relative_mobility_funct_id_;
    };
    // class Scatra

    /*----------------------------------------------------------------------*/
    /// parameters for scalar transport material (species in volume fraction)
    class ScatraMatMultiPoroVolFrac : public ScatraMat
    {
     public:
      /// standard constructor
      ScatraMatMultiPoroVolFrac(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// ID of fluid phase the scalar is associated with
      const int phaseID_;
      /// delta used for modelling dependency of diffusivity on volfrac^delta
      const double delta_;
      /// function ID of relative mobility function
      const int relative_mobility_funct_id_;
    };
    // class Scatra
    /*----------------------------------------------------------------------*/
    /// parameters for scalar transport material (species in solid)
    class ScatraMatMultiPoroSolid : public ScatraMat
    {
     public:
      /// standard constructor
      ScatraMatMultiPoroSolid(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// delta used for modelling dependency of diffusivity on
      /// (saturation*porosity)^delta
      //  as in G. Sciume, William G. Gray, F. Hussain, M. Ferrari, P. Decuzzi, and
      //  B. A. Schrefler. Three phase flow dynamics in tumor growth. Computational
      //  Mechanics, 53:465-484, 2014.
      const double delta_;
    };
    // class Scatra
    /*----------------------------------------------------------------------*/
    /// parameters for scalar transport material (temperature)
    class ScatraMatMultiPoroTemperature : public ScatraMat
    {
     public:
      /// standard constructor
      ScatraMatMultiPoroTemperature(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      const int numfluidphases_;
      const int numvolfrac_;
      // heat capacities for fluids, volfracs and solid
      const std::vector<double> cp_fluid_;
      const std::vector<double> cp_volfrac_;
      const double cp_solid_;

      // thermal diffusivity for fluids, volfracs and solid
      const std::vector<double> kappa_fluid_;
      const std::vector<double> kappa_volfrac_;
      const double kappa_solid_;
    };
    // class Scatra

  }  // namespace PAR

  class ScatraMatMultiPoroFluidType : public ScatraMatType
  {
   public:
    std::string name() const override { return "ScatraMatMultiPoroFluidType"; }

    static ScatraMatMultiPoroFluidType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ScatraMatMultiPoroFluidType instance_;
  };

  class ScatraMatMultiPoroVolFracType : public ScatraMatType
  {
   public:
    std::string name() const override { return "ScatraMatMultiPoroVolFracType"; }

    static ScatraMatMultiPoroVolFracType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ScatraMatMultiPoroVolFracType instance_;
  };

  class ScatraMatMultiPoroSolidType : public ScatraMatType
  {
   public:
    std::string name() const override { return "ScatraMatMultiPoroSolidType"; }

    static ScatraMatMultiPoroSolidType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ScatraMatMultiPoroSolidType instance_;
  };
  class ScatraMatMultiPoroTemperatureType : public ScatraMatType
  {
   public:
    std::string name() const override { return "ScatraMatMultiPoroTemperatureType"; }

    static ScatraMatMultiPoroTemperatureType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ScatraMatMultiPoroTemperatureType instance_;
  };
  /*----------------------------------------------------------------------*/
  /// wrapper for scalar transport material (species in fluid)
  class ScatraMatMultiPoroFluid : public ScatraMat
  {
   public:
    /// construct empty material object
    ScatraMatMultiPoroFluid();

    /// construct the material object given material parameters
    explicit ScatraMatMultiPoroFluid(Mat::PAR::ScatraMatMultiPoroFluid* params);

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return ScatraMatMultiPoroFluidType::instance().unique_par_object_id();
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
      return Core::Materials::m_scatra_in_fluid_porofluid_pressure_based;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ScatraMatMultiPoroFluid>(*this);
    }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// return phase ID
    virtual int phase_id() const { return params_->phaseID_; }

    /// return delta
    virtual double delta() const { return params_->delta_; }

    /// return minimum saturation
    virtual double min_sat() const { return params_->min_sat_; }

    /// return ID of relative mobility function
    [[nodiscard]] int relative_mobility_funct_id() const
    {
      return params_->relative_mobility_funct_id_;
    }

   private:
    /// my material parameters
    Mat::PAR::ScatraMatMultiPoroFluid* params_;
  };

  /*----------------------------------------------------------------------*/
  /// wrapper for scalar transport material (species in volfrac)
  class ScatraMatMultiPoroVolFrac : public ScatraMat
  {
   public:
    /// construct empty material object
    ScatraMatMultiPoroVolFrac();

    /// construct the material object given material parameters
    explicit ScatraMatMultiPoroVolFrac(Mat::PAR::ScatraMatMultiPoroVolFrac* params);

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return ScatraMatMultiPoroVolFracType::instance().unique_par_object_id();
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
      return Core::Materials::m_scatra_in_volfrac_porofluid_pressure_based;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ScatraMatMultiPoroVolFrac>(*this);
    }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// return phase ID
    virtual int phase_id() const { return params_->phaseID_; }

    /// return delta
    virtual double delta() const { return params_->delta_; }

    /// return ID of relative mobility function
    [[nodiscard]] int relative_mobility_funct_id() const
    {
      return params_->relative_mobility_funct_id_;
    }

   private:
    /// my material parameters
    Mat::PAR::ScatraMatMultiPoroVolFrac* params_;
  };

  /*----------------------------------------------------------------------*/
  /// wrapper for scalar transport material (species in solid)
  class ScatraMatMultiPoroSolid : public ScatraMat
  {
   public:
    /// construct empty material object
    ScatraMatMultiPoroSolid();

    /// construct the material object given material parameters
    explicit ScatraMatMultiPoroSolid(Mat::PAR::ScatraMatMultiPoroSolid* params);

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return ScatraMatMultiPoroSolidType::instance().unique_par_object_id();
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
      return Core::Materials::m_scatra_in_solid_porofluid_pressure_based;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ScatraMatMultiPoroSolid>(*this);
    }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// return delta
    virtual double delta() const { return params_->delta_; }

   private:
    /// my material parameters
    Mat::PAR::ScatraMatMultiPoroSolid* params_;
  };

  /*----------------------------------------------------------------------*/
  /// wrapper for scalar transport material (species in temperature)
  class ScatraMatMultiPoroTemperature : public ScatraMat
  {
   public:
    /// construct empty material object
    ScatraMatMultiPoroTemperature();

    /// construct the material object given material parameters
    explicit ScatraMatMultiPoroTemperature(Mat::PAR::ScatraMatMultiPoroTemperature* params);

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return ScatraMatMultiPoroTemperatureType::instance().unique_par_object_id();
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
      return Core::Materials::m_scatra_as_temperature_porofluid_pressure_based;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ScatraMatMultiPoroTemperature>(*this);
    }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    std::vector<double> cp_fluid() const { return params_->cp_fluid_; }
    double cp_fluid(int phase) const { return params_->cp_fluid_[phase]; }
    std::vector<double> cp_volfrac() const { return params_->cp_volfrac_; }
    double cp_volfrac(int phase) const { return params_->cp_volfrac_[phase]; }
    double cp_solid() const { return params_->cp_solid_; }

    std::vector<double> kappa_fluid() const { return params_->kappa_fluid_; }
    double kappa_fluid(int phase) const { return params_->kappa_fluid_[phase]; };
    std::vector<double> kappa_volfrac() const { return params_->kappa_volfrac_; }
    double kappa_volfrac(int phase) const { return params_->kappa_volfrac_[phase]; }
    double kappa_solid() const { return params_->kappa_solid_; }

   private:
    /// my material parameters
    Mat::PAR::ScatraMatMultiPoroTemperature* params_;
  };

}  // namespace Mat



FOUR_C_NAMESPACE_CLOSE

#endif
