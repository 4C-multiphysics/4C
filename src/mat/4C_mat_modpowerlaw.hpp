/*----------------------------------------------------------------------*/
/*! \file
\brief
Nonlinear viscosity according to a modified power law


\level 3
*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_MAT_MODPOWERLAW_HPP
#define FOUR_C_MAT_MODPOWERLAW_HPP



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
    /*----------------------------------------------------------------------*/
    /// material parameters
    class ModPowerLaw : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      ModPowerLaw(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      const double m_cons_;  /* consistency */
      const double delta_;   /* safety factor */
      const double a_exp_;   /* exponent */
      const double density_; /* density */
      //@}

      /// create material instance of matching type with my parameters
      Teuchos::RCP<Core::Mat::Material> create_material() override;

    };  // class ModPowerLaw

  }  // namespace PAR

  class ModPowerLawType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ModPowerLawType"; }

    static ModPowerLawType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ModPowerLawType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Nonlinear viscosity according to a modified power law
  class ModPowerLaw : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    ModPowerLaw();

    /// construct the material object given material parameters
    explicit ModPowerLaw(Mat::PAR::ModPowerLaw* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */

    int unique_par_object_id() const override
    {
      return ModPowerLawType::instance().unique_par_object_id();
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
      return Core::Materials::m_modpowerlaw;
    }

    /// return copy of this material object
    Teuchos::RCP<Core::Mat::Material> clone() const override
    {
      return Teuchos::rcp(new ModPowerLaw(*this));
    }

    /// return material parameters for element calculation

    /// consistency constant
    double m_cons() const { return params_->m_cons_; }
    /// safety factor
    double delta() const { return params_->delta_; }
    /// exponent
    double a_exp() const { return params_->a_exp_; }
    /// density
    double density() const override { return params_->density_; }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /// my material parameters
    Mat::PAR::ModPowerLaw* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
