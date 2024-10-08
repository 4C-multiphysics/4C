/*----------------------------------------------------------------------------*/
/*! \file
\brief material stores parameters for ion species in electrolyte solution. The scl material is
derived for a binary electrolyte assuming a mobile ionic species in a fixed anion lattice.
Local electroneutrality is dismissed, which enables formation of Space-Charge-Layers (SCLs).

\level 2


*/
/*----------------------------------------------------------------------------*/

#ifndef FOUR_C_MAT_SCL_HPP
#define FOUR_C_MAT_SCL_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_elchsinglemat.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for electrolytes including space-charge-layer formation
    class Scl : public ElchSingleMat
    {
     public:
      Scl(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      /// valence (= charge number)
      const double valence_;

      /// definition of transference number
      /// (by function number or implemented concentration dependence)
      const int transnrcurve_;

      /// number of parameter needed for implemented concentration dependence
      const int transnrparanum_;

      /// parameter needed for implemented concentration dependence
      const std::vector<double> transnr_;

      //! maximum concentration of species
      const double cmax_;

      //! strategy for extrapolation of diffusion coefficient
      const int extrapolation_diffussion_coeff_strategy_;

      //! limit concentration for extrapolation strategy
      const double clim_;

      //! bulk concentration i.e. anion concentration for equal transference numbers
      const double cbulk_;

      //! dieelectric susceptibility of electrolyte material
      const double susceptibility_;

      //! difference in partial molar volumes (vacancy <=> interstitial)
      const double delta_nu_;

      //! Faraday constant
      const double faraday_;

      //! vacuum Permittivity
      const double epsilon_0_;
      //@}

      /// create material instance of matching type with my parameters
      Teuchos::RCP<Core::Mat::Material> create_material() override;
    };

  }  // namespace PAR

  class SclType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "SclType"; }

    static SclType& instance() { return instance_; }

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static SclType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for the material properties of an ion species in an electrolyte solution
  class Scl : public ElchSingleMat
  {
   public:
    /// construct empty material object
    Scl();

    /// construct the material object given material parameters
    explicit Scl(Mat::PAR::Scl* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of drt_parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override { return SclType::instance().unique_par_object_id(); }

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
    Core::Materials::MaterialType material_type() const override { return Core::Materials::m_scl; }

    /// return copy of this material object
    Teuchos::RCP<Core::Mat::Material> clone() const override
    {
      return Teuchos::RCP(new Scl(*this));
    }

    /// valence (= charge number)
    double valence() const { return params_->valence_; }

    /// computation of the transference number based on the defined curve
    double compute_transference_number(const double cint) const;

    /// computation of the first derivative of the transference number based on the defined curve
    double compute_first_deriv_trans(const double cint) const;

    double compute_diffusion_coefficient(
        const double concentration, const double temperature) const override;

    double compute_concentration_derivative_of_diffusion_coefficient(
        const double concentration, const double temperature) const override;

    /// computation of dielectric susceptibility (currently a constant)
    double compute_susceptibility() const { return params_->susceptibility_; }

    /// computation of 1/(z^2F^2) with valence of cations
    double inv_val_valence_faraday_squared() const;

    /// Computation of dielectric permittivity based on dieelectric susceptibility
    double compute_permittivity() const;

    /// Returns Value of cation concentration in the neutral bulk (= anion concentration)
    double bulk_concentration() const { return params_->cbulk_; }

    /// computation of mobility factor in linear onsager ansatz
    double compute_onsager_coefficient(const double concentration, const double temperature) const;

    /// computation of the derivative of the mobility factor w.r.t to cation concentration
    double compute_concentration_derivative_of_onsager_coefficient(
        const double concentration, const double temperature) const;

   private:
    /// return curve defining the transference number
    int trans_nr_curve() const { return params_->transnrcurve_; }

    /// parameter needed for implemented concentration dependence
    const std::vector<double>& trans_nr_params() const { return params_->transnr_; }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// my material parameters
    Mat::PAR::Scl* params_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
