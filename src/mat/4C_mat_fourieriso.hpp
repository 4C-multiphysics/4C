#ifndef FOUR_C_MAT_FOURIERISO_HPP
#define FOUR_C_MAT_FOURIERISO_HPP

/*----------------------------------------------------------------------*
 |  headers                                                  dano 09/09 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_thermo.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |                                                           dano 09/09 |
 *----------------------------------------------------------------------*/
namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for FourierIso material
    ///
    /// <h3>Input line</h3>
    /// MAT 1 THERM_FourierIsoIso CAPA 1.0 COND 1.0
    class FourierIso : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      FourierIso(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// volumetric heat capacity
      const double capa_;
      /// heat conductivity
      const double conduct_;

      //@}

      /// create material instance of matching type with my parameters
      Teuchos::RCP<Core::Mat::Material> create_material() override;

    };  // class FourierIso

  }  // namespace PAR

  class FourierIsoType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "FourierIsoType"; }

    static FourierIsoType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static FourierIsoType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// FourierIso material according to [1]
  ///
  /// This is a FourierIso's law of isotropic, instationary heat conduction
  ///
  /// <h3>References</h3>
  /// <ul>
  /// <li> [1] GA Holzapfel, "Nonlinear solid mechanics", Wiley, 2000.
  /// </ul>
  ///
  /// \author dano
  /// \date 09/09
  class FourierIso : public ThermoMaterial
  {
   public:
    /// empty constructor
    FourierIso();

    /// constructor with given material parameters
    FourierIso(Mat::PAR::FourierIso* params);

    /// @name Packing and Unpacking
    //@{

    /// Return unique ParObject id
    ///
    ///  every class implementing ParObject needs a unique id defined at the
    ///  top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return FourierIsoType::instance().unique_par_object_id();
    }

    /// Pack this class so it can be communicated
    ///
    /// Resizes the vector data and stores all information of a class in it.
    /// The first information to be stored in data has to be the
    /// unique parobject id delivered by unique_par_object_id() which will then
    /// identify the exact class on the receiving processor.
    void pack(
        Core::Communication::PackBuffer& data  ///< (in/out): char vector to store class information
    ) const override;

    /// \brief Unpack data from a char vector into this class
    ///
    /// The vector data contains all information to rebuild the
    /// exact copy of an instance of a class on a different processor.
    /// The first entry in data has to be an integer which is the unique
    /// parobject id defined at the top of this file and delivered by
    /// unique_par_object_id().
    ///
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    /// @name Access material constants
    //@{

    /// conductivity
    double conductivity() const { return params_->conduct_; }

    /// volumetric heat capacity
    double capacity() const override { return params_->capa_; }

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_th_fourier_iso;
    }

    /// return copy of this material object
    Teuchos::RCP<Core::Mat::Material> clone() const override
    {
      return Teuchos::make_rcp<FourierIso>(*this);
    }

    //@}

    void evaluate(const Core::LinAlg::Matrix<1, 1>& gradtemp, Core::LinAlg::Matrix<1, 1>& cmat,
        Core::LinAlg::Matrix<1, 1>& heatflux) const override;

    void evaluate(const Core::LinAlg::Matrix<2, 1>& gradtemp, Core::LinAlg::Matrix<2, 2>& cmat,
        Core::LinAlg::Matrix<2, 1>& heatflux) const override;

    void evaluate(const Core::LinAlg::Matrix<3, 1>& gradtemp, Core::LinAlg::Matrix<3, 3>& cmat,
        Core::LinAlg::Matrix<3, 1>& heatflux) const override;

    void conductivity_deriv_t(Core::LinAlg::Matrix<3, 3>& dCondDT) const override
    {
      dCondDT.clear();
    }

    void conductivity_deriv_t(Core::LinAlg::Matrix<2, 2>& dCondDT) const override
    {
      dCondDT.clear();
    }

    void conductivity_deriv_t(Core::LinAlg::Matrix<1, 1>& dCondDT) const override
    {
      dCondDT.clear();
    }

    double capacity_deriv_t() const override { return 0; }

    void reinit(double temperature, unsigned gp) override
    {
      // do nothing
    }

    void reset_current_state() override
    {
      // do nothing
    }

    void commit_current_state() override
    {
      // do nothing
    }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /// my material parameters
    Mat::PAR::FourierIso* params_;

  };  // FourierIso

}  // namespace Mat

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
