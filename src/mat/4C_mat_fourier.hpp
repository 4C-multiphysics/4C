// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_FOURIER_HPP
#define FOUR_C_MAT_FOURIER_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_io_input_field.hpp"
#include "4C_mat_thermo.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

/* Fourier material according to [1]
 *
 * This is a Fourier's law of instationary heat conduction.
 * The anisotropic version is the extension of the isotropic case with the
 * thermal conductivity treated as tensor.
 *
 * <h3>References</h3>
 * <ul>
 * <li> [1] GA Holzapfel, "Nonlinear solid mechanics", Wiley, 2000.
 * </ul>
 */

namespace Mat
{
  namespace PAR
  {
    /// material parameters for Fourier material
    class Fourier : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      Fourier(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// volumetric heat capacity
      /// be careful: capa_ := rho * C_V, e.g contains the density
      const double capa_;

      /// conductivity tensor as row-wise vector
      const Core::IO::InputField<std::vector<double>> conductivity_;

      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;
    };
  }  // namespace PAR

  class FourierType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "FourierType"; }

    static FourierType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static FourierType instance_;
  };

  class Fourier : public ThermoMaterial
  {
   public:
    Fourier();

    explicit Fourier(Mat::PAR::Fourier* params);

    /// @name Packing and Unpacking
    //@{

    /// Return unique ParObject id
    ///
    ///  every class implementing ParObject needs a unique id defined at the
    ///  top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return FourierType::instance().unique_par_object_id();
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

    /// get element defined conductivity, defaults back to globally defined one if no element gid is
    /// given
    std::vector<double> conductivity(int eleGID) const override;

    /// volumetric heat capacity
    double capacity() const override { return params_->capa_; }

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_thermo_fourier;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<Fourier>(*this);
    }

    //@}

    void evaluate(const Core::LinAlg::Matrix<1, 1>& gradtemp, Core::LinAlg::Matrix<1, 1>& cmat,
        Core::LinAlg::Matrix<1, 1>& heatflux, const int eleGID) const override;

    void evaluate(const Core::LinAlg::Matrix<2, 1>& gradtemp, Core::LinAlg::Matrix<2, 2>& cmat,
        Core::LinAlg::Matrix<2, 1>& heatflux, const int eleGID) const override;

    void evaluate(const Core::LinAlg::Matrix<3, 1>& gradtemp, Core::LinAlg::Matrix<3, 3>& cmat,
        Core::LinAlg::Matrix<3, 1>& heatflux, const int eleGID) const override;

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

    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    Mat::PAR::Fourier* params_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
