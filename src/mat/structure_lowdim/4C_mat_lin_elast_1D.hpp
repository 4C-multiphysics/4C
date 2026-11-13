// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_LIN_ELAST_1D_HPP
#define FOUR_C_MAT_LIN_ELAST_1D_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    class LinElast1D : public Core::Mat::PAR::Parameter
    {
     public:
      LinElast1D(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      /// Young's modulus
      const double youngs_;

      /// mass density
      const double density_;
      //@}

      std::shared_ptr<Core::Mat::Material> create_material() override;
    };

    class LinElast1DGrowth : public LinElast1D
    {
     public:
      LinElast1DGrowth(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      /// reference concentration without inelastic deformation
      const double c0_;

      /// order of polynomial for inelastic growth
      const int poly_num_;

      /// parameters of polynomial for inelastic growth
      const std::vector<double> poly_params_;

      /// growth proportional to amount of substance (true) or proportional to concentration (false)
      const bool amount_prop_growth_;
      //@}

      std::shared_ptr<Core::Mat::Material> create_material() override;
    };
  }  // namespace PAR

  class LinElast1DType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "LinElast1DType"; }

    static LinElast1DType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static LinElast1DType instance_;
  };

  class LinElast1D : public Core::Mat::Material
  {
   public:
    explicit LinElast1D(Mat::PAR::LinElast1D* params);

    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<LinElast1D>(*this);
    }

    /// mass density
    double density() const override { return params_->density_; }

    /// elastic energy based on @p epsilon
    double evaluate_elastic_energy(const double epsilon) const
    {
      return 0.5 * evaluate_p_k2(epsilon) * epsilon;
    }

    /// evaluate 2nd Piola-Kirchhoff stress based on @param epsilon (Green-Lagrange strain)
    double evaluate_p_k2(const double epsilon) const { return params_->youngs_ * epsilon; }

    /// evaluate stiffness of material i.e. derivative of 2nd Piola Kirchhoff stress w.r.t.
    /// Green-Lagrange strain
    double evaluate_stiffness() const { return params_->youngs_; }

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_linelast1D;
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    int unique_par_object_id() const override
    {
      return LinElast1DType::instance().unique_par_object_id();
    }

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

   private:
    /// my material parameters
    Mat::PAR::LinElast1D* params_;
  };


  class LinElast1DGrowthType : public Core::Communication::ParObjectType
  {
   public:
    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

    static LinElast1DGrowthType& instance() { return instance_; }

    std::string name() const override { return "LinElast1DGrowthType"; }

   private:
    static LinElast1DGrowthType instance_;
  };

  class LinElast1DGrowth : public LinElast1D
  {
   public:
    explicit LinElast1DGrowth(Mat::PAR::LinElast1DGrowth* params);

    /// growth proportional to amount of substance or to concentration
    bool amount_prop_growth() const { return growth_params_->amount_prop_growth_; }

    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<LinElast1DGrowth>(*this);
    }
    /// elastic energy based on @p def_grad and @p conc
    double evaluate_elastic_energy(double def_grad, double conc) const;

    /// 2nd Piola-Kirchhoff stress based on @p def_grad and @p conc
    double evaluate_p_k2(double def_grad, double conc) const;

    /// stiffness, i.e. derivative of 2nd Piola-Kirchhoff stress w.r.t. @p def_grad based on @p
    /// def_grad and @p conc
    double evaluate_stiffness(double def_grad, double conc) const;

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_linelast1D_growth;
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    Core::Mat::PAR::Parameter* parameter() const override { return growth_params_; }

    int unique_par_object_id() const override
    {
      return LinElast1DGrowthType::instance().unique_par_object_id();
    }

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

   private:
    /// polynomial growth factor based on amount of substance (@p conc * @p def_grad)
    double get_growth_factor_ao_s_prop(double conc, double def_grad) const;

    /// derivative of polynomial growth factor based on amount of substance w.r.t @p def_grad
    double get_growth_factor_ao_s_prop_deriv(double conc, double def_grad) const;

    /// polynomial growth factor based on concentration (@p conc)
    double get_growth_factor_conc_prop(double conc) const;

    /// my material parameters
    Mat::PAR::LinElast1DGrowth* growth_params_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
