// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PLASTICNLNLOGNEOHOOKE_HPP
#define FOUR_C_MAT_PLASTICNLNLOGNEOHOOKE_HPP

/*----------------------------------------------------------------------*
 | headers                                                              |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core
{
  namespace Utils
  {
    class FunctionOfAnything;
  }
}  // namespace Core

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    //! material parameters for neo-Hooke
    class PlasticNlnLogNeoHooke : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      PlasticNlnLogNeoHooke(const Core::Mat::PAR::Parameter::Data& matdata);

      //! @name material parameters
      //@{

      //! Young's modulus
      const double youngs_;
      //! Possion's ratio
      const double poissonratio_;
      //! mass density
      const double density_;
      //! initial yield stress (constant)
      const double yield_;
      //! linear isotropic hardening modulus
      const double isohard_;
      //! saturation yield stress
      const double infyield_;
      //! nonlinear hardening exponent
      const double hardexp_;
      //! viscosity
      const double visc_;
      //! rate dependency
      const double rate_dependency_;
      //! newton Raphson tolerance
      const double tolerance_nr_;
      //! function ID for evaluation of saturation
      const int functionID_hardening_;
      //! maximum number of Newton Raphson iterations
      const int max_iterations_;


      //@}

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class PlasticNlnLogNeoHooke

  }  // namespace PAR


  class PlasticNlnLogNeoHookeType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "PlasticNlnLogNeoHookeType"; }

    static PlasticNlnLogNeoHookeType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static PlasticNlnLogNeoHookeType instance_;

  };  // class PlasticNlnLogNeoHookeType

  /*----------------------------------------------------------------------*/
  //! wrapper for finite strain elasto-plastic material

  class PlasticNlnLogNeoHooke : public So3Material
  {
   public:
    //! construct empty material object
    PlasticNlnLogNeoHooke();

    //! construct the material object given material parameters
    explicit PlasticNlnLogNeoHooke(Mat::PAR::PlasticNlnLogNeoHooke* params);

    //! @name Packing and Unpacking

    /*!
    \brief Return unique ParObject id

    every class implementing ParObject needs a unique id defined at the
    top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return PlasticNlnLogNeoHookeType::instance().unique_par_object_id();
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

    //! @name Access methods

    //! material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_plnlnlogneohooke;
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (!(kinem == Inpar::Solid::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    //! return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<PlasticNlnLogNeoHooke>(*this);
    }

    //! density
    double density() const override { return params_->density_; }

    //! return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //! return accumulated strain at Gauss points
    //! use the old vector (last_) for postprocessing
    //! Output is called after(!!) Update, so that newest values are included in
    //! the old history vectors last_, while the current history vectors curr_
    //! are reset
    double accumulated_strain(int gp) const { return (accplstrainlast_.at(gp)); }

    //! return 1 if the material point is actively yielding (gamma > 0); else 0
    double active_yielding(int gp) const { return (activeyield_.at(gp)); }

    //! //! check if history variables are already initialized
    bool initialized() const { return (isinit_ and !accplstraincurr_.empty()); }

    //! return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    //! return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    //! return names of visualization data available for direct VTK output
    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    //! return visualization data for direct VTK output
    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

    //@}

    //! @name Evaluation methods

    //! initialise internal stress variables
    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    //! update internal stress variables
    void update() override;

    //! evaluate material law
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

   private:
    //! my material parameters
    Mat::PAR::PlasticNlnLogNeoHooke* params_;

    //! inverse right cauchy green of plastic strain
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> invplrcglast_;
    //! inverse right cauchy green of plastic strain
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> invplrcgcurr_;

    //! old (i.e. at t_n) accumulated plastic strain
    std::vector<double> accplstrainlast_;
    //! current (i.e. at t_n+1) accumulated plastic strain
    std::vector<double> accplstraincurr_;
    //! active yielding between t_n and t_n+1
    std::vector<double> activeyield_;

    const Core::Utils::FunctionOfAnything* hardening_function_{nullptr};

    //! indicator if #Initialize routine has been called
    bool isinit_;
  };  // class PlasticNlnLogNeoHooke
}  // namespace Mat

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
