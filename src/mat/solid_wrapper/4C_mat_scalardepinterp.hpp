// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCALARDEPINTERP_HPP
#define FOUR_C_MAT_SCALARDEPINTERP_HPP



#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declaration due to avoid header definition
namespace Mat
{
  // forward declaration
  class ScalarDepInterp;

  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /*! \class ScalarDepInterp
     *  \brief Common parameters for scalar dependent interpolation between two material laws
     *

     */
    class ScalarDepInterp : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      ScalarDepInterp(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      /// elastic material number
      const int id_lambda_zero_;
      /// growthlaw material number
      const int id_lambda_unit_;
      //@}

      // create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class ScalarDepInterp

  }  // namespace PAR

  class ScalarDepInterpType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ScalarDepInterpType"; }

    static ScalarDepInterpType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ScalarDepInterpType instance_;
  };


  /*----------------------------------------------------------------------*/
  /// Collection of hyperelastic materials
  ///
  /// This collection offers to additively compose a stress response
  /// based on summands defined separately.  This is possible, because
  /// we deal with hyperelastic materials, which are composed
  /// of (Helmholtz free energy density) potentials.  Effectively, we want
  ///\f[
  ///  \Psi(\boldsymbol{C}) = \sum_i \Psi_i(\boldsymbol{C})
  ///\f]
  /// in which the individual \f$\Psi_i\f$ is implemented as #Mat::Elastic::Summand.
  ///
  /// Quite often the right Cauchy-Green 2-tensor \f$\boldsymbol{C}\f$
  /// is replaced by its various invariant forms as argument.
  ///
  /// The task of ElastHyper is the evaluation of the
  /// potential energies and their derivatives to obtain the actual
  /// stress response and the elasticity tensor. The storage is located
  /// at the associated member #params_.
  ///
  /// <h3>References</h3>
  /// <ul>
  /// <li> [1] GA Holzapfel, "Nonlinear solid mechanics", Wiley, 2000.
  /// </ul>
  ///


  class ScalarDepInterp : public So3Material
  {
   public:
    /// construct empty material object
    ScalarDepInterp();

    /// construct the material object given material parameters
    explicit ScalarDepInterp(Mat::PAR::ScalarDepInterp* params);

    ///@name Packing and Unpacking
    //@{

    /// \brief Return unique ParObject id
    ///
    /// every class implementing ParObject needs a unique id defined at the
    /// top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return ScalarDepInterpType::instance().unique_par_object_id();
    }

    /// \brief Pack this class so it can be communicated
    ///
    /// Resizes the vector data and stores all information of a class in it.
    /// The first information to be stored in data has to be the
    /// unique parobject id delivered by unique_par_object_id() which will then
    /// identify the exact class on the receiving processor.
    ///
    /// \param data (in/out): char vector to store class information
    void pack(Core::Communication::PackBuffer& data) const override;

    /// \brief Unpack data from a char vector into this class
    ///
    /// The vector data contains all information to rebuild the
    /// exact copy of an instance of a class on a different processor.
    /// The first entry in data has to be an integer which is the unique
    /// parobject id defined at the top of this file and delivered by
    /// unique_par_object_id().
    ///
    /// \param data (in) : vector storing all data to be unpacked into this
    ///                    instance.
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_sc_dep_interp;
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (!(kinem == Solid::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ScalarDepInterp>(*this);
    }

    /// number of materials
    //    virtual int NumMat() const { return params_->nummat_; }

    /// Update
    void update() override;

    /// Reset time step
    void reset_step() override;

    double density() const override
    {
      // Note: we have already check that both densities are equal in Setuo()
      return lambda_zero_mat_->density();
    };

    /// provide access to material by its ID
    //     virtual std::shared_ptr<const Mat::Elastic::Summand> MaterialById(const int id) const {
    //     return params_->MaterialById(id); }


    /// hyperelastic stress response plus elasticity tensor
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    /// setup
    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// Return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    /// Return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    /// evaluate strain energy function
    [[nodiscard]] double strain_energy(const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const EvaluationContext<3>& context, const int gp, const int eleGID) const override;

   private:
    /// my material parameters
    Mat::PAR::ScalarDepInterp* params_;

    /// indicates if material is initialized
    bool isinit_;

    /// elastic material for zero lambda
    std::shared_ptr<Mat::So3Material> lambda_zero_mat_;

    /// elastic material for unit lambda
    std::shared_ptr<Mat::So3Material> lambda_unit_mat_;

    /// interpolation parameter (0-->IDMATZEROSC,1-->IDMATUNITSC)
    std::vector<double> lambda_;
  };


}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
