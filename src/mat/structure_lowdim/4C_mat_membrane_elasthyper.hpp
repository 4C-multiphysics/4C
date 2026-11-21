// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MEMBRANE_ELASTHYPER_HPP
#define FOUR_C_MAT_MEMBRANE_ELASTHYPER_HPP

/*----------------------------------------------------------------------*
 | headers                                                              |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_mat_elasthyper.hpp"
#include "4C_mat_membrane_material_interfaces.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | hyperelastic material for membranes                                  |
 *----------------------------------------------------------------------*/
namespace Mat
{
  // forward declaration
  namespace Elastic
  {
    class Summand;
  }  // namespace Elastic

  class MembraneElastHyper;

  namespace PAR
  {
    class MembraneElastHyper : public Mat::PAR::ElastHyper
    {
      friend class Mat::MembraneElastHyper;

     public:
      /// standard constructor
      MembraneElastHyper(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class MembraneElastHyper

  }  // namespace PAR

  class MembraneElastHyperType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "Membrane_ElastHyperType"; }

    static MembraneElastHyperType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static MembraneElastHyperType instance_;
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


  // forward declaration
  class Material;

  class MembraneElastHyper : public ElastHyper, public Mat::MembraneMaterialLocalCoordinates
  {
   public:
    /// construct empty material object
    MembraneElastHyper();

    /// construct the material object given material parameters
    explicit MembraneElastHyper(Mat::PAR::MembraneElastHyper* params);

    ///@name Packing and Unpacking
    //@{

    /// \brief Return unique ParObject id
    ///
    /// every class implementing ParObject needs a unique id defined at the
    /// top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return MembraneElastHyperType::instance().unique_par_object_id();
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
      return Core::Materials::m_membrane_elasthyper;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<MembraneElastHyper>(*this);
    }

    /// setup
    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    void update_membrane(const Core::LinAlg::Matrix<3, 3>& defgrd,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        const Core::LinAlg::Matrix<3, 3>& Q_trafo, int gp, int eleGID) override
    {
      // nothing to do
    }

    void evaluate_membrane(const Core::LinAlg::Matrix<3, 3>& defgrd,
        const Core::LinAlg::Matrix<3, 3>& cauchygreen, const Teuchos::ParameterList& params,
        const EvaluationContext<3>& context, const Core::LinAlg::Matrix<3, 3>& Q_trafo,
        Core::LinAlg::Matrix<3, 1>& stress, Core::LinAlg::Matrix<3, 3>& cmat, int gp,
        int eleGID) override;

    /// evaluate strain energy function
    virtual void strain_energy(
        Core::LinAlg::Matrix<3, 3>& cauchygreen,  ///< right Cauchy-Green tensor
        double& psi,                              ///< Strain energy function
        int gp,                                   ///< Gauss point
        int eleGID                                ///< Element GID
    );

   protected:
    /// calculate anisotropic stress and elasticity tensor
    virtual void evaluate_anisotropic_stress_cmat(Core::LinAlg::Matrix<3, 1>& stress_aniso,
        Core::LinAlg::Matrix<3, 3>& cmat_aniso, const Core::LinAlg::Matrix<3, 3>& Q_trafo,
        const Core::LinAlg::Matrix<3, 1>& rcg, const double& rcg33,
        const Teuchos::ParameterList& params, int gp, int eleGID);

    /// vector of fiber vectors
    std::vector<Core::LinAlg::Tensor<double, 3>> fibervecs_;
  };

}  // namespace Mat

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
