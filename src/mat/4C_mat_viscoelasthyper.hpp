// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCOELASTHYPER_HPP
#define FOUR_C_MAT_VISCOELASTHYPER_HPP



#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_elasthyper.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declaration due to avoid header definition
namespace Mat
{
  namespace Elastic
  {
    class Summand;
  }

  // forward declaration
  class ViscoElastHyper;

  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// Collection of viscohyperelastic materials
    ///
    /// Storage map of hyperelastic summands.


    class ViscoElastHyper : public Mat::PAR::ElastHyper
    //    class ViscoElastHyper : public Core::Mat::PAR::Parameter
    {
      friend class Mat::ViscoElastHyper;

     public:
      /// standard constructor
      ///
      /// This constructor recursively calls the constructors of the
      /// parameter sets of the hyperelastic summands.
      ViscoElastHyper(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //@}

    };  // class ViscoElastHyper

  }  // namespace PAR

  class ViscoElastHyperType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ViscoElastHyperType"; }

    static ViscoElastHyperType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ViscoElastHyperType instance_;
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

  class Material;

  class ViscoElastHyper : public Mat::ElastHyper
  {
   public:
    /// construct empty material object
    ViscoElastHyper();

    /// construct the material object given material parameters
    explicit ViscoElastHyper(Mat::PAR::ViscoElastHyper* params);

    ///@name Packing and Unpacking
    //@{

    /// \brief Return unique ParObject id
    ///
    /// every class implementing ParObject needs a unique id defined at the
    /// top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return ViscoElastHyperType::instance().unique_par_object_id();
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
      return Core::Materials::m_viscoelasthyper;
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (!(kinem == Inpar::Solid::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ViscoElastHyper>(*this);
    }

    /// Check if history variables are already initialized
    virtual bool initialized() const { return isinitvis_ && (histscgcurr_ != nullptr); }

    /// hyperelastic stress response plus elasticity tensor
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp,
        int eleGID) override;  ///< Constitutive matrix

    /// setup material description
    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    /// update history variables
    void update() override;

   protected:
    /// calculates the kinematic quantities and tensors used afterwards for viscous part
    virtual void evaluate_kin_quant_vis(Core::LinAlg::Matrix<6, 1>& rcg,
        Core::LinAlg::Matrix<6, 1>& scg, Core::LinAlg::Matrix<6, 1>& icg,
        Core::LinAlg::Matrix<3, 1>& prinv, Core::LinAlg::Matrix<7, 1>& rateinv,
        Core::LinAlg::Matrix<6, 1>& modrcg, const Teuchos::ParameterList& params,
        const EvaluationContext<3>& context, Core::LinAlg::Matrix<6, 1>& scgrate,
        Core::LinAlg::Matrix<6, 1>& modrcgrate, Core::LinAlg::Matrix<7, 1>& modrateinv, int gp);

    /// calculates the factors associated to the viscous laws
    virtual void evaluate_mu_xi(Core::LinAlg::Matrix<3, 1>& inv, Core::LinAlg::Matrix<3, 1>& modinv,
        Core::LinAlg::Matrix<8, 1>& mu, Core::LinAlg::Matrix<8, 1>& modmu,
        Core::LinAlg::Matrix<33, 1>& xi, Core::LinAlg::Matrix<33, 1>& modxi,
        Core::LinAlg::Matrix<7, 1>& rateinv, Core::LinAlg::Matrix<7, 1>& modrateinv,
        const Teuchos::ParameterList& params, int gp, int eleGID);

    /// calculates the isotropic stress and elasticity tensor for viscous principal configuration
    virtual void evaluate_iso_visco_principal(Core::LinAlg::Matrix<6, 1>& stress,
        Core::LinAlg::Matrix<6, 6>& cmat, Core::LinAlg::Matrix<8, 1>& mu,
        Core::LinAlg::Matrix<33, 1>& xi, Core::LinAlg::Matrix<6, 6>& id4sharp,
        Core::LinAlg::Matrix<6, 1>& scgrate);

    /// calculates the isotropic stress and elasticity tensor for viscous decoupled configuration
    virtual void evaluate_iso_visco_modified(Core::LinAlg::Matrix<6, 1>& stressisomodisovisco,
        Core::LinAlg::Matrix<6, 1>& stressisomodvolvisco,
        Core::LinAlg::Matrix<6, 6>& cmatisomodisovisco,
        Core::LinAlg::Matrix<6, 6>& cmatisomodvolvisco, Core::LinAlg::Matrix<3, 1>& prinv,
        Core::LinAlg::Matrix<3, 1>& modinv, Core::LinAlg::Matrix<8, 1>& modmu,
        Core::LinAlg::Matrix<33, 1>& modxi, Core::LinAlg::Matrix<6, 1>& rcg,
        Core::LinAlg::Matrix<6, 1>& id2, Core::LinAlg::Matrix<6, 1>& icg,
        Core::LinAlg::Matrix<6, 6>& id4, Core::LinAlg::Matrix<6, 1>& modrcgrate);

    /// calculates the stress and elasticitiy tensor for the generalized Maxwell material
    virtual void evaluate_visco_generalized_maxwell(Core::LinAlg::Matrix<6, 1>& Q,
        Core::LinAlg::Matrix<6, 6>& cmatq, const Teuchos::ParameterList& params,
        const Mat::EvaluationContext<3>& context, const Core::LinAlg::Matrix<6, 1>* glstrain,
        int gp, int eleGID);

    /// calculates the stress and elasticity tensor for the VISCO_FSLS model
    /// depending on the viscoelastic material isochoric-principal, isochoric-modified
    /// (volumetric and isochoric equal treaded) or anisotropic stresses and elasticity
    /// tensors are added
    virtual void evaluate_visco_fsls(Core::LinAlg::Matrix<6, 1> stress,
        Core::LinAlg::Matrix<6, 6> cmat, Core::LinAlg::Matrix<6, 1>& Q,
        Core::LinAlg::Matrix<6, 6>& cmatq, const Teuchos::ParameterList& params,
        const Mat::EvaluationContext<3>& context, int gp);

    /// @name Flags to specify the viscous formulations
    //@{
    bool isovisco_;                   ///< global indicator for isotropic split viscous formulation
    bool visco_generalized_maxwell_;  ///< global indicator for viscous contribution of branches
                                      ///< according to the generalized Maxwell model
    bool visco_fsls_;  ///< global indicator for viscous contribution according the FSLS model
                       //@}

   private:
    /// viscous history Cauchy-Green-Tensor
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<6, 1>>>
        histscgcurr_;  ///< current Cauchy-Green-Tensor
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<6, 1>>>
        histscglast_;  ///< Cauchy-Green-Tensor of last converged state
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<6, 1>>>
        histmodrcgcurr_;  ///< current decoupled Cauchy-Green-Tensor
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<6, 1>>>
        histmodrcglast_;  ///< decoupled Cauchy-Green-Tensor of last converged state
    std::shared_ptr<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>
        histbranchstresscurr_;  ///< current stress of the branches
    std::shared_ptr<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>
        histbranchstresslast_;  ///< stress of the branches of last converged state
    std::shared_ptr<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>
        histbranchelaststresscurr_;  ///< current elastic stress of the branches
    std::shared_ptr<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>
        histbranchelaststresslast_;  ///< elastic stress of the branches of last converged state
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>
        histfslsartstresscurr_;  ///< current artificial stress of FSLS model
    std::shared_ptr<std::vector<std::vector<Core::LinAlg::Matrix<NUM_STRESS_3D, 1>>>>
        histfslsartstresslastall_;  ///< artificial FSLS stress history of converged states

    bool isinitvis_;  ///< indicates if #Initialized routine has been called
  };  // class ViscoElastHyper

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
