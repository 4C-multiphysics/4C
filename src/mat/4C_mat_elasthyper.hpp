// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELASTHYPER_HPP
#define FOUR_C_MAT_ELASTHYPER_HPP



#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_elasthyper_service.hpp"
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
  class ElastHyper;

  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// Collection of hyperelastic materials
    ///
    /// Storage map of hyperelastic summands.
    ///
    /// \date 05/09

    class ElastHyper : public Core::Mat::PAR::Parameter
    {
      friend class Mat::ElastHyper;

     public:
      /// standard constructor
      ///
      /// This constructor recursively calls the constructors of the
      /// parameter sets of the hyperelastic summands.
      explicit ElastHyper(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      //       /// provide access to material/summand by its ID
      //       std::shared_ptr<const Mat::Elastic::Summand> MaterialById(
      //         const int id  ///< ID to look for in collection of summands
      //         ) const;

      /// length of material list
      const int nummat_;

      /// the list of material IDs
      const std::vector<int> matids_;

      /// material mass density
      const double density_;

      /// 1.0 if polyconvexity of system is checked (0 = no = default)
      const int polyconvex_;

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //@}

    };  // class ElastHyper

  }  // namespace PAR

  class ElastHyperType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ElastHyperType"; }

    static ElastHyperType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ElastHyperType instance_;
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
  /// \date 05/09
  class Material;

  // forward declaration of unit test
  class ElastHyperService_TestSuite;

  class ElastHyper : public So3Material
  {
   public:
    friend class Mat::PAR::ElastHyper;

    /// construct empty material object
    ElastHyper();

    /// construct the material object given material parameters
    explicit ElastHyper(Mat::PAR::ElastHyper* params);

    ///@name Packing and Unpacking
    //@{

    /// \brief Return unique ParObject id
    ///
    /// every class implementing ParObject needs a unique id defined at the
    /// top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return ElastHyperType::instance().unique_par_object_id();
    }

    /// \brief Pack this class so it can be communicated
    ///
    /// Resizes the vector data and stores all information of a class in it.
    /// The first information to be stored in data has to be the
    /// unique parobject id delivered by unique_par_object_id() which will then
    /// identify the exact class on the receiving processor.
    ///
    /// \param data (in/out) : char vector to store class information
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
      return Core::Materials::m_elasthyper;
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (kinem != Inpar::Solid::KinemType::nonlinearTotLag)
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ElastHyper>(*this);
    }

    /// number of materials
    virtual int num_mat() const { return params_->nummat_; }

    /*!
     * @brief deliver material ID of index i'th potential summand in collection
     *
     * @param(in) index index
     * @return material id
     */
    virtual int mat_id(unsigned index) const;

    /// material mass density
    double density() const override { return params_->density_; }

    /// a shear modulus equivalent
    virtual double shear_mod() const;

    /// a young's modulus equivalent
    virtual double get_young();

    /// evaluate strain energy function
    void strain_energy(const Core::LinAlg::Matrix<6, 1>& glstrain,  ///< Green-Lagrange strain
        double& psi,                                                ///< Strain energy function
        int gp,                                                     ///< Gauss point
        int eleGID                                                  ///< Element GID
    ) const override;

    /*!
     * \brief Hyperelastic stress response plus elasticity tensor
     *
     * \param defgrd(in) : Deformation gradient
     * \param glstrain(in) : Green-Lagrange strain
     * \param params(in) : Container for additional information
     * \param stress(in,out) : 2nd Piola-Kirchhoff stresses
     * \param cmat(in,out) : Constitutive matrix
     * \param gp(in) : Gauss point
     * \param eleGID(in) : Element GID
     */
    void evaluate(const Core::LinAlg::Matrix<3, 3>* defgrd,
        const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
        Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat, int gp,
        int eleGID) override;

    void evaluate_cauchy_n_dir_and_derivatives(const Core::LinAlg::Matrix<3, 3>& defgrd,
        const Core::LinAlg::Matrix<3, 1>& n, const Core::LinAlg::Matrix<3, 1>& dir,
        double& cauchy_n_dir, Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dn,
        Core::LinAlg::Matrix<3, 1>* d_cauchyndir_ddir, Core::LinAlg::Matrix<9, 1>* d_cauchyndir_dF,
        Core::LinAlg::Matrix<9, 9>* d2_cauchyndir_dF2,
        Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_dn,
        Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_ddir, int gp, int eleGID,
        const double* concentration, const double* temp, double* d_cauchyndir_dT,
        Core::LinAlg::Matrix<9, 1>* d2_cauchyndir_dF_dT) override;

    /**
     * \brief The derivatives of the strain energy function w.r.t. the principle invariants are
     * filled up to the third derivative, can be overloaded if temperature is given.
     *
     * \param prinv(in) : principle invariants of the tensor
     * \param gp(in) : Gauss point
     * \param eleGID(in) : global ID of the element
     * \param dPI(in,out) : 1st derivative of strain energy function w.r.t. principle invariants
     * \param ddPII(in,out) : 2nd derivative of strain energy function w.r.t. principle invariants
     * \param dddPIII(in,out) : 3rd derivative of strain energy function w.r.t. principle invariants
     * \param temp(in) : temperature
     */
    virtual void evaluate_cauchy_derivs(const Core::LinAlg::Matrix<3, 1>& prinv, int gp, int eleGID,
        Core::LinAlg::Matrix<3, 1>& dPI, Core::LinAlg::Matrix<6, 1>& ddPII,
        Core::LinAlg::Matrix<10, 1>& dddPIII, const double* temp);

    /**
     * \brief Evaluate the thermal dependency of the linearizations of the cauchy stress, to be
     *        overloaded
     *
     * \param prinv(in) : principle invariants of the tensor
     * \param ndt(in) : dot product of vectors n and t (\f[\mathbf{t} \cdot \mathbf{t}\f])
     * \param bdndt(in) : left Cauchy-Green tensor contracted using the vector n and t (\f[
              \mathbf{b} \cdot \mathbf{n} \cdot \mathbf{t} \f])
     * \param ibdndt(in) : inverse of left Cauchy-Green tensor contracted using the vector n and t
              (\f[ \mathbf{b}^{-1} \cdot \mathbf{n} \cdot \mathbf{t} \f])
     * \param temp(in) : temperature
     * \param DsntDT(out) : derivative of cauchy stress contracted with vectors n and t w.r.t.
              temperature (\f[\frac{\mathrm{d} \mathbf{\sigma} \cdot \mathbf{n} \cdot \mathbf{t} }{
              \mathrm{d} T }\f])
     * \param iFTV(in) : inverse transposed of the deformation gradient
     * \param DbdndtDFV(in) : derivative of bdndt w.r.t. deformation gradient (\f[\frac{ \mathrm{d}
              \mathbf{b} \cdot \mathbf{n} \cdot \mathbf{t}}{\mathrm{d} \mathbf{F}}\f])
     * \param DibdndtDFV(in) : derivative of ibdndt w.r.t. deformation gradient
              (\f[\frac{\mathrm{d} \mathbf{b}^{-1} \cdot \mathbf{n} \cdot \mathbf{t}}{\mathrm{d}
              \mathbf{F} }\f])
     * \param DI1DF(in) : derivative of first invariant w.r.t. deformation gradient
     * \param DI2DF(in) : derivative of second invariant w.r.t. deformation gradient
     * \param DI3DF(in) : derivative of third invariant w.r.t. deformation gradient
     * \param D2sntDFDT(out) : second derivative of cauchy stress contracted with vectors n and t
              w.r.t. temperature and deformation gradient (\f[ \frac{\mathrm{d}^2 \mathbf{\sigma}
              \cdot \mathbf{n} \cdot \mathbf{t}} {\mathrm{d} \mathbf{F} \mathrm{d} T } \f])
     */
    virtual void evaluate_cauchy_temp_deriv(const Core::LinAlg::Matrix<3, 1>& prinv,
        const double ndt, const double bdndt, const double ibdndt, const double* temp,
        double* DsntDT, const Core::LinAlg::Matrix<9, 1>& iFTV,
        const Core::LinAlg::Matrix<9, 1>& DbdndtDFV, const Core::LinAlg::Matrix<9, 1>& DibdndtDFV,
        const Core::LinAlg::Matrix<9, 1>& DI1DF, const Core::LinAlg::Matrix<9, 1>& DI2DF,
        const Core::LinAlg::Matrix<9, 1>& DI3DF, Core::LinAlg::Matrix<9, 1>* D2sntDFDT)
    {
    }

    /// setup
    void setup(int numgp, const Core::IO::InputParameterContainer& container) override;

    /*!
     * Post setup routine that is executed after the input. This class will forward the call to all
     * summands
     *
     * @param params Container for additional information
     * @param eleGID Global element id
     */
    void post_setup(Teuchos::ParameterList& params, int eleGID) override;

    /// update
    void update() override;

    /// setup patient-specific AAA stuff
    virtual void setup_aaa(Teuchos::ParameterList& params, int eleGID);

    /// return if anisotropic not split formulation
    virtual bool anisotropic_principal() const { return summandProperties_.anisoprinc; }

    /// return if anisotropic and split formulation
    virtual bool anisotropic_modified() const { return summandProperties_.anisomod; }

    /// get fiber vectors
    virtual void get_fiber_vecs(std::vector<Core::LinAlg::Matrix<3, 1>>& fibervecs) const;

    /// evaluate fiber directions from locsys and alignment angle, pull back
    virtual void evaluate_fiber_vecs(double newgamma,  ///< new angle
        const Core::LinAlg::Matrix<3, 3>& locsys,      ///< local coordinate system
        const Core::LinAlg::Matrix<3, 3>& defgrd       ///< deformation gradient
    );

    /// Return potential summand pointer for the given material type
    std::shared_ptr<const Mat::Elastic::Summand> get_pot_summand_ptr(
        const Core::Materials::MaterialType& materialtype) const;

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// Return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    /// Return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    /// Return whether the material requires the deformation gradient for its evaluation
    bool needs_defgrd() const override
    {
      // only the polyconvexity check needs the deformation gradient. Regular materials don't need
      // it
      return params_->polyconvex_ != 0;
    };

   protected:
    /// @name Flags to specify the elastic formulations (initialize with false)
    //@{
    /// Class indicating the formulation used by the summands
    SummandProperties summandProperties_;
    //@}

    /// my material parameters
    Mat::PAR::ElastHyper* params_;

    /// map to materials/potential summands
    std::vector<std::shared_ptr<Mat::Elastic::Summand>> potsum_;

    /// Holder of anisotropy
    Mat::Anisotropy anisotropy_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
