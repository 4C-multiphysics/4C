// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_STRUCTPORO_REACTION_HPP
#define FOUR_C_MAT_STRUCTPORO_REACTION_HPP

#include "4C_config.hpp"

#include "4C_mat_structporo.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  // forward declaration
  class StructPoroReaction;

  namespace PAR
  {
    class StructPoroReaction : public PAR::StructPoro
    {
      friend class Mat::StructPoroReaction;

     public:
      /// standard constructor
      StructPoroReaction(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// @name material parameters
      //@{

      int dofIDReacScalar_;
      //@}

    };  // class StructPoroReaction

  }  // namespace PAR

  class StructPoroReactionType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "StructPoroReactionType"; }

    static StructPoroReactionType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static StructPoroReactionType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for StructPoro material
  ///
  /// This object exists (several times) at every element
  class StructPoroReaction : public StructPoro
  {
   public:
    /// construct empty material object
    StructPoroReaction();

    /// construct the material object given material parameters
    explicit StructPoroReaction(Mat::PAR::StructPoroReaction* params);

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return StructPoroReactionType::instance().unique_par_object_id();
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
      return Core::Materials::m_structpororeaction;
    }

    /// return initial porosity
    double ref_porosity_time_deriv() const override { return dphiDphiref_ * refporositydot_; }

    /// compute current porosity and save it
    void compute_porosity(const Teuchos::ParameterList& params,  ///< (i) element parameter list
        double press,                                            ///< (i) pressure at gauss point
        double J,          ///< (i) determinant of jacobian at gauss point
        int gp,            ///< (i) number of current gauss point
        double& porosity,  ///< (o) porosity at gauss point
        double* dphi_dp,   ///< (o) first derivative of porosity w.r.t. pressure at gauss point
        double* dphi_dJ,   ///< (o) first derivative of porosity w.r.t. jacobian at gauss point
        double*
            dphi_dJdp,  ///< (o) derivative of porosity w.r.t. pressure and jacobian at gauss point
        double* dphi_dJJ,  ///< (o) second derivative of porosity w.r.t. jacobian at gauss point
        double* dphi_dpp,  ///< (o) second derivative of porosity w.r.t. pressure at gauss point
        bool save = true) override;

    //! evaluate constitutive relation for porosity and compute derivatives
    void constitutive_derivatives(const Teuchos::ParameterList& params,  ///< (i) parameter list
        double press,        ///< (i) fluid pressure at gauss point
        double J,            ///< (i) Jacobian determinant at gauss point
        double porosity,     ///< (i) porosity at gauss point
        double* dW_dp,       ///< (o) derivative of potential w.r.t. pressure
        double* dW_dphi,     ///< (o) derivative of potential w.r.t. porosity
        double* dW_dJ,       ///< (o) derivative of potential w.r.t. jacobian
        double* dW_dphiref,  ///< (o) derivative of potential w.r.t. reference porosity
        double* W            ///< (o) inner potential
        ) override;

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<StructPoroReaction>(*this);
    }

    /// Initialize internal variables
    void setup(int numgp,  ///< number of Gauss points
        const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// return reference porosity average
    double ref_porosity_av() const;

    //! @name Evaluation methods

    /// evaluate material law
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    //@}

    //! @name Visualization methods

    /// Return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    /// Return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    //@}

   protected:
    virtual void reaction(const double porosity, const double J,
        std::shared_ptr<std::vector<double>> scalars, const Teuchos::ParameterList& params);

    /// my material parameters
    Mat::PAR::StructPoroReaction* params_;

    /// reference porosity
    double refporosity_;

    /// derivative of porosity w.r.t. reference porosity
    double dphiDphiref_;

    /// time derivative of reference porosity
    double refporositydot_;
  };

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
