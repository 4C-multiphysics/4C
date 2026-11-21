// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_STRUCTPORO_REACTION_ECM_HPP
#define FOUR_C_MAT_STRUCTPORO_REACTION_ECM_HPP

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_structporo_reaction.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  // forward declaration
  class StructPoroReactionECM;

  namespace PAR
  {
    class StructPoroReactionECM : public PAR::StructPoroReaction
    {
      friend class Mat::StructPoroReactionECM;

     public:
      /// standard constructor
      StructPoroReactionECM(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// @name material parameters
      //@{

      /// density of collagen
      double densCollagen_;
      //@}
    };
    // class StructPoroReactionECM

  }  // namespace PAR

  class StructPoroReactionECMType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "StructPoroReactionECMType"; }

    static StructPoroReactionECMType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static StructPoroReactionECMType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for StructPoroReactionECM material
  ///
  /// This object exists (several times) at every element
  class StructPoroReactionECM : public StructPoroReaction
  {
   public:
    /// construct empty material object
    StructPoroReactionECM();

    /// construct the material object given material parameters
    explicit StructPoroReactionECM(Mat::PAR::StructPoroReactionECM* params);

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return StructPoroReactionECMType::instance().unique_par_object_id();
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
      return Core::Materials::m_structpororeactionECM;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<StructPoroReactionECM>(*this);
    }

    /// Initialize internal variables
    void setup(int numgp,  ///< number of Gauss points
        const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// evaluate chemical potential
    virtual void chem_potential(
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,  ///< (i) green lagrange strain
        const double porosity,                                        ///< (i) porosity
        const double press,  ///< (i) pressure at gauss point
        const double J,      ///< (i) determinant of jacobian at gauss point
        int EleID,           ///< (i) element GID
        double& pot,         ///< (o) chemical potential
        const int gp);

    /// Update of GP data (e.g., history variables)
    void update() override;

    //! @name Visualization methods

    /// Return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    /// Return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

   protected:
    void reaction(const double porosity, const double J,
        std::shared_ptr<std::vector<double>> scalars,
        const Teuchos::ParameterList& params) override;

    /// reference porosity at time step n
    double refporosity_old_;

    /// time derivative of reference porosity at time step n
    double refporositydot_old_;

    /// chemical potential
    std::vector<double> chempot_;

    /// initial chemical potential
    std::vector<double> chempot_init_;

    /// my material parameters
    Mat::PAR::StructPoroReactionECM* params_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
