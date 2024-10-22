// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_LINK_TRUSS_HPP
#define FOUR_C_BEAMINTERACTION_LINK_TRUSS_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_link_pinjointed.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SerialDenseMatrix;
}  // namespace Core::LinAlg

namespace Discret
{
  namespace ELEMENTS
  {
    class Truss3;
  }
}  // namespace Discret

namespace BEAMINTERACTION
{
  class BeamLinkTrussType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "BeamLinkTrussType"; };

    static BeamLinkTrussType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static BeamLinkTrussType instance_;
  };


  /*!
   \brief element for link between two 3D beam elements via a truss element
   */
  class BeamLinkTruss : public BeamLinkPinJointed
  {
   public:
    //! @name Friends
    // no friend classes defined
    //@}

    //! @name Constructors and destructors and related methods
    /*!
    \brief Standard Constructor
    */
    BeamLinkTruss();

    /*!
    \brief Copy Constructor

    Makes a deep copy of a Element

    */
    BeamLinkTruss(const BeamLinkTruss& old);



    //! Initialization [derived]
    void init(int id, const std::vector<std::pair<int, int>>& eleids,
        const std::vector<Core::LinAlg::Matrix<3, 1>>& initpos,
        const std::vector<Core::LinAlg::Matrix<3, 3>>& inittriad,
        Inpar::BEAMINTERACTION::CrosslinkerType linkertype, double timelinkwasset) override;

    //! Setup [derived]
    void setup(const int matnum) override;

    /*!
    \brief Return unique ParObject id [derived]

    Every class implementing ParObject needs a unique id defined at the
    top of parobject.H
    */
    int unique_par_object_id() const override
    {
      return BeamLinkTrussType::instance().unique_par_object_id();
    };

    /*!
    \brief Pack this class so it can be communicated [derived]

    \ref pack and \ref unpack are used to communicate this element

    */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
    \brief Unpack data from a char vector into this class [derived]

    \ref pack and \ref unpack are used to communicate this element

    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    /// return copy of this linking object
    Teuchos::RCP<BeamLink> clone() const override;

    //@}


    //! @name Access methods

    //! get internal linker energy
    double get_internal_energy() const override;

    //! get kinetic linker energy
    double get_kinetic_energy() const override;

    //! scale linker element reference length
    void scale_linker_reference_length(double scalefac) override;

    //! get force in first or second binding spot
    void get_binding_spot_force(
        int bspotid, Core::LinAlg::SerialDenseVector& bspotforce) const override;

    double get_current_linker_length() const override;

    //@}

    //! @name Public evaluation methods

    /*!
    \brief Evaluate forces and stiffness contribution [derived]
    */
    bool evaluate_force(Core::LinAlg::SerialDenseVector& forcevec1,
        Core::LinAlg::SerialDenseVector& forcevec2) override;

    /*!
    \brief Evaluate stiffness contribution [derived]
    */
    bool evaluate_stiff(Core::LinAlg::SerialDenseMatrix& stiffmat11,
        Core::LinAlg::SerialDenseMatrix& stiffmat12, Core::LinAlg::SerialDenseMatrix& stiffmat21,
        Core::LinAlg::SerialDenseMatrix& stiffmat22) override;

    /*!
    \brief Evaluate forces and stiffness contribution [derived]
    */
    bool evaluate_force_stiff(Core::LinAlg::SerialDenseVector& forcevec1,
        Core::LinAlg::SerialDenseVector& forcevec2, Core::LinAlg::SerialDenseMatrix& stiffmat11,
        Core::LinAlg::SerialDenseMatrix& stiffmat12, Core::LinAlg::SerialDenseMatrix& stiffmat21,
        Core::LinAlg::SerialDenseMatrix& stiffmat22) override;

    /*
    \brief Update position and triad of both connection sites (a.k.a. binding spots)
    */
    void reset_state(std::vector<Core::LinAlg::Matrix<3, 1>>& bspotpos,
        std::vector<Core::LinAlg::Matrix<3, 3>>& bspottriad) override;


    //@}

   private:
    //! @name Private evaluation methods

    /*!
    \brief Fill absolute nodal positions and nodal quaternions with current values
    */
    void fill_state_variables_for_element_evaluation(
        Core::LinAlg::Matrix<6, 1, double>& absolute_nodal_positions) const;

    void get_disp_for_element_evaluation(
        std::map<std::string, std::vector<double>>& ele_state) const;

    //@}

   private:
    //! @name member variables

    //! new connecting element
    Teuchos::RCP<Discret::ELEMENTS::Truss3> linkele_;

    //! the following variables are for output purposes only (no need to pack or unpack)
    std::vector<Core::LinAlg::SerialDenseVector> bspotforces_;

    //@}
  };

}  // namespace BEAMINTERACTION

FOUR_C_NAMESPACE_CLOSE

#endif
