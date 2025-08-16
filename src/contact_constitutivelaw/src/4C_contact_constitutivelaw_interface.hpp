// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_INTERFACE_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_INTERFACE_HPP

#include "4C_config.hpp"

#include "4C_contact_interface.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  namespace CONSTITUTIVELAW
  {
    class ConstitutiveLaw;
  }

  class ConstitutivelawInterface : public Interface
  {
   public:
    /*!
    \brief Standard constructor creating empty contact interface

    This initializes the employed shape function set for lagrangian multipliers
    to a specific setting. Throughout the evaluation process, this set will be employed
    for the field of lagrangian multipliers.

    \param idata_ptr (in): data container
    \param id (in): Unique interface id
    \param comm (in): A communicator object
    \param dim (in): Global problem dimension
    \param icontact (in): Global contact parameter list
    \param selfcontact (in): Flag for self contact status
    */
    ConstitutivelawInterface(const std::shared_ptr<Mortar::InterfaceDataContainer>& interfaceData,
        const int id, MPI_Comm comm, const int dim, const Teuchos::ParameterList& icontact,
        bool selfcontact, const int contactconstitutivelawid);

    /**
     * \brief Evaluate regularized normal forces at slave nodes
     *
     * Assemble gap-computed lagrange multipliers and nodal linlambda derivatives into nodal
     * quantities using the Macauley bracket
     *
     * When dealing with penalty methods, the lagrange multipliers are not independent variables
     * anymore. Instead, they can be computed in terms of the weighted gap and the penalty
     * parameter. This is done here so every node stores the correct lm and thus we integrate
     * smoothly into the overlaying algorithm.
     *
     * Additionally, we use the performed loop over all nodes to store the nodal derivlambda_j
     * matrix right there.
     *
     * As a result, the function notifies the calling routine if any negative gap was detected
     * and thus whether the interface is in contact or not. In consequence, after calling this
     * routine from within the penalty strategy object, the contact status is known at a global
     * level.
     *
     * Note: To be able to perform this computation, weighted gaps and normals have to be available
     * within every node! Since this computation is done via Interface::evaluate() in the integrator
     * class, these corresponding methods have to be called before AssembleMacauley()!
     *
     * \param[in/out] localisincontact true if at least one node is in contact
     * \param[in/out] localactivesetchange true if the active set changed
     *
     */
    void assemble_reg_normal_forces(bool& localisincontact, bool& localactivesetchange) override;

    /**
     * \brief Evaluate regularized normal forces at slave nodes
     *
     * Throws an error since frictional contact is not implemented, yet.
     */
    void assemble_reg_tangent_forces_penalty() override;

   private:
    /** \brief multi-scale constitutive law used for the contact containing information
     i.e. on the micro roughness
     */
    std::shared_ptr<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw> coconstlaw_;
  };  // class Interface
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
