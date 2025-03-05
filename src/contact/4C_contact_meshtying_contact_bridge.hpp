// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_MESHTYING_CONTACT_BRIDGE_HPP
#define FOUR_C_CONTACT_MESHTYING_CONTACT_BRIDGE_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::IO
{
  class DiscretizationWriter;
  class DiscretizationReader;
}  // namespace Core::IO

namespace Core::LinAlg
{
  class MapExtractor;
}  // namespace Core::LinAlg

namespace Mortar
{
  class ManagerBase;
  class StrategyBase;
}  // namespace Mortar

namespace CONTACT
{
  /*!
  \brief Bridge to enable unified access to contact and meshtying managers

  This bridge wraps contact and meshtying managers, such that the structure time integration does
  not have to distinguish between contact and meshtying operations, but has a single interface to
  both of them. The distinction between contact and meshtying operations is hidden in here.
  */
  class MeshtyingContactBridge
  {
   public:
    /*!
    \brief Constructor

    @param dis Structure discretization
    @param meshtyingConditions List of meshtying conditions as given in input file
    @param contactConditions List of contact conditions as given in input file
    @param timeIntegrationMidPoint Generalized mid-point of time integration scheme
    */
    MeshtyingContactBridge(Core::FE::Discretization& dis,
        std::vector<Core::Conditions::Condition*>& meshtyingConditions,
        std::vector<Core::Conditions::Condition*>& contactConditions,
        double timeIntegrationMidPoint);

    /*!
    \brief Destructor

    */
    virtual ~MeshtyingContactBridge() = default;

    //! @name Access methods

    /*!
    \brief Get Epetra communicator

    */
    MPI_Comm get_comm() const;

    /*!
    \brief Get contact manager

    */
    std::shared_ptr<Mortar::ManagerBase> contact_manager() const;

    /*!
    \brief Get meshtying manager

    */
    std::shared_ptr<Mortar::ManagerBase> mt_manager() const;

    /*!
    \brief Get strategy of meshtying/contact problem

    */
    Mortar::StrategyBase& get_strategy() const;

    /*!
    \brief return bool indicating if contact is defined

    */
    bool have_contact() const { return (cman_ != nullptr); }

    /*!
    \brief return bool indicating if meshtying is defined

    */
    bool have_meshtying() const { return (mtman_ != nullptr); }

    /*!
    \brief Write results for visualization for meshtying/contact problems

    This routine does some postprocessing (e.g. computing interface tractions) and then writes
    results to disk through the structure discretization's output writer \c output.

    \param[in] output Output writer of structure discretization to write results to disk
    */
    void postprocess_quantities(Core::IO::DiscretizationWriter& output);

    /*!
    \brief Write results for visualization separately for each meshtying/contact interface

    Call each interface, such that each interface can handle its own output of results.

    \param[in] outputParams Parameter list with stuff required by interfaces to write output
    */
    void postprocess_quantities_per_interface(std::shared_ptr<Teuchos::ParameterList> outputParams);

    /*!
    \brief read restart

    */
    void read_restart(Core::IO::DiscretizationReader& reader,
        std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<Core::LinAlg::Vector<double>> zero);
    /*!
    \brief recover lagr. mult. for contact/meshtying and slave displ for mesht.

    */
    void recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi);

    /*!
    \brief set state vector

    */
    void set_state(Core::LinAlg::Vector<double>& zeros);

    /*!
    \brief store dirichlet status

    */
    void store_dirichlet_status(std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps);

    /*!
    \brief update

    */
    void update(std::shared_ptr<Core::LinAlg::Vector<double>> dis);

    /*!
    \brief write restart

    @param[in] output Output writer to be used for writing outpu
    @param[in] forcedrestart Force to write restart data

    */
    void write_restart(Core::IO::DiscretizationWriter& output, bool forcedrestart = false);

   private:
    //! don't want cctor (= operator impossible anyway for abstract class)
    MeshtyingContactBridge(const MeshtyingContactBridge& old) = delete;

    //! Contact manager
    std::shared_ptr<Mortar::ManagerBase> cman_;

    //! Meshtying manager
    std::shared_ptr<Mortar::ManagerBase> mtman_;

  };  // class meshtying_contact_bridge
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
