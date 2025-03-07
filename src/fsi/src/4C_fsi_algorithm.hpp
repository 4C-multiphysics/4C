// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_ALGORITHM_HPP
#define FOUR_C_FSI_ALGORITHM_HPP

#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_adapter_fld_fluid_ale.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


// forward declarations
namespace Adapter
{
  class FSIStructureWrapper;
  class StructureBaseAlgorithmNew;
}  // namespace Adapter


namespace FSI
{
  /// FSI algorithm base
  /*!

    Base class of FSI algorithms with generalized fluid field. There can (and
    will) be different subclasses that implement different coupling schemes.

    \note The generalized fluid field hides any ale or xfem handling of the
    variable fluid domain. This is the appropriate base class if direct access
    to ale or xfem is not required. If a coupling algorithm needs an ale
    field, MonolithicBase is the better choice for a base class.

    \warning The order of calling the three BaseAlgorithm-constructors (that
    is the order in which we list the base classes) is important here! In the
    constructors control file entries are written. And these entries define
    the order in which the filters handle the Discretizations, which in turn
    defines the dof number ordering of the Discretizations... Don't get
    confused. Just always list structure, fluid, ale. In that order.

   */
  class Algorithm : public Adapter::AlgorithmBase
  {
   public:
    explicit Algorithm(MPI_Comm comm);


    /// setup this object
    virtual void setup();

    /*!
     * @brief Perform all necessary tasks after setting up the FSI
     * algorithm. Currently, this only calls the post_setup routine of the
     * structure field.
     */
    void post_setup();

    /// access to structure field
    const std::shared_ptr<Adapter::FSIStructureWrapper>& structure_field() { return structure_; }

    /// access to fluid field
    const std::shared_ptr<Adapter::FluidMovingBoundary>& mb_fluid_field() { return fluid_; }

    /// read restart data
    void read_restart(int step) override;

   protected:
    //! @name Time loop building blocks

    /// start a new time step
    void prepare_time_step() override;

    /// take current results for converged and save for next time step
    void update() override;

    /// calculate stresses, strains, energies
    virtual void prepare_output(bool force_prepare);

    /// write output
    void output() override;

    //@}

    //! @name Transfer helpers
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> struct_to_fluid(
        std::shared_ptr<Core::LinAlg::Vector<double>> iv);
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> fluid_to_struct(
        std::shared_ptr<Core::LinAlg::Vector<double>> iv);
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> struct_to_fluid(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const;
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> fluid_to_struct(
        std::shared_ptr<const Core::LinAlg::Vector<double>> iv) const;
    //@}

    /// return the structure fluid coupling object
    Coupling::Adapter::Coupling& structure_fluid_coupling();

    /// return const version of structure fluid coupling object
    const Coupling::Adapter::Coupling& structure_fluid_coupling() const;

   protected:
    /// underlying structure of the FSI problem
    std::shared_ptr<Adapter::FSIStructureWrapper> structure_;

    /// underlying fluid of the FSI problem
    std::shared_ptr<Adapter::FluidMovingBoundary> fluid_;

    /// RCP pointer to the base algorithm of the structure
    std::shared_ptr<Adapter::StructureBaseAlgorithmNew> adapterbase_ptr_;

    /// use deprecated old structural time integration. todo Has to be removed !
    bool use_old_structure_;

   private:
    /// coupling of structure and fluid at the interface
    std::shared_ptr<Coupling::Adapter::Coupling> coupsf_;
  };
}  // namespace FSI

FOUR_C_NAMESPACE_CLOSE

#endif
