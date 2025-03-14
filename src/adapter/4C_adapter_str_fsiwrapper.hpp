// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_STR_FSIWRAPPER_HPP
#define FOUR_C_ADAPTER_STR_FSIWRAPPER_HPP

#include "4C_config.hpp"

#include "4C_adapter_str_wrapper.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Solid
{
  class MapExtractor;

  namespace ModelEvaluator
  {
    class PartitionedFSI;
  }
}  // namespace Solid


namespace Adapter
{
  class FSIStructureWrapper : public StructureWrapper
  {
   public:
    /// constructor
    explicit FSIStructureWrapper(std::shared_ptr<Structure> structure);

    /// communication object at the interface
    virtual std::shared_ptr<const Solid::MapExtractor> interface() const { return interface_; }

    /// switch structure field to block matrix in fsi simulations
    virtual void use_block_matrix();

    /// linear structure solve with just a interface load
    ///
    /// The very special solve done in steepest descent relaxation
    /// calculation (and matrix free Newton Krylov).
    ///
    /// \note Can only be called after a valid structural solve.
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> relaxation_solve(
        std::shared_ptr<Core::LinAlg::Vector<double>> iforce);

    /// @name Extract interface values

    /// extract interface displacements at \f$t_{n}\f$
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_dispn();

    /// extract interface displacements at \f$t_{n+1}\f$
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> extract_interface_dispnp();

    /// Predictor for interface displacements
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> predict_interface_dispnp();

    /// @name Apply interface forces

    /// apply interface forces to structural solver
    ///
    /// This prepares a new solve of the structural field within one time
    /// step. The middle values are newly created.
    ///
    /// \note This is not yet the most efficient implementation.
    virtual void apply_interface_forces(std::shared_ptr<Core::LinAlg::Vector<double>> iforce);

    /// remove as soon as new structure is fully usable ! todo
    /// only 3 nightly tests use this method:
    /// fsi_dc3D_part_ait_ga_ost_xwall
    /// fsi_ow3D_mtr_drt
    /// constr2D_fsi (newtonlinuzawa not implemented; but really needed ?)
    virtual void apply_interface_forces_temporary_deprecated(
        std::shared_ptr<Core::LinAlg::Vector<double>> iforce);

    /// rebuild FSI interface from structure side
    virtual void rebuild_interface();

    /// set pointer to model evaluator
    void set_model_evaluator_ptr(std::shared_ptr<Solid::ModelEvaluator::PartitionedFSI> me)
    {
      fsi_model_evaluator_ = me;
      return;
    }

   protected:
    /// the interface map setup for interface <-> full translation
    std::shared_ptr<Solid::MapExtractor> interface_;

    /// predictor type
    std::string predictor_;

    /// access the fsi model evaluator
    std::shared_ptr<Solid::ModelEvaluator::PartitionedFSI> fsi_model_evaluator();

   private:
    /// The structural model evaluator object.
    /// Your FSI algorithm calls methods in this adapter.
    /// If this method is related to the structural field,
    /// a corresponding method in the model evaluator may be
    /// called, if necessary.
    /// See e.g. \ref Adapter::FSIStructureWrapper::RelaxationSolve()
    std::shared_ptr<Solid::ModelEvaluator::PartitionedFSI> fsi_model_evaluator_;

  };  // class FSIStructureWrapper
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
