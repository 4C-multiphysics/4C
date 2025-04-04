// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROELAST_BASE_HPP
#define FOUR_C_POROELAST_BASE_HPP


#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_adapter_field.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_volmortar.hpp"
#include "4C_fem_condition.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_poroelast_utils.hpp"
#include "4C_utils_exceptions.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class MapExtractor;
  class SparseMatrix;
  class BlockSparseMatrixBase;
}  // namespace Core::LinAlg

namespace Adapter
{
  class FluidPoro;
  class FPSIStructureWrapper;
  class MortarVolCoupl;
}  // namespace Adapter

namespace PoroElast
{
  class NoPenetrationConditionHandle;

  //! base class for porous media algorithm
  class PoroBase : public Adapter::AlgorithmBase, public Adapter::Field
  {
   public:
    explicit PoroBase(MPI_Comm comm, const Teuchos::ParameterList& timeparams,
        std::shared_ptr<Core::LinAlg::MapExtractor> porosity_splitter);

    //! read restart data
    void read_restart(const int step) override;

    /*!
     * @brief Perform all necessary tasks after setting up the PoroElast
     * object. Currently, this only calls the post_setup routine of the
     * structural field.
     */
    void post_setup();

    //! outer level time loop
    virtual void time_loop();

    //! initialize system
    virtual void setup_system() = 0;

    //! perform result tests
    virtual void test_results(MPI_Comm comm);

    //! build combined dirichlet map for the monolithic problem
    virtual void build_combined_dbc_map()
    {
      FOUR_C_THROW(
          "build_combined_dbc_map() not implemented in base class. must be implemented in sub "
          "classes.");
    }

    //! @name access methods

    //! access to structural field
    const std::shared_ptr<Adapter::FPSIStructureWrapper>& structure_field() { return structure_; }

    //! access to fluid field
    const std::shared_ptr<Adapter::FluidPoro>& fluid_field() { return fluid_; }

    //! composed system matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() override
    {
      FOUR_C_THROW("system_matrix() only available for monolithic schemes!");
      return nullptr;
    }

    //! block system matrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override
    {
      FOUR_C_THROW("block_system_matrix() only available for monolithic schemes!");
      return nullptr;
    }

    //! full monolithic dof row map
    std::shared_ptr<const Core::LinAlg::Map> dof_row_map() override
    {
      FOUR_C_THROW("dof_row_map() only available for monolithic schemes!");
      return nullptr;
    }

    //! dof row map of Structure field
    virtual std::shared_ptr<const Core::LinAlg::Map> dof_row_map_structure() = 0;

    //! dof row map of Fluid field
    virtual std::shared_ptr<const Core::LinAlg::Map> dof_row_map_fluid() = 0;

    //! extractor to communicate between full monolithic map and block maps
    virtual std::shared_ptr<const Core::LinAlg::MultiMapExtractor> extractor() const
    {
      FOUR_C_THROW("ExtractorPointer only available for monolithic schemes!");
      return nullptr;
    }

    //! unique map of all dofs that should be constrained with DBC
    virtual std::shared_ptr<const Core::LinAlg::Map> combined_dbc_map() const
    {
      FOUR_C_THROW("combined_dbc_map() only available for monolithic schemes!");
      return nullptr;
    }

    //! return rhs of poro problem
    std::shared_ptr<const Core::LinAlg::Vector<double>> rhs() override
    {
      FOUR_C_THROW("RHS() only available for monolithic schemes!");
      return nullptr;
    }

    //!@}

    //! update all fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    void update_state_incrementally(std::shared_ptr<const Core::LinAlg::Vector<double>>
            iterinc  //!< increment between iteration i and i+1
        ) override
    {
      FOUR_C_THROW("update_state_incrementally() only available for monolithic schemes!");
    }

    //! evaluate all fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>>
            iterinc  //!< increment between iteration i and i+1
        ) override
    {
      FOUR_C_THROW("evaluate() only available for monolithic schemes!");
    }

    //! evaluate all fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    virtual void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fx)
    {
      FOUR_C_THROW("evaluate(sx,fx) only available for monolithic schemes!");
    }

    //! evaluate all fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>>
                      iterinc,  //!< increment between iteration i and i+1
        bool firstiter) override
    {
      FOUR_C_THROW("evaluate() only available for monolithic schemes!");
    }

    //! evaluate all fields at x^n+1_i+1 with x^n+1_i+1 = x_n+1_i + iterinc
    virtual void evaluate(std::shared_ptr<const Core::LinAlg::Vector<double>> sx,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fx, bool firstiter)
    {
      FOUR_C_THROW("evaluate(sx,fx) only available for monolithic schemes!");
    }

    //! solve time step (depending on algorithm)
    virtual void solve() = 0;

    //! perform one time step (setup + solve + output)
    virtual void do_time_step() = 0;

    //! setup solver (for monolithic only)
    virtual bool setup_solver() { return false; }

    virtual void setup_rhs(bool firstcall = false)
    {
      FOUR_C_THROW("setup_rhs() only available for monolithic schemes!");
    }

    //! @name Time loop building blocks

    //! start a new time step
    void prepare_time_step() override;

    //! take current results for converged and save for next time step
    void update() override;

    //! calculate stresses, strains, energies
    void prepare_output(bool force_prepare_timestep) override;

    //! output
    void output(bool forced_writerestart = false) override;

    //!@}

    //! return whether the poro discretization contains submeshes (i.e. it is coupled with a pure
    //! solid)
    bool has_submeshes() { return submeshes_; }

    //! return coupling object
    Coupling::Adapter::Coupling& fluid_structure_coupling() { return *coupling_fluid_structure_; }

   protected:
    //! @name Transfer helpers

    //! field transform
    std::shared_ptr<Core::LinAlg::Vector<double>> structure_to_fluid_field(
        const Core::LinAlg::Vector<double>& iv);

    //!@}

    //! @name Transfer methods

    //! set fluid solution on structure
    void set_fluid_solution();

    //! set structure solution on fluid
    void set_struct_solution();

    //!@}

    //! Extractor used for constraint structure
    std::shared_ptr<Core::LinAlg::MapExtractor> cond_splitter_;

    //! true if the poroelast problem is only part of a larger problem (like e.g. in FPSI)
    bool is_part_of_multifield_problem_;

    //! Extractor used for poro structure interaction
    std::shared_ptr<Core::LinAlg::MapExtractor> psi_extractor_;

    //! helper class for no penetration condition
    std::shared_ptr<NoPenetrationConditionHandle> nopen_handle_;

    //! flag for partial integration condition of porous fluid continuity equation
    bool part_int_cond_;

    //! flag for pressure coupling condition
    bool pres_int_cond_;

    //! flag for additional porosity degree of freedom
    bool porosity_dof_;

    std::shared_ptr<Core::LinAlg::MapExtractor> porosity_splitter_;



    //! @name Volume Mortar stuff

    //! flag for matchinggrid
    const bool matchinggrid_;

    //! volume coupling (using mortar) adapter
    std::shared_ptr<Coupling::Adapter::MortarVolCoupl> volcoupl_;
    //!@}

    //! flag for old time integration
    const bool oldstructimint_;  // delete this once unused!

    //! flag indicating if nitsche contact is active
    bool nit_contact_;

   private:
    //! setup of everything, that is needed for the volumetric coupling
    void setup_coupling();

    //! add dof set of structure/fluid discretization to fluid/structure discretization
    void replace_dof_sets();

    //! check for special poro conditions and set bools
    void check_for_poro_conditions();

    //! flag indicating not fully overlapping fluid and structure discretization
    bool submeshes_;

    //! coupling of fluid and structure (whole field)
    std::shared_ptr<Coupling::Adapter::Coupling> coupling_fluid_structure_;

    //! @name Underlying fields

    //! underlying structure of the poro problem
    std::shared_ptr<Adapter::FPSIStructureWrapper> structure_;

    //! underlying fluid of the poro problem
    std::shared_ptr<Adapter::FluidPoro> fluid_;

    //!@}
  };

  //! helper class for no penetration condition
  class NoPenetrationConditionHandle
  {
   public:
    explicit NoPenetrationConditionHandle(std::vector<Core::Conditions::Condition*> nopencond)
        : cond_ids_(nullptr),
          cond_dofs_(nullptr),
          cond_rhs_(nullptr),
          nopencond_(std::move(nopencond)),
          nopenetration_(nullptr),
          has_cond_(false),
          fluid_fluid_constraint_matrix_(nullptr),
          fluid_structure_constraint_matrix_(nullptr),
          structure_vel_constraint_matrix_(nullptr)
    {
      if (nopencond_.size())
      {
        has_cond_ = true;
        cond_ids_ = std::make_shared<std::set<int>>();
      }
    }

    //! build map containing dofs with no penetration condition (fluid)
    void build_no_penetration_map(
        MPI_Comm comm, std::shared_ptr<const Core::LinAlg::Map> dofRowMap);

    //! apply rhs terms of no penetration condition to global rhs vector
    void apply_cond_rhs(Core::LinAlg::Vector<double>& iterinc, Core::LinAlg::Vector<double>& rhs);

    //! return no penetration map extractor
    std::shared_ptr<const Core::LinAlg::MapExtractor> extractor() { return nopenetration_; }

    //! return vector containing global IDs of dofs with no penetration condition
    std::shared_ptr<std::set<int>> cond_ids() { return cond_ids_; }

    //! return vector containing global IDs of dofs with no penetration condition
    std::shared_ptr<Core::LinAlg::Vector<double>> cond_vector() { return cond_dofs_; }

    //! check if a no penetration condition exists
    bool has_cond() { return has_cond_; }

    //! return condrhs
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs() { return cond_rhs_; }

    //! clear everything that is needed for coupling
    void clear(PoroElast::Coupltype coupltype = PoroElast::undefined);

    //! setup coupling matrixes and vectors
    void setup(const Core::LinAlg::Map& dofRowMap, const Core::LinAlg::Map* dofRowMapFluid);

    //! return constraint matrix, that fits to coupling type
    std::shared_ptr<Core::LinAlg::SparseMatrix> constraint_matrix(PoroElast::Coupltype coupltype);

    //! return constraint matrix for structure velocity coupling
    std::shared_ptr<Core::LinAlg::SparseMatrix> struct_vel_constraint_matrix(
        PoroElast::Coupltype coupltype);

   private:
    //! set containing global IDs of dofs with no penetration condition
    std::shared_ptr<std::set<int>> cond_ids_;

    //! vector marking dofs with no penetration condition
    std::shared_ptr<Core::LinAlg::Vector<double>> cond_dofs_;

    //! vector containing rhs terms from no penetration condition
    std::shared_ptr<Core::LinAlg::Vector<double>> cond_rhs_;

    //! vector containing no penetration - conditions
    std::vector<Core::Conditions::Condition*> nopencond_;

    //! Extractor used for no penetration condition
    std::shared_ptr<Core::LinAlg::MapExtractor> nopenetration_;

    //! flag indicating if a no penetration condition exists
    bool has_cond_;

    //! @name coupling matrices
    std::shared_ptr<Core::LinAlg::SparseMatrix> fluid_fluid_constraint_matrix_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> fluid_structure_constraint_matrix_;
    std::shared_ptr<Core::LinAlg::SparseMatrix> structure_vel_constraint_matrix_;
    //!@}
  };
}  // namespace PoroElast

FOUR_C_NAMESPACE_CLOSE

#endif
