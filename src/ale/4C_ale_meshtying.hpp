// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ALE_MESHTYING_HPP
#define FOUR_C_ALE_MESHTYING_HPP


#include "4C_config.hpp"

#include "4C_ale_input.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    class FluidEleParameter;
  }
}  // namespace Discret

namespace Mortar
{
  class Interface;
}

namespace Adapter
{
  class CouplingMortar;
}

namespace Core::LinAlg
{
  class Solver;
  class SparseMatrix;
  class MultiMapExtractor;
  class MapExtractor;
  class BlockSparseMatrixBase;
  class SparseOperator;
  class KrylovProjector;
}  // namespace Core::LinAlg

namespace ALE
{
  namespace Utils
  {
    class MapExtractor;
    class InterfaceSplitStrategy;
  }  // namespace Utils
  class Meshtying
  {
   public:
    //! Constructor
    Meshtying(std::shared_ptr<Core::FE::Discretization> dis,    ///> actual discretisation
        Core::LinAlg::Solver& solver,                           ///> solver
        int msht,                                               ///> meshting parameter list
        int nsd,                                                ///> number space dimensions
        const Utils::MapExtractor* surfacesplitter = nullptr);  ///> surface splitter

    virtual ~Meshtying() = default;

    //! Set up mesh-tying framework
    virtual std::shared_ptr<Core::LinAlg::SparseOperator> setup(
        std::vector<int> coupleddof, std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp);

    //! Use the split of the ale mesh tying for the sysmat
    std::shared_ptr<Core::LinAlg::SparseOperator> msht_split();

    //! Check weather Dirichlet BC are defined on the master
    void dirichlet_on_master(
        std::shared_ptr<const Core::LinAlg::Map> bmaps  ///> map of boundary condition
    );

    //! Prepare matrix and residual for meshtying
    void prepare_meshtying_system(std::shared_ptr<Core::LinAlg::SparseOperator>&
                                      sysmat,  ///> sysmat established by the element routine
        std::shared_ptr<Core::LinAlg::Vector<double>>&
            residual,  ///> residual established by the element routine
        std::shared_ptr<Core::LinAlg::Vector<double>>&
            dispnp);  ///> current ALE displacement vector

    //! Set the flag for multifield problems
    void is_multifield(const Core::LinAlg::MultiMapExtractor&
                           interface,  ///< interface maps for split of ale matrix
        bool ismultifield              ///< flag for multifield problems
    );

    //! Use the split of the ale mesh tying for the sysmat
    void msht_split(std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat);

    //! Use the split of the multifield problem for the sysmat
    void multifield_split(std::shared_ptr<Core::LinAlg::SparseOperator>& sysmat);

    //! Call the constructor and the setup of the mortar coupling adapter
    virtual void adapter_mortar(std::vector<int> coupleddof);

    //! Compare the size of the slave and master dof row map
    virtual void compare_num_dof();

    //! Get function for the slave and master dof row map
    virtual void dof_row_maps();

    //! Get function for the P matrix
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> get_mortar_matrix_p();

    //! Condensation operation for a block matrix (including ALE case):
    /// the original blocks (nn, nm, mn, mm) are manipulated directly;
    /// the remaining blocks (ns, ms, ss, sn, sm) are not touched at all,
    /// since finally a 2x2 block matrix is solved
    virtual void condensation_operation_block_matrix(
        std::shared_ptr<Core::LinAlg::SparseOperator>&
            sysmat,  ///> sysmat established by the element routine
        std::shared_ptr<Core::LinAlg::Vector<double>>&
            residual,  ///> residual established by the element routine
        std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp);  ///> current displacement vector

    //! Compute and update the increments of the slave node
    virtual void update_source_dof(std::shared_ptr<Core::LinAlg::Vector<double>>& inc,
        std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp);

    //! Recover method for Lagrange multipliers (do nothing in mesh tying case)
    virtual void recover(std::shared_ptr<Core::LinAlg::Vector<double>>& inc) {};

    //! Solve ALE mesh tying problem
    virtual int solve_meshtying(Core::LinAlg::Solver& solver,
        std::shared_ptr<Core::LinAlg::SparseOperator> sysmat,
        std::shared_ptr<Core::LinAlg::Vector<double>>& disi,
        std::shared_ptr<Core::LinAlg::Vector<double>> residual,
        std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp);

    //! Split vector and save parts in a std::vector<std::shared_ptr<Core::LinAlg::Vector<double>> >
    void split_vector(Core::LinAlg::Vector<double>& vector,  ///> vector to split
        std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>&
            splitvector);  ///> container for the split vector

   protected:
    //! discretisation
    std::shared_ptr<Core::FE::Discretization> discret_;

    Core::LinAlg::Solver& solver_;  // standard solver object

    //! dof row map of the complete system
    const Core::LinAlg::Map* dofrowmap_;

    //! slave dof rowmap
    std::shared_ptr<const Core::LinAlg::Map> gsdofrowmap_;

    //! master dof rowmap
    std::shared_ptr<const Core::LinAlg::Map> gmdofrowmap_;

    //! merged map for saddle point system and 2x2 block matrix
    std::shared_ptr<Core::LinAlg::Map> mergedmap_;

   private:
    //! Split vector and save parts in a std::vector<std::shared_ptr<Core::LinAlg::Vector<double>> >
    void split_vector_based_on3x3(
        Core::LinAlg::Vector<double>& orgvector,  ///> original vector based on 3x3 blockmatrix
        Core::LinAlg::Vector<double>& vectorbasedon2x2);  ///> split vector based on 2x2 blockmatrix

   private:
    //! meshting options
    /// 0: no_meshtying     -> no mesh-tying
    /// 1: yes_meshtying   -> condensation in a block matrix
    //    int msht_;    // Todo commented to avoid compiler warning, grill 04/17

    //! the processor ID from the communicator
    int myrank_;

    // interface splitter
    const Utils::MapExtractor* surfacesplitter_;

    //! dof row map of the complete system
    std::shared_ptr<Core::LinAlg::Map> problemrowmap_;

    //! dof rowmap of all nodes, which are not on the interface
    std::shared_ptr<Core::LinAlg::Map> gndofrowmap_;

    //! slave & master dof rowmap
    std::shared_ptr<Core::LinAlg::Map> gsource_target_dof_row_map_;

    //! vector containing time-depending values of the dirichlet condition
    /// valuesdc_ = (dispnp after applying DC) - (dispn)
    std::shared_ptr<Core::LinAlg::Vector<double>> valuesdc_;

    //! adapter to mortar framework
    std::shared_ptr<Coupling::Adapter::CouplingMortar> adaptermeshtying_;

    //! 2x2 (3x3) block matrix for solving condensed system (3x3 block matrix)
    std::shared_ptr<Core::LinAlg::SparseOperator> sysmatsolve_;
    std::shared_ptr<Core::LinAlg::Vector<double>> residual_;

    //! flag defining if Dirichlet  or Dirichlet-like boundary conditions are defined on the master
    //! side of the internal interface
    bool dconmaster_;

    //! flag for identifying first Newton iteration in each time step
    bool firstnonliniter_;

    //! number of space dimensions
    //    int nsd_;    // Todo commented to avoid compiler warning, grill 04/17

    //! interface maps for split of ale matrix in multifield simulation
    Core::LinAlg::MultiMapExtractor multifield_interface_;

    //! flag for multifield problems in multifield simulation
    bool is_multifield_;

  };  // end  class Meshtying
}  // end namespace ALE


FOUR_C_NAMESPACE_CLOSE

#endif
