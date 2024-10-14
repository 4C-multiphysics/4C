/*--------------------------------------------------------------------------*/
/*! \file

\brief Mesh tying for ale problems

\level 2

*/
/*--------------------------------------------------------------------------*/
#ifndef FOUR_C_ALE_MESHTYING_HPP
#define FOUR_C_ALE_MESHTYING_HPP


#include "4C_config.hpp"

#include "4C_coupling_adapter_mortar.hpp"
#include "4C_inpar_ale.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_vector.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace ELEMENTS
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
    Meshtying(Teuchos::RCP<Core::FE::Discretization> dis,       ///> actual discretisation
        Core::LinAlg::Solver& solver,                           ///> solver
        int msht,                                               ///> meshting parameter list
        int nsd,                                                ///> number space dimensions
        const Utils::MapExtractor* surfacesplitter = nullptr);  ///> surface splitter

    virtual ~Meshtying() = default;

    //! Set up mesh-tying framework
    virtual Teuchos::RCP<Core::LinAlg::SparseOperator> setup(
        std::vector<int> coupleddof, Teuchos::RCP<Core::LinAlg::Vector<double>>& dispnp);

    //! Use the split of the ale mesh tying for the sysmat
    Teuchos::RCP<Core::LinAlg::SparseOperator> msht_split();

    //! Check weather Dirichlet BC are defined on the master
    void dirichlet_on_master(Teuchos::RCP<const Epetra_Map> bmaps  ///> map of boundary condition
    );

    //! Prepare matrix and residual for meshtying
    void prepare_meshtying_system(Teuchos::RCP<Core::LinAlg::SparseOperator>&
                                      sysmat,  ///> sysmat established by the element routine
        Teuchos::RCP<Core::LinAlg::Vector<double>>&
            residual,  ///> residual established by the element routine
        Teuchos::RCP<Core::LinAlg::Vector<double>>& dispnp);  ///> current ALE displacement vector

    //! Set the flag for multifield problems
    void is_multifield(const Core::LinAlg::MultiMapExtractor&
                           interface,  ///< interface maps for split of ale matrix
        bool ismultifield              ///< flag for multifield problems
    );

    //! Use the split of the ale mesh tying for the sysmat
    void msht_split(Teuchos::RCP<Core::LinAlg::SparseOperator>& sysmat);

    //! Use the split of the multifield problem for the sysmat
    void multifield_split(Teuchos::RCP<Core::LinAlg::SparseOperator>& sysmat);

    //! Call the constructor and the setup of the mortar coupling adapter
    virtual void adapter_mortar(std::vector<int> coupleddof);

    //! Compare the size of the slave and master dof row map
    virtual void compare_num_dof();

    //! Get function for the slave and master dof row map
    virtual void dof_row_maps();

    //! Get function for the P matrix
    virtual Teuchos::RCP<Core::LinAlg::SparseMatrix> get_mortar_matrix_p();

    //! Condensation operation for a block matrix (including ALE case):
    /// the original blocks (nn, nm, mn, mm) are manipulated directly;
    /// the remaining blocks (ns, ms, ss, sn, sm) are not touched at all,
    /// since finally a 2x2 block matrix is solved
    virtual void condensation_operation_block_matrix(
        Teuchos::RCP<Core::LinAlg::SparseOperator>&
            sysmat,  ///> sysmat established by the element routine
        Teuchos::RCP<Core::LinAlg::Vector<double>>&
            residual,  ///> residual established by the element routine
        Teuchos::RCP<Core::LinAlg::Vector<double>>& dispnp);  ///> current displacement vector

    //! Compute and update the increments of the slave node
    virtual void update_slave_dof(Teuchos::RCP<Core::LinAlg::Vector<double>>& inc,
        Teuchos::RCP<Core::LinAlg::Vector<double>>& dispnp);

    //! Recover method for Lagrange multipliers (do nothing in mesh tying case)
    virtual void recover(Teuchos::RCP<Core::LinAlg::Vector<double>>& inc){};

    //! Solve ALE mesh tying problem
    virtual int solve_meshtying(Core::LinAlg::Solver& solver,
        Teuchos::RCP<Core::LinAlg::SparseOperator> sysmat,
        Teuchos::RCP<Core::LinAlg::Vector<double>>& disi,
        Teuchos::RCP<Core::LinAlg::Vector<double>> residual,
        Teuchos::RCP<Core::LinAlg::Vector<double>>& dispnp);

    //! Split vector and save parts in a std::vector<Teuchos::RCP<Core::LinAlg::Vector<double>> >
    void split_vector(Core::LinAlg::Vector<double>& vector,  ///> vector to split
        std::vector<Teuchos::RCP<Core::LinAlg::Vector<double>>>&
            splitvector);  ///> container for the split vector

   protected:
    //! discretisation
    Teuchos::RCP<Core::FE::Discretization> discret_;

    Core::LinAlg::Solver& solver_;  // standard solver object

    //! dof row map of the complete system
    const Epetra_Map* dofrowmap_;

    //! slave dof rowmap
    Teuchos::RCP<const Epetra_Map> gsdofrowmap_;

    //! master dof rowmap
    Teuchos::RCP<const Epetra_Map> gmdofrowmap_;

    //! merged map for saddle point system and 2x2 block matrix
    Teuchos::RCP<Epetra_Map> mergedmap_;

   private:
    //! Split vector and save parts in a std::vector<Teuchos::RCP<Core::LinAlg::Vector<double>> >
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
    Teuchos::RCP<Epetra_Map> problemrowmap_;

    //! dof rowmap of all nodes, which are not on the interface
    Teuchos::RCP<Epetra_Map> gndofrowmap_;

    //! slave & master dof rowmap
    Teuchos::RCP<Epetra_Map> gsmdofrowmap_;

    //! vector containing time-depending values of the dirichlet condition
    /// valuesdc_ = (dispnp after applying DC) - (dispn)
    Teuchos::RCP<Core::LinAlg::Vector<double>> valuesdc_;

    //! adapter to mortar framework
    Teuchos::RCP<Coupling::Adapter::CouplingMortar> adaptermeshtying_;

    //! 2x2 (3x3) block matrix for solving condensed system (3x3 block matrix)
    Teuchos::RCP<Core::LinAlg::SparseOperator> sysmatsolve_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> residual_;

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
