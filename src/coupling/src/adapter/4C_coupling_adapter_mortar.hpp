// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_COUPLING_ADAPTER_MORTAR_HPP
#define FOUR_C_COUPLING_ADAPTER_MORTAR_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter_base.hpp"
#include "4C_fem_general_shape_function_type.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_map.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SparseMatrix;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::Nodes
{
  class Node;
}

namespace Core::Utils
{
  class FunctionManager;
}

namespace Mortar
{
  class Interface;
  class IntCell;
}  // namespace Mortar

/// Couple non-matching interface meshes using mortar method
namespace Coupling::Adapter
{
  /*!
  This is a generic class used to couple any non-matching meshes
  (or more general: discretizations) at interfaces. The current
  applications in 4C encompass FSI coupling algorithms (i.e. to
  interpolate between fluid and structure fields at the interface)
  and fluid mesh tying algorithms (i.e. to couple non-matching
  Eulerian fluid meshes). All the hard work is actually done by
  the Mortar::Interface class (thus we use the mortar method).

  The major part of this code is the setup() method that gets the
  non-matching interface meshes on input, initializes the mortar
  interface and computes the so-called coupling matrices \f$D\f$ and \f$M\f$.

  The actual coupling methods master_to_slave() and slave_to_master()
  just evaluate one simple equation each, i.e. primal variables
  are projected from master to slave side via \f$D^{-1} M\f$ when
  calling master_to_slave(), and dual variables are projected from
  slave to master side via \f$M^T D^{-T}\f$ when calling slave_to_master().

  Whenever you want to add a new problem class, check whether you
  can re-use one of the already existing setup() methods. If not,
  feel free to write your own tailored setup() method.
  */
  class CouplingMortar : public CouplingBase
  {
   public:
    /// Construct the CouplingMortar with basic parameters.
    CouplingMortar(int spatial_dimension, Teuchos::ParameterList mortar_coupling_params,
        Teuchos::ParameterList contact_dynamic_params,
        Core::FE::ShapeFunctionType shape_function_type);

    /*! setup the machinery (generalized version)
     *
     *  \note
     *  - Master and slave discretizations are identical in case of sliding ALE or fluid/scatra
     * meshtying
     *  - ALE discretization is nullptr in case of sliding ALE or fluid/scatra meshtying
     */
    void setup(
        const std::shared_ptr<Core::FE::Discretization>& masterdis,  ///< master discretization
        const std::shared_ptr<Core::FE::Discretization>& slavedis,   ///< slave discretization
        const std::shared_ptr<Core::FE::Discretization>& aledis,     ///< ALE discretization
        const std::vector<int>& coupleddof,  ///< vector defining coupled degrees of freedom
        const std::string& couplingcond,     ///< string for coupling condition
        MPI_Comm comm,                       ///< communicator
        const Core::Utils::FunctionManager& function_manager,  ///< function manager
        const Teuchos::ParameterList& binning_params,          ///< parameters for binning strategy
        const std::map<std::string, std::shared_ptr<Core::FE::Discretization>>& discretization_map,
        std::shared_ptr<Core::IO::OutputControl> output_control,
        Core::FE::ShapeFunctionType spatial_approximation_type,
        const bool slavewithale = false,  ///< flag defining if slave is ALE
        const bool slidingale = false,    ///< flag indicating sliding ALE case
        const int nds_master = 0,         ///< master dofset number
        const int nds_slave = 0           ///< slave dofset number
    );

    /*! setup the machinery (generalized version)
     *
     *  \note
     *  - Master and slave discretizations are identical in case of sliding ALE or fluid/scatra
     * meshtying
     *  - ALE discretization is nullptr in case of sliding ALE or fluid/scatra meshtying
     */
    void setup_interface(
        const std::shared_ptr<Core::FE::Discretization>& masterdis,  ///< master discretization
        const std::shared_ptr<Core::FE::Discretization>& slavedis,   ///< slave discretization
        const std::vector<int>& coupleddof,  ///< vector defining coupled degrees of freedom
        const std::map<int, Core::Nodes::Node*>&
            mastergnodes,  ///< master nodes, including ghosted nodes
        const std::map<int, Core::Nodes::Node*>&
            slavegnodes,  ///< slave nodes, including ghosted nodes
        const std::map<int, std::shared_ptr<Core::Elements::Element>>&
            masterelements,  ///< master elements
        const std::map<int, std::shared_ptr<Core::Elements::Element>>&
            slaveelements,                             ///< slave elements
        MPI_Comm comm,                                 ///< communicator
        const Teuchos::ParameterList& binning_params,  ///< parameters for binning strategy
        const std::map<std::string, std::shared_ptr<Core::FE::Discretization>>& discretization_map,
        std::shared_ptr<Core::IO::OutputControl> output_control,
        Core::FE::ShapeFunctionType spatial_approximation_type,
        const bool slavewithale = false,  ///< flag defining if slave is ALE
        const bool slidingale = false,    ///< flag indicating sliding ALE case
        const int nds_master = 0,         ///< master dofset number
        const int nds_slave = 0           ///< slave dofset number
    );

    /// create integration cells
    virtual void evaluate_geometry(std::vector<std::shared_ptr<Mortar::IntCell>>&
            intcells  //!< vector of mortar integration cells
    );

    /// Compute mortar matrices by using mortar interface using reference configuration
    virtual void evaluate();

    /// Compute mortar matrices
    virtual void evaluate(std::shared_ptr<Core::LinAlg::Vector<double>> idisp  ///< [in] ??
    );

    /// Compute mortar matrices (case of transferring same dofs on two different meshes)
    virtual void evaluate(std::shared_ptr<Core::LinAlg::Vector<double>> idispma,  ///< [in] ??
        std::shared_ptr<Core::LinAlg::Vector<double>> idispsl                     ///< [in] ??
    );

    //! Compute mortar matrices after performing a mesh correction step
    virtual void evaluate_with_mesh_relocation(
        std::shared_ptr<Core::FE::Discretization> slavedis,    ///< slave discretization
        std::shared_ptr<Core::FE::Discretization> aledis,      ///< ALE discretization
        std::shared_ptr<Core::LinAlg::Vector<double>>& idisp,  ///< ALE displacements
        MPI_Comm comm,                                         ///< communicator
        bool slavewithale                                      ///< flag defining if slave is ALE
    );

    //! Get the mortar interface itself
    std::shared_ptr<Mortar::Interface> interface() const { return interface_; }

    //! Access to slave side mortar matrix \f$D\f$
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> get_mortar_matrix_d() const
    {
      if (D_ == nullptr) FOUR_C_THROW("D Matrix is null pointer!");
      return D_;
    };

    //! Access to inverse of slave side mortar matrix \f$D^{-1}\f$
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> get_mortar_matrix_dinv() const
    {
      if (Dinv_ == nullptr) FOUR_C_THROW("DInv Matrix is null pointer!");
      return Dinv_;
    };

    //! Access to master side mortar matrix \f$M\f$
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> get_mortar_matrix_m() const
    {
      if (M_ == nullptr) FOUR_C_THROW("M Matrix is null pointer!");
      return M_;
    };

    //! Access to mortar projection operator \f$P\f$
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> get_mortar_matrix_p() const
    {
      if (P_ == nullptr) FOUR_C_THROW("P Matrix is null pointer!");
      return P_;
    };

    /// @name Conversion between master and slave
    //@{


    /*! \brief Transfer a dof vector from master to slave (const version)
     *
     *  \return Slave vector
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> master_to_slave(
        const Core::LinAlg::Vector<double>& mv  ///< [in] master vector (to be transferred)
    ) const override;

    /*! \brief Transfer a dof vector from master to slave (const version)
     *
     *  \return Slave vector
     */
    std::shared_ptr<Core::LinAlg::MultiVector<double>> master_to_slave(
        const Core::LinAlg::MultiVector<double>& mv  ///< [in] master vector (to be transferred)
    ) const override;


    /*! \brief Transfer a dof vector from slave to master (const version)
     *
     *  \return Master vector
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> slave_to_master(
        const Core::LinAlg::Vector<double>& sv  ///< [in] slave vector (to be transferred)
    ) const override;

    /*! \brief Transfer a dof vector from slave to master (const version)
     *
     *  \return Master vector
     */
    std::shared_ptr<Core::LinAlg::MultiVector<double>> slave_to_master(
        const Core::LinAlg::MultiVector<double>& sv  ///< [in] slave vector (to be transferred)
    ) const override;

    /// transfer a dof vector from master to slave
    void master_to_slave(
        const Core::LinAlg::MultiVector<double>& mv,  ///< [in] master vector (to be transferred)
        Core::LinAlg::MultiVector<double>& sv         ///< [out] slave vector (containing result)
    ) const override;

    /// transfer a dof vector from slave to master
    void slave_to_master(
        const Core::LinAlg::MultiVector<double>& sv,  ///< [in] slave vector (to be transferred)
        Core::LinAlg::MultiVector<double>& mv         ///< [out] master vector (containing result)
    ) const override;

    //@}

    /** \name Coupled maps */
    //@{

    /// Get the interface dof row map of the master side
    std::shared_ptr<const Core::LinAlg::Map> master_dof_map() const override
    {
      return pmasterdofrowmap_;
    }

    /// Get the interface dof row map of the slave side
    std::shared_ptr<const Core::LinAlg::Map> slave_dof_map() const override
    {
      return pslavedofrowmap_;
    }

    //@}

    /** \name Condensation methods */
    //@{

    /// do condensation of Lagrange multiplier and slave-sided dofs
    void mortar_condensation(
        std::shared_ptr<Core::LinAlg::SparseMatrix>& k,  ///< in:  tangent matrix w/o condensation
                                                         ///< out: tangent matrix w/  condensation
        Core::LinAlg::Vector<double>& rhs                ///< in:  rhs vector     w/o condensation
                                                         ///< out: rhs vector     w/  condensation
    ) const;

    /// recover slave-sided dofs
    void mortar_recover(Core::LinAlg::SparseMatrix& k,  ///< in: tangent matrix
        Core::LinAlg::Vector<double>& inc  ///< in:  solution vector     w/o condensation
                                           ///< out: solution vector     w/  condensation
    ) const;

    //@}

   protected:
    /// Create mortar projection operator \f$P=D{^1}M\f$
    virtual void create_p();

    /*! \brief Check if slave dofs have Dirichlet constraints
     *
     *  Slave DOFs are not allowed to carry Dirichlet boundary conditions to
     *  avoid over-constraining the problem, cf. [1]
     *
     *  <h3> References </h3>
     *  [1] Puso, M and Laursen, TA: Mesh tying on curved interfaces in 3D,
     *      Engineering Computation, 20:305-319 (2003)
     */
    void check_slave_dirichlet_overlap(const std::shared_ptr<Core::FE::Discretization>& slavedis,
        MPI_Comm comm, const Core::Utils::FunctionManager& function_manager);

    /// back transformation to initial parallel distribution
    void matrix_row_col_transform();

    /// check setup call
    const bool& is_setup() const { return issetup_; };

    /// check init and setup call
    virtual void check_setup() const
    {
      if (!is_setup()) FOUR_C_THROW("Call setup() first!");
    }

   private:
    /// perform mesh relocation
    void mesh_relocation(Core::FE::Discretization& slavedis,  ///< [in] Slave discretization
        std::shared_ptr<Core::FE::Discretization> aledis,     ///< [in] ALE discretization
        std::shared_ptr<const Core::LinAlg::Map>
            masterdofrowmap,  ///< [in] DOF row map of master discretization
        std::shared_ptr<const Core::LinAlg::Map>
            slavedofrowmap,  ///< [in] DOF row map of slave discretization
        std::shared_ptr<Core::LinAlg::Vector<double>>& idisp,  ///< [in] ALE displacements
        MPI_Comm comm,                                         ///< [in] Communicator
        bool slavewithale  ///< [in] Flag defining if slave is ALE
    );

   protected:
    //! Spatial dimension of problem
    int spatial_dimension_;

    //! Parameters for mortar coupling
    Teuchos::ParameterList mortar_coupling_params_;
    //! Parameters for contact dynamic
    Teuchos::ParameterList contact_dynamic_params_;

    //! Shape functions used in coupled discretizations
    Core::FE::ShapeFunctionType shape_function_type_;

    /// Check for setup
    bool issetup_;

    /// Interface
    std::shared_ptr<Mortar::Interface> interface_;

    /// Map of master row dofs (after parallel redist.)
    std::shared_ptr<const Core::LinAlg::Map> masterdofrowmap_;

    /// Map of slave row dofs  (after parallel redist.)
    std::shared_ptr<const Core::LinAlg::Map> slavedofrowmap_;

    /// Map of master row dofs (before parallel redist.)
    std::shared_ptr<const Core::LinAlg::Map> pmasterdofrowmap_;

    /// Map of slave row dofs  (before parallel redist.)
    std::shared_ptr<const Core::LinAlg::Map> pslavedofrowmap_;

    /// Slave side mortar matrix \f$D\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> D_;

    /// Inverse \f$D^{-1}\f$ of slave side mortar matrix \f$D\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> Dinv_;

    /// Master side mortar matrix \f$M\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> M_;

    /// Mortar projection operator \f$P=D^{-1}M\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> P_;
  };
}  // namespace Coupling::Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
