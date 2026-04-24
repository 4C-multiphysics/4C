// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_XFLUID_STATE_HPP
#define FOUR_C_FLUID_XFLUID_STATE_HPP

#include "4C_config.hpp"

#include "4C_cut_input.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_xfem_input.hpp"

#include <map>
#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Cut
{
  class CutWizard;
}

namespace Core::LinAlg
{
  class SparseMatrix;
  class MultiMapExtractor;
  class MapExtractor;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
}

namespace XFEM
{
  class ConditionManager;
  class DiscretizationXFEM;
  class XFEMDofSet;
}  // namespace XFEM

namespace FLD
{
  namespace Utils
  {
    class KSPMapExtractor;
  }


  /**
   * Container class for the state vectors and maps of the intersected background
   * fluid - tied to a specific intersection state (interface position,
   * independent from the fact, in which form the interface is given (boundary mesh or level-set
   * field)).
   */
  class XFluidState
  {
   public:
    /*!
     * \brief Container class for coupling state matrices and vectors for a certain coupling object
     */
    class CouplingState
    {
     public:
      //! ctor
      CouplingState()
          : C_sx_(nullptr),
            C_xs_(nullptr),
            C_ss_(nullptr),
            rhC_s_(nullptr),
            rhC_s_col_(nullptr),
            is_active_(false)
      {
      }

      //! ctor  Initialize coupling matrices with fluid sysmat and fluid rhs vector
      CouplingState(const std::shared_ptr<Core::LinAlg::SparseMatrix>& C_sx,
          const std::shared_ptr<Core::LinAlg::SparseMatrix>& C_xs,
          const std::shared_ptr<Core::LinAlg::SparseMatrix>& C_ss,
          const std::shared_ptr<Core::LinAlg::Vector<double>>& rhC_s,
          const std::shared_ptr<Core::LinAlg::Vector<double>>& rhC_s_col)
          : C_sx_(C_sx),  // this rcp constructor using const references copies also the ownership
                          // and increases the specific weak/strong reference counter
            C_xs_(C_xs),
            C_ss_(C_ss),
            rhC_s_(rhC_s),
            rhC_s_col_(rhC_s_col),
            is_active_(true)
      {
      }

      //! ctor  Initialize coupling matrices
      CouplingState(const std::shared_ptr<const Core::LinAlg::Map>& xfluiddofrowmap,
          const std::shared_ptr<Core::FE::Discretization>& slavediscret_mat,
          const std::shared_ptr<Core::FE::Discretization>& slavediscret_rhs);

      //! zero coupling matrices and rhs vectors
      void zero_coupling_matrices_and_rhs();

      //! complete coupling matrices and rhs vectors
      void complete_coupling_matrices_and_rhs(
          const Core::LinAlg::Map& xfluiddofrowmap, const Core::LinAlg::Map& slavedofrowmap);

      //! destroy the coupling objects and it's content
      void destroy(bool throw_exception = true);

      //! @name coupling matrices x: xfluid, s: coupling slave (structure, ALE-fluid, xfluid-element
      //! with other active dofset, etc.)
      //@{
      std::shared_ptr<Core::LinAlg::SparseMatrix> C_sx_;         ///< slave - xfluid coupling block
      std::shared_ptr<Core::LinAlg::SparseMatrix> C_xs_;         ///< xfluid - slave coupling block
      std::shared_ptr<Core::LinAlg::SparseMatrix> C_ss_;         ///< slave - slave coupling block
      std::shared_ptr<Core::LinAlg::Vector<double>> rhC_s_;      ///< slave rhs block
      std::shared_ptr<Core::LinAlg::Vector<double>> rhC_s_col_;  ///< slave rhs block
      //@}

      bool is_active_;
    };

    /*!
     ctor for one-sided problems
     @param xfluiddofrowmap dof-rowmap of intersected fluid
     */
    explicit XFluidState(const std::shared_ptr<XFEM::ConditionManager>& condition_manager,
        const std::shared_ptr<Cut::CutWizard>& wizard,
        const std::shared_ptr<XFEM::XFEMDofSet>& dofset,
        const std::shared_ptr<const Core::LinAlg::Map>& xfluiddofrowmap,
        const std::shared_ptr<const Core::LinAlg::Map>& xfluiddofcolmap);

    /// dtor
    virtual ~XFluidState() = default;
    /// setup map extractors for dirichlet maps & velocity/pressure maps
    void setup_map_extractors(
        const std::shared_ptr<Core::FE::Discretization>& xfluiddiscret, const double& time);

    /// zero system matrix and related rhs vectors
    virtual void zero_system_matrix_and_rhs();

    /// zero all coupling matrices and rhs vectors for all coupling objects
    virtual void zero_coupling_matrices_and_rhs();

    /// Complete coupling matrices and rhs vectors
    virtual void complete_coupling_matrices_and_rhs();

    /// destroy the stored objects
    virtual bool destroy();

    /// update the coordinates of the cut boundary cells
    void update_boundary_cell_coords();


    //! @name Accessors
    //@{

    /// access to the cut wizard
    std::shared_ptr<Cut::CutWizard> wizard() const
    {
      if (wizard_ == nullptr) FOUR_C_THROW("Cut wizard is uninitialized!");
      return wizard_;
    }


    //! access to the xfem-dofset
    std::shared_ptr<XFEM::XFEMDofSet> dof_set() { return dofset_; }

    virtual std::shared_ptr<Core::LinAlg::MapExtractor> dbc_map_extractor() { return dbcmaps_; }

    virtual std::shared_ptr<Core::LinAlg::MapExtractor> vel_pres_splitter()
    {
      return velpressplitter_;
    }

    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> system_matrix() { return sysmat_; }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>>& residual() { return residual_; }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>>& zeros() { return zeros_; }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>>& inc_vel() { return incvel_; }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>>& velnp() { return velnp_; }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>>& veln() { return veln_; }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>>& accnp() { return accnp_; }
    virtual std::shared_ptr<Core::LinAlg::Vector<double>>& accn() { return accn_; }
    //@}

    /// dof-rowmap of intersected fluid
    std::shared_ptr<const Core::LinAlg::Map> xfluiddofrowmap_;

    /// dof-colmap of intersected fluid
    std::shared_ptr<const Core::LinAlg::Map> xfluiddofcolmap_;

    /// system matrix
    std::shared_ptr<Core::LinAlg::SparseMatrix> sysmat_;

    /// a vector of zeros to be used to enforce zero dirichlet boundary conditions
    std::shared_ptr<Core::LinAlg::Vector<double>> zeros_;

    /// the vector containing body and surface forces
    std::shared_ptr<Core::LinAlg::Vector<double>> neumann_loads_;

    /// (standard) residual vector (rhs for the incremental form),
    std::shared_ptr<Core::LinAlg::Vector<double>> residual_;

    /// (standard) residual vector (rhs for the incremental form) wrt col map,
    std::shared_ptr<Core::LinAlg::Vector<double>> residual_col_;

    /// true (rescaled) residual vector without zeros at dirichlet positions
    std::shared_ptr<Core::LinAlg::Vector<double>> trueresidual_;

    /// nonlinear iteration increment vector
    std::shared_ptr<Core::LinAlg::Vector<double>> incvel_;

    //! @name acceleration/(scalar time derivative) at time n+1, n and n+alpha_M/(n+alpha_M/n)
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> accnp_;
    std::shared_ptr<Core::LinAlg::Vector<double>> accn_;
    std::shared_ptr<Core::LinAlg::Vector<double>> accam_;
    //@}

    //! @name velocity and pressure at time n+1, n, n-1 and n+alpha_F
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> velnp_;
    std::shared_ptr<Core::LinAlg::Vector<double>> veln_;
    std::shared_ptr<Core::LinAlg::Vector<double>> velnm_;
    std::shared_ptr<Core::LinAlg::Vector<double>> velaf_;
    //@}

    //! @name scalar at time n+alpha_F/n+1 and n+alpha_M/n
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> scaaf_;
    std::shared_ptr<Core::LinAlg::Vector<double>> scaam_;
    //@}

    //! @name displacemets at time n+1, n and n-1 (if we have an XFEM-ALE-fluid)
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> dispnp_;
    //  std::shared_ptr<Core::LinAlg::Vector<double>>         dispn_;
    //  std::shared_ptr<Core::LinAlg::Vector<double>>         dispnm_;
    //@}

    /// grid velocity (if we have an XFEM-ALE-fluid) (set from the adapter!)
    std::shared_ptr<Core::LinAlg::Vector<double>> gridvnp_;

    /// histvector --- a linear combination of velnm, veln (BDF)
    ///                or veln, accn (One-Step-Theta)
    std::shared_ptr<Core::LinAlg::Vector<double>> hist_;

    //! @name map extractors
    //@{
    /// maps for extracting Dirichlet and free DOF sets
    std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps_;

    /// velocity/pressure map extractor, used for convergence check
    std::shared_ptr<Core::LinAlg::MapExtractor> velpressplitter_;

    //@}


    //! @name coupling matrices for each coupling object (=key); x: xfluid, s: coupling slave
    //! (structure, ALE-fluid, xfluid-element with other active dofset, etc.)
    //@{
    std::map<int, std::shared_ptr<CouplingState>> coup_state_;
    //@}

   protected:
    /// initialize all state members based on the xfluid dof-rowmap
    void init_state_vectors();

    /// initialize the system matrix of the intersected fluid
    void init_system_matrix();

    /// initialize coupling matrices and rhs vectors for all coupling objects
    void init_coupling_matrices_and_rhs();

    /// Complete coupling matrices and rhs vectors
    void complete_coupling_matrices_and_rhs(const Core::LinAlg::Map& fluiddofrowmap);


   public:
    /*!
     \brief initialize ALE state vectors
     @param dispnp and grivnp vectors w.r.t initial full dofrowmap
     */
    void init_ale_state_vectors(XFEM::DiscretizationXFEM& xdiscret,
        const Core::LinAlg::Vector<double>& dispnp_initmap,
        const Core::LinAlg::Vector<double>& gridvnp_initmap);

    /// XFEM dofset
    std::shared_ptr<XFEM::XFEMDofSet> dofset_;

    /// cut wizard
    std::shared_ptr<Cut::CutWizard> wizard_;

    /// condition manager
    std::shared_ptr<XFEM::ConditionManager> condition_manager_;
  };

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
