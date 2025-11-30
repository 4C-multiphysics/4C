// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_UTILS_HPP
#define FOUR_C_FLUID_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_linalg_blocksparsematrix.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::DOFSets
{
  class DofSet;
  class DofSetInterface;
}  // namespace Core::DOFSets

namespace Core::LinAlg
{
  class MultiMapExtractor;
  class MapExtractor;
}  // namespace Core::LinAlg

namespace FLD
{
  namespace Utils
  {
    /// velocity-pressure block matrix split strategy
    class VelPressSplitStrategy
    {
     public:
      /// construct with a block matrix base
      explicit VelPressSplitStrategy(Core::LinAlg::BlockSparseMatrixBase& mat)
          : mat_(mat),
            matrix00_(mat_.matrix(0, 0)),
            matrix01_(mat_.matrix(0, 1)),
            matrix10_(mat_.matrix(1, 0)),
            matrix11_(mat_.matrix(1, 1)),
            numdim_(-1),
            numdofpernode_(-1)
      {
      }

      /// find row block to a given row gid
      int row_block(int lrow, int rgid)
      {
        if ((lrow % numdofpernode_) < numdim_) return 0;
        return 1;
      }

      /// find column block to a given column gid
      int col_block(int rblock, int lcol, int cgid)
      {
        if ((lcol % numdofpernode_) < numdim_) return 0;
        return 1;
      }

      /// assemble into the given block
      void assemble(int eid, int myrank, const std::vector<int>& lmstride,
          const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
          const std::vector<int>& lmrowowner, const std::vector<int>& lmcol)
      {
        const int lrowdim = (int)lmrow.size();
        const int lcoldim = (int)lmcol.size();

        if (mat_.filled())
        {
          // We use the maps of the matrix to gain fast access to the LID's.
          // Assembling with SumIntoMyValues based on LID's is two times faster
          // than inserting single values based on the global row and column ids.

          // There is the case of nodes without dofs (XFEM).
          // If no row dofs are present on this proc, their is nothing to assemble.
          // However, the subsequent check for coldofs (in DEBUG mode) would incorrectly fail.
          bool doit = false;
          for (int lrow = 0; lrow < lrowdim; ++lrow)
            if (lmrowowner[lrow] == myrank)
            {
              doit = true;  // This proc owns at least one row of interest
              break;
            }
          if (!doit) return;

          // get the maps
          const Core::LinAlg::Map& colmap00 = Core::LinAlg::Map(mat_.matrix(0, 0).col_map());
          const Core::LinAlg::Map& colmap01 = Core::LinAlg::Map(mat_.matrix(0, 1).col_map());
          const Core::LinAlg::Map& colmap10 = Core::LinAlg::Map(mat_.matrix(1, 0).col_map());
          const Core::LinAlg::Map& colmap11 = Core::LinAlg::Map(mat_.matrix(1, 1).col_map());
          const Core::LinAlg::Map& rowmap00 = Core::LinAlg::Map(mat_.matrix(0, 0).row_map());
          const Core::LinAlg::Map& rowmap01 = Core::LinAlg::Map(mat_.matrix(0, 1).row_map());
          const Core::LinAlg::Map& rowmap10 = Core::LinAlg::Map(mat_.matrix(1, 0).row_map());
          const Core::LinAlg::Map& rowmap11 = Core::LinAlg::Map(mat_.matrix(1, 1).row_map());

          // prepare vectors for holding column local ids and the values to be assembled
          const int nnode = lcoldim / numdofpernode_;
          std::vector<double> values0(numdim_ * nnode);
          std::vector<double> values1(nnode);
          std::vector<int> localcol00(numdim_ * nnode);
          std::vector<int> localcol01(nnode);
          std::vector<int> localcol10(numdim_ * nnode);
          std::vector<int> localcol11(nnode);

          // fill vectors with the LID's
          int nodespassed = 0;
          for (int lcol = 0; lcol < lcoldim; ++lcol)
          {
            const int cgid = lmcol[lcol];
            const int rest = (lcol % numdofpernode_);
            if (rest < numdim_)
            {
              const int pos = nodespassed * numdim_ + rest;
              localcol00[pos] = (colmap00.lid(cgid));
              localcol10[pos] = (colmap10.lid(cgid));
            }
            else
            {
              const int pos = nodespassed;
              localcol01[pos] = (colmap01.lid(cgid));
              localcol11[pos] = (colmap11.lid(cgid));
              nodespassed++;
            }
          }

          // loop rows of local matrix
          for (int lrow = 0; lrow < lrowdim; ++lrow)
          {
            // check ownership of row
            if (lmrowowner[lrow] != myrank) continue;

            const int rgid = lmrow[lrow];
            int rlid0;
            int rlid1;
            int rowblock = row_block(lrow, rgid);
            if (rowblock == 0)
            {
              rlid0 = rowmap00.lid(rgid);
              rlid1 = rowmap01.lid(rgid);
            }
            else
            {
              rlid0 = rowmap10.lid(rgid);
              rlid1 = rowmap11.lid(rgid);
            }
#ifdef FOUR_C_ENABLE_ASSERTIONS
            if (rlid0 < 0) FOUR_C_THROW("Sparse matrix A does not have global row {}", rgid);
            if (rlid1 < 0) FOUR_C_THROW("Sparse matrix A does not have global row {}", rgid);
#endif
            // separate the values of the current row
            nodespassed = 0;
            for (int lcol = 0; lcol < lcoldim; ++lcol)
            {
              double val = Aele(lrow, lcol);
              const int rest = lcol % numdofpernode_;
              if (rest < numdim_)
              {
                int pos = nodespassed * numdim_ + rest;
                values0[pos] = val;
              }
              else
              {
                values1[nodespassed] = val;
                nodespassed++;
              }
            }

            // now assemble
            if (rowblock == 0)
            {  // rowblock 0
              matrix00_.sum_into_my_values(
                  rlid0, nnode * numdim_, values0.data(), localcol00.data());
              matrix01_.sum_into_my_values(rlid1, nnode, values1.data(), localcol01.data());
            }
            else
            {  // rowblock 1
              matrix10_.sum_into_my_values(
                  rlid0, nnode * numdim_, values0.data(), localcol10.data());
              matrix11_.sum_into_my_values(rlid1, nnode, values1.data(), localcol11.data());
            }
          }  // for (int lrow=0; lrow<ldim; ++lrow)
        }
        else
        {
          // the maps of the blockmatrix are not yet available; use global ids instead

          // loop rows of local matrix
          for (int lrow = 0; lrow < lrowdim; ++lrow)
          {
            // check ownership of row
            if (lmrowowner[lrow] != myrank) continue;

            int rgid = lmrow[lrow];
            int rblock = row_block(lrow, rgid);

            for (int lcol = 0; lcol < lcoldim; ++lcol)
            {
              double val = Aele(lrow, lcol);
              int cgid = lmcol[lcol];
              int cblock = col_block(rblock, lcol, cgid);

              Core::LinAlg::SparseMatrix& matrix = mat_.matrix(rblock, cblock);
              matrix.assemble(val, rgid, cgid);
            }
          }
        }
      }

      /// assemble into the given block
      void assemble(double val, int rgid, int cgid)
      {
        int rblock = row_block(0, rgid);
        int cblock = col_block(rblock, 0, cgid);
        Core::LinAlg::SparseMatrix& matrix = mat_.matrix(rblock, cblock);
        matrix.assemble(val, rgid, cgid);
      }

      /// assemble the remaining ghost entries
      void complete() {}

      /// set number of velocity dofs
      void set_numdim(int numdim)
      {
        numdim_ = numdim;
        numdofpernode_ = numdim + 1;
      }

     private:
      /// my block matrix base
      Core::LinAlg::BlockSparseMatrixBase& mat_;

      // the four sub-matrices of the whole matrix
      Core::LinAlg::SparseMatrix& matrix00_;
      Core::LinAlg::SparseMatrix& matrix01_;
      Core::LinAlg::SparseMatrix& matrix10_;
      Core::LinAlg::SparseMatrix& matrix11_;

      /// number of velocity dofs
      int numdim_;

      /// number of dofs per node (= numdim_ +1)
      int numdofpernode_;
    };


    /// (FSI) interface block matrix split strategy
    class InterfaceSplitStrategy : public Core::LinAlg::DefaultBlockMatrixStrategy
    {
     public:
      explicit InterfaceSplitStrategy(Core::LinAlg::BlockSparseMatrixBase& mat)
          : Core::LinAlg::DefaultBlockMatrixStrategy(mat)
      {
      }

      /// assemble into the given block
      void assemble(int eid, int myrank, const std::vector<int>& lmstride,
          const Core::LinAlg::SerialDenseMatrix& Aele, const std::vector<int>& lmrow,
          const std::vector<int>& lmrowowner, const std::vector<int>& lmcol)
      {
        if (condelements_->find(eid) != condelements_->end())
        {
          // if we have an element with conditioned nodes, we have to do the
          // default assembling
          Core::LinAlg::DefaultBlockMatrixStrategy::assemble(
              eid, myrank, lmstride, Aele, lmrow, lmrowowner, lmcol);
        }
        else
        {
          // if there are no conditioned nodes we can simply assemble to the
          // internal matrix
          Core::LinAlg::SparseMatrix& matrix = mat().matrix(0, 0);
          matrix.assemble(eid, lmstride, Aele, lmrow, lmrowowner, lmcol);
        }
      }

      void assemble(double val, int rgid, int cgid)
      {
        // forward single value assembling
        Core::LinAlg::DefaultBlockMatrixStrategy::assemble(val, rgid, cgid);
      }

      void set_cond_elements(std::shared_ptr<std::set<int>> condelements)
      {
        condelements_ = condelements;
      }

     private:
      std::shared_ptr<std::set<int>> condelements_;
    };


    /// Stress manager manages everything to do with stresses and wallshearstresses
    class StressManager
    {
     public:
      /// constructor
      StressManager(std::shared_ptr<Core::FE::Discretization> discret,
          std::shared_ptr<Core::LinAlg::Vector<double>> dispnp, const bool alefluid,
          const int numdim);

      /// initialize smoothing of stresses
      void init_aggr(std::shared_ptr<Core::LinAlg::SparseOperator> sysmat);

      /// update and return WSS vector
      std::shared_ptr<Core::LinAlg::Vector<double>> get_wall_shear_stresses(
          const Core::LinAlg::Vector<double>& trueresidual, const double dt);

      /// return WSS vector (without updating the mean wss vector)
      std::shared_ptr<Core::LinAlg::Vector<double>> get_pre_calc_wall_shear_stresses(
          const Core::LinAlg::Vector<double>& trueresidual);

      /// return WSS vector always without aggregation, even if scale separation matrix exists
      std::shared_ptr<Core::LinAlg::Vector<double>> get_wall_shear_stresses_wo_agg(
          const Core::LinAlg::Vector<double>& trueresidual);

      /// update and return stress vector
      std::shared_ptr<Core::LinAlg::Vector<double>> get_stresses(
          const Core::LinAlg::Vector<double>& trueresidual, const double dt);

      /// return stress vector (without updating the mean stress vector)
      std::shared_ptr<Core::LinAlg::Vector<double>> get_pre_calc_stresses(
          const Core::LinAlg::Vector<double>& trueresidual);

      /// return stress vector always without aggregation, even if scale separation matrix exists
      std::shared_ptr<Core::LinAlg::Vector<double>> get_stresses_wo_agg(
          const Core::LinAlg::Vector<double>& trueresidual);

      /// return flag if StressManager has already been initialized
      bool is_init() { return isinit_; };

     private:
      /// return stress vector
      std::shared_ptr<Core::LinAlg::Vector<double>> calc_stresses(
          const Core::LinAlg::Vector<double>& trueresidual);

      /// integrate shape functions at nodes marked by condition
      std::shared_ptr<Core::LinAlg::Vector<double>> integrate_interface_shape(std::string condname);

      /// calculate WSS based on residual
      std::shared_ptr<Core::LinAlg::Vector<double>> calc_wall_shear_stresses(
          std::shared_ptr<Core::LinAlg::Vector<double>> stresses);

      /// smooth stress/wss via ML-aggregation
      std::shared_ptr<Core::LinAlg::Vector<double>> aggreagte_stresses(
          Core::LinAlg::Vector<double>& wss);

      /// time average stresses
      std::shared_ptr<Core::LinAlg::Vector<double>> time_average_stresses(
          const Core::LinAlg::Vector<double>& stresses, double dt);

      /// time average wss
      std::shared_ptr<Core::LinAlg::Vector<double>> time_average_wss(
          const Core::LinAlg::Vector<double>& wss, double dt);

      /// Calculate Aggregation Matrix
      void calc_sep_enr(std::shared_ptr<Core::LinAlg::SparseOperator> sysmat);

      /// fluid discretization
      const std::shared_ptr<Core::FE::Discretization> discret_;

      /// displacement at time \f$t^{n+1}\f$
      const std::shared_ptr<Core::LinAlg::Vector<double>> dispnp_;

      /// do we move the fluid mesh and calculate the fluid on this moving mesh?
      const bool alefluid_;

      /// number of spatial dimensions
      const int numdim_;

      /// filtering matrix for wall shear stress
      std::shared_ptr<Core::LinAlg::SparseMatrix> sep_enr_;

      /// wss calculation type
      const Inpar::FLUID::WSSType wss_type_;

      /// weighted sum of all prior stresses
      std::shared_ptr<Core::LinAlg::Vector<double>> sum_stresses_;

      /// weighted sum of all prior wss
      std::shared_ptr<Core::LinAlg::Vector<double>> sum_wss_;

      /// time the stresses are averaged for
      double sum_dt_stresses_;

      /// time the wss are averaged for
      double sum_dt_wss_;

      bool isinit_;
    };


    void setup_fluid_fluid_vel_pres_split(const Core::FE::Discretization& fluiddis, int ndim,
        const Core::FE::Discretization& alefluiddis, Core::LinAlg::MapExtractor& extractor,
        std::shared_ptr<Core::LinAlg::Map> fullmap);

    /**
     * \brief Calculate lift and drag forces, and angular momenta.
     *
     * This routine computes lift and drag forces based on the right-hand side
     * true-residual values of the corresponding nodes. The contribution of the end node
     * of a line is entirely added to the current lift and drag (L&D) forces.
     *
     * The basic idea:
     *  - A map is created: `map<label, std::set<Core::Nodes::Node*>>`,
     *    where each L&D ID corresponds to a set of nodes.
     *  - Nodal forces from all nodes within a set are added to compute the total L&D forces.
     *
     * \note Angular moments obtained from lift and drag forces currently refer to the
     *       initial configuration, meaning they are calculated using the coordinates X
     *       of a node in its initial state, not its current position.
     *

     *
     * \param dis          Fluid discretization, including node distribution and
     *                     boundary conditions.
     * \param trueresidual Vector of nodal forces (true-residual values).
     * \param dispnp       Solution vector containing velocities and pressures.
     * \param ndim         Number of spatial dimensions (e.g., 2D or 3D).
     * \param liftdragvals Output parameter that stores the computed lift and drag values in
     *                     an array.
     * \param alefluid     Boolean flag indicating if the Arbitrary Lagrangian-Eulerian (ALE)
     *                     formulation is used.
     */
    void lift_drag(const std::shared_ptr<const Core::FE::Discretization> dis,
        const Core::LinAlg::Vector<double>& trueresidual,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> dispnp, const int ndim,
        std::shared_ptr<std::map<int, std::vector<double>>>& liftdragvals, bool alefluid);


    /**
     * \brief Process 0 writes transient lift and drag values to files (one file per label).
     *
     * This function writes the computed lift and drag values to files, with one file being
     * generated per label. It is typically called by process 0.
     *

     *
     * \param time         The current real time at which the values are being written.
     * \param step         The current time step.
     * \param liftdragvals The computed lift and drag values, stored in a map where each entry
     *                     corresponds to a label and its associated lift and drag data.
     */
    void write_lift_drag_to_file(
        const double time, const int step, const std::map<int, std::vector<double>>& liftdragvals);

    /**
     * \brief Integrate mass flow over surfaces for each condition ID.
     *
     * This function computes the flow rate through a boundary condition for each
     * condition ID. The result is returned as a map where the flow rate is associated
     * with each condition ID. The flow rate's sign indicates net inflow (positive)
     * or outflow (negative).
     *

     *
     * \param dis         The discretization, including node distribution and conditions.
     * \param velnp       Solution vector containing velocities (and pressure).
     * \param condstring  Name of the condition (e.g., "LineFlowRate" or "SurfaceFlowRate").
     * \param physicaltype The physical type of flow, defined by the fluid's properties.
     *
     * \return A map where each condition ID corresponds to the computed flow rate.
     *         The sign of the flow rate indicates net inflow (positive) or outflow (negative).
     */
    std::map<int, double> compute_flow_rates(Core::FE::Discretization& dis,
        Core::LinAlg::Vector<double>& velnp, const std::string& condstring,
        const Inpar::FLUID::PhysicalType physicaltype);

    /**
     * \param dis          The discretization, including node distribution and conditions.
     * \param velnp        Solution vector containing velocities (and pressure).
     * \param gridvel      Solution vector containing grid velocities for ALE formulation.
     * \param dispnp       Solution vector containing mesh displacements for ALE formulation.
     * \param condstring   Name of the condition (e.g., "LineFlowRate" or "SurfaceFlowRate").
     * \param physicaltype The physical type of flow, defined by the fluid's properties.
     *
     * \return A map where each condition ID corresponds to the computed flow rate.
     *         The sign of the flow rate indicates net inflow (positive) or outflow (negative).
     */
    std::map<int, double> compute_flow_rates(Core::FE::Discretization& dis,
        Core::LinAlg::Vector<double>& velnp,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& gridvel,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp, const std::string& condstring,
        const Inpar::FLUID::PhysicalType physicaltype);

    /**
     * \param dis          The discretization, including node distribution and conditions.
     * \param velnp        Solution vector containing velocities (and pressure).
     * \param gridvel      Solution vector containing grid velocities for ALE formulation.
     * \param dispnp       Solution vector containing mesh displacements for ALE formulation.
     * \param physicaltype The physical type of flow, defined by the fluid's properties.
     *
     * \return A map where each condition ID corresponds to the computed volume.
     */
    std::map<int, double> compute_volume(Core::FE::Discretization& dis,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& velnp,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& gridvel,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& dispnp,
        const Inpar::FLUID::PhysicalType physicaltype);

    /*!
     * \brief proc 0 writes the flow rate values for each condition ID to a file
     *

     */
    void write_doubles_to_file(const double time, const int step, const std::map<int, double>& data,
        const std::string& name);


    void write_volume_to_file(
        const double time, const int step, const std::map<int, double>& flowrates);

    /*!
    \brief Project gradient and store vector in param list

    */
    void project_gradient_and_set_param(Core::FE::Discretization& discret,
        Teuchos::ParameterList& eleparams, const Core::LinAlg::Vector<double>& vel,
        const std::string paraname, bool alefluid);

    /*!
    \brief Project velocity gradient, depends on time integrator used

    */
    std::shared_ptr<Core::LinAlg::MultiVector<double>> project_gradient(
        Core::FE::Discretization& discret, const Core::LinAlg::Vector<double>& vel, bool alefluid);

  }  // namespace Utils
}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
