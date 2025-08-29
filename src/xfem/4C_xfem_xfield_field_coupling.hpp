// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_XFEM_XFIELD_FIELD_COUPLING_HPP
#define FOUR_C_XFEM_XFIELD_FIELD_COUPLING_HPP


#include "4C_config.hpp"

#include "4C_coupling_adapter.hpp"
#include "4C_utils_exceptions.hpp"
// due to enum use ...
#include "4C_xfem_multi_field_mapextractor.hpp"

FOUR_C_NAMESPACE_OPEN

namespace XFEM
{
  namespace XFieldField
  {
    class Coupling : public ::FourC::Coupling::Adapter::Coupling
    {
     public:
      /** \brief enumerator specifying which interface discretization (slave
       *  or master) has the minimum number of DoF's at each node
       *
       *  min_dof_slave  : The slave discretization has the minimum number
       *                   of DoF's at each interface node.
       *
       *  min_dof_master : The master discretization has the minimum number
       *                   of DoF's at each interface node.
       *
       *  min_dof_unknown: The discretization with the minimum number of DoF's
       *                   per node is unknown or cannot be identified, since it
       *                   changes from node to node. This case needs extra
       *                   communication effort and is currently unsupported.
       *
       */
      enum MinDofDiscretization
      {
        min_dof_slave,
        min_dof_master,
        min_dof_unknown
      };

     public:
      /// constructor
      Coupling();

      /// initialize class member variables
      void init(const enum MinDofDiscretization& min_dof_dis);

      /** \name Conversion between master and slave
       *
       *  In contradiction to the base class versions, nodal distributed as well
       *  as dof distributed vectors can be transferred. */
      //@{
      /// There are different versions to satisfy all needs. The basic
      /// idea is the same for all of them.

      /** \brief transfer a nodel/dof vector from master to slave
       *
       *  \param mv       (in) : master vector (to be transferred)
       *  \param map_type (in) : map type of the master vector */
      inline std::shared_ptr<Core::LinAlg::Vector<double>> master_to_slave(
          const std::shared_ptr<Core::LinAlg::Vector<double>>& mv,
          const enum XFEM::MapType& map_type) const
      {
        return master_to_slave(
            *std::const_pointer_cast<const Core::LinAlg::Vector<double>>(mv), map_type);
      }

      /** \brief transfer a nodal/dof vector from slave to master
       *
       *  \param sv       (in) : slave vector (to be transferred)
       *  \param map_type (in) : map type of the slave vector */
      inline std::shared_ptr<Core::LinAlg::Vector<double>> slave_to_master(
          std::shared_ptr<Core::LinAlg::Vector<double>> sv,
          const enum XFEM::MapType& map_type) const
      {
        return slave_to_master(
            *std::const_pointer_cast<const Core::LinAlg::Vector<double>>(sv), map_type);
      }

      /** \brief transfer a nodal/dof multi vector from master to slave
       *
       *  \param mv       (in) : master multi vector (to be transferred)
       *  \param map_type (in) : map type of the master vector */
      inline std::shared_ptr<Core::LinAlg::MultiVector<double>> master_to_slave(
          std::shared_ptr<Core::LinAlg::MultiVector<double>> mv,
          const enum XFEM::MapType& map_type) const
      {
        return master_to_slave(
            *std::const_pointer_cast<const Core::LinAlg::MultiVector<double>>(mv), map_type);
      }

      /** \brief transfer a nodal/dof multi vector from slave to master
       *
       *  \param sv       (in) : slave multi vector (to be transferred)
       *  \param map_type (in) : map type of the slave vector */
      inline std::shared_ptr<Core::LinAlg::MultiVector<double>> slave_to_master(
          std::shared_ptr<Core::LinAlg::MultiVector<double>> sv,
          const enum XFEM::MapType& map_type) const
      {
        return slave_to_master(
            *std::const_pointer_cast<const Core::LinAlg::MultiVector<double>>(sv), map_type);
      }

      /** \brief transfer a nodel/dof vector from master to slave
       *
       *  \param mv       (in) : master vector (to be transferred)
       *  \param map_type (in) : map type of the master vector */
      std::shared_ptr<Core::LinAlg::Vector<double>> master_to_slave(
          const Core::LinAlg::Vector<double>& mv, const enum XFEM::MapType& map_type) const;

      /** \brief transfer a nodal/dof vector from slave to master
       *
       *  \param sv       (in) : slave vector (to be transferred)
       *  \param map_type (in) : map type of the slave vector */
      std::shared_ptr<Core::LinAlg::Vector<double>> slave_to_master(
          const Core::LinAlg::Vector<double>& sv, const enum XFEM::MapType& map_type) const;

      /** \brief transfer a nodel/dof vector from master to slave
       *
       *  \param mv       (in) : master vector (to be transferred)
       *  \param map_type (in) : map type of the master vector */
      std::shared_ptr<Core::LinAlg::MultiVector<double>> master_to_slave(
          const Core::LinAlg::MultiVector<double>& mv, const enum XFEM::MapType& map_type) const;

      /** \brief transfer a nodal/dof multi vector from slave to master
       *
       *  \param sv       (in) : slave multi vector (to be transferred)
       *  \param map_type (in) : map type of the slave vector */
      std::shared_ptr<Core::LinAlg::MultiVector<double>> slave_to_master(
          const Core::LinAlg::MultiVector<double>& sv, const enum XFEM::MapType& map_type) const;

      /** \brief transfer a nodel/dof multi vector from master to slave
       *
       *  \param mv       (in) : master multi vector (to be transferred/source)
       *  \param map_type (in) : map type of the master vector
       *  \param sv       (out): slave multi vector (target)*/
      void master_to_slave(const Core::LinAlg::MultiVector<double>& mv,
          const enum XFEM::MapType& map_type, Core::LinAlg::MultiVector<double>& sv) const;

      /** \brief transfer a nodal/dof multi vector from slave to master
       *
       *  \param sv       (in) : slave multi vector (to be transferred)
       *  \param map_type (in) : map type of the slave vector
       *  \param mv       (out): master multi vector (target)*/
      void slave_to_master(const Core::LinAlg::MultiVector<double>& sv,
          const enum XFEM::MapType& map_type, Core::LinAlg::MultiVector<double>& mv) const;

      //@}

     protected:
      /// check the isinit_ flag
      inline void check_init() const
      {
        if (not isinit_) FOUR_C_THROW("Call Init first!");
      }

      /** \brief build dof maps from node maps
       *
       *  \note It is assumed that the first numdof DoF's of each
       *  node are of interest. If numdof is equal -1 the maximum
       *  number of shared DoF's per node will be coupled, i.e.
       *  numdof is equivalent to the dof number of MinDofDiscretization
       *  at each node.
       *  Otherwise the base class variant is called.
       *
       */
      void build_dof_maps(const Core::FE::Discretization& masterdis,
          const Core::FE::Discretization& slavedis,
          const std::shared_ptr<const Core::LinAlg::Map>& masternodemap,
          const std::shared_ptr<const Core::LinAlg::Map>& slavenodemap,
          const std::shared_ptr<const Core::LinAlg::Map>& permmasternodemap,
          const std::shared_ptr<const Core::LinAlg::Map>& permslavenodemap,
          const std::vector<int>& masterdofs, const std::vector<int>& slavedofs,
          const int nds_master = 0, const int nds_slave = 0) override;

     private:
      /** \brief store the nodal maps and create the corresponding exporter objects
       *
       *  This becomes necessary, since we want to transfer dof's as well as nodal
       *  distributed vectors
       *
       */
      void save_node_maps(const std::shared_ptr<const Core::LinAlg::Map>& masternodemap,
          const std::shared_ptr<const Core::LinAlg::Map>& slavenodemap,
          const std::shared_ptr<const Core::LinAlg::Map>& permmasternodemap,
          const std::shared_ptr<const Core::LinAlg::Map>& permslavenodemap);

      /** \brief Identify the number of DoF's of the min-dof discretization as well
       *  as the actual dof GID's
       *
       *  This method takes all current DoF's of each node in the min-dof-discretization
       *  and sets the necessary maps for the coupling.
       */
      void build_min_dof_maps(const Core::FE::Discretization& min_dis,
          const Core::LinAlg::Map& min_nodemap, const Core::LinAlg::Map& min_permnodemap,
          std::shared_ptr<const Core::LinAlg::Map>& min_dofmap,
          std::shared_ptr<const Core::LinAlg::Map>& min_permdofmap,
          std::shared_ptr<Core::LinAlg::Export>& min_exporter, const Core::LinAlg::Map& max_nodemap,
          std::map<int, unsigned>& my_mindofpernode) const;

      void build_max_dof_maps(const Core::FE::Discretization& max_dis,
          const Core::LinAlg::Map& max_nodemap, const Core::LinAlg::Map& max_permnodemap,
          std::shared_ptr<const Core::LinAlg::Map>& max_dofmap,
          std::shared_ptr<const Core::LinAlg::Map>& max_permdofmap,
          std::shared_ptr<Core::LinAlg::Export>& max_exporter,
          const std::map<int, unsigned>& my_mindofpernode) const;

      inline const enum MinDofDiscretization& min_dof_dis() const { return min_dof_dis_; }

     private:
      bool isinit_;

      enum MinDofDiscretization min_dof_dis_;

      std::shared_ptr<const Core::LinAlg::Map> masternodemap_;

      std::shared_ptr<const Core::LinAlg::Map> slavenodemap_;

      std::shared_ptr<const Core::LinAlg::Map> permmasternodemap_;

      std::shared_ptr<const Core::LinAlg::Map> permslavenodemap_;

      //! @name Nodal communication objects
      //@{

      //! permuted master dof map to master dof map exporter
      std::shared_ptr<Core::LinAlg::Export> nodal_masterexport_;

      //! permuted slave dof map to slave dof map exporter
      std::shared_ptr<Core::LinAlg::Export> nodal_slaveexport_;

      //@}
    };  // class Coupling
  }  // namespace XFieldField
}  // namespace XFEM



FOUR_C_NAMESPACE_CLOSE

#endif
