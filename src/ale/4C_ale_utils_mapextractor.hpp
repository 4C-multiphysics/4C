/*----------------------------------------------------------------------------*/
/*! \file


\brief map extractor functionalities for interface problems for ALE field

\level 1

*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
#ifndef FOUR_C_ALE_UTILS_MAPEXTRACTOR_HPP
#define FOUR_C_ALE_UTILS_MAPEXTRACTOR_HPP

#include "4C_config.hpp"

#include "4C_linalg_mapextractor.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace ALE
{
  namespace Utils
  {
    /// specific MultiMapExtractor to handle the ale field
    class MapExtractor : public Core::LinAlg::MultiMapExtractor
    {
     public:
      enum
      {
        cond_other = 0,
        cond_fsi = 1,
        cond_fs = 2,
        cond_lung_asi = 3,
        cond_ale_wear = 4,
        cond_bio_gr = 5,
        cond_au = 6,
        cond_fpsi = 7,
        cond_mortar = 8
      };

      /// application-specific types of Dirichlet sets
      enum AleDBCSetType
      {
        dbc_set_std = 0,       ///< type of Dirichlet set in standard ALE-FSI
        dbc_set_x_ff = 1,      ///< Dirichlet sets include fluid-fluid-interface DOF (during Newton
                               ///< iteration of XFFSI)
        dbc_set_x_fsi = 2,     ///< Dirichlet sets include ALE-sided FSI interface DOF (in
                               ///< ALE-relaxation step of XFFSI)
        dbc_set_biofilm = 3,   ///< type of Dirichlet set for biofilm applications
        dbc_set_part_fsi = 4,  ///< type of Dirichlet set for partitioned FSI
        dbc_set_wear = 5,      ///< type of Dirichlet set for wear application
        dbc_set_count = 6      ///< total number of types
      };

      /// setup the whole thing
      void setup(const Core::FE::Discretization& dis, bool overlapping = false);

      /// get all element gids those nodes are touched by any condition
      Teuchos::RCP<std::set<int>> conditioned_element_map(
          const Core::FE::Discretization& dis) const;

      MAP_EXTRACTOR_VECTOR_METHODS(other, cond_other)
      MAP_EXTRACTOR_VECTOR_METHODS(fsi_cond, cond_fsi)
      MAP_EXTRACTOR_VECTOR_METHODS(fs_cond, cond_fs)
      MAP_EXTRACTOR_VECTOR_METHODS(lung_asi_cond, cond_lung_asi)
      MAP_EXTRACTOR_VECTOR_METHODS(ale_wear_cond, cond_ale_wear)
      MAP_EXTRACTOR_VECTOR_METHODS(au_cond, cond_au)
      MAP_EXTRACTOR_VECTOR_METHODS(fpsi_cond, cond_fpsi)
      MAP_EXTRACTOR_VECTOR_METHODS(mortar, cond_mortar)
    };

    /// specific MultiMapExtractor to handle the fsi and ale meshtying at the same time
    class FsiMapExtractor : public Core::LinAlg::MultiMapExtractor
    {
     public:
      enum
      {
        cond_other = 0,
        cond_fsi = 1
      };

      /// setup the whole thing
      void setup(const Core::FE::Discretization& dis);

      MAP_EXTRACTOR_VECTOR_METHODS(other, cond_other)
      MAP_EXTRACTOR_VECTOR_METHODS(FSI, cond_fsi)
    };

    /// specific MultiMapExtractor to handle the fluid_fluid_Coupling
    class XFluidFluidMapExtractor : public Core::LinAlg::MultiMapExtractor
    {
     public:
      enum
      {
        cond_other = 0,
        cond_xfluidfluid = 1
      };

      /// setup the whole thing
      void setup(const Core::FE::Discretization& dis);

      MAP_EXTRACTOR_VECTOR_METHODS(other, cond_other)
      MAP_EXTRACTOR_VECTOR_METHODS(xfluid_fluid_cond, cond_xfluidfluid)
    };
  }  // namespace Utils
}  // namespace ALE

FOUR_C_NAMESPACE_CLOSE

#endif
