/*---------------------------------------------------------------------------*/
/*! \file
\brief history pair handler for discrete element method (DEM) interactions
\level 3
*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*
 | definitions                                                               |
 *---------------------------------------------------------------------------*/
#ifndef FOUR_C_PARTICLE_INTERACTION_DEM_HISTORY_PAIRS_HPP
#define FOUR_C_PARTICLE_INTERACTION_DEM_HISTORY_PAIRS_HPP

/*---------------------------------------------------------------------------*
 | headers                                                                   |
 *---------------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_particle_engine_enums.hpp"
#include "4C_particle_engine_typedefs.hpp"
#include "4C_particle_interaction_dem_history_pair_struct.hpp"

#include <Epetra_Comm.h>

#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------------*
 | forward declarations                                                      |
 *---------------------------------------------------------------------------*/
namespace Core::IO
{
  class DiscretizationReader;
}

namespace PARTICLEENGINE
{
  class ParticleEngineInterface;
}

/*---------------------------------------------------------------------------*
 | type definitions                                                          |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  using TouchedDEMHistoryPairTangential =
      std::pair<bool, ParticleInteraction::DEMHistoryPairTangential>;
  using DEMHistoryPairTangentialData =
      std::unordered_map<int, std::unordered_map<int, TouchedDEMHistoryPairTangential>>;

  using TouchedDEMHistoryPairRolling = std::pair<bool, ParticleInteraction::DEMHistoryPairRolling>;
  using DEMHistoryPairRollingData =
      std::unordered_map<int, std::unordered_map<int, TouchedDEMHistoryPairRolling>>;

  using TouchedDEMHistoryPairAdhesion =
      std::pair<bool, ParticleInteraction::DEMHistoryPairAdhesion>;
  using DEMHistoryPairAdhesionData =
      std::unordered_map<int, std::unordered_map<int, TouchedDEMHistoryPairAdhesion>>;
}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*
 | class declarations                                                        |
 *---------------------------------------------------------------------------*/
namespace ParticleInteraction
{
  class DEMHistoryPairs final
  {
   public:
    //! constructor
    explicit DEMHistoryPairs(const Epetra_Comm& comm);

    //! init history pair handler
    void init();

    //! setup history pair handler
    void setup(
        const std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface);

    //! write restart of history pair handler
    void write_restart() const;

    //! read restart of history pair handler
    void read_restart(const std::shared_ptr<Core::IO::DiscretizationReader> reader);

    //! get reference to particle tangential history pair data
    inline DEMHistoryPairTangentialData& get_ref_to_particle_tangential_history_data()
    {
      return particletangentialhistorydata_;
    };

    //! get reference to particle-wall tangential history pair data
    inline DEMHistoryPairTangentialData& get_ref_to_particle_wall_tangential_history_data()
    {
      return particlewalltangentialhistorydata_;
    };

    //! get reference to particle rolling history pair data
    inline DEMHistoryPairRollingData& get_ref_to_particle_rolling_history_data()
    {
      return particlerollinghistorydata_;
    };

    //! get reference to particle-wall rolling history pair data
    inline DEMHistoryPairRollingData& get_ref_to_particle_wall_rolling_history_data()
    {
      return particlewallrollinghistorydata_;
    };

    //! get reference to particle adhesion history pair data
    inline DEMHistoryPairAdhesionData& get_ref_to_particle_adhesion_history_data()
    {
      return particleadhesionhistorydata_;
    };

    //! get reference to particle-wall adhesion history pair data
    inline DEMHistoryPairAdhesionData& get_ref_to_particle_wall_adhesion_history_data()
    {
      return particlewalladhesionhistorydata_;
    };

    //! distribute history pairs
    void distribute_history_pairs();

    //! communicate history pairs
    void communicate_history_pairs();

    //! update history pairs
    void update_history_pairs();

   private:
    //! communicate specific history pairs
    template <typename Historypairtype>
    void communicate_specific_history_pairs(const std::vector<std::vector<int>>& particletargets,
        std::unordered_map<int, std::unordered_map<int, std::pair<bool, Historypairtype>>>&
            historydata);

    //! erase untouched history pairs
    template <typename Historypairtype>
    void erase_untouched_history_pairs(
        std::unordered_map<int, std::unordered_map<int, std::pair<bool, Historypairtype>>>&
            historydata);

    //! pack all history pairs
    template <typename Historypairtype>
    void pack_all_history_pairs(std::vector<char>& buffer,
        const std::unordered_map<int, std::unordered_map<int, std::pair<bool, Historypairtype>>>&
            historydata) const;

    //! unpack history pairs
    template <typename Historypairtype>
    void unpack_history_pairs(const std::vector<char>& buffer,
        std::unordered_map<int, std::unordered_map<int, std::pair<bool, Historypairtype>>>&
            historydata);

    //! add history pair to buffer
    template <typename Historypairtype>
    void add_history_pair_to_buffer(std::vector<char>& buffer, int globalid_i, int globalid_j,
        const Historypairtype& historypair) const;

    //! communication
    const Epetra_Comm& comm_;

    //! particle tangential history pair data
    DEMHistoryPairTangentialData particletangentialhistorydata_;

    //! particle-wall tangential history pair data
    DEMHistoryPairTangentialData particlewalltangentialhistorydata_;

    //! particle rolling history pair data
    DEMHistoryPairRollingData particlerollinghistorydata_;

    //! particle-wall rolling history pair data
    DEMHistoryPairRollingData particlewallrollinghistorydata_;

    //! particle adhesion history pair data
    DEMHistoryPairAdhesionData particleadhesionhistorydata_;

    //! particle-wall adhesion history pair data
    DEMHistoryPairAdhesionData particlewalladhesionhistorydata_;

    //! interface to particle engine
    std::shared_ptr<PARTICLEENGINE::ParticleEngineInterface> particleengineinterface_;
  };

}  // namespace ParticleInteraction

/*---------------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
