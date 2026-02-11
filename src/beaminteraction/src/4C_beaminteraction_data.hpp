// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_DATA_HPP
#define FOUR_C_BEAMINTERACTION_DATA_HPP

#include "4C_config.hpp"

#include "4C_beaminteraction_crosslinking_submodel_evaluator.hpp"
#include "4C_beaminteraction_input.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

#include <map>
#include <set>

FOUR_C_NAMESPACE_OPEN


// forward declarations
namespace Discret
{
  class ParObject;
  class PackBuffer;
}  // namespace Discret
// forward declaration
namespace Solid
{
  namespace TimeInt
  {
    class BaseDataGlobalState;
  }
}  // namespace Solid
namespace BeamInteraction
{
  /*!
   * data container for input file parameters for submodel crosslinking in beam interaction
   * author eichinger*/
  class BeamInteractionParams
  {
   public:
    //! constructor
    BeamInteractionParams();

    //! destructor
    virtual ~BeamInteractionParams() = default;

    //! initialize with the stuff coming from input file
    void init();

    //! setup member variables
    void setup();

    //! returns the isinit_ flag
    inline const bool& is_init() const { return isinit_; };

    //! returns the issetup_ flag
    inline const bool& is_setup() const { return issetup_; };

    //! Checks the init and setup status
    inline void check_init_setup() const
    {
      if (!is_init() or !is_setup()) FOUR_C_THROW("Call init() and setup() first!");
    }

    //! Checks the init status
    inline void check_init() const
    {
      if (!is_init()) FOUR_C_THROW("init() has not been called, yet!");
    }

    /// number of crosslinkers per type
    BeamInteraction::RepartitionStrategy get_repartition_strategy() const
    {
      check_init_setup();
      return rep_strategy_;
    };

   private:
    bool isinit_;

    bool issetup_;

    /// number of crosslinker that are initially set
    BeamInteraction::RepartitionStrategy rep_strategy_;
  };



  namespace Data
  {
    //! struct to store crosslinker data and enable parallel redistribution
    struct CrosslinkerData
    {
     public:
      //! constructor
      CrosslinkerData();

      //! destructor
      virtual ~CrosslinkerData() = default;


      //!@name data access functions
      //! @{

      int get_id() const { return id_; };

      Core::LinAlg::Matrix<3, 1> const& get_position() const { return pos_; };

      int get_number_of_bonds() const { return numbond_; };

      std::vector<std::pair<int, int>> const& get_b_spots() const { return bspots_; };

      void set_id(int id) { id_ = id; };

      void set_position(Core::LinAlg::Matrix<3, 1> const& clpos) { pos_ = clpos; };

      void set_number_of_bonds(int clnumbond) { numbond_ = clnumbond; };

      void set_b_spots(std::vector<std::pair<int, int>> const& clbspots) { bspots_ = clbspots; };
      void set_bspot(int bspotid, std::pair<int, int> bpsotone) { bspots_[bspotid] = bpsotone; };

      //! @}

      //!@name methods for enabling parallel use of data container
      //! @{

      /*!
      \brief Pack this class so it can be communicated
      */
      void pack(Core::Communication::PackBuffer& data) const;

      /*!
      \brief Unpack data into this container
      */
      void unpack(Core::Communication::UnpackBuffer& buffer);

      //! @}

     private:
      //!@name linker member variables
      //! @{

      /// linker gid
      int id_;

      /// current position of crosslinker
      Core::LinAlg::Matrix<3, 1> pos_;

      /// number of active bonds
      int numbond_;

      /// gid of element to local number of bspot, [0] and [1] first and second bspot
      std::vector<std::pair<int, int>> bspots_;


      //! @}
    };

    //! struct to store crosslinker data and enable parallel redistribution
    struct BeamData
    {
     public:
      //! constructor
      BeamData();

      //! destructor
      virtual ~BeamData() = default;


      //!@name data access functions
      //! @{

      int get_id() const { return id_; };

      std::map<BeamInteraction::CrosslinkerType, std::map<int, Core::LinAlg::Matrix<3, 1>>> const&
      get_b_spot_positions() const
      {
        return bspotpos_;
      };

      Core::LinAlg::Matrix<3, 1> const& get_b_spot_position(
          BeamInteraction::CrosslinkerType linkertype, int bspotid) const
      {
        return bspotpos_.at(linkertype).at(bspotid);
      };

      std::map<BeamInteraction::CrosslinkerType, std::map<int, Core::LinAlg::Matrix<3, 3>>> const&
      get_b_spot_triads() const
      {
        return bspottriad_;
      };
      Core::LinAlg::Matrix<3, 3> const& get_b_spot_triad(
          BeamInteraction::CrosslinkerType linkertype, int bspotid) const
      {
        return bspottriad_.at(linkertype).at(bspotid);
      };

      std::map<BeamInteraction::CrosslinkerType, std::map<int, std::set<int>>> const&
      get_b_spot_status() const
      {
        return bspotstatus_;
      };

      std::set<int> const& get_b_spot_status_at(
          BeamInteraction::CrosslinkerType linkertype, int bspotid) const
      {
        return bspotstatus_.at(linkertype).at(bspotid);
      };

      // not [] necessary here in case element has no binding spot of certain type
      unsigned int get_number_of_binding_spots_of_type(BeamInteraction::CrosslinkerType linkertype)
      {
        return bspotstatus_[linkertype].size();
      };


      void set_id(int id) { id_ = id; };

      void set_b_spot_positions(std::map<BeamInteraction::CrosslinkerType,
          std::map<int, Core::LinAlg::Matrix<3, 1>>> const& bspotpos)
      {
        bspotpos_ = bspotpos;
      };
      void set_b_spot_position(BeamInteraction::CrosslinkerType linkertype, int bspotid,
          Core::LinAlg::Matrix<3, 1> const& bspotpos)
      {
        bspotpos_[linkertype][bspotid] = bspotpos;
      };

      void set_b_spot_triads(std::map<BeamInteraction::CrosslinkerType,
          std::map<int, Core::LinAlg::Matrix<3, 3>>> const& bspottriad)
      {
        bspottriad_ = bspottriad;
      };
      void set_b_spot_triad(BeamInteraction::CrosslinkerType linkertype, int bspotid,
          Core::LinAlg::Matrix<3, 3> const& bspottriad)
      {
        bspottriad_[linkertype][bspotid] = bspottriad;
      };

      void set_b_spot_status(
          std::map<BeamInteraction::CrosslinkerType, std::map<int, std::set<int>>> const&
              bspotstatus)
      {
        bspotstatus_ = bspotstatus;
      };
      void set_b_spot_status(
          BeamInteraction::CrosslinkerType linkertype, int bspotid, std::set<int> clgids)
      {
        bspotstatus_[linkertype][bspotid] = clgids;
      };

      void erase_bond_from_binding_spot(
          BeamInteraction::CrosslinkerType linkertype, int locbspotid, int clgid)
      {
        bspotstatus_.at(linkertype).at(locbspotid).erase(clgid);
      }
      void add_bond_to_binding_spot(
          BeamInteraction::CrosslinkerType linkertype, int locbspotid, int clgid)
      {
        bspotstatus_.at(linkertype).at(locbspotid).insert(clgid);
      }


      //! @}

      //!@name methods for enabling parallel use of data container
      //! @{

      /*!
      \brief Pack this class so it can be communicated
      */
      void pack(Core::Communication::PackBuffer& data) const;

      /*!
      \brief Unpack data from a char vector into this container
      */
      void unpack(Core::Communication::UnpackBuffer& buffer);

      //! @}

     private:
      //!@name linker member variables
      //! @{

      /// beam gid
      int id_;

      /// current position at bindingspots (xi) (key is local number of binding spot)
      std::map<BeamInteraction::CrosslinkerType, std::map<int, Core::LinAlg::Matrix<3, 1>>>
          bspotpos_;

      /// current triad at bindingspots (xi) (key is local number of binding spot)
      std::map<BeamInteraction::CrosslinkerType, std::map<int, Core::LinAlg::Matrix<3, 3>>>
          bspottriad_;

      /// key is locn of bspot, holds gid of crosslinker to which it is bonded
      std::map<BeamInteraction::CrosslinkerType, std::map<int, std::set<int>>> bspotstatus_;

      //! @}
    };

    struct BindEventData
    {
      //! constructor
      BindEventData();

      //! destructor
      virtual ~BindEventData() = default;

      //! Init
      void init(int clgid, int elegid, int bspotlocn, int requestproc, int permission);


      //!@name data access functions
      //! @{

      int get_cl_id() const { return clgid_; };

      int get_ele_id() const { return elegid_; };

      int get_b_spot_loc_n() const { return bspotlocn_; };

      int get_request_proc() const { return requestproc_; };

      int get_permission() const { return permission_; };

      void set_cl_id(int clgid) { clgid_ = clgid; };

      void set_ele_id(int elegid) { elegid_ = elegid; };

      void set_b_spot_loc_n(int bspotlocn) { bspotlocn_ = bspotlocn; };

      void set_request_proc(int requestproc) { requestproc_ = requestproc; };

      void set_permission(int permission) { permission_ = permission; };

      //! @}

      //!@name methods for enabling parallel use of data container
      //! @{

      /*!
      \brief Pack this class so it can be communicated
      */
      void pack(Core::Communication::PackBuffer& data) const;

      /*!
      \brief Unpack data from a char vector into this container
      */
      void unpack(Core::Communication::UnpackBuffer& buffer);

      //! @}

     private:
      //!@name member variables
      //! @{

      // gid of crosslinker
      int clgid_;

      // ele gid crosslinker wants to bind to
      int elegid_;

      // loc number of bspot on ele cl wants to bind to
      int bspotlocn_;

      // myrank, processor that is requesting
      int requestproc_;

      // permission/veto, if crosslinker is allowed to bind
      int permission_;

      //! @}
    };

    struct UnBindEventData
    {
      //! constructor
      UnBindEventData();

      //! destructor
      virtual ~UnBindEventData() = default;

      //!@name data access functions
      //! @{

      int get_cl_id() const { return clgid_; };

      std::pair<int, int> const& get_ele_toupdate() const { return eletoupdate_; };

      BeamInteraction::CrosslinkerType get_linker_type() const { return linkertype_; };


      void set_cl_id(int clgid) { clgid_ = clgid; };

      void set_ele_toupdate(std::pair<int, int> eletoupdate) { eletoupdate_ = eletoupdate; };

      void set_linker_type(BeamInteraction::CrosslinkerType linkertype)
      {
        linkertype_ = linkertype;
      };


      //! @}


      //!@name methods for enabling parallel use of data container
      //! @{

      /*!
      \brief Pack this class so it can be communicated
      */
      void pack(Core::Communication::PackBuffer& data) const;

      /*!
      \brief Unpack data from a char vector into this container
      */
      void unpack(Core::Communication::UnpackBuffer& buffer);

      //! @}

     private:
      //!@name member variables
      //! @{

      /// crosslinker (gid) that is no longer bonded to above binding spot
      int clgid_;

      /// element gid (first) that needs to be updated at local binding (second)
      std::pair<int, int> eletoupdate_;

      /// type of binding spot where unbinding takes place
      BeamInteraction::CrosslinkerType linkertype_;


      //! @}
    };

    struct BspotLinkerData
    {
      //! constructor
      BspotLinkerData();

      //! destructor
      virtual ~BspotLinkerData() = default;

      //!@name data access functions
      //! @{

      int get_ele_gid1() const { return elegid_1_; };
      int get_ele_gid2() const { return elegid_2_; };

      int get_loc_bspot_id1() const { return locbspot_1_; };
      int get_loc_bspot_id2() const { return locbspot_2_; };

      int get_type() const { return type_; };

      int get_mat_id() const { return mat_id_; };

      int get_number_of_bonds1() const { return number_of_bonds_1_; };
      int get_number_of_bonds2() const { return number_of_bonds_2_; };

      void set_ele_gid1(int elegid) { elegid_1_ = elegid; };
      void set_ele_gid2(int elegid) { elegid_2_ = elegid; };

      void set_loc_bspot_id1(int locbspot) { locbspot_1_ = locbspot; };
      void set_loc_bspot_id2(int locbspot) { locbspot_2_ = locbspot; };

      void set_type(int type) { type_ = type; };

      void set_mat_id(int mat_id) { mat_id_ = mat_id; };

      void set_number_of_bonds1(int number_of_bonds) { number_of_bonds_1_ = number_of_bonds; };
      void set_number_of_bonds2(int number_of_bonds) { number_of_bonds_2_ = number_of_bonds; };

      bool same_as(BspotLinkerData bspotlinker);

      //! @}

     private:
      //!@name member variables
      //! @{

      /// element ids
      int elegid_1_;
      int elegid_2_;

      /// binding spot local ids
      int locbspot_1_;
      int locbspot_2_;

      /// crosslinker type
      int type_;

      /// crosslinker mat id
      int mat_id_;

      /// number of bonds
      int number_of_bonds_1_;
      int number_of_bonds_2_;

      /// element

      //! @}
    };

    /// create respective data container from char vector
    template <typename T>
    T* create_data_container(Core::Communication::UnpackBuffer& data)
    {
      T* new_container = new T();
      new_container->unpack(data);
      return new_container;
    };

  }  // namespace Data
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
