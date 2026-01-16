// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_ROUGH_NODE_HPP
#define FOUR_C_CONTACT_ROUGH_NODE_HPP

#include "4C_config.hpp"

#include "4C_contact_node.hpp"
#include "4C_linalg_serialdensematrix.hpp"

#ifdef FOUR_C_WITH_MIRCO
#include <mirco_kokkostypes.h>
#endif

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  class RoughNodeType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const final { return "RoughNodeType"; }

    static RoughNodeType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static RoughNodeType instance_;
  };


  class RoughNode : public Node
  {
   public:
    //! @name Enums and Friends

    /*!
     \brief The discretization is a friend of RoughNode
     */
    friend class Core::FE::Discretization;

    //@}

    //! @name Constructors and destructors and related methods

    /*!
     \brief Standard Constructor

     \param id     (in): A globally unique node id
     \param coords (in): span of nodal coordinates, length 3
     \param owner  (in): Owner of this node.
     \param dofs   (in): list of global degrees of freedom
     \param isslave(in): flag indicating whether node is slave or master
     \param initactive (in): flag indicating whether initially set to active
     \param hurstexponentfunction (in): Function for Hurst exponent of the surface
     \param initialtopologystddeviationfunction (in): function for topology standard deviation
     \param resolution (in): resolution of the surface
     \param randomtopologyflag (in): Use random midpoint generator if true
     \param randomseedflag (in): Use random seed for the random midpoint generator
     \param randomgeneratorseed (in): Seed for the random midpoint generator
     \\ add remaining params here

     */
    RoughNode(int id, std::span<const double> coords, const int owner, const std::vector<int>& dofs,
        const bool isslave, const bool initactive, const int hurstexponentfunction,
        int initialtopologystddeviationfunction, int resolution, bool randomtopologyflag,
        bool randomseedflag, int randomgeneratorseed);

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of lib/parobject.H.

     */
    int unique_par_object_id() const override
    {
      return RoughNodeType::instance().unique_par_object_id();
    }

    /*!
     \brief Pack this class so it can be communicated

     \ref pack and \ref unpack are used to communicate this node

    */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
     \brief Unpack data from a char vector into this class

     \ref pack and \ref unpack are used to communicate this node

     */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    // //! @name Access methods
#ifdef FOUR_C_WITH_MIRCO
    const MIRCO::ViewMatrix_d* get_topology() const { return &topology_; };
#endif
    double get_max_topology_height() const { return maxTopologyHeight_; };

   protected:
    int hurstexponentfunction_;
    int initialtopologystddeviationfunction_;
    int resolution_;
    bool randomtopologyflag_;
    bool randomseedflag_;
    int randomgeneratorseed_;

    double hurstExponent_ = 0;
    double initialTopologyStdDeviation_ = 0;

#ifdef FOUR_C_WITH_MIRCO
    MIRCO::ViewMatrix_d topology_;
#endif

    double maxTopologyHeight_;
  };
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
