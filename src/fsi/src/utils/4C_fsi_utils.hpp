// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FSI_UTILS_HPP
#define FOUR_C_FSI_UTILS_HPP

#include "4C_config.hpp"

#include "4C_coupling_adapter_mortar.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_inpar_fsi.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <NOX_Abstract_Group.H>
#include <NOX_Abstract_Vector.H>

#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Adapter
{
  class CouplingMortar;
  class FSIStructureWrapper;
}  // namespace Adapter

namespace ALE
{
  class Ale;
}

namespace Core::IO
{
  class DiscretizationReader;
  class DiscretizationWriter;
}  // namespace Core::IO

namespace FSI
{
  namespace Utils
  {
    /// Check whether fluid node numbers and ALE node numbers are equal.
    bool fluid_ale_nodes_disjoint(
        Core::FE::Discretization& fluiddis,  ///< pointer to fluid discretization
        Core::FE::Discretization& aledis     ///< pointer to ALE discretization
    );

    /*!
    \brief implementation of sliding ALE stuff
     */
    class SlideAleUtils
    {
     public:
      /// constructor initializing internal variables
      SlideAleUtils(
          std::shared_ptr<Core::FE::Discretization> structdis,  ///< structure discretization
          std::shared_ptr<Core::FE::Discretization> fluiddis,   ///< fluid discretization
          Coupling::Adapter::CouplingMortar& coupsf,            ///< mortar adapter
          bool structcoupmaster,            ///< is structure master of adapter coupling?
          Inpar::FSI::SlideALEProj aleproj  ///< projection enum
      );

      /// empty destructor
      virtual ~SlideAleUtils() = default;
      /// remesh ALE corresponding
      void remeshing(Adapter::FSIStructureWrapper& structure,  ///< structure adapter
          Core::FE::Discretization& fluiddis,                  ///< fluid discretization
          Core::LinAlg::Vector<double>& idispale,      ///< standard ALE interface displacement
          Core::LinAlg::Vector<double>& iprojdispale,  ///< projected ALE interface displacement
          Coupling::Adapter::CouplingMortar& coupsf,   ///< mortar adapter
          MPI_Comm comm                                ///< communicator
      );

      /// Compute new coupling matrices D and M for solid/fluid
      void evaluate_mortar(
          Core::LinAlg::Vector<double>& idispstruct,  ///< displacement of structure
          Core::LinAlg::Vector<double>& idispfluid,   ///< (proj.) displacement of ale
          Coupling::Adapter::CouplingMortar& coupsf   ///< mortar adapter
      );

      /// Compute new coupling matrices D and M for solid/ale coupling
      void evaluate_fluid_mortar(
          std::shared_ptr<Core::LinAlg::Vector<double>> ima,  ///< displacement of structure
          std::shared_ptr<Core::LinAlg::Vector<double>> isl   ///< (proj.) displacement of ale
      );

      /// use fluid-fluid mortar interface to interpolate between fluid quantities before and after
      /// sliding
      std::shared_ptr<Core::LinAlg::Vector<double>> interpolate_fluid(
          const Core::LinAlg::Vector<double>&
              uold  ///< fluid velocity in configuration before sliding
      );

      /// write history vectors for restart
      void output_restart(Core::IO::DiscretizationWriter& output);

      /// read history values for restart
      void read_restart(Core::IO::DiscretizationReader& reader);

     protected:
      /// compute average interface displacement
      std::vector<double> centerdisp(
          Adapter::FSIStructureWrapper& structure,  ///< structure adapter
          MPI_Comm comm                             ///< communicator
      );

      /// compute approximate interface rotation (structuresplit)
      void rotation(Core::FE::Discretization& mtrdis,  ///< mtr interface  discretization
          Core::LinAlg::Vector<double>& idispale,      ///< vector of ALE displacements
          MPI_Comm comm,                               ///< communicator
          std::map<int, double>& rotrat,  ///< rotation ratio of tangential displacements
          Core::LinAlg::Vector<double>&
              rotfull  ///< vector of full displacements in tangential directions
      );


      /// calculate current position of structure interface nodes
      std::map<int, Core::LinAlg::Matrix<3, 1>> current_struct_pos(
          Core::LinAlg::Vector<double>& reddisp,   ///< redundant version of structure displacements
          Core::FE::Discretization& interfacedis,  ///< interface discretization
          std::map<int, double>& maxcoord);


      /// project ALE nodes onto the structure surface
      void slide_projection(Adapter::FSIStructureWrapper& structure,  ///< structure adapter
          Core::FE::Discretization& fluiddis,                         ///< fluid discretization
          Core::LinAlg::Vector<double>& idispale,      ///< standard ALE interface displacement
          Core::LinAlg::Vector<double>& iprojdispale,  ///< projected ALE interface displacement
          Coupling::Adapter::CouplingMortar& coupsf,   ///< mortar adapter
          MPI_Comm comm                                ///< communicator
      );

      /// Build full redundant structure and fluid elements.
      /// Necessary for search-trees since MORTAR elements do not know about their facets and edges.
      /// Furthermore, this function builds SolidSurface elements from the fluid outer surface
      /// for rotation.
      void redundant_elements(Coupling::Adapter::CouplingMortar& coupsf, MPI_Comm comm);

     private:
      const Inpar::FSI::SlideALEProj aletype_;
      std::shared_ptr<Core::LinAlg::Vector<double>>
          idispms_;  ///< merged vector of displacements (struct and fluid interface)
      std::vector<double> centerdisptotal_;  ///< sum over all center displacement increments
      double maxmindist_;                    ///< maximal distance between fluidpairs

      //      std::map<int, std::shared_ptr<Core::Elements::Element> > istructslideles_;  ///<
      //      sliding struct elements in the interface
      std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>>
          istructslideles_;  ///< sliding struct elements in the interface
      std::map<int, Core::Nodes::Node*>
          istructdispnodes_;  ///< struct nodes in the interface used for centerdisp calculation
      std::map<int, std::shared_ptr<Core::Elements::Element>>
          istructdispeles_;  ///< struct elements in the interface used for centerdisp calc
      //      std::map<int, std::shared_ptr<Core::LinAlg::Map> >  slideeleredmap_;      ///<
      //      redundant version of sliding struct elements
      std::map<int, std::map<int, Core::Nodes::Node*>>
          ifluidslidnodes_;  ///< sliding fluid nodes in the interface
      std::map<int, Core::Nodes::Node*>
          ifluidconfnodes_;  ///< sticking fluid nodes in the interface

      std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>> ifluidslideles_;
      std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>> ifluidslidstructeles_;

      std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>> structreduelements_;

      std::shared_ptr<const Core::LinAlg::Map> structdofrowmap_;
      std::shared_ptr<const Core::LinAlg::Map> fluiddofrowmap_;
      std::shared_ptr<Core::LinAlg::Map> structfullnodemap_;
      std::shared_ptr<Core::LinAlg::Map> structfullelemap_;
      std::shared_ptr<Core::LinAlg::Map> fluidfullnodemap_;
      std::shared_ptr<Core::LinAlg::Map> fluidfullelemap_;

      int maxid_;

      std::shared_ptr<Core::LinAlg::Vector<double>> iprojhist_;  ///< history of final displacements

      bool structcoupmaster_;  ///< is structure master of coupling?

      /// coupling of fluid before and fluid after the sliding
      std::shared_ptr<Coupling::Adapter::CouplingMortar> coupff_;

    };  // class SlideAleUtils

  }  // namespace Utils
}  // namespace FSI

FOUR_C_NAMESPACE_CLOSE

#endif
