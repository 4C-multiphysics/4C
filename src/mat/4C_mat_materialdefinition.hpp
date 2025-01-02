// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MATERIALDEFINITION_HPP
#define FOUR_C_MAT_MATERIALDEFINITION_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_io_input_spec_builders.hpp"
#include "4C_io_linecomponent.hpp"
#include "4C_mat_par_bundle.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Input
{
  class MaterialDefinition;
}  // namespace Input

namespace Core::IO
{
  class InputFile;
}

namespace Mat
{
  /// Definition of a valid material in 4C input
  ///
  /// This is basically a clone of \see ConditionDefinition which
  /// was adjusted to adhere the heterogeneous lines occurring
  /// in material descriptions in the DAT file.
  ///
  /// A MaterialDefinition is the definition of a --MATERIALS in DAT file
  /// section. This definition includes the knowledge what this section looks
  /// like, how to read it and how to write it. In particular given a
  /// MaterialDefinition object it is possible to (a) write an empty DAT file
  /// section that describes digestible materials, (b) read a DAT file and create
  /// Mat::PAR::Material objects for each line in this section and (c) write the DAT
  /// file section filled with all corresponding materials from a given
  /// Core::FE::Discretization.
  ///
  /// So this is quite sophisticated internal stuff here. If you want to
  /// introduce a new material to 4C, all you have to do is add an
  /// appropriate definition in valid_materials(). This will take care of the
  /// reading part and you will get your Core::FE::Discretization filled with proper
  /// Discret::Material objects.
  ///
  /// \author bborn
  /// \date 02/09
  class MaterialDefinition
  {
   public:
    /// construction of a material definition
    MaterialDefinition(std::string materialname,  ///< name of materials in Core::FE::Discretization
        std::string description,                  ///< description of material type
        Core::Materials::MaterialType mattype     ///< type of materials to be build
    );

    /// add a concrete component to the material line definition
    ///
    /// Add new components to the input line. One at a time.
    void add_component(Core::IO::InputSpec&& c);

    /// print my DAT file section and possible materials from the discretization
    std::ostream& print(std::ostream& stream,  ///< the output stream
        const Core::FE::Discretization* dis = nullptr);

    /// my material name
    std::string name() const { return materialname_; }

    // my material type
    Core::Materials::MaterialType type() const { return mattype_; }

    /// my material description
    std::string description() const { return description_; }

    /// Read access to the InputSpecs that make up the material definition.
    ///
    /// Legacy: only used for rtd emitter.
    const std::vector<Core::IO::InputSpec>& specs() const { return components_; }

   private:
    /// name of material
    std::string materialname_;
    /// description of material type
    std::string description_;
    /// type of materials to be build
    Core::Materials::MaterialType mattype_;

    /// the list of valid components
    std::vector<Core::IO::InputSpec> components_;
  };


  /// add material definition to list of defined materials
  ///
  /// this method checks for coincidental material names or types
  void append_material_definition(
      std::vector<std::shared_ptr<MaterialDefinition>>& matlist,  ///< list of defined materials
      const std::shared_ptr<MaterialDefinition>& mat              ///< material to add
  );

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
