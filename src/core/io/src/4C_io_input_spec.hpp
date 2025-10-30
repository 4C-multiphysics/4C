// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_SPEC_HPP
#define FOUR_C_IO_INPUT_SPEC_HPP

#include "4C_config.hpp"

#include <limits>
#include <memory>
#include <optional>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class InputParameterContainer;
  class ValueParser;
  class YamlNodeRef;
  class ConstYamlNodeRef;

  namespace Internal
  {
    class InputSpecImpl;
  }  // namespace Internal

  struct InputSpecEmitOptions
  {
    /**
     * If set to true, entries which have a default value are emitted even if their values are equal
     * to the default.
     */
    bool emit_defaulted_values{false};
  };

  struct InputSpecEmitMetadataOptions
  {
    /**
     * If an InputSpec is duplicated more often than this number, condense all occurrences into a
     * single shared reference in the metadata. This means that if multiple InputSpecs have the same
     * content, they will not be emitted separately, but instead a single reference is created in
     * the metadata and the place where the InputSpec is used will point to that reference.
     *
     * By default, this is set to the maximum value of an unsigned integer, which means that no
     * condensation is done. A useful value is around 5, as a trade-off between condensing too
     * much and not condensing enough.
     */
    unsigned condense_duplicated_specs_threshold{std::numeric_limits<unsigned>::max()};

    /**
     * The name under which the metadata of duplicated InputSpecs is stored as a shared reference in
     * the metadata. This key will _always_ be placed in the root of the YAML tree.
     */
    std::string references_node_name{"$references"};

    /**
     * The key under which a reference to a shared InputSpec is stored in the YAML tree. This is the
     * key used in place of the full InputSpec metadata when a shared reference is used.
     */
    std::string reference_key{"$ref"};
  };

  /**
   * Objects of this class encapsulate knowledge about the input. You can create objects using the
   * helper functions in the InputSpecBuilders namespace. See the function
   * InputSpecBuilders::entry() for more information on how to create InputSpecs.
   */
  class InputSpec
  {
   public:
    InputSpec() = default;

    InputSpec(std::unique_ptr<Internal::InputSpecImpl> pimpl);

    /**
     * Use the @p parser to parse whatever this InputSpec expects. The results are stored in the
     * @p container. If parsing fails, an exception is thrown.
     */
    void fully_parse(ValueParser& parser, InputParameterContainer& container) const;

    /**
     * Match the content in @p yaml to the expected input format of this InputSpec. If the content
     * matches, fill the @p container with the parsed data. If the content does not match, throws an
     * exception. A successful match means that @p yaml contains all required entries and no unknown
     * content.
     */
    void match(ConstYamlNodeRef yaml, InputParameterContainer& container) const;

    /**
     * Emit the data in @p container to @p yaml. The data in @p container has to fit the
     * specification encoded in this InputSpec; otherwise an exception is thrown. This function is
     * the inverse of match().
     */
    void emit(YamlNodeRef yaml, InputParameterContainer& container,
        InputSpecEmitOptions options = {}) const;

    /**
     * Emit metadata about the InputSpec to the @p yaml emitter.
     */
    void emit_metadata(YamlNodeRef yaml, InputSpecEmitMetadataOptions options = {}) const;

    /**
     * Access the opaque implementation class. This is used in the implementation files where the
     * definition is known. There is nothing you can or should do with this function in the user
     * code.
     */
    Internal::InputSpecImpl& impl();

    /**
     * Access the opaque implementation class. This is used in the implementation files where the
     * definition is known. There is nothing you can or should do with this function in the user
     * code.
     */
    [[nodiscard]] const Internal::InputSpecImpl& impl() const;

    /**
     * Returns the number of times this exact InputSpec is used. Since InputSpecs are copied
     * shallowly, this function is useful to export knowledge about shared specs to the metadata.
     */
    [[nodiscard]] std::size_t use_count() const;

    /**
     * Get the name of this InputSpec. The name is used as a key in the input file.
     * The name may be an empty string in case the spec does not have a name.
     */
    [[nodiscard]] const std::string& name() const;

   private:
    std::shared_ptr<Internal::InputSpecImpl> pimpl_;
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
