// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_SPEC_HPP
#define FOUR_C_IO_INPUT_SPEC_HPP

#include "4C_config.hpp"

#include <memory>
#include <optional>

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
     * Print the expected input format of this InputSpec to @p stream in dat format.
     */
    void print_as_dat(std::ostream& stream) const;

    /**
     * Emit metadata about the InputSpec to the @p yaml emitter.
     */
    void emit_metadata(YamlNodeRef yaml) const;

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

   private:
    std::shared_ptr<Internal::InputSpecImpl> pimpl_;
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
