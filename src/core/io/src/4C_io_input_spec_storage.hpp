// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_INPUT_SPEC_STORAGE_HPP
#define FOUR_C_IO_INPUT_SPEC_STORAGE_HPP

#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"

#include <any>
#include <functional>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  namespace InputSpecBuilders
  {
    /**
     * We can store to different types of container:
     *  - the dynamic InputParameterContainer, which will type-erase the value and requires
     * explicit knowledge of the type to retrieve it later.
     *  - a std::any which will usually contain a value of a struct type. An appropriate
     * StoreFunction needs to be provided which knows how to store the value in the struct.
     * Specifically, it needs to know the type of the struct. See in_struct().
     */
    using Storage = std::any;

    /**
     * A function type to store a value of type T in a Storage.
     */
    template <typename T>
    class StoreFunction
    {
     public:
      StoreFunction() = default;

      explicit StoreFunction(std::nullptr_t) {}

      StoreFunction(std::function<void(Storage&, T&&)> fn, const std::type_info& stores_to)
          : fn_(std::move(fn)), stores_to_(&stores_to)
      {
        FOUR_C_ASSERT(stores_to != typeid(void), "Cannot store to void type.");
      }

      void operator()(Storage& storage, T&& value) const { fn_(storage, std::move(value)); }

      explicit operator bool() const { return static_cast<bool>(fn_); }

      [[nodiscard]] const std::type_info& stores_to() const { return *stores_to_; }

     private:
      std::function<void(Storage&, T&&)> fn_{nullptr};
      const std::type_info* stores_to_{nullptr};
    };


    /**
     * Create a StoreFunction which stores a value to a specific member of a struct. The funny
     * looking type of the parameter is a pointer to a member of the struct, which can be specified
     * as seen in an example:
     *
     * @code
     *   struct MyStruct
     *   {
     *     int my_value;
     *   };
     *
     *   // Here, we refer to the member `my_value` of `MyStruct` using a pointer to member syntax.
     *   auto store_my_value = in_struct(&MyStruct::my_value);
     * @endcode
     *
     * You probably want to use this function in the context of an InputSpec, where you can pass the
     * result of this function to the `.store` field.
     *
     */
    template <typename StructType, typename MemberType>
    auto in_struct(MemberType StructType::* p);
  }  // namespace InputSpecBuilders

  namespace Internal
  {
    template <typename T>
    [[nodiscard]] bool holds(const InputSpecBuilders::Storage& storage)
    {
      return storage.type() == typeid(T);
    }

    inline void init_storage_with_container(InputSpecBuilders::Storage& storage)
    {
      storage.emplace<InputParameterContainer>();
    }

    /**
     * Get a StoreFunction that stores a value in an InputParameterContainer. This is only used
     * internally as the default if a storage function is not user-provided.
     */
    template <typename T>
    auto store_value_in_container(std::string name)
    {
      return InputSpecBuilders::StoreFunction<T>(
          [name](InputSpecBuilders::Storage& storage, T&& value)
          {
            FOUR_C_ASSERT(storage.has_value(), "Storage must be initialized before storing.");
            std::any_cast<InputParameterContainer&>(storage).add(name, std::move(value));
          },
          typeid(InputParameterContainer));
    }

    /**
     * Get a StoreFunction that stores a group container in an InputParameterContainer. This works
     * if both the storage and the group_storage are of type InputParameterContainer.
     */
    inline auto store_container_in_container(std::string name)
    {
      return InputSpecBuilders::StoreFunction<InputSpecBuilders::Storage>(
          [name](InputSpecBuilders::Storage& storage, InputSpecBuilders::Storage&& group_storage)
          {
            FOUR_C_ASSERT(holds<InputParameterContainer>(storage),
                "Internal error: storage must be an InputParameterContainer.");
            FOUR_C_ASSERT(holds<InputParameterContainer>(group_storage),
                "Internal error: value must be an InputParameterContainer.");
            std::any_cast<InputParameterContainer&>(storage).group(name) =
                std::any_cast<InputParameterContainer&&>(std::move(group_storage));
          },
          typeid(InputParameterContainer));
    }

    template <typename S>
    auto wrap_group_in_container(InputSpecBuilders::StoreFunction<S> f)
    {
      return InputSpecBuilders::StoreFunction<InputSpecBuilders::Storage>(
          [f](InputSpecBuilders::Storage& storage, InputSpecBuilders::Storage&& group_storage)
          {
            FOUR_C_ASSERT(holds<S>(group_storage), "Internal error: group_storage must be {}.",
                typeid(S).name());
            f(storage, std::any_cast<S&&>(std::move(group_storage)));
          },
          f.stores_to());
    }


  }  // namespace Internal


  // --- template definitions --- //

  template <typename StructType, typename MemberType>
  auto InputSpecBuilders::in_struct(MemberType StructType::* p)
  {
    return InputSpecBuilders::StoreFunction<MemberType>(
        [p](Storage& obj, MemberType&& val)
        {
          FOUR_C_ASSERT(Internal::holds<StructType>(obj),
              "Implementation error: expected an object of type {}, but got {}",
              typeid(StructType).name(), obj.type().name());
          std::any_cast<StructType&>(obj).*p = std::move(val);
        },
        typeid(StructType));
  }
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
