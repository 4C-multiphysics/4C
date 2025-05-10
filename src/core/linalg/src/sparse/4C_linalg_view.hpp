// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_VIEW_HPP
#define FOUR_C_LINALG_VIEW_HPP


#include "4C_config.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /**
   * A helper struct to easily specify that one of our linear algebra classes is a wrapper for
   * another class. This struct should be specialized for each of our linear algebra classes and
   * contain a single typedef `type` for the class that is wrapped.
   */
  template <typename T>
  struct WrapperFor
  {
    static_assert("You need to specialize this struct for the wrapper class.");
  };

  namespace Internal
  {
    /**
     * Helper type to work with const and non-const references.
     */
    template <typename T>
    using WrapperForWithQualifiers = std::conditional_t<std::is_const_v<std::remove_reference_t<T>>,
        const typename WrapperFor<std::decay_t<T>>::type,
        typename WrapperFor<std::decay_t<T>>::type>;
  }  // namespace Internal

  /**
   * Concept for a SourceType which can be viewed as a WrapperType. The WrapperType must have a
   * static method create_view which takes a SourceType and returns a shared pointer to a
   * WrapperType.
   */
  template <typename SourceType>
  concept Viewable = requires(SourceType source) {
    {
      Internal::WrapperForWithQualifiers<SourceType>::create_view(source)
    } -> std::same_as<std::unique_ptr<Internal::WrapperForWithQualifiers<SourceType>>>;
  };

  /**
   * Temporary helper class for migration from raw Trilinos classes to our own wrappers. Views one
   * of the Trilinos linear algebra types as one of ours. It is the users responsibility to ensure
   * that the viewed source object outlives the view. Example:
   *
   * @code
   *   // Function taking our Vector type.
   *   void f(const Core::LinAlg::Vector<double>& vec);
   *
   *   // We only have an Epetra_Vector object.
   *   const Epetra_Vector& source = ...;
   *   // Create a view of the source object.
   *   Core::LinAlg::View view(source);
   *   // It behaves like our type Core::LinAlg::Vector<double> and converts implicitly.
   *   f(view); // works
   *   // Convert to our type explicitly
   *   const Core::LinAlg::Vector<double>& vec = view.underlying();
   * @endcode
   *
   */
  template <typename WrapperType>
  class View
  {
   public:
    /**
     * Construct a view from a source object.
     */
    template <Viewable SourceType>
    View(SourceType& source) : view_(WrapperType::create_view(source))
    {
    }

    // Make the class hard to misuse and disallow copy and move.
    View(const View& other) = delete;
    View& operator=(const View& other) = delete;
    View(View&& other) = delete;
    View& operator=(View&& other) = delete;
    ~View() = default;

    /**
     * Allow implicit conversion to the underlying type. This allows us to use the view as if it
     * were the underlying type. This is useful for passing the view to functions which expect the
     * underlying wrapper type.
     */
    operator WrapperType&() { return *view_; }

    /**
     * Explicitly ask for the viewed underlying wrapper. This is simply another way to access
     * the viewed object when implicit conversion is not happening automatically.
     */
    WrapperType& underlying() { return *view_; }

   private:
    //! Source content wrapped in our own type.
    std::unique_ptr<WrapperType> view_;
  };


  // Deduction guide
  template <typename SourceType>
  View(SourceType&) -> View<Internal::WrapperForWithQualifiers<SourceType>>;

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE


#endif
