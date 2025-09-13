// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_VIEW_HPP
#define FOUR_C_LINALG_VIEW_HPP


#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

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
  struct EnableViewFor
  {
    static_assert("You need to specialize this struct for the wrapper class.");
  };

  namespace Internal
  {
    /**
     * Helper type to work with const and non-const references.
     */
    template <typename T>
    using EnableViewForWithQualifiers =
        std::conditional_t<std::is_const_v<std::remove_reference_t<T>>,
            const typename EnableViewFor<std::decay_t<T>>::type,
            typename EnableViewFor<std::decay_t<T>>::type>;
  }  // namespace Internal

  /**
   * Concept for a SourceType which can be viewed as a WrapperType. The WrapperType must have a
   * static method create_view which takes a SourceType and returns a unique_ptr to a WrapperType
   * which behaves like a view of the SourceType. For the implementation, the WrapperType can use
   * Core::Utils::OwnerOrView to hold the pointer to the SourceType.
   */
  template <typename SourceType>
  concept Viewable = requires(SourceType source) {
    {
      Internal::EnableViewForWithQualifiers<SourceType>::create_view(source)
    } -> std::same_as<std::unique_ptr<Internal::EnableViewForWithQualifiers<SourceType>>>;
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
   * Another use case of this class is the following case. Suppose we want to wrap a class ExtA
   * which can return a reference to ExtB. We have our own wrappers for ExtA and ExtB, let us call
   * them MyA and MyB, respectively. Now we want to implement a function that returns a reference to
   * MyB from MyA, where the MyB is a view of the ExtB object returned by ExtA wrapped by MyA. The
   * code would look as follows:
   *
   * @code
   * //! The external class we want to wrap.
   * class ExtA
   * {
   *  public:
   *   const ExtB& get_b() { return b_; }
   *   void set_b(const ExtB& b) { b_ = b; }
   *  private:
   *   ExtB b_;
   * };
   *
   *  //! Our wrapper class for ExtA.
   * class MyA
   * {
   *  public:
   *   const MyB& get_b() { return b_view_.sync(ext_a_.get_b()); }
   *
   *  private:
   *   View<const MyB> b_view_;
   *   ExtA ext_a_;
   * };
   * @endcode
   *
   * Essentially, this ensures that whenever we call get_b() on MyA, we get a view of the _current_
   * ExtB object from ExtA. In the example above, one could call ExtA::set_b() and a subsequent
   * call to MyA::get_b() would return a view of the new ExtB object.
   */
  template <typename WrapperType>
  class View
  {
   public:
    /**
     * Construct an empty view. You either need to assign a non-empty view to it or call
     * sync() to make it point to a valid object before using it.
     */
    View() = default;

    /**
     * Construct a view from a source object.
     */
    template <Viewable SourceType>
    View(SourceType& source)
    {
      sync(source);
    }

    /**
     * Copying the View creates a new View of the same underlying object.
     */
    View(const View& other) = default;
    View& operator=(const View& other) = default;

    View(View&& other) noexcept = default;
    View& operator=(View&& other) noexcept = default;

    ~View() = default;

    /**
     * Allow implicit conversion to the underlying type. This allows us to use the view as if it
     * were the underlying type. This is useful for passing the view to functions which expect the
     * underlying wrapper type.
     */
    operator WrapperType&() const { return underlying(); }

    /**
     * Explicitly ask for the viewed underlying wrapper. This is simply another way to access
     * the viewed object when implicit conversion is not happening automatically.
     */
    WrapperType& underlying() const
    {
      FOUR_C_ASSERT_ALWAYS(view_, "View is not viewing anything.");
      return *view_;
    }

    /**
     * @brief Ensure that the View is referencing the @p source object.
     *
     * This method exists for a typical use case where we want to keep a view of a source object
     * up-to-date with changes in another object. If the @p source is not the same as the viewed
     * object, this View is updated to point to the new @p source object.
     *
     * @return The viewed underlying wrapper which is behaving like a view of the @p source object.
     */
    template <Viewable SourceType>
    WrapperType& sync(SourceType& source)
    {
      if (viewed_object_ != &source)
      {
        view_ = WrapperType::create_view(source);
        viewed_object_ = &source;
      }
      return *view_;
    }

    /**
     * Invalidate the view, so that it no longer views any object. You need to call sync()  or
     * assign a new object to the view before using it again.
     *
     * @note This is useful in the implementation of wrappers where we want to trigger a re-sync of
     * the view.
     */
    void invalidate()
    {
      view_.reset();
      viewed_object_ = nullptr;
    }

   private:
    //! Source content wrapped in our own type.
    //! We need to share ownership so we can make copies of the view (which refer to the same
    //! underlying object).
    std::shared_ptr<WrapperType> view_;

    const void* viewed_object_{nullptr};
  };


  // Deduction guide
  template <typename SourceType>
  View(SourceType&) -> View<Internal::EnableViewForWithQualifiers<SourceType>>;

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE


#endif
