#ifndef FOUR_C_BEAM3_TRIAD_INTERPOLATION_HPP
#define FOUR_C_BEAM3_TRIAD_INTERPOLATION_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"

#include <Teuchos_RCP.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace LargeRotations
{
  /**
   * \brief abstract base class for a triad interpolation scheme
   */
  template <typename T>
  class TriadInterpolation
  {
   public:
    //! @name Constructors and destructors and related methods

    /** \brief Standard Constructor
     *
     *  \author grill
     *  \date 01/17 */
    TriadInterpolation();

    /** \brief Destructor
     *
     *  \author grill
     *  \date 01/17 */
    virtual ~TriadInterpolation() = default;

    /** \brief return appropriate derived (templated) class (acts as a simple factory)
     *
     *  \author grill
     *  \date 01/17 */
    static Teuchos::RCP<TriadInterpolation<T>> create(unsigned int numnodes);
    //@}


    //! @name Public evaluation methods

    /** \brief reset interpolation scheme with nodal quaternions
     *
     *  \author grill
     *  \date 01/17 */
    virtual void reset(std::vector<Core::LinAlg::Matrix<4, 1, T>> const& nodal_quaternions) = 0;

    /** \brief reset interpolation scheme with nodal triads
     *
     *  \author grill
     *  \date 01/17 */
    virtual void reset(std::vector<Core::LinAlg::Matrix<3, 3, T>> const& nodal_triads) = 0;

    /** \brief compute the interpolated triad at any point \xi \in [-1,1] in parameter space
     *
     *  \author grill
     *  \date 01/17 */
    virtual void get_interpolated_triad_at_xi(
        Core::LinAlg::Matrix<3, 3, T>& triad, const double xi) const = 0;

    /** \brief compute the interpolated quaternion at any point \xi \in [-1,1] in parameter space
     *
     *  \author grill
     *  \date 01/17 */
    virtual void get_interpolated_quaternion_at_xi(
        Core::LinAlg::Matrix<4, 1, T>& quaternion, const double xi) const = 0;
    //@}

   private:
    //! @name Private evaluation methods

    //@}

   private:
    //! @name member variables

    //@}
  };

}  // namespace LargeRotations

FOUR_C_NAMESPACE_CLOSE

#endif
