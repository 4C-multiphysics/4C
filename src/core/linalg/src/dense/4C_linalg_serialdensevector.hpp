#ifndef FOUR_C_LINALG_SERIALDENSEVECTOR_HPP
#define FOUR_C_LINALG_SERIALDENSEVECTOR_HPP


#include "4C_config.hpp"

#include <Teuchos_SerialDenseVector.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /*!
 \brief A class that wraps Teuchos::SerialDenseVector

      This is done in favor of typedef to allow forward declaration
 */
  class SerialDenseVector : public Teuchos::SerialDenseVector<int, double>
  {
   public:
    /// Base type definition
    using Base = Teuchos::SerialDenseVector<int, double>;

    /// Using the base class constructor
    using Base::SerialDenseVector;
  };

  // type definition for serial integer vector
  typedef Teuchos::SerialDenseVector<int, int> IntSerialDenseVector;

  /*!
    \brief Update vector components with scaled values of a,
           b = alpha*a + beta*b
    */
  void update(double alpha, const SerialDenseVector& a, double beta, SerialDenseVector& b);

  // wrapper function to compute Norm of vector
  double norm2(const SerialDenseVector& v);

  // output stream operator
  inline std::ostream& operator<<(std::ostream& out, const SerialDenseVector& vec)
  {
    vec.print(out);
    return out;
  }

  // output stream operator
  inline std::ostream& operator<<(std::ostream& out, const IntSerialDenseVector& vec)
  {
    vec.print(out);
    return out;
  }
}  // namespace Core::LinAlg


FOUR_C_NAMESPACE_CLOSE

#endif
