#ifndef FOUR_C_STRUCTURE_NEW_NLN_SOLVER_UZAWA_HPP
#define FOUR_C_STRUCTURE_NEW_NLN_SOLVER_UZAWA_HPP

#include "4C_config.hpp"

#include "4C_structure_new_nln_solver_generic.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  namespace Nln
  {
    namespace SOLVER
    {
      // Can be deleted in an upcoming commit, since unnecessary/deprecated   Hiermeier 01/12/2015
      class Uzawa : public Generic
      {
       public:
        Uzawa(){};
      };
    }  // namespace SOLVER
  }    // namespace Nln
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
