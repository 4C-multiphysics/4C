/*-----------------------------------------------------------------------*/
/*! \file

\brief

\level 1

*/

/*----------------------------------------------------------------------*/

#ifndef FOUR_C_INPAR_VOLMORTAR_HPP
#define FOUR_C_INPAR_VOLMORTAR_HPP


/*----------------------------------------------------------------------*/
#include "baci_config.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>

BACI_NAMESPACE_OPEN

namespace INPAR
{
  namespace VOLMORTAR
  {
    // Type of integration procedure
    enum IntType
    {
      inttype_segments,  ///< cut procedure of volume meshes
      inttype_elements   ///< fast, elementwise integration
    };

    // Type of cut procedure
    enum CutType
    {
      cuttype_directdivergence,  ///< direct divergence for integration
      cuttype_tessellation       ///< tessellation of volume meshes
    };

    // Type weighting function for quadr. problems
    enum DualQuad
    {
      dualquad_no_mod,   ///< no modification
      dualquad_lin_mod,  ///< linear modification
      dualquad_quad_mod  ///< quadr. modification
    };

    // Type of coupling
    enum CouplingType
    {
      couplingtype_volmortar,  ///< volmortar
      couplingtype_coninter    ///< consist. interpolation
    };

    // Type of coupling
    enum Shapefcn
    {
      shape_dual,  ///< dual shape functions
      shape_std    ///< std. shape functions --> lumped
    };

    /// set the volmortar parameters
    void SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list);

  }  // namespace VOLMORTAR
}  // namespace INPAR

BACI_NAMESPACE_CLOSE

#endif  // INPAR_VOLMORTAR_H
