#ifndef FOUR_C_XFEM_DOFSET_TRANSPARENT_INDEPENDENT_HPP
#define FOUR_C_XFEM_DOFSET_TRANSPARENT_INDEPENDENT_HPP

#include "4C_config.hpp"

#include "4C_fem_dofset_transparent_independent.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE


namespace Cut
{
  class CutWizard;
}

namespace XFEM
{
  /// Alias dofset that shares dof numbers with another dofset
  /*!
    A special set of degrees of freedom, implemented in order to assign the same degrees of freedom
    to nodes belonging to two discretizations. This way two discretizations can assemble into the
    same position of the system matrix. As internal variable it holds a source discretization
    (Constructor). If such a nodeset is assigned to a sub-discretization, its dofs are assigned
    according to the dofs of the source. The source discretization can be a xfem discretization. In
    this case this  should be called with a Fluidwizard not equal to zero to determine the  number
    of xfem dofs.

   */
  class XFEMTransparentIndependentDofSet
      : public virtual Core::DOFSets::TransparentIndependentDofSet
  {
   public:
    /*!
      \brief Standard Constructor
     */
    explicit XFEMTransparentIndependentDofSet(Teuchos::RCP<Core::FE::Discretization> sourcedis,
        bool parallel, Teuchos::RCP<Cut::CutWizard> wizard);



   protected:
    int num_dof_per_node(const Core::Nodes::Node& node) const override;


   private:
    Teuchos::RCP<Cut::CutWizard> wizard_;
  };
}  // namespace XFEM

FOUR_C_NAMESPACE_CLOSE

#endif
