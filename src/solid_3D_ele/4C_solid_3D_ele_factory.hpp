#ifndef FOUR_C_SOLID_3D_ELE_FACTORY_HPP
#define FOUR_C_SOLID_3D_ELE_FACTORY_HPP

#include "4C_config.hpp"

#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_inpar_structure.hpp"
#include "4C_solid_3D_ele_calc_displacement_based.hpp"
#include "4C_solid_3D_ele_calc_displacement_based_linear_kinematics.hpp"
#include "4C_solid_3D_ele_calc_eas.hpp"
#include "4C_solid_3D_ele_calc_fbar.hpp"
#include "4C_solid_3D_ele_calc_mulf.hpp"
#include "4C_solid_3D_ele_calc_mulf_fbar.hpp"
#include "4C_solid_3D_ele_calc_shell_ans.hpp"
#include "4C_solid_3D_ele_factory_lib.hpp"
#include "4C_solid_3D_ele_properties.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Solid::ELEMENTS
{
  enum class EasType;
}
namespace Discret::ELEMENTS
{
  namespace Internal
  {
    using ImplementedSolidCellTypes = Core::FE::CelltypeSequence<Core::FE::CellType::hex8,
        Core::FE::CellType::hex18, Core::FE::CellType::hex20, Core::FE::CellType::hex27,
        Core::FE::CellType::nurbs27, Core::FE::CellType::tet4, Core::FE::CellType::tet10,
        Core::FE::CellType::wedge6, Core::FE::CellType::pyramid5>;

    using DisplacementBasedEvaluators =
        Core::FE::apply_celltype_sequence<DisplacementBasedSolidIntegrator,
            ImplementedSolidCellTypes>;

    using DisplacementBasedLinearKinematicsEvaluators =
        Core::FE::apply_celltype_sequence<DisplacementBasedLinearKinematicsSolidIntegrator,
            ImplementedSolidCellTypes>;

    using FbarEvaluators = Core::FE::apply_celltype_sequence<FBarSolidIntegrator,
        Core::FE::CelltypeSequence<Core::FE::CellType::hex8, Core::FE::CellType::pyramid5>>;
    using EASEvaluators = Core::FE::BaseTypeList<
        SolidEleCalcEas<Core::FE::CellType::hex8, Solid::ELEMENTS::EasType::eastype_h8_9,
            Inpar::Solid::KinemType::nonlinearTotLag>,
        SolidEleCalcEas<Core::FE::CellType::hex8, Solid::ELEMENTS::EasType::eastype_h8_21,
            Inpar::Solid::KinemType::nonlinearTotLag>,
        SolidEleCalcEas<Core::FE::CellType::hex8, Solid::ELEMENTS::EasType::eastype_sh8_7,
            Inpar::Solid::KinemType::nonlinearTotLag>,
        SolidEleCalcEas<Core::FE::CellType::hex8, Solid::ELEMENTS::EasType::eastype_h8_9,
            Inpar::Solid::KinemType::linear>,
        SolidEleCalcEas<Core::FE::CellType::hex8, Solid::ELEMENTS::EasType::eastype_h8_21,
            Inpar::Solid::KinemType::linear>>;
    using MulfEvaluators =
        Core::FE::apply_celltype_sequence<MulfSolidIntegrator, ImplementedSolidCellTypes>;
    using FBarMulfEvaluators = Core::FE::apply_celltype_sequence<MulfFBarSolidIntegrator,
        Core::FE::CelltypeSequence<Core::FE::CellType::hex8, Core::FE::CellType::pyramid5>>;

    using SolidShellEvaluators = Core::FE::apply_celltype_sequence<ANSSolidShellIntegrator,
        Core::FE::CelltypeSequence<Core::FE::CellType::hex8>>;

    using SolidEvaluators = Core::FE::Join<DisplacementBasedEvaluators,
        DisplacementBasedLinearKinematicsEvaluators, FbarEvaluators, EASEvaluators, MulfEvaluators,
        FBarMulfEvaluators, SolidShellEvaluators>;
  }  // namespace Internal

  using SolidCalcVariant = CreateVariantType<Internal::SolidEvaluators>;

  // forward declaration
  class SolidEleCalcInterface;
  class Solid;

  SolidCalcVariant create_solid_calculation_interface(Core::FE::CellType celltype,
      const Discret::ELEMENTS::SolidElementProperties& element_properties);

}  // namespace Discret::ELEMENTS


FOUR_C_NAMESPACE_CLOSE

#endif
