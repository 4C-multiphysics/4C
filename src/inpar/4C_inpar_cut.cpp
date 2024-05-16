/*----------------------------------------------------------------------*/
/*! \file
\brief Input parameters for cut library

\level 2


*/

/*----------------------------------------------------------------------*/

#include "4C_inpar_cut.hpp"

#include "4C_io_condition_definition.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN



void INPAR::CUT::SetValidParameters(Teuchos::RCP<Teuchos::ParameterList> list)
{
  using namespace INPUT;
  using Teuchos::setStringToIntegralParameter;
  using Teuchos::tuple;

  Teuchos::ParameterList& cut_general = list->sublist("CUT GENERAL", false, "");

  // Intersection precision (double or cln)
  setStringToIntegralParameter<int>("KERNEL_INTERSECTION_FLOATTYPE", "double",
      "The floattype of the cut surface-edge intersection", tuple<std::string>("cln", "double"),
      tuple<int>(INPAR::CUT::floattype_cln, INPAR::CUT::floattype_double), &cut_general);

  // Computing disctance surface to point precision (double or cln)
  setStringToIntegralParameter<int>("KERNEL_DISTANCE_FLOATTYPE", "double",
      "The floattype of the cut distance computation", tuple<std::string>("cln", "double"),
      tuple<int>(INPAR::CUT::floattype_cln, INPAR::CUT::floattype_double), &cut_general);

  // A general floattype for CORE::GEO::CUT::Position for Embedded Elements (ComputeDistance)
  // If specified this floattype is used for all computations of CORE::GEO::CUT::Position with
  // embedded elements
  setStringToIntegralParameter<int>("GENERAL_POSITON_DISTANCE_FLOATTYPE", "none",
      "A general floattype for CORE::GEO::CUT::Position for Embedded Elements (ComputeDistance)",
      tuple<std::string>("none", "cln", "double"),
      tuple<int>(
          INPAR::CUT::floattype_none, INPAR::CUT::floattype_cln, INPAR::CUT::floattype_double),
      &cut_general);

  // A general floattype for CORE::GEO::CUT::Position for Elements (ComputePosition)
  // If specified this floattype is used for all computations of CORE::GEO::CUT::Position
  setStringToIntegralParameter<int>("GENERAL_POSITON_POSITION_FLOATTYPE", "none",
      "A general floattype for CORE::GEO::CUT::Position Elements (ComputePosition)",
      tuple<std::string>("none", "cln", "double"),
      tuple<int>(
          INPAR::CUT::floattype_none, INPAR::CUT::floattype_cln, INPAR::CUT::floattype_double),
      &cut_general);

  // Specifiy which Referenceplanes are used in DirectDivergence
  setStringToIntegralParameter<int>("DIRECT_DIVERGENCE_REFPLANE", "all",
      "Specifiy which Referenceplanes are used in DirectDivergence",
      tuple<std::string>("all", "diagonal_side", "facet", "diagonal", "side", "none"),
      tuple<int>(INPAR::CUT::DirDiv_refplane_all, INPAR::CUT::DirDiv_refplane_diagonal_side,
          INPAR::CUT::DirDiv_refplane_facet, INPAR::CUT::DirDiv_refplane_diagonal,
          INPAR::CUT::DirDiv_refplane_side, INPAR::CUT::DirDiv_refplane_none),
      &cut_general);

  // Specifiy is Cutsides are triangulated
  CORE::UTILS::BoolParameter(
      "SPLIT_CUTSIDES", "Yes", "Split Quad4 CutSides into Tri3-Subtriangles?", &cut_general);

  // Do the Selfcut before standard CUT
  CORE::UTILS::BoolParameter("DO_SELFCUT", "Yes", "Do the SelfCut?", &cut_general);

  // Do meshcorrection in Selfcut
  CORE::UTILS::BoolParameter(
      "SELFCUT_DO_MESHCORRECTION", "Yes", "Do meshcorrection in the SelfCut?", &cut_general);

  // Selfcut meshcorrection multiplicator
  CORE::UTILS::IntParameter("SELFCUT_MESHCORRECTION_MULTIPLICATOR", 30,
      "ISLANDS with maximal size of the bounding box of h*multiplacator will be removed in the "
      "meshcorrection",
      &cut_general);

  // Cubaturedegree utilized for the numerical integration on the CUT BoundaryCells.
  CORE::UTILS::IntParameter("BOUNDARYCELL_CUBATURDEGREE", 20,
      "Cubaturedegree utilized for the numerical integration on the CUT BoundaryCells.",
      &cut_general);
}

FOUR_C_NAMESPACE_CLOSE
