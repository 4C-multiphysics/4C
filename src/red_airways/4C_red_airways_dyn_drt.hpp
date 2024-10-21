#ifndef FOUR_C_RED_AIRWAYS_DYN_DRT_HPP
#define FOUR_C_RED_AIRWAYS_DYN_DRT_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_io.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_red_airways_implicitintegration.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN



void dyn_red_airways_drt();
void redairway_tissue_dyn();

Teuchos::RCP<Airway::RedAirwayImplicitTimeInt> dyn_red_airways_drt(bool CoupledTo3D);


FOUR_C_NAMESPACE_CLOSE

#endif
