#include "4C_contact_nitsche_strategy_ssi_elch.hpp"

FOUR_C_NAMESPACE_OPEN

/*------------------------------------------------------------------------*
/-------------------------------------------------------------------------*/
void CONTACT::NitscheStrategySsiElch::integrate(const CONTACT::ParamsInterface& cparams)
{
  CONTACT::NitscheStrategy::integrate(cparams);

  fs_ = create_rhs_block_ptr(CONTACT::VecBlockType::elch);
  kss_ = create_matrix_block_ptr(CONTACT::MatBlockType::elch_elch);
  ksd_ = create_matrix_block_ptr(CONTACT::MatBlockType::elch_displ);
  kds_ = create_matrix_block_ptr(CONTACT::MatBlockType::displ_elch);
}

FOUR_C_NAMESPACE_CLOSE
