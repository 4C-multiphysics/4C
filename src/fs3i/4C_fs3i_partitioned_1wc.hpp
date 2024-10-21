#ifndef FOUR_C_FS3I_PARTITIONED_1WC_HPP
#define FOUR_C_FS3I_PARTITIONED_1WC_HPP


#include "4C_config.hpp"

#include "4C_fs3i_partitioned.hpp"

FOUR_C_NAMESPACE_OPEN


namespace FS3I
{
  class PartFS3I1Wc : public PartFS3I
  {
   public:
    PartFS3I1Wc(const Epetra_Comm& comm);

    void init() override;

    void setup() override;

    void timeloop() override;

    void do_fsi_step();

    void do_scatra_step();

    void prepare_time_step() override;

    bool scatra_convergence_check(int itnum) override;
  };

}  // namespace FS3I

FOUR_C_NAMESPACE_CLOSE

#endif
