#ifndef FOUR_C_MORTAR_MATRIX_TRANSFORM_HPP
#define FOUR_C_MORTAR_MATRIX_TRANSFORM_HPP

#include "4C_config.hpp"

#include "4C_contact_utils.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_pairedvector.hpp"

#include <Epetra_Export.h>
#include <Epetra_Map.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SparseMatrix;
}  // namespace Core::LinAlg
namespace Mortar
{
  class MatrixRowColTransformer
  {
    typedef Core::Gen::Pairedvector<CONTACT::MatBlockType, Teuchos::RCP<Epetra_Export>>
        plain_block_export_pairs;

   public:
    typedef Core::Gen::Pairedvector<CONTACT::MatBlockType, Teuchos::RCP<Epetra_Map>*>
        plain_block_map_pairs;

   public:
    /// constructor
    MatrixRowColTransformer(const unsigned num_transformer);


    void init(const plain_block_map_pairs& redistributed_row,
        const plain_block_map_pairs& redistributed_column,
        const plain_block_map_pairs& unredistributed_row,
        const plain_block_map_pairs& unredistributed_column);

    void setup();

    Teuchos::RCP<Core::LinAlg::SparseMatrix> redistributed_to_unredistributed(
        const CONTACT::MatBlockType bt, const Core::LinAlg::SparseMatrix& src_mat);

    void redistributed_to_unredistributed(const CONTACT::MatBlockType bt,
        const Core::LinAlg::SparseMatrix& src_mat, Core::LinAlg::SparseMatrix& dst_mat);

    Teuchos::RCP<Core::LinAlg::SparseMatrix> unredistributed_to_redistributed(
        const CONTACT::MatBlockType bt, const Core::LinAlg::SparseMatrix& src_mat);

    void unredistributed_to_redistributed(const CONTACT::MatBlockType bt,
        const Core::LinAlg::SparseMatrix& src_mat, Core::LinAlg::SparseMatrix& dst_mat);

   private:
    void set_slave_map_pairs(const plain_block_map_pairs& redistributed_row,
        const plain_block_map_pairs& redistributed_column);

    void set_master_map_pairs(const plain_block_map_pairs& unredistributed_row,
        const plain_block_map_pairs& unredistributed_column);

    // hide empty constructor
    MatrixRowColTransformer();

    inline void throw_if_not_init() const
    {
      if (not isinit_) FOUR_C_THROW("Call init() first!");
    }

    inline void throw_if_not_init_and_setup() const
    {
      throw_if_not_init();
      if (not issetup_) FOUR_C_THROW("Call setup() first!");
    }

    void reset_exporter(Teuchos::RCP<Epetra_Export>& exporter) const;

   private:
    bool isinit_;
    bool issetup_;

    plain_block_export_pairs slave_to_master_;
    plain_block_export_pairs master_to_slave_;

    plain_block_map_pairs slave_row_;
    plain_block_map_pairs slave_col_;

    plain_block_map_pairs master_row_;
    plain_block_map_pairs master_col_;

  };  // class MatrixTransform

}  // namespace Mortar



FOUR_C_NAMESPACE_CLOSE

#endif
