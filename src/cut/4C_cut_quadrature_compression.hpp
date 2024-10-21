#ifndef FOUR_C_CUT_QUADRATURE_COMPRESSION_HPP
#define FOUR_C_CUT_QUADRATURE_COMPRESSION_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_linalg_serialdensevector.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SerialDenseMatrix;
}

namespace Cut
{
  class VolumeCell;


  class QuadratureCompression
  {
   public:
    QuadratureCompression();

    bool perform_compression_of_quadrature(
        Core::FE::GaussPointsComposite& gin, Cut::VolumeCell* vc);

    // Core::FE::GaussIntegration get_compressed_quadrature(){ return *gout_; }
    Teuchos::RCP<Core::FE::GaussPoints> get_compressed_quadrature() { return gout_; }

   private:
    void form_matrix_system(Core::FE::GaussPointsComposite& gin,
        Core::LinAlg::SerialDenseMatrix& mat, Core::LinAlg::SerialDenseVector& rhs);

    void teuchos_gels(Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>& mat,
        Teuchos::RCP<Core::LinAlg::SerialDenseVector>& rhs, Core::LinAlg::SerialDenseVector& sol);

    /*!
    \brief Solve the under/over determined system by performing QR decomposition, achived using
    Teuchos framework
     */
    void qr_decomposition_teuchos(Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>& mat,
        Teuchos::RCP<Core::LinAlg::SerialDenseVector>& rhs,
        Teuchos::RCP<Core::LinAlg::SerialDenseVector>& sol);

    /*!
    \brief Solve the under/over determined system by performing QR decomposition, achived using
    LAPACK
     */
    void qr_decomposition_lapack(Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>& mat,
        Teuchos::RCP<Core::LinAlg::SerialDenseVector>& rhs, Core::LinAlg::SerialDenseVector& sol);

    bool compress_leja_points(Core::FE::GaussPointsComposite& gin,
        Teuchos::RCP<Core::LinAlg::SerialDenseMatrix>& mat,
        Teuchos::RCP<Core::LinAlg::SerialDenseVector>& rhs,
        Teuchos::RCP<Core::LinAlg::SerialDenseVector>& sol);

    void compute_and_print_error(Core::FE::GaussPointsComposite& gin,
        Core::LinAlg::SerialDenseVector& rhs, Core::LinAlg::SerialDenseVector& sol,
        std::vector<int>& work, int& na);

    void get_pivotal_rows(Core::LinAlg::IntSerialDenseVector& work_temp, std::vector<int>& work);

    bool is_this_value_already_in_dense_vector(
        int& input, std::vector<int>& vec, int upper_range, int& index);

    Teuchos::RCP<Core::FE::GaussPoints> form_new_quadrature_rule(
        Core::FE::GaussPointsComposite& gin, Core::LinAlg::SerialDenseVector& sol,
        std::vector<int>& work, int& na);
    int get_correct_index(int& input, std::vector<int>& vec, int upper_range);

    void write_compressed_quadrature_gmsh(Core::FE::GaussPointsComposite& gin, Cut::VolumeCell* vc);

    void integrate_predefined_polynomials(Core::FE::GaussPointsComposite& gin);

    Teuchos::RCP<Core::FE::GaussPoints> gout_;
  };

}  // namespace Cut

FOUR_C_NAMESPACE_CLOSE

#endif
