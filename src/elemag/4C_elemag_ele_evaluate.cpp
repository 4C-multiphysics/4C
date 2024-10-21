#include "4C_elemag_diff_ele.hpp"
#include "4C_elemag_ele.hpp"
#include "4C_elemag_ele_factory.hpp"
#include "4C_elemag_ele_interface.hpp"
#include "4C_inpar_elemag.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::ELEMENTS::ElemagType::pre_evaluate(Core::FE::Discretization& dis,
    Teuchos::ParameterList& p, Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix1,
    Teuchos::RCP<Core::LinAlg::SparseOperator> systemmatrix2,
    Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector1,
    Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector2,
    Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector3)
{
  return;
}

/*---------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Discret::ELEMENTS::Elemag::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  Teuchos::RCP<Core::Mat::Material> mat = material();

  if (dynamic_cast<const Discret::ELEMENTS::ElemagDiff*>(this))
    return Discret::ELEMENTS::ElemagFactory::provide_impl(shape(), "diff")
        ->evaluate(
            this, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2, elevec3);
  else
    return Discret::ELEMENTS::ElemagFactory::provide_impl(shape(), "std")
        ->evaluate(
            this, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2, elevec3);

  return -1;
}

/*---------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Discret::ELEMENTS::Elemag::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  return 0;
}

FOUR_C_NAMESPACE_CLOSE
