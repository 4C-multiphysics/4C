/*----------------------------------------------------------------------*/
/*! \file

\brief Write (time and space) averaged values to file for
turbulent flow over a periodic hill

o Create sets for various evaluation lines in domain
  (Construction based on a round robin communication pattern):
  - 21 lines in x2-direction
  - line along lower wall

  - generate 4 toggle vectors (u,v,w,p), for example

                            /  1  u dof in homogeneous plane
                 toggleu_  |
                            \  0  elsewhere

  - pointwise multiplication velnp.*velnp for second order
    moments

o values on lines are averaged in time over all steps between two
  outputs

Required parameters are the number of velocity degrees of freedom (3)
and the basename of the statistics outfile. These parameters are
expected to be contained in the fluid time integration parameter list
given on input.


\level 2

*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_FLUID_TURBULENCE_STATISTICS_PH_HPP
#define FOUR_C_FLUID_TURBULENCE_STATISTICS_PH_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Epetra_MpiComm.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN


namespace FLD
{
  class TurbulenceStatisticsPh
  {
   public:
    /*!
    \brief Standard Constructor (public)

        o Create sets for lines

    o Allocate distributed vector for squares

    */
    TurbulenceStatisticsPh(Teuchos::RCP<Core::FE::Discretization> actdis,
        Teuchos::ParameterList& params, const std::string& statistics_outfilename);

    /*!
    \brief Destructor

    */
    virtual ~TurbulenceStatisticsPh() = default;


    //! @name functions for averaging

    /*!
    \brief The values of velocity and its squared values are added to
    global vectors. This method allows to do the time average of the
    nodal values after a certain amount of timesteps.
    */
    void do_time_sample(Teuchos::RCP<Core::LinAlg::Vector<double>> velnp,
        Teuchos::RCP<Core::LinAlg::Vector<double>> stresses);

    /*!
    \brief Dump the result to file.

    step on input is used to print the timesteps which belong to the
    statistic to the file
    */

    void dump_statistics(int step);


   protected:
    /*!
    \brief sort criterium for double values up to a tolerance of 10-9

    This is used to create sets of doubles (e.g. coordinates)

    */
    class LineSortCriterion
    {
     public:
      bool operator()(const double& p1, const double& p2) const { return (p1 < p2 - 1E-9); }

     protected:
     private:
    };

   private:
    //! number of samples taken
    int numsamp_;

    //! number of coordinates in x1- and x2-direction
    int numx1coor_;
    int numx2coor_;

    //! number of locations in x1- and x2-direction for statistical evaluation
    int numx1statlocations_;

    //! The discretisation (required for nodes, dofs etc;)
    Teuchos::RCP<Core::FE::Discretization> discret_;

    //! parameter list
    Teuchos::ParameterList& params_;

    //! name of statistics output file, despite the ending
    const std::string statistics_outfilename_;

    //! pointer to vel/pres^2 field (space allocated in constructor)
    Teuchos::RCP<Core::LinAlg::Vector<double>> squaredvelnp_;

    //! toogle vectors: sums are computed by scalarproducts
    Teuchos::RCP<Core::LinAlg::Vector<double>> toggleu_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> togglev_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> togglew_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> togglep_;

    //! available x1- and x2-coordinates
    Teuchos::RCP<std::vector<double>> x1coordinates_;
    Teuchos::RCP<std::vector<double>> x2coordinates_;

    //! coordinates of locations in x1- and x2-direction for statistical evaluation
    Core::LinAlg::SerialDenseMatrix x1statlocations_;

    //! matrix for r-coordinates (columns are evaluation planes
    Core::LinAlg::SerialDenseMatrix x2statlocations_;

    //! set coordinates of locations in x1-direction for statistical evaluation
    Teuchos::RCP<std::vector<double>> x1setstatlocations_;

    //! coordinates in x1-direction for sampling velocity gradient at the middle bottom

    //! matrices containing values
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x1sumu_;
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x1sump_;
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x1sumf_;

    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x2sumu_;
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x2sumv_;
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x2sumw_;
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x2sump_;


    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x2sumsqu_;
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x2sumsqv_;
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x2sumsqw_;
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x2sumsqp_;

    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x2sumuv_;
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x2sumuw_;
    Teuchos::RCP<Core::LinAlg::SerialDenseMatrix> x2sumvw_;
  };

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
