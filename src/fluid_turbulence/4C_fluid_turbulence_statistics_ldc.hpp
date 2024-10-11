/*----------------------------------------------------------------------*/
/*! \file

\brief Write (time and space) averaged values to file.

o Create sets for centerlines in x1-, x2- and x3-direction
  (Construction based on a round robin communication pattern)

o loop nodes closest to centerlines

  - generate 4 toggle vectors (u,v,w,p), for example

                            /  1  u dof in homogeneous plane
                 toggleu_  |
                            \  0  elsewhere

  - pointwise multiplication velnp.*velnp for second order
    moments

o values on centerlines are averaged in time over all steps between two
  outputs

Required parameters are the number of velocity degrees of freedom (3)
and the basename of the statistics outfile. These parameters are
expected to be contained in the fluid time integration parameter list
given on input.

This method is intended to be called every upres_ steps during fluid
output.


\level 2

*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_FLUID_TURBULENCE_STATISTICS_LDC_HPP
#define FOUR_C_FLUID_TURBULENCE_STATISTICS_LDC_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_inpar_fluid.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Epetra_MpiComm.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class DiscretizationReader;
  class DiscretizationWriter;
}  // namespace Core::IO

namespace FLD
{
  class TurbulenceStatisticsLdc
  {
   public:
    /*!
    \brief Standard Constructor (public)

        o Create sets for centerlines in x1-, x2- and x3-direction

    o Allocate distributed vector for squares

    */
    TurbulenceStatisticsLdc(Teuchos::RCP<Core::FE::Discretization> actdis,
        Teuchos::ParameterList& params, const std::string& statistics_outfilename);

    /*!
    \brief Destructor

    */
    virtual ~TurbulenceStatisticsLdc() = default;


    //! @name functions for averaging

    /*!
    \brief The values of velocity, pressure and its squared values are
    added to global vectors. This method allows to do the time average
    of the nodal values after a certain amount of timesteps.
    */
    void do_time_sample(Core::LinAlg::Vector<double>& velnp);

    /*!
    \brief The values of velocity, pressure, temperature and its squared
    values are added to global vectors. This method allows to do the time
    average of the nodal values after a certain amount of timesteps.
    */
    void do_loma_time_sample(Core::LinAlg::Vector<double>& velnp,
        Core::LinAlg::Vector<double>& scanp, Core::LinAlg::Vector<double>& force,
        const double eosfac);

    /*!
    \brief Dump the result to file for incompressible flow.

    step on input is used to print the timesteps which belong to the
    statistic to the file
    */

    void dump_statistics(int step);

    /*!
    \brief Dump the result to file for low-Mach-number flow.

    step on input is used to print the timesteps which belong to the
    statistic to the file
    */

    void dump_loma_statistics(int step);

    /*!
    \brief Reset sums and number of samples to zero
    */

    void clear_statistics();


    /*!
    \brief Input of statistics data after restart
    */

    void read_restart(Core::IO::DiscretizationReader& reader);

    /*!
    \brief Write output file of statistics data to allow restart
    */
    void write_restart(Core::IO::DiscretizationWriter& writer);

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

    //! bounds for extension of cavity in x1-direction
    double x1min_;
    double x1max_;
    //! bounds for extension of cavity in x2-direction
    double x2min_;
    double x2max_;
    //! bounds for extension of cavity in x3-direction
    double x3min_;
    double x3max_;

    //! The discretisation (required for nodes, dofs etc;)
    Teuchos::RCP<Core::FE::Discretization> discret_;

    //! parameter list
    Teuchos::ParameterList& params_;

    //! name of statistics output file, despite the ending
    const std::string statistics_outfilename_;

    //! toogle vectors: sums are computed by scalarproducts
    Teuchos::RCP<Core::LinAlg::Vector<double>> toggleu_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> togglev_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> togglew_;
    Teuchos::RCP<Core::LinAlg::Vector<double>> togglep_;

    //! the coordinates of the centerlines in x1-, x2- and x3-direction
    Teuchos::RCP<std::vector<double>> x1coordinates_;
    Teuchos::RCP<std::vector<double>> x2coordinates_;
    Teuchos::RCP<std::vector<double>> x3coordinates_;

    //! sum over u (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumu_;
    Teuchos::RCP<std::vector<double>> x2sumu_;
    Teuchos::RCP<std::vector<double>> x3sumu_;
    //! sum over v (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumv_;
    Teuchos::RCP<std::vector<double>> x2sumv_;
    Teuchos::RCP<std::vector<double>> x3sumv_;
    //! sum over w (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumw_;
    Teuchos::RCP<std::vector<double>> x2sumw_;
    Teuchos::RCP<std::vector<double>> x3sumw_;
    //! sum over p (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sump_;
    Teuchos::RCP<std::vector<double>> x2sump_;
    Teuchos::RCP<std::vector<double>> x3sump_;
    //! sum over density (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumrho_;
    Teuchos::RCP<std::vector<double>> x2sumrho_;
    Teuchos::RCP<std::vector<double>> x3sumrho_;
    //! sum over T (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sum_t_;
    Teuchos::RCP<std::vector<double>> x2sum_t_;
    Teuchos::RCP<std::vector<double>> x3sum_t_;

    //! sum over u^2 (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumsqu_;
    Teuchos::RCP<std::vector<double>> x2sumsqu_;
    Teuchos::RCP<std::vector<double>> x3sumsqu_;
    //! sum over v^2 (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumsqv_;
    Teuchos::RCP<std::vector<double>> x2sumsqv_;
    Teuchos::RCP<std::vector<double>> x3sumsqv_;
    //! sum over w^2 (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumsqw_;
    Teuchos::RCP<std::vector<double>> x2sumsqw_;
    Teuchos::RCP<std::vector<double>> x3sumsqw_;
    //! sum over p^2 (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumsqp_;
    Teuchos::RCP<std::vector<double>> x2sumsqp_;
    Teuchos::RCP<std::vector<double>> x3sumsqp_;
    //! sum over density^2 (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumsqrho_;
    Teuchos::RCP<std::vector<double>> x2sumsqrho_;
    Teuchos::RCP<std::vector<double>> x3sumsqrho_;
    //! sum over T^2 (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumsq_t_;
    Teuchos::RCP<std::vector<double>> x2sumsq_t_;
    Teuchos::RCP<std::vector<double>> x3sumsq_t_;

    //! sum over uv (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumuv_;
    Teuchos::RCP<std::vector<double>> x2sumuv_;
    Teuchos::RCP<std::vector<double>> x3sumuv_;
    //! sum over uw (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumuw_;
    Teuchos::RCP<std::vector<double>> x2sumuw_;
    Teuchos::RCP<std::vector<double>> x3sumuw_;
    //! sum over vw (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumvw_;
    Teuchos::RCP<std::vector<double>> x2sumvw_;
    Teuchos::RCP<std::vector<double>> x3sumvw_;
    //! sum over uT (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumu_t_;
    Teuchos::RCP<std::vector<double>> x2sumu_t_;
    Teuchos::RCP<std::vector<double>> x3sumu_t_;
    //! sum over vT (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumv_t_;
    Teuchos::RCP<std::vector<double>> x2sumv_t_;
    Teuchos::RCP<std::vector<double>> x3sumv_t_;
    //! sum over wT (over the centerlines in x1-, x2- and x3-direction)
    Teuchos::RCP<std::vector<double>> x1sumw_t_;
    Teuchos::RCP<std::vector<double>> x2sumw_t_;
    Teuchos::RCP<std::vector<double>> x3sumw_t_;
  };

}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
