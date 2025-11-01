// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_MANAGER_HPP
#define FOUR_C_CONSTRAINT_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::IO
{
  class DiscretizationReader;
}

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class SparseOperator;
  class MultiMapExtractor;
}  // namespace Core::LinAlg

namespace Constraints
{
  // forward declarations
  class Constraint;
  class ConstraintPenalty;
  class MPConstraint3;
  class MPConstraint3Penalty;
  class MPConstraint2;
  class Monitor;
  class ConstraintDofSet;

  /*!
  \brief Class controlling constraints and containing the necessary data
  */
  class ConstrManager
  {
   public:
    //! Constructor of constraint manager, allocating the constraints
    ConstrManager();


    //! initialize this class
    void init(
        std::shared_ptr<Core::FE::Discretization> discr, const Teuchos::ParameterList& params);

    /*! \brief Setup all class internal objects and members

     setup() is not supposed to have any input arguments !

     Must only be called after init().

     Construct all objects depending on the parallel distribution and
     relying on valid maps like, e.g. the state vectors, system matrices, etc.

     Call all setup() routines on previously initialized internal objects and members.

    \note Must only be called after parallel (re-)distribution of discretizations is finished !
          Otherwise, e.g. vectors may have wrong maps.

    \warning Here setup() needs to get an input argument because of the faulty implementation
             of this manager class. This needs to be fixed.

    \warning none
    \return void

    */
    void setup(
        std::shared_ptr<const Core::LinAlg::Vector<double>> disp, Teuchos::ParameterList params);

    /*!
      \brief Change stiffness matrix and force vector according to the constraints.
      Values of lagrange multiplier are taken from intern variable.
      Difference between current and prescribed values is calculated and stored as well.
    */
    void evaluate_force_stiff(const double time,  ///< time at end of time step
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            displast,  ///< displacement at beginning of time step
        std::shared_ptr<const Core::LinAlg::Vector<double>>
            disp,                                             ///< displacement at end of time step
        std::shared_ptr<Core::LinAlg::Vector<double>> fint,   ///< vector of internal forces
        std::shared_ptr<Core::LinAlg::SparseOperator> stiff,  ///< stiffness matrix
        Teuchos::ParameterList scalelist);

    /*!
     \brief Return norm of difference between actual and constraint values
    */
    double get_error_norm() const
    {
      double foo;
      constrainterr_->norm_2(&foo);
      return foo;
    };

    /*!
         \brief Return number of constraints
    */
    int get_number_of_constraints() const { return num_constr_id_; };

    /*!
     \brief Scale all lagrange multipliers by a double d
    */
    void scale_lagr_mult(double d  ///< scale factor
    )
    {
      lagr_mult_vec_->scale(d);
      return;
    };

    /*!
         \brief Update constraint variables
    */
    void update();

    /*!
         \brief Update lagrange multiplier \f$\lambda_{n+1}=\lambda_{n}+factor*\f$(volerr)
    */
    void update_lagr_mult(double factor);

    /// Add a vector as residual increment to the vector of Lagrange multipliers
    void update_lagr_mult(Core::LinAlg::Vector<double>& vect  ///< vector to add
    );

    /// Add a vector as total increment to the vector of Lagrange multipliers
    void update_tot_lagr_mult(Core::LinAlg::Vector<double>& vect  ///< vector to add
    );

    /*!
         \brief Compute difference between current and prescribed values at a given time and a given
       displacement
    */
    void compute_error(double time,  ///< time, at which the error is to compute at
        std::shared_ptr<Core::LinAlg::Vector<double>>
            disp  ///< displacement vector at the given time
    );

    /*!
         \brief Return differences between prescribed and actual value of constraint number i
    */
    double get_error(int i  ///< ID of constraint of interest
    ) const
    {
      return (*constrainterr_).local_values_as_span()[i];
    }

    /// return vector of differences between prescribed and actual values
    std::shared_ptr<Core::LinAlg::Vector<double>> get_error() const { return constrainterr_; }

    /*!
     \brief Return map that determined distribution of constraints and lagrange
     multiplier over processors
    */
    std::shared_ptr<Core::LinAlg::Map> get_constraint_map() const { return constrmap_; };

    //! Return the additional rectangular matrix, constructed for lagrange multiplier evaluation
    std::shared_ptr<Core::LinAlg::SparseOperator> get_constr_matrix()  // const
    {
      return constr_matrix_;
    };

    /*!
      \brief Return lagrange multiplier for constraint i
    */
    double get_lagr_mult(int i  ///< ID of constraint of interest
    ) const
    {
      return (*lagr_mult_vec_).local_values_as_span()[i];
    };

    /*!
      \brief Return lagrange multiplier vector
    */
    std::shared_ptr<Core::LinAlg::Vector<double>> get_lagr_mult_vector() const
    {
      return lagr_mult_vec_;
    };

    /*!
      \brief Return lagrange multiplier of last converged step
    */
    std::shared_ptr<Core::LinAlg::Vector<double>> get_lagr_mult_vector_old() const
    {
      return lagr_mult_vec_old_;
    };

    /*!
     \brief Return if there are constraints
    */
    bool have_constraint() const { return haveconstraint_; };

    /*!
     \brief Return if there are constraints
    */
    bool have_constraint_lagr() const { return havelagrconstr_; };

    /*!
     \brief Return if there are constraints
    */
    bool have_constraint_pen() const { return havepenaconstr_; };

    /*!
       \brief Return if there are monitors
    */
    bool have_monitor() const { return havemonitor_; };

    /*!
     \brief Read restart information
    */
    void read_restart(Core::IO::DiscretizationReader& reader, const double& time);

    /*!
     \brief Return current value
    */
    double get_curr_value(int i  ///< ID of constraint of interest
    ) const
    {
      return (*actvalues_).local_values_as_span()[i];
    };

    /*!
         \brief Print out the values of current monitor values
     */
    void print_monitor_values() const;

    /*!
       \brief Compute values described by a monitor boundary condition
    */
    void compute_monitor_values(
        std::shared_ptr<Core::LinAlg::Vector<double>> disp  ///< current displacement
    );

    /*!
       \brief Compute values described by a monitor boundary condition
    */
    void compute_monitor_values(const Core::LinAlg::Vector<double>& disp  ///< current displacement
    );

    /// Reset reference base values for restart computations
    void set_ref_base_values(
        Core::LinAlg::Vector<double>& newrefvals,  ///< new reference base values
        const double& time                         ///< current time
    );

    /// Reset lagrange multipliers
    void set_lagr_mult_vector(
        Core::LinAlg::Vector<double>& newlagrmult  ///< new lagrange multipliers
    )
    {
      lagr_mult_vec_->update(1.0, newlagrmult, 0.0);
      lagr_mult_vec_old_->update(1.0, newlagrmult, 0.0);
      return;
    }

    /// Return Reference base values to write restart
    std::shared_ptr<Core::LinAlg::Vector<double>> get_ref_base_values() const
    {
      return refbasevalues_;
    }

    //! switch constraint matrix to block matrix
    void use_block_matrix(std::shared_ptr<const Core::LinAlg::MultiMapExtractor> domainmaps,
        std::shared_ptr<const Core::LinAlg::MultiMapExtractor> rangemaps);


   private:
    // don't want = operator, cctor and destructor
    ConstrManager operator=(const ConstrManager& old);
    ConstrManager(const ConstrManager& old);

    /// Build Monitor type Vector
    void build_moni_type();

    std::shared_ptr<Core::FE::Discretization>
        actdisc_;  ///< discretization, elements to constraint live in
    std::shared_ptr<ConstraintDofSet>
        constrdofset_;                              ///< degrees of freedom of lagrange multipliers
    std::shared_ptr<Core::LinAlg::Map> constrmap_;  ///< unique map of constraint values
    std::shared_ptr<Core::LinAlg::Map> redconstrmap_;  ///< fully redundant map of constraint values
    std::shared_ptr<Core::LinAlg::Export>
        conimpo_;  ///< importer for fully redundant constraint vector into distributed one
    std::shared_ptr<Core::LinAlg::Map> monitormap_;  ///< unique map of monitor values
    std::shared_ptr<Core::LinAlg::Map> redmonmap_;   ///< fully redundant map of monitor values
    std::shared_ptr<Core::LinAlg::Export>
        monimpo_;  ///< importer for fully redundant monitor vector into distributed one
    std::shared_ptr<Core::LinAlg::Vector<double>>
        referencevalues_;  ///< reference at current time step to constrain values to
    std::shared_ptr<Core::LinAlg::Vector<double>>
        refbasevalues_;  ///< reference base values at activation time of constrained structures
    std::shared_ptr<Core::LinAlg::Vector<double>>
        actvalues_;  ///< current values of constrained structures
    std::shared_ptr<Core::LinAlg::Vector<double>>
        constrainterr_;  ///< vector with deflection between reference and current values
    std::shared_ptr<Core::LinAlg::Vector<double>>
        monitorvalues_;  ///< current values of monitored structures
    std::shared_ptr<Core::LinAlg::Vector<double>>
        initialmonvalues_;  ///< initial values of monitored structures
    std::shared_ptr<Core::LinAlg::Vector<double>>
        monitortypes_;    ///< vector containing type of monitors
    int offset_id_;       ///< smallest constraint boundary condition ID
    int max_constr_id_;   ///< max number of constraints
    int num_constr_id_;   ///< number of constraint boundary conditions
    int num_monitor_id_;  ///< smallest monitor boundary condition ID
    int min_monitor_id_;  ///< number monitor boundary condition ID
    std::shared_ptr<Core::LinAlg::Vector<double>> fact_;  ///< vector with current time curve values
    std::shared_ptr<Core::LinAlg::Vector<double>> lagr_mult_vec_;      ///< lagrange multipliers
    std::shared_ptr<Core::LinAlg::Vector<double>> lagr_mult_vec_old_;  ///< lagrange multipliers
    std::shared_ptr<Core::LinAlg::SparseOperator>
        constr_matrix_;    ///< additional rectangular matrix
    bool haveconstraint_;  ///< are there constraints at all?
    bool havelagrconstr_;  ///< are there constraints controlled by Lagrange multiplier?
    bool havepenaconstr_;  ///< are there constraints controlled by Penalty approach?
    bool havemonitor_;     ///< are there monitor conditions?
    double uzawaparam_;    ///< parameter of Uzawa algorithm (only for the case the linear uzawa is
                           ///< not used)

    std::shared_ptr<Constraint> volconstr3d_;   ///< 3d volume constraints defined on surfaces
    std::shared_ptr<Constraint> areaconstr3d_;  ///< 3d area constraints defined on surfaces
    std::shared_ptr<Constraint> areaconstr2d_;  ///< 2d area constraints defined on lines
    std::shared_ptr<ConstraintPenalty> volconstr3dpen_;
    std::shared_ptr<ConstraintPenalty> areaconstr3dpen_;
    std::shared_ptr<MPConstraint3> mpconplane3d_;   ///< 3d multipoint constraint prescribing the
                                                    ///< motion of a node relatively to a plane
    std::shared_ptr<MPConstraint3> mpcnormcomp3d_;  ///< 3d multipoint constraint prescribing the
                                                    ///< motion of a node to a plane masternode
    std::shared_ptr<MPConstraint2>
        mpconline2d_;  ///< 2d multipoint constraint prescribing the motion
                       ///< of a node relatively to a straight line
    std::shared_ptr<MPConstraint3Penalty> mpcnormcomp3dpen_;


    std::shared_ptr<Monitor> volmonitor3d_;   ///< 3d volume monitors defined on surfaces
    std::shared_ptr<Monitor> areamonitor3d_;  ///< 3d area monitors defined on surfaces
    std::shared_ptr<Monitor> areamonitor2d_;  ///< 2d area monitors defined on lines


   private:
    //! flag indicating if class is setup
    bool issetup_;

    //! flag indicating if class is initialized
    bool isinit_;

   protected:
    //! returns true if setup() was called and is still valid
    bool is_setup() { return issetup_; };

    //! returns true if init(..) was called and is still valid
    bool is_init() { return isinit_; };

    //! check if \ref setup() was called
    void check_is_setup()
    {
      if (not is_setup()) FOUR_C_THROW("setup() was not called.");
    };

    //! check if \ref init() was called
    void check_is_init()
    {
      if (not is_init()) FOUR_C_THROW("init(...) was not called.");
    };

   public:
    //! set flag true after setup or false if setup became invalid
    void set_is_setup(bool trueorfalse) { issetup_ = trueorfalse; };

    //! set flag true after init or false if init became invalid
    void set_is_init(bool trueorfalse) { isinit_ = trueorfalse; };

  };  // class
}  // namespace Constraints
FOUR_C_NAMESPACE_CLOSE

#endif
