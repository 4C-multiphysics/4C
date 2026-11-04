// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCATRA_REACTION_COUPLING_HPP
#define FOUR_C_MAT_SCATRA_REACTION_COUPLING_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_scatra_reaction.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Utils
  {
    class FunctionOfAnything;
  }  // namespace Utils
}  // namespace Discret

namespace Mat
{
  namespace PAR
  {
    namespace REACTIONCOUPLING
    {
      //! interface class for generic reaction coupling
      class ReactionInterface
      {
       public:
        /// factory method
        static std::shared_ptr<ReactionInterface> create_reaction(
            Mat::PAR::ReactionCoupling couplingtype,  //!< coupling type defining reaction
            bool isreacstart,                         //!< flag for reaction start feature
            const std::vector<double>& reacstart      //!< reaction start vector
        );

        /// standard constructor
        ReactionInterface() {};

        /// destructor
        virtual ~ReactionInterface() = default;
        /// initialization (to be called by derived classes)
        virtual void initialize(int numscal,     //!< number of scalars
            const std::vector<double>& couprole  //!< coupling role vector
            ) = 0;

        /// check for initialization
        virtual bool is_init() const = 0;

        /// helper for calculating advanced reaction terms
        virtual double calc_rea_body_force_term(int k,  //!< current scalar id
            int numscal,                                //!< number of scalars
            const std::vector<double>& phinp,           //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                                //!< stoichometry)
            double
                scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
            ) = 0;

        /// helper for calculating advanced reaction term derivatives
        virtual void calc_rea_body_force_deriv(int k,  //!< current scalar id
            int numscal,                               //!< number of scalars
            std::vector<double>& derivs,               //!< vector with derivatives (to be filled)
            const std::vector<double>& phinp,          //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                                //!< stoichometry)
            double
                scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
            ) = 0;

        /// add additional variables for by-function reaction
        virtual void add_additional_variables(const int k,                 //!< current scalar id
            const std::vector<std::pair<std::string, double>>& variables,  //!< variables
            const std::vector<double>& couprole                            //!< coupling role vector
        )
        {
          // do nothing in this case --> only the by-function reaction will override this method
          // if you want to evaluate any coupling terms in your own reaction coupling just override
          // it in your own function
          FOUR_C_THROW("Only the by-function coupling is capable of adding additional variables");
          return;
        }

        /// helper for calculating advanced reaction term derivatives w.r.t. additional variables
        virtual void calc_rea_body_force_deriv_add_variables(const int k,  //!< current scalar id
            std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
            const std::vector<std::pair<std::string, double>>& variables,  //!< variables
            const std::vector<std::pair<std::string, double>>&
                constants,                        //!< constants (including scalar values phinp)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                                //!< stoichometry)
            double
                scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
        )
        {
          // do nothing in this case --> only the by-function reaction will override this method
          // if you want to evaluate any coupling terms in your own reaction coupling just override
          // it in your own function
          FOUR_C_THROW(
              "Only the by-function coupling is capable of calculating additional derivatives");
          return;
        }
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      //! reaction start feature for reaction coupling
      //! it wraps another reaction class
      class ReacStart : public ReactionInterface
      {
       public:
        /// standard constructor
        ReacStart(std::shared_ptr<ReactionInterface> reaction, const std::vector<double>& reacstart)
            : reaction_(reaction), reacstart_(reacstart) {};

        /// initialization
        void initialize(int numscal,             //!< number of scalars
            const std::vector<double>& couprole  //!< coupling role vector
            ) override;

        /// check for initialization
        bool is_init() const override { return reaction_->is_init(); };

        /// helper for calculating advanced reaction terms
        double calc_rea_body_force_term(int k,  //!< current scalar id
            int numscal,                        //!< number of scalars
            const std::vector<double>& phinp,   //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                                //!< stoichometry)
            double
                scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
            ) override;

        /// helper for calculating advanced reaction term derivatives
        void calc_rea_body_force_deriv(int k,  //!< current scalar id
            int numscal,                       //!< number of scalars
            std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
            const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                                //!< stoichometry)
            double
                scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
            ) override;

       private:
        /// helper for reaction start feature
        virtual std::vector<double> modify_phi(const std::vector<double>& phinp);

        /// actual reaction
        std::shared_ptr<ReactionInterface> reaction_;
        /// reacstart vector
        const std::vector<double>& reacstart_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      //! wrapper class for reaction coupling with potential phi scaling
      //! it wraps another reaction class
      class ReactionWithPhiScaling : public ReactionInterface
      {
       public:
        /// standard constructor
        ReactionWithPhiScaling(std::shared_ptr<ReactionInterface> reaction)
            : reaction_(reaction) {};

        /// initialization (to be called by derived classes)
        void initialize(int numscal,             //!< number of scalars
            const std::vector<double>& couprole  //!< coupling role vector
            ) override
        {
          reaction_->initialize(numscal, couprole);
        };

        /// check for initialization
        bool is_init() const override { return reaction_->is_init(); };

        /// helper for calculating advanced reaction terms
        double calc_rea_body_force_term(int k,  //!< current scalar id
            int numscal,                        //!< number of scalars
            const std::vector<double>& phinp,   //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                                //!< stoichometry)
            double
                scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
            ) override;

        /// helper for calculating advanced reaction term derivatives
        void calc_rea_body_force_deriv(int k,  //!< current scalar id
            int numscal,                       //!< number of scalars
            std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
            const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                                //!< stoichometry)
            double
                scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
            ) override;

        /// helper for calculating advanced reaction term derivatives after additional variables
        void calc_rea_body_force_deriv_add_variables(const int k,  //!< current scalar id
            std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
            const std::vector<std::pair<std::string, double>>& variables,  //!< variables
            const std::vector<std::pair<std::string, double>>&
                constants,                        //!< constants (including scalar values phinp)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                                //!< stoichometry)
            double
                scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
            ) override;

        /// add additional variables for by-function reaction
        void add_additional_variables(const int k,                         //!< current scalar id
            const std::vector<std::pair<std::string, double>>& variables,  //!< variables
            const std::vector<double>& couprole                            //!< coupling role vector
            ) override
        {
          reaction_->add_additional_variables(k, variables, couprole);
        }

       protected:
        /// actual reaction
        std::shared_ptr<ReactionInterface> reaction_;

        /// helper for scaling
        virtual std::vector<double> modify_phi(const std::vector<double>& phinp, double scale_phi);
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      //! base class for reaction coupling kinetics
      class ReactionBase : public ReactionInterface
      {
       public:
        /// standard constructor
        ReactionBase() : isinit_(false) {};

        /// initialization (to be called by derived classes)
        void initialize(int numscal,             //!< number of scalars
            const std::vector<double>& couprole  //!< coupling role vector
            ) override
        {
          isinit_ = true;
        };

        /// check for initialization
        bool is_init() const override { return isinit_; };

        /// helper for calculating advanced reaction terms
        double calc_rea_body_force_term(int k,  //!< current scalar id
            int numscal,                        //!< number of scalars
            const std::vector<double>& phinp,   //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                                //!< stoichometry)
            double
                scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
            ) override
        {
          // check
          FOUR_C_ASSERT(is_init(), "Reaction class has not been initialized!");

          // call the real evaluation (scale_phi should have been applied in wrapper class)
          return calc_rea_body_force_term(k, numscal, phinp, constants, couprole, scale_reac);
        };

        /// helper for calculating advanced reaction term derivatives
        void calc_rea_body_force_deriv(int k,  //!< current scalar id
            int numscal,                       //!< number of scalars
            std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
            const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                                //!< stoichometry)
            double
                scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
            ) override
        {
          // check
          FOUR_C_ASSERT(is_init(), "Reaction class has not been initialized!");

          // call the real evaluation (scale_phi should have been applied in wrapper class)
          calc_rea_body_force_deriv(k, numscal, derivs, phinp, constants, couprole, scale_reac);
          return;
        };

       private:
        /// helper for calculating advanced reaction terms
        virtual double calc_rea_body_force_term(int k,  //!< current scalar id
            int numscal,                                //!< number of scalars
            const std::vector<double>& phinp,           //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) = 0;

        /// helper for calculating advanced reaction term derivatives
        virtual void calc_rea_body_force_deriv(int k,  //!< current scalar id
            int numscal,                               //!< number of scalars
            std::vector<double>& derivs,               //!< vector with derivatives (to be filled)
            const std::vector<double>& phinp,          //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) = 0;

       private:
        // initialization flag
        bool isinit_;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      //! simple multiplicative reaction coupling
      class SimpleMultiplicative : public ReactionBase
      {
       public:
        /// standard constructor
        SimpleMultiplicative() {};

        /// initialization
        void initialize(int numscal,             //!< number of scalars
            const std::vector<double>& couprole  //!< coupling role vector
            ) override;

       private:
        /// helper for calculating advanced reaction terms
        double calc_rea_body_force_term(int k,  //!< current scalar id
            int numscal,                        //!< number of scalars
            const std::vector<double>& phinp,   //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) override;

        /// helper for calculating advanced reaction term derivatives
        void calc_rea_body_force_deriv(int k,  //!< current scalar id
            int numscal,                       //!< number of scalars
            std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
            const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) override;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      //! power multiplicative reaction coupling
      class PowerMultiplicative : public ReactionBase
      {
       public:
        /// standard constructor
        PowerMultiplicative() {};

        /// initialization
        void initialize(int numscal,             //!< number of scalars
            const std::vector<double>& couprole  //!< coupling role vector
            ) override;

        /// helper for calculating advanced reaction terms
        double calc_rea_body_force_term(int k,  //!< current scalar id
            int numscal,                        //!< number of scalars
            const std::vector<double>& phinp,   //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) override;

        /// helper for calculating advanced reaction term derivatives
        void calc_rea_body_force_deriv(int k,  //!< current scalar id
            int numscal,                       //!< number of scalars
            std::vector<double>& derivs,       //!< vector with derivatives (to be filled)
            const std::vector<double>& phinp,  //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) override;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      //! simple multiplicative reaction coupling
      class Constant : public ReactionBase
      {
       public:
        /// standard constructor
        Constant() {};

        /// initialization
        void initialize(int numscal,             //!< number of scalars
            const std::vector<double>& couprole  //!< coupling role vector
            ) override;

       private:
        /// helper for calculating advanced reaction terms
        double calc_rea_body_force_term(int k,  //!< current scalar id
            int numscal,                        //!< number of scalars
            const std::vector<double>& phinp,   //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) override;

        /// helper for calculating advanced reaction term derivatives
        void calc_rea_body_force_deriv(const int k,  //!< current scalar id
            int numscal,                             //!< number of scalars
            std::vector<double>& derivs,             //!< vector with derivatives (to be filled)
            const std::vector<double>& phinp,        //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) override;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      //! simple multiplicative reaction coupling
      class MichaelisMenten : public ReactionBase
      {
       public:
        /// standard constructor
        MichaelisMenten() {};

        /// initialization
        void initialize(int numscal,             //!< number of scalars
            const std::vector<double>& couprole  //!< coupling role vector
            ) override;

       private:
        /// helper for calculating advanced reaction terms
        double calc_rea_body_force_term(const int k,  //!< current scalar id
            int numscal,                              //!< number of scalars
            const std::vector<double>& phinp,         //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) override;

        /// helper for calculating advanced reaction term derivatives
        void calc_rea_body_force_deriv(const int k,  //!< current scalar id
            int numscal,                             //!< number of scalars
            std::vector<double>& derivs,             //!< vector with derivatives (to be filled)
            const std::vector<double>& phinp,        //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) override;
      };

      /*----------------------------------------------------------------------*
       * **********************************************************************
       *----------------------------------------------------------------------*/

      //! simple multiplicative reaction coupling
      class ByFunction : public ReactionBase
      {
       public:
        /// standard constructor
        ByFunction() {};

        /// initialization
        void initialize(int numscal,             //!< number of scalars
            const std::vector<double>& couprole  //!< coupling role vector
            ) override;

        /// helper for calculating advanced reaction term derivatives after additional variables
        void calc_rea_body_force_deriv_add_variables(const int k,  //!< current scalar id
            std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
            const std::vector<std::pair<std::string, double>>& variables,  //!< variables
            const std::vector<std::pair<std::string, double>>&
                constants,                        //!< constants (including scalar values phinp)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac,  //!< scaling factor for reaction term (= reaction coefficient *
                                //!< stoichometry)
            double
                scale_phi  //!< scaling factor for scalar values (used for reference concentrations)
            ) override
        {
          // check
          FOUR_C_ASSERT(is_init(), "Reaction class has not been initialized!");

          // call the real evaluation (scale_phi should have been applied in wrapper class)
          calc_rea_body_force_deriv_add_variables(
              k, derivs, variables, constants, couprole, scale_reac);
        }

        /// add additional variables for by-function reaction
        void add_additional_variables(const int k,                         //!< current scalar id
            const std::vector<std::pair<std::string, double>>& variables,  //!< variables
            const std::vector<double>& couprole                            //!< coupling role vector
            ) override;

       private:
        /// helper for calculating advanced reaction terms
        double calc_rea_body_force_term(const int k,  //!< current scalar id
            int numscal,                              //!< number of scalars
            const std::vector<double>& phinp,         //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) override;

        /// helper for calculating advanced reaction term derivatives
        void calc_rea_body_force_deriv(const int k,  //!< current scalar id
            int numscal,                             //!< number of scalars
            std::vector<double>& derivs,             //!< vector with derivatives (to be filled)
            const std::vector<double>& phinp,        //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
            ) override;

        /// helper for calculating advanced reaction term derivatives after additional variables
        void calc_rea_body_force_deriv_add_variables(const int k,  //!< current scalar id
            std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
            const std::vector<std::pair<std::string, double>>& variables,  //!< variables
            const std::vector<std::pair<std::string, double>>&
                constants,                        //!< constants (including scalar values phinp)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
        );

        /// helper for evaluation by function
        void build_phi_vector_for_function(
            const std::vector<double>& phinp_org,  //!< scalar values at t_(n+1)
            int numscal                            //!< number of scalars
        );


        //! templated internal add_additional_variables implementation
        template <int dim>
        void add_additional_variables_internal(const int k,                //!< current scalar id
            const std::vector<std::pair<std::string, double>>& variables,  //!< variables
            const std::vector<double>& couprole                            //!< coupling role vector
        );

        //! templated internal calc_rea_body_force_deriv_add_variables implementation
        template <int dim>
        void calc_rea_body_force_deriv_add_variables_internal(const int k,  //!< current scalar id
            std::vector<double>& derivs,  //!< vector with derivatives (to be filled)
            const std::vector<std::pair<std::string, double>>& variables,  //!< variables
            const std::vector<std::pair<std::string, double>>&
                constants,                        //!< constants (including scalar values phinp)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
        );

        //! templated internal calc_rea_body_force_deriv implementation
        template <int dim>
        void calc_rea_body_force_deriv_internal(int k,  //!< current scalar id
            int numscal,                                //!< number of scalars
            std::vector<double>& derivs,                //!< vector with derivatives (to be filled)
            const std::vector<double>& phinp,           //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
        );

        //! templated internal calc_rea_body_force_term implementation
        template <int dim>
        double calc_rea_body_force_term_internal(int k,  //!< current scalar id
            int numscal,                                 //!< number of scalars
            const std::vector<double>& phinp,            //!< scalar values at t_(n+1)
            const std::vector<std::pair<std::string, double>>&
                constants,  //!< vector containing values which are independent of the scalars (e.g.
                            //!< t,x,y,z)
            const std::vector<double>& couprole,  //!< coupling role vector
            double scale_reac  //!< scaling factor for reaction term (= reaction coefficient *
                               //!< stoichometry)
        );

        //! templated internal Initialize implementation
        template <int dim>
        void initialize_internal(int numscal,    //!< number of scalars
            const std::vector<double>& couprole  //!< coupling role vector
        );

        /// variable vector for function evaluation
        std::vector<std::pair<std::string, double>> variables_;
      };

    }  // namespace REACTIONCOUPLING

  }  // namespace PAR
}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
