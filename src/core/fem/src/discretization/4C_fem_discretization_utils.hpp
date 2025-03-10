// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_DISCRETIZATION_UTILS_HPP
#define FOUR_C_FEM_DISCRETIZATION_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_condition.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>
#include <set>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class MapExtractor;
}  // namespace Core::LinAlg

namespace Core::FE
{
  class AssembleStrategy;
}

namespace Core::Utils
{
  class FunctionManager;
}

namespace Core::FE
{
  class Discretization;
  class DiscretizationFaces;
  namespace Utils
  {
    class Dbc;

    /** \brief Evaluate the elements of the given discretization and fill the
     *         system matrix and vector
     *
     *  This evaluate routine supports the evaluation of a subset of all column
     *  elements inside the given discretization. If the \c col_ele_map pointer
     *  is not set or set to \c nullptr, this routine generates almost no overhead
     *  and is equivalent to the more familiar implementation in Core::FE::Discretization.
     *
     *  \param discret      (in)  : discretization containing the considered elements
     *  \param eparams      (in)  : element parameter list
     *  \param systemmatrix (out) : system-matrix which is supposed to be filled
     *  \param systemvector (out) : system-vector which is supposed to be filled
     *  \param col_ele_map  (in)  : column element map, which can be a subset of the
     *                              discretization column map ( optional )
     */
    void evaluate(Core::FE::Discretization& discret, Teuchos::ParameterList& eparams,
        const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvector,
        const Epetra_Map* col_ele_map = nullptr);

    /** \brief Evaluate the elements of the given discretization and fill the
     *         system matrices and vectors
     *
     *  This evaluate routine supports the evaluation of a subset of all column
     *  elements inside the given discretization. If the \c col_ele_map pointer
     *  is not set or set to \c nullptr, this routine generates almost no overhead
     *  and is equivalent to the more familiar implementation in Core::FE::Discretization.
     *
     *  \param discret      (in)  : discretization containing the considered elements
     *  \param eparams      (in)  : element parameter list
     *  \param systemmatrix (out) : system-matrix vector which is supposed to be filled
     *  \param systemvector (out) : system-vector vector which is supposed to be filled
     *  \param col_ele_map  (in)  : column element map, which can be a subset of the
     *                              discretization column map ( optional )
     */
    void evaluate(Core::FE::Discretization& discret, Teuchos::ParameterList& eparams,
        std::vector<std::shared_ptr<Core::LinAlg::SparseOperator>>& systemmatrices,
        std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>>& systemvector,
        const Epetra_Map* col_ele_map = nullptr);

    /** \brief Evaluate the elements of the given discretization and fill the
     *         system matrices and vectors
     *
     *  This evaluate routine supports the evaluation of a subset of all column
     *  elements inside the given discretization. If the \c col_ele_map pointer
     *  is not set or set to \c nullptr, this routine generates almost no overhead
     *  and is equivalent to the more familiar implementation in Core::FE::Discretization.
     *
     *  \param discret      (in)  : discretization containing the considered elements
     *  \param eparams      (in)  : element parameter list
     *  \param strategy     (out) : assemble strategy containing all the system vectors
     *                              and matrices
     *  \param col_ele_map  (in)  : column element map, which can be a subset of the
     *                              discretization column map ( optional )
     */
    void evaluate(Core::FE::Discretization& discret, Teuchos::ParameterList& eparams,
        Core::FE::AssembleStrategy& strategy, const Epetra_Map* col_ele_map = nullptr);

    /** \brief Evaluate Dirichlet boundary conditions
     *
     *  non-member functions to call the dbc public routines
     */
    void evaluate_dirichlet(const Core::FE::Discretization& discret,
        const Teuchos::ParameterList& params,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvector,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvectord,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvectordd,
        const std::shared_ptr<Core::LinAlg::Vector<int>>& toggle,
        const std::shared_ptr<Core::LinAlg::MapExtractor>& dbcmapextractor);

    /** \brief Evaluate Dirichlet boundary conditions
     *
     *  Call this variant, if you need no new dbc map extractor.
     *  See the corresponding called function for more detailed information.
     */
    inline void evaluate_dirichlet(const Core::FE::Discretization& discret,
        const Teuchos::ParameterList& params,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvector,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvectord,
        const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvectordd,
        const std::shared_ptr<Core::LinAlg::Vector<int>>& toggle)
    {
      evaluate_dirichlet(
          discret, params, systemvector, systemvectord, systemvectordd, toggle, nullptr);
    }

    /*!
    \brief Evaluate a specified initial field (scalar or vector field)

    Loop all initial field conditions attached to the discretization @p discret and evaluate them to
    @p fieldvector if their names match the user-provided string @p fieldstring. Information on
    which local DOFs ids are addressed by the condition MUST be pre-defined and is represented by
    the @p locids vector. As an example, if we provide an initial velocity for a 3D structural
    dynamics simulation, locids must contain the local DOF ids {0,1,2}. Another example would be
    prescribing an initial pressure in a 3D fluid dynamics simulation, where locids would have to
    contain only the local pressure DOF id, namely {3}.
    */
    void evaluate_initial_field(const Core::Utils::FunctionManager& function_manager,
        const Core::FE::Discretization& discret, const std::string& fieldstring,
        Core::LinAlg::Vector<double>& fieldvector, const std::vector<int>& locids);


    /*!
    \brief Evaluate a specified initial field (scalar or vector field)

    This is the actual evaluation method.

    */
    void do_initial_field(const Core::Utils::FunctionManager& function_manager,
        const Core::FE::Discretization& discret, Core::Conditions::Condition& cond,
        Core::LinAlg::Vector<double>& fieldvector, const std::vector<int>& locids);

    /** \brief Build a Dbc object
     *
     *  The Dbc object is build in dependency of the given discretization.
     */
    std::shared_ptr<const Dbc> build_dbc(const Core::FE::Discretization* discret_ptr);

    /** \brief Default Dirchilet boundary condition evaluation class
     */
    class Dbc
    {
     protected:
      enum DbcSet
      {
        set_row = 0,  ///< access the dbc row GID set
        set_col = 1   ///< access the dbc column GID set
      };

     public:
      struct DbcInfo
      {
        /*!
         * \brief toggle vector to store the fix/free state of a dof
         */
        Core::LinAlg::Vector<int> toggle;

        /*!
         * \brief record the lowest geometrical order that a dof applies
         */
        Core::LinAlg::Vector<int> hierarchy;

        /*!
         * \brief record the last condition id prescribed and assign value to dof
         */
        Core::LinAlg::Vector<int> condition;

        /*!
         * \brief the prescribed value assigned to dof
         * \note This is necessary to check the DBC consistency
         */
        Core::LinAlg::Vector<double> values;

        /*!
         * \brief constructor using the toggle vector as input
         * \note all the vectors use the same map
         */
        DbcInfo(const Core::LinAlg::Vector<int>& toggle_input)
            : toggle(toggle_input),
              hierarchy(Core::LinAlg::Vector<int>(toggle_input.Map())),
              condition(Core::LinAlg::Vector<int>(toggle_input.Map())),
              values(Core::LinAlg::Vector<double>(toggle_input.Map(), true))
        {
          hierarchy.PutValue(std::numeric_limits<int>::max());
          condition.PutValue(-1);
        }

        /*!
         * \brief constructor using the vector map as input
         * \note all the vectors use the same map
         */
        DbcInfo(const Epetra_BlockMap& toggle_map)
            : toggle(Core::LinAlg::Vector<int>(toggle_map)),
              hierarchy(Core::LinAlg::Vector<int>(toggle_map)),
              condition(Core::LinAlg::Vector<int>(toggle_map)),
              values(Core::LinAlg::Vector<double>(toggle_map, true))
        {
          hierarchy.PutValue(std::numeric_limits<int>::max());
          condition.PutValue(-1);
        }
      };

      /** \brief constructor
       *
       *  Intentionally left blank! */
      Dbc() {};

      /// destructor
      virtual ~Dbc() = default;

      /** \brief Extract parameters and setup some temporal variables, before the actual
       *  evaluation process can start
       */
      void operator()(const Core::FE::Discretization& discret, const Teuchos::ParameterList& params,
          const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvector,
          const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvectord,
          const std::shared_ptr<Core::LinAlg::Vector<double>>& systemvectordd,
          const std::shared_ptr<Core::LinAlg::Vector<int>>& toggle,
          const std::shared_ptr<Core::LinAlg::MapExtractor>& dbcmapextractor) const;

     protected:
      /// create the toggle vector based on the given systemvector maps
      std::shared_ptr<Core::LinAlg::Vector<int>> create_toggle_vector(
          const std::shared_ptr<Core::LinAlg::Vector<int>> toggle_input,
          const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors) const;

      /** \brief Evaluate Dirichlet boundary conditions
       *
       *  Loop all Dirichlet conditions attached to the discretization and evaluate
       *  them. This method considers all conditions in condition_ with the names
       *  "PointDirichlet", "LineDirichlet", "SurfaceDirichlet" and "VolumeDirichlet".
       *  It takes a current time from the parameter list params named "total time"
       *  and evaluates the appropriate time curves at that time for each
       *  Dirichlet condition separately. If "total time" is not included
       *  in the parameters, no time curves are used.
       *
       *  \note Opposed to the other 'Evaluate' method does this one NOT assembly
       *        but OVERWRITE values in the output vector systemvector. For this
       *        reason, dirichlet boundary conditions are evaluated in the
       *        following order: First "VolumeDirichlet", then "SurfaceDirichlet",
       *        then "LineDirichlet" and finally "PointDirichlet". This way, the
       *        lower entity dirichlet BCs override the higher ones and a point
       *        Dirichlet BCs has priority over other dirichlet BCs in the input
       *        file.
       *
       *  Parameters recognized by this method:
       *  \code
       *  params.set("total time",acttime); // current total time
       *  \endcode
       *
       *  \param params           (in): Teuchos Parameter List
       *  \param discret          (in): discretization corresponding to the input
       *                                system vectors
       *  \param params           (in): List of parameters
       *  \param systemvector    (out): Vector holding prescribed Dirichlet values
       *  \param systemvectord   (out): Vector holding 1st time derivative of
       *                                prescribed Dirichlet values
       *  \param systemvectordd  (out): Vector holding 2nd time derivative prescribed
       *                                Dirichlet values
       *  \param toggle          (out): Vector containing 1.0 for each Dirichlet dof
       *                                and 0 for everything else
       *  \param dbcmapextractor (out): Map extractor containing maps for the DOFs
       *                                subjected to Dirichlet boundary conditions
       *                                and the remaining/free DOFs
       */
      virtual void evaluate(const Teuchos::ParameterList& params,
          const Core::FE::Discretization& discret, double time,
          const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors, DbcInfo& info,
          std::shared_ptr<std::set<int>>* dbcgids) const;

      /** \brief loop through Dirichlet conditions and evaluate them
       *
       *  Note that this method does not sum up but 'sets' values in systemvector.
       *  For this reason, Dirichlet BCs are evaluated hierarchical meaning
       *  in this order:
       *                 VolumeDirichlet
       *                 SurfaceDirichlet
       *                 LineDirichlet
       *                 PointDirichlet
       *  This way, lower entities override higher ones which is
       *  equivalent to inheritance of dirichlet BCs.
       *
       *  Lower entities MUST NOT set dof values in systemvector before
       *  we know if higher entities also prescribe/release Dirichlet BCs
       *  for the same dofs!
       *
       *  Therefore, we first have to assess the full hierarchy and
       *  set the toggle vector for a dof to 1 if an entity prescribes a
       *  dirichlet BC and we have to set it to 0 again if a higher entity
       *  does NOT prescribe a dirichlet BC for the same dof. This is done
       *  in read_dirichlet_condition(...). We do this for each type of entity,
       *  starting with volume DBCs.
       *
       *  Only then we call do_dirichlet_condition(...) for each type of entity,
       *  starting with volume DBCs.
       *
       *  This way, it is guaranteed, that the highest entity defined in
       *  the input file determines if the systemvector for the corresponding
       *  dofs is actually touched, or not, irrespective of the dirichlet BC
       *  definition of a lower entity.
       */
      void read_dirichlet_condition(const Teuchos::ParameterList& params,
          const Core::FE::Discretization& discret,
          const std::vector<std::shared_ptr<Core::Conditions::Condition>>& conds, double time,
          DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids) const;

      /// loop over the conditions and read the given type
      void read_dirichlet_condition(const Teuchos::ParameterList& params,
          const Core::FE::Discretization& discret,
          const std::vector<std::shared_ptr<Core::Conditions::Condition>>& conds, double time,
          DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids,
          const enum Core::Conditions::ConditionType& type) const;

      /** \brief Determine dofs subject to Dirichlet condition from input file
       *
       *  \param params   (in)  : Teuchos Parameter List
       *  \param discret  (in)  :  discretization corresponding to the input
       *                           system vectors
       *  \param cond     (in)  :  The condition object
       *  \param toggle   (out) :  Its i-th component is set 1 if it has a
       *                           DBC, otherwise this component remains untouched
       *  \param dbcgids  (out) :  Map containing DOFs subjected to Dirichlet
       *                           boundary conditions (row and optional column)
       *
       *  \remark If you want to be sure which Dirichlet values are set, look at
       *          the highest entity for a certain node in your input file.
       *
       *  The corresponding condition, e.g.:
       *  ---------DESIGN LINE DIRICH CONDITIONS
       *  // example_line
       *  E 1 - NUMDOF 6 ONOFF 0 1 1 VAL 0.0 1.0 1.0 FUNCT 1 none none
       *
       *  tells you which dofs are actually conditioned. In the example given
       *  above, the highest entity which contains a certain node is a LINE.
       *  The first dof has an ONOFF-toggle of ZERO. This means, that the nodes
       *  in LINE 'E 1' definitely DO NOT HAVE a dirichlet BC on their first dof.
       *  No matter what is defined in a condition line of lower priority. The
       *  corresponding entries in the systemvectors remain untouched.
       */
      virtual void read_dirichlet_condition(const Teuchos::ParameterList& params,
          const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond,
          double time, DbcInfo& info, const std::shared_ptr<std::set<int>>* dbcgids,
          int hierarchical_order) const;

      /** \brief Assignment of the values to the system vectors.
       *
       *  (1) Assign VolumeDirichlet DBC GIDs
       *  (2) Assign SurfaceDirichlet DBC GIDs
       *  (3) Assign LineDirichlet DBC GIDs
       *  (4) Assign PointDirichlet DBC GIDs
       */
      void do_dirichlet_condition(const Teuchos::ParameterList& params,
          const Core::FE::Discretization& discret,
          const std::vector<std::shared_ptr<Core::Conditions::Condition>>& conds, double time,
          const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
          const Core::LinAlg::Vector<int>& toggle,
          const std::shared_ptr<std::set<int>>* dbcgids) const;

      /// loop over the conditions and assign the given type
      void do_dirichlet_condition(const Teuchos::ParameterList& params,
          const Core::FE::Discretization& discret,
          const std::vector<std::shared_ptr<Core::Conditions::Condition>>& conds, double time,
          const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
          const Core::LinAlg::Vector<int>& toggle, const std::shared_ptr<std::set<int>>* dbcgids,
          const enum Core::Conditions::ConditionType& type) const;

      /** \brief Apply the Dirichlet values to the system vectors
       *
       *  \param params          (in): Teuchos Parameter List
       *  \param discret         (in): discretization corresponding to the input
       *                               system vectors
       *  \param cond            (in): The condition object
       *  \param time            (in): Evaluation time
       *  \param systemvector   (out): Vector to apply DBCs to (eg displ. in
       *                               structure, vel. in fluids)
       *  \param systemvectord  (out): First time derivative of DBCs
       *  \param systemvectordd (out): Second time derivative of DBCs
       *  \param toggle          (in): Its i-th component is set 1 if it has
       *                               a DBC, otherwise this component remains
       *                               untouched
       *
       *  \remark If you want to be sure which Dirichlet values are set, look at
       *          the highest entity for a certain node in your input file.
       *
       *  The corresponding condition, e.g.:
       *  ---------DESIGN LINE DIRICH CONDITIONS
       *  // example_line
       *  E 1 - NUMDOF 6 ONOFF 0 1 1 VAL 0.0 1.0 1.0 FUNCT 1 none none
       *
       *  tells you which dofs are actually conditioned. In the example given
       *  above, the highest entity which contains a certain node is a LINE.
       *  The first dof has an ONOFF-toggle of ZERO. This means, that the
       *  nodes in LINE 'E 1' definitely DO NOT HAVE a dirichlet BC on their
       *  first dof. No matter what is defined in a condition line of lower
       *  priority. The corresponding entries in the systemvectors remain
       *  untouched.
       *
       *  \version rauch 06/2016
       *  Shifted and rearranged parts of the former implementation to
       *  read_dirichlet_condition(...). This fixes inconsistency in hierarchy
       *  handling. Now, the Dirichlet conditions are first read by
       *  read_dirichlet_condition(...). Then, they are applied by
       *  do_dirichlet_condition(...).
       */
      virtual void do_dirichlet_condition(const Teuchos::ParameterList& params,
          const Core::FE::Discretization& discret, const Core::Conditions::Condition& cond,
          double time, const std::shared_ptr<Core::LinAlg::Vector<double>>* systemvectors,
          const Core::LinAlg::Vector<int>& toggle,
          const std::shared_ptr<std::set<int>>* dbcgids) const;

      /** \brief Create a Dbc map extractor, if desired
       */
      void build_dbc_map_extractor(const Core::FE::Discretization& discret,
          const std::shared_ptr<const std::set<int>>& dbcrowgids,
          const std::shared_ptr<Core::LinAlg::MapExtractor>& dbcmapextractor) const;

    };  // class Dbc
  }  // namespace Utils
}  // namespace Core::FE


FOUR_C_NAMESPACE_CLOSE

#endif
