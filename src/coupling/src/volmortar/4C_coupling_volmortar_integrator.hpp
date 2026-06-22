// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_COUPLING_VOLMORTAR_INTEGRATOR_HPP
#define FOUR_C_COUPLING_VOLMORTAR_INTEGRATOR_HPP

/*---------------------------------------------------------------------*
 | headers                                                 farah 01/14 |
 *---------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_coupling_volmortar.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------*
 | forward declarations                                    farah 01/14 |
 *---------------------------------------------------------------------*/
namespace Core::LinAlg
{
  class SerialDenseVector;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Mortar
{
  class IntCell;
}

namespace Coupling::VolMortar
{
  class Cell;

  template <Core::FE::CellType distype_source, Core::FE::CellType distype_target>
  class VolMortarIntegrator
  {
   public:
    // constructor
    VolMortarIntegrator(Teuchos::ParameterList& params);

    // destructor
    virtual ~VolMortarIntegrator() = default;

    //! nsource_: number of source element nodes
    static constexpr int nsource_ = Core::FE::num_nodes(distype_source);

    //! ntarget_: number of target element nodes
    static constexpr int ntarget_ = Core::FE::num_nodes(distype_target);

    //! number of space dimensions ("+1" due to considering only interface elements)
    static constexpr int ndim_ = Core::FE::dim<distype_source>;

    /*!
    \brief Initialize Gauss rule (points, weights)

    */
    void initialize_gp(bool integrateele = false, int domain = 0,
        Core::FE::CellType shape = Core::FE::CellType::tet4);

    /*!
    \brief Integrate cell for 2D problems

    */
    void integrate_cells_2d(Core::Elements::Element& source_ele,
        Core::Elements::Element& target_ele, Mortar::IntCell& cell,
        Core::LinAlg::SparseMatrix& dmatrix, Core::LinAlg::SparseMatrix& mmatrix,
        const Core::FE::Discretization& source_dis, const Core::FE::Discretization& target_dis,
        int source_dofset, int target_dofset);

    /*!
    \brief Integrate cell for 3D problems

    */
    void integrate_cells_3d(Core::Elements::Element& Aele, Core::Elements::Element& Bele,
        Coupling::VolMortar::Cell& cell, Core::LinAlg::SparseMatrix& dmatrix_A,
        Core::LinAlg::SparseMatrix& mmatrix_A, Core::LinAlg::SparseMatrix& dmatrix_B,
        Core::LinAlg::SparseMatrix& mmatrix_B, const Core::FE::Discretization& Adis,
        const Core::FE::Discretization& Bdis, int source_dofset_A, int target_dofset_A,
        int source_dofset_B, int target_dofset_B);

    /*!
    \brief Integrate cell for 3D problems

    */
    void integrate_cells_3d_direct_diveregence(Core::Elements::Element& Aele,
        Core::Elements::Element& Bele, Cut::VolumeCell& vc,
        std::shared_ptr<Core::FE::GaussPoints> intpoints, bool switched_conf,
        Core::LinAlg::SparseMatrix& dmatrix_A, Core::LinAlg::SparseMatrix& mmatrix_A,
        Core::LinAlg::SparseMatrix& dmatrix_B, Core::LinAlg::SparseMatrix& mmatrix_B,
        const Core::FE::Discretization& Adis, const Core::FE::Discretization& Bdis,
        int source_dofset_A, int target_dofset_A, int source_dofset_B, int target_dofset_B);

    /*!
    \brief Integrate ele for 3D problems

    */
    void integrate_ele_3d(int domain, Core::Elements::Element& Aele, Core::Elements::Element& Bele,
        Core::LinAlg::SparseMatrix& dmatrix_A, Core::LinAlg::SparseMatrix& mmatrix_A,
        Core::LinAlg::SparseMatrix& dmatrix_B, Core::LinAlg::SparseMatrix& mmatrix_B,
        const Core::FE::Discretization& Adis, const Core::FE::Discretization& Bdis,
        int source_dofset_A, int target_dofset_A, int source_dofset_B, int target_dofset_B);

    /*!
    \brief Integrate ele for 3D problems

    */
    void integrate_ele_based_3d_a_dis(Core::Elements::Element& Aele, std::vector<int>& foundeles,
        Core::LinAlg::SparseMatrix& dmatrix_A, Core::LinAlg::SparseMatrix& mmatrix_A,
        const Core::FE::Discretization& Adiscret, const Core::FE::Discretization& Bdiscret,
        int dofsetA, int dofsetB);

    /*!
    \brief Integrate ele for 3D problems

    */
    void integrate_ele_based_3d_b_dis(Core::Elements::Element& Bele, std::vector<int>& foundeles,
        Core::LinAlg::SparseMatrix& dmatrix_B, Core::LinAlg::SparseMatrix& mmatrix_B,
        const Core::FE::Discretization& Adiscret, const Core::FE::Discretization& Bdiscret,
        int dofsetA, int dofsetB);

   protected:
    /*!
    \brief Check integration point mapping (2D)

    */
    bool check_mapping_2d(Core::Elements::Element& source_ele, Core::Elements::Element& target_ele,
        double* source_xi, double* target_xi);

    /*!
    \brief Check integration point mapping (3D)

    */
    bool check_mapping_3d(Core::Elements::Element& source_ele, Core::Elements::Element& target_ele,
        double* source_xi, double* target_xi);

    //@}
    int ngp_;                                 // number of Gauss points
    Core::LinAlg::SerialDenseMatrix coords_;  // Gauss point coordinates
    std::vector<double> weights_;             // Gauss point weights

    // input parameter
    DualQuad dualquad_;  // type of quadratic weighting interpolation
    Shapefcn shape_;     // type of test function
  };

  template <Core::FE::CellType distype_source>
  class VolMortarIntegratorEleBased
  {
   public:
    // constructor
    VolMortarIntegratorEleBased(Teuchos::ParameterList& params);

    // destructor
    virtual ~VolMortarIntegratorEleBased() = default;

    //! nsource_: number of source element nodes
    static constexpr int nsource_ = Core::FE::num_nodes(distype_source);

    //! number of space dimensions ("+1" due to considering only interface elements)
    static constexpr int ndim_ = Core::FE::dim<distype_source>;

    /*!
    \brief Initialize Gauss rule (points, weights)

    */
    void initialize_gp();

    /*!
    \brief Integrate ele for 3D problems

    */
    void integrate_ele_based_3d(Core::Elements::Element& source_ele, std::vector<int>& foundeles,
        Core::LinAlg::SparseMatrix& dmatrixA, Core::LinAlg::SparseMatrix& mmatrixA,
        const Core::FE::Discretization& Adiscret, const Core::FE::Discretization& Bdiscret,
        int dofseta, int dofsetb, const Core::LinAlg::Map& PAB_dofrowmap,
        const Core::LinAlg::Map& PAB_dofcolmap);

   protected:
    /*!
    \brief Check integration point mapping (3D)

    */
    bool check_mapping_3d(Core::Elements::Element& source_ele, Core::Elements::Element& target_ele,
        double* source_xi, double* target_xi);

    //@}
    int ngp_;                                 // number of Gauss points
    Core::LinAlg::SerialDenseMatrix coords_;  // Gauss point coordinates
    std::vector<double> weights_;             // Gauss point weights

    // input parameter
    DualQuad dualquad_;  // type of quadratic weighting interpolation
    Shapefcn shape_;     // type of test function
  };

  /*----------------------------------------------------------------------*
   |  Check paraspace mapping                                  farah 02/15|
   *----------------------------------------------------------------------*/
  template <Core::FE::CellType distype_source, Core::FE::CellType distype_target>
  bool check_mapping(Core::Elements::Element& source_ele, Core::Elements::Element& target_ele,
      double* source_xi, double* target_xi)
  {
    // check GP projection (SOURCE)
    double tol = 1e-5;
    if (distype_source == Core::FE::CellType::hex8 || distype_source == Core::FE::CellType::hex20 ||
        distype_source == Core::FE::CellType::hex27)
    {
      if (source_xi[0] < -1.0 - tol || source_xi[1] < -1.0 - tol || source_xi[2] < -1.0 - tol ||
          source_xi[0] > 1.0 + tol || source_xi[1] > 1.0 + tol || source_xi[2] > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype_source == Core::FE::CellType::pyramid5)
    {
      if (source_xi[2] < 0.0 - tol || -source_xi[0] + source_xi[2] > 1.0 + tol ||
          source_xi[0] + source_xi[2] > 1.0 + tol || -source_xi[1] + source_xi[2] > 1.0 + tol ||
          source_xi[1] + source_xi[2] > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype_source == Core::FE::CellType::tet4 ||
             distype_source == Core::FE::CellType::tet10)
    {
      if (source_xi[0] < 0.0 - tol || source_xi[1] < 0.0 - tol || source_xi[2] < 0.0 - tol ||
          (source_xi[0] + source_xi[1] + source_xi[2]) > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype_source == Core::FE::CellType::tri3 ||
             distype_source == Core::FE::CellType::tri6)
    {
      if (source_xi[0] < 0.0 - tol || source_xi[1] < 0.0 - tol ||
          (source_xi[0] + source_xi[1]) > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype_source == Core::FE::CellType::quad4 ||
             distype_source == Core::FE::CellType::quad8 ||
             distype_source == Core::FE::CellType::quad9)
    {
      if (source_xi[0] < -1.0 - tol || source_xi[1] < -1.0 - tol || source_xi[0] > 1.0 + tol ||
          source_xi[1] > 1.0 + tol)
      {
        return false;
      }
    }
    else
      FOUR_C_THROW("Wrong element type!");

    // check GP projection (TARGET)
    if (distype_target == Core::FE::CellType::hex8 || distype_target == Core::FE::CellType::hex20 ||
        distype_target == Core::FE::CellType::hex27)
    {
      if (target_xi[0] < -1.0 - tol || target_xi[1] < -1.0 - tol || target_xi[2] < -1.0 - tol ||
          target_xi[0] > 1.0 + tol || target_xi[1] > 1.0 + tol || target_xi[2] > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype_target == Core::FE::CellType::pyramid5)
    {
      if (target_xi[2] < 0.0 - tol || -target_xi[0] + target_xi[2] > 1.0 + tol ||
          target_xi[0] + target_xi[2] > 1.0 + tol || -target_xi[1] + target_xi[2] > 1.0 + tol ||
          target_xi[1] + target_xi[2] > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype_target == Core::FE::CellType::tet4 ||
             distype_target == Core::FE::CellType::tet10)
    {
      if (target_xi[0] < 0.0 - tol || target_xi[1] < 0.0 - tol || target_xi[2] < 0.0 - tol ||
          (target_xi[0] + target_xi[1] + target_xi[2]) > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype_target == Core::FE::CellType::tri3 ||
             distype_target == Core::FE::CellType::tri6)
    {
      if (target_xi[0] < 0.0 - tol || target_xi[1] < 0.0 - tol ||
          (target_xi[0] + target_xi[1]) > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype_target == Core::FE::CellType::quad4 ||
             distype_target == Core::FE::CellType::quad8 ||
             distype_target == Core::FE::CellType::quad9)
    {
      if (target_xi[0] < -1.0 - tol || target_xi[1] < -1.0 - tol || target_xi[0] > 1.0 + tol ||
          target_xi[1] > 1.0 + tol)
      {
        return false;
      }
    }
    else
      FOUR_C_THROW("Wrong element type!");

    return true;
  };

  /*----------------------------------------------------------------------*
   |  Check mapping                                            farah 06/14|
   *----------------------------------------------------------------------*/
  template <Core::FE::CellType distype>
  bool check_mapping(Core::Elements::Element& ele, double* xi)
  {
    // check node projection
    double tol = 1e-5;
    if (distype == Core::FE::CellType::hex8 || distype == Core::FE::CellType::hex20 ||
        distype == Core::FE::CellType::hex27)
    {
      if (xi[0] < -1.0 - tol || xi[1] < -1.0 - tol || xi[2] < -1.0 - tol || xi[0] > 1.0 + tol ||
          xi[1] > 1.0 + tol || xi[2] > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype == Core::FE::CellType::pyramid5)
    {
      if (xi[2] < 0.0 - tol || -xi[0] + xi[2] > 1.0 + tol || xi[0] + xi[2] > 1.0 + tol ||
          -xi[1] + xi[2] > 1.0 + tol || xi[1] + xi[2] > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype == Core::FE::CellType::tet4 || distype == Core::FE::CellType::tet10)
    {
      if (xi[0] < 0.0 - tol || xi[1] < 0.0 - tol || xi[2] < 0.0 - tol ||
          (xi[0] + xi[1] + xi[2]) > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype == Core::FE::CellType::quad4 || distype == Core::FE::CellType::quad8 ||
             distype == Core::FE::CellType::quad9)
    {
      if (xi[0] < -1.0 - tol || xi[1] < -1.0 - tol || xi[0] > 1.0 + tol || xi[1] > 1.0 + tol)
      {
        return false;
      }
    }
    else if (distype == Core::FE::CellType::tri3 || distype == Core::FE::CellType::tri6)
    {
      if (xi[0] < -tol || xi[1] < -tol || xi[0] > 1.0 + tol || xi[1] > 1.0 + tol ||
          xi[0] + xi[1] > 1.0 + 2 * tol)
      {
        return false;
      }
    }
    else
    {
      FOUR_C_THROW("ERROR: Wrong element type!");
    }

    return true;
  }

  //===================================
  template <Core::FE::CellType distype_source, Core::FE::CellType distype_target>
  bool vol_mortar_ele_based_gp(Core::Elements::Element& source_ele,
      Core::Elements::Element* target_ele, std::vector<int>& foundeles, int& found, int& gpid,
      double& jac, double& wgt, double& gpdist, double* Axi, double* AuxXi, double* globgp,
      DualQuad& dq, Shapefcn& shape, Core::LinAlg::SparseMatrix& dmatrix_A,
      Core::LinAlg::SparseMatrix& mmatrix_A, const Core::FE::Discretization& Adis,
      const Core::FE::Discretization& Bdis, int dofseta, int dofsetb,
      const Core::LinAlg::Map& PAB_dofrowmap, const Core::LinAlg::Map& PAB_dofcolmap);

  // evaluation of nts approach
  template <Core::FE::CellType distype>
  bool cons_interpolator_eval(Core::Nodes::Node* node, Core::Elements::Element* ele,
      Core::LinAlg::SparseMatrix& pmatrix, const Core::FE::Discretization& nodediscret,
      const Core::FE::Discretization& elediscret, std::vector<int>& foundeles, int& found,
      int& eleid, double& dist, double* AuxXi, double* nodepos, std::pair<int, int>& dofset,
      const Core::LinAlg::Map& P_dofrowmap, const Core::LinAlg::Map& P_dofcolmap);

  //===================================
  // Alternative to VolMortar coupling:
  class ConsInterpolator
  {
   public:
    // constructor
    ConsInterpolator();

    // destructor
    virtual ~ConsInterpolator() = default;

    /*!
    \brief interpolation functionality

    */
    void interpolate(Core::Nodes::Node* node, Core::LinAlg::SparseMatrix& pmatrix_,
        const Core::FE::Discretization& nodediscret, const Core::FE::Discretization& elediscret,
        std::vector<int>& foundeles, std::pair<int, int>& dofset,
        const Core::LinAlg::Map& P_dofrowmap, const Core::LinAlg::Map& P_dofcolmap);
  };

}  // namespace Coupling::VolMortar

FOUR_C_NAMESPACE_CLOSE

#endif
