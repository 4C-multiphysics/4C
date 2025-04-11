// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FEM_GENERAL_UTILS_GAUSSPOINTS_HPP
#define FOUR_C_FEM_GENERAL_UTILS_GAUSSPOINTS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_geometry_element_coordtrafo.hpp"

#include <map>
#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  /// base class of gauss point collection
  class GaussPoints
  {
   public:
    virtual ~GaussPoints() = default;
    /// number of gauss points
    virtual int num_points() const = 0;

    /// spacial dimension
    virtual int num_dimension() const = 0;

    /// gauss point coordinates
    virtual const double* point(int point) const = 0;

    /// gauss weight
    virtual double weight(int point) const = 0;

    /// debug print
    virtual void print() const = 0;
  };

  /// specific collected gauss points for xfem usage
  class CollectedGaussPoints : public GaussPoints
  {
   public:
    CollectedGaussPoints(int size = 0) { gp_.reserve(size); }

    void append(double x, double y, double z, double w)
    {
      // if ( w > 1e-15 )
      gp_.push_back(Entry(x, y, z, w));
    }

    void append(const Core::LinAlg::Matrix<2, 1>& xi, double w)
    {
      gp_.push_back(Entry(xi(0), xi(1), 0., w));
    }

    void append(const Core::LinAlg::Matrix<3, 1>& xi, double w)
    {
      gp_.push_back(Entry(xi(0), xi(1), xi(2), w));
    }

    void increase_reserved(int size) { gp_.reserve(gp_.size() + size); }

    int num_points() const override { return gp_.size(); }

    int num_dimension() const override { return 3; }

    const double* point(int point) const override { return &gp_[point].data[0]; }

    double weight(int point) const override { return gp_[point].data[3]; }

    void print() const override
    {
      std::cout << " collected gauss points:\n";
      for (int i = 0; i < num_points(); ++i)
      {
        std::cout << "    ";
        for (int j = 0; j < num_dimension(); ++j) std::cout << gp_[i].data[j] << " ";
        std::cout << gp_[i].data[3] << "\n";
      }
    }

   private:
    struct Entry
    {
      Entry(double x, double y, double z, double w)
      {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
      }
      double data[4];
    };
    std::vector<Entry> gp_;
  };

  /// specific collected gauss points for xfem usage
  ///
  /// This is meant to be an inverted collection, where the first set of gauss
  /// points integrates a whole element and all following sets subtract from
  /// the element.
  class GaussPointsComposite : public GaussPoints
  {
   public:
    explicit GaussPointsComposite(int size) { gp_.reserve(size); }

    void append(std::shared_ptr<GaussPoints> gp) { gp_.push_back(gp); }

    /// number of gauss points
    int num_points() const override
    {
      int numpoints = 0;
      for (std::vector<std::shared_ptr<GaussPoints>>::const_iterator i = gp_.begin();
          i != gp_.end(); ++i)
      {
        std::shared_ptr<GaussPoints> gp = *i;
        numpoints += gp->num_points();
      }
      return numpoints;
    }

    /// spacial dimension
    int num_dimension() const override { return gp_[0]->num_dimension(); }

    /// gauss point coordinates
    const double* point(int point) const override
    {
      std::shared_ptr<GaussPoints> gp = find(point);
      return gp->point(point);
    }

    /// gauss weight
    double weight(int point) const override
    {
      std::shared_ptr<GaussPoints> gp = find(point);
      return gp->weight(point);
      //     if ( gp_[0]==gp )
      //     {
      //       return gp->Weight( point );
      //     }
      //     else
      //     {
      //       return -1. * gp->Weight( point );
      //     }
    }

    /// debug print
    void print() const override
    {
      for (std::vector<std::shared_ptr<GaussPoints>>::const_iterator i = gp_.begin();
          i != gp_.end(); ++i)
      {
        std::shared_ptr<GaussPoints> gp = *i;
        gp->print();
      }
    }

   private:
    std::shared_ptr<GaussPoints> find(int& point) const
    {
      for (std::vector<std::shared_ptr<GaussPoints>>::const_iterator i = gp_.begin();
          i != gp_.end(); ++i)
      {
        std::shared_ptr<GaussPoints> gp = *i;
        int numpoints = gp->num_points();
        if (numpoints > point)
        {
          return gp;
        }
        point -= numpoints;
      }
      FOUR_C_THROW("gauss point not available");
    }

    std::vector<std::shared_ptr<GaussPoints>> gp_;
  };

  /**
   * Create GaussPoints object for a given element type and degree.
   */
  std::shared_ptr<GaussPoints> create_gauss_points(Core::FE::CellType distype, int degree);

  /**
   * Create GaussPoints object for a given element type with a default degree.
   */
  std::shared_ptr<GaussPoints> create_gauss_points_default(Core::FE::CellType distype);

  /// gauss integration interface
  class GaussIntegration
  {
   public:
    /// Very simple internal gauss point iterator.
    ///
    /// With this iterator our gauss point loop looks familiar. Furthermore, we
    /// store a raw pointer and avoid the Teuchos::rcp indirection. This is the gauss
    /// loop, after all!
    class GaussPointIterator
    {
     public:
      GaussPointIterator(GaussPoints* gp, int point) : gp_(gp), point_(point) {}

      /// increment iterator
      void operator++() { point_ += 1; }

      /// point coordinates
      const double* point() const { return gp_->point(point_); }

      /// gauss weight at point
      double weight() const { return gp_->weight(point_); }

      /// actual point we are at
      int operator*() const { return point_; }

      /// actual point we are at
      int operator->() const { return point_; }

      /// compare
      bool operator==(const GaussPointIterator& other) const
      {
        return gp_ == other.gp_ and point_ == other.point_;
      }

      /// compare
      bool operator!=(const GaussPointIterator& other) const
      {
        return gp_ != other.gp_ or point_ != other.point_;
      }

     private:
      GaussPoints* gp_;
      int point_;
    };

    typedef GaussPointIterator iterator;
    typedef GaussPointIterator const_iterator;

    /// construct the optimal (normal) rule for a given element shape
    explicit GaussIntegration(Core::FE::CellType distype);

    /// construct rule for a given element shape
    GaussIntegration(Core::FE::CellType distype, int degree);

    /// construct with a known set of gauss points
    explicit GaussIntegration(std::shared_ptr<GaussPoints> gp) : gp_(gp) {}

    void clear() { gp_ = nullptr; }

    iterator begin() { return GaussPointIterator(&*gp_, 0); }

    const_iterator begin() const { return GaussPointIterator(&*gp_, 0); }

    iterator end() { return GaussPointIterator(&*gp_, gp_->num_points()); }

    const_iterator end() const { return GaussPointIterator(&*gp_, gp_->num_points()); }

    /// number of gauss points
    int num_points() const { return gp_->num_points(); }

    /// spacial dimension
    int num_dimension() const { return gp_->num_dimension(); }

    /// gauss point coordinates
    const double* point(int point) const { return gp_->point(point); }

    /// gauss weight
    double weight(int point) const { return gp_->weight(point); }

    /// debug print
    void print() const { gp_->print(); }

    std::shared_ptr<GaussPoints> points() const { return gp_; }

    void set_points(std::shared_ptr<GaussPoints> gp) { gp_ = gp; }

    /// Create Gauss integration rule of given degree
    template <Core::FE::CellType distype>
    static std::shared_ptr<GaussPoints> create_projected(
        const Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>>& xie,
        int degree)
    {
      std::shared_ptr<GaussPoints> gp = create_gauss_points(distype, degree);
      std::shared_ptr<CollectedGaussPoints> cgp =
          std::make_shared<CollectedGaussPoints>(gp->num_points());

      GaussIntegration intpoints(gp);

      project_gauss_points_local_to_global<distype>(xie, intpoints, *cgp);

      return cgp;
    }

    /// Project the given Gauss points in local coordinate system of the element to its global
    /// coordinate system
    template <Core::FE::CellType distype>
    static void project_gauss_points_local_to_global(
        const Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>>& xie,
        GaussIntegration& intpoints, CollectedGaussPoints& cgp)
    {
      const int nsd = Core::FE::dim<distype>;
      const int nen = Core::FE::num_nodes<distype>;

      Core::LinAlg::Matrix<nen, 1> funct;
      Core::LinAlg::Matrix<nsd, nen> deriv;

      Core::LinAlg::Matrix<nsd, nsd> xjm;
      Core::LinAlg::Matrix<nsd, 1> xi;

      for (Core::FE::GaussIntegration::iterator iquad = intpoints.begin(); iquad != intpoints.end();
          ++iquad)
      {
        Core::LinAlg::Matrix<nsd, 1> eta(iquad.point());

        // cell shape functions and their first derivatives
        Core::FE::shape_function<distype>(eta, funct);
        Core::FE::shape_function_deriv1<distype>(eta, deriv);

        // local coordinates of gauss point w.r.to background element
        xi.multiply(xie, funct);

        // get transposed of the jacobian matrix d x / d \xi
        // xjm(i,j) = deriv(i,k)*xyze(j,k)
        xjm.multiply_nt(deriv, xie);

        double det = xjm.determinant();

        cgp.append(xi, iquad.weight() * det);
      }
    }

    /// Project the given Gauss points in global coordinate system of the element to its local
    /// coordinate system
    template <Core::FE::CellType distype>
    static std::shared_ptr<GaussPoints> project_gauss_points_global_to_local(
        const Core::LinAlg::Matrix<Core::FE::dim<distype>, Core::FE::num_nodes<distype>>& xie,
        GaussIntegration& intpoints, const bool& throw_error = true)
    {
      const int nsd = Core::FE::dim<distype>;
      const int nen = Core::FE::num_nodes<distype>;

      Core::LinAlg::Matrix<nen, 1> funct;
      Core::LinAlg::Matrix<nsd, nen> deriv;

      Core::LinAlg::Matrix<nsd, nsd> xjm;
      Core::LinAlg::Matrix<nsd, 1> xi;

      std::shared_ptr<Core::FE::CollectedGaussPoints> cgp =
          std::make_shared<Core::FE::CollectedGaussPoints>(intpoints.num_points());

      for (Core::FE::GaussIntegration::iterator iquad = intpoints.begin(); iquad != intpoints.end();
          ++iquad)
      {
        Core::LinAlg::Matrix<nsd, 1> glo(iquad.point());

        bool insideele = Core::Geo::current_to_volume_element_coordinates(distype, xie, glo, xi);
        if (not insideele && throw_error)
        {
          FOUR_C_THROW("Given Gauss points not inside the element?");
        }

        // cell shape functions and their first derivatives
        Core::FE::shape_function<distype>(xi, funct);
        Core::FE::shape_function_deriv1<distype>(xi, deriv);

        // get transposed of the jacobian matrix d x / d \xi
        // xjm(i,j) = deriv(i,k)*xyze(j,k)
        xjm.multiply_nt(deriv, xie);

        double det = xjm.determinant();

        cgp->append(xi, iquad.weight() / det);
      }
      return cgp;
    }

   private:
    /// internal collection
    std::shared_ptr<GaussPoints> gp_;
  };

  /*!
   * @brief Create a Gauss integration interface from a given Gauss rule type
   */
  template <Core::FE::CellType celltype, typename GaussRuleType>
  GaussIntegration create_gauss_integration(GaussRuleType rule)
  {
    // setup default integration
    IntPointsAndWeights<Core::FE::dim<celltype>> intpoints(rule);

    // format as Discret::Utils::GaussIntegration
    std::shared_ptr<Core::FE::CollectedGaussPoints> gp =
        std::make_shared<Core::FE::CollectedGaussPoints>();

    std::array<double, 3> xi = {0., 0., 0.};
    for (int i = 0; i < intpoints.ip().nquad; ++i)
    {
      for (int d = 0; d < Core::FE::dim<celltype>; ++d) xi[d] = intpoints.ip().qxg[i][d];
      gp->append(xi[0], xi[1], xi[2], intpoints.ip().qwgt[i]);
    }

    return Core::FE::GaussIntegration(gp);
  }

}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif
