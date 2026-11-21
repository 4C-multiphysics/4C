// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_anisotropy_extension.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_parobject.hpp"
#include "4C_mat_anisotropy_utils.hpp"

FOUR_C_NAMESPACE_OPEN

template <unsigned int numfib>
Mat::FiberAnisotropyExtension<numfib>::FiberAnisotropyExtension(
    const std::shared_ptr<Elastic::StructuralTensorStrategyBase>& stucturalTensorStrategy)
    : fibers_(0), fiber_structural_tensors_(0), structural_tensor_strategy_(stucturalTensorStrategy)
{
}

template <unsigned int numfib>
Mat::FiberAnisotropyExtension<numfib>::FiberAnisotropyExtension()
    : fibers_(0), fiber_structural_tensors_(0), structural_tensor_strategy_(nullptr)
{
}

template <unsigned int numfib>
void Mat::FiberAnisotropyExtension<numfib>::compute_structural_tensors()
{
  Mat::compute_structural_tensors<Core::LinAlg::SymmetricTensor<double, 3, 3>, numfib>(
      fibers_, fiber_structural_tensors_, structural_tensor_strategy_);
}

template <unsigned int numfib>
void Mat::FiberAnisotropyExtension<numfib>::set_fibers(
    int gp, const std::array<Core::LinAlg::Tensor<double, 3>, numfib>& fibers)
{
  if (gp >= numgp_)
  {
    FOUR_C_THROW(
        "The current Gauss point {} is out of range of the expected number of Gauss points {}.", gp,
        numgp_);
  }

  if (fibers_.empty())
  {
    fibers_.resize(get_fibers_per_element());
  }

  // Store fibers
  for (std::vector<Core::LinAlg::Tensor<double, 3>>::size_type i = 0; i < fibers.size(); ++i)
  {
    fibers_[gp][i] = fibers[i];
  }

  // Compute needed structural tensors
  compute_needed_structural_tensors();

  // Call the notifier method
  on_fibers_initialized();
}

template <unsigned int numfib>
void Mat::FiberAnisotropyExtension<numfib>::set_fibers(
    const std::vector<std::array<Core::LinAlg::Tensor<double, 3>, numfib>>& fibers)
{
  fibers_ = fibers;

  // Compute needed structural tensors
  compute_needed_structural_tensors();

  // Call the notifier method
  on_fibers_initialized();
}

template <unsigned int numfib>
void Mat::FiberAnisotropyExtension<numfib>::compute_needed_structural_tensors()
{
  if (static_cast<bool>(tensor_flags_ & STRUCTURAL_TENSOR))
  {
    compute_structural_tensors();
  }
}

template <unsigned int numfib>
const Core::LinAlg::Tensor<double, 3>& Mat::FiberAnisotropyExtension<numfib>::get_fiber(
    int gp, int i) const
{
  switch (fiber_location_)
  {
    case FiberLocation::ElementFibers:
      return fibers_[GPDEFAULT][i];
    case FiberLocation::GPFibers:
      return fibers_[gp][i];
    default:
      FOUR_C_THROW(
          "You have not specified, whether you want fibers on GP level or on element level.");
  }

  // just for compilation reasons. We will never land here because of the FOUR_C_THROW() above
  std::abort();
}

template <unsigned int numfib>
const Core::LinAlg::SymmetricTensor<double, 3, 3>&
Mat::FiberAnisotropyExtension<numfib>::get_structural_tensor(int gp, int i) const
{
  if (not static_cast<bool>(tensor_flags_ & STRUCTURAL_TENSOR))
  {
    FOUR_C_THROW("You have not specified that you need the structural tensor.");
  }
  switch (fiber_location_)
  {
    case FiberLocation::ElementFibers:
      return fiber_structural_tensors_[GPDEFAULT][i];
    case FiberLocation::GPFibers:
      return fiber_structural_tensors_[gp][i];
    default:
      FOUR_C_THROW(
          "You have not specified, whether you want fibers on GP level or on element level.");
  }

  // just for compilation reasons. We will never land here because of the FOUR_C_THROW() above
  std::abort();
}

template <unsigned int numfib>
void Mat::FiberAnisotropyExtension<numfib>::pack_anisotropy(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, numgp_);
  add_to_pack(data, fibers_);
  add_to_pack(data, fiber_structural_tensors_);
  add_to_pack(data, tensor_flags_);
  add_to_pack(data, fiber_location_);
}

template <unsigned int numfib>
void Mat::FiberAnisotropyExtension<numfib>::unpack_anisotropy(
    Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, numgp_);
  extract_from_pack(buffer, fibers_);
  extract_from_pack(buffer, fiber_structural_tensors_);
  extract_from_pack(buffer, tensor_flags_);
  extract_from_pack(buffer, fiber_location_);
}

template <unsigned int numfib>
void Mat::FiberAnisotropyExtension<numfib>::set_fiber_location(FiberLocation location)
{
  fiber_location_ = location;
}

template <unsigned int numfib>
int Mat::FiberAnisotropyExtension<numfib>::get_virtual_gauss_point(const int gp) const
{
  if (fiber_location_ == FiberLocation::ElementFibers)
  {
    return GPDEFAULT;
  }

  return gp;
}

template <unsigned int numfib>
int Mat::FiberAnisotropyExtension<numfib>::get_fibers_per_element() const
{
  if (fiber_location_ == FiberLocation::ElementFibers)
  {
    return 1;
  }

  return numgp_;
}

// explicit instantiations of template classes
template class Mat::FiberAnisotropyExtension<1u>;
template class Mat::FiberAnisotropyExtension<2u>;
FOUR_C_NAMESPACE_CLOSE
