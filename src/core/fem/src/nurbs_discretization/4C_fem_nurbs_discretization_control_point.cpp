#include "4C_fem_nurbs_discretization_control_point.hpp"

#include "4C_comm_pack_helpers.hpp"

FOUR_C_NAMESPACE_OPEN

Core::FE::Nurbs::ControlPointType Core::FE::Nurbs::ControlPointType::instance_;


Core::Communication::ParObject* Core::FE::Nurbs::ControlPointType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  std::vector<double> dummycoord(3, 999.0);
  double dummyweight = 999.;
  Core::FE::Nurbs::ControlPoint* object =
      new Core::FE::Nurbs::ControlPoint(-1, dummycoord, dummyweight, -1);
  object->unpack(buffer);
  return object;
}

/*
  Standard ctor
 */
Core::FE::Nurbs::ControlPoint::ControlPoint(
    int id, const std::vector<double>& coords, const double weight, const int owner)
    : Core::Nodes::Node(id, coords, owner), w_(weight)
{
  return;
}

/*
  Copy Constructor

  Makes a deep copy of a control point

*/
Core::FE::Nurbs::ControlPoint::ControlPoint(const Core::FE::Nurbs::ControlPoint& old)
    : Core::Nodes::Node(old), w_(old.w())
{
  return;
}

/*
  Deep copy the derived class and return pointer to it

*/
Core::FE::Nurbs::ControlPoint* Core::FE::Nurbs::ControlPoint::clone() const
{
  Core::FE::Nurbs::ControlPoint* newcp = new Core::FE::Nurbs::ControlPoint(*this);

  return newcp;
}


/*
  Pack this class so it can be communicated

  Pack and Unpack are used to communicate this control point

*/
void Core::FE::Nurbs::ControlPoint::pack(Core::Communication::PackBuffer& data) const
{
  Core::Communication::PackBuffer::SizeMarker sm(data);

  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class of control point
  Core::Nodes::Node::pack(data);
  // add weight
  add_to_pack(data, &w_, sizeof(double));

  return;
}

/*
  Unpack data from a char vector into this class

  Pack and Unpack are used to communicate this control point
*/
void Core::FE::Nurbs::ControlPoint::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Node
  std::vector<char> basedata(0);
  extract_from_pack(buffer, basedata);
  Core::Communication::UnpackBuffer basedata_buffer(basedata);
  Core::Nodes::Node::unpack(basedata_buffer);
  // extract weight
  extract_from_pack(buffer, w_);

  return;
}

/*
  Print this control point
*/
void Core::FE::Nurbs::ControlPoint::print(std::ostream& os) const
{
  os << "Control Point :";
  Core::Nodes::Node::print(os);
  os << "\n+ additional weight ";
  os << w_ << "\n";
  return;
}

FOUR_C_NAMESPACE_CLOSE
