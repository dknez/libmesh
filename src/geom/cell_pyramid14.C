// The libMesh Finite Element Library.
// Copyright (C) 2002-2016 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


// C++ includes

// Local includes
#include "libmesh/side.h"
#include "libmesh/cell_pyramid14.h"
#include "libmesh/edge_edge3.h"
#include "libmesh/face_tri6.h"
#include "libmesh/face_quad9.h"

namespace libMesh
{




// ------------------------------------------------------------
// Pyramid14 class static member initializations
const unsigned int Pyramid14::side_nodes_map[5][9] =
  {
    {0, 1, 4, 5, 10,  9, 99, 99, 99}, // Side 0 (front)
    {1, 2, 4, 6, 11, 10, 99, 99, 99}, // Side 1 (right)
    {2, 3, 4, 7, 12, 11, 99, 99, 99}, // Side 2 (back)
    {3, 0, 4, 8,  9, 12, 99, 99, 99}, // Side 3 (left)
    {0, 3, 2, 1,  8,  7,  6,  5, 13}  // Side 4 (base)
  };

const unsigned int Pyramid14::edge_nodes_map[8][3] =
  {
    {0, 1,  5}, // Edge 0
    {1, 2,  6}, // Edge 1
    {2, 3,  7}, // Edge 2
    {0, 3,  8}, // Edge 3
    {0, 4,  9}, // Edge 4
    {1, 4, 10}, // Edge 5
    {2, 4, 11}, // Edge 6
    {3, 4, 12}  // Edge 7
  };



// ------------------------------------------------------------
// Pyramid14 class member functions

bool Pyramid14::is_vertex(const unsigned int i) const
{
  if (i < 5)
    return true;
  return false;
}



bool Pyramid14::is_edge(const unsigned int i) const
{
  if (i < 5)
    return false;
  if (i == 13)
    return false;
  return true;
}



bool Pyramid14::is_face(const unsigned int i) const
{
  if (i == 13)
    return true;
  return false;
}



bool Pyramid14::is_node_on_side(const unsigned int n,
                                const unsigned int s) const
{
  libmesh_assert_less (s, n_sides());
  for (unsigned int i = 0; i != 9; ++i)
    if (side_nodes_map[s][i] == n)
      return true;
  return false;
}

bool Pyramid14::is_node_on_edge(const unsigned int n,
                                const unsigned int e) const
{
  libmesh_assert_less (e, n_edges());
  for (unsigned int i = 0; i != 3; ++i)
    if (edge_nodes_map[e][i] == n)
      return true;
  return false;
}



bool Pyramid14::has_affine_map() const
{
  // TODO: If the base is a parallelogram and all the triangular faces are planar,
  // the map should be linear, but I need to test this theory...
  return false;
}



dof_id_type Pyramid14::key (const unsigned int s) const
{
  libmesh_assert_less (s, this->n_sides());

  switch (s)
    {
    case 0: // triangular face 1
    case 1: // triangular face 2
    case 2: // triangular face 3
    case 3: // triangular face 4
      return Pyramid::key(s);

    case 4:  // the quad face at z=0
      return this->compute_key (this->node(13));

    default:
      libmesh_error_msg("Invalid side s = " << s);
    }

  libmesh_error_msg("We'll never get here!");
  return 0;
}



UniquePtr<Elem> Pyramid14::build_side (const unsigned int i, bool proxy) const
{
  libmesh_assert_less (i, this->n_sides());

  if (proxy)
    {
      switch (i)
        {
        case 0:
        case 1:
        case 2:
        case 3:
          return UniquePtr<Elem>(new Side<Tri6,Pyramid14>(this,i));

        case 4:
          return UniquePtr<Elem>(new Side<Quad9,Pyramid14>(this,i));

        default:
          libmesh_error_msg("Invalid side i = " << i);
        }
    }

  else
    {
      // Create NULL pointer to be initialized, returned later.
      Elem * face = NULL;

      switch (i)
        {
        case 0: // triangular face 1
        case 1: // triangular face 2
        case 2: // triangular face 3
        case 3: // triangular face 4
          {
            face = new Tri6;
            break;
          }
        case 4: // the quad face at z=0
          {
            face = new Quad9;
            break;
          }
        default:
          libmesh_error_msg("Invalid side i = " << i);
        }

      face->subdomain_id() = this->subdomain_id();

      // Set the nodes
      for (unsigned n=0; n<face->n_nodes(); ++n)
        face->set_node(n) = this->get_node(Pyramid14::side_nodes_map[i][n]);

      return UniquePtr<Elem>(face);
    }

  libmesh_error_msg("We'll never get here!");
  return UniquePtr<Elem>();
}



UniquePtr<Elem> Pyramid14::build_edge (const unsigned int i) const
{
  libmesh_assert_less (i, this->n_edges());

  return UniquePtr<Elem>(new SideEdge<Edge3,Pyramid14>(this,i));
}



void Pyramid14::connectivity(const unsigned int libmesh_dbg_var(sc),
                             const IOPackage iop,
                             std::vector<dof_id_type> & /*conn*/) const
{
  libmesh_assert(_nodes);
  libmesh_assert_less (sc, this->n_sub_elem());
  libmesh_assert_not_equal_to (iop, INVALID_IO_PACKAGE);

  switch (iop)
    {
    case TECPLOT:
      {
        // TODO
        libmesh_not_implemented();
      }

    case VTK:
      {
        // TODO
        libmesh_not_implemented();
      }

    default:
      libmesh_error_msg("Unsupported IO package " << iop);
    }
}



unsigned int Pyramid14::n_second_order_adjacent_vertices (const unsigned int n) const
{
  switch (n)
    {
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
      return 2;

    case 13:
      return 4;

    default:
      libmesh_error_msg("Invalid node n = " << n);
    }

  libmesh_error_msg("We'll never get here!");
  return libMesh::invalid_uint;
}


unsigned short int Pyramid14::second_order_adjacent_vertex (const unsigned int n,
                                                            const unsigned int v) const
{
  libmesh_assert_greater_equal (n, this->n_vertices());
  libmesh_assert_less (n, this->n_nodes());

  switch (n)
    {
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
      {
        libmesh_assert_less (v, 2);

        // This is the analog of the static, const arrays
        // {Hex,Prism,Tet10}::_second_order_adjacent_vertices
        // defined in the respective source files... possibly treat
        // this similarly once the Pyramid13 has been added?
        unsigned short node_list[8][2] =
          {
            {0,1},
            {1,2},
            {2,3},
            {0,3},
            {0,4},
            {1,4},
            {2,4},
            {3,4}
          };

        return node_list[n-5][v];
      }

      // mid-face node on bottom
    case 13:
      {
        libmesh_assert_less (v, 4);

        // The vertex nodes surrounding node 13 are 0, 1, 2, and 3.
        // Thus, the v'th node is simply = v.
        return cast_int<unsigned short>(v);
      }

    default:
      libmesh_error_msg("Invalid n = " << n);

    }

  libmesh_error_msg("We'll never get here!");
  return static_cast<unsigned short int>(-1);
}

} // namespace libMesh
