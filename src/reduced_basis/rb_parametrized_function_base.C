// rbOOmit: An implementation of the Certified Reduced Basis method.
// Copyright (C) 2009, 2010 David J. Knezevic

// This file is part of rbOOmit.

// rbOOmit is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// rbOOmit is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

// libmesh includes
#include "libmesh/rb_parametrized_function_base.h"
#include "libmesh/int_range.h"
#include "libmesh/point.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/utility.h"
#include "libmesh/rb_parameters.h"
#include "libmesh/system.h"
#include "libmesh/elem.h"
#include "libmesh/fem_context.h"

namespace libMesh
{

RBParametrizedFunctionBase::RBParametrizedFunctionBase()
:
requires_xyz_perturbations(false),
is_lookup_table(false),
fd_delta(1.e-6)
{}

RBParametrizedFunctionBase::~RBParametrizedFunctionBase() = default;

void RBParametrizedFunctionBase::initialize_lookup_table()
{
  // No-op by default, override in subclasses as needed
}

Number RBParametrizedFunctionBase::get_parameter_independent_data(const std::string & property_name,
                                                                  subdomain_id_type sbd_id) const
{
  return libmesh_map_find(libmesh_map_find(_parameter_independent_data, property_name), sbd_id);
}

}
