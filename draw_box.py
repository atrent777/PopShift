"""
Make a PDB of a fake molecule that is in the shape of the
docking box you are considering using--for co-visualizing 
with aligned frames.
    Copyright (C) 2023 Louis G. Smith 

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""
import loos
import argparse as ap
from sys import stdout

p = ap.ArgumentParser(argument_default=True)
p.add_argument('center_x', type=float, help='Center x coordinate')
p.add_argument('center_y', type=float, help='Center y coordinate')
p.add_argument('center_z', type=float, help='Center z coordinate')
p.add_argument('edge_length_x', type=int, help='X edge length')
p.add_argument('edge_length_y', type=int, help='Y edge length')
p.add_argument('edge_length_z', type=int, help='Z edge length')
p.add_argument('-g', '--grid-spacing', type=float, default=1.0,
               help='Grid spacing for autodock-style tools; will be multiplied '
                    'through each edge length.')
args = p.parse_args()


box = loos.AtomicGroup()
center_crd = loos.GCoord(args.center_x, args.center_y, args.center_z)
center = loos.Atom(1, 'cen', center_crd)
gs = args.grid_spacing
half_x = (args.edge_length_x / 2) * gs
half_y = (args.edge_length_y / 2) * gs
half_z = (args.edge_length_z / 2) * gs
longest = max(half_x, half_y, half_z) * 2
updown = dict(u=1, d=-1)
count = 1
for updown_x in updown:
    for updown_y in updown:
        for updown_z in updown:
            count += 1
            atom_name_fstring = f'{updown_x}{updown_y}{updown_z}'
            xf, yf, zf = updown[updown_x], updown[updown_y], updown[updown_z]
            corner_crd = loos.GCoord(xf * half_x, yf * half_y, zf * half_z) \
                + center_crd
            corner = loos.Atom(count, atom_name_fstring, corner_crd)

            box.append(corner)
box.findBonds(longest + gs/2)
box.append(center)
pdb = loos.PDB.fromAtomicGroup(box)
stdout.write(str(pdb))
