// Inverted airfoil above a moving ground: gmsh template.
// groundGen.py fills in the placeholders (names between @ signs) and writes airfoil.geo.
//
// Changes from the original airfoil_template.geo:
//   - the bottom edge of the domain is the ground, at a variable height
//   - a refinement box gives finer cells in the gap between airfoil and ground
//   - the domain size is a parameter (original: 5 chords from the airfoil)
//   - optional boundary layer (--bl): thin quadrilateral cells along the airfoil.
//     The airfoil is then one spline plus a short straight trailing-edge segment,
//     so that the trailing-edge corners are real corners where gmsh can fan out
//     the layers.
// Without --bl the airfoil curve and the surface numbering are exactly as in the
// original.

@POINTS@

@AIRFOIL_CURVES@

edge_lc = @EDGE_LC@;
Point(1900) = { @X_OUT@, @Y_TOP@, 0, edge_lc};
Point(1901) = { @X_OUT@, @GROUND_Y@, 0, edge_lc};
Point(1902) = { @X_IN@, @GROUND_Y@, 0, edge_lc};
Point(1903) = { @X_IN@, @Y_TOP@, 0, edge_lc};

Line(1) = {1900,1901};
Line(2) = {1901,1902};
Line(3) = {1902,1903};
Line(4) = {1903,1900};

Line Loop (1) = {1,2,3,4};
Line Loop (2) = {@AIRFOIL_LOOP@};
Plane Surface(1) = {1,2};

@EXTRUDE_AND_PHYSICAL@

// Finer cells between the airfoil and the ground (about 10 across the gap),
// growing smoothly back to the normal size over a 0.3-chord transition.
Field[1] = Box;
Field[1].VIn = @LC_GAP@;
Field[1].VOut = edge_lc;
Field[1].XMin = -0.5;
Field[1].XMax = 2.0;
Field[1].YMin = @BOX_YMIN@;
Field[1].YMax = @BOX_YMAX@;
Field[1].ZMin = -1;
Field[1].ZMax = 2;
Field[1].Thickness = 0.3;
Background Field = 1;

@BL_FIELD@
