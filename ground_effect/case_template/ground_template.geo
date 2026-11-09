// Inverted airfoil above a moving ground: gmsh template.
// groundGen.py fills in the placeholders (names between @ signs) and writes airfoil.geo.
//
// Same as the original airfoil_template.geo, except:
//   - the bottom edge of the domain is the ground, at a variable height
//   - a refinement box gives finer cells in the gap between airfoil and ground
// The curves are kept in the same order as the original, so the surface
// numbers in the Physical Surface lines stay the same.

@POINTS@

Spline(1000) = {1000:@LAST_POINT@,1000};

edge_lc = 0.2;
Point(1900) = { 5, 5, 0, edge_lc};
Point(1901) = { 5, @GROUND_Y@, 0, edge_lc};
Point(1902) = { -5, @GROUND_Y@, 0, edge_lc};
Point(1903) = { -5, 5, 0, edge_lc};

Line(1) = {1900,1901};
Line(2) = {1901,1902};
Line(3) = {1902,1903};
Line(4) = {1903,1900};

Line Loop (1) = {1,2,3,4};
Line Loop (2) = {1000};
Plane Surface(1) = {1,2};

Extrude {0, 0, 1} {
  Surface{1};
  Layers{1};
  Recombine;
}
Physical Surface("back") = {1027};
Physical Surface("front") = {1};
Physical Surface("top") = {1022};
Physical Surface("exit") = {1010};
Physical Surface("ground") = {1014};
Physical Surface("inlet") = {1018};
Physical Surface("aerofoil") = {1026};
Physical Volume("internal") = {1};

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
