h0 = 2.;
h1 = 0.2*h0;

Point(1) = {0, 0, 0, h1};
Point(2) = {5, 0, 0, h1};
Point(3) = {10, 0, 0, h0};
Point(4) = {10, 18, 0, h0};
Point(5) = {0, 18, 0, h0};
Point(6) = {0, 5, 0, h1};

Line(1) = {2, 3};
Line(2) = {3, 4};
Line(3) = {4, 5};
Line(4) = {5, 6};
Circle(5) = {2, 1, 6};

Line Loop(1) = {5, -4, -3, -2, -1};
Plane Surface(1) = {1};

Physical Line("hsymmetry") = {1};
Physical Line("vsymmetry") = {4};
Physical Line("load") = {3};
Physical Surface("interior") = {1};
