overlap = 0.06;
over_left = 0.9;

Point(1) = {0,0,0};  // Bottom left
Point(2) = {over_left, 0, 0}; // Bottom right
Point(3) = {over_left, 1, 0}; // Top right
Point(4) = {0, 1, 0}; // Top left

Point(5) = {over_left + overlap, 0, 0}; // Bot point
Point(6) = {over_left + overlap, 1, 0}; // Top point

Point(7) = {1,0,0};
Point(8) = {1,1,0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {2, 5};
Line(6) = {5, 6};
Line(7) = {6,3};

Line(8) = {5, 7};
Line(9) = {7,8};
Line(10) = {8, 6};

Curve Loop(1) = {1,2,3,4};
Curve Loop(2) = {5,6,7,-2};
Curve Loop(3) = {8,9,10,-6};

Surface(1) = {1};
Surface(2) = {2};
Surface(3) = {3};

Transfinite Line {1, -3} = 45 Using Progression 1;
Transfinite Line {4, -2, -6, -9} = 50 Using Progression 1;

Transfinite Line {5, -7} = 4 Using Progression 1;

Transfinite Line {8, -10} = 10 Using Progression 0.5;

Transfinite Surface {1,2,3};

Physical Line(1) = {1,5,8,9,10,7,3,4};
Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};



