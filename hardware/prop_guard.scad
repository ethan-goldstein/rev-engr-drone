// Contixo F19 Arvo prop guard: parametric DRAFT. One per motor; print 4 (all four are identical).
//
// Numbers marked EST were estimated from photos (about +/- 3-5 mm). Numbers marked MEASURE are
// guesses. Check both with calipers before printing.
//
// How it mounts: the flat saddle sits on top of the arm, right next to the motor, and two mini zip
// ties (2.5 mm wide) go through the slots and around the arm. The curved wall wraps the outer side
// of the prop, from arm height up past the prop plane.
//
// Open in OpenSCAD (free: brew install --cask openscad), press F6 to render, File > Export > STL.
// Print flat side down, PETG or PLA, 0.2 mm layers, no supports.
// Weight estimate in PLA at the defaults: about 5 g each, 20 g for all four. Weigh the drone first:
// the total must stay under 250 g.

/* [Drone, measure these] */
prop_d       = 110;  // EST: prop tip-to-tip diameter (photos showed 105-111 mm)
prop_z       = 12;   // MEASURE: height of the prop blades above the top of the arm
arm_w        = 11;   // EST: arm width next to the motor
motor_to_arm = 16;   // MEASURE: motor center to where the saddle sits on the arm

/* [Guard] */
clearance = 6;    // gap between prop tips and the wall
wall      = 1.0;  // wall thickness: thinner is lighter, 1.0-1.4 mm prints fine
wall_top  = 3;    // how far the wall reaches above the prop plane
arc_deg   = 220;  // how much of the circle is guarded (outer side; 360 = full ring)
base_t    = 1.6;  // spokes and saddle thickness
spoke_w   = 3;    // spoke width
saddle_l  = 14;   // saddle length along the arm
tie_w     = 2.8;  // zip-tie slot width (for 2.5 mm ties)
tie_t     = 1.4;  // zip-tie slot thickness

$fn = 96;

r_in  = prop_d / 2 + clearance;
r_mid = r_in + wall / 2;
wall_h = prop_z + wall_top;
end_a = arc_deg / 2;  // the arc is centered on +X, pointing away from the body along the arm

// The curved wall.
module wall_arc() {
    rotate([0, 0, -end_a])
        rotate_extrude(angle = arc_deg)
            translate([r_in, 0]) square([wall, wall_h]);
}

// Saddle on the arm (the arm runs from the motor at the origin toward the body, along -X).
module saddle() {
    sx = motor_to_arm + saddle_l / 2;
    difference() {
        translate([-sx - saddle_l / 2, -arm_w / 2 - spoke_w, 0])
            cube([saddle_l, arm_w + 2 * spoke_w, base_t]);
        for (dx = [-saddle_l / 4, saddle_l / 4], side = [-1, 1])  // zip-tie slots on both sides of the arm
            translate([-sx + dx - tie_w / 2, side * (arm_w / 2 + spoke_w / 2) - tie_t / 2, -1])
                cube([tie_w, tie_t, base_t + 2]);
    }
}

// Spokes from the saddle's outer corners to the two ends of the wall, at arm height, under the prop.
module spokes() {
    sx = motor_to_arm + saddle_l / 2;
    for (side = [-1, 1])
        hull() {
            translate([-sx, side * (arm_w / 2 + spoke_w / 2), 0]) cylinder(d = spoke_w, h = base_t);
            rotate([0, 0, side * end_a]) translate([r_mid, 0, 0]) cylinder(d = spoke_w, h = base_t);
        }
}

wall_arc();
saddle();
spokes();
