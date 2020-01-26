#[derive(Default, Copy, Clone)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

pub const WINDOW_NAME: &str = "Learn Gfx";

#[rustfmt::skip]
pub const QUAD_DATA: [f32; 8] = [
    -0.5, -0.5, 
    -0.5,  0.5, 
     0.5,  0.5, 
     0.5, -0.5, 
];

#[rustfmt::skip]
pub const QUAD_INDICES: [u16; 6] = [
    0, 1, 2,
    0, 2, 3,
];

// Matches mailbox presentation, which
// uses three images for vsync
pub const FRAMES_IN_FLIGHT: usize = 3;
