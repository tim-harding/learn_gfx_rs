extern crate winit;

use winit::{EventsLoop, WindowBuilder, Window};

pub struct GfxWindow {
    pub window: Window,
    pub events_loop: EventsLoop,
}

impl GfxWindow {
    pub fn new() -> Self {
        let mut events_loop = EventsLoop::new();
        let window = WindowBuilder::new()
            .with_title("Learn gfx-rs")
            .build(&events_loop)
            .expect("Failed to build window");
        Self { window, events_loop }
    }
}