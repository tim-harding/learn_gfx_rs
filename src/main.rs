pub mod window;

use window::GfxWindow;
use winit::{Event, WindowEvent};
use simple_logger;

#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;

fn main() {
    match simple_logger::init() {
        Err(_) => { println!("Couldn't start logger.") },
        _ => { },
    };
    let mut window = GfxWindow::new();

    let mut running = true;
    while running {
        window.events_loop.poll_events(|event| match event {
            Event::WindowEvent { 
                event: WindowEvent::CloseRequested,
                ..
            } => running = false,
            _ => (),
        });
    }
}
