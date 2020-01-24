mod hal_state;
use hal_state::HalState;

use simple_logger;
use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() -> Result<(), &'static str> {
    // simple_logger::init().map_err(|_| "Could not start logger")?;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = HalState::new(&window)?;

    state.draw_clear_frame([0.8, 0.5, 0.2, 1.0])?;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                }
                | WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit
                },
                _ => { },
            },
            Event::RedrawRequested(_) => {
                if let Err(e) = state.draw_clear_frame([0.8, 0.5, 0.2, 1.0]) {
                    println!("{}", e);
                }
            }
            Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            _ => (),
        }
    })
}
