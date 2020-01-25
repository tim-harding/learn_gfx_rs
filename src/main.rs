mod hal_state;
use hal_state::HalState;

use fern::colors::ColoredLevelConfig;
use winit::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Default)]
struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[derive(Default)]
struct InputState {
    pub mouse: Vec2,
}

fn main() -> Result<(), &'static str> {
    let colors = ColoredLevelConfig::default();
    fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "[{}][{}] {}",
                colors.color(record.level()),
                record.target(),
                message
            ))
        })
        .level(log::LevelFilter::Warn)
        .chain(std::io::stdout())
        .apply()
        .map_err(|_| "Failed to start logger")?;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut hal_state = HalState::new(&window)?;
    let mut input_state = InputState::default();

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
                | WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(_) => {
                    // Winit logs some warnings from this,
                    // but it seems to work alright
                    hal_state.free();
                    hal_state = match HalState::new(&window) {
                        Ok(state) => state,
                        Err(e) => panic!(e),
                    }
                }

                WindowEvent::CursorMoved { position, .. } => {
                    input_state.mouse = Vec2 {
                        x: position.x as f32 / window.inner_size().width as f32,
                        y: position.y as f32 / window.inner_size().height as f32,
                    };
                    window.request_redraw();
                }

                _ => {}
            },

            Event::RedrawRequested(_) => {
                render(&mut hal_state, &input_state);
            }

            _ => (),
        }
    });
}

fn render(hal_state: &mut HalState, input_state: &InputState) {
    let color = [input_state.mouse.x, input_state.mouse.y, 0.2, 1.0];
    if let Err(e) = hal_state.draw_frame(color) {
        println!("{}", e);
    }
}
