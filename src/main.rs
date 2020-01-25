mod hal_state;
use hal_state::HalState;

use fern::colors::ColoredLevelConfig;
use winit::{
    event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub mod utils;
use utils::Vec2;

mod buffer_info;
pub use buffer_info::BufferInfo;

mod pipeline_info;
pub use pipeline_info::PipelineInfo;

#[derive(Default, Copy, Clone)]
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
        .level(log::LevelFilter::Error)
        .chain(std::io::stdout())
        .apply()
        .map_err(|_| "Failed to start logger")?;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(utils::WINDOW_NAME)
        .build(&event_loop)
        .unwrap();

    let mut hal_state = HalState::new(&window)?;
    let mut input_state = InputState::default();

    render(&mut hal_state, &input_state);
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
    if let Err(e) = hal_state.draw_frame([0.2, 0.2, 0.2, 1.0], input_state.mouse) {
        println!("{}", e);
    }
}
