use crate::{utils, GfxState};
use arrayvec::ArrayVec;
use gfx_hal::{
    buffer::IndexBufferView,
    command::{self, CommandBuffer},
    device::Device,
    pso,
    queue::{CommandQueue, Submission},
    window::Swapchain,
    IndexType,
};
use std::mem;

pub fn draw_frame(
    state: &mut GfxState,
    color: [f32; 4],
    mouse: utils::Vec2,
) -> Result<(), &'static str> {
    let image_available = &state.image_available_semaphores[state.current_frame];
    let render_finished = &state.render_finished_semaphores[state.current_frame];
    state.current_frame = (state.current_frame + 1) % utils::FRAMES_IN_FLIGHT;

    let (image_i, _suboptimal) = unsafe {
        state
            .swapchain
            .acquire_image(core::u64::MAX, Some(image_available), None)
    }
    .map_err(|_| "Failed to acquire an image from the swapchain")?;
    let image_i = image_i as usize;

    let flight_fence = &state.in_flight_fences[image_i];
    unsafe { state.device.wait_for_fence(flight_fence, core::u64::MAX) }
        .map_err(|_| "Failed to wait on the fence")?;
    unsafe { state.device.reset_fence(flight_fence) }.map_err(|_| "Failed to reset the fence")?;

    state.vertices.load_data(&state.device, &utils::QUAD_DATA)?;
    state
        .indices
        .load_data(&state.device, &utils::QUAD_INDICES)?;

    {
        let commands = &mut state.command_buffers[image_i];
        let buffers: ArrayVec<[_; 1]> = [(&*state.vertices.buffer, 0)].into();
        unsafe {
            // A primary command buffer may optionally call into
            // secondary command buffers, which are usually prerecorded
            // steps the primary buffer can reuse or switch between
            commands.begin_primary(command::CommandBufferFlags::EMPTY);
            commands.bind_graphics_pipeline(&state.pipeline.handle);
            commands.bind_vertex_buffers(0, buffers);
            commands.bind_index_buffer(IndexBufferView {
                buffer: &state.indices.buffer,
                offset: 0,
                index_type: IndexType::U16,
            });
            commands.push_graphics_constants(
                &state.pipeline.layout,
                pso::ShaderStageFlags::VERTEX,
                0,
                &[
                    mem::transmute::<f32, u32>(mouse.x),
                    mem::transmute::<f32, u32>(mouse.y),
                ],
            );
            // A renderpass is a bunch of work done with a
            // particular set of attachments. 
            commands.begin_render_pass(
                &state.render_pass,
                &state.framebuffers[image_i],
                state.content_size,
                [command::ClearValue {
                    color: command::ClearColor { float32: color },
                }]
                .iter(),
                command::SubpassContents::Inline,
            );
            // Subpasses may change attachment behaviour,
            // for example changing intermediate buffers 
            // from write to read in the case of 
            // deferred rendering. Subpasses are also likely
            // to be faster, and their use is preferrable where
            // limitations don't restrict their use. Each pixel of output
            // can only read its corresponding pixel of input, 
            // so things like blur are not possible within subpasses.
            commands.draw_indexed(0..6, 0, 0..1);
            commands.end_render_pass();
            commands.finish();
        }
    }

    let submission = {
        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        Submission {
            command_buffers: &state.command_buffers.get(image_i),
            wait_semaphores,
            signal_semaphores,
        }
    };

    let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
    let command_queue = &mut state.queue_group.queues[0];
    unsafe {
        command_queue.submit(submission, Some(flight_fence));
        state
            .swapchain
            .present(command_queue, image_i as u32, present_wait_semaphores)
    }
    // Discard suboptimal warning
    .map(|_| ())
    .map_err(|_| "Failed to present into the swapchain")
}
