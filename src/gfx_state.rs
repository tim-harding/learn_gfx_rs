use crate::{
    utils::{self, Vec2},
    BufferInfo, PipelineInfo,
};
use arrayvec::ArrayVec;
use gfx_backend_vulkan as back;
use gfx_hal::{
    adapter::{Gpu, PhysicalDevice},
    buffer::{IndexBufferView, Usage},
    command::{ClearColor, ClearValue, CommandBuffer, CommandBufferFlags, Level, SubpassContents},
    device::Device,
    format::{Aspects, Format, Swizzle},
    image::{Extent, SubresourceRange, ViewKind},
    pass::{
        self, Attachment, AttachmentLayout, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp,
        SubpassDesc,
    },
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{PipelineStage, Rect, ShaderStageFlags},
    queue::{
        family::{QueueFamily, QueueGroup},
        CommandQueue, Submission,
    },
    window::{Extent2D, PresentMode, Surface, Swapchain, SwapchainConfig},
    Backend, Features, IndexType, Instance,
};
use std::{
    mem::{self, ManuallyDrop},
    ops::Drop,
    ptr,
};
use winit::window::Window;

#[rustfmt::skip]
const QUAD_DATA: [f32; 8] = [
    -0.5, -0.5, 
    -0.5,  0.5, 
     0.5,  0.5, 
     0.5, -0.5, 
];

#[rustfmt::skip]
const QUAD_INDICES: [u16; 6] = [
    0, 1, 2,
    0, 2, 3,
];

// Matches mailbox presentation, which
// uses three images for vsync
const FRAMES_IN_FLIGHT: usize = 3;
const FORMAT: Format = Format::Rgba8Srgb;

pub struct GfxState {
    current_frame: usize,
    content_size: Rect,

    device: back::Device,
    queue_group: QueueGroup<back::Backend>,

    in_flight_fences: Vec<<back::Backend as Backend>::Fence>,
    render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    command_buffers: Vec<<back::Backend as Backend>::CommandBuffer>,
    framebuffers: Vec<<back::Backend as Backend>::Framebuffer>,
    image_views: Vec<<back::Backend as Backend>::ImageView>,

    command_pool: ManuallyDrop<<back::Backend as Backend>::CommandPool>,
    render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
    swapchain: ManuallyDrop<<back::Backend as Backend>::Swapchain>,

    pipeline: PipelineInfo,
    vertices: BufferInfo,
    indices: BufferInfo,
}

impl GfxState {
    pub fn new(window: &Window) -> Result<Self, &'static str> {
        // Backend handle
        let instance =
            back::Instance::create(utils::WINDOW_NAME, 1).map_err(|_| "Unsupported backend")?;

        // Window drawing surface
        // TODO: Does this need to be paired with a destroy_surface call?
        let mut surface = unsafe { instance.create_surface(window) }
            .map_err(|_| "Could not get drawing surface")?;

        // Supports our backend, probably a GPU
        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .find(|a| {
                a.queue_families.iter().any(|qf| {
                    qf.queue_type().supports_graphics() && surface.supports_queue_family(qf)
                })
            })
            .ok_or("No adapter supporting Vulkan")?;

        // A set of queues with identical properties
        let queue_family = adapter
            .queue_families
            .iter()
            .find(|qf| qf.queue_type().supports_graphics() && surface.supports_queue_family(qf))
            .ok_or("No queue family with graphics.")?;

        // The adapter's underlying device
        let gpu = unsafe {
            adapter
                .physical_device
                // Request graphics queue with full priority and core features only
                .open(&[(queue_family, &[1.0f32])], Features::empty())
        }
        .map_err(|_| "Could not open physical device")?;

        // Take ownership of contents so the gpu can go
        // out of scope while the queue group lives on
        let Gpu {
            device,
            queue_groups,
        } = gpu;

        // Queue group contains queues matching the queue family
        let queue_group = queue_groups
            .into_iter()
            .find(|qg| qg.family == queue_family.id())
            .ok_or("Matching queue group not found")?;

        if !queue_group.queues.is_empty() {
            Ok(())
        } else {
            Err("Queue group contains no command queues")
        }?;

        let content_size = window.inner_size();
        let content_size = Extent2D {
            width: content_size.width,
            height: content_size.height,
        };
        let capabilities = surface.capabilities(&adapter.physical_device);
        let swapchain_config = SwapchainConfig::from_caps(&capabilities, FORMAT, content_size)
            .with_present_mode(PresentMode::MAILBOX);

        // Swapchain manages a collection of images
        // Backbuffer contains handles to swapchain image memory
        let (swapchain, backbuffer) =
            unsafe { device.create_swapchain(&mut surface, swapchain_config, None) }
                .map_err(|_| "Could not create swapchain")?;

        // Collection of subpasses,
        // defines which attachment will be written
        let render_pass = unsafe {
            device.create_render_pass(
                &[
                    // Describes a render target,
                    // to be attached as input or output
                    Attachment {
                        format: Some(FORMAT),
                        // Don't have MSAA yet anyway
                        samples: 1,
                        // Clear the render target to the clear color and preserve the result
                        ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
                        stencil_ops: AttachmentOps::DONT_CARE,
                        // Begin uninitialized, end ready to present
                        layouts: AttachmentLayout::Undefined..AttachmentLayout::Present,
                    },
                ],
                &[
                    // Render pass stage, distinct from multipass rendering
                    SubpassDesc {
                        // Zero is color attachment ID
                        colors: &[(0, AttachmentLayout::ColorAttachmentOptimal)],
                        depth_stencil: None,
                        inputs: &[],
                        // For MSAA
                        resolves: &[],
                        // Attachments not used by subpass but which must preserved
                        preserves: &[],
                    },
                ],
                &[],
            )
        }
        .map_err(|_| "Could not create render pass")?;

        // Describe access to the underlying image memory,
        // possibly a subregion
        let image_views = backbuffer
            .into_iter()
            .map(|image| {
                unsafe {
                    device.create_image_view(
                        &image,
                        ViewKind::D2,
                        FORMAT,
                        Swizzle::NO,
                        SubresourceRange {
                            // Image format properties that further specify the format,
                            // especially if the format is ambiguous
                            aspects: Aspects::COLOR,
                            // Mipmaps
                            levels: 0..1,
                            // Image array layers
                            layers: 0..1,
                        },
                    )
                }
                .map_err(|_| "Could not create a backbuffer image view")
            })
            .collect::<Result<Vec<_>, &str>>()?;

        // A framebuffer defines which image view
        // is to be which attachment
        let framebuffers = image_views
            .iter()
            .map(|view| {
                let view_vec: ArrayVec<[_; 1]> = [view].into();
                unsafe {
                    device.create_framebuffer(
                        &render_pass,
                        view_vec,
                        Extent {
                            width: content_size.width,
                            height: content_size.height,
                            // Layers
                            depth: 1,
                        },
                    )
                }
                .map_err(|_| "Could not create framebuffer")
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Allocator for command buffers
        let mut command_pool = unsafe {
            device.create_command_pool(queue_group.family, CommandPoolCreateFlags::RESET_INDIVIDUAL)
        }
        .map_err(|_| "Could not create command pool")?;

        let content_size = content_size.to_extent().rect();

        let make_semaphore = || {
            device
                .create_semaphore()
                .map_err(|_| "Could not create semaphore")
        };

        Ok(Self {
            image_available_semaphores: full_flight(make_semaphore)?,
            render_finished_semaphores: full_flight(make_semaphore)?,
            in_flight_fences: full_flight(|| {
                device
                    .create_fence(true)
                    .map_err(|_| "Could not create fence")
            })?,

            command_buffers: framebuffers
                .iter()
                // Primary command buffers cannot be reused across sub passes
                .map(|_| unsafe { command_pool.allocate_one(Level::Primary) })
                .collect::<Vec<_>>(),

            pipeline: PipelineInfo::new(
                &device,
                pass::Subpass {
                    index: 0,
                    main_pass: &render_pass,
                },
                content_size,
            )?,

            vertices: BufferInfo::new(&device, &adapter, &QUAD_DATA, Usage::VERTEX)?,
            indices: BufferInfo::new(&device, &adapter, &QUAD_INDICES, Usage::INDEX)?,

            command_pool: ManuallyDrop::new(command_pool),
            render_pass: ManuallyDrop::new(render_pass),
            swapchain: ManuallyDrop::new(swapchain),

            current_frame: 0,
            content_size,
            queue_group,
            framebuffers,
            image_views,
            device,
        })
    }

    pub fn draw_frame(&mut self, color: [f32; 4], mouse: Vec2) -> Result<(), &'static str> {
        let image_available = &self.image_available_semaphores[self.current_frame];
        let render_finished = &self.render_finished_semaphores[self.current_frame];
        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;

        let (image_i, _suboptimal) = unsafe {
            self.swapchain
                .acquire_image(core::u64::MAX, Some(image_available), None)
        }
        .map_err(|_| "Failed to acquire an image from the swapchain")?;
        let image_i = image_i as usize;

        let flight_fence = &self.in_flight_fences[image_i];
        unsafe { self.device.wait_for_fence(flight_fence, core::u64::MAX) }
            .map_err(|_| "Failed to wait on the fence")?;
        unsafe { self.device.reset_fence(flight_fence) }
            .map_err(|_| "Failed to reset the fence")?;

        self.vertices.load_data(&self.device, &QUAD_DATA)?;
        self.indices.load_data(&self.device, &QUAD_INDICES)?;

        let commands = &mut self.command_buffers[image_i];
        let buffers: ArrayVec<[_; 1]> = [(&*self.vertices.buffer, 0)].into();
        unsafe {
            commands.begin_primary(CommandBufferFlags::EMPTY);
            commands.bind_graphics_pipeline(&self.pipeline.handle);
            commands.bind_vertex_buffers(0, buffers);
            commands.bind_index_buffer(IndexBufferView {
                buffer: &self.indices.buffer,
                offset: 0,
                index_type: IndexType::U16,
            });
            commands.push_graphics_constants(
                &self.pipeline.layout,
                ShaderStageFlags::VERTEX,
                0,
                &[
                    mem::transmute::<f32, u32>(mouse.x),
                    mem::transmute::<f32, u32>(mouse.y),
                ],
            );
            commands.begin_render_pass(
                &self.render_pass,
                &self.framebuffers[image_i],
                self.content_size,
                [ClearValue {
                    color: ClearColor { float32: color },
                }]
                .iter(),
                SubpassContents::Inline,
            );
            commands.draw_indexed(0..6, 0, 0..1);
            commands.end_render_pass();
            commands.finish();
        }

        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let submission = Submission {
            command_buffers: &self.command_buffers.get(image_i),
            wait_semaphores,
            signal_semaphores,
        };
        let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let command_queue = &mut self.queue_group.queues[0];
        unsafe {
            command_queue.submit(submission, Some(flight_fence));
            self.swapchain
                .present(command_queue, image_i as u32, present_wait_semaphores)
        }
        // Discard suboptimal warning
        .map(|_| ())
        .map_err(|_| "Failed to present into the swapchain")
    }

    pub fn free(&mut self) {
        let _ = self.device.wait_idle();

        // Don't need to destroy command buffers,
        // they are freed with their pool

        for fence in self.in_flight_fences.drain(..) {
            unsafe { self.device.destroy_fence(fence) }
        }

        for semaphore in self.render_finished_semaphores.drain(..) {
            unsafe { self.device.destroy_semaphore(semaphore) }
        }

        for semaphore in self.image_available_semaphores.drain(..) {
            unsafe { self.device.destroy_semaphore(semaphore) }
        }

        for framebuffer in self.framebuffers.drain(..) {
            unsafe { self.device.destroy_framebuffer(framebuffer) }
        }

        for view in self.image_views.drain(..) {
            unsafe { self.device.destroy_image_view(view) }
        }

        self.vertices.free(&self.device);
        self.indices.free(&self.device);
        self.pipeline.free(&self.device);

        unsafe {
            self.device
                .destroy_command_pool(ManuallyDrop::into_inner(ptr::read(&self.command_pool)));
            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(ptr::read(&self.render_pass)));
            self.device
                .destroy_swapchain(ManuallyDrop::into_inner(ptr::read(&self.swapchain)));
        }
    }
}

impl Drop for GfxState {
    fn drop(&mut self) {
        self.free()
    }
}

fn full_flight<T, F>(cb: F) -> Result<Vec<T>, &'static str>
where
    F: Fn() -> Result<T, &'static str>,
{
    (0..FRAMES_IN_FLIGHT)
        .map(|_| cb())
        .collect::<Result<Vec<_>, _>>()
}
