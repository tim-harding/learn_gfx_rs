use arrayvec::ArrayVec;
use gfx_backend_vulkan as back;
use gfx_hal::{
    adapter::Adapter,
    adapter::{Gpu, PhysicalDevice},
    command::{
        ClearColor, ClearValue, CommandBuffer, CommandBufferFlags,
        Level, SubpassContents,
    },
    device::Device,
    format::{Aspects, Format, Swizzle},
    image::{Extent, SubresourceRange, ViewKind},
    pass::{
        Attachment, AttachmentLayout, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp,
        SubpassDesc,
    },
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{PipelineStage, Rect},
    queue::{
        family::{QueueFamily, QueueGroup},
        CommandQueue, Submission,
    },
    window::{
        Extent2D,
        PresentMode,
        // Required import but compiler still complains.
        // Can't find a way of silencing it.
        Surface,
        Swapchain,
        SwapchainConfig,
    },
    Backend, Features, Instance,
};
use std::{mem::ManuallyDrop, ops::Drop, ptr};
use winit::window::Window;

pub struct HalState {
    current_frame: usize,
    in_flight_fences: Vec<<back::Backend as Backend>::Fence>,
    render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    command_buffers: Vec<<back::Backend as Backend>::CommandBuffer>,
    command_pool: ManuallyDrop<<back::Backend as Backend>::CommandPool>,
    framebuffers: Vec<<back::Backend as Backend>::Framebuffer>,
    image_views: Vec<<back::Backend as Backend>::ImageView>,
    render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
    render_area: Rect,
    queue_group: ManuallyDrop<QueueGroup<back::Backend>>,
    swapchain: ManuallyDrop<<back::Backend as Backend>::Swapchain>,
    device: ManuallyDrop<back::Device>,

    adapter: Adapter<back::Backend>,
    surface: <back::Backend as Backend>::Surface,
    instance: ManuallyDrop<back::Instance>,
}

// Should move this where Winit can use it too
const VERSION: u32 = 1;
const WINDOW_NAME: &str = "Learn Gfx";

// Matches mailbox presentation, which
// uses three images for vsync
const FRAMES_IN_FLIGHT: usize = 3;

const FORMAT: Format = Format::Rgba8Srgb;

impl HalState {
    pub fn new(window: &Window) -> Result<Self, &'static str> {
        // Backend handle
        let instance =
            back::Instance::create(WINDOW_NAME, VERSION).map_err(|_| "Unsupported backend")?;

        // Window drawing surface
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
                .map_err(|_| "Could not open physical device")
        }?;

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

        if queue_group.queues.len() > 0 {
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

        // Semaphores provide GPU-side syncronization
        let make_semaphore = || {
            device
                .create_semaphore()
                .map_err(|_| "Could not create semaphore")
        };

        // Image available flagged when...
        let image_available_semaphores = flight(make_semaphore)?;

        // Render finished flagged when...
        let render_finished_semaphores = flight(make_semaphore)?;

        // In flight fences flagged when...
        let in_flight_fences = flight(|| {
            device
                .create_fence(true)
                .map_err(|_| "Could not create fence")
        })?;

        // Describes a render target,
        // to be attached as input or output
        let attachment = Attachment {
            format: Some(FORMAT),
            // Don't have MSAA yet anyway
            samples: 1,
            // Clear the render target to the clear color and preserve the result
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
            stencil_ops: AttachmentOps::DONT_CARE,
            // Begin uninitialized, end ready to present
            layouts: AttachmentLayout::Undefined..AttachmentLayout::Present,
        };

        // Render pass stage, distinct from multipass rendering
        let subpass = SubpassDesc {
            // Zero is color attachment ID
            colors: &[(0, AttachmentLayout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            // For MSAA
            resolves: &[],
            // Attachments not used by subpass but which must preserved
            preserves: &[],
        };

        // Collection of subpasses,
        // defines which attachment will be written
        let render_pass = unsafe { device.create_render_pass(&[attachment], &[subpass], &[]) }
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
                let mut view_vec = ArrayVec::<[_; 1]>::new();
                view_vec.push(view);
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

        // Used to build up lists of commands for execution
        let command_buffers = framebuffers
            .iter()
            // Primary command buffers cannot be reused across sub passes
            .map(|_| unsafe { command_pool.allocate_one(Level::Primary) })
            .collect::<Vec<_>>();

        let render_area = Rect {
            x: 0,
            y: 0,
            w: content_size.width as i16,
            h: content_size.height as i16,
        };

        Ok(Self {
            current_frame: 0,
            in_flight_fences,
            render_finished_semaphores,
            image_available_semaphores,
            command_buffers,
            command_pool: ManuallyDrop::new(command_pool),
            framebuffers,
            image_views,
            render_pass: ManuallyDrop::new(render_pass),
            render_area,
            queue_group: ManuallyDrop::new(queue_group),
            swapchain: ManuallyDrop::new(swapchain),
            device: ManuallyDrop::new(device),

            adapter,
            surface,
            instance: ManuallyDrop::new(instance),
        })
    }

    pub fn draw_clear_frame(&mut self, color: [f32; 4]) -> Result<(), &'static str> {
        let image_available = &self.image_available_semaphores[self.current_frame];
        let render_finished = &self.render_finished_semaphores[self.current_frame];
        self.current_frame = (self.current_frame + 1) % FRAMES_IN_FLIGHT;

        let (image_i, suboptimal) = unsafe {
            self.swapchain
                .acquire_image(core::u64::MAX, Some(image_available), None)
        }
        .map_err(|_| "Failed to acquire an image from the swapchain")?;

        let image_i = image_i as usize;
        if let Some(_) = suboptimal {
            println!("Swapchain no longer matches drawing surface");
        }

        let flight_fence = &self.in_flight_fences[image_i];
        unsafe { self.device.wait_for_fence(flight_fence, core::u64::MAX) }
            .map_err(|_| "Failed to wait on the fence")?;
        unsafe { self.device.reset_fence(flight_fence) }
            .map_err(|_| "Failed to reset the fence")?;

        let buffer = &mut self.command_buffers[image_i];
        let clear_values = [ClearValue {
            color: ClearColor { float32: color },
        }];
        unsafe {
            buffer.begin_primary(CommandBufferFlags::EMPTY);
            buffer.begin_render_pass(
                &self.render_pass,
                &self.framebuffers[image_i],
                self.render_area,
                clear_values.iter(),
                SubpassContents::Inline,
            );
            buffer.finish();
        }

        let command_buffers = &self.command_buffers.iter().nth(image_i);
        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let submission = Submission {
            command_buffers,
            wait_semaphores,
            signal_semaphores,
        };
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
}

impl Drop for HalState {
    fn drop(&mut self) {
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

        unsafe {
            self.device
                .destroy_command_pool(ManuallyDrop::into_inner(ptr::read(&mut self.command_pool)));
            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(ptr::read(&mut self.render_pass)));
            self.device
                .destroy_swapchain(ManuallyDrop::into_inner(ptr::read(&mut self.swapchain)));

            ManuallyDrop::drop(&mut self.queue_group);
            ManuallyDrop::drop(&mut self.device);
            ManuallyDrop::drop(&mut self.instance);
        }
    }
}

fn flight<T, F>(cb: F) -> Result<Vec<T>, &'static str>
where
    F: Fn() -> Result<T, &'static str>,
{
    (0..FRAMES_IN_FLIGHT)
        .map(|_| cb())
        .collect::<Result<Vec<_>, _>>()
}
