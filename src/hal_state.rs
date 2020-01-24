use gfx_backend_vulkan as back;
use gfx_hal::{
    adapter::{Gpu, PhysicalDevice},
    command::Level,
    device::Device,
    format::Format,
    format::{Aspects, Swizzle},
    image::{Extent, SubresourceRange, ViewKind},
    pass::{
        Attachment, AttachmentLayout, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp,
        SubpassDesc,
    },
    pool::{CommandPool, CommandPoolCreateFlags},
    queue::family::QueueFamily,
    window::{
        Extent2D,
        PresentMode,
        // Required import but compiler still complains.
        // Can't find a way of silencing it.
        Surface,
        SwapchainConfig,
    },
    Features, Instance,
};
use winit::window::Window;

pub struct HalState {
    instance: back::Instance,
}

// Should move this where Winit can use it too
const VERSION: u32 = 1;
const WINDOW_NAME: &str = "Learn Gfx";

// Matches mailbox presentation, which
// uses three images for vsync
const FRAMES_IN_FLIGHT: usize = 3;

const FORMAT: Format = Format::Rgba32Sfloat;

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
            .map(|image| unsafe {
                device
                    .create_image_view(
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
                    .map_err(|_| "Could not create a backbuffer image view")
            })
            .collect::<Result<Vec<_>, &str>>()?;

        // A framebuffer defines which image view
        // is to be which attachment
        let framebuffers = image_views
            .iter()
            .map(|view| unsafe {
                device
                    .create_framebuffer(
                        &render_pass,
                        vec![view],
                        Extent {
                            width: content_size.width,
                            height: content_size.height,
                            // Layers
                            depth: 1,
                        },
                    )
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

        Ok(Self { instance })
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
