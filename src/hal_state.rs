use gfx_backend_vulkan as back;
use gfx_hal::{
    adapter::{Gpu, PhysicalDevice},
    device::Device,
    format::Format,
    queue::family::QueueFamily,
    window::{Extent2D, PresentMode, Surface, SwapchainConfig},
    Features, Instance,
};
use winit::window::Window;

pub struct HalState {
    instance: back::Instance,
}

const VERSION: u32 = 1;
const WINDOW_NAME: &str = "Learn Gfx";
// Matches mailbox presentation, which
// uses three images for vsync
const FRAMES_IN_FLIGHT: usize = 3;

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

        let (device, queue_group) = {
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

            (device, queue_group)
        };

        let content_size = window.inner_size();
        let content_size = Extent2D {
            width: content_size.width,
            height: content_size.height,
        };
        let capabilities = surface.capabilities(&adapter.physical_device);
        let swapchain_config =
            SwapchainConfig::from_caps(&capabilities, Format::Rgba32Sfloat, content_size)
                .with_present_mode(PresentMode::MAILBOX);

        // Swapchain manages a collection of images
        // Backbuffer contains handles to swapchain images
        let (swapchain, backbuffer) =
            unsafe { device.create_swapchain(&mut surface, swapchain_config, None) }
                .map_err(|_| "Could not create swapchain")?;

        Ok(Self { instance })
    }
}
