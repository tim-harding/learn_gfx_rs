use crate::{utils, BufferInfo, PipelineInfo};
use arrayvec::ArrayVec;
use gfx_backend_vulkan as back;
use gfx_hal::{
    adapter::{Gpu, PhysicalDevice},
    buffer::Usage,
    command::Level,
    device::Device,
    format::{self, Format},
    image,
    pass::{self, AttachmentLayout, AttachmentOps},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::Rect,
    queue::family::{QueueFamily, QueueGroup},
    window::{self, Surface},
    Backend, Features, Instance,
};
use std::mem::ManuallyDrop;

const FORMAT: Format = Format::Rgba8Srgb;

pub struct GfxState {
    pub current_frame: usize,
    pub content_size: Rect,

    pub device: back::Device,
    pub queue_group: QueueGroup<back::Backend>,

    pub in_flight_fences: Vec<<back::Backend as Backend>::Fence>,
    pub render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    pub image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    pub command_buffers: Vec<<back::Backend as Backend>::CommandBuffer>,
    pub framebuffers: Vec<<back::Backend as Backend>::Framebuffer>,
    pub image_views: Vec<<back::Backend as Backend>::ImageView>,

    pub command_pool: ManuallyDrop<<back::Backend as Backend>::CommandPool>,
    pub render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,
    pub swapchain: ManuallyDrop<<back::Backend as Backend>::Swapchain>,

    pub pipeline: PipelineInfo,
    pub vertices: BufferInfo,
    pub indices: BufferInfo,
}

impl GfxState {
    pub fn new(window: &winit::window::Window) -> Result<Self, &'static str> {
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

            (device, queue_group)
        };

        if !queue_group.queues.is_empty() {
            Ok(())
        } else {
            Err("Queue group contains no command queues")
        }?;

        let content_size = window.inner_size();
        let content_size = window::Extent2D {
            width: content_size.width,
            height: content_size.height,
        };

        let swapchain_config = {
            let capabilities = surface.capabilities(&adapter.physical_device);
            window::SwapchainConfig::from_caps(&capabilities, FORMAT, content_size)
                .with_present_mode(window::PresentMode::MAILBOX)
        };

        let (swapchain, image_views) = {
            // Swapchain manages a collection of images
            // Backbuffer contains handles to swapchain image memory
            let (swapchain, backbuffer) =
                unsafe { device.create_swapchain(&mut surface, swapchain_config, None) }
                    .map_err(|_| "Could not create swapchain")?;

            // Describe access to the underlying image memory,
            // possibly a subregion
            let image_views = backbuffer
                .into_iter()
                .map(|image| {
                    unsafe {
                        device.create_image_view(
                            &image,
                            image::ViewKind::D2,
                            FORMAT,
                            format::Swizzle::NO,
                            image::SubresourceRange {
                                // Properties that further specify the image format,
                                // especially if it is ambiguous
                                aspects: format::Aspects::COLOR,
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

            (swapchain, image_views)
        };

        // A render pass is collection of subpasses describing
        // the type of images used during rendering operations,
        // how they will be used,
        // and the treatment of their contents
        let render_pass = unsafe {
            device.create_render_pass(
                &[
                    // Describes a render target,
                    // to be attached as input or output
                    pass::Attachment {
                        format: Some(FORMAT),
                        // Don't have MSAA yet anyway
                        samples: 1,
                        // Clear the render target to the clear color and preserve the result
                        ops: AttachmentOps::new(
                            pass::AttachmentLoadOp::Clear,
                            pass::AttachmentStoreOp::Store,
                        ),
                        stencil_ops: AttachmentOps::DONT_CARE,
                        // Begin uninitialized, end ready to present
                        layouts: AttachmentLayout::Undefined..AttachmentLayout::Present,
                    },
                ],
                &[
                    // Render pass stage, distinct from multipass rendering
                    pass::SubpassDesc {
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

        // Where a render pass describes the types of image attachments,
        // a framebuffer binds specific images to its attachements
        let framebuffers = image_views
            .iter()
            .map(|view| {
                let view_vec: ArrayVec<[_; 1]> = [view].into();
                unsafe {
                    device.create_framebuffer(
                        &render_pass,
                        view_vec,
                        image::Extent {
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

            vertices: BufferInfo::new(&device, &adapter, &utils::QUAD_DATA, Usage::VERTEX)?,
            indices: BufferInfo::new(&device, &adapter, &utils::QUAD_INDICES, Usage::INDEX)?,

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

    pub fn free(&mut self) {
        use std::ptr::read;

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
                .destroy_command_pool(ManuallyDrop::into_inner(read(&self.command_pool)));
            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(read(&self.render_pass)));
            self.device
                .destroy_swapchain(ManuallyDrop::into_inner(read(&self.swapchain)));
        }
    }
}

impl std::ops::Drop for GfxState {
    fn drop(&mut self) {
        self.free()
    }
}

fn full_flight<T, F>(cb: F) -> Result<Vec<T>, &'static str>
where
    F: Fn() -> Result<T, &'static str>,
{
    (0..utils::FRAMES_IN_FLIGHT)
        .map(|_| cb())
        .collect::<Result<Vec<_>, _>>()
}
