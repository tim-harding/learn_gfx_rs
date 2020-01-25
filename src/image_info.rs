use gfx_backend_vulkan as back;
use gfx_hal::{memory::Requirements, Backend};
use std::mem::ManuallyDrop;

pub struct ImageInfo {
    pub requirements: Requirements,
    pub image: ManuallyDrop<<back::Backend as Backend>::Image>,
    pub memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
    pub image_view: ManuallyDrop<<back::Backend as Backend>::ImageView>,
    pub sampler: ManuallyDrop<<back::Backend as Backend>::Sampler>,
}

impl ImageInfo {
    pub fn new(device: &back::Device) -> Result<(), &'static str> {

        Ok(())
    }

    pub fn free(&mut self, device: &back::Device) {

    }
}