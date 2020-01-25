use gfx_backend_vulkan as back;
use gfx_hal::{
    adapter::{Adapter, PhysicalDevice},
    buffer::{Usage},
    device::Device,
    memory::{Properties, Requirements},
    Backend, MemoryTypeId,
};
use std::{
    mem::{ManuallyDrop},
    ptr,
};

pub struct BufferInfo {
    pub buffer: ManuallyDrop<<back::Backend as Backend>::Buffer>,
    pub memory: ManuallyDrop<<back::Backend as Backend>::Memory>,
    pub requirements: Requirements,
}

impl BufferInfo {
    pub fn new<T>(
        device: &back::Device,
        adapter: &Adapter<back::Backend>,
        data: &[T],
        usage: Usage,
    ) -> Result<Self, &'static str> {
        let mut buffer = unsafe { device.create_buffer(array_size(data) as u64, usage) }
            .map_err(|_| "Failed to create a buffer for the vertices")?;

        // Creation of the buffer does not imply allocation.
        // We can now query it's prerequistes and allocate memory to match.
        let requirements = unsafe { device.get_buffer_requirements(&buffer) };

        // Find id of CPU-visible memory for vertex buffer
        let memory_type_id = adapter
            .physical_device
            .memory_properties()
            .memory_types
            .iter()
            .enumerate()
            .find(|&(id, memory_type)| {
                requirements.type_mask & (1 << id) != 0
                    && memory_type.properties.contains(Properties::CPU_VISIBLE)
            })
            .map(|(id, _)| MemoryTypeId(id))
            .ok_or("Failed to find a memory type to support the vertex buffer")?;

        // Allocate vertex buffer
        let memory = unsafe { device.allocate_memory(memory_type_id, requirements.size) }
            .map_err(|_| "Failed to allocate vertex buffer memory")?;

        // Make the buffer use the allocation
        unsafe { device.bind_buffer_memory(&memory, 0, &mut buffer) }
            .map_err(|_| "Failed to bind the buffer memory")?;

        Ok(Self{
            buffer: ManuallyDrop::new(buffer), 
            memory: ManuallyDrop::new(memory), 
            requirements
        })
    }

    pub fn load_data<T>(&self, device: &back::Device, data: &[T]) -> Result<(), &'static str> {
        let mapped_memory = unsafe {
            device
                .map_memory(&self.memory, 0..self.requirements.size)
        }
        .map_err(|_| "Failed to memory map buffer")?;

        unsafe {
            std::ptr::copy(
                data.as_ptr() as *const u8,
                mapped_memory,
                array_size(data),
            );
            device.unmap_memory(&self.memory)
        }

        Ok(())
    }

    pub fn free(&mut self, device: &back::Device) {
        unsafe {
            device.destroy_buffer(ManuallyDrop::into_inner(ptr::read(&self.buffer)));
            device.free_memory(ManuallyDrop::into_inner(ptr::read(&self.memory)));
        }
    }
}

fn array_size<T>(array: &[T]) -> usize {
    array.len() * std::mem::size_of::<T>()
}