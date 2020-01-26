use gfx_backend_vulkan as back;
use gfx_hal::{device::Device, format::Format, pass::Subpass, pso, Backend};
use shaderc::{Compiler, ShaderKind};
use std::{mem::ManuallyDrop, ops::Range};

pub struct PipelineInfo {
    pub descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub handle: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
}

// A pipeline describes all configurable and programmable state on the GPU
impl PipelineInfo {
    pub fn new(
        device: &back::Device,
        subpass: Subpass<back::Backend>,
        content_size: pso::Rect,
    ) -> Result<Self, &'static str> {
        use std::mem::size_of;

        let (vert, frag) = {
            let mut compiler = Compiler::new().ok_or("Failed to create shader compiler")?;
            let mut compile = |src, kind| compile_shader(src, &mut compiler, &device, kind);
            let vert = compile("shaders/vert.glsl", ShaderKind::Vertex)?;
            let frag = compile("shaders/frag.glsl", ShaderKind::Fragment)?;
            (vert, frag)
        };

        // This machinery is only used when graphics pipeline data
        // comes from somewhere other than the vertex buffer.
        // We still have to explicitly declare all these empty
        // bits and bobs.
        let descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> =
            vec![unsafe {
                device.create_descriptor_set_layout(
                    Vec::<pso::DescriptorSetLayoutBinding>::new(),
                    Vec::<<back::Backend as Backend>::Sampler>::new(),
                )
            }
            .map_err(|_| "Failed to create a descriptor set layout")?];

        let layout = unsafe {
            device.create_pipeline_layout(
                &descriptor_set_layouts,
                Vec::<(pso::ShaderStageFlags, Range<u32>)>::new(),
            )
        }
        .map_err(|_| "Failed to create a pipeline layout")?;

        let handle = unsafe {
            device.create_graphics_pipeline(
                &pso::GraphicsPipelineDesc {
                    shaders: pso::GraphicsShaderSet::<back::Backend> {
                        vertex: pso::EntryPoint {
                            entry: "main",
                            module: &vert,
                            // Not sure what this is used for
                            specialization: pso::Specialization::EMPTY,
                        },
                        domain: None,
                        geometry: None,
                        hull: None,
                        fragment: Some(pso::EntryPoint {
                            entry: "main",
                            module: &frag,
                            specialization: pso::Specialization::EMPTY,
                        }),
                    },

                    vertex_buffers: vec![pso::VertexBufferDesc {
                        // Not the location listed on the shader,
                        // this is just a unique id for the buffer
                        binding: 0,
                        stride: (size_of::<f32>() * 2) as u32,
                        rate: pso::VertexInputRate::Vertex,
                    }],

                    attributes: vec![pso::AttributeDesc {
                        // This is the attribute location in the shader
                        location: 0,
                        // Matches vertex buffer description
                        binding: 0,
                        element: pso::Element {
                            // Float vec2
                            format: Format::Rg32Sfloat,
                            offset: 0,
                        },
                    }],

                    input_assembler: pso::InputAssemblerDesc {
                        primitive: pso::Primitive::TriangleList,
                        with_adjacency: false,
                        restart_index: None,
                    },

                    blender: pso::BlendDesc {
                        logic_op: Some(pso::LogicOp::Copy),
                        targets: vec![pso::ColorBlendDesc {
                            mask: pso::ColorMask::ALL,
                            blend: Some(pso::BlendState::ALPHA),
                        }],
                    },

                    depth_stencil: pso::DepthStencilDesc {
                        depth: None,
                        depth_bounds: false,
                        stencil: None,
                    },

                    multisampling: None,
                    baked_states: pso::BakedStates {
                        viewport: Some(pso::Viewport {
                            rect: content_size,
                            depth: 0.0..1.0,
                        }),
                        scissor: Some(content_size),
                        blend_color: None,
                        depth_bounds: None,
                    },

                    rasterizer: pso::Rasterizer::FILL,
                    layout: &layout,
                    subpass: subpass,
                    flags: pso::PipelineCreationFlags::empty(),
                    parent: pso::BasePipeline::None,
                },
                None,
            )
        }
        .map_err(|_| "Failed to create graphics pipeline")?;

        unsafe {
            // Not needed after pipeline is built
            device.destroy_shader_module(vert);
            device.destroy_shader_module(frag);
        }

        Ok(Self {
            descriptor_set_layouts,
            layout: ManuallyDrop::new(layout),
            handle: ManuallyDrop::new(handle),
        })
    }

    pub fn free(&mut self, device: &back::Device) {
        use std::ptr::read;

        for layout in self.descriptor_set_layouts.drain(..) {
            unsafe { device.destroy_descriptor_set_layout(layout) }
        }

        unsafe {
            device.destroy_pipeline_layout(ManuallyDrop::into_inner(read(&self.layout)));
            device.destroy_graphics_pipeline(ManuallyDrop::into_inner(read(&self.handle)));
        }
    }
}

fn compile_shader(
    src_file: &str,
    compiler: &mut Compiler,
    device: &back::Device,
    kind: ShaderKind,
) -> Result<<back::Backend as Backend>::ShaderModule, &'static str> {
    let src = std::fs::read_to_string(src_file).map_err(|_| "Could not read shader source file")?;
    let spirv = compiler
        .compile_into_spirv(&src, kind, src_file, "main", None)
        .map_err(|e| {
            log::error!("{}", e);
            "Failed to compile fragment program"
        })?;
    unsafe { device.create_shader_module(spirv.as_binary()) }
        .map_err(|_| "Failed to create shader module")
}
