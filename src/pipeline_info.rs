use gfx_backend_vulkan as back;
use gfx_hal::{
    device::Device,
    format::Format,
    pass::Subpass,
    // TODO: Maybe just use namespace inline
    pso::{
        AttributeDesc, BakedStates, BasePipeline, BlendDesc, BlendState, ColorBlendDesc, ColorMask,
        DepthStencilDesc, DescriptorSetLayoutBinding, Element, EntryPoint, GraphicsPipelineDesc,
        GraphicsShaderSet, InputAssemblerDesc, LogicOp, PipelineCreationFlags, Primitive,
        Rasterizer, Rect, ShaderStageFlags, Specialization, VertexBufferDesc, VertexInputRate,
        Viewport,
    },
    Backend,
};
use shaderc::{Compiler, ShaderKind};
use std::{
    mem::{self, ManuallyDrop},
    ops::Range,
    ptr,
};

pub struct PipelineInfo {
    pub descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    pub layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    pub handle: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,
}

// To be replaced as more upstream stuff is bundled up
pub struct CreationInfo<'a> {
    pub device: &'a back::Device,
    pub subpass: Subpass<'a, back::Backend>,
    pub content_size: Rect,
}

impl PipelineInfo {
    pub fn new(info: CreationInfo) -> Result<Self, &'static str> {
        let mut compiler = Compiler::new().ok_or("Failed to create shader compiler")?;

        let vert = compile_shader(
            "shaders/vert.glsl",
            &mut compiler,
            &info.device,
            ShaderKind::Vertex,
        )?;
        let frag = compile_shader(
            "shaders/frag.glsl",
            &mut compiler,
            &info.device,
            ShaderKind::Fragment,
        )?;

        let shaders = GraphicsShaderSet::<back::Backend> {
            vertex: EntryPoint {
                entry: "main",
                module: &vert,
                // Not sure what this is used for
                specialization: Specialization::EMPTY,
            },
            domain: None,
            geometry: None,
            hull: None,
            fragment: Some(EntryPoint {
                entry: "main",
                module: &frag,
                specialization: Specialization::EMPTY,
            }),
        };

        let input_assembler = InputAssemblerDesc {
            primitive: Primitive::TriangleList,
            with_adjacency: false,
            restart_index: None,
        };

        let vertex_buffers = vec![VertexBufferDesc {
            // Not the location listed on the shader,
            // this is just a unique id for the buffer
            binding: 0,
            stride: (mem::size_of::<f32>() * 2) as u32,
            rate: VertexInputRate::Vertex,
        }];

        let attributes = vec![AttributeDesc {
            // This is the attribute location in the shader
            location: 0,
            // Matches vertex buffer description
            binding: 0,
            element: Element {
                // Float vec2
                format: Format::Rg32Sfloat,
                offset: 0,
            },
        }];

        // No depth test for now
        let depth_stencil = DepthStencilDesc {
            depth: None,
            depth_bounds: false,
            stencil: None,
        };

        let blender = BlendDesc {
            logic_op: Some(LogicOp::Copy),
            targets: vec![ColorBlendDesc {
                mask: ColorMask::ALL,
                blend: Some(BlendState::ALPHA),
            }],
        };

        // Baked-in pipeline attributes
        let baked_states = BakedStates {
            viewport: Some(Viewport {
                rect: info.content_size,
                depth: 0.0..1.0,
            }),
            scissor: Some(info.content_size),
            blend_color: None,
            depth_bounds: None,
        };

        // This machinery is only used when graphics pipeline data
        // comes from somewhere other than the vertex buffer.
        // We still have to explicitly declare all these empty
        // bits and bobs.
        let bindings = Vec::<DescriptorSetLayoutBinding>::new();
        let immutable_samplers = Vec::<<back::Backend as Backend>::Sampler>::new();
        let descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout> =
            vec![unsafe {
                info.device
                    .create_descriptor_set_layout(bindings, immutable_samplers)
            }
            .map_err(|_| "Failed to create a descriptor set layout")?];
        let push_constants = Vec::<(ShaderStageFlags, Range<u32>)>::new();
        let layout = unsafe {
            info.device
                .create_pipeline_layout(&descriptor_set_layouts, push_constants)
        }
        .map_err(|_| "Failed to create a pipeline layout")?;

        let pipeline_desc = GraphicsPipelineDesc {
            shaders,
            rasterizer: Rasterizer::FILL,
            vertex_buffers,
            attributes,
            input_assembler,
            blender,
            depth_stencil,
            multisampling: None,
            baked_states,
            layout: &layout,
            subpass: info.subpass,
            flags: PipelineCreationFlags::empty(),
            parent: BasePipeline::None,
        };

        let handle = unsafe { info.device.create_graphics_pipeline(&pipeline_desc, None) }
            .map_err(|_| "Failed to create graphics pipeline")?;

        unsafe {
            // Not needed after pipeline is built
            info.device.destroy_shader_module(vert);
            info.device.destroy_shader_module(frag);
        }

        Ok(Self {
            descriptor_set_layouts,
            layout: ManuallyDrop::new(layout),
            handle: ManuallyDrop::new(handle),
        })
    }

    pub fn free(&mut self, device: &back::Device) {
        for layout in self.descriptor_set_layouts.drain(..) {
            unsafe { device.destroy_descriptor_set_layout(layout) }
        }

        unsafe {
            device.destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(&self.layout)));
            device.destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(&self.handle)));
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
