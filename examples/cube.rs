use wgpu::include_wgsl;
use wgpu_examples::*;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VertexInput {
    position: Vec4,
    texture_coordinates: Vec2,
}

struct RenderState {
    bind_group_layout: wgpu::BindGroupLayout,
}

mod cube {
    use std::num::NonZeroU64;

    use wgpu::UncapturedErrorHandler;

    use super::*;

    #[repr(C)]
    pub struct UniformBuffer {
        matrix: Mat4,
    }

    impl UniformBuffer {
        pub const SIZE: NonZeroU64 =
            unsafe { NonZeroU64::new_unchecked(size_of!(UniformBuffer) as u64) };
    }

    macro_rules! vertex {
        ([$x:expr, $y:expr, $z:expr], [$tex_x:expr, $tex_y:expr]) => {
            VertexInput {
                position: Vec4::new($x as f32, $y as f32, $z as f32, 1.),
                texture_coordinates: Vec2::new($tex_x as f32, $tex_y as f32),
            }
        };
    }

    pub fn create_vertices() -> Vec<VertexInput> {
        [
            vertex!([-1, -1, 1], [0, 0]),
            vertex!([1, -1, 1], [1, 0]),
            vertex!([1, 1, 1], [1, 1]),
            vertex!([-1, 1, 1], [0, 1]),
            vertex!([-1, 1, -1], [1, 0]),
            vertex!([1, 1, -1], [0, 0]),
            vertex!([1, -1, -1], [0, 1]),
            vertex!([-1, -1, -1], [1, 1]),
            vertex!([1, -1, -1], [0, 0]),
            vertex!([1, 1, -1], [1, 0]),
            vertex!([1, 1, 1], [1, 1]),
            vertex!([1, -1, 1], [0, 1]),
            vertex!([-1, -1, 1], [1, 0]),
            vertex!([-1, 1, 1], [0, 0]),
            vertex!([-1, 1, -1], [0, 1]),
            vertex!([-1, -1, -1], [1, 1]),
            vertex!([1, 1, -1], [1, 0]),
            vertex!([-1, 1, -1], [0, 0]),
            vertex!([-1, 1, 1], [0, 1]),
            vertex!([1, 1, 1], [1, 1]),
            vertex!([1, -1, 1], [0, 0]),
            vertex!([-1, -1, 1], [1, 0]),
            vertex!([-1, -1, -1], [1, 1]),
            vertex!([1, -1, -1], [0, 1]),
        ]
        .to_vec()
    }

    pub fn create_indices() -> Vec<Index> {
        [
            0, 1, 2, 2, 3, 0, // top
            4, 5, 6, 6, 7, 4, // bottom
            8, 9, 10, 10, 11, 8, // right
            12, 13, 14, 14, 15, 12, // left
            16, 17, 18, 18, 19, 16, // front
            20, 21, 22, 22, 23, 20, // back
        ]
        .to_vec()
    }

    #[rustfmt::skip]
    pub fn create_matrix(aspect_ratio: f32) -> Mat4 {
        let view_matrix = Mat4::look_at_lh(
            Vec3::new(1.5, -5., 3.), 
            Vec3::new(0., 0., 0.), 
            Vec3::new(0., 0., 1.)
        );
        let projection_matrix = Mat4::perspective_lh(
            45_f32.to_radians(), 
            aspect_ratio, 
            1., 
            10.
        );
        projection_matrix * view_matrix
    }

    pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cube Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(UniformBuffer::SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn create_pipeline_layout(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::PipelineLayout {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        })
    }

    pub fn create_bind_group(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        uniform: &[u8],
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        offset: 0,
                        buffer: &gpu.create_uniform_buffer(uniform),
                        size: Some(UniformBuffer::SIZE),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
            ],
            label: None,
        })
    }

    pub fn create_vertex_buffer_layouts<'a>() -> [wgpu::VertexBufferLayout<'a>; 1] {
        [wgpu::VertexBufferLayout {
            array_stride: size_of!(VertexInput) as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: size_of!([f32; 4]) as wgpu::BufferAddress,
                    shader_location: 1,
                },
            ],
        }]
    }
}

fn main() {
    let main_loop = MainLoop::new("cube");
    let mut gpu = Gpu::new(main_loop.window());

    let bind_group_layout = cube::create_bind_group_layout(gpu.device());
    let pipeline_layout = cube::create_pipeline_layout(gpu.device(), &bind_group_layout);
    let vertex_buffer_layouts = cube::create_vertex_buffer_layouts();
    let shader_module = gpu
        .device()
        .create_shader_module(&include_wgsl!("./cube.wgsl"));

    let cube_texture = gpu.create_texture_from_image(include_image!("./cube.png"));

    let render_pipeline = gpu
        .device()
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                buffers: &vertex_buffer_layouts,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[gpu.preferred_texture_format().into()],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: gpu.multisample_state(),
        });

    main_loop.run(move |window, event, _| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::Resized(new_inner_size) => {
                gpu.resize_surface(new_inner_size.width, new_inner_size.height);
            }

            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                gpu.resize_surface(new_inner_size.width, new_inner_size.height);
            }

            _ => {}
        },

        Event::RedrawRequested(_) => {
            let mut command_encoder = gpu.device().create_command_encoder(&Default::default());
            {
                let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: 
                })
            }
        }

        _ => {}
    })
}
