use wgpu::include_wgsl;
use wgpu_examples::*;

use crate::cube::UniformBuffer;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VertexInput {
    position: Vec4,
    texture_coordinates: Vec2,
}

mod cube {
    use super::*;
    use std::num::NonZeroU64;

    #[repr(C)]
    pub struct UniformBuffer {
        pub matrix: Mat4,
    }

    impl UniformBuffer {
        pub const SIZE: NonZeroU64 =
            unsafe { NonZeroU64::new_unchecked(size_of!(UniformBuffer) as u64) };

        pub const USIZE: usize = Self::SIZE.get() as usize;

        pub fn as_buffer_contents(&self) -> &[u8] {
            unsafe {
                let bytes = std::mem::transmute::<&Self, &[u8; Self::USIZE]>(self).as_ptr();
                std::slice::from_raw_parts(bytes, Self::USIZE)
            }
        }
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
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        comparison: false,
                        filtering: true,
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

    pub fn create_sampler(device: &wgpu::Device) -> wgpu::Sampler {
        let filter_mode = wgpu::FilterMode::Nearest;
        device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: filter_mode,
            min_filter: filter_mode,
            ..Default::default()
        })
    }

    pub fn create_bind_group(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        uniform: &[u8],
        texture_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
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
    env_logger::init();

    let main_loop = MainLoop::new("cube");
    let mut gpu = Gpu::new(main_loop.window());

    let bind_group_layout = cube::create_bind_group_layout(gpu.device());
    let pipeline_layout = cube::create_pipeline_layout(gpu.device(), &bind_group_layout);
    let vertex_buffer_layouts = cube::create_vertex_buffer_layouts();
    let shader_module = gpu
        .device()
        .create_shader_module(&include_wgsl!("./cube.wgsl"));

    let cube_texture = gpu.create_texture_from_image(include_image!("./cube.png"));
    let sampler = cube::create_sampler(&gpu.device());
    let mut vertices = cube::create_vertices();
    let indices = cube::create_indices();

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

    let mut aspect_ratio = 1.;

    main_loop.run(move |delta_time, window, event, _| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::Resized(new_inner_size) => {
                gpu.resize_surface(new_inner_size.width, new_inner_size.height);
                aspect_ratio = new_inner_size.width as f32 / new_inner_size.height as f32;
            }

            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                gpu.resize_surface(new_inner_size.width, new_inner_size.height);
            }

            _ => {}
        },

        Event::RedrawRequested(_) => {
            if let Ok(mut render_pass_resources) = gpu.create_render_pass_resources() {
                let rotation_matrix = Mat4::from_quat(Quat::from_euler(
                    EulerRot::ZXY,
                    0.,
                    0.,
                    90f32.to_radians() * delta_time.as_secs_f32(),
                ));
                dbg!(&delta_time);
                for vertex in vertices.iter_mut() {
                    vertex.position = rotation_matrix * vertex.position;
                }

                let uniform = UniformBuffer {
                    matrix: cube::create_matrix(aspect_ratio),
                };

                let bind_group = cube::create_bind_group(
                    &gpu,
                    &bind_group_layout,
                    uniform.as_buffer_contents(),
                    &cube_texture.create_view(&Default::default()),
                    &sampler,
                );

                let index_buffer = gpu.create_index_buffer(&indices);
                let vertex_buffer = gpu.create_vertex_buffer(&vertices);

                {
                    let (view, resolve_target) =
                        render_pass_resources.create_view_and_resolve_target();
                    let mut render_pass = render_pass_resources.command_encoder.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Render Pass"),
                            color_attachments: &[wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: resolve_target.as_ref(),
                                ops: CLEAR_WHITE_OPERATIONS,
                            }],
                            depth_stencil_attachment: None,
                        },
                    );

                    render_pass.set_pipeline(&render_pipeline);
                    render_pass.set_bind_group(0, &bind_group, &[]);
                    render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    render_pass.draw_indexed(0..indices.len() as _, 0, 0..1);
                }
                gpu.queue()
                    .submit([render_pass_resources.command_encoder.finish()]);
            }
        }

        _ => {}
    })
}
