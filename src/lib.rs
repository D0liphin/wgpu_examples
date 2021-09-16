pub use glam::*;
use image::DynamicImage;
pub use std::time;
pub use wgpu::util::DeviceExt;
use wgpu::ShaderModule;
pub use winit::{
    dpi::{PhysicalSize, Size},
    event::{Event, *},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};

pub type Index = u16;

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SampleCount {
    Single = 1,
    Msaa4x = 4,
}

impl From<u32> for SampleCount {
    fn from(sample_count: u32) -> Self {
        match sample_count {
            1 => Self::Single,
            4 => Self::Msaa4x,
            _ => panic!("a sample count of {} is invalid", sample_count),
        }
    }
}

pub struct FrameBuffer {
    texture: wgpu::Texture,
    sample_count: u32,
}

impl FrameBuffer {
    pub fn new(device: &wgpu::Device, config: &SurfaceHandlerConfiguration) -> Self {
        let sample_count = config.sample_count as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Mutlisampled Texture"),
            sample_count,
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        });

        Self {
            texture,
            sample_count,
        }
    }

    pub fn create_view(&self) -> wgpu::TextureView {
        self.texture.create_view(&Default::default())
    }
}

pub struct SurfaceHandler {
    surface: wgpu::Surface,
    surface_configuration: wgpu::SurfaceConfiguration,
    frame_buffer: Option<FrameBuffer>,
}

pub struct SurfaceHandlerConfiguration {
    pub width: u32,
    pub height: u32,
    pub sample_count: SampleCount,
}

impl SurfaceHandler {
    pub fn multisample_state(&self) -> wgpu::MultisampleState {
        wgpu::MultisampleState {
            count: self.sample_count() as u32,
            ..Default::default()
        }
    }

    pub fn sample_count(&self) -> SampleCount {
        if let Some(FrameBuffer { sample_count, .. }) = self.frame_buffer {
            sample_count.into()
        } else {
            SampleCount::Single
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.configure(
            &device,
            &SurfaceHandlerConfiguration {
                width,
                height,
                sample_count: self.sample_count(),
            },
        );
    }

    pub fn configure(&mut self, device: &wgpu::Device, config: &SurfaceHandlerConfiguration) {
        self.surface_configuration = wgpu::SurfaceConfiguration {
            width: config.width,
            height: config.height,
            ..self.surface_configuration
        };
        self.surface.configure(&device, &self.surface_configuration);

        match config.sample_count {
            SampleCount::Single => {
                self.frame_buffer = None;
            }
            SampleCount::Msaa4x => self.frame_buffer = Some(FrameBuffer::new(&device, &config)),
        }
    }

    pub fn create_view_and_resolve_target(
        &self,
        surface_texture: &wgpu::SurfaceTexture,
    ) -> (wgpu::TextureView, Option<wgpu::TextureView>) {
        let surface_texture_view = surface_texture.texture.create_view(&Default::default());
        if let Some(ref frame_buffer) = self.frame_buffer {
            (frame_buffer.create_view(), Some(surface_texture_view))
        } else {
            (surface_texture_view, None)
        }
    }
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_handler: SurfaceHandler,
}

impl Gpu {
    pub fn new(window: &Window) -> Self {
        pollster::block_on(Self::new_async(window))
    }

    pub async fn new_async(window: &Window) -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
            })
            .await
            .expect("request adapter error");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("request device error");

        let preferred_texture_format = surface
            .get_preferred_format(&adapter)
            .expect("get preferred format error");
        let window_size = window.inner_size();
        let surface_configuration = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: preferred_texture_format,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &surface_configuration);

        let frame_buffer = FrameBuffer::new(
            &device,
            &SurfaceHandlerConfiguration {
                width: window_size.width,
                height: window_size.height,
                sample_count: SampleCount::Msaa4x,
            },
        );

        let surface_handler = SurfaceHandler {
            surface,
            surface_configuration,
            frame_buffer: Some(frame_buffer),
        };

        Self {
            device,
            queue,
            surface_handler,
        }
    }

    pub fn resize_surface(&mut self, width: u32, height: u32) {
        self.surface_handler.resize(&self.device, width, height);
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    fn as_buffer_contents<T>(slice: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                std::mem::size_of::<T>() * slice.len(),
            )
        }
    }

    fn create_buffer<T>(
        &self,
        label: &str,
        contents: &[T],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: Self::as_buffer_contents(contents),
                usage,
            })
    }

    pub fn create_index_buffer(&self, contents: &[Index]) -> wgpu::Buffer {
        self.create_buffer(
            "Index Buffer",
            Self::as_buffer_contents(contents),
            wgpu::BufferUsages::INDEX,
        )
    }

    pub fn create_vertex_buffer<T>(&self, contents: &[T]) -> wgpu::Buffer {
        self.create_buffer(
            "Vertex Buffer",
            Self::as_buffer_contents(contents),
            wgpu::BufferUsages::VERTEX,
        )
    }

    pub fn create_uniform_buffer<T>(&self, contents: &[T]) -> wgpu::Buffer {
        self.create_buffer(
            "Uniform Buffer",
            Self::as_buffer_contents(contents),
            wgpu::BufferUsages::UNIFORM,
        )
    }

    pub fn create_texture_from_image(&self, image: DynamicImage) -> wgpu::Texture {
        use image::GenericImageView;

        let image_buffer = image.as_rgba8().expect("image format error");
        let dimensions = image.dimensions();

        let texture_extent_3d = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: texture_extent_3d,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        });

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            image_buffer,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(dimensions.0 << 2),
                rows_per_image: std::num::NonZeroU32::new(dimensions.1),
            },
            texture_extent_3d,
        );

        texture
    }

    pub fn preferred_texture_format(&self) -> wgpu::TextureFormat {
        self.surface_handler.surface_configuration.format
    }

    pub fn multisample_state(&self) -> wgpu::MultisampleState {
        self.surface_handler.multisample_state()
    }

    pub fn create_render_pass_resources(&self) -> Result<RenderPassResources, wgpu::SurfaceError> {
        Ok(RenderPassResources {
            command_encoder: self.device.create_command_encoder(&Default::default()),
            surface_texture: self.surface_handler.surface.get_current_frame()?.output,
            gpu: &self,
        })
    }
}

pub struct RenderPassResources<'a> {
    pub command_encoder: wgpu::CommandEncoder,
    surface_texture: wgpu::SurfaceTexture,
    gpu: &'a Gpu,
}

impl RenderPassResources<'_> {
    pub fn create_view_and_resolve_target(&self) -> (wgpu::TextureView, Option<wgpu::TextureView>) {
        self.gpu
            .surface_handler
            .create_view_and_resolve_target(&self.surface_texture)
    }
}

pub struct MainLoop {
    event_loop: EventLoop<()>,
    window: Window,
}

impl MainLoop {
    pub fn new(title: &str) -> MainLoop {
        let event_loop = EventLoop::new();
        let mut window_builder = winit::window::WindowBuilder::new();
        window_builder.window = WindowAttributes {
            title: title.to_owned(),
            min_inner_size: Some(Size::Physical(PhysicalSize {
                width: 16,
                height: 16,
            })),
            inner_size: Some(Size::Physical(PhysicalSize {
                width: 16 * 2u32.pow(6),
                height: 9 * 2u32.pow(6),
            })),
            ..Default::default()
        };
        let window = window_builder.build(&event_loop).unwrap();

        Self { event_loop, window }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    const DURATION_500MS: time::Duration = time::Duration::from_millis(500);

    pub fn run(
        self,
        mut event_handler: impl 'static + FnMut(time::Duration, &Window, Event<()>, &mut ControlFlow),
    ) -> ! {
        let mut last_update_instant = time::Instant::now();
        let mut last_fps_update_instant = time::Instant::now();
        let mut update_count = 0u32;

        let event_loop = self.event_loop;
        let window = self.window;

        event_loop.run(move |event, _, control_flow| {
            let now = time::Instant::now();
            let duration_since_last_update = now.duration_since(last_update_instant);
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == window.id() => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    _ => {}
                },

                Event::MainEventsCleared => {
                    last_update_instant = now;

                    let duration_since_last_fps_update =
                        now.duration_since(last_fps_update_instant);
                    if duration_since_last_fps_update > Self::DURATION_500MS {
                        // print!(
                        //     "\r{: >12} fps",
                        //     update_count as f32 / duration_since_last_fps_update.as_secs_f32(),
                        // );
                        // use std::io::Write;
                        // std::io::stdout().flush().unwrap_or(());
                        last_fps_update_instant = now;
                        update_count = 0;
                    }

                    window.request_redraw();
                    update_count += 1;
                }

                _ => {}
            }
            event_handler(dbg!(duration_since_last_update), &window, event, control_flow);
        })
    }
}

#[macro_export]
macro_rules! size_of {
    ($T:ty) => {
        std::mem::size_of::<$T>()
    };
}

pub struct RenderBundle {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[macro_export]
macro_rules! include_image {
    ($file:expr $(,)?) => {
        image::load_from_memory(include_bytes!($file)).expect("load image error")
    };
}

pub const ALPHA_BLEND_STATE: Option<wgpu::BlendState> = Some(wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
});

pub const CLEAR_WHITE_OPERATIONS: wgpu::Operations<wgpu::Color> = wgpu::Operations {
    load: wgpu::LoadOp::Clear(wgpu::Color {
        r: 0.1,
        g: 0.2,
        b: 0.3,
        a: 1.0,
    }),
    store: true,
};
