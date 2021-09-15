use glam::*;
use wgpu::util::DeviceExt;
use winit::{
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

    fn as_buffer_contents<T>(slice: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr() as *const u8,
                std::mem::size_of::<T>() * slice.len(),
            )
        }
    }

    pub fn create_index_buffer(&self, contents: &[Index]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: Self::as_buffer_contents(contents),
                usage: wgpu::BufferUsages::INDEX,
            })
    }

    pub fn create_vertex_buffer<T>(&self, contents: &[T]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: Self::as_buffer_contents(contents),
                usage: wgpu::BufferUsages::VERTEX,
            })
    }
}

mod main_loop {
    use super::*;

    fn build_window() -> (EventLoop<()>, Window) {
        let event_loop = EventLoop::new();

        let mut window_builder = winit::window::WindowBuilder::new();
        window_builder.window = WindowAttributes {
            title: "Testing".to_owned(),
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

        (event_loop, window)
    }

    pub fn run(mut event_handler: impl 'static + FnMut(Event<()>, &mut ControlFlow)) -> ! {
        let (event_loop, window) = build_window();
        let event_loop_proxy = event_loop.create_proxy();

        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == window.id() => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    _ => {}
                },

                Event::MainEventsCleared => {
                    window.request_redraw();
                }

                _ => {}
            }
            event_handler(event, control_flow);
        })
    }
}
