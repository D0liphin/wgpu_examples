```rs
use wgpu_examples::*;

struct RenderState {
    bind_group_layout: wgpu::BindGroupLayout,
}

mod _ {
    use std::num::NonZeroU64;
    use super::*;

    #[repr(C)]
    pub struct UniformBuffer {}

    impl UniformBuffer {
        pub const SIZE: NonZeroU64 =
            unsafe { NonZeroU64::new_unchecked(size_of!(UniformBuffer) as u64) };
    }

    pub fn create_vertices() -> Vec<VertexInput> { 
        todo!() 
    }

    pub fn create_indices() -> Vec<Index> { 
        todo!() 
    }
}

fn main() {
    let main_loop = MainLoop::new("cube");
    let mut gpu = Gpu::new(main_loop.window());

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
        _ => {}
    })
}

```