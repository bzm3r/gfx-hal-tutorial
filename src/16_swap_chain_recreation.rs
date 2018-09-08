extern crate env_logger;
#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;
extern crate gfx_hal as hal;
extern crate glsl_to_spirv;
extern crate winit;

use hal::{
    Capability, Device, Instance, PhysicalDevice, QueueFamily, Surface, Swapchain, SwapchainConfig,
};
use std::io::Read;

static WINDOW_NAME: &str = "16_swap_chain_recreation";

const MAX_FRAMES_IN_FLIGHT: usize = 2;

fn main() {
    env_logger::init();
    let (window, events_loop) = init_window();

    let (
        _instance,
        mut command_queues,
        mut swapchain_state,
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
    ) = init_hal(&window);

    main_loop(
        events_loop,
        &mut command_queues,
        &mut swapchain_state,
        &image_available_semaphores,
        &render_finished_semaphores,
        &in_flight_fences,
    );

    clean_up(
        swapchain_state,
        vec![image_available_semaphores, render_finished_semaphores],
        in_flight_fences,
    );
}

struct SwapchainState {
    adapter: hal::Adapter<back::Backend>,
    surface: <back::Backend as hal::Backend>::Surface,
    device: <back::Backend as hal::Backend>::Device,
    queue_type: hal::queue::QueueType,
    qf_id: hal::queue::family::QueueFamilyId,
    swapchain: <back::Backend as hal::Backend>::Swapchain,
    frame_images: Vec<(
        <back::Backend as hal::Backend>::Image,
        <back::Backend as hal::Backend>::ImageView,
    )>,
    render_pass: <back::Backend as hal::Backend>::RenderPass,
    swapchain_framebuffers: Vec<<back::Backend as hal::Backend>::Framebuffer>,
    descriptor_set_layouts: Vec<<back::Backend as hal::Backend>::DescriptorSetLayout>,
    pipeline_layout: <back::Backend as hal::Backend>::PipelineLayout,
    gfx_pipeline: <back::Backend as hal::Backend>::GraphicsPipeline,
    command_pool: hal::pool::CommandPool<back::Backend, hal::Graphics>,
    submission_command_buffers: Vec<
        hal::command::Submit<
            back::Backend,
            hal::Graphics,
            hal::command::MultiShot,
            hal::command::Primary,
        >,
    >,
}

impl SwapchainState {
    fn create(
        adapter: hal::Adapter<back::Backend>,
        mut surface: <back::Backend as hal::Backend>::Surface,
        device: <back::Backend as hal::Backend>::Device,
        queue_type: hal::queue::QueueType,
        qf_id: hal::queue::family::QueueFamilyId,
    ) -> SwapchainState {
        let (swapchain, extent, backbuffer, format) =
            create_swap_chain(&adapter, &device, &mut surface, None);
        let frame_images = create_image_views(backbuffer, format, &device);
        let render_pass = create_render_pass(&device, Some(format));
        let swapchain_framebuffers =
            create_framebuffers(&device, &render_pass, &frame_images, extent);
        let (descriptor_set_layouts, pipeline_layout, gfx_pipeline) =
            create_graphics_pipeline(&device, extent, &render_pass);
        let mut command_pool = create_command_pool(&device, queue_type, qf_id);
        let submission_command_buffers = create_command_buffers(
            &mut command_pool,
            &render_pass,
            &swapchain_framebuffers,
            extent,
            &gfx_pipeline,
        );

        SwapchainState {
            adapter,
            surface,
            device,
            queue_type,
            qf_id,
            swapchain,
            frame_images,
            render_pass,
            swapchain_framebuffers,
            descriptor_set_layouts,
            pipeline_layout,
            gfx_pipeline,
            command_pool,
            submission_command_buffers,
        }
    }

    fn clean_up(
        mut swapchain_state: SwapchainState,
        destroy_command_pool: bool,
    ) -> (
        hal::Adapter<back::Backend>,
        <back::Backend as hal::Backend>::Surface,
        <back::Backend as hal::Backend>::Device,
        hal::queue::QueueType,
        hal::queue::family::QueueFamilyId,
    ) {
        let (adapter, surface, device, queue_type, qf_id) = (
            swapchain_state.adapter,
            swapchain_state.surface,
            swapchain_state.device,
            swapchain_state.queue_type,
            swapchain_state.qf_id,
        );

        if destroy_command_pool {
            device.destroy_command_pool(swapchain_state.command_pool.into_raw());
        } else {
            swapchain_state.command_pool.reset();
        }

        for fb in swapchain_state.swapchain_framebuffers.into_iter() {
            device.destroy_framebuffer(fb);
        }

        device.destroy_graphics_pipeline(swapchain_state.gfx_pipeline);

        for dsl in swapchain_state.descriptor_set_layouts.into_iter() {
            device.destroy_descriptor_set_layout(dsl);
        }

        device.destroy_pipeline_layout(swapchain_state.pipeline_layout);

        device.destroy_render_pass(swapchain_state.render_pass);

        for (_, image_view) in swapchain_state.frame_images.into_iter() {
            device.destroy_image_view(image_view);
        }

        device.destroy_swapchain(swapchain_state.swapchain);

        (adapter, surface, device, queue_type, qf_id)
    }

    fn recreate(&mut self) {
        self.device.wait_idle().expect("Queues are not going idle!");

        let (swapchain, extent, backbuffer, format) =
            create_swap_chain(&self.adapter, &self.device, &mut self.surface, None);
        self.swapchain = swapchain;

        self.frame_images = create_image_views(backbuffer, format, &self.device);
        self.render_pass = create_render_pass(&self.device, Some(format));
        self.swapchain_framebuffers =
            create_framebuffers(&self.device, &self.render_pass, &self.frame_images, extent);
        let (descriptor_set_layouts, pipeline_layout, gfx_pipeline) =
            create_graphics_pipeline(&self.device, extent, &self.render_pass);
        self.descriptor_set_layouts = descriptor_set_layouts;
        self.pipeline_layout = pipeline_layout;
        self.gfx_pipeline = gfx_pipeline;
        self.command_pool = create_command_pool(&self.device, self.queue_type, self.qf_id);
        self.submission_command_buffers = create_command_buffers(
            &mut self.command_pool,
            &self.render_pass,
            &self.swapchain_framebuffers,
            extent,
            &self.gfx_pipeline,
        );
    }
}

fn draw_frame(
    command_queues: &mut Vec<hal::queue::CommandQueue<back::Backend, hal::Graphics>>,
    swapchain_state: &mut SwapchainState,
    image_available_semaphore: &<back::Backend as hal::Backend>::Semaphore,
    render_finished_semaphore: &<back::Backend as hal::Backend>::Semaphore,
    in_flight_fence: &<back::Backend as hal::Backend>::Fence,
) {
    swapchain_state
        .device
        .wait_for_fence(in_flight_fence, std::u64::MAX);

    let image_index = match swapchain_state.swapchain.acquire_image(
        std::u64::MAX,
        hal::window::FrameSync::Semaphore(image_available_semaphore),
    ) {
        Ok(image_index) => image_index,
        Err(acquire_error) => match acquire_error {
            hal::window::AcquireError::NotReady => {
                panic!("No timeout provided, yet image acquire timed out!")
            }
            hal::window::AcquireError::OutOfDate => {
                swapchain_state.recreate();
                return;
            }
            hal::window::AcquireError::SurfaceLost => panic!("Cannot acquire image: surface lost!"),
        },
    };

    {
        let submission = {
            hal::queue::submission::Submission::new()
                .wait_on(&[(
                    image_available_semaphore,
                    hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                )]).signal(vec![render_finished_semaphore])
                .submit(Some(&swapchain_state.submission_command_buffers[image_index as usize]))
        };

        swapchain_state.device.reset_fence(in_flight_fence);
        command_queues[0].submit(submission, Some(in_flight_fence));
    }

    swapchain_state
        .swapchain
        .present(
            &mut command_queues[0],
            image_index,
            vec![render_finished_semaphore],
        ).expect("presentation failed!");

    // if Swapchain::present returns an error, that means it got one of the results SUBOPTIMAL_KHR or ERROR_OUT_OF_DATE_KHR
    // (see documentation for hal::window::Swapchain::present)
    match {
        swapchain_state.swapchain.present(&mut command_queues[0], image_index, vec![render_finished_semaphore])
    } {
        Ok(()) => {},
        Err(()) => {
            swapchain_state.recreate();
        }
    }
}

fn create_command_buffers<'a>(
    command_pool: &'a mut hal::pool::CommandPool<back::Backend, hal::Graphics>,
    render_pass: &<back::Backend as hal::Backend>::RenderPass,
    framebuffers: &Vec<<back::Backend as hal::Backend>::Framebuffer>,
    extent: hal::window::Extent2D,
    pipeline: &<back::Backend as hal::Backend>::GraphicsPipeline,
) -> Vec<
    hal::command::Submit<
        back::Backend,
        hal::Graphics,
        hal::command::MultiShot,
        hal::command::Primary,
    >,
> {
    command_pool.reserve(framebuffers.iter().count());

    let mut submission_command_buffers: Vec<
        hal::command::Submit<
            back::Backend,
            hal::Graphics,
            hal::command::MultiShot,
            hal::command::Primary,
        >,
    > = Vec::new();

    for fb in framebuffers.iter() {
        let mut command_buffer: hal::command::CommandBuffer<
            back::Backend,
            hal::Graphics,
            hal::command::MultiShot,
            hal::command::Primary,
        > = command_pool.acquire_command_buffer(true);

        command_buffer.bind_graphics_pipeline(pipeline);
        {
            let render_area = hal::pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as _,
                h: extent.height as _,
            };
            let clear_values = vec![hal::command::ClearValue::Color(
                hal::command::ClearColor::Float([0.0, 0.0, 0.0, 4.0]),
            )];

            let mut render_pass_inline_encoder = command_buffer.begin_render_pass_inline(
                render_pass,
                fb,
                render_area,
                clear_values.iter(),
            );
            render_pass_inline_encoder.draw(0..3, 0..1);
        }

        let submission_command_buffer = command_buffer.finish();
        submission_command_buffers.push(submission_command_buffer);
    }

    submission_command_buffers
}

fn create_command_pool(
    device: &<back::Backend as hal::Backend>::Device,
    queue_type: hal::queue::QueueType,
    qf_id: hal::queue::family::QueueFamilyId,
) -> hal::pool::CommandPool<back::Backend, hal::Graphics> {
    let raw_command_pool =
        device.create_command_pool(qf_id, hal::pool::CommandPoolCreateFlags::empty());

    assert_eq!(hal::Graphics::supported_by(queue_type), true);
    unsafe { hal::pool::CommandPool::new(raw_command_pool) }
}

fn create_framebuffers(
    device: &<back::Backend as hal::Backend>::Device,
    render_pass: &<back::Backend as hal::Backend>::RenderPass,
    frame_images: &Vec<(
        <back::Backend as hal::Backend>::Image,
        <back::Backend as hal::Backend>::ImageView,
    )>,
    extent: hal::window::Extent2D,
) -> Vec<<back::Backend as hal::Backend>::Framebuffer> {
    let mut swapchain_framebuffers: Vec<<back::Backend as hal::Backend>::Framebuffer> = Vec::new();

    for (_, image_view) in frame_images.iter() {
        swapchain_framebuffers.push(
            device
                .create_framebuffer(
                    render_pass,
                    vec![image_view],
                    hal::image::Extent {
                        width: extent.width as _,
                        height: extent.height as _,
                        depth: 1,
                    },
                ).expect("failed to create framebuffer!"),
        );
    }

    swapchain_framebuffers
}

fn create_render_pass(
    device: &<back::Backend as hal::Backend>::Device,
    format: Option<hal::format::Format>,
) -> <back::Backend as hal::Backend>::RenderPass {
    let samples: u8 = 1;

    let ops = hal::pass::AttachmentOps {
        load: hal::pass::AttachmentLoadOp::Clear,
        store: hal::pass::AttachmentStoreOp::Store,
    };

    let stencil_ops = hal::pass::AttachmentOps::DONT_CARE;

    let layouts = hal::image::Layout::Undefined..hal::image::Layout::Present;

    let color_attachment = hal::pass::Attachment {
        format,
        samples,
        ops,
        stencil_ops,
        layouts,
    };

    let color_attachment_ref: hal::pass::AttachmentRef =
        (0, hal::image::Layout::ColorAttachmentOptimal);

    let subpass = hal::pass::SubpassDesc {
        colors: &[color_attachment_ref],
        depth_stencil: None,
        inputs: &[],
        resolves: &[],
        preserves: &[],
    };

    let subpass_dependency = hal::pass::SubpassDependency {
        passes: hal::pass::SubpassRef::External..hal::pass::SubpassRef::Pass(0),
        stages: hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT
            ..hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
        accesses: hal::image::Access::empty()
            ..hal::image::Access::COLOR_ATTACHMENT_READ
            | hal::image::Access::COLOR_ATTACHMENT_WRITE,
    };

    device.create_render_pass(&[color_attachment], &[subpass], &[subpass_dependency])
}

fn create_graphics_pipeline(
    device: &<back::Backend as hal::Backend>::Device,
    extent: hal::window::Extent2D,
    render_pass: &<back::Backend as hal::Backend>::RenderPass,
) -> (
    Vec<<back::Backend as hal::Backend>::DescriptorSetLayout>,
    <back::Backend as hal::Backend>::PipelineLayout,
    <back::Backend as hal::Backend>::GraphicsPipeline,
) {
    let vert_shader_code = glsl_to_spirv::compile(
        include_str!("09_shader_base.vert"),
        glsl_to_spirv::ShaderType::Vertex,
    ).expect("Error compiling vertex shader code.")
        .bytes()
        .map(|b| b.unwrap())
        .collect::<Vec<u8>>();

    let frag_shader_code = glsl_to_spirv::compile(
        include_str!("09_shader_base.frag"),
        glsl_to_spirv::ShaderType::Fragment,
    ).expect("Error compiling fragment shader code.")
        .bytes()
        .map(|b| b.unwrap())
        .collect::<Vec<u8>>();

    let vert_shader_module = device
        .create_shader_module(&vert_shader_code)
        .expect("Error creating shader module.");
    let frag_shader_module = device
        .create_shader_module(&frag_shader_code)
        .expect("Error creating fragment module.");

    let (ds_layouts, pipeline_layout, gfx_pipeline) = {
        let (vs_entry, fs_entry) = (
            hal::pso::EntryPoint::<back::Backend> {
                entry: "main",
                module: &vert_shader_module,
                specialization: &[],
            },
            hal::pso::EntryPoint::<back::Backend> {
                entry: "main",
                module: &frag_shader_module,
                specialization: &[],
            },
        );

        let shaders = hal::pso::GraphicsShaderSet {
            vertex: vs_entry,
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(fs_entry),
        };

        let rasterizer = hal::pso::Rasterizer {
            depth_clamping: false,
            polygon_mode: hal::pso::PolygonMode::Fill,
            cull_face: <hal::pso::Face>::BACK,
            front_face: hal::pso::FrontFace::Clockwise,
            depth_bias: None,
            conservative: false,
        };

        let vertex_buffers: Vec<hal::pso::VertexBufferDesc> = Vec::new();
        let attributes: Vec<hal::pso::AttributeDesc> = Vec::new();

        let input_assembler = hal::pso::InputAssemblerDesc::new(hal::Primitive::TriangleList);

        let blender = {
            let blend_state = hal::pso::BlendState::On {
                color: hal::pso::BlendOp::Add {
                    src: hal::pso::Factor::One,
                    dst: hal::pso::Factor::Zero,
                },
                alpha: hal::pso::BlendOp::Add {
                    src: hal::pso::Factor::One,
                    dst: hal::pso::Factor::Zero,
                },
            };

            hal::pso::BlendDesc {
                logic_op: Some(hal::pso::LogicOp::Copy),
                targets: vec![hal::pso::ColorBlendDesc(
                    hal::pso::ColorMask::ALL,
                    blend_state,
                )],
            }
        };

        let depth_stencil = hal::pso::DepthStencilDesc {
            depth: hal::pso::DepthTest::Off,
            depth_bounds: false,
            stencil: hal::pso::StencilTest::Off,
        };

        let multisampling: Option<hal::pso::Multisampling> = None;

        let baked_states = hal::pso::BakedStates {
            viewport: Some(hal::pso::Viewport {
                rect: hal::pso::Rect {
                    x: 0,
                    y: 0,
                    w: extent.width as i16,
                    h: extent.width as i16,
                },
                depth: (0.0..1.0),
            }),
            scissor: Some(hal::pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as i16,
                h: extent.height as i16,
            }),
            blend_color: None,
            depth_bounds: None,
        };

        let bindings = Vec::<hal::pso::DescriptorSetLayoutBinding>::new();
        let immutable_samplers = Vec::<<back::Backend as hal::Backend>::Sampler>::new();
        let ds_layouts: Vec<<back::Backend as hal::Backend>::DescriptorSetLayout> =
            vec![device.create_descriptor_set_layout(bindings, immutable_samplers)];
        let push_constants = Vec::<(hal::pso::ShaderStageFlags, std::ops::Range<u32>)>::new();
        let layout = device.create_pipeline_layout(&ds_layouts, push_constants);

        let subpass = hal::pass::Subpass {
            index: 0,
            main_pass: render_pass,
        };

        let flags = hal::pso::PipelineCreationFlags::empty();

        let parent = hal::pso::BasePipeline::None;

        let gfx_pipeline = {
            let desc = hal::pso::GraphicsPipelineDesc {
                shaders,
                rasterizer,
                vertex_buffers,
                attributes,
                input_assembler,
                blender,
                depth_stencil,
                multisampling,
                baked_states,
                layout: &layout,
                subpass,
                flags,
                parent,
            };

            device
                .create_graphics_pipeline(&desc, None)
                .expect("failed to create graphics pipeline!")
        };

        (ds_layouts, layout, gfx_pipeline)
    };

    device.destroy_shader_module(vert_shader_module);
    device.destroy_shader_module(frag_shader_module);

    (ds_layouts, pipeline_layout, gfx_pipeline)
}

fn create_image_views(
    backbuffer: hal::Backbuffer<back::Backend>,
    format: hal::format::Format,
    device: &<back::Backend as hal::Backend>::Device,
) -> Vec<(
    <back::Backend as hal::Backend>::Image,
    <back::Backend as hal::Backend>::ImageView,
)> {
    match backbuffer {
        hal::window::Backbuffer::Images(images) => images
            .into_iter()
            .map(|image| {
                let image_view = match device.create_image_view(
                    &image,
                    hal::image::ViewKind::D2,
                    format,
                    hal::format::Swizzle::NO,
                    hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                ) {
                    Ok(image_view) => image_view,
                    Err(_) => panic!("Error creating image view for an image!"),
                };

                (image, image_view)
            }).collect(),
        _ => unimplemented!(),
    }
}

fn create_swap_chain(
    adapter: &hal::Adapter<back::Backend>,
    device: &<back::Backend as hal::Backend>::Device,
    surface: &mut <back::Backend as hal::Backend>::Surface,
    previous_swapchain: Option<<back::Backend as hal::Backend>::Swapchain>,
) -> (
    <back::Backend as hal::Backend>::Swapchain,
    hal::window::Extent2D,
    hal::Backbuffer<back::Backend>,
    hal::format::Format,
) {
    let (caps, formats, _present_modes) = surface.compatibility(&adapter.physical_device);

    let format = formats.map_or(hal::format::Format::Rgba8Srgb, |formats| {
        formats
            .iter()
            .find(|format| format.base_format().1 == hal::format::ChannelType::Srgb)
            .map(|format| *format)
            .unwrap_or(formats[0])
    });

    // we already create the swapchain from the size of the surface
    let swap_config = SwapchainConfig::from_caps(&caps, format);
    let extent = swap_config.extent.clone();
    let (swapchain, backbuffer) = device.create_swapchain(surface, swap_config, previous_swapchain);

    (swapchain, extent, backbuffer, format)
}

fn create_surface(
    instance: &back::Instance,
    window: &winit::Window,
) -> <back::Backend as hal::Backend>::Surface {
    instance.create_surface(window)
}

fn create_device_with_graphics_queues(
    adapter: &mut hal::Adapter<back::Backend>,
    surface: &<back::Backend as hal::Backend>::Surface,
) -> (
    <back::Backend as hal::Backend>::Device,
    Vec<hal::queue::CommandQueue<back::Backend, hal::Graphics>>,
    hal::queue::QueueType,
    hal::queue::family::QueueFamilyId,
) {
    let family = adapter
        .queue_families
        .iter()
        .find(|family| {
            hal::Graphics::supported_by(family.queue_type())
                && family.max_queues() > 0
                && surface.supports_queue_family(family)
        }).expect("Could not find a queue family supporting graphics.");

    let priorities = vec![1.0; 1];

    let families = [(family, priorities.as_slice())];

    let hal::Gpu { device, mut queues } = adapter
        .physical_device
        .open(&families)
        .expect("Could not create device.");

    let mut queue_group = queues
        .take::<hal::Graphics>(family.id())
        .expect("Could not take ownership of relevant queue group.");

    let command_queues: Vec<_> = queue_group.queues.drain(..1).collect();

    (device, command_queues, family.queue_type(), family.id())
}

fn create_sync_objects(
    device: &<back::Backend as hal::Backend>::Device,
) -> (
    Vec<<back::Backend as hal::Backend>::Semaphore>,
    Vec<<back::Backend as hal::Backend>::Semaphore>,
    Vec<<back::Backend as hal::Backend>::Fence>,
) {
    let mut image_available_semaphores: Vec<<back::Backend as hal::Backend>::Semaphore> =
        Vec::new();
    let mut render_finished_semaphores: Vec<<back::Backend as hal::Backend>::Semaphore> =
        Vec::new();
    let mut in_flight_fences: Vec<<back::Backend as hal::Backend>::Fence> = Vec::new();

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        image_available_semaphores.push(device.create_semaphore());
        render_finished_semaphores.push(device.create_semaphore());
        in_flight_fences.push(device.create_fence(true));
    }

    (
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
    )
}

fn find_queue_families(adapter: &hal::Adapter<back::Backend>) -> QueueFamilyIds {
    let mut queue_family_ids = QueueFamilyIds::default();

    for queue_family in &adapter.queue_families {
        if queue_family.max_queues() > 0 && queue_family.supports_graphics() {
            queue_family_ids.graphics_family = Some(queue_family.id());
        }

        if queue_family_ids.is_complete() {
            break;
        }
    }

    queue_family_ids
}

fn is_adapter_suitable(adapter: &hal::Adapter<back::Backend>) -> bool {
    find_queue_families(adapter).is_complete()
}

fn pick_adapter(instance: &back::Instance) -> hal::Adapter<back::Backend> {
    let adapters = instance.enumerate_adapters();
    for adapter in adapters {
        if is_adapter_suitable(&adapter) {
            return adapter;
        }
    }
    panic!("No suitable adapter");
}

#[derive(Default)]
struct QueueFamilyIds {
    graphics_family: Option<hal::queue::QueueFamilyId>,
}

impl QueueFamilyIds {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some()
    }
}

fn init_window() -> (winit::Window, winit::EventsLoop) {
    let events_loop = winit::EventsLoop::new();
    let window_builder = winit::WindowBuilder::new()
        .with_dimensions(winit::dpi::LogicalSize::new(1024., 768.))
        .with_title(WINDOW_NAME.to_string());
    let window = window_builder.build(&events_loop).unwrap();
    (window, events_loop)
}

fn create_instance() -> back::Instance {
    back::Instance::create(WINDOW_NAME, 1)
}

fn init_hal(
    window: &winit::Window,
) -> (
    back::Instance,
    Vec<hal::queue::CommandQueue<back::Backend, hal::Graphics>>,
    SwapchainState,
    Vec<<back::Backend as hal::Backend>::Semaphore>,
    Vec<<back::Backend as hal::Backend>::Semaphore>,
    Vec<<back::Backend as hal::Backend>::Fence>,
) {
    let instance = create_instance();
    let surface = create_surface(&instance, window);
    let mut adapter = pick_adapter(&instance);
    let (device, command_queues, queue_type, qf_id) =
        create_device_with_graphics_queues(&mut adapter, &surface);
    let swapchain_state = SwapchainState::create(adapter, surface, device, queue_type, qf_id);
    let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
        create_sync_objects(&swapchain_state.device);
    (
        instance,
        command_queues,
        swapchain_state,
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
    )
}

fn clean_up(
    swapchain_state: SwapchainState,
    semaphores_per_case: Vec<Vec<<back::Backend as hal::Backend>::Semaphore>>,
    in_flight_fences: Vec<<back::Backend as hal::Backend>::Fence>,
) {
    {
        let device = &swapchain_state.device;

        for fence in in_flight_fences.into_iter() {
            device.destroy_fence(fence);
        }

        for semaphores_per_frame in semaphores_per_case.into_iter() {
            for semaphore in semaphores_per_frame.into_iter() {
                device.destroy_semaphore(semaphore);
            }
        }
    }

    // all returned data of SwapchainState::clean_up is dropped automatically
    SwapchainState::clean_up(swapchain_state, true);
}

fn main_loop(
    mut events_loop: winit::EventsLoop,
    command_queues: &mut Vec<hal::queue::CommandQueue<back::Backend, hal::Graphics>>,
    swapchain_state: &mut SwapchainState,
    image_available_semaphores: &Vec<<back::Backend as hal::Backend>::Semaphore>,
    render_finished_semaphores: &Vec<<back::Backend as hal::Backend>::Semaphore>,
    in_flight_fences: &Vec<<back::Backend as hal::Backend>::Fence>,
) {
    let mut current_frame: usize = 0;

    events_loop.run_forever(|event| match event {
        winit::Event::WindowEvent {
            event: winit::WindowEvent::CloseRequested,
            ..
        } => {
            swapchain_state
                .device
                .wait_idle()
                .expect("Queues are not going idle!");
            winit::ControlFlow::Break
        }
        _ => {
            draw_frame(
                command_queues,
                swapchain_state,
                &image_available_semaphores[current_frame],
                &render_finished_semaphores[current_frame],
                &in_flight_fences[current_frame],
            );
            current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
            winit::ControlFlow::Continue
        }
    });
}
