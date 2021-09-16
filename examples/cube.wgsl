struct VertexOutput {
    [[location(0)]] texture_coordinates: vec2<f32>;
    [[builtin(position)]] position: vec4<f32>;
};

[[block]]
struct UniformBuffer {
    matrix: mat4x4<f32>;
};

[[group(0), binding(0)]]
var<uniform> uniform_buffer: UniformBuffer;

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] position: vec4<f32>,
    [[location(1)]] texture_coordinates: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.texture_coordinates = texture_coordinates;
    out.position = uniform_buffer.matrix * position;
    return out;
}

[[group(0), binding(1)]]
var texture: texture_2d<f32>;

[[group(0), binding(2)]]
var sampler: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(texture, sampler, in.texture_coordinates);
}