#version 450

layout (push_constant) uniform PushConsts {
    float mouse_x;
    float mouse_y;
} push;

layout (location = 0) in vec2 position;

// Equivalent to OpenGL gl_Position
layout (location = 0) out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    vec2 offset = vec2(push.mouse_x, push.mouse_y) * 2.0 - 0.5;
    gl_Position = vec4(position + offset, 0.0, 1.0);
}