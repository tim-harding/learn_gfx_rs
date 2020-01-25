#version 450

layout (location = 1) in vec3 frag_color;

// Locations are required for SPIRV compilation
layout (location = 0) out vec4 color;

void main() {
    color = vec4(1.0);
}