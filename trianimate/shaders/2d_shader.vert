#version 330

// Input values
in vec2 in_vert;
in vec3 in_color;

// Output values for the shader. They end up in the buffer.
out vec3 v_color;

void main() {
    v_color = in_color;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}