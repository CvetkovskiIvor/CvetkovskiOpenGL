#version 330 core

layout(location = 0) in vec3 vertex_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in  vec2 vertexUV;

out vec3 FragPos;
out vec3 Normal;
out vec2 UV;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
    FragPos = vec3(model * vec4(vertex_pos, 1.0));
    Normal = mat3(transpose(inverse(model))) * in_normal;

    UV = vertexUV;

    gl_Position = proj * view * vec4(FragPos, 1.0);
}
