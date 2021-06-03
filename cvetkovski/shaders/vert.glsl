#version 330 core
// layout(location = 0) in vec4 position;
// 0 je koo, a 1 je boja, sve dobijamo iz glavnog programa
layout(location = 0) in  vec3 in_Position;
layout(location = 1) in  vec3 in_Color;

uniform mat4 transform;

uniform mat4 view;
uniform mat4 projection;

// ex_Color jednostavno ide dalje u protocnom sustavu
out vec3 ex_Color;
void main()
{
    gl_Position = projection * view * transform * vec4(in_Position.x, in_Position.y, in_Position.z, 1.0);
    //gl_Position = projection * view * transform * position;

    // ne mijenjamo boju, samo je saljemo dalje
    ex_Color = in_Color;
}
