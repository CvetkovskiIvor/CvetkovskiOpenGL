#version 330 core
// boju dobijamo iz vertex shadera
// ime varijeble je isto...
in  vec3 ex_Color;
out vec4 fragColor;


void main(void)
{
    fragColor = vec4(ex_Color,1.0);
}
