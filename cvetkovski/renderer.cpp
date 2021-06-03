#include "renderer.h"
#include "GL/glew.h"

void Renderer::clear() { glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); }
void Renderer::drawTriangles(const VertexArray &va, const IndexBuffer &ib, const Shader &shader)
{
    shader.bind();
    va.bind();
    ib.bind();
    glDrawElements(GL_TRIANGLES, ib.count(), GL_UNSIGNED_INT, nullptr);
}

void Renderer::drawTriangleStrip(const VertexArray &va, const IndexBuffer &ib, const Shader &shader)
{
    shader.bind();
    va.bind();
    ib.bind();
    glDrawElements(GL_TRIANGLE_STRIP, ib.count(), GL_UNSIGNED_INT, nullptr);
}

void Renderer::drawPoints(const VertexArray &va, const IndexBuffer &ib, const Shader &shader)
{
    shader.bind();
    va.bind();
    ib.bind();
    glDrawElements(GL_POINTS, ib.count(), GL_UNSIGNED_INT, nullptr);
}

void Renderer::drawLines(const VertexArray &va, const IndexBuffer &ib, const Shader &shader)
{
    shader.bind();
    va.bind();
    ib.bind();
    glDrawElements(GL_LINE_STRIP, ib.count(), GL_UNSIGNED_INT, nullptr);
}

void Renderer::unBind(/*const VertexArray &va, const IndexBuffer &ib, const Shader &shader*/)
{
    Shader::unBind();
    VertexArray::unBind();
    IndexBuffer::unBind();
}