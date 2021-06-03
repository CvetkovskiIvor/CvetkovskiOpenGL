#include "vertexbuffer.h"
#include "GL/glew.h"

VertexBuffer::VertexBuffer(const std::vector<float> &arr)
{
    glGenBuffers(1, &m_bufferID);
    glBindBuffer(GL_ARRAY_BUFFER, m_bufferID);
    glBufferData(GL_ARRAY_BUFFER, arr.size() * sizeof(float), arr.data(), /*GL_DYNAMIC_DRAW*/GL_STATIC_DRAW);
}

VertexBuffer::VertexBuffer(const std::vector<glm::vec3> &arr)
{
    glGenBuffers(1, &m_bufferID);
    glBindBuffer(GL_ARRAY_BUFFER, m_bufferID);
    glBufferData(GL_ARRAY_BUFFER, arr.size() * sizeof(glm::vec3), &arr.front(), /*GL_DYNAMIC_DRAW*/GL_STATIC_DRAW);
}

VertexBuffer::~VertexBuffer() { glDeleteBuffers(1, &m_bufferID); }

void VertexBuffer::updateBufferData(const std::vector<float> &arr)
{
    glBindBuffer(GL_ARRAY_BUFFER, m_bufferID);
    glBufferSubData(GL_ARRAY_BUFFER, 0, arr.size() * sizeof(float), arr.data());
}

void VertexBuffer::bind() const { glBindBuffer(GL_ARRAY_BUFFER, m_bufferID); }
void VertexBuffer::unBind() { glBindBuffer(GL_ARRAY_BUFFER, 0); }
