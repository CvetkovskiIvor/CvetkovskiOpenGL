#include "indexbuffer.h"
#include "GL/glew.h"
// #include <cassert>

IndexBuffer::IndexBuffer(const std::vector<unsigned int> &indices) : m_count(indices.size())
{
    //assert(sizeof(unsigned int) == sizeof(GLuint));

    glGenBuffers(1, &m_indexBufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBufferID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(float), indices.data(),
                 GL_STATIC_DRAW);
}

IndexBuffer::~IndexBuffer() { glDeleteBuffers(1, &m_indexBufferID); }

void IndexBuffer::bind() const { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBufferID); }
void IndexBuffer::unBind() { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); }
