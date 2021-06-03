#include "vertexarray.h"

VertexArray::VertexArray() { glGenVertexArrays(1, &m_vertexArrayID); }
VertexArray::~VertexArray() { glDeleteVertexArrays(1, &m_vertexArrayID); }

void VertexArray::addBuffer(const VertexBuffer &vb, const VertexBufferLayout &layout) const
{
    bind();
    vb.bind();
    const std::vector<VertexBufferElement> &elements = layout.elements();
    unsigned int offset = 0;
    for (unsigned int i = 0; i < elements.size(); i++)
    {
        const VertexBufferElement element = elements[i];
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, element.count, element.type, element.normalized, layout.stride(),
                              (void *) (uintptr_t)(offset));
        offset += element.count * VertexBufferElement::sizeOfType(element.type);
    }
}

void VertexArray::bind() const { glBindVertexArray(m_vertexArrayID); }
void VertexArray::unBind() { glBindVertexArray(0); }
