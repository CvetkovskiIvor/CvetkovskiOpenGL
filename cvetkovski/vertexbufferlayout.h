#pragma once
#include "GL/glew.h"
#include <vector>

struct VertexBufferElement {
    unsigned int type;
    unsigned int count;
    unsigned char normalized;

    static unsigned int sizeOfType(unsigned int type)
    {
        switch (type)
        {
            default:
                return 0;
            case GL_FLOAT:
                return sizeof(GLfloat);
            case GL_UNSIGNED_INT:
                return sizeof(GLuint);
            case GL_UNSIGNED_BYTE:
                return sizeof(GLbyte);
        }
    }
};

class VertexBufferLayout
{
private:
    unsigned int m_Stride = 0;
    std::vector<VertexBufferElement> m_Elements;

public:
    VertexBufferLayout() = default;

    void addFloat(unsigned int count) { add(GL_FLOAT, count, GL_FALSE); }
    //    void addUnsignedInt(unsigned int count)  { add(GL_UNSIGNED_INT, count, GL_FALSE); }
    //    void aAddUnsignedByte(unsigned int count) { add(GL_UNSIGNED_BYTE, count, GL_TRUE); }

    [[nodiscard]] inline const std::vector<VertexBufferElement> &elements() const { return m_Elements; }
    [[nodiscard]] inline unsigned int stride() const { return m_Stride; }

private:
    void add(unsigned int type, unsigned int count, unsigned char normalized)
    {
        m_Elements.emplace_back(VertexBufferElement({type, count, normalized}));
        m_Stride += count * VertexBufferElement::sizeOfType(type);
    }
};