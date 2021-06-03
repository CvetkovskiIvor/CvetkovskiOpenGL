#pragma once
#include "vertexbuffer.h"
#include "vertexbufferlayout.h"


struct VertexArray
{
private:
    unsigned int m_vertexArrayID = 0;

public:
    VertexArray();
    ~VertexArray();

    void addBuffer(const VertexBuffer &vb, const VertexBufferLayout &layout) const;

    void bind() const;
    static void unBind();
};