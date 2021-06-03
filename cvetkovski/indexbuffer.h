#pragma once
#include <vector>

struct IndexBuffer {
private:
    unsigned int m_indexBufferID=0;
    unsigned int m_count;

public:
    explicit IndexBuffer(const std::vector<unsigned int> &indices);
    ~IndexBuffer();

    void bind() const;
    static void unBind();

    [[nodiscard]] inline unsigned int count() const {return m_count;}
};