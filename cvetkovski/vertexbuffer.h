#pragma once
#include <vector>
#include <glm/glm.hpp>

struct VertexBuffer {
private:
    unsigned int m_bufferID=0;

public:
    explicit VertexBuffer(const std::vector<float> &arr);
    explicit VertexBuffer(const std::vector<glm::vec3> &arr);
    ~VertexBuffer();

    void updateBufferData(const std::vector<float> &arr);

    void bind() const;
    static void unBind();
};