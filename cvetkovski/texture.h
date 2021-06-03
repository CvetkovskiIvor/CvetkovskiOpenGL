#pragma once

#include <string>
class Texture
{
private:
    unsigned int m_textureID; // texture id
    std::string m_FilePath;
    unsigned char* m_localBuffer;
    int m_width, m_height, m_bitsPerPixel; // bits per pixels

public:
    explicit Texture(const std::string& path);
    ~Texture();

    void bind(unsigned int slot = 0) const;
    void unBind() const;

    int width() const { return m_width; }
    int height() const { return m_height; }
};
