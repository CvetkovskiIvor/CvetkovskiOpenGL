#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "texture.h"
#include "GL/glew.h"


Texture::Texture(const std::string& path)
    : m_textureID(0),
      m_FilePath(path),
      m_localBuffer(nullptr),
      m_width(0), m_height(0),
      m_bitsPerPixel(0)
{
    stbi_set_flip_vertically_on_load(1);
    m_localBuffer = stbi_load(path.c_str(),
                              &m_width, &m_height,
                              &m_bitsPerPixel, 4); // 4 jer rgba
    glGenTextures(1, &m_textureID);
    glBindTexture(GL_TEXTURE_2D, m_textureID); // Bind without slot selection

    // mora se postaviti ova 4 parametra (fje)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                 m_width, m_height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, m_localBuffer);
    unBind();

    if (m_localBuffer)
        stbi_image_free(m_localBuffer);
};

Texture::~Texture()
{
    glDeleteTextures(1, &m_textureID);
}

void Texture::bind(unsigned int slot) const
{
    // koji slot postavljamo 0-31
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, m_textureID);
}

void Texture::unBind() const
{
    glBindTexture(GL_TEXTURE_2D, 0);
}
