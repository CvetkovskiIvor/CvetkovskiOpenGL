#ifndef GRAFIKA_VJEZBE_SHADER_H
#define GRAFIKA_VJEZBE_SHADER_H

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <vector>

#include <unordered_map>
struct ShaderInfo {
    unsigned int type;
    std::string file_path;
};

struct Shader {

private:
    std::unordered_map<std::string, int> m_uniformLocationCache;
    int uniformLocation(const std::string& name);

public:
    const unsigned int ID;
    explicit Shader(const std::vector<ShaderInfo> &shaders);

    ~Shader()
    {
        glDeleteProgram(ID);
    }
    // ------------------------------------------------------------------------
    void bind() const { glUseProgram(ID); }

    // ------------------------------------------------------------------------
    static void unBind() { glUseProgram(0); }

    // ------------------------------------------------------------------------
    void setBool(const std::string &name, bool value)
    {
        glUniform1i(uniformLocation(name), (int) value);
    }
    // ------------------------------------------------------------------------
    void setInt(const std::string &name, int value)
    {
        glUniform1i(uniformLocation(name), value);
    }
    // ------------------------------------------------------------------------
    void setFloat(const std::string &name, float value)
    {
        glUniform1f(uniformLocation(name), value);
    }
    // ------------------------------------------------------------------------
    void setVec2(const std::string &name, const glm::vec2 &value)
    {
        glUniform2fv(uniformLocation(name), 1, &value[0]);
    }
    void setVec2(const std::string &name, float x, float y)
    {
        glUniform2f(uniformLocation(name), x, y);
    }
    // ------------------------------------------------------------------------
    void setVec3(const std::string &name, const glm::vec3 &value)
    {
        glUniform3fv(uniformLocation(name), 1, glm::value_ptr(value));
    }
    void setVec3(const std::string &name, float x, float y, float z)
    {
        glUniform3f(uniformLocation(name), x, y, z);
    }
    // ------------------------------------------------------------------------
    void setVec4(const std::string &name, const glm::vec4 &value)
    {
        glUniform4fv(uniformLocation(name), 1, glm::value_ptr(value));
    }
    void setVec4(const std::string &name, float x, float y, float z, float w)
    {
        glUniform4f(uniformLocation(name), x, y, z, w);
    }
    // ------------------------------------------------------------------------
    void setMat2(const std::string &name, const glm::mat2 &mat, const int& transpose = GL_FALSE)
    {
        glUniformMatrix2fv(uniformLocation(name), 1, transpose, glm::value_ptr(mat));
    }
    // ------------------------------------------------------------------------
    void setMat3(const std::string &name, const glm::mat3 &mat, const int& transpose = GL_FALSE)
    {
        glUniformMatrix3fv(uniformLocation(name), 1, transpose, glm::value_ptr(mat));
    }
    // ------------------------------------------------------------------------
    void setMat4(const std::string &name, const glm::mat4 &mat, const int& transpose = GL_FALSE)
    {
        glUniformMatrix4fv(uniformLocation(name), 1, transpose, glm::value_ptr(mat));
    }
};


#endif//GRAFIKA_VJEZBE_SHADER_H
