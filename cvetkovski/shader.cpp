//
// Created by flood on 01. 04. 2021..
//

#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
using namespace std;

//#include <stdlib.h>
#include <string.h>

#include <GL/glew.h>

#include "shader.h"

Shader::Shader(const std::vector<ShaderInfo>& shaders) : ID( glCreateProgram())
{
    const GLuint &ProgramID = ID; // glCreateProgram();
    GLint Result = GL_FALSE;
    int InfoLogLength;

    std::vector<GLuint> shader_ids;
    for(const auto& s : shaders)
    {
        GLuint id =glCreateShader(s.type);
        shader_ids.push_back(id);

        std::string VertexShaderCode;
        std::ifstream VertexShaderStream(s.file_path, std::ios::in);
        if(VertexShaderStream.is_open()){
            std::stringstream sstr;
            sstr << VertexShaderStream.rdbuf();
            VertexShaderCode = sstr.str();
            VertexShaderStream.close();
        }else{
            printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n",
                   s.file_path.c_str());
            getchar();
            return ;
        }

        // Compile Vertex Shader
        printf("Compiling shader : %s\n", s.file_path.c_str());
        char const * VertexSourcePointer = VertexShaderCode.c_str();
        glShaderSource(id, 1, &VertexSourcePointer , NULL);
        glCompileShader(id);

        // Check Vertex Shader
        glGetShaderiv(id, GL_COMPILE_STATUS, &Result);
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &InfoLogLength);
        if ( InfoLogLength > 0 ){
            std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
            glGetShaderInfoLog(id, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
            printf("%s\n", &VertexShaderErrorMessage[0]);
        }

        glAttachShader(ProgramID, id);
    }

    glLinkProgram(ProgramID);
    // Check the program
    glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
    glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
    if ( InfoLogLength > 0 ){
        std::vector<char> ProgramErrorMessage(InfoLogLength+1);
        glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
        printf("%s\n", &ProgramErrorMessage[0]);
    }

    for(const auto& id : shader_ids)
    {
        glDetachShader(ProgramID, id);
        glDeleteShader(id);
    }
}

int Shader::uniformLocation(const std::string& name)
{
    if (m_uniformLocationCache.find(name) != m_uniformLocationCache.end())
        return m_uniformLocationCache[name];

    const int location = glGetUniformLocation(ID, name.c_str());

    if (location == -1)
        std::cout << "No active uniform variable with name " << name << " found" << std::endl;

    m_uniformLocationCache[name] = location;

    return location;
}