#include "ShaderProgram.h"

ShaderProgram::ShaderProgram(const GLchar* vertexPath, const GLchar* fragmentPath) {
    // Retrieve the vertex/fragment source code from file path
    std::string vertexCode;
    std::string fragmentCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;

    // ensure ifstream objects can throw exceptions:
    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        // open files
        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);
        std::stringstream vShaderStream, fShaderStream;

        // read file's buffer contents into streams
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();

        // close file handlers
        vShaderFile.close();
        fShaderFile.close();

        // convert stream into string
        vertexCode = vShaderStream.str();
        fragmentCode = fShaderStream.str();
    }
    catch (std::ifstream::failure e) {
        std::cout << "Shader file not successfully read" << std::endl;
    }

    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();

    // Shader compilation
    // 2. compile shaders
    unsigned int vertex, fragment;
    int success;
    char infoLog[512];

    // Vertex Shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    // print compile errors if any
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertex, 512, NULL, infoLog);
        std::cout << "Vertex Shader compilation failed\n" << infoLog << std::endl;
    };

    // Fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    // print compile errors if any
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragment, 512, NULL, infoLog);
        std::cout << "Fragment Shader compilation failed\n" << infoLog << std::endl;
    };

    // ShaderProgram
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    // print linking errors if any
    glGetProgramiv(ID, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(ID, 512, NULL, infoLog);
        std::cout << "Shader Program linking failed\n" << infoLog << std::endl;
    }

    // Get various attribute locations
    glBindAttribLocation(ID, 0, "vsPos");
    glBindAttribLocation(ID, 1, "vsNor");

    // Don't need these anymore
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

void ShaderProgram::Draw(Drawable& d) {
    use();

    // Position
    int attrPos = glGetAttribLocation(ID, "vsPos");
    if (attrPos != -1 && d.bindBufPos()) {
        glVertexAttribPointer(attrPos, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(attrPos);
    }

    // Normal
    int attrNor = glGetAttribLocation(ID, "vsNor");
    if (attrNor != -1 && d.bindBufNor()) {
        glVertexAttribPointer(attrNor, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(attrNor);
    }

    d.bindBufIdx();
    glDrawElements(d.drawMode(), d.idxCount(), GL_UNSIGNED_INT, 0);

    if (attrPos != -1) { glDisableVertexAttribArray(0); }
    if (attrNor != -1) { glDisableVertexAttribArray(1); }
}

void ShaderProgram::setCameraViewProj(const char* uniformName, const glm::mat4& camViewProj) {
    use();
    glUniformMatrix4fv(glGetUniformLocation(ID, uniformName), 1, GL_FALSE, glm::value_ptr(camViewProj));
}

void ShaderProgram::setUniformColor(const char* uniformName, const glm::vec3& color) {
    use();
    glUniform3fv(glGetUniformLocation(ID, uniformName), 1, glm::value_ptr(color));
}
