#pragma once

#include "glad/glad.h"

class Drawable {
protected:
    int count; // Size of index buffer
    GLuint bufIdx; // Index buffer 
    GLuint bufPos; // Position buffer
    GLuint bufNor; // Normals buffer

    bool idxBound; // Flags for whether or not each buffer has been bound
    bool posBound;
    bool norBound;
public:
    Drawable() : count(-1), bufIdx(), bufPos(), bufNor(), idxBound(false), posBound(false), norBound(false) {}
    inline int idxCount() { return count; }
    void destroy();
    bool bindBufIdx();
    bool bindBufPos();
    bool bindBufNor();
    void genBufIdx();
    void genBufPos();
    void genBufNor();

    // Inheritable functions
    virtual void create() = 0;
    virtual GLenum drawMode() = 0;
};
