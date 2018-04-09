#include "Drawable.h"

void Drawable::destroy() {
    glDeleteBuffers(1, &bufIdx);
    glDeleteBuffers(1, &bufPos);
    glDeleteBuffers(1, &bufNor);
}

bool Drawable::bindBufIdx() {
    if (idxBound) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufIdx);
    }
    return idxBound;
}

bool Drawable::bindBufPos() {
    if (posBound) {
        glBindBuffer(GL_ARRAY_BUFFER, bufPos);
    }
    return posBound;
}

bool Drawable::bindBufNor() {
    if (norBound) {
        glBindBuffer(GL_ARRAY_BUFFER, bufNor);
    }
    return norBound;
}

void Drawable::genBufIdx() {
    idxBound = true;
    glGenBuffers(1, &bufIdx);
}

void Drawable::genBufPos() {
    posBound = true;
    glGenBuffers(1, &bufPos);
}

void Drawable::genBufNor() {
    norBound = true;
    glGenBuffers(1, &bufNor);
}
