#pragma once

#include <SDL3/SDL_opengl.h>

// opengl 1.1 functions come straight from the system library, everything newer is loaded at runtime.
// usage: gl::CreateShader(...) instead of glCreateShader(...).
#define SPH_GL_FUNCTIONS(FUNCTION)                                     \
   FUNCTION(PFNGLACTIVETEXTUREPROC, ActiveTexture)                     \
   FUNCTION(PFNGLATTACHSHADERPROC, AttachShader)                       \
   FUNCTION(PFNGLBINDFRAMEBUFFERPROC, BindFramebuffer)                 \
   FUNCTION(PFNGLCHECKFRAMEBUFFERSTATUSPROC, CheckFramebufferStatus)   \
   FUNCTION(PFNGLDELETEFRAMEBUFFERSPROC, DeleteFramebuffers)           \
   FUNCTION(PFNGLFRAMEBUFFERTEXTURE2DPROC, FramebufferTexture2D)       \
   FUNCTION(PFNGLGENFRAMEBUFFERSPROC, GenFramebuffers)                 \
   FUNCTION(PFNGLUNIFORM2FPROC, Uniform2f)                             \
   FUNCTION(PFNGLUNIFORM3FPROC, Uniform3f)                             \
   FUNCTION(PFNGLUNIFORMMATRIX3FVPROC, UniformMatrix3fv)               \
   FUNCTION(PFNGLBINDBUFFERPROC, BindBuffer)                           \
   FUNCTION(PFNGLBINDVERTEXARRAYPROC, BindVertexArray)                 \
   FUNCTION(PFNGLBUFFERDATAPROC, BufferData)                           \
   FUNCTION(PFNGLCOMPILESHADERPROC, CompileShader)                     \
   FUNCTION(PFNGLCREATEPROGRAMPROC, CreateProgram)                     \
   FUNCTION(PFNGLCREATESHADERPROC, CreateShader)                       \
   FUNCTION(PFNGLDELETEBUFFERSPROC, DeleteBuffers)                     \
   FUNCTION(PFNGLDELETEPROGRAMPROC, DeleteProgram)                     \
   FUNCTION(PFNGLDELETESHADERPROC, DeleteShader)                       \
   FUNCTION(PFNGLDELETEVERTEXARRAYSPROC, DeleteVertexArrays)           \
   FUNCTION(PFNGLENABLEVERTEXATTRIBARRAYPROC, EnableVertexAttribArray) \
   FUNCTION(PFNGLGENBUFFERSPROC, GenBuffers)                           \
   FUNCTION(PFNGLGENVERTEXARRAYSPROC, GenVertexArrays)                 \
   FUNCTION(PFNGLGETPROGRAMINFOLOGPROC, GetProgramInfoLog)             \
   FUNCTION(PFNGLGETPROGRAMIVPROC, GetProgramiv)                       \
   FUNCTION(PFNGLGETSHADERINFOLOGPROC, GetShaderInfoLog)               \
   FUNCTION(PFNGLGETSHADERIVPROC, GetShaderiv)                         \
   FUNCTION(PFNGLGETUNIFORMLOCATIONPROC, GetUniformLocation)           \
   FUNCTION(PFNGLLINKPROGRAMPROC, LinkProgram)                         \
   FUNCTION(PFNGLSHADERSOURCEPROC, ShaderSource)                       \
   FUNCTION(PFNGLUNIFORM1FPROC, Uniform1f)                             \
   FUNCTION(PFNGLUNIFORM1IPROC, Uniform1i)                             \
   FUNCTION(PFNGLUNIFORMMATRIX4FVPROC, UniformMatrix4fv)               \
   FUNCTION(PFNGLUSEPROGRAMPROC, UseProgram)                           \
   FUNCTION(PFNGLVERTEXATTRIBPOINTERPROC, VertexAttribPointer)

namespace gl
{

#define SPH_DECLARE_GL_FUNCTION(type, name) extern type name;
SPH_GL_FUNCTIONS(SPH_DECLARE_GL_FUNCTION)
#undef SPH_DECLARE_GL_FUNCTION

/// \brief loads all functions above from the current context.
/// \return false if any of them is missing
[[nodiscard]] bool load();

}  // namespace gl
