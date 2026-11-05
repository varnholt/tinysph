#include "gl_functions.h"

#include <SDL3/SDL_log.h>
#include <SDL3/SDL_video.h>

namespace gl
{

#define SPH_DEFINE_GL_FUNCTION(type, name) type name = nullptr;
SPH_GL_FUNCTIONS(SPH_DEFINE_GL_FUNCTION)
#undef SPH_DEFINE_GL_FUNCTION

bool load()
{
   auto all_loaded = true;

#define SPH_LOAD_GL_FUNCTION(type, name)                                              \
   name = reinterpret_cast<type>(SDL_GL_GetProcAddress("gl" #name));                  \
   if (name == nullptr)                                                               \
   {                                                                                  \
      SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "missing opengl function gl" #name); \
      all_loaded = false;                                                             \
   }
   SPH_GL_FUNCTIONS(SPH_LOAD_GL_FUNCTION)
#undef SPH_LOAD_GL_FUNCTION

   return all_loaded;
}

}  // namespace gl
