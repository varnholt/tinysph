#pragma once

// tinysph follows semantic versioning. until 1.0 the api may still change between minor versions;
// from 1.0 on, breaking changes only happen with a new major version.

#define TINYSPH_VERSION_MAJOR 0
#define TINYSPH_VERSION_MINOR 1
#define TINYSPH_VERSION_PATCH 0

namespace tinysph
{

struct Version
{
   int major;
   int minor;
   int patch;
};

inline constexpr Version version{TINYSPH_VERSION_MAJOR, TINYSPH_VERSION_MINOR, TINYSPH_VERSION_PATCH};

}  // namespace tinysph
