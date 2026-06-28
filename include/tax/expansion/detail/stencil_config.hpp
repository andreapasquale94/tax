#pragma once

#include <cstddef>

// Kernel dispatch configuration. Defaults ON here (not in the build system)
// so every consumer gets the fast paths regardless of how the headers are
// consumed. A project may pre-define either macro to 0 to fall back to the
// loop kernel, but the value MUST be identical in every translation unit
// linked together — differing values change inline definitions (ODR).
#ifndef TAX_USE_UNROLL
#define TAX_USE_UNROLL 1
#endif
#ifndef TAX_USE_STENCIL
#define TAX_USE_STENCIL 1
#endif
// Upper bound (bytes) on any precomputed stencil table (Cauchy or recurrence).
// Beyond this the dispatch falls back to the constexpr loop kernel instead of
// materialising a huge static table — which would otherwise be a hard compile
// error for many-variable, high-order expansions. Configured in-header like the
// other knobs; a project may pre-define it, but the value must be identical in
// every translation unit (ODR).
#ifndef TAX_STENCIL_MAX_BYTES
#define TAX_STENCIL_MAX_BYTES ( static_cast< std::size_t >( 64 ) << 20 )
#endif
