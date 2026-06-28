#pragma once

// Back-compatibility shim. The basis-generic carrier `Expansion< T, Basis,
// Scheme, Storage >` (with `TaylorExpansion` as its TaylorBasis alias) now lives
// in <tax/core/expansion.hpp>. Include that directly in new code.

#include <tax/core/expansion.hpp>
