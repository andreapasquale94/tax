#pragma once
// Facade for the core expansion capability: the Expansion carrier, schemes, storage, the Taylor
// basis policy, named + mixed expansions, and the operator surface. Kernels are an internal detail
// (expansion/detail/).

#include <tax/expansion/concepts.hpp>
#include <tax/expansion/enumeration.hpp>
#include <tax/expansion/expansion.hpp>
#include <tax/expansion/mixed_named.hpp>
#include <tax/expansion/multi_index.hpp>
#include <tax/expansion/named.hpp>
#include <tax/expansion/ops/arithmetic.hpp>
#include <tax/expansion/ops/math_binary.hpp>
#include <tax/expansion/ops/math_unary.hpp>
#include <tax/expansion/ops/mixed_arithmetic.hpp>
#include <tax/expansion/ops/mixed_math_binary.hpp>
#include <tax/expansion/ops/mixed_math_unary.hpp>
#include <tax/expansion/ops/named_arithmetic.hpp>
#include <tax/expansion/ops/named_math_binary.hpp>
#include <tax/expansion/ops/named_math_unary.hpp>
#include <tax/expansion/ops/sparse.hpp>
#include <tax/expansion/promote.hpp>
#include <tax/expansion/storage/dense.hpp>
#include <tax/expansion/storage/sparse.hpp>
