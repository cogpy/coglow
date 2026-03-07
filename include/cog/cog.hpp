// cog/cog.hpp — Unified umbrella header for all CogPy modules
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
//
// Include this single header to access all modules:
//   cog::core   — Shared types (Atom, AtomSpace, TruthValue, AttentionValue)
//   cog::plan9  — Plan 9 cognitive OS (9P protocol, CogFS, MachSpace)
//   cog::pilot  — Deep Tree Echo Reservoir (A000081, B-Series, J-Surface, P-System)
//   cog::mach   — Mach microkernel cognitive (fixed-point tensors, IPC, VM)
//   cog::lux    — Cognitive node graph (typed nodes/edges, traversal, DOT export)
//   cog::glow   — Neural network compiler (graph IR, passes, interpreter)
//   cog::gml    — Tensor library for ML (quantization, auto-diff, optimizers)
//   cog::prime  — AGI architecture (cognitive cycle, ontogenesis, memory systems)
//   cog::webvm  — Web AtomSpace VM (Scheme REPL, JSON serialization)
//
#ifndef COG_HPP
#define COG_HPP

#include "core/core.hpp"
#include "plan9/plan9.hpp"
#include "pilot/pilot.hpp"
#include "mach/mach.hpp"
#include "lux/lux.hpp"
#include "glow/glow.hpp"
#include "gml/gml.hpp"
#include "prime/prime.hpp"
#include "webvm/webvm.hpp"

#endif // COG_HPP
