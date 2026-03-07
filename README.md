# CogPy Header-Only C++11 Library Suite

A unified, header-only C++11 library implementing all CogPy cognitive architecture modules with **zero external dependencies**.

## Modules

| Module | Namespace | Description |
|--------|-----------|-------------|
| **core** | `cog` | AtomSpace, Atom, TruthValue, AttentionValue, Arena, Spinlock |
| **plan9** | `cog::plan9` | Plan 9 cognitive OS — 9P2000 protocol, CogFS, MachSpace |
| **pilot** | `cog::pilot` | Deep Tree Echo reservoir — A000081, B-Series, J-Surface, P-System, ESN |
| **mach** | `cog::mach` | Mach microkernel cognitive — Q16.16 fixed-point, IPC, VM, kernel AtomSpace |
| **lux** | `cog::lux` | Cognitive node graph — typed nodes/edges, BFS/DFS, PageRank, DOT export |
| **glow** | `cog::glow` | Neural network compiler — graph IR, optimization passes, interpreter |
| **gml** | `cog::gml` | Tensor library — N-dim tensors, Q4/Q8 quantization, Adam/L-BFGS |
| **prime** | `cog::prime` | AGI architecture — cognitive cycle, PLN, pattern matching, memory systems |
| **webvm** | `cog::webvm` | Web AtomSpace VM — S-expression parser, Scheme REPL, JSON serialization |

## Quick Start

```cpp
#include <cog/cog.hpp>  // Everything

int main() {
    cog::AtomSpace as;
    auto cat = as.add_node(cog::AtomType::CONCEPT_NODE, "cat");
    auto animal = as.add_node(cog::AtomType::CONCEPT_NODE, "animal");
    as.add_link(cog::AtomType::INHERITANCE_LINK, {cat, animal});
    return 0;
}
```

```bash
g++ -std=c++11 -I include -o myapp myapp.cpp
```

## Build & Test

```bash
# Direct
g++ -std=c++11 -I include -o test_all test/test_all.cpp && ./test_all

# CMake
mkdir build && cd build && cmake .. && make && ctest
```

## License

MIT
