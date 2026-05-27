#include <cassert>
#include <iostream>

#include <cog/glow/glow.hpp>

int main() {
    cog::glow::TypeRef type(cog::glow::ElemKind::FloatTy, {2, 2});
    assert(type.num_elements() == 4);
    assert(type.size() == 16);

    cog::glow::Tensor tensor(type);
    tensor.fill(1.5f);
    assert(tensor.at_f(0, 0) == 1.5f);
    assert(tensor.at_f(1, 1) == 1.5f);

    cog::glow::Function fn("smoke");
    auto input = fn.add_placeholder("input", type);
    fn.set_output(input);
    auto order = fn.topo_sort();
    assert(!order.empty());
    assert(fn.node(input) != nullptr);

    std::cout << "CogLow smoke test passed" << std::endl;
    return 0;
}
