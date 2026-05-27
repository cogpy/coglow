// cog/glow/glow.hpp — Neural Network Graph Compiler & Optimizer
// Graph IR, type system, optimization passes, lowering, interpreter
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
#ifndef COG_GLOW_HPP
#define COG_GLOW_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <algorithm>
#include <sstream>
#include <cassert>
#include <numeric>
#include <queue>

namespace cog { namespace glow {

// ─────────────────────────────────────────────────────────────────────────────
// Element Types
// ─────────────────────────────────────────────────────────────────────────────
enum class ElemKind : uint8_t {
    FloatTy   = 0,
    Float16Ty = 1,
    Int8QTy   = 2,
    Int16QTy  = 3,
    Int32QTy  = 4,
    Int32ITy  = 5,
    Int64ITy  = 6,
    BoolTy    = 7
};

inline size_t elem_size(ElemKind k) {
    switch (k) {
        case ElemKind::FloatTy:   return 4;
        case ElemKind::Float16Ty: return 2;
        case ElemKind::Int8QTy:   return 1;
        case ElemKind::Int16QTy:  return 2;
        case ElemKind::Int32QTy:  return 4;
        case ElemKind::Int32ITy:  return 4;
        case ElemKind::Int64ITy:  return 8;
        case ElemKind::BoolTy:    return 1;
    }
    return 0;
}

inline const char* elem_name(ElemKind k) {
    static const char* names[] = {
        "float32","float16","int8q","int16q","int32q","int32","int64","bool"
    };
    return names[static_cast<uint8_t>(k)];
}

// ─────────────────────────────────────────────────────────────────────────────
// TypeRef — Shape + element type
// ─────────────────────────────────────────────────────────────────────────────
struct TypeRef {
    ElemKind elem;
    std::vector<size_t> dims;
    float scale;    // For quantized types
    int32_t offset; // For quantized types

    TypeRef() : elem(ElemKind::FloatTy), scale(1.0f), offset(0) {}
    TypeRef(ElemKind e, const std::vector<size_t>& d)
        : elem(e), dims(d), scale(1.0f), offset(0) {}

    size_t size() const {
        size_t s = elem_size(elem);
        for (auto d : dims) s *= d;
        return s;
    }

    size_t num_elements() const {
        size_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }

    size_t ndim() const { return dims.size(); }

    bool operator==(const TypeRef& o) const {
        return elem == o.elem && dims == o.dims;
    }

    std::string to_string() const {
        std::ostringstream os;
        os << elem_name(elem) << "<";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) os << "x";
            os << dims[i];
        }
        os << ">";
        return os.str();
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Tensor — Runtime tensor with data
// ─────────────────────────────────────────────────────────────────────────────
class Tensor {
public:
    Tensor() {}

    explicit Tensor(const TypeRef& type) : type_(type) {
        data_.resize(type.size(), 0);
    }

    Tensor(ElemKind elem, const std::vector<size_t>& dims)
        : type_(elem, dims)
    {
        data_.resize(type_.size(), 0);
    }

    // Float access
    float& at_f(size_t idx) {
        assert(type_.elem == ElemKind::FloatTy);
        assert(idx < type_.num_elements());
        return reinterpret_cast<float*>(data_.data())[idx];
    }
    float at_f(size_t idx) const {
        assert(type_.elem == ElemKind::FloatTy);
        assert(idx < type_.num_elements());
        return reinterpret_cast<const float*>(data_.data())[idx];
    }

    // 2D float access
    float& at_f(size_t r, size_t c) {
        assert(type_.ndim() >= 2);
        return at_f(r * type_.dims[1] + c);
    }
    float at_f(size_t r, size_t c) const {
        assert(type_.ndim() >= 2);
        return at_f(r * type_.dims[1] + c);
    }

    // Fill with constant
    void fill(float val) {
        assert(type_.elem == ElemKind::FloatTy);
        size_t n = type_.num_elements();
        float* ptr = reinterpret_cast<float*>(data_.data());
        for (size_t i = 0; i < n; ++i) ptr[i] = val;
    }

    // Zero
    void zero() { std::fill(data_.begin(), data_.end(), 0); }

    const TypeRef& type() const { return type_; }
    uint8_t* raw() { return data_.data(); }
    const uint8_t* raw() const { return data_.data(); }
    size_t raw_size() const { return data_.size(); }

private:
    TypeRef type_;
    std::vector<uint8_t> data_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Node — Graph IR node (operation)
// ─────────────────────────────────────────────────────────────────────────────
enum class OpCode : uint16_t {
    Placeholder = 0,
    Constant    = 1,
    // Arithmetic
    Add         = 10,
    Sub         = 11,
    Mul         = 12,
    Div         = 13,
    // Matrix
    MatMul      = 20,
    Transpose   = 21,
    // Activation
    Relu        = 30,
    Sigmoid     = 31,
    Tanh        = 32,
    Softmax     = 33,
    // Reduction
    ReduceSum   = 40,
    ReduceMean  = 41,
    ReduceMax   = 42,
    // Shape
    Reshape     = 50,
    Concat      = 51,
    Slice       = 52,
    // Convolution
    Conv2D      = 60,
    MaxPool     = 61,
    AvgPool     = 62,
    // Normalization
    BatchNorm   = 70,
    LayerNorm   = 71,
    // Loss
    CrossEntropy = 80,
    MSELoss     = 81,
    // Custom
    Custom      = 255
};

inline const char* opcode_name(OpCode op) {
    switch (op) {
        case OpCode::Placeholder: return "Placeholder";
        case OpCode::Constant:    return "Constant";
        case OpCode::Add:         return "Add";
        case OpCode::Sub:         return "Sub";
        case OpCode::Mul:         return "Mul";
        case OpCode::Div:         return "Div";
        case OpCode::MatMul:      return "MatMul";
        case OpCode::Transpose:   return "Transpose";
        case OpCode::Relu:        return "Relu";
        case OpCode::Sigmoid:     return "Sigmoid";
        case OpCode::Tanh:        return "Tanh";
        case OpCode::Softmax:     return "Softmax";
        case OpCode::ReduceSum:   return "ReduceSum";
        case OpCode::ReduceMean:  return "ReduceMean";
        case OpCode::ReduceMax:   return "ReduceMax";
        case OpCode::Reshape:     return "Reshape";
        case OpCode::Concat:      return "Concat";
        case OpCode::Slice:       return "Slice";
        case OpCode::Conv2D:      return "Conv2D";
        case OpCode::MaxPool:     return "MaxPool";
        case OpCode::AvgPool:     return "AvgPool";
        case OpCode::BatchNorm:   return "BatchNorm";
        case OpCode::LayerNorm:   return "LayerNorm";
        case OpCode::CrossEntropy:return "CrossEntropy";
        case OpCode::MSELoss:     return "MSELoss";
        case OpCode::Custom:      return "Custom";
    }
    return "Unknown";
}

struct Node {
    uint32_t id;
    std::string name;
    OpCode op;
    TypeRef result_type;
    std::vector<uint32_t> inputs;  // IDs of input nodes
    std::shared_ptr<Tensor> data;  // For Constant nodes
    bool dead;                     // Marked for dead code elimination

    Node() : id(0), op(OpCode::Placeholder), dead(false) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// Function — A graph of nodes
// ─────────────────────────────────────────────────────────────────────────────
class Function {
public:
    explicit Function(const std::string& name) : name_(name), next_id_(1) {}

    // Add placeholder (input)
    uint32_t add_placeholder(const std::string& name, const TypeRef& type) {
        Node n;
        n.id = next_id_++;
        n.name = name;
        n.op = OpCode::Placeholder;
        n.result_type = type;
        nodes_[n.id] = n;
        inputs_.push_back(n.id);
        return n.id;
    }

    // Add constant
    uint32_t add_constant(const std::string& name, std::shared_ptr<Tensor> data) {
        Node n;
        n.id = next_id_++;
        n.name = name;
        n.op = OpCode::Constant;
        n.result_type = data->type();
        n.data = data;
        nodes_[n.id] = n;
        return n.id;
    }

    // Add operation
    uint32_t add_op(const std::string& name, OpCode op,
                    const std::vector<uint32_t>& inputs,
                    const TypeRef& result_type) {
        Node n;
        n.id = next_id_++;
        n.name = name;
        n.op = op;
        n.inputs = inputs;
        n.result_type = result_type;
        nodes_[n.id] = n;
        return n.id;
    }

    // Set output node
    void set_output(uint32_t id) { output_ = id; }

    // Get node
    Node* node(uint32_t id) {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }
    const Node* node(uint32_t id) const {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    // Topological sort
    std::vector<uint32_t> topo_sort() const {
        std::unordered_map<uint32_t, int> in_degree;
        for (auto& kv : nodes_) {
            if (in_degree.find(kv.first) == in_degree.end())
                in_degree[kv.first] = 0;
            for (auto inp : kv.second.inputs) {
                in_degree[kv.first]++;
                (void)in_degree[inp]; // ensure exists
            }
        }
        // Kahn's algorithm
        std::queue<uint32_t> q;
        for (auto& kv : in_degree) {
            if (kv.second == 0) q.push(kv.first);
        }
        std::vector<uint32_t> order;
        while (!q.empty()) {
            uint32_t cur = q.front(); q.pop();
            order.push_back(cur);
            for (auto& kv : nodes_) {
                for (auto inp : kv.second.inputs) {
                    if (inp == cur) {
                        if (--in_degree[kv.first] == 0) {
                            q.push(kv.first);
                        }
                    }
                }
            }
        }
        return order;
    }

    const std::string& name() const { return name_; }
    uint32_t output() const { return output_; }
    const std::vector<uint32_t>& inputs() const { return inputs_; }
    size_t size() const { return nodes_.size(); }

    template<typename Fn>
    void foreach_node(Fn fn) { for (auto& kv : nodes_) fn(kv.second); }

    template<typename Fn>
    void foreach_node(Fn fn) const { for (auto& kv : nodes_) fn(kv.second); }

private:
    std::string name_;
    uint32_t next_id_;
    uint32_t output_ = 0;
    std::unordered_map<uint32_t, Node> nodes_;
    std::vector<uint32_t> inputs_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Optimization Passes
// ─────────────────────────────────────────────────────────────────────────────

// Dead Code Elimination
inline size_t dce_pass(Function& fn) {
    // Mark all nodes reachable from output
    std::unordered_set<uint32_t> reachable;
    std::queue<uint32_t> worklist;
    worklist.push(fn.output());
    while (!worklist.empty()) {
        uint32_t id = worklist.front(); worklist.pop();
        if (reachable.count(id)) continue;
        reachable.insert(id);
        auto* n = fn.node(id);
        if (n) {
            for (auto inp : n->inputs) worklist.push(inp);
        }
    }
    // Mark unreachable as dead
    size_t eliminated = 0;
    fn.foreach_node([&](Node& n) {
        if (!reachable.count(n.id)) {
            n.dead = true;
            ++eliminated;
        }
    });
    return eliminated;
}

// Constant Folding: fold Add/Sub/Mul/Div of two constants
inline size_t constant_fold_pass(Function& fn) {
    size_t folded = 0;
    fn.foreach_node([&](Node& n) {
        if (n.dead) return;
        if (n.inputs.size() != 2) return;
        if (n.op != OpCode::Add && n.op != OpCode::Sub &&
            n.op != OpCode::Mul && n.op != OpCode::Div) return;

        auto* lhs = fn.node(n.inputs[0]);
        auto* rhs = fn.node(n.inputs[1]);
        if (!lhs || !rhs) return;
        if (lhs->op != OpCode::Constant || rhs->op != OpCode::Constant) return;
        if (!lhs->data || !rhs->data) return;
        if (lhs->data->type().elem != ElemKind::FloatTy) return;

        size_t num = lhs->data->type().num_elements();
        if (num != rhs->data->type().num_elements()) return;

        auto result = std::make_shared<Tensor>(lhs->data->type());
        for (size_t i = 0; i < num; ++i) {
            float a = lhs->data->at_f(i);
            float b = rhs->data->at_f(i);
            float r = 0;
            switch (n.op) {
                case OpCode::Add: r = a + b; break;
                case OpCode::Sub: r = a - b; break;
                case OpCode::Mul: r = a * b; break;
                case OpCode::Div: r = (std::fabs(b) > 1e-12f) ? a / b : 0; break;
                default: break;
            }
            result->at_f(i) = r;
        }

        n.op = OpCode::Constant;
        n.data = result;
        n.inputs.clear();
        ++folded;
    });
    return folded;
}

// ─────────────────────────────────────────────────────────────────────────────
// Interpreter — Execute a function graph
// ─────────────────────────────────────────────────────────────────────────────
class Interpreter {
public:
    using TensorMap = std::unordered_map<uint32_t, std::shared_ptr<Tensor>>;

    // Execute function with given input tensors
    std::shared_ptr<Tensor> run(const Function& fn,
                                 const std::unordered_map<uint32_t, std::shared_ptr<Tensor>>& inputs) {
        TensorMap values;

        // Bind inputs
        for (auto& kv : inputs) values[kv.first] = kv.second;

        // Execute in topological order
        auto order = fn.topo_sort();
        for (auto id : order) {
            const Node* n = fn.node(id);
            if (!n || n->dead) continue;

            switch (n->op) {
                case OpCode::Placeholder:
                    // Already bound
                    break;
                case OpCode::Constant:
                    values[id] = n->data;
                    break;
                case OpCode::Add:
                    values[id] = binary_op(values, n, [](float a, float b) { return a + b; });
                    break;
                case OpCode::Sub:
                    values[id] = binary_op(values, n, [](float a, float b) { return a - b; });
                    break;
                case OpCode::Mul:
                    values[id] = binary_op(values, n, [](float a, float b) { return a * b; });
                    break;
                case OpCode::Div:
                    values[id] = binary_op(values, n, [](float a, float b) {
                        return (std::fabs(b) > 1e-12f) ? a / b : 0.0f;
                    });
                    break;
                case OpCode::Relu:
                    values[id] = unary_op(values, n, [](float a) { return a > 0 ? a : 0.0f; });
                    break;
                case OpCode::Sigmoid:
                    values[id] = unary_op(values, n, [](float a) { return 1.0f / (1.0f + std::exp(-a)); });
                    break;
                case OpCode::Tanh:
                    values[id] = unary_op(values, n, [](float a) { return std::tanh(a); });
                    break;
                case OpCode::MatMul:
                    values[id] = matmul(values, n);
                    break;
                case OpCode::Softmax:
                    values[id] = softmax(values, n);
                    break;
                default:
                    // Unsupported op: pass through first input
                    if (!n->inputs.empty() && values.count(n->inputs[0])) {
                        values[id] = values[n->inputs[0]];
                    }
                    break;
            }
        }

        auto it = values.find(fn.output());
        return (it != values.end()) ? it->second : nullptr;
    }

private:
    std::shared_ptr<Tensor> binary_op(TensorMap& vals, const Node* n,
                                       std::function<float(float, float)> op) {
        auto a = vals[n->inputs[0]];
        auto b = vals[n->inputs[1]];
        if (!a || !b) return nullptr;
        auto result = std::make_shared<Tensor>(a->type());
        size_t num = a->type().num_elements();
        for (size_t i = 0; i < num; ++i) {
            result->at_f(i) = op(a->at_f(i), b->at_f(i % b->type().num_elements()));
        }
        return result;
    }

    std::shared_ptr<Tensor> unary_op(TensorMap& vals, const Node* n,
                                      std::function<float(float)> op) {
        auto a = vals[n->inputs[0]];
        if (!a) return nullptr;
        auto result = std::make_shared<Tensor>(a->type());
        size_t num = a->type().num_elements();
        for (size_t i = 0; i < num; ++i) {
            result->at_f(i) = op(a->at_f(i));
        }
        return result;
    }

    std::shared_ptr<Tensor> matmul(TensorMap& vals, const Node* n) {
        auto a = vals[n->inputs[0]];
        auto b = vals[n->inputs[1]];
        if (!a || !b) return nullptr;
        assert(a->type().ndim() == 2 && b->type().ndim() == 2);
        size_t M = a->type().dims[0];
        size_t K = a->type().dims[1];
        size_t N = b->type().dims[1];
        assert(K == b->type().dims[0]);
        auto result = std::make_shared<Tensor>(ElemKind::FloatTy, std::vector<size_t>{M, N});
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float sum = 0;
                for (size_t k = 0; k < K; ++k) {
                    sum += a->at_f(i, k) * b->at_f(k, j);
                }
                result->at_f(i, j) = sum;
            }
        }
        return result;
    }

    std::shared_ptr<Tensor> softmax(TensorMap& vals, const Node* n) {
        auto a = vals[n->inputs[0]];
        if (!a) return nullptr;
        auto result = std::make_shared<Tensor>(a->type());
        size_t num = a->type().num_elements();
        // Find max for numerical stability
        float max_val = a->at_f(0);
        for (size_t i = 1; i < num; ++i) {
            if (a->at_f(i) > max_val) max_val = a->at_f(i);
        }
        float sum = 0;
        for (size_t i = 0; i < num; ++i) {
            result->at_f(i) = std::exp(a->at_f(i) - max_val);
            sum += result->at_f(i);
        }
        for (size_t i = 0; i < num; ++i) {
            result->at_f(i) /= sum;
        }
        return result;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Module — Top-level container for functions
// ─────────────────────────────────────────────────────────────────────────────
class Module {
public:
    Module() {}

    Function& create_function(const std::string& name) {
        functions_.emplace_back(new Function(name));
        return *functions_.back();
    }

    Function* get_function(const std::string& name) {
        for (auto& f : functions_) {
            if (f->name() == name) return f.get();
        }
        return nullptr;
    }

    size_t num_functions() const { return functions_.size(); }

    // Run optimization pipeline
    void optimize() {
        for (auto& f : functions_) {
            constant_fold_pass(*f);
            dce_pass(*f);
        }
    }

private:
    std::vector<std::unique_ptr<Function>> functions_;
};

}} // namespace cog::glow

#endif // COG_GLOW_HPP
