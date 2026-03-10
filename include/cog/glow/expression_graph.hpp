// cog/glow/expression_graph.hpp — Expression Pipeline Graph IR
// Compiles the meta-echo-dna expression pipeline into an optimizable graph
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
//
// The expression pipeline is represented as a dataflow graph:
//   HormoneInput → EndocrineMap → AU_Blend ← CognitivePreset
//                                    ↓
//                              ChaoticNoise → AestheticBias → MorphOutput
//
// This graph IR enables optimization passes (constant folding, dead AU
// elimination, subexpression sharing) before interpretation or code generation.
//
#ifndef COG_GLOW_EXPRESSION_GRAPH_HPP
#define COG_GLOW_EXPRESSION_GRAPH_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <sstream>

namespace cog { namespace glow {

// ─────────────────────────────────────────────────────────────────────────────
// Node Types for the Expression Graph
// ─────────────────────────────────────────────────────────────────────────────
enum class ExprNodeKind : uint8_t {
    INPUT_HORMONE,      // Input: hormone concentration [0,1]
    INPUT_COGNITIVE,    // Input: cognitive mode / load / valence / arousal
    ENDOCRINE_MAP,      // Hormone → AU mapping (matrix multiply)
    COGNITIVE_PRESET,   // Cognitive mode → AU preset
    AU_BLEND,           // Sum multiple AU sources, clamp [0,1]
    CHAOTIC_NOISE,      // Lorenz attractor micro-expression injection
    AESTHETIC_BIAS,     // SuperHotGirl aesthetic parameter application
    MORPH_OUTPUT,       // Final MetaHuman CTRL_ morph target
    MATERIAL_OUTPUT,    // Dynamic material parameter output
    CONSTANT,           // Constant value
    MULTIPLY,           // Element-wise multiply
    ADD,                // Element-wise add
    CLAMP,              // Clamp to range
    EMA_SMOOTH,         // Exponential moving average smoothing
};

inline const char* expr_node_kind_name(ExprNodeKind k) {
    static const char* names[] = {
        "InputHormone", "InputCognitive", "EndocrineMap", "CognitivePreset",
        "AUBlend", "ChaoticNoise", "AestheticBias", "MorphOutput",
        "MaterialOutput", "Constant", "Multiply", "Add", "Clamp", "EMASmooth"
    };
    return names[static_cast<int>(k)];
}

// ─────────────────────────────────────────────────────────────────────────────
// ExprNode — A node in the expression pipeline graph
// ─────────────────────────────────────────────────────────────────────────────
struct ExprNode {
    uint32_t id;
    ExprNodeKind kind;
    std::string name;
    std::vector<uint32_t> inputs;  // IDs of input nodes
    std::vector<float> params;     // Node-specific parameters
    std::vector<float> output;     // Cached output values

    ExprNode() : id(0), kind(ExprNodeKind::CONSTANT) {}
    ExprNode(uint32_t id_, ExprNodeKind k, const std::string& n)
        : id(id_), kind(k), name(n) {}
};

// ─────────────────────────────────────────────────────────────────────────────
// ExpressionGraph — The complete expression pipeline as a graph
// ─────────────────────────────────────────────────────────────────────────────
class ExpressionGraph {
public:
    ExpressionGraph() : next_id_(1) {}

    // Add a node to the graph, return its ID
    uint32_t add_node(ExprNodeKind kind, const std::string& name,
                      const std::vector<uint32_t>& inputs = {},
                      const std::vector<float>& params = {}) {
        uint32_t id = next_id_++;
        ExprNode node(id, kind, name);
        node.inputs = inputs;
        node.params = params;
        nodes_[id] = node;
        return id;
    }

    // Get a node by ID
    const ExprNode* get_node(uint32_t id) const {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    ExprNode* get_node_mut(uint32_t id) {
        auto it = nodes_.find(id);
        return (it != nodes_.end()) ? &it->second : nullptr;
    }

    // Build the standard meta-echo-dna expression pipeline graph
    void build_standard_pipeline() {
        // Inputs
        uint32_t h_in = add_node(ExprNodeKind::INPUT_HORMONE, "HormoneInput");
        uint32_t c_in = add_node(ExprNodeKind::INPUT_COGNITIVE, "CognitiveInput");

        // Endocrine mapping
        uint32_t endo = add_node(ExprNodeKind::ENDOCRINE_MAP, "EndocrineMap", {h_in});

        // Cognitive preset
        uint32_t cog = add_node(ExprNodeKind::COGNITIVE_PRESET, "CogPreset", {c_in});

        // AU blend (sum endocrine + cognitive)
        uint32_t blend = add_node(ExprNodeKind::AU_BLEND, "AUBlend", {endo, cog});

        // Chaotic noise injection
        uint32_t chaos = add_node(ExprNodeKind::CHAOTIC_NOISE, "LorenzNoise", {blend},
                                  {0.15f}); // chaos_intensity

        // Aesthetic bias
        uint32_t aesthetic = add_node(ExprNodeKind::AESTHETIC_BIAS, "AestheticBias", {chaos},
                                      {0.7f, 0.6f, 0.5f}); // confidence, charisma, sparkle

        // Smoothing
        uint32_t smooth = add_node(ExprNodeKind::EMA_SMOOTH, "Smoothing", {aesthetic},
                                   {0.3f}); // smoothing factor

        // Morph target output
        add_node(ExprNodeKind::MORPH_OUTPUT, "MorphTargets", {smooth});

        // Material output (from aesthetic + hormone input)
        add_node(ExprNodeKind::MATERIAL_OUTPUT, "MaterialParams", {aesthetic, h_in});
    }

    // Export graph as DOT format for visualization
    std::string to_dot() const {
        std::ostringstream ss;
        ss << "digraph ExpressionPipeline {\n";
        ss << "  rankdir=TB;\n";
        ss << "  node [shape=record, fontname=\"Courier\"];\n";
        for (const auto& kv : nodes_) {
            const auto& n = kv.second;
            ss << "  n" << n.id << " [label=\"{" << expr_node_kind_name(n.kind)
               << "|" << n.name << "}\"];\n";
            for (auto inp : n.inputs) {
                ss << "  n" << inp << " -> n" << n.id << ";\n";
            }
        }
        ss << "}\n";
        return ss.str();
    }

    // Count nodes by kind
    size_t count_nodes(ExprNodeKind kind) const {
        size_t count = 0;
        for (const auto& kv : nodes_) {
            if (kv.second.kind == kind) ++count;
        }
        return count;
    }

    size_t total_nodes() const { return nodes_.size(); }

    // Topological sort for evaluation order
    std::vector<uint32_t> topological_order() const {
        std::vector<uint32_t> order;
        std::unordered_map<uint32_t, bool> visited;
        for (const auto& kv : nodes_) {
            if (!visited[kv.first]) {
                topo_visit(kv.first, visited, order);
            }
        }
        return order;
    }

private:
    uint32_t next_id_;
    std::unordered_map<uint32_t, ExprNode> nodes_;

    void topo_visit(uint32_t id,
                    std::unordered_map<uint32_t, bool>& visited,
                    std::vector<uint32_t>& order) const {
        if (visited[id]) return;
        visited[id] = true;
        auto it = nodes_.find(id);
        if (it != nodes_.end()) {
            for (auto inp : it->second.inputs) {
                topo_visit(inp, visited, order);
            }
        }
        order.push_back(id);
    }
};

}} // namespace cog::glow

#endif // COG_GLOW_EXPRESSION_GRAPH_HPP
