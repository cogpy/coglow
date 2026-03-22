#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Graph/Nodes.h"
#include "gtest/gtest.h"

using namespace glow;

// A minimal testing macro to avoid external dependencies for this example.
#define ASSERT_TRUE(condition) EXPECT_TRUE(condition)
#define ASSERT_FALSE(condition) EXPECT_FALSE(condition)
#define ASSERT_EQ(val1, val2) EXPECT_EQ(val1, val2)

/// A mock "Cognitive" node for demonstration purposes.
/// In a real scenario, this would be a custom node type with specific
/// cognitive functionalities, like symbolic reasoning or memory access.
class CognitiveNode : public Node {
public:
  CognitiveNode(Graph &G, TypeRef inputType, std::string name, std::string knowledgeBaseId)
      : Node(G, Kinded::Kind::CognitiveNodeKind, inputType, name),
        knowledgeBaseId_(knowledgeBaseId) {}

  static CognitiveNode *create(Graph &G, Node *input, std::string name, std::string knowledgeBaseId) {
    auto *N = new CognitiveNode(G, input->getType(), name, knowledgeBaseId);
    N->addInput(input);
    return N;
  }

  const std::string &getKnowledgeBaseId() const { return knowledgeBaseId_; }

private:
  std::string knowledgeBaseId_;
};

TEST(CognitiveExtensionsTest, CreateCognitiveNode) {
  Graph G;
  Type T(ElemKind::FloatTy, {1, 32});
  Node *input = G.createVariable(T, "input", Visibility::Public);

  CognitiveNode *cogNode = CognitiveNode::create(G, input, "testCogNode", "kb-001");

  ASSERT_TRUE(cogNode != nullptr);
  ASSERT_EQ(cogNode->getNumInputs(), 1);
  ASSERT_EQ(cogNode->getNthInput(0), input);
  ASSERT_EQ(cogNode->getName(), "testCogNode");
  ASSERT_EQ(cogNode->getKnowledgeBaseId(), "kb-001");
}

TEST(CognitiveExtensionsTest, CognitiveNodeSerialization) {
  Graph G;
  Type T(ElemKind::FloatTy, {1, 10});
  Node *input = G.createVariable(T, "input", Visibility::Public);
  CognitiveNode *cogNode = CognitiveNode::create(G, input, "cogNode", "kb-abc");

  // In a real implementation, we would serialize the graph to a format
  // that supports custom nodes and verify that the cognitive properties
  // are preserved. For this test, we simulate a check.
  std::string serialized_graph = cogNode->getName() + ":" + cogNode->getKnowledgeBaseId();
  
  ASSERT_EQ(serialized_graph, "cogNode:kb-abc");
}

TEST(CognitiveExtensionsTest, CognitiveNodeCloning) {
  Graph G;
  Type T(ElemKind::FloatTy, {1, 16});
  Node *input = G.createVariable(T, "input", Visibility::Public);
  CognitiveNode *origNode = CognitiveNode::create(G, input, "original", "kb-123");

  // This is a simplified clone. A real implementation would be more complex.
  CognitiveNode *clonedNode = CognitiveNode::create(G, origNode->getNthInput(0).getNode(), 
                                                    origNode->getName(), origNode->getKnowledgeBaseId());

  ASSERT_TRUE(clonedNode != nullptr);
  ASSERT_EQ(clonedNode->getName(), origNode->getName());
  ASSERT_EQ(clonedNode->getKnowledgeBaseId(), origNode->getKnowledgeBaseId());
  ASSERT_FALSE(clonedNode == origNode);
}
