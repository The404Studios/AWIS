using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.Core
{
    /// <summary>
    /// Types of relationships between knowledge nodes
    /// </summary>
    public enum RelationType
    {
        IsA,
        PartOf,
        HasProperty,
        Causes,
        Requires,
        SimilarTo,
        OppositeOf,
        Friend,
        Enables,
        Prevents,
        CreatedBy,
        UsedFor,
        LocatedAt,
        TemporallyBefore,
        TemporallyAfter
    }

    /// <summary>
    /// A node in the knowledge graph
    /// </summary>
    public class KnowledgeNode
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public string Name { get; set; } = string.Empty;
        public string Type { get; set; } = "Concept";
        public Dictionary<string, object> Properties { get; set; } = new();
        public DateTime Created { get; set; } = DateTime.UtcNow;
        public DateTime LastModified { get; set; } = DateTime.UtcNow;
        public double Confidence { get; set; } = 1.0;
        public List<string> Tags { get; set; } = new();

        public KnowledgeNode(string name, string type = "Concept")
        {
            Name = name;
            Type = type;
        }

        public void SetProperty(string key, object value)
        {
            Properties[key] = value;
            LastModified = DateTime.UtcNow;
        }

        public T? GetProperty<T>(string key)
        {
            if (Properties.TryGetValue(key, out var value) && value is T typedValue)
            {
                return typedValue;
            }
            return default;
        }
    }

    /// <summary>
    /// A relationship between two knowledge nodes
    /// </summary>
    public class KnowledgeRelation
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public Guid FromNodeId { get; set; }
        public Guid ToNodeId { get; set; }
        public RelationType Type { get; set; }
        public double Strength { get; set; } = 1.0;
        public DateTime Created { get; set; } = DateTime.UtcNow;
        public Dictionary<string, object> Metadata { get; set; } = new();

        public KnowledgeRelation(Guid fromNode, Guid toNode, RelationType type, double strength = 1.0)
        {
            FromNodeId = fromNode;
            ToNodeId = toNode;
            Type = type;
            Strength = strength;
        }
    }

    /// <summary>
    /// Hierarchical knowledge base with graph structure
    /// </summary>
    public class HierarchicalKnowledgeBase
    {
        private readonly Dictionary<Guid, KnowledgeNode> nodes = new();
        private readonly Dictionary<string, Guid> nameIndex = new();
        private readonly List<KnowledgeRelation> relations = new();
        private readonly Dictionary<Guid, List<KnowledgeRelation>> outgoingRelations = new();
        private readonly Dictionary<Guid, List<KnowledgeRelation>> incomingRelations = new();

        public KnowledgeNode AddNode(string name, string type = "Concept")
        {
            lock (nodes)
            {
                // Check if node already exists
                if (nameIndex.TryGetValue(name.ToLower(), out var existingId))
                {
                    return nodes[existingId];
                }

                var node = new KnowledgeNode(name, type);
                nodes[node.Id] = node;
                nameIndex[name.ToLower()] = node.Id;
                outgoingRelations[node.Id] = new List<KnowledgeRelation>();
                incomingRelations[node.Id] = new List<KnowledgeRelation>();

                return node;
            }
        }

        public KnowledgeRelation AddRelation(Guid fromNodeId, Guid toNodeId, RelationType type, double strength = 1.0)
        {
            lock (nodes)
            {
                if (!nodes.ContainsKey(fromNodeId) || !nodes.ContainsKey(toNodeId))
                {
                    throw new ArgumentException("Both nodes must exist");
                }

                // Check for existing relation
                var existing = outgoingRelations[fromNodeId]
                    .FirstOrDefault(r => r.ToNodeId == toNodeId && r.Type == type);

                if (existing != null)
                {
                    // Strengthen existing relation
                    existing.Strength = Math.Min(1.0, existing.Strength + strength * 0.1);
                    return existing;
                }

                var relation = new KnowledgeRelation(fromNodeId, toNodeId, type, strength);
                relations.Add(relation);
                outgoingRelations[fromNodeId].Add(relation);
                incomingRelations[toNodeId].Add(relation);

                return relation;
            }
        }

        public KnowledgeRelation AddRelation(string fromName, string toName, RelationType type, double strength = 1.0)
        {
            var fromNode = FindNode(fromName) ?? AddNode(fromName);
            var toNode = FindNode(toName) ?? AddNode(toName);
            return AddRelation(fromNode.Id, toNode.Id, type, strength);
        }

        public KnowledgeNode? FindNode(string name)
        {
            lock (nodes)
            {
                if (nameIndex.TryGetValue(name.ToLower(), out var id))
                {
                    return nodes[id];
                }
                return null;
            }
        }

        public List<KnowledgeNode> SearchNodes(string query)
        {
            lock (nodes)
            {
                query = query.ToLower();
                return nodes.Values
                    .Where(n => n.Name.ToLower().Contains(query) ||
                               n.Type.ToLower().Contains(query) ||
                               n.Tags.Any(t => t.ToLower().Contains(query)))
                    .OrderByDescending(n => n.Confidence)
                    .ToList();
            }
        }

        public List<(KnowledgeNode node, RelationType relationType, double strength)> GetRelatedNodes(Guid nodeId)
        {
            lock (nodes)
            {
                if (!outgoingRelations.ContainsKey(nodeId))
                {
                    return new List<(KnowledgeNode, RelationType, double)>();
                }

                return outgoingRelations[nodeId]
                    .Where(r => nodes.ContainsKey(r.ToNodeId))
                    .Select(r => (nodes[r.ToNodeId], r.Type, r.Strength))
                    .OrderByDescending(t => t.Item3)
                    .ToList();
            }
        }

        public List<(KnowledgeNode node, RelationType relationType, double strength)> GetIncomingNodes(Guid nodeId)
        {
            lock (nodes)
            {
                if (!incomingRelations.ContainsKey(nodeId))
                {
                    return new List<(KnowledgeNode, RelationType, double)>();
                }

                return incomingRelations[nodeId]
                    .Where(r => nodes.ContainsKey(r.FromNodeId))
                    .Select(r => (nodes[r.FromNodeId], r.Type, r.Strength))
                    .OrderByDescending(t => t.Item3)
                    .ToList();
            }
        }

        public List<KnowledgeNode> InferRelatedConcepts(string conceptName, int depth = 2)
        {
            var node = FindNode(conceptName);
            if (node == null)
            {
                return new List<KnowledgeNode>();
            }

            var visited = new HashSet<Guid>();
            var queue = new Queue<(Guid id, int currentDepth)>();
            var results = new List<KnowledgeNode>();

            queue.Enqueue((node.Id, 0));
            visited.Add(node.Id);

            while (queue.Count > 0)
            {
                var (currentId, currentDepth) = queue.Dequeue();
                if (currentDepth >= depth) continue;

                var related = GetRelatedNodes(currentId);
                foreach (var (relatedNode, _, _) in related)
                {
                    if (!visited.Contains(relatedNode.Id))
                    {
                        visited.Add(relatedNode.Id);
                        results.Add(relatedNode);
                        queue.Enqueue((relatedNode.Id, currentDepth + 1));
                    }
                }
            }

            return results.OrderByDescending(n => n.Confidence).ToList();
        }

        public KnowledgeStatistics GetStatistics()
        {
            lock (nodes)
            {
                var typeCounts = nodes.Values
                    .GroupBy(n => n.Type)
                    .ToDictionary(g => g.Key, g => g.Count());

                var relationTypeCounts = relations
                    .GroupBy(r => r.Type)
                    .ToDictionary(g => g.Key, g => g.Count());

                return new KnowledgeStatistics
                {
                    TotalNodes = nodes.Count,
                    TotalRelations = relations.Count,
                    NodesByType = typeCounts,
                    RelationsByType = relationTypeCounts,
                    AverageConnections = nodes.Count > 0 ?
                        outgoingRelations.Values.Average(list => list.Count) : 0
                };
            }
        }
    }

    public class KnowledgeStatistics
    {
        public int TotalNodes { get; set; }
        public int TotalRelations { get; set; }
        public Dictionary<string, int> NodesByType { get; set; } = new();
        public Dictionary<RelationType, int> RelationsByType { get; set; } = new();
        public double AverageConnections { get; set; }
    }
}
