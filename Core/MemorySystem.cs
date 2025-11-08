using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.Core
{
    /// <summary>
    /// Types of memory
    /// </summary>
    public enum MemoryType
    {
        ShortTerm,
        LongTerm,
        Working,
        Episodic,
        Semantic,
        Procedural
    }

    /// <summary>
    /// A memory item with content and metadata
    /// </summary>
    public class MemoryItem
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public MemoryType Type { get; set; }
        public string Content { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public int AccessCount { get; set; } = 0;
        public DateTime LastAccessed { get; set; } = DateTime.UtcNow;
        public double Importance { get; set; } = 0.5;
        public Dictionary<string, object> Metadata { get; set; } = new();
        public List<Guid> Associations { get; set; } = new();

        public void RecordAccess()
        {
            AccessCount++;
            LastAccessed = DateTime.UtcNow;
            // Accessing a memory strengthens it
            Importance = Math.Min(1.0, Importance + 0.05);
        }

        public double GetStrength()
        {
            // Memory strength based on recency, frequency, and importance
            var recency = 1.0 / (1.0 + (DateTime.UtcNow - LastAccessed).TotalHours / 24.0);
            var frequency = Math.Log(1.0 + AccessCount) / 10.0;
            return (recency * 0.4 + frequency * 0.3 + Importance * 0.3);
        }

        public bool ShouldRetain(TimeSpan retentionPeriod)
        {
            if (Type == MemoryType.LongTerm || Type == MemoryType.Semantic || Type == MemoryType.Procedural)
            {
                return true; // Always retain these types
            }

            var age = DateTime.UtcNow - Timestamp;
            return age < retentionPeriod || GetStrength() > 0.6;
        }
    }

    /// <summary>
    /// Hierarchical memory architecture with different memory types
    /// </summary>
    public class MemoryArchitecture : IMemorySystem
    {
        public string Name => "MemoryArchitecture";
        public bool IsInitialized { get; private set; }

        private readonly Dictionary<MemoryType, List<MemoryItem>> memoryStores = new();
        private readonly Dictionary<Guid, MemoryItem> memoryIndex = new();
        private readonly int shortTermCapacity = 100;
        private readonly int workingMemoryCapacity = 20;
        private readonly int longTermCapacity = 10000;

        public MemoryArchitecture()
        {
            foreach (MemoryType type in Enum.GetValues(typeof(MemoryType)))
            {
                memoryStores[type] = new List<MemoryItem>();
            }
        }

        public void Store(string content, MemoryType type = MemoryType.ShortTerm, double importance = 0.5)
        {
            var memory = new MemoryItem
            {
                Content = content,
                Type = type,
                Importance = importance
            };

            lock (memoryStores)
            {
                memoryStores[type].Add(memory);
                memoryIndex[memory.Id] = memory;

                // Manage capacity
                ManageCapacity(type);
            }
        }

        public MemoryItem? Recall(string query, MemoryType? type = null)
        {
            lock (memoryStores)
            {
                var searchStores = type.HasValue ?
                    new[] { memoryStores[type.Value] } :
                    memoryStores.Values;

                MemoryItem? bestMatch = null;
                double bestScore = 0;

                foreach (var store in searchStores)
                {
                    foreach (var memory in store)
                    {
                        var score = ComputeSimilarity(query, memory.Content) * memory.GetStrength();
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestMatch = memory;
                        }
                    }
                }

                if (bestMatch != null)
                {
                    bestMatch.RecordAccess();
                }

                return bestMatch;
            }
        }

        public List<MemoryItem> RecallMultiple(string query, int limit = 10, MemoryType? type = null)
        {
            lock (memoryStores)
            {
                var searchStores = type.HasValue ?
                    new[] { memoryStores[type.Value] } :
                    memoryStores.Values;

                var results = new List<(MemoryItem memory, double score)>();

                foreach (var store in searchStores)
                {
                    foreach (var memory in store)
                    {
                        var score = ComputeSimilarity(query, memory.Content) * memory.GetStrength();
                        if (score > 0.1)
                        {
                            results.Add((memory, score));
                        }
                    }
                }

                var topResults = results
                    .OrderByDescending(r => r.score)
                    .Take(limit)
                    .Select(r => r.memory)
                    .ToList();

                foreach (var memory in topResults)
                {
                    memory.RecordAccess();
                }

                return topResults;
            }
        }

        public void Consolidate()
        {
            // Move important short-term memories to long-term
            lock (memoryStores)
            {
                var shortTermMems = memoryStores[MemoryType.ShortTerm]
                    .Where(m => m.GetStrength() > 0.7 || m.AccessCount > 5)
                    .ToList();

                foreach (var memory in shortTermMems)
                {
                    memory.Type = MemoryType.LongTerm;
                    memoryStores[MemoryType.ShortTerm].Remove(memory);
                    memoryStores[MemoryType.LongTerm].Add(memory);
                }

                // Clean up old weak memories
                CleanUpMemories();
            }
        }

        private void ManageCapacity(MemoryType type)
        {
            var store = memoryStores[type];
            int capacity = type switch
            {
                MemoryType.ShortTerm => shortTermCapacity,
                MemoryType.Working => workingMemoryCapacity,
                MemoryType.LongTerm => longTermCapacity,
                _ => 1000
            };

            if (store.Count > capacity)
            {
                // Remove weakest memories
                var toRemove = store
                    .OrderBy(m => m.GetStrength())
                    .Take(store.Count - capacity)
                    .ToList();

                foreach (var memory in toRemove)
                {
                    store.Remove(memory);
                    memoryIndex.Remove(memory.Id);
                }
            }
        }

        private void CleanUpMemories()
        {
            foreach (var type in new[] { MemoryType.ShortTerm, MemoryType.Working, MemoryType.Episodic })
            {
                var retention = type switch
                {
                    MemoryType.Working => TimeSpan.FromHours(1),
                    MemoryType.ShortTerm => TimeSpan.FromDays(7),
                    MemoryType.Episodic => TimeSpan.FromDays(30),
                    _ => TimeSpan.FromDays(365)
                };

                var toRemove = memoryStores[type]
                    .Where(m => !m.ShouldRetain(retention))
                    .ToList();

                foreach (var memory in toRemove)
                {
                    memoryStores[type].Remove(memory);
                    memoryIndex.Remove(memory.Id);
                }
            }
        }

        private double ComputeSimilarity(string query, string content)
        {
            // Simple word overlap similarity
            var queryWords = query.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var contentWords = content.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);

            if (!queryWords.Any() || !contentWords.Any())
            {
                return 0;
            }

            var matches = queryWords.Count(qw => contentWords.Any(cw => cw.Contains(qw) || qw.Contains(cw)));
            return matches / (double)queryWords.Length;
        }

        public MemoryStatistics GetStatistics()
        {
            lock (memoryStores)
            {
                return new MemoryStatistics
                {
                    TotalMemories = memoryIndex.Count,
                    ShortTermCount = memoryStores[MemoryType.ShortTerm].Count,
                    LongTermCount = memoryStores[MemoryType.LongTerm].Count,
                    WorkingMemoryCount = memoryStores[MemoryType.Working].Count,
                    EpisodicCount = memoryStores[MemoryType.Episodic].Count,
                    SemanticCount = memoryStores[MemoryType.Semantic].Count,
                    ProceduralCount = memoryStores[MemoryType.Procedural].Count
                };
            }
        }

        public void AssociateMemories(Guid memory1, Guid memory2)
        {
            if (memoryIndex.TryGetValue(memory1, out var mem1) &&
                memoryIndex.TryGetValue(memory2, out var mem2))
            {
                if (!mem1.Associations.Contains(memory2))
                {
                    mem1.Associations.Add(memory2);
                }
                if (!mem2.Associations.Contains(memory1))
                {
                    mem2.Associations.Add(memory1);
                }
            }
        }

        public List<MemoryItem> GetAssociatedMemories(Guid memoryId)
        {
            if (memoryIndex.TryGetValue(memoryId, out var memory))
            {
                return memory.Associations
                    .Where(id => memoryIndex.ContainsKey(id))
                    .Select(id => memoryIndex[id])
                    .ToList();
            }
            return new List<MemoryItem>();
        }

        // IMemorySystem implementation
        public Task InitializeAsync()
        {
            IsInitialized = true;
            return Task.CompletedTask;
        }

        public Task ShutdownAsync()
        {
            IsInitialized = false;
            return Task.CompletedTask;
        }

        public Task<HealthStatus> GetHealthAsync()
        {
            var stats = GetStatistics();
            return Task.FromResult(new HealthStatus
            {
                IsHealthy = IsInitialized,
                Status = IsInitialized ? "Operational" : "Not Initialized",
                Metrics = new Dictionary<string, object>
                {
                    ["TotalMemories"] = stats.TotalMemories,
                    ["ShortTermCount"] = stats.ShortTermCount,
                    ["LongTermCount"] = stats.LongTermCount
                }
            });
        }

        public Task StoreAsync(string content, MemoryType type, double importance = 0.5)
        {
            Store(content, type, importance);
            return Task.CompletedTask;
        }

        public Task<Memory?> RecallAsync(string query, MemoryType? type = null)
        {
            var item = Recall(query, type);
            if (item == null) return Task.FromResult<Memory?>(null);

            return Task.FromResult<Memory?>(new Memory
            {
                Id = item.Id.ToString(),
                Content = item.Content,
                Type = item.Type,
                Importance = item.Importance,
                Strength = item.Strength,
                CreatedAt = item.Timestamp,
                LastAccessedAt = item.LastAccessed,
                AccessCount = item.AccessCount
            });
        }

        public Task<IEnumerable<Memory>> RecallMultipleAsync(string query, int limit = 10, MemoryType? type = null)
        {
            var items = RecallMultiple(query, type, limit);
            var memories = items.Select(item => new Memory
            {
                Id = item.Id.ToString(),
                Content = item.Content,
                Type = item.Type,
                Importance = item.Importance,
                Strength = item.Strength,
                CreatedAt = item.Timestamp,
                LastAccessedAt = item.LastAccessed,
                AccessCount = item.AccessCount
            });

            return Task.FromResult(memories);
        }

        public Task ConsolidateAsync()
        {
            Consolidate();
            return Task.CompletedTask;
        }
    }

    public class MemoryStatistics
    {
        public int TotalMemories { get; set; }
        public int ShortTermCount { get; set; }
        public int LongTermCount { get; set; }
        public int WorkingMemoryCount { get; set; }
        public int EpisodicCount { get; set; }
        public int SemanticCount { get; set; }
        public int ProceduralCount { get; set; }
    }
}
