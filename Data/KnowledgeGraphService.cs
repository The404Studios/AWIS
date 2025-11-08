using System;
using System.Collections.Generic;
using System.Data;
using Microsoft.Data.Sqlite;
using System.Linq;
using System.Threading.Tasks;
using Dapper;
using AWIS.Core;

namespace AWIS.Data
{

/// <summary>
/// Persistent knowledge graph backed by SQLite with advanced inference
/// </summary>
public class KnowledgeGraphService : IKnowledgeStore, ISubsystem
{
    private readonly string _connectionString;
    private readonly IEventBus? _eventBus;
    private bool _isInitialized;

    public string Name => "KnowledgeGraphService";
    public bool IsInitialized => _isInitialized;

    public KnowledgeGraphService(string databasePath = "knowledge.db", IEventBus? eventBus = null)
    {
        _connectionString = $"Data Source={databasePath};Version=3;";
        _eventBus = eventBus;
    }

    public async Task InitializeAsync()
    {
        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync();

        // Create tables
        await connection.ExecuteAsync(@"
            CREATE TABLE IF NOT EXISTS Facts (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                Subject TEXT NOT NULL,
                Predicate TEXT NOT NULL,
                Object TEXT NOT NULL,
                Confidence REAL NOT NULL DEFAULT 1.0,
                CreatedAt TEXT NOT NULL,
                LastAccessedAt TEXT,
                AccessCount INTEGER DEFAULT 0,
                Metadata TEXT,
                UNIQUE(Subject, Predicate, Object)
            );

            CREATE INDEX IF NOT EXISTS idx_subject ON Facts(Subject);
            CREATE INDEX IF NOT EXISTS idx_predicate ON Facts(Predicate);
            CREATE INDEX IF NOT EXISTS idx_object ON Facts(Object);
            CREATE INDEX IF NOT EXISTS idx_confidence ON Facts(Confidence);

            CREATE TABLE IF NOT EXISTS InferenceRules (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL UNIQUE,
                Pattern TEXT NOT NULL,
                Conclusion TEXT NOT NULL,
                Confidence REAL NOT NULL DEFAULT 1.0,
                Enabled INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS ConceptHierarchy (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                Child TEXT NOT NULL,
                Parent TEXT NOT NULL,
                Distance INTEGER DEFAULT 1,
                UNIQUE(Child, Parent)
            );

            CREATE INDEX IF NOT EXISTS idx_child ON ConceptHierarchy(Child);
            CREATE INDEX IF NOT EXISTS idx_parent ON ConceptHierarchy(Parent);
        ");

        // Insert default inference rules
        await connection.ExecuteAsync(@"
            INSERT OR IGNORE INTO InferenceRules (Name, Pattern, Conclusion, Confidence) VALUES
            ('Transitivity', 'IsA', 'IsA', 0.9),
            ('PartOfTransitivity', 'PartOf', 'PartOf', 0.85),
            ('PropertyInheritance', 'IsA,HasProperty', 'HasProperty', 0.8),
            ('CauseTransitivity', 'Causes', 'Causes', 0.75);
        ");

        _isInitialized = true;
    }

    public async Task ShutdownAsync()
    {
        _isInitialized = false;
        await Task.CompletedTask;
    }

    public async Task<HealthStatus> GetHealthAsync()
    {
        try
        {
            using var connection = new SqliteConnection(_connectionString);
            var count = await connection.ExecuteScalarAsync<int>("SELECT COUNT(*) FROM Facts");

            return new HealthStatus
            {
                IsHealthy = true,
                Status = "Operational",
                Metrics = new Dictionary<string, object>
                {
                    ["TotalFacts"] = count,
                    ["IsInitialized"] = _isInitialized
                }
            };
        }
        catch (Exception ex)
        {
            return new HealthStatus
            {
                IsHealthy = false,
                Status = $"Error: {ex.Message}"
            };
        }
    }

    public async Task AddFactAsync(string subject, string predicate, string obj, double confidence = 1.0)
    {
        using var connection = new SqliteConnection(_connectionString);

        await connection.ExecuteAsync(@"
            INSERT OR REPLACE INTO Facts (Subject, Predicate, Object, Confidence, CreatedAt, Metadata)
            VALUES (@Subject, @Predicate, @Object, @Confidence, @CreatedAt, @Metadata)",
            new
            {
                Subject = subject,
                Predicate = predicate,
                Object = obj,
                Confidence = confidence,
                CreatedAt = DateTime.UtcNow.ToString("O"),
                Metadata = "{}"
            });

        // Update concept hierarchy for IsA relationships
        if (predicate.Equals("IsA", StringComparison.OrdinalIgnoreCase))
        {
            await UpdateConceptHierarchyAsync(subject, obj);
        }

        // Publish event
        if (_eventBus != null)
        {
            await _eventBus.PublishAsync(new KnowledgeLearnedEvent
            {
                Subject = subject,
                Predicate = predicate,
                Object = obj,
                Confidence = confidence
            });
        }
    }

    public async Task<IEnumerable<KnowledgeFact>> QueryAsync(string subject, string? predicate = null)
    {
        using var connection = new SqliteConnection(_connectionString);

        var sql = "SELECT * FROM Facts WHERE Subject = @Subject";
        if (predicate != null)
        {
            sql += " AND Predicate = @Predicate";
        }
        sql += " ORDER BY Confidence DESC";

        var results = await connection.QueryAsync<FactDto>(sql, new { Subject = subject, Predicate = predicate });

        // Update access statistics
        var ids = results.Select(r => r.Id).ToArray();
        if (ids.Any())
        {
            await connection.ExecuteAsync(@"
                UPDATE Facts
                SET AccessCount = AccessCount + 1, LastAccessedAt = @Now
                WHERE Id IN @Ids",
                new { Now = DateTime.UtcNow.ToString("O"), Ids = ids });
        }

        return results.Select(dto => new KnowledgeFact
        {
            Subject = dto.Subject,
            Predicate = dto.Predicate,
            Object = dto.Object,
            Confidence = dto.Confidence,
            CreatedAt = DateTime.Parse(dto.CreatedAt)
        });
    }

    public async Task<IEnumerable<KnowledgeFact>> InferAsync(string subject, int depth = 2)
    {
        var allFacts = new List<KnowledgeFact>();
        var visited = new HashSet<string>();
        var queue = new Queue<(string Subject, int CurrentDepth)>();

        queue.Enqueue((subject, 0));
        visited.Add(subject);

        using var connection = new SqliteConnection(_connectionString);

        while (queue.Count > 0)
        {
            var (currentSubject, currentDepth) = queue.Dequeue();

            if (currentDepth >= depth)
                continue;

            // Get direct facts
            var directFacts = await QueryAsync(currentSubject);
            allFacts.AddRange(directFacts);

            // Apply inference rules
            var inferredFacts = await ApplyInferenceRulesAsync(connection, directFacts.ToList());
            allFacts.AddRange(inferredFacts);

            // Add related concepts to queue
            foreach (var fact in directFacts)
            {
                if (!visited.Contains(fact.Object))
                {
                    visited.Add(fact.Object);
                    queue.Enqueue((fact.Object, currentDepth + 1));
                }
            }
        }

        return allFacts.Distinct(new KnowledgeFactComparer());
    }

    public async Task<double> GetConfidenceAsync(string subject, string predicate, string obj)
    {
        using var connection = new SqliteConnection(_connectionString);

        var result = await connection.QueryFirstOrDefaultAsync<double?>(@"
            SELECT Confidence FROM Facts
            WHERE Subject = @Subject AND Predicate = @Predicate AND Object = @Object",
            new { Subject = subject, Predicate = predicate, Object = obj });

        return result ?? 0.0;
    }

    /// <summary>
    /// Apply inference rules to derive new facts
    /// </summary>
    private async Task<List<KnowledgeFact>> ApplyInferenceRulesAsync(IDbConnection connection, List<KnowledgeFact> facts)
    {
        var inferred = new List<KnowledgeFact>();

        // Transitivity: If A -> B and B -> C, then A -> C
        var grouped = facts.GroupBy(f => f.Predicate);
        foreach (var group in grouped)
        {
            var predicate = group.Key;
            var factList = group.ToList();

            for (int i = 0; i < factList.Count; i++)
            {
                for (int j = 0; j < factList.Count; j++)
                {
                    if (i != j && factList[i].Object == factList[j].Subject)
                    {
                        var newConfidence = Math.Min(factList[i].Confidence, factList[j].Confidence) * 0.9;

                        inferred.Add(new KnowledgeFact
                        {
                            Subject = factList[i].Subject,
                            Predicate = predicate,
                            Object = factList[j].Object,
                            Confidence = newConfidence,
                            CreatedAt = DateTime.UtcNow,
                            Metadata = new Dictionary<string, object>
                            {
                                ["Inferred"] = true,
                                ["Rule"] = "Transitivity"
                            }
                        });
                    }
                }
            }
        }

        // Property inheritance: If A IsA B and B HasProperty P, then A HasProperty P
        var isAFacts = facts.Where(f => f.Predicate == "IsA").ToList();
        var propertyFacts = facts.Where(f => f.Predicate == "HasProperty").ToList();

        foreach (var isA in isAFacts)
        {
            var inheritedProperties = await connection.QueryAsync<FactDto>(@"
                SELECT * FROM Facts
                WHERE Subject = @Parent AND Predicate = 'HasProperty'",
                new { Parent = isA.Object });

            foreach (var prop in inheritedProperties)
            {
                inferred.Add(new KnowledgeFact
                {
                    Subject = isA.Subject,
                    Predicate = "HasProperty",
                    Object = prop.Object,
                    Confidence = Math.Min(isA.Confidence, prop.Confidence) * 0.8,
                    CreatedAt = DateTime.UtcNow,
                    Metadata = new Dictionary<string, object>
                    {
                        ["Inferred"] = true,
                        ["Rule"] = "PropertyInheritance"
                    }
                });
            }
        }

        return inferred;
    }

    /// <summary>
    /// Update concept hierarchy for efficient ancestor/descendant queries
    /// </summary>
    private async Task UpdateConceptHierarchyAsync(string child, string parent)
    {
        using var connection = new SqliteConnection(_connectionString);

        // Add direct relationship
        await connection.ExecuteAsync(@"
            INSERT OR IGNORE INTO ConceptHierarchy (Child, Parent, Distance)
            VALUES (@Child, @Parent, 1)",
            new { Child = child, Parent = parent });

        // Add transitive relationships
        var ancestors = await connection.QueryAsync<(string Parent, int Distance)>(@"
            SELECT Parent, Distance FROM ConceptHierarchy WHERE Child = @Parent",
            new { Parent = parent });

        foreach (var (ancestor, distance) in ancestors)
        {
            await connection.ExecuteAsync(@"
                INSERT OR IGNORE INTO ConceptHierarchy (Child, Parent, Distance)
                VALUES (@Child, @Ancestor, @Distance)",
                new { Child = child, Ancestor = ancestor, Distance = distance + 1 });
        }
    }

    /// <summary>
    /// Get all ancestors of a concept
    /// </summary>
    public async Task<IEnumerable<string>> GetAncestorsAsync(string concept)
    {
        using var connection = new SqliteConnection(_connectionString);

        var ancestors = await connection.QueryAsync<string>(@"
            SELECT Parent FROM ConceptHierarchy
            WHERE Child = @Concept
            ORDER BY Distance",
            new { Concept = concept });

        return ancestors;
    }

    /// <summary>
    /// Get all descendants of a concept
    /// </summary>
    public async Task<IEnumerable<string>> GetDescendantsAsync(string concept)
    {
        using var connection = new SqliteConnection(_connectionString);

        var descendants = await connection.QueryAsync<string>(@"
            SELECT Child FROM ConceptHierarchy
            WHERE Parent = @Concept
            ORDER BY Distance",
            new { Concept = concept });

        return descendants;
    }

    /// <summary>
    /// Find shortest path between two concepts
    /// </summary>
    public async Task<IEnumerable<KnowledgeFact>> FindPathAsync(string start, string end)
    {
        using var connection = new SqliteConnection(_connectionString);

        var visited = new HashSet<string>();
        var queue = new Queue<(string Node, List<KnowledgeFact> Path)>();
        queue.Enqueue((start, new List<KnowledgeFact>()));

        while (queue.Count > 0)
        {
            var (current, path) = queue.Dequeue();

            if (current == end)
                return path;

            if (visited.Contains(current))
                continue;

            visited.Add(current);

            var neighbors = await QueryAsync(current);
            foreach (var fact in neighbors)
            {
                if (!visited.Contains(fact.Object))
                {
                    var newPath = new List<KnowledgeFact>(path) { fact };
                    queue.Enqueue((fact.Object, newPath));
                }
            }
        }

        return Enumerable.Empty<KnowledgeFact>();
    }

    /// <summary>
    /// Get statistics about the knowledge graph
    /// </summary>
    public async Task<KnowledgeGraphStats> GetStatisticsAsync()
    {
        using var connection = new SqliteConnection(_connectionString);

        var stats = new KnowledgeGraphStats
        {
            TotalFacts = await connection.ExecuteScalarAsync<int>("SELECT COUNT(*) FROM Facts"),
            TotalSubjects = await connection.ExecuteScalarAsync<int>("SELECT COUNT(DISTINCT Subject) FROM Facts"),
            TotalPredicates = await connection.ExecuteScalarAsync<int>("SELECT COUNT(DISTINCT Predicate) FROM Facts"),
            TotalObjects = await connection.ExecuteScalarAsync<int>("SELECT COUNT(DISTINCT Object) FROM Facts"),
            AverageConfidence = await connection.ExecuteScalarAsync<double>("SELECT AVG(Confidence) FROM Facts")
        };

        var predicateDistribution = await connection.QueryAsync<(string Predicate, int Count)>(@"
            SELECT Predicate, COUNT(*) as Count
            FROM Facts
            GROUP BY Predicate
            ORDER BY Count DESC");

        stats.PredicateDistribution = predicateDistribution.ToDictionary(x => x.Predicate, x => x.Count);

        return stats;
    }

    /// <summary>
    /// Prune low-confidence facts
    /// </summary>
    public async Task PruneAsync(double minConfidence = 0.1)
    {
        using var connection = new SqliteConnection(_connectionString);

        var deleted = await connection.ExecuteAsync(@"
            DELETE FROM Facts WHERE Confidence < @MinConfidence",
            new { MinConfidence = minConfidence });

        Console.WriteLine($"[KnowledgeGraph] Pruned {deleted} low-confidence facts");
    }

    /// <summary>
    /// Export knowledge graph to JSON
    /// </summary>
    public async Task<string> ExportAsync()
    {
        using var connection = new SqliteConnection(_connectionString);

        var facts = await connection.QueryAsync<FactDto>("SELECT * FROM Facts");

        return System.Text.Json.JsonSerializer.Serialize(facts, new System.Text.Json.JsonSerializerOptions
        {
            WriteIndented = true
        });
    }

    /// <summary>
    /// Import knowledge graph from JSON
    /// </summary>
    public async Task ImportAsync(string json)
    {
        var facts = System.Text.Json.JsonSerializer.Deserialize<List<FactDto>>(json);

        if (facts == null)
            return;

        foreach (var fact in facts)
        {
            await AddFactAsync(fact.Subject, fact.Predicate, fact.Object, fact.Confidence);
        }
    }

    private class FactDto
    {
        public int Id { get; set; }
        public string Subject { get; set; } = string.Empty;
        public string Predicate { get; set; } = string.Empty;
        public string Object { get; set; } = string.Empty;
        public double Confidence { get; set; }
        public string CreatedAt { get; set; } = string.Empty;
    }

    private class KnowledgeFactComparer : IEqualityComparer<KnowledgeFact>
    {
        public bool Equals(KnowledgeFact? x, KnowledgeFact? y)
        {
            if (x == null || y == null) return false;
            return x.Subject == y.Subject && x.Predicate == y.Predicate && x.Object == y.Object;
        }

        public int GetHashCode(KnowledgeFact obj)
        {
            return HashCode.Combine(obj.Subject, obj.Predicate, obj.Object);
        }
    }
}

/// <summary>
/// Knowledge graph statistics
/// </summary>
public class KnowledgeGraphStats
{
    public int TotalFacts { get; set; }
    public int TotalSubjects { get; set; }
    public int TotalPredicates { get; set; }
    public int TotalObjects { get; set; }
    public double AverageConfidence { get; set; }
    public Dictionary<string, int> PredicateDistribution { get; set; } = new();
}
}
