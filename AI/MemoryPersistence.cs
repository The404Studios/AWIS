using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;

namespace AWIS.AI
{
    /// <summary>
    /// Persists learned knowledge, goals, and experiences to disk for continuous learning
    /// </summary>
    public class MemoryPersistence
    {
        private readonly string dataDirectory;
        private const string GOALS_FILE = "learned_goals.json";
        private const string PERSONALITY_FILE = "personality_state.json";
        private const string CONVERSATIONS_FILE = "conversation_history.json";
        private const string LLM_VOCAB_FILE = "llm_vocabulary.json";

        public MemoryPersistence(string? customDirectory = null)
        {
            dataDirectory = customDirectory ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "AWIS", "Memory");

            Directory.CreateDirectory(dataDirectory);
            Console.WriteLine($"[MEMORY] Persistence initialized: {dataDirectory}");
        }

        /// <summary>
        /// Save goal learning data
        /// </summary>
        public async Task SaveGoalDataAsync(Dictionary<string, GoalGradients> gradients, List<Goal> completedGoals)
        {
            try
            {
                var data = new
                {
                    Gradients = gradients,
                    CompletedGoals = completedGoals,
                    SavedAt = DateTime.UtcNow
                };

                var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
                var filePath = Path.Combine(dataDirectory, GOALS_FILE);
                await File.WriteAllTextAsync(filePath, json);

                Console.WriteLine($"[MEMORY] ✅ Saved {completedGoals.Count} goals and gradients");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MEMORY] ❌ Failed to save goals: {ex.Message}");
            }
        }

        /// <summary>
        /// Load goal learning data
        /// </summary>
        public async Task<(Dictionary<string, GoalGradients>? gradients, List<Goal>? goals)> LoadGoalDataAsync()
        {
            try
            {
                var filePath = Path.Combine(dataDirectory, GOALS_FILE);
                if (!File.Exists(filePath))
                {
                    Console.WriteLine("[MEMORY] No saved goal data found");
                    return (null, null);
                }

                var json = await File.ReadAllTextAsync(filePath);
                var data = JsonSerializer.Deserialize<GoalSaveData>(json);

                if (data != null)
                {
                    Console.WriteLine($"[MEMORY] ✅ Loaded {data.CompletedGoals?.Count ?? 0} goals from {data.SavedAt}");
                    return (data.Gradients, data.CompletedGoals);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MEMORY] ❌ Failed to load goals: {ex.Message}");
            }

            return (null, null);
        }

        /// <summary>
        /// Save personality evolution
        /// </summary>
        public async Task SavePersonalityAsync(PersonalitySystem personality)
        {
            try
            {
                var data = new
                {
                    Curiosity = personality.Curiosity,
                    Friendliness = personality.Friendliness,
                    Assertiveness = personality.Assertiveness,
                    Playfulness = personality.Playfulness,
                    Caution = personality.Caution,
                    Creativity = personality.Creativity,
                    Patience = personality.Patience,
                    Helpfulness = personality.Helpfulness,
                    SavedAt = DateTime.UtcNow
                };

                var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
                var filePath = Path.Combine(dataDirectory, PERSONALITY_FILE);
                await File.WriteAllTextAsync(filePath, json);

                Console.WriteLine($"[MEMORY] ✅ Saved personality state");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MEMORY] ❌ Failed to save personality: {ex.Message}");
            }
        }

        /// <summary>
        /// Load personality evolution
        /// </summary>
        public async Task<PersonalityData?> LoadPersonalityAsync()
        {
            try
            {
                var filePath = Path.Combine(dataDirectory, PERSONALITY_FILE);
                if (!File.Exists(filePath))
                {
                    Console.WriteLine("[MEMORY] No saved personality found");
                    return null;
                }

                var json = await File.ReadAllTextAsync(filePath);
                var data = JsonSerializer.Deserialize<PersonalityData>(json);

                if (data != null)
                {
                    Console.WriteLine($"[MEMORY] ✅ Loaded personality from {data.SavedAt}");
                    return data;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MEMORY] ❌ Failed to load personality: {ex.Message}");
            }

            return null;
        }

        /// <summary>
        /// Save conversation history for learning
        /// </summary>
        public async Task SaveConversationsAsync(List<string> conversations)
        {
            try
            {
                var data = new
                {
                    Conversations = conversations,
                    Count = conversations.Count,
                    SavedAt = DateTime.UtcNow
                };

                var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
                var filePath = Path.Combine(dataDirectory, CONVERSATIONS_FILE);
                await File.WriteAllTextAsync(filePath, json);

                Console.WriteLine($"[MEMORY] ✅ Saved {conversations.Count} conversation exchanges");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MEMORY] ❌ Failed to save conversations: {ex.Message}");
            }
        }

        /// <summary>
        /// Load conversation history
        /// </summary>
        public async Task<List<string>?> LoadConversationsAsync()
        {
            try
            {
                var filePath = Path.Combine(dataDirectory, CONVERSATIONS_FILE);
                if (!File.Exists(filePath))
                {
                    Console.WriteLine("[MEMORY] No saved conversations found");
                    return null;
                }

                var json = await File.ReadAllTextAsync(filePath);
                var data = JsonSerializer.Deserialize<ConversationData>(json);

                if (data != null && data.Conversations != null)
                {
                    Console.WriteLine($"[MEMORY] ✅ Loaded {data.Count} conversations from {data.SavedAt}");
                    return data.Conversations;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MEMORY] ❌ Failed to load conversations: {ex.Message}");
            }

            return null;
        }

        /// <summary>
        /// Save LLM vocabulary for faster startup
        /// </summary>
        public async Task SaveLLMVocabularyAsync(Dictionary<string, int> vocabulary, Dictionary<string, double[]> embeddings)
        {
            try
            {
                var data = new
                {
                    Vocabulary = vocabulary,
                    VocabularySize = vocabulary.Count,
                    // Save embeddings as list for JSON serialization
                    Embeddings = ConvertEmbeddingsToSerializable(embeddings),
                    SavedAt = DateTime.UtcNow
                };

                var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = false });
                var filePath = Path.Combine(dataDirectory, LLM_VOCAB_FILE);
                await File.WriteAllTextAsync(filePath, json);

                Console.WriteLine($"[MEMORY] ✅ Saved LLM vocabulary ({vocabulary.Count} tokens)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MEMORY] ❌ Failed to save vocabulary: {ex.Message}");
            }
        }

        /// <summary>
        /// Load LLM vocabulary
        /// </summary>
        public async Task<(Dictionary<string, int>? vocab, Dictionary<string, double[]>? embeddings)> LoadLLMVocabularyAsync()
        {
            try
            {
                var filePath = Path.Combine(dataDirectory, LLM_VOCAB_FILE);
                if (!File.Exists(filePath))
                {
                    Console.WriteLine("[MEMORY] No saved vocabulary found");
                    return (null, null);
                }

                var json = await File.ReadAllTextAsync(filePath);
                var data = JsonSerializer.Deserialize<VocabularyData>(json);

                if (data != null && data.Vocabulary != null && data.Embeddings != null)
                {
                    var embeddings = ConvertEmbeddingsFromSerializable(data.Embeddings);
                    Console.WriteLine($"[MEMORY] ✅ Loaded {data.VocabularySize} tokens from {data.SavedAt}");
                    return (data.Vocabulary, embeddings);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MEMORY] ❌ Failed to load vocabulary: {ex.Message}");
            }

            return (null, null);
        }

        private static Dictionary<string, List<double>> ConvertEmbeddingsToSerializable(Dictionary<string, double[]> embeddings)
        {
            var result = new Dictionary<string, List<double>>();
            foreach (var kvp in embeddings)
            {
                result[kvp.Key] = new List<double>(kvp.Value);
            }
            return result;
        }

        private static Dictionary<string, double[]> ConvertEmbeddingsFromSerializable(Dictionary<string, List<double>> embeddings)
        {
            var result = new Dictionary<string, double[]>();
            foreach (var kvp in embeddings)
            {
                result[kvp.Key] = kvp.Value.ToArray();
            }
            return result;
        }

        /// <summary>
        /// Clear all saved memory
        /// </summary>
        public void ClearAllMemory()
        {
            try
            {
                if (Directory.Exists(dataDirectory))
                {
                    Directory.Delete(dataDirectory, recursive: true);
                    Directory.CreateDirectory(dataDirectory);
                    Console.WriteLine("[MEMORY] ✅ All memory cleared");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MEMORY] ❌ Failed to clear memory: {ex.Message}");
            }
        }
    }

    // Data classes for serialization
    public class GoalSaveData
    {
        public Dictionary<string, GoalGradients>? Gradients { get; set; }
        public List<Goal>? CompletedGoals { get; set; }
        public DateTime SavedAt { get; set; }
    }

    public class PersonalityData
    {
        public double Curiosity { get; set; }
        public double Friendliness { get; set; }
        public double Assertiveness { get; set; }
        public double Playfulness { get; set; }
        public double Caution { get; set; }
        public double Creativity { get; set; }
        public double Patience { get; set; }
        public double Helpfulness { get; set; }
        public DateTime SavedAt { get; set; }
    }

    public class ConversationData
    {
        public List<string>? Conversations { get; set; }
        public int Count { get; set; }
        public DateTime SavedAt { get; set; }
    }

    public class VocabularyData
    {
        public Dictionary<string, int>? Vocabulary { get; set; }
        public Dictionary<string, List<double>>? Embeddings { get; set; }
        public int VocabularySize { get; set; }
        public DateTime SavedAt { get; set; }
    }
}
