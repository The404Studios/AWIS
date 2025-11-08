using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.Core
{
    /// <summary>
    /// Represents a learned experience with context and outcomes
    /// </summary>
    public class Experience
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public string Context { get; set; } = string.Empty;
        public AIAction Action { get; set; }
        public ActionResult Result { get; set; }
        public double Reward { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new();
        public int AccessCount { get; set; } = 0;
        public DateTime LastAccessed { get; set; } = DateTime.UtcNow;

        public Experience(AIAction action, ActionResult result)
        {
            Action = action;
            Result = result;
            Reward = result.Success ? 1.0 : -1.0;
        }

        public void RecordAccess()
        {
            AccessCount++;
            LastAccessed = DateTime.UtcNow;
        }

        public double GetRelevanceScore()
        {
            // Combines recency and access frequency
            var recencyScore = 1.0 / (1.0 + (DateTime.UtcNow - LastAccessed).TotalHours);
            var frequencyScore = Math.Log(1.0 + AccessCount);
            return (recencyScore + frequencyScore) / 2.0;
        }
    }

    /// <summary>
    /// Manages and learns from experiences
    /// </summary>
    public class ExperienceManager
    {
        private readonly List<Experience> experiences = new();
        private readonly Dictionary<string, List<Experience>> contextIndex = new();
        private readonly int maxExperiences;

        public ExperienceManager(int maxExperiences = 10000)
        {
            this.maxExperiences = maxExperiences;
        }

        public void AddExperience(Experience experience)
        {
            lock (experiences)
            {
                experiences.Add(experience);

                // Index by context
                if (!contextIndex.ContainsKey(experience.Context))
                {
                    contextIndex[experience.Context] = new List<Experience>();
                }
                contextIndex[experience.Context].Add(experience);

                // Prune old experiences if needed
                if (experiences.Count > maxExperiences)
                {
                    PruneExperiences();
                }
            }
        }

        public List<Experience> GetSimilarExperiences(string context, int limit = 10)
        {
            lock (experiences)
            {
                if (contextIndex.TryGetValue(context, out var contextExperiences))
                {
                    return contextExperiences
                        .OrderByDescending(e => e.GetRelevanceScore())
                        .Take(limit)
                        .ToList();
                }

                // Fuzzy matching if exact context not found
                return experiences
                    .Where(e => e.Context.Contains(context, StringComparison.OrdinalIgnoreCase))
                    .OrderByDescending(e => e.GetRelevanceScore())
                    .Take(limit)
                    .ToList();
            }
        }

        public double GetExpectedReward(string context, ActionType actionType)
        {
            var similarExperiences = GetSimilarExperiences(context)
                .Where(e => e.Action.Type == actionType)
                .ToList();

            if (!similarExperiences.Any())
            {
                return 0.5; // Neutral expected reward for unknown actions
            }

            return similarExperiences.Average(e => e.Reward);
        }

        public ActionType GetBestAction(string context)
        {
            var similarExperiences = GetSimilarExperiences(context, 50);

            if (!similarExperiences.Any())
            {
                return ActionType.Observe; // Default to observing when uncertain
            }

            // Find action with highest average reward
            var actionRewards = similarExperiences
                .GroupBy(e => e.Action.Type)
                .Select(g => new { ActionType = g.Key, AvgReward = g.Average(e => e.Reward) })
                .OrderByDescending(x => x.AvgReward)
                .ToList();

            return actionRewards.First().ActionType;
        }

        private void PruneExperiences()
        {
            // Remove lowest relevance experiences
            var toRemove = experiences
                .OrderBy(e => e.GetRelevanceScore())
                .Take(experiences.Count - (int)(maxExperiences * 0.9))
                .ToList();

            foreach (var exp in toRemove)
            {
                experiences.Remove(exp);

                if (contextIndex.TryGetValue(exp.Context, out var list))
                {
                    list.Remove(exp);
                    if (list.Count == 0)
                    {
                        contextIndex.Remove(exp.Context);
                    }
                }
            }
        }

        public ExperienceStatistics GetStatistics()
        {
            lock (experiences)
            {
                return new ExperienceStatistics
                {
                    TotalExperiences = experiences.Count,
                    UniqueContexts = contextIndex.Count,
                    AverageReward = experiences.Any() ? experiences.Average(e => e.Reward) : 0,
                    SuccessRate = experiences.Any() ?
                        experiences.Count(e => e.Result.Success) / (double)experiences.Count : 0
                };
            }
        }
    }

    public class ExperienceStatistics
    {
        public int TotalExperiences { get; set; }
        public int UniqueContexts { get; set; }
        public double AverageReward { get; set; }
        public double SuccessRate { get; set; }
    }
}
