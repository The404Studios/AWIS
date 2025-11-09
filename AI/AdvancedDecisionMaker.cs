using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.AI
{
    /// <summary>
    /// Advanced decision-making system using multi-criteria analysis and learning
    /// </summary>
    public class AdvancedDecisionMaker
    {
        private readonly Dictionary<string, DecisionNode> decisionTree;
        private readonly Dictionary<string, double> actionSuccessRates;
        private readonly Random random;

        // Context weights (learned over time)
        private double explorationWeight = 0.3;
        private double safetyWeight = 0.2;
        private double goalProgressWeight = 0.35;
        private double socialWeight = 0.15;

        public AdvancedDecisionMaker()
        {
            decisionTree = new Dictionary<string, DecisionNode>();
            actionSuccessRates = new Dictionary<string, double>();
            random = new Random();

            BuildDecisionTree();
            Console.WriteLine("[DECISION] Advanced decision maker initialized");
        }

        private void BuildDecisionTree()
        {
            // Root decision nodes
            decisionTree["idle"] = new DecisionNode
            {
                Name = "idle",
                Criteria = new List<DecisionCriterion>
                {
                    new() { Name = "has_active_goal", Weight = 0.8 },
                    new() { Name = "low_energy", Weight = 0.1 },
                    new() { Name = "random_exploration", Weight = 0.1 }
                },
                Outcomes = new Dictionary<string, string>
                {
                    ["has_active_goal"] = "work_on_goal",
                    ["low_energy"] = "rest",
                    ["random_exploration"] = "explore"
                }
            };

            decisionTree["work_on_goal"] = new DecisionNode
            {
                Name = "work_on_goal",
                Criteria = new List<DecisionCriterion>
                {
                    new() { Name = "goal_is_exploration", Weight = 0.4 },
                    new() { Name = "goal_is_combat", Weight = 0.3 },
                    new() { Name = "goal_is_learning", Weight = 0.3 }
                },
                Outcomes = new Dictionary<string, string>
                {
                    ["goal_is_exploration"] = "explore_area",
                    ["goal_is_combat"] = "combat_action",
                    ["goal_is_learning"] = "practice_skills"
                }
            };

            decisionTree["explore"] = new DecisionNode
            {
                Name = "explore",
                Criteria = new List<DecisionCriterion>
                {
                    new() { Name = "see_new_area", Weight = 0.5 },
                    new() { Name = "scan_surroundings", Weight = 0.3 },
                    new() { Name = "analyze_objects", Weight = 0.2 }
                },
                Outcomes = new Dictionary<string, string>
                {
                    ["see_new_area"] = "move_forward",
                    ["scan_surroundings"] = "look_around_360",
                    ["analyze_objects"] = "focus_on_object"
                }
            };

            decisionTree["encounter_obstacle"] = new DecisionNode
            {
                Name = "encounter_obstacle",
                Criteria = new List<DecisionCriterion>
                {
                    new() { Name = "can_overcome", Weight = 0.6 },
                    new() { Name = "should_avoid", Weight = 0.3 },
                    new() { Name = "need_help", Weight = 0.1 }
                },
                Outcomes = new Dictionary<string, string>
                {
                    ["can_overcome"] = "tackle_obstacle",
                    ["should_avoid"] = "find_alternative_path",
                    ["need_help"] = "request_assistance"
                }
            };

            decisionTree["social_interaction"] = new DecisionNode
            {
                Name = "social_interaction",
                Criteria = new List<DecisionCriterion>
                {
                    new() { Name = "respond_friendly", Weight = 0.7 },
                    new() { Name = "analyze_intent", Weight = 0.2 },
                    new() { Name = "learn_from_user", Weight = 0.1 }
                },
                Outcomes = new Dictionary<string, string>
                {
                    ["respond_friendly"] = "friendly_response",
                    ["analyze_intent"] = "understand_request",
                    ["learn_from_user"] = "record_for_learning"
                }
            };
        }

        /// <summary>
        /// Make a decision based on current context using multi-criteria analysis
        /// </summary>
        public DecisionResult MakeDecision(DecisionContext context)
        {
            var currentNode = decisionTree.GetValueOrDefault(context.CurrentState)
                            ?? decisionTree["idle"];

            // Evaluate all criteria with context-specific weights
            var criteriaScores = EvaluateCriteria(currentNode, context);

            // Select best criterion based on weighted scores
            var selectedCriterion = SelectBestCriterion(criteriaScores);

            // Get outcome action
            var action = currentNode.Outcomes.GetValueOrDefault(selectedCriterion)
                       ?? "default_action";

            // Calculate confidence based on score distribution
            var confidence = CalculateConfidence(criteriaScores, selectedCriterion);

            // Learn from this decision (update weights)
            UpdateWeights(context, confidence);

            return new DecisionResult
            {
                Action = action,
                Confidence = confidence,
                Reasoning = $"Selected '{selectedCriterion}' with {confidence:P0} confidence",
                AlternativeActions = GetAlternatives(criteriaScores, selectedCriterion)
            };
        }

        private Dictionary<string, double> EvaluateCriteria(DecisionNode node, DecisionContext context)
        {
            var scores = new Dictionary<string, double>();

            foreach (var criterion in node.Criteria)
            {
                var baseScore = criterion.Weight;

                // Adjust score based on context
                if (context.HasActiveGoal && criterion.Name.Contains("goal"))
                    baseScore *= (1 + goalProgressWeight);

                if (context.RecentFailures > 2 && criterion.Name.Contains("avoid"))
                    baseScore *= (1 + safetyWeight);

                if (context.ExplorationDesire > 0.7 && criterion.Name.Contains("explore"))
                    baseScore *= (1 + explorationWeight);

                if (context.SocialInteraction && criterion.Name.Contains("social"))
                    baseScore *= (1 + socialWeight);

                // Factor in past success rates
                var historicalSuccess = actionSuccessRates.GetValueOrDefault(criterion.Name, 0.5);
                baseScore *= (0.5 + historicalSuccess * 0.5);

                // Add small random variation for diversity
                baseScore *= (0.9 + random.NextDouble() * 0.2);

                scores[criterion.Name] = baseScore;
            }

            return scores;
        }

        private string SelectBestCriterion(Dictionary<string, double> scores)
        {
            if (scores.Count == 0)
                return "default";

            // Use softmax for probabilistic selection (exploration vs exploitation)
            var expScores = scores.ToDictionary(
                kvp => kvp.Key,
                kvp => Math.Exp(kvp.Value * 2.0) // Temperature = 0.5
            );

            var totalExp = expScores.Values.Sum();
            var threshold = random.NextDouble() * totalExp;

            double cumulative = 0;
            foreach (var kvp in expScores.OrderByDescending(k => k.Value))
            {
                cumulative += kvp.Value;
                if (cumulative >= threshold)
                    return kvp.Key;
            }

            return scores.OrderByDescending(k => k.Value).First().Key;
        }

        private static double CalculateConfidence(Dictionary<string, double> scores, string selected)
        {
            if (scores.Count == 0) return 0.5;

            var selectedScore = scores.GetValueOrDefault(selected, 0);
            var avgScore = scores.Values.Average();
            var maxScore = scores.Values.Max();

            // Confidence based on how much better the selected option is
            var confidence = selectedScore / (maxScore + 0.001);

            // Penalize if scores are very close (uncertain decision)
            var variance = scores.Values.Select(s => Math.Pow(s - avgScore, 2)).Average();
            if (variance < 0.01)
                confidence *= 0.7;

            return Math.Clamp(confidence, 0.0, 1.0);
        }

        private static List<string> GetAlternatives(Dictionary<string, double> scores, string selected)
        {
            return scores
                .Where(kvp => kvp.Key != selected)
                .OrderByDescending(kvp => kvp.Value)
                .Take(2)
                .Select(kvp => kvp.Key)
                .ToList();
        }

        private void UpdateWeights(DecisionContext context, double confidence)
        {
            // Adjust context weights based on decision outcomes
            var adjustment = 0.01 * confidence;

            if (context.HasActiveGoal)
                goalProgressWeight = Math.Clamp(goalProgressWeight + adjustment, 0.1, 0.6);

            if (context.RecentFailures > 0)
                safetyWeight = Math.Clamp(safetyWeight + adjustment * 0.5, 0.1, 0.4);

            if (context.ExplorationDesire > 0.5)
                explorationWeight = Math.Clamp(explorationWeight + adjustment * 0.3, 0.1, 0.5);

            if (context.SocialInteraction)
                socialWeight = Math.Clamp(socialWeight + adjustment * 0.2, 0.05, 0.3);

            // Normalize weights to sum to ~1.0
            var total = explorationWeight + safetyWeight + goalProgressWeight + socialWeight;
            if (total > 0)
            {
                explorationWeight /= total;
                safetyWeight /= total;
                goalProgressWeight /= total;
                socialWeight /= total;
            }
        }

        /// <summary>
        /// Learn from action outcome to improve future decisions
        /// </summary>
        public void LearnFromOutcome(string criterion, bool success)
        {
            if (!actionSuccessRates.ContainsKey(criterion))
                actionSuccessRates[criterion] = 0.5;

            // Exponential moving average
            var currentRate = actionSuccessRates[criterion];
            var newRate = currentRate * 0.9 + (success ? 1.0 : 0.0) * 0.1;
            actionSuccessRates[criterion] = newRate;

            Console.WriteLine($"[DECISION] Learned: {criterion} → Success rate: {newRate:P1}");
        }

        /// <summary>
        /// Get current decision-making statistics
        /// </summary>
        public string GetStatistics()
        {
            var stats = "Decision Making Statistics:\n";
            stats += $"  Exploration Weight: {explorationWeight:F3}\n";
            stats += $"  Safety Weight: {safetyWeight:F3}\n";
            stats += $"  Goal Progress Weight: {goalProgressWeight:F3}\n";
            stats += $"  Social Weight: {socialWeight:F3}\n";
            stats += $"  Learned Actions: {actionSuccessRates.Count}\n";

            if (actionSuccessRates.Count > 0)
            {
                stats += "  Top Actions:\n";
                foreach (var kvp in actionSuccessRates.OrderByDescending(k => k.Value).Take(5))
                {
                    stats += $"    - {kvp.Key}: {kvp.Value:P0}\n";
                }
            }

            return stats;
        }
    }

    public class DecisionNode
    {
        public string Name { get; set; } = string.Empty;
        public List<DecisionCriterion> Criteria { get; set; } = new();
        public Dictionary<string, string> Outcomes { get; set; } = new();
    }

    public class DecisionCriterion
    {
        public string Name { get; set; } = string.Empty;
        public double Weight { get; set; }
    }

    public class DecisionContext
    {
        public string CurrentState { get; set; } = "idle";
        public bool HasActiveGoal { get; set; }
        public int RecentFailures { get; set; }
        public double ExplorationDesire { get; set; }
        public bool SocialInteraction { get; set; }
        public Dictionary<string, object> AdditionalContext { get; set; } = new();
    }

    public class DecisionResult
    {
        public string Action { get; set; } = string.Empty;
        public double Confidence { get; set; }
        public string Reasoning { get; set; } = string.Empty;
        public List<string> AlternativeActions { get; set; } = new();
    }
}
