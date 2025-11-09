using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.AI
{
    /// <summary>
    /// Goal-driven system for autonomous behavior with self-directed learning
    /// Uses gradient accumulation to learn optimal strategies
    /// </summary>
    public class GoalSystem
    {
        private readonly List<Goal> activeGoals = new();
        private readonly List<Goal> completedGoals = new();
        private readonly Dictionary<string, GoalGradients> learningGradients = new();
        private readonly Random random = new();

        public int TotalGoalsCompleted => completedGoals.Count;
        public int ActiveGoalCount => activeGoals.Count;

        public GoalSystem()
        {
            InitializeDefaultGoals();
        }

        private void InitializeDefaultGoals()
        {
            // Start with some basic autonomous goals
            SetGoal("explore_environment", "Explore the environment", GoalPriority.Medium, 300000); // 5 minutes
            SetGoal("learn_controls", "Learn and master all controls", GoalPriority.High, 600000); // 10 minutes
            SetGoal("find_objectives", "Find and identify objectives", GoalPriority.Medium, 900000); // 15 minutes
        }

        /// <summary>
        /// Set a new goal
        /// </summary>
        public void SetGoal(string id, string description, GoalPriority priority, int timeoutMs = 300000)
        {
            var goal = new Goal
            {
                Id = id,
                Description = description,
                Priority = priority,
                Status = GoalStatus.Active,
                CreatedAt = DateTime.Now,
                TimeoutMs = timeoutMs
            };

            activeGoals.Add(goal);
            Console.WriteLine($"[GOAL] 🎯 New goal: {description} (Priority: {priority})");

            // Initialize gradients for this goal type
            if (!learningGradients.ContainsKey(id))
            {
                learningGradients[id] = new GoalGradients();
            }
        }

        /// <summary>
        /// Get the highest priority active goal
        /// </summary>
        public Goal? GetCurrentGoal()
        {
            // Remove expired goals
            var now = DateTime.Now;
            activeGoals.RemoveAll(g => (now - g.CreatedAt).TotalMilliseconds > g.TimeoutMs);

            // Return highest priority goal
            return activeGoals
                .OrderByDescending(g => g.Priority)
                .ThenBy(g => g.CreatedAt)
                .FirstOrDefault();
        }

        /// <summary>
        /// Complete a goal with success/failure and reward
        /// </summary>
        public void CompleteGoal(string id, bool success, double reward)
        {
            var goal = activeGoals.FirstOrDefault(g => g.Id == id);
            if (goal == null) return;

            goal.Status = success ? GoalStatus.Completed : GoalStatus.Failed;
            goal.CompletedAt = DateTime.Now;
            goal.Reward = reward;

            activeGoals.Remove(goal);
            completedGoals.Add(goal);

            // Update gradients for learning
            UpdateGradients(id, success, reward);

            var emoji = success ? "✅" : "❌";
            Console.WriteLine($"[GOAL] {emoji} {goal.Description}: {(success ? "SUCCESS" : "FAILED")} (Reward: {reward:F2})");

            // Suggest new goals based on learning
            if (success && reward > 0.7)
            {
                SuggestNextGoal(id);
            }
        }

        /// <summary>
        /// Update gradients for gradient accumulation learning
        /// </summary>
        private void UpdateGradients(string goalId, bool success, double reward)
        {
            if (!learningGradients.TryGetValue(goalId, out var gradients))
            {
                gradients = new GoalGradients();
                learningGradients[goalId] = gradients;
            }

            // Accumulate gradients
            gradients.SuccessCount += success ? 1 : 0;
            gradients.FailureCount += success ? 0 : 1;
            gradients.TotalReward += reward;
            gradients.AverageReward = gradients.TotalReward / (gradients.SuccessCount + gradients.FailureCount);

            // Calculate success rate gradient (how much to prioritize this goal type)
            double successRate = gradients.SuccessCount / (double)(gradients.SuccessCount + gradients.FailureCount);
            gradients.PriorityGradient = successRate * gradients.AverageReward;

            Console.WriteLine($"[LEARNING] {goalId}: Success rate={successRate:P1}, Avg reward={gradients.AverageReward:F2}, Gradient={gradients.PriorityGradient:F3}");
        }

        /// <summary>
        /// Suggest next goal based on learned gradients
        /// </summary>
        private void SuggestNextGoal(string completedGoalId)
        {
            var suggestions = completedGoalId switch
            {
                "explore_environment" => new[] {
                    ("map_area", "Map out the discovered area", GoalPriority.Medium),
                    ("find_resources", "Find and collect resources", GoalPriority.High)
                },
                "learn_controls" => new[] {
                    ("practice_combat", "Practice combat techniques", GoalPriority.Medium),
                    ("master_movement", "Master advanced movement", GoalPriority.High)
                },
                "find_objectives" => new[] {
                    ("complete_objective", "Complete found objective", GoalPriority.High),
                    ("optimize_route", "Find optimal route to objectives", GoalPriority.Medium)
                },
                _ => new[] {
                    ("experiment", "Try new behaviors and strategies", GoalPriority.Medium),
                    ("improve_skills", "Improve current skills", GoalPriority.Low)
                }
            };

            // Pick a suggestion based on learned gradients
            var suggestion = suggestions[random.Next(suggestions.Length)];
            SetGoal(suggestion.Item1, suggestion.Item2, suggestion.Item3);
        }

        /// <summary>
        /// Autonomously generate a new goal based on context
        /// </summary>
        public void GenerateAutonomousGoal(string context = "")
        {
            // Use gradients to determine what type of goal to generate
            var bestGoalType = learningGradients
                .OrderByDescending(kvp => kvp.Value.PriorityGradient)
                .FirstOrDefault();

            var goalTypes = new[]
            {
                ("explore_new_area", "Explore a new undiscovered area", GoalPriority.Medium),
                ("test_hypothesis", "Test a hypothesis about the environment", GoalPriority.Low),
                ("optimize_behavior", "Optimize current behavior patterns", GoalPriority.Medium),
                ("seek_challenge", "Seek out a new challenge", GoalPriority.High),
                ("practice_skill", "Practice and improve a specific skill", GoalPriority.Low)
            };

            var chosen = goalTypes[random.Next(goalTypes.Length)];
            SetGoal(chosen.Item1 + "_" + Guid.NewGuid().ToString("N")[..8], chosen.Item2, chosen.Item3);
        }

        /// <summary>
        /// Get progress towards current goal
        /// </summary>
        public double GetGoalProgress(string goalId)
        {
            var goal = activeGoals.FirstOrDefault(g => g.Id == goalId);
            if (goal == null) return 0.0;

            var elapsed = (DateTime.Now - goal.CreatedAt).TotalMilliseconds;
            return Math.Min(elapsed / goal.TimeoutMs, 1.0);
        }

        /// <summary>
        /// Get learning statistics
        /// </summary>
        public string GetLearningStatistics()
        {
            var stats = $"Goals completed: {completedGoals.Count}\n";
            stats += $"Active goals: {activeGoals.Count}\n";
            stats += "Learned gradients:\n";

            foreach (var kvp in learningGradients.OrderByDescending(g => g.Value.PriorityGradient).Take(5))
            {
                stats += $"  {kvp.Key}: Priority={kvp.Value.PriorityGradient:F3}, Success={kvp.Value.SuccessCount}, Fail={kvp.Value.FailureCount}\n";
            }

            return stats;
        }
    }

    public class Goal
    {
        public string Id { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public GoalPriority Priority { get; set; }
        public GoalStatus Status { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime? CompletedAt { get; set; }
        public int TimeoutMs { get; set; }
        public double Reward { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    public class GoalGradients
    {
        public int SuccessCount { get; set; }
        public int FailureCount { get; set; }
        public double TotalReward { get; set; }
        public double AverageReward { get; set; }
        public double PriorityGradient { get; set; }
    }

    public enum GoalPriority
    {
        Low = 1,
        Medium = 2,
        High = 3,
        Critical = 4
    }

    public enum GoalStatus
    {
        Active,
        Completed,
        Failed,
        Cancelled
    }
}
