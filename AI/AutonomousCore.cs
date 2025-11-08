using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AWIS.Core;

namespace AWIS.AI
{
    /// <summary>
    /// Represents an AI decision with rationale
    /// </summary>
    public class AIDecision
    {
        public AIAction RecommendedAction { get; set; }
        public double Confidence { get; set; }
        public string Rationale { get; set; } = string.Empty;
        public List<AIAction> AlternativeActions { get; set; } = new();
        public Dictionary<string, double> Factors { get; set; } = new();

        public AIDecision(AIAction action, double confidence, string rationale = "")
        {
            RecommendedAction = action;
            Confidence = confidence;
            Rationale = rationale;
        }
    }

    /// <summary>
    /// Context analysis result
    /// </summary>
    public class ContextAnalysis
    {
        public string Summary { get; set; } = string.Empty;
        public List<string> KeyElements { get; set; } = new();
        public Dictionary<string, double> Features { get; set; } = new();
        public double Complexity { get; set; }
        public double Urgency { get; set; }
        public List<string> PotentialRisks { get; set; } = new();
        public List<string> Opportunities { get; set; } = new();
    }

    /// <summary>
    /// Autonomous intelligence core that makes decisions
    /// </summary>
    public class AutonomousIntelligenceCore
    {
        private readonly ExperienceManager experienceManager;
        private readonly EmotionalSocketManager emotionalManager;
        private readonly MemoryArchitecture memoryArchitecture;
        private readonly HierarchicalKnowledgeBase knowledgeBase;
        private readonly Random random = new();

        public AutonomousIntelligenceCore()
        {
            experienceManager = new ExperienceManager();
            emotionalManager = new EmotionalSocketManager();
            memoryArchitecture = new MemoryArchitecture();
            knowledgeBase = new HierarchicalKnowledgeBase();
        }

        public AutonomousIntelligenceCore(ExperienceManager experiences, EmotionalSocketManager emotions,
                                         MemoryArchitecture memory, HierarchicalKnowledgeBase knowledge)
        {
            experienceManager = experiences;
            emotionalManager = emotions;
            memoryArchitecture = memory;
            knowledgeBase = knowledge;
        }

        /// <summary>
        /// Analyzes the current context
        /// </summary>
        public ContextAnalysis AnalyzeContext(string context)
        {
            var analysis = new ContextAnalysis
            {
                Summary = context
            };

            // Extract key elements
            var words = context.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            analysis.KeyElements = words.Take(10).ToList();

            // Compute complexity based on context length and vocabulary
            analysis.Complexity = Math.Min(1.0, words.Length / 100.0);

            // Check memory for similar contexts
            var memories = memoryArchitecture.RecallMultiple(context, 5);
            if (memories.Any())
            {
                analysis.Features["memory_relevance"] = memories.Average(m => m.GetStrength());
            }

            // Check knowledge base
            var relatedConcepts = new List<string>();
            foreach (var word in words.Take(5))
            {
                var concepts = knowledgeBase.SearchNodes(word);
                relatedConcepts.AddRange(concepts.Select(c => c.Name));
            }
            analysis.Features["knowledge_coverage"] = relatedConcepts.Distinct().Count() / Math.Max(1.0, words.Length);

            // Assess urgency based on keywords
            var urgentKeywords = new[] { "emergency", "urgent", "critical", "immediate", "now", "quickly" };
            analysis.Urgency = words.Count(w => urgentKeywords.Contains(w.ToLower())) / 6.0;

            // Identify risks
            var riskKeywords = new[] { "error", "fail", "problem", "issue", "danger" };
            if (words.Any(w => riskKeywords.Contains(w.ToLower())))
            {
                analysis.PotentialRisks.Add("Potential failure scenario detected");
            }

            // Identify opportunities
            var opportunityKeywords = new[] { "improve", "optimize", "enhance", "better", "new" };
            if (words.Any(w => opportunityKeywords.Contains(w.ToLower())))
            {
                analysis.Opportunities.Add("Improvement opportunity detected");
            }

            return analysis;
        }

        /// <summary>
        /// Makes a decision based on context and experience
        /// </summary>
        public AIDecision MakeDecision(string context)
        {
            // Analyze the context
            var contextAnalysis = AnalyzeContext(context);

            // Get best action from experience
            var bestAction = experienceManager.GetBestAction(context);
            var expectedReward = experienceManager.GetExpectedReward(context, bestAction);

            // Consider emotional state
            var emotionalState = emotionalManager.CurrentState;
            var emotionalModifier = emotionalState.GetValence() * 0.1; // Small emotional influence

            // Compute confidence
            var confidence = Math.Min(0.95, Math.Max(0.3, expectedReward + emotionalModifier));

            // Build decision
            var decision = new AIDecision(
                new AIAction(bestAction, $"Execute {bestAction} in context: {context}"),
                confidence,
                $"Based on {experienceManager.GetStatistics().TotalExperiences} experiences. " +
                $"Context complexity: {contextAnalysis.Complexity:F2}. " +
                $"Emotional state: {emotionalState.GetDominantEmotion()}."
            );

            // Add factors
            decision.Factors["experience_quality"] = expectedReward;
            decision.Factors["emotional_influence"] = emotionalModifier;
            decision.Factors["context_complexity"] = contextAnalysis.Complexity;
            decision.Factors["urgency"] = contextAnalysis.Urgency;

            // Add alternative actions
            var alternatives = Enum.GetValues(typeof(ActionType)).Cast<ActionType>()
                .Where(a => a != bestAction)
                .OrderByDescending(a => experienceManager.GetExpectedReward(context, a))
                .Take(3)
                .ToList();

            foreach (var alt in alternatives)
            {
                decision.AlternativeActions.Add(new AIAction(alt, $"Alternative: {alt}"));
            }

            return decision;
        }

        /// <summary>
        /// Learns from an action result
        /// </summary>
        public void LearnFromOutcome(string context, AIAction action, ActionResult result)
        {
            // Create experience
            var experience = new Experience(action, result)
            {
                Context = context
            };

            // Add to experience manager
            experienceManager.AddExperience(experience);

            // Update emotional state
            emotionalManager.ProcessExperience(experience);

            // Store in memory
            var importance = result.Success ? 0.7 : 0.5;
            memoryArchitecture.Store(
                $"{context} -> {action.Type}: {result.Message}",
                MemoryType.Episodic,
                importance
            );

            // Update knowledge base
            var actionNode = knowledgeBase.AddNode(action.Type.ToString(), "Action");
            var contextNode = knowledgeBase.AddNode(context, "Context");
            var outcome = result.Success ? "Success" : "Failure";
            var outcomeNode = knowledgeBase.AddNode(outcome, "Outcome");

            knowledgeBase.AddRelation(contextNode.Id, actionNode.Id, RelationType.Enables, result.Success ? 0.8 : 0.2);
            knowledgeBase.AddRelation(actionNode.Id, outcomeNode.Id, RelationType.Causes, 0.7);
        }

        /// <summary>
        /// Gets the current system status
        /// </summary>
        public string GetSystemStatus()
        {
            var expStats = experienceManager.GetStatistics();
            var memStats = memoryArchitecture.GetStatistics();
            var kbStats = knowledgeBase.GetStatistics();
            var mood = emotionalManager.GetMoodReport();

            return $@"=== Autonomous Intelligence Core Status ===
Experiences: {expStats.TotalExperiences} total, {expStats.SuccessRate:P0} success rate
Memories: {memStats.TotalMemories} total ({memStats.ShortTermCount} short-term, {memStats.LongTermCount} long-term)
Knowledge: {kbStats.TotalNodes} concepts, {kbStats.TotalRelations} relations
Emotional State: {mood}
Average Reward: {expStats.AverageReward:F2}";
        }

        public ExperienceManager GetExperienceManager() => experienceManager;
        public EmotionalSocketManager GetEmotionalManager() => emotionalManager;
        public MemoryArchitecture GetMemoryArchitecture() => memoryArchitecture;
        public HierarchicalKnowledgeBase GetKnowledgeBase() => knowledgeBase;
    }

    /// <summary>
    /// Advanced cognitive processor for higher-level thinking
    /// </summary>
    public class AdvancedCognitiveProcessor
    {
        private readonly AutonomousIntelligenceCore core;
        private readonly List<string> thoughtLog = new();

        public AdvancedCognitiveProcessor(AutonomousIntelligenceCore core)
        {
            this.core = core;
        }

        /// <summary>
        /// Performs deep reasoning about a problem
        /// </summary>
        public async Task<List<string>> Reason(string problem, int depth = 3)
        {
            var reasoning = new List<string>();
            reasoning.Add($"Problem: {problem}");

            for (int i = 0; i < depth; i++)
            {
                await Task.Delay(10); // Simulate thinking time

                var analysis = core.AnalyzeContext(problem);
                reasoning.Add($"Step {i + 1}: Complexity {analysis.Complexity:F2}, Urgency {analysis.Urgency:F2}");

                // Recall similar situations
                var memories = core.GetMemoryArchitecture().RecallMultiple(problem, 3);
                if (memories.Any())
                {
                    reasoning.Add($"  Recalled {memories.Count} similar situations");
                }

                // Make decision
                var decision = core.MakeDecision(problem);
                reasoning.Add($"  Recommended: {decision.RecommendedAction.Type} (confidence: {decision.Confidence:F2})");

                // Update problem with new insights
                problem += $" considering {decision.RecommendedAction.Type}";
            }

            thoughtLog.AddRange(reasoning);
            return reasoning;
        }

        /// <summary>
        /// Generates creative solutions to a problem
        /// </summary>
        public List<AIAction> GenerateCreativeSolutions(string problem, int numSolutions = 5)
        {
            var solutions = new List<AIAction>();
            var usedActions = new HashSet<ActionType>();

            // Get standard solution
            var standardDecision = core.MakeDecision(problem);
            solutions.Add(standardDecision.RecommendedAction);
            usedActions.Add(standardDecision.RecommendedAction.Type);

            // Generate creative alternatives
            var allActions = Enum.GetValues(typeof(ActionType)).Cast<ActionType>().ToList();
            var random = new Random();

            while (solutions.Count < numSolutions && usedActions.Count < allActions.Count)
            {
                var actionType = allActions[random.Next(allActions.Count)];
                if (!usedActions.Contains(actionType))
                {
                    usedActions.Add(actionType);
                    solutions.Add(new AIAction(actionType, $"Creative solution: {actionType}"));
                }
            }

            return solutions;
        }

        public List<string> GetThoughtLog() => thoughtLog.ToList();
    }
}
