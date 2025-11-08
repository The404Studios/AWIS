using System;
using System.Threading.Tasks;
using System.Drawing;
using AWIS.AI;
using AWIS.Vision;
using AWIS.Voice;

namespace AWIS.Core
{
    /// <summary>
    /// Comprehensive demonstration of all AWIS systems
    /// </summary>
    public class SystemDemo
    {
        public static async Task RunFullSystemDemo()
        {
            Console.Clear();
            PrintHeader("AWIS v8.0 - Complete System Demonstration");

            // Create core AI systems
            var experienceManager = new ExperienceManager();
            var emotionalManager = new EmotionalSocketManager();
            var memoryArchitecture = new MemoryArchitecture();
            var knowledgeBase = new HierarchicalKnowledgeBase();

            var core = new AutonomousIntelligenceCore(
                experienceManager,
                emotionalManager,
                memoryArchitecture,
                knowledgeBase
            );

            var cognitiveProcessor = new AdvancedCognitiveProcessor(core);

            await DemoMemorySystem(memoryArchitecture);
            await DemoKnowledgeBase(knowledgeBase);
            await DemoEmotionalSystem(emotionalManager, experienceManager);
            await DemoAutonomousDecisionMaking(core);
            await DemoCognitiveReasoning(cognitiveProcessor);
            await DemoVoiceCommands();
            DemoComputerVision();

            Console.WriteLine("\n" + new string('=', 70));
            Log.Success("All system demonstrations completed successfully!");
            Console.WriteLine(core.GetSystemStatus());
        }

        private static async Task DemoMemorySystem(MemoryArchitecture memory)
        {
            PrintSection("Memory System");

            Log.Information("Storing memories across different types...");

            memory.Store("User prefers dark mode interface", MemoryType.LongTerm, 0.9);
            memory.Store("Current task: Analyze data from report.csv", MemoryType.Working, 0.8);
            memory.Store("Meeting with team at 2 PM today", MemoryType.ShortTerm, 0.7);
            memory.Store("The capital of France is Paris", MemoryType.Semantic, 1.0);
            memory.Store("How to initialize a C# project", MemoryType.Procedural, 0.85);
            memory.Store("Yesterday: Successfully deployed version 8.0", MemoryType.Episodic, 0.75);

            await Task.Delay(50);

            Log.Information("Recalling memories...");
            var darkModeMemory = memory.Recall("dark mode");
            if (darkModeMemory != null)
            {
                Console.WriteLine($"  Recalled: {darkModeMemory.Content} (Strength: {darkModeMemory.GetStrength():F2})");
            }

            var taskMemories = memory.RecallMultiple("task", 3);
            Console.WriteLine($"  Found {taskMemories.Count} task-related memories");

            memory.Consolidate();
            Log.Information("Memory consolidation completed");

            var stats = memory.GetStatistics();
            Console.WriteLine($"\n  Memory Statistics:");
            Console.WriteLine($"    Total: {stats.TotalMemories}");
            Console.WriteLine($"    Short-term: {stats.ShortTermCount}");
            Console.WriteLine($"    Long-term: {stats.LongTermCount}");
            Console.WriteLine($"    Working: {stats.WorkingMemoryCount}");
            Console.WriteLine($"    Semantic: {stats.SemanticCount}");
        }

        private static async Task DemoKnowledgeBase(HierarchicalKnowledgeBase kb)
        {
            PrintSection("Knowledge Base & Graph System");

            Log.Information("Building knowledge graph...");

            // Add programming concepts
            var csharpNode = kb.AddNode("C#", "ProgrammingLanguage");
            var dotnetNode = kb.AddNode(".NET", "Framework");
            var classNode = kb.AddNode("Class", "Concept");

            kb.AddRelation(csharpNode.Id, dotnetNode.Id, RelationType.PartOf, 1.0);
            kb.AddRelation(csharpNode.Id, classNode.Id, RelationType.HasProperty, 0.9);

            // Add AI concepts
            kb.AddRelation("Machine Learning", "Artificial Intelligence", RelationType.PartOf);
            kb.AddRelation("Neural Network", "Machine Learning", RelationType.IsA);
            kb.AddRelation("Deep Learning", "Neural Network", RelationType.UsedFor);

            await Task.Delay(50);

            Log.Information("Querying knowledge graph...");
            var mlNode = kb.FindNode("Machine Learning");
            if (mlNode != null)
            {
                var related = kb.GetRelatedNodes(mlNode.Id);
                Console.WriteLine($"\n  Concepts related to Machine Learning:");
                foreach (var (node, relType, strength) in related)
                {
                    Console.WriteLine($"    - {node.Name} ({relType}, strength: {strength:F2})");
                }
            }

            Log.Information("Performing inference...");
            var inferred = kb.InferRelatedConcepts("Artificial Intelligence", 2);
            Console.WriteLine($"\n  Inferred related concepts: {string.Join(", ", inferred.ConvertAll(n => n.Name))}");

            var kbStats = kb.GetStatistics();
            Console.WriteLine($"\n  Knowledge Base Statistics:");
            Console.WriteLine($"    Nodes: {kbStats.TotalNodes}");
            Console.WriteLine($"    Relations: {kbStats.TotalRelations}");
            Console.WriteLine($"    Avg Connections: {kbStats.AverageConnections:F2}");
        }

        private static async Task DemoEmotionalSystem(EmotionalSocketManager emotions, ExperienceManager experiences)
        {
            PrintSection("Emotional System");

            Log.Information("Processing experiences and emotional responses...");

            // Simulate positive experience
            var successAction = new AIAction(ActionType.Save, "Save project successfully");
            var successResult = ActionResult.SuccessResult("Project saved");
            var successExp = new Experience(successAction, successResult) { Reward = 0.9 };
            experiences.AddExperience(successExp);
            emotions.ProcessExperience(successExp);

            await Task.Delay(100);
            Console.WriteLine($"  After success: {emotions.GetMoodReport()}");
            Console.WriteLine($"    Joy: {emotions.CurrentState.Joy:F2}, Trust: {emotions.CurrentState.Trust:F2}");

            // Simulate failure experience
            var failAction = new AIAction(ActionType.Load, "Load corrupted file");
            var failResult = ActionResult.FailureResult("File corrupted");
            var failExp = new Experience(failAction, failResult) { Reward = -0.7 };
            experiences.AddExperience(failExp);
            emotions.ProcessExperience(failExp);

            await Task.Delay(100);
            Console.WriteLine($"\n  After failure: {emotions.GetMoodReport()}");
            Console.WriteLine($"    Sadness: {emotions.CurrentState.Sadness:F2}, Fear: {emotions.CurrentState.Fear:F2}");

            // Show emotional trends
            var recentEmotion = emotions.GetAverageEmotion(TimeSpan.FromMinutes(1));
            Console.WriteLine($"\n  Recent emotional trend (1 min avg):");
            Console.WriteLine($"    Valence: {recentEmotion.GetValence():F2} (Positive/Negative)");
            Console.WriteLine($"    Arousal: {recentEmotion.GetArousal():F2} (Energy Level)");
        }

        private static async Task DemoAutonomousDecisionMaking(AutonomousIntelligenceCore core)
        {
            PrintSection("Autonomous Decision Making");

            Log.Information("Analyzing context and making decisions...");

            var contexts = new[]
            {
                "User requested to open a file urgently",
                "System error detected in module A",
                "New optimization opportunity found in database",
                "User wants to learn about machine learning"
            };

            foreach (var context in contexts)
            {
                Console.WriteLine($"\n  Context: {context}");

                var analysis = core.AnalyzeContext(context);
                Console.WriteLine($"    Complexity: {analysis.Complexity:F2}, Urgency: {analysis.Urgency:F2}");
                if (analysis.Opportunities.Count > 0)
                {
                    Console.WriteLine($"    Opportunities: {string.Join(", ", analysis.Opportunities)}");
                }

                var decision = core.MakeDecision(context);
                Console.WriteLine($"    Decision: {decision.RecommendedAction.Type} (confidence: {decision.Confidence:F2})");
                Console.WriteLine($"    Rationale: {decision.Rationale.Substring(0, Math.Min(80, decision.Rationale.Length))}...");

                // Simulate execution and learning
                var result = new Random().NextDouble() > 0.3 ?
                    ActionResult.SuccessResult() :
                    ActionResult.FailureResult("Simulated failure");

                core.LearnFromOutcome(context, decision.RecommendedAction, result);

                await Task.Delay(50);
            }

            var expStats = core.GetExperienceManager().GetStatistics();
            Console.WriteLine($"\n  Learning Progress:");
            Console.WriteLine($"    Experiences: {expStats.TotalExperiences}");
            Console.WriteLine($"    Success Rate: {expStats.SuccessRate:P0}");
            Console.WriteLine($"    Average Reward: {expStats.AverageReward:F2}");
        }

        private static async Task DemoCognitiveReasoning(AdvancedCognitiveProcessor processor)
        {
            PrintSection("Cognitive Reasoning");

            Log.Information("Performing deep reasoning...");

            var problem = "How to improve system performance";
            Console.WriteLine($"  Problem: {problem}\n");

            var reasoning = await processor.Reason(problem, 3);
            foreach (var thought in reasoning)
            {
                Console.WriteLine($"    {thought}");
            }

            Console.WriteLine();
            Log.Information("Generating creative solutions...");
            var solutions = processor.GenerateCreativeSolutions(problem, 5);
            Console.WriteLine($"  Generated {solutions.Count} potential solutions:");
            for (int i = 0; i < solutions.Count; i++)
            {
                Console.WriteLine($"    {i + 1}. {solutions[i].Type} - {solutions[i].Description}");
            }
        }

        private static async Task DemoVoiceCommands()
        {
            PrintSection("Voice Command System");

            using var voiceSystem = new VoiceCommandSystem();

            // Register custom handler
            voiceSystem.RegisterHandler("open browser", async (cmd) =>
            {
                await Task.Delay(10);
                Log.Success($"Opening browser as requested: {cmd.Text}");
            });

            voiceSystem.StartProcessing();

            Log.Information("Processing voice commands...");

            var commands = new[]
            {
                "open browser and navigate to example.com",
                "click on the red button at the top left",
                "find documents about machine learning",
                "remember this important note for later",
                "save the current project"
            };

            foreach (var cmd in commands)
            {
                Console.WriteLine($"\n  Command: \"{cmd}\"");
                voiceSystem.ProcessTextCommand(cmd, 0.95);
                await Task.Delay(150);
            }

            await Task.Delay(500); // Let commands process

            var stats = voiceSystem.GetStatistics();
            Console.WriteLine($"\n  Voice Command Statistics:");
            Console.WriteLine($"    Processed: {stats.TotalProcessed}");
            Console.WriteLine($"    Success Rate: {stats.SuccessRate:P0}");
            Console.WriteLine($"    Queued: {stats.QueuedCommands}");

            voiceSystem.StopProcessing();
        }

        private static void DemoComputerVision()
        {
            PrintSection("Computer Vision System");

            Log.Information("Initializing computer vision...");

            try
            {
                var vision = new AdvancedComputerVision(640, 480); // Smaller size for demo

                Log.Information("Creating test image...");
                using var testImage = new Bitmap(200, 200);
                using (var g = Graphics.FromImage(testImage))
                {
                    g.Clear(Color.White);
                    g.FillEllipse(new SolidBrush(Color.Red), 20, 20, 60, 60);
                    g.FillRectangle(new SolidBrush(Color.Blue), 120, 20, 60, 60);
                    g.FillRectangle(new SolidBrush(Color.Green), 70, 120, 60, 60);
                }

                Console.WriteLine("  Test image created (200x200)");

                Log.Information("Detecting objects...");
                var objects = vision.DetectObjects(testImage, 0.5);
                Console.WriteLine($"  Detected {objects.Count} objects:");
                foreach (var obj in objects)
                {
                    Console.WriteLine($"    - {obj.Label} at [{obj.BoundingBox.X},{obj.BoundingBox.Y}] " +
                                    $"size: {obj.BoundingBox.Width}x{obj.BoundingBox.Height} " +
                                    $"(confidence: {obj.Confidence:F2})");
                }

                Log.Information("Analyzing colors...");
                var colors = vision.AnalyzeDominantColors(testImage, 5);
                Console.WriteLine($"  Dominant colors:");
                foreach (var color in colors)
                {
                    Console.WriteLine($"    - {color.Key.Name}: {color.Value} occurrences");
                }

                Log.Information("Finding red regions...");
                var redRegions = vision.FindColorRegions(testImage, Color.Red, 30);
                Console.WriteLine($"  Found {redRegions.Count} red regions");

                Log.Success("Computer vision demo completed");
            }
            catch (Exception ex)
            {
                Log.Error(ex, "Computer vision demo failed (this is normal if GDI+ is not available)");
                Console.WriteLine("  Skipping vision demo...");
            }
        }

        private static void PrintHeader(string title)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(new string('=', 70));
            Console.WriteLine($"  {title}");
            Console.WriteLine(new string('=', 70));
            Console.ResetColor();
            Console.WriteLine();
        }

        private static void PrintSection(string section)
        {
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"▶ {section}");
            Console.WriteLine(new string('-', 70));
            Console.ResetColor();
        }
    }
}
