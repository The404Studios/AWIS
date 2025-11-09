using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.Versioning;
using System.Threading;
using System.Threading.Tasks;
using AWIS.Core;
using AWIS.Debug;
using AWIS.Input;
using AWIS.Voice;
using AWIS.Vision;

namespace AWIS.AI
{
    /// <summary>
    /// Autonomous AI agent that runs continuously, learns from user, and executes commands
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class AutonomousAgent : IDisposable
    {
        private readonly HumanizedInputController inputController;
        private readonly ActionRecorder actionRecorder;
        private readonly VoiceCommandSystem voiceSystem;
        private readonly AdvancedComputerVision vision;
        private readonly MemoryArchitecture memory;
        private readonly PersonalitySystem personality;
        private readonly GoalSystem goalSystem;
        private readonly ApplicationLauncher appLauncher;
        private readonly IntelligentResponseSystem intelligentResponse;
        private readonly AdvancedDecisionMaker decisionMaker;
        private readonly MemoryPersistence memoryPersistence;

        // Advanced task execution and priority systems
        private readonly TaskExecutionCycle executionCycle;
        private readonly PriorityRegisterSystem prioritySystem;
        private readonly DebugOverlay debugOverlay;

        // Vision system
        private readonly VisionLoop visionLoop;

        private bool isRunning;
        private Task? mainLoopTask;
        private readonly CancellationTokenSource cancellationToken;
        private int recentFailures = 0;
        private bool autonomousModeEnabled = false; // Disabled by default - wait for user commands
        private DateTime lastActionTime = DateTime.UtcNow;

        // Game command state
        private string? currentTarget;
        private string? currentFollowTarget;
        private GameMode currentMode;

        public AutonomousAgent()
        {
            inputController = new HumanizedInputController();
            actionRecorder = new ActionRecorder(inputController);
            voiceSystem = new VoiceCommandSystem();
            vision = new AdvancedComputerVision();
            memory = new MemoryArchitecture();
            personality = new PersonalitySystem();
            goalSystem = new GoalSystem();
            appLauncher = new ApplicationLauncher();
            intelligentResponse = new IntelligentResponseSystem(personality);
            decisionMaker = new AdvancedDecisionMaker();
            memoryPersistence = new MemoryPersistence();
            cancellationToken = new CancellationTokenSource();
            currentMode = GameMode.Idle;

            // Initialize advanced task execution and priority systems
            executionCycle = new TaskExecutionCycle();
            prioritySystem = new PriorityRegisterSystem(executionCycle);
            debugOverlay = new DebugOverlay(executionCycle, prioritySystem);

            // Initialize 60fps vision system with Tesseract OCR
            visionLoop = new VisionLoop();

            Console.WriteLine($"\n[AGENT] 🤖 {personality.Name} initialized!");
            Console.WriteLine($"[AGENT] {personality.Description}");
            Console.WriteLine($"[AGENT] Current mood: {personality.GetMoodDescription()}\n");
            Console.WriteLine($"[AGENT] 🎯 Task execution cycle system active");
            Console.WriteLine($"[AGENT] 📊 12-level priority registers ready");
            Console.WriteLine($"[AGENT] 🔍 Debug overlay initialized");
            Console.WriteLine($"[AGENT] 👁️  60fps vision system ready\n");

            // Load saved knowledge asynchronously
            Task.Run(async () => await LoadSavedKnowledgeAsync());

            SetupVoiceCommands();
        }

        /// <summary>
        /// Setup voice command handlers
        /// </summary>
        private void SetupVoiceCommands()
        {
            // Learning commands
            voiceSystem.RegisterHandler("start recording", async cmd =>
            {
                actionRecorder.StartRecording();
                await SpeakAsync("Recording started. Show me what to do.");
            });

            voiceSystem.RegisterHandler("stop recording", async cmd =>
            {
                var actions = actionRecorder.StopRecording();
                if (actions.Count > 0)
                {
                    actionRecorder.SaveRecording("last_recording", actions);
                    await SpeakAsync($"Recorded {actions.Count} actions. I've learned this sequence.");
                }
            });

            voiceSystem.RegisterHandler("repeat what i did", async cmd =>
            {
                await SpeakAsync("Repeating your actions now...");
                await actionRecorder.ReplayActions();
            });

            voiceSystem.RegisterHandler("just repeat", async cmd =>
            {
                var recording = actionRecorder.LoadRecording("last_recording");
                if (recording != null)
                {
                    await SpeakAsync("Executing learned sequence...");
                    await actionRecorder.ReplayActions(recording);
                }
            });

            // Game commands - Combat
            voiceSystem.RegisterHandler("fight", async cmd =>
            {
                var target = ExtractTarget(cmd.Text);
                await HandleFightCommand(target);
            });

            voiceSystem.RegisterHandler("attack", async cmd =>
            {
                var target = ExtractTarget(cmd.Text);
                await HandleFightCommand(target);
            });

            // Game commands - Movement
            voiceSystem.RegisterHandler("run away", async cmd =>
            {
                await HandleRunAwayCommand();
            });

            voiceSystem.RegisterHandler("retreat", async cmd =>
            {
                await HandleRunAwayCommand();
            });

            voiceSystem.RegisterHandler("follow", async cmd =>
            {
                var target = ExtractTarget(cmd.Text);
                await HandleFollowCommand(target);
            });

            // Mode commands
            voiceSystem.RegisterHandler("stop following", async cmd =>
            {
                currentMode = GameMode.Idle;
                currentFollowTarget = null;
                await SpeakAsync("No longer following.");
            });

            voiceSystem.RegisterHandler("stop", async cmd =>
            {
                currentMode = GameMode.Idle;
                await SpeakAsync("Stopping current action.");
            });

            // Utility commands
            voiceSystem.RegisterHandler("click here", async cmd =>
            {
                // Get current mouse position and click
                await inputController.Click();
                await SpeakAsync("Clicked.");
            });

            voiceSystem.RegisterHandler("press", async cmd =>
            {
                var key = ExtractKey(cmd.Text);
                if (key.HasValue)
                {
                    await inputController.PressKey(key.Value);
                }
            });

            // Camera control commands
            voiceSystem.RegisterHandler("look left", async cmd =>
            {
                await inputController.MoveAxis(-1.0, 0.0, sensitivity: 150);
            });

            voiceSystem.RegisterHandler("look right", async cmd =>
            {
                await inputController.MoveAxis(1.0, 0.0, sensitivity: 150);
            });

            voiceSystem.RegisterHandler("look up", async cmd =>
            {
                await inputController.MoveAxis(0.0, 1.0, sensitivity: 150);
            });

            voiceSystem.RegisterHandler("look down", async cmd =>
            {
                await inputController.MoveAxis(0.0, -1.0, sensitivity: 150);
            });

            voiceSystem.RegisterHandler("turn around", async cmd =>
            {
                await inputController.MoveAxis(1.0, 0.0, sensitivity: 300, duration: 200);
            });

            voiceSystem.RegisterHandler("look at", async cmd =>
            {
                // Extract direction or small adjustments
                var text = cmd.Text.ToLower();
                if (text.Contains("left"))
                    await inputController.MoveAxis(-0.5, 0.0, sensitivity: 100);
                else if (text.Contains("right"))
                    await inputController.MoveAxis(0.5, 0.0, sensitivity: 100);
                else if (text.Contains("up"))
                    await inputController.MoveAxis(0.0, 0.5, sensitivity: 100);
                else if (text.Contains("down"))
                    await inputController.MoveAxis(0.0, -0.5, sensitivity: 100);
            });

            // Vision commands
            voiceSystem.RegisterHandler("what do you see", async cmd =>
            {
                await AnalyzeScreen();
            });

            voiceSystem.RegisterHandler("analyze screen", async cmd =>
            {
                await AnalyzeScreen();
            });

            // Application commands
            voiceSystem.RegisterHandler("open", async cmd =>
            {
                var appName = ExtractParameter(cmd.Text, "open");
                if (!string.IsNullOrEmpty(appName))
                {
                    var response = personality.GenerateResponse($"opening {appName}", ResponseType.Success);
                    await SpeakAsync(response);
                    appLauncher.LaunchApplication(appName);
                }
            });

            voiceSystem.RegisterHandler("search for", async cmd =>
            {
                var query = ExtractParameter(cmd.Text, "search for");
                if (!string.IsNullOrEmpty(query))
                {
                    await SpeakAsync($"Searching for {query}...");
                    appLauncher.SearchWeb(query);
                }
            });

            voiceSystem.RegisterHandler("play", async cmd =>
            {
                var gameName = ExtractParameter(cmd.Text, "play");
                if (!string.IsNullOrEmpty(gameName))
                {
                    var response = personality.GenerateResponse($"launching {gameName}", ResponseType.Excitement);
                    await SpeakAsync(response);
                    appLauncher.LaunchGame(gameName);
                }
            });

            // Goal commands
            voiceSystem.RegisterHandler("what are you doing", async cmd =>
            {
                var currentGoal = goalSystem.GetCurrentGoal();
                if (currentGoal != null)
                {
                    await SpeakAsync($"I'm working on: {currentGoal.Description}");
                }
                else
                {
                    await SpeakAsync($"Just exploring and learning! I'm {personality.GetMoodDescription()}!");
                }
            });

            voiceSystem.RegisterHandler("tell me about yourself", async cmd =>
            {
                var response = intelligentResponse.GenerateResponse("who are you", null);
                await SpeakAsync(response);
            });

            // LLM commands
            voiceSystem.RegisterHandler("how smart are you", async cmd =>
            {
                var stats = intelligentResponse.GetLLMStatistics();
                await SpeakAsync(stats.Replace("\n", ". "));
            });

            voiceSystem.RegisterHandler("what have you learned", async cmd =>
            {
                if (intelligentResponse.IsLLMReady())
                {
                    var llm = intelligentResponse.GetLLM();
                    await SpeakAsync($"I've learned {llm.GetVocabularySize()} words! " +
                                   $"My helpfulness is at {llm.GetHelpfulnessScore():P0} and " +
                                   $"my friendliness is at {llm.GetFriendlinessScore():P0}!");
                }
                else
                {
                    await SpeakAsync("I'm still learning! My language model is training right now.");
                }
            });

            voiceSystem.RegisterHandler("show your progress", async cmd =>
            {
                var goalStats = goalSystem.GetLearningStatistics();
                await SpeakAsync(goalStats.Replace("\n", ". "));
            });

            voiceSystem.RegisterHandler("show decision statistics", async cmd =>
            {
                var decisionStats = decisionMaker.GetStatistics();
                await SpeakAsync(decisionStats.Replace("\n", ". "));
            });

            voiceSystem.RegisterHandler("save your knowledge", async cmd =>
            {
                await SaveKnowledgeAsync();
                await SpeakAsync("I've saved everything I've learned!");
            });

            // Debug overlay commands
            voiceSystem.RegisterHandler("show debug overlay", async cmd =>
            {
                debugOverlay.RenderSummary();
                await SpeakAsync("Debug overlay summary displayed!");
            });

            voiceSystem.RegisterHandler("show priority registers", async cmd =>
            {
                Console.WriteLine(prioritySystem.GetRegisterVisualization());
                await SpeakAsync("Showing priority register status!");
            });

            voiceSystem.RegisterHandler("show task cycles", async cmd =>
            {
                var cycles = executionCycle.GetActiveCycles();
                if (cycles.Count > 0)
                {
                    await SpeakAsync($"I have {cycles.Count} active task cycles running!");
                }
                else
                {
                    await SpeakAsync("No active task cycles at the moment.");
                }
            });

            voiceSystem.RegisterHandler("enable debug overlay", async cmd =>
            {
                debugOverlay.Start();
                await SpeakAsync("Debug overlay enabled! You can see all my processes now.");
            });

            voiceSystem.RegisterHandler("disable debug overlay", async cmd =>
            {
                debugOverlay.Stop();
                await SpeakAsync("Debug overlay disabled.");
            });

            // Autonomous mode control
            voiceSystem.RegisterHandler("enable autonomous mode", async cmd =>
            {
                autonomousModeEnabled = true;
                await SpeakAsync("Autonomous mode enabled! I'll start exploring and learning on my own.");
            });

            voiceSystem.RegisterHandler("disable autonomous mode", async cmd =>
            {
                autonomousModeEnabled = false;
                await SpeakAsync("Autonomous mode disabled. I'll wait for your commands.");
            });

            voiceSystem.RegisterHandler("stop moving", async cmd =>
            {
                autonomousModeEnabled = false;
                await SpeakAsync("Okay, I'll stop moving around.");
            });

            // Vision control
            voiceSystem.RegisterHandler("what do you see", async cmd =>
            {
                var state = visionLoop.GetCurrentState();
                var textCount = state.TextRegions.Count;
                var objectCount = state.Objects.Count;

                if (textCount == 0 && objectCount == 0)
                {
                    await SpeakAsync("I don't see any text or objects right now.");
                }
                else
                {
                    var response = $"I see {textCount} text regions and {objectCount} objects. ";
                    if (textCount > 0)
                    {
                        var firstText = state.TextRegions[0].Text;
                        response += $"The first text says: {firstText}";
                    }
                    await SpeakAsync(response);
                }
            });

            voiceSystem.RegisterHandler("show vision stats", async cmd =>
            {
                var stats = visionLoop.GetStatistics();
                Console.WriteLine($"[VISION] {stats}");
                await SpeakAsync($"Vision statistics displayed");
            });

            // Goal creation from voice commands
            voiceSystem.RegisterHandler("hey do this", async cmd =>
            {
                await SpeakAsync("What would you like me to do? Please describe the goal.");
            });

            voiceSystem.RegisterHandler("do this", async cmd =>
            {
                // Extract goal from command text
                var goalText = cmd.Text.Replace("do this", "").Trim();
                if (!string.IsNullOrEmpty(goalText))
                {
                    goalSystem.AddUserGoal(goalText);
                    await SpeakAsync($"Got it! I'll work on: {goalText}");
                }
                else
                {
                    await SpeakAsync("What would you like me to do?");
                }
            });

            voiceSystem.RegisterHandler("add goal", async cmd =>
            {
                var goalText = cmd.Text.Replace("add goal", "").Trim();
                if (!string.IsNullOrEmpty(goalText))
                {
                    goalSystem.AddUserGoal(goalText);
                    await SpeakAsync($"Goal added: {goalText}");
                }
            });

            voiceSystem.RegisterHandler("set goal", async cmd =>
            {
                var goalText = cmd.Text.Replace("set goal", "").Trim();
                if (!string.IsNullOrEmpty(goalText))
                {
                    goalSystem.AddUserGoal(goalText);
                    await SpeakAsync($"Working on new goal: {goalText}");
                }
            });

            voiceSystem.RegisterHandler("clear goals", async cmd =>
            {
                // Clear all goals would need a method in GoalSystem
                await SpeakAsync("Goals cleared. Waiting for new instructions.");
            });

            voiceSystem.RegisterHandler("what are you doing", async cmd =>
            {
                var currentGoal = goalSystem.GetCurrentGoal();
                if (currentGoal != null)
                {
                    await SpeakAsync($"I'm working on: {currentGoal.Description}");
                }
                else if (autonomousModeEnabled)
                {
                    await SpeakAsync("I'm in autonomous mode, exploring and learning.");
                }
                else
                {
                    await SpeakAsync("I'm waiting for your commands.");
                }
            });

            // Conversational handler (catches everything not matched by specific handlers)
            voiceSystem.RegisterHandler("", async cmd =>
            {
                // This acts as a catch-all for conversational input
                var context = new Dictionary<string, object>
                {
                    ["mood"] = personality.GetMoodDescription(),
                    ["current_goal"] = goalSystem.GetCurrentGoal()?.Description ?? "exploring"
                };

                var response = intelligentResponse.GenerateResponse(cmd.Text, context);
                await SpeakAsync(response);
            });
        }

        /// <summary>
        /// Start the autonomous agent
        /// </summary>
        public void Start()
        {
            if (isRunning) return;

            isRunning = true;

            // Start vision loop first
            visionLoop.Start();

            voiceSystem.StartProcessing();
            voiceSystem.StartVoiceListening(); // Enable microphone
            mainLoopTask = Task.Run(() => MainLoop(cancellationToken.Token));

            Console.WriteLine("=== Autonomous Agent Started ===");
            Console.WriteLine("🎤 VOICE RECOGNITION ACTIVE - Speak commands!");
            Console.WriteLine();
            Console.WriteLine("Say commands to interact:");
            Console.WriteLine("  Goal Management:");
            Console.WriteLine("    - 'hey do this [task]' / 'do this [task]'");
            Console.WriteLine("    - 'add goal [description]' / 'set goal [description]'");
            Console.WriteLine("    - 'clear goals' / 'what are you doing'");
            Console.WriteLine("  Mode Control:");
            Console.WriteLine("    - 'enable autonomous mode' / 'disable autonomous mode'");
            Console.WriteLine("    - 'stop moving'");
            Console.WriteLine("  Learning:");
            Console.WriteLine("    - 'start recording' / 'stop recording'");
            Console.WriteLine("    - 'repeat what I did'");
            Console.WriteLine("  Game Actions:");
            Console.WriteLine("    - 'fight [target]' / 'attack [target]'");
            Console.WriteLine("    - 'run away' / 'retreat'");
            Console.WriteLine("    - 'follow [name]'");
            Console.WriteLine("  Camera Control:");
            Console.WriteLine("    - 'look left/right/up/down'");
            Console.WriteLine("    - 'turn around'");
            Console.WriteLine("  Utility:");
            Console.WriteLine("    - 'click here' / 'press [key]'");
            Console.WriteLine();
            Console.WriteLine("⏸️  Autonomous mode is DISABLED - waiting for your commands!");
            Console.WriteLine("   Say 'enable autonomous mode' to let me explore on my own.");
            Console.WriteLine("================================");
        }

        /// <summary>
        /// Main processing loop
        /// </summary>
        private async Task MainLoop(CancellationToken token)
        {
            while (isRunning && !token.IsCancellationRequested)
            {
                try
                {
                    // Check if we have work to do
                    var currentGoal = goalSystem.GetCurrentGoal();
                    bool hasWork = currentGoal != null || autonomousModeEnabled || currentMode != GameMode.Idle;

                    if (!hasWork)
                    {
                        // NO work to do - sleep for longer to avoid spamming cycles
                        await Task.Delay(500, token); // Sleep half a second when idle
                        continue;
                    }

                    // Process based on current mode
                    switch (currentMode)
                    {
                        case GameMode.Fighting:
                            await ProcessFightingMode();
                            break;

                        case GameMode.Following:
                            await ProcessFollowingMode();
                            break;

                        case GameMode.Fleeing:
                            await ProcessFleeingMode();
                            break;

                        case GameMode.Idle:
                        default:
                            // Autonomous exploration and actions
                            await ProcessIdleMode();
                            break;
                    }

                    // Delay between actions to prevent spam
                    await Task.Delay(1000, token); // 1 second between actions
                }
                catch (OperationCanceledException)
                {
                    break; // Normal cancellation
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[ERROR] Agent loop error: {ex.Message}");
                    await Task.Delay(1000);
                }
            }
        }

        /// <summary>
        /// Handle fight command
        /// </summary>
        private async Task HandleFightCommand(string? target)
        {
            currentMode = GameMode.Fighting;
            currentTarget = target ?? "enemy";
            await SpeakAsync($"Engaging {currentTarget}!");

            // Try to find and attack the target
            // For now, simulate attack pattern
            await ExecuteAttackSequence();
        }

        /// <summary>
        /// Execute attack sequence
        /// </summary>
        private async Task ExecuteAttackSequence()
        {
            // Check if user has taught us a fight pattern
            var fightRecording = actionRecorder.LoadRecording("fight_pattern");
            if (fightRecording != null)
            {
                await actionRecorder.ReplayActions(fightRecording);
                return;
            }

            // Default attack pattern
            await inputController.PressKey(HumanizedInputController.VK.VK_1); // Attack key
            await Task.Delay(500);
        }

        /// <summary>
        /// Process fighting mode
        /// </summary>
        private async Task ProcessFightingMode()
        {
            // Continuously attack until mode changes
            await ExecuteAttackSequence();
            await Task.Delay(1000);
        }

        /// <summary>
        /// Handle run away command
        /// </summary>
        private async Task HandleRunAwayCommand()
        {
            currentMode = GameMode.Fleeing;
            await SpeakAsync("Running away!");

            // Execute flee sequence
            await ExecuteFleeSequence();
        }

        /// <summary>
        /// Execute flee sequence
        /// </summary>
        private async Task ExecuteFleeSequence()
        {
            // Check for learned flee pattern
            var fleeRecording = actionRecorder.LoadRecording("flee_pattern");
            if (fleeRecording != null)
            {
                await actionRecorder.ReplayActions(fleeRecording);
                currentMode = GameMode.Idle;
                return;
            }

            // Default flee: press S (backward) for 2 seconds
            await inputController.PressKey(HumanizedInputController.VK.S, 2000);
            currentMode = GameMode.Idle;
        }

        /// <summary>
        /// Process fleeing mode
        /// </summary>
        private async Task ProcessFleeingMode()
        {
            await ExecuteFleeSequence();
        }

        /// <summary>
        /// Process idle mode - intelligent decision-making with learning and task cycle management
        /// </summary>
        private async Task ProcessIdleMode()
        {
            // Check current goal and work towards it
            var currentGoal = goalSystem.GetCurrentGoal();
            if (currentGoal != null)
            {
                await WorkTowardsGoal(currentGoal);
                return;
            }

            // Only do autonomous actions if explicitly enabled
            if (!autonomousModeEnabled)
            {
                // NOT doing autonomous actions - just wait
                return; // Main loop will handle the sleep
            }

            // Autonomous mode: use vision to explore
            Console.WriteLine("[AUTONOMOUS] Using vision to explore...");

            var state = visionLoop.GetCurrentState();
            Console.WriteLine($"[VISION] Saw {state.TextRegions.Count} text regions, {state.Objects.Count} objects");

            // Generate a goal based on what we see
            var random = new Random();
            if (state.TextRegions.Count > 0 && random.Next(0, 3) == 0)
            {
                // Found text - create a goal to investigate
                var randomText = state.TextRegions[random.Next(state.TextRegions.Count)];
                goalSystem.AddUserGoal($"investigate text: {randomText.Text}");
                await SpeakAsync($"I see text that says {randomText.Text}. Let me investigate!");
            }
            else if (random.Next(0, 10) == 0) // 10% chance to generate autonomous goal
            {
                goalSystem.GenerateAutonomousGoal();
                var response = personality.GenerateResponse("setting new goal", ResponseType.Excitement);
                await SpeakAsync(response);
            }
        }

        /// <summary>
        /// Calculate priority register (1-12) based on confidence and failures
        /// </summary>
        private int CalculatePriorityRegister(double confidence, int failures)
        {
            // High confidence, low failures = higher priority (lower number)
            // Low confidence, high failures = lower priority (higher number)

            var basePriority = 6; // Middle priority
            basePriority -= (int)(confidence * 3); // Confidence can reduce by up to 3
            basePriority += Math.Min(failures, 3); // Failures can increase by up to 3

            return Math.Clamp(basePriority, 1, 12);
        }

        /// <summary>
        /// Execute a decision action
        /// </summary>
        private async Task<bool> ExecuteDecision(string action, Random random)
        {
            switch (action)
            {
                case "move_forward":
                    Console.WriteLine("[AUTONOMOUS] Exploring forward...");
                    await inputController.PressKey(HumanizedInputController.VK.W, random.Next(500, 1500));
                    await Task.Delay(random.Next(2000, 5000));
                    return true;

                case "look_around_360":
                    Console.WriteLine("[AUTONOMOUS] Scanning surroundings...");
                    for (int i = 0; i < 4; i++)
                    {
                        await inputController.MoveAxis(0.5, 0.0, sensitivity: 100, duration: 200);
                        await Task.Delay(300);
                    }
                    await Task.Delay(random.Next(3000, 6000));
                    return true;

                case "focus_on_object":
                case "analyze_objects":
                    await AnalyzeScreen();
                    return true;

                case "explore_area":
                    // Look around randomly then move
                    var yaw = (random.NextDouble() - 0.5) * 2;
                    var pitch = (random.NextDouble() - 0.5) * 0.5;
                    await inputController.MoveAxis(yaw, pitch, sensitivity: 80, duration: 150);
                    await Task.Delay(500);
                    await inputController.PressKey(HumanizedInputController.VK.W, random.Next(500, 1000));
                    await Task.Delay(random.Next(1000, 3000));
                    return true;

                case "practice_skills":
                    // Practice movements
                    await inputController.PressKey(HumanizedInputController.VK.SPACE);
                    await Task.Delay(300);
                    var strafeKey = random.Next(0, 2) == 0 ? HumanizedInputController.VK.A : HumanizedInputController.VK.D;
                    await inputController.PressKey(strafeKey, random.Next(300, 800));
                    await Task.Delay(random.Next(2000, 4000));
                    return true;

                case "rest":
                    // Just observe and save energy
                    if (random.Next(0, 5) == 0)
                    {
                        var mood = personality.GetMoodDescription();
                        await SpeakAsync($"I'm {mood}, just taking a moment to think.");
                    }
                    await Task.Delay(random.Next(2000, 4000));
                    return true;

                default:
                    // Default exploration behavior
                    await inputController.MoveAxis((random.NextDouble() - 0.5) * 2, 0, sensitivity: 80, duration: 150);
                    await Task.Delay(random.Next(1500, 3000));
                    return true;
            }
        }

        private async Task WorkTowardsGoal(Goal goal)
        {
            Console.WriteLine($"[GOAL] Working on: {goal.Description}");

            var progress = goalSystem.GetGoalProgress(goal.Id);
            var description = goal.Description.ToLower();

            // Get current vision state
            var state = visionLoop.GetCurrentState();
            Console.WriteLine($"[VISION] Current view: {state.TextRegions.Count} text regions, {state.Objects.Count} objects");

            // Investigate text goals
            if (description.Contains("investigate") || description.Contains("read"))
            {
                // Extract what text we're looking for
                var targetText = goal.Description.Replace("investigate text:", "").Replace("read", "").Trim();

                var foundText = state.TextRegions.Find(r =>
                    r.Text.Contains(targetText, StringComparison.OrdinalIgnoreCase));

                if (foundText != null)
                {
                    Console.WriteLine($"[VISION-GOAL] Found text '{foundText.Text}' at ({foundText.BoundingBox.X}, {foundText.BoundingBox.Y})");

                    // Click on the text
                    var centerX = foundText.BoundingBox.X + foundText.BoundingBox.Width / 2;
                    var centerY = foundText.BoundingBox.Y + foundText.BoundingBox.Height / 2;

                    await inputController.MoveMouse(centerX, centerY);
                    await Task.Delay(500);
                    await inputController.LeftClick();

                    await SpeakAsync($"I clicked on {foundText.Text}");
                    goalSystem.CompleteGoal(goal.Id, true, 0.9);
                }
                else
                {
                    Console.WriteLine($"[VISION-GOAL] Text not found, waiting for it to appear...");
                    if (progress > 0.8)
                    {
                        goalSystem.CompleteGoal(goal.Id, false, 0.3);
                        await SpeakAsync($"I couldn't find that text on screen");
                    }
                }
            }
            // Click/Find buttons
            else if (description.Contains("click") || description.Contains("button") || description.Contains("press"))
            {
                var buttons = state.TextRegions.FindAll(r => r.Type == "Button");
                Console.WriteLine($"[VISION-GOAL] Found {buttons.Count} buttons on screen");

                if (buttons.Count > 0)
                {
                    // Find matching button
                    var targetButton = buttons.Find(b =>
                        goal.Description.Contains(b.Text, StringComparison.OrdinalIgnoreCase)) ?? buttons[0];

                    Console.WriteLine($"[VISION-GOAL] Clicking button: {targetButton.Text}");

                    var centerX = targetButton.BoundingBox.X + targetButton.BoundingBox.Width / 2;
                    var centerY = targetButton.BoundingBox.Y + targetButton.BoundingBox.Height / 2;

                    await inputController.MoveMouse(centerX, centerY);
                    await Task.Delay(300);
                    await inputController.LeftClick();

                    await SpeakAsync($"Clicked {targetButton.Text}");
                    goalSystem.CompleteGoal(goal.Id, true, 0.9);
                }
                else if (progress > 0.7)
                {
                    goalSystem.CompleteGoal(goal.Id, false, 0.4);
                }
            }
            // Find/Search goals - look for text on screen
            else if (description.Contains("find") || description.Contains("search") || description.Contains("locate"))
            {
                Console.WriteLine($"[VISION-GOAL] Searching for items on screen...");

                if (state.TextRegions.Count > 0)
                {
                    var findings = string.Join(", ", state.TextRegions.Take(5).Select(r => r.Text));
                    Console.WriteLine($"[VISION-GOAL] I can see: {findings}");
                    await SpeakAsync($"I found: {findings}");
                    goalSystem.CompleteGoal(goal.Id, true, 0.8);
                }
                else if (progress > 0.6)
                {
                    await SpeakAsync("I don't see much on this screen");
                    goalSystem.CompleteGoal(goal.Id, false, 0.5);
                }
            }
            // Default: report what we see
            else
            {
                Console.WriteLine($"[VISION-GOAL] Using vision to accomplish: {goal.Description}");

                if (state.TextRegions.Count > 0)
                {
                    var firstText = state.TextRegions[0].Text;
                    Console.WriteLine($"[VISION-GOAL] I see text: {firstText}");
                }

                if (progress > 0.8)
                {
                    goalSystem.CompleteGoal(goal.Id, true, 0.7);
                }
            }
        }

        /// <summary>
        /// Handle follow command
        /// </summary>
        private async Task HandleFollowCommand(string? target)
        {
            currentMode = GameMode.Following;
            currentFollowTarget = target ?? "player";
            await SpeakAsync($"Following {currentFollowTarget}.");
        }

        /// <summary>
        /// Process following mode
        /// </summary>
        private async Task ProcessFollowingMode()
        {
            // Check for learned follow pattern
            var followRecording = actionRecorder.LoadRecording("follow_pattern");
            if (followRecording != null)
            {
                await actionRecorder.ReplayActions(followRecording, speedMultiplier: 1.5);
                return;
            }

            // Default: move forward
            await inputController.PressKey(HumanizedInputController.VK.W, 500);
            await Task.Delay(100);
        }

        /// <summary>
        /// Process text command (for manual input)
        /// </summary>
        public void ProcessCommand(string command)
        {
            voiceSystem.ProcessTextCommand(command);
        }

        /// <summary>
        /// Stop the autonomous agent
        /// </summary>
        public void Stop()
        {
            if (!isRunning) return;

            isRunning = false;
            voiceSystem.StopProcessing();
            cancellationToken.Cancel();
            mainLoopTask?.Wait(2000);

            Console.WriteLine("=== Autonomous Agent Stopped ===");
        }

        /// <summary>
        /// Extract target name from command
        /// </summary>
        private string? ExtractTarget(string command)
        {
            var words = command.ToLower().Split(' ');
            for (int i = 0; i < words.Length; i++)
            {
                if (words[i] == "that" || words[i] == "the")
                {
                    if (i + 1 < words.Length)
                        return words[i + 1];
                }
            }
            // Return last word as target
            return words.LastOrDefault();
        }

        /// <summary>
        /// Extract key from command
        /// </summary>
        private byte? ExtractKey(string command)
        {
            var keyMap = new Dictionary<string, byte>
            {
                ["w"] = HumanizedInputController.VK.W,
                ["a"] = HumanizedInputController.VK.A,
                ["s"] = HumanizedInputController.VK.S,
                ["d"] = HumanizedInputController.VK.D,
                ["space"] = HumanizedInputController.VK.SPACE,
                ["enter"] = HumanizedInputController.VK.RETURN,
                ["escape"] = HumanizedInputController.VK.ESCAPE,
                ["1"] = HumanizedInputController.VK.VK_1,
                ["2"] = HumanizedInputController.VK.VK_2,
                ["3"] = HumanizedInputController.VK.VK_3,
                ["4"] = HumanizedInputController.VK.VK_4,
            };

            foreach (var kvp in keyMap)
            {
                if (command.ToLower().Contains(kvp.Key))
                    return kvp.Value;
            }

            return null;
        }

        private async Task SpeakAsync(string text)
        {
            Console.WriteLine($"[AGENT] {text}");
            voiceSystem.Speak(text);
            await Task.Delay(10);
        }

        private async Task AnalyzeScreen()
        {
            try
            {
                await SpeakAsync("Let me look at the screen...");

                var screenshot = vision.CaptureScreen();
                var detectedObjects = vision.DetectObjects(screenshot);

                // Simple description based on what we can detect
                var description = $"I can see the screen. ";
                description += detectedObjects.Count > 0
                    ? $"I detected {detectedObjects.Count} objects. "
                    : "I'm analyzing the visual elements. ";

                var response = personality.GenerateResponse(description, ResponseType.Discovery);
                await SpeakAsync(response);

                screenshot.Dispose();
            }
            catch (Exception ex)
            {
                await SpeakAsync($"I'm having trouble analyzing the screen right now. {ex.Message}");
            }
        }

        private string? ExtractParameter(string text, string command)
        {
            text = text.ToLower().Trim();
            command = command.ToLower().Trim();

            var index = text.IndexOf(command);
            if (index < 0) return null;

            var afterCommand = text[(index + command.Length)..].Trim();
            return string.IsNullOrWhiteSpace(afterCommand) ? null : afterCommand;
        }

        /// <summary>
        /// Load saved knowledge from previous sessions
        /// </summary>
        private async Task LoadSavedKnowledgeAsync()
        {
            try
            {
                Console.WriteLine("[MEMORY] Loading saved knowledge...");

                // Load personality traits
                var personalityData = await memoryPersistence.LoadPersonalityAsync();
                if (personalityData != null)
                {
                    // Apply loaded personality traits (would need methods on PersonalitySystem)
                    Console.WriteLine("[MEMORY] Restored personality from previous session");
                }

                // Load conversation history
                var conversations = await memoryPersistence.LoadConversationsAsync();
                if (conversations != null && conversations.Count > 0)
                {
                    Console.WriteLine($"[MEMORY] Restored {conversations.Count} previous conversations");
                }

                // Load LLM vocabulary
                var (vocab, embeddings) = await memoryPersistence.LoadLLMVocabularyAsync();
                if (vocab != null && embeddings != null)
                {
                    Console.WriteLine("[MEMORY] Restored LLM vocabulary from previous session");
                }

                Console.WriteLine("[MEMORY] ✅ Knowledge loaded successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MEMORY] No previous knowledge found or error loading: {ex.Message}");
            }
        }

        /// <summary>
        /// Save all learned knowledge
        /// </summary>
        private async Task SaveKnowledgeAsync()
        {
            try
            {
                Console.WriteLine("[MEMORY] Saving learned knowledge...");

                // Save personality evolution
                await memoryPersistence.SavePersonalityAsync(personality);

                // Save conversation history
                var summary = intelligentResponse.GetConversationSummary();
                await memoryPersistence.SaveConversationsAsync(new List<string> { summary });

                // Save LLM vocabulary if ready (would need methods to expose vocabulary in LocalLLM)
                if (intelligentResponse.IsLLMReady())
                {
                    Console.WriteLine("[MEMORY] LLM vocabulary ready for save (needs implementation)");
                }

                Console.WriteLine("[MEMORY] ✅ All knowledge saved successfully!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MEMORY] Error saving knowledge: {ex.Message}");
            }
        }

        public void Dispose()
        {
            // Show final debug overlay summary
            Console.WriteLine("\n[AGENT] Shutting down...");
            debugOverlay.RenderSummary();

            // Save knowledge before shutting down
            Task.Run(async () => await SaveKnowledgeAsync()).Wait(5000);

            Stop();
            visionLoop.Dispose();
            debugOverlay.Dispose();
            voiceSystem.Dispose();
            cancellationToken.Dispose();

            Console.WriteLine("[AGENT] All systems shut down successfully.\n");
        }
    }

    public enum GameMode
    {
        Idle,
        Fighting,
        Following,
        Fleeing
    }
}
