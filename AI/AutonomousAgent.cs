using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.Versioning;
using System.Threading;
using System.Threading.Tasks;
using AWIS.Core;
using AWIS.Input;
using AWIS.Voice;
using AWIS.Vision;

namespace AWIS.AI
{
    /// <summary>
    /// Autonomous AI agent that runs continuously, learns from user, and executes commands
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class AutonomousAgent
    {
        private readonly HumanizedInputController inputController;
        private readonly ActionRecorder actionRecorder;
        private readonly VoiceCommandSystem voiceSystem;
        private readonly AdvancedComputerVision vision;
        private readonly MemoryArchitecture memory;

        private bool isRunning;
        private Task? mainLoopTask;
        private readonly CancellationTokenSource cancellationToken;

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
            cancellationToken = new CancellationTokenSource();
            currentMode = GameMode.Idle;

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
                if (actions.Any())
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
        }

        /// <summary>
        /// Start the autonomous agent
        /// </summary>
        public void Start()
        {
            if (isRunning) return;

            isRunning = true;
            voiceSystem.StartProcessing();
            mainLoopTask = Task.Run(() => MainLoop(cancellationToken.Token));

            Console.WriteLine("=== Autonomous Agent Started ===");
            Console.WriteLine("Say commands to interact:");
            Console.WriteLine("  - 'start recording' / 'stop recording'");
            Console.WriteLine("  - 'repeat what I did'");
            Console.WriteLine("  - 'fight [target]'");
            Console.WriteLine("  - 'run away'");
            Console.WriteLine("  - 'follow [name]'");
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
                            // Just monitor and wait for commands
                            await Task.Delay(100);
                            break;
                    }

                    // Small delay to prevent CPU overuse
                    await Task.Delay(50);
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

        public void Dispose()
        {
            Stop();
            voiceSystem.Dispose();
            cancellationToken.Dispose();
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
