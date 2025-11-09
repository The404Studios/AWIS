using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AWIS.Core;

namespace AWIS.Voice
{
    /// <summary>
    /// A voice command recognized from speech
    /// </summary>
    public class VoiceCommand
    {
        public string Text { get; set; } = string.Empty;
        public double Confidence { get; set; }
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public Dictionary<string, string> Parameters { get; set; } = new();
        public ActionType? DetectedAction { get; set; }

        public VoiceCommand(string text, double confidence = 1.0)
        {
            Text = text;
            Confidence = confidence;
        }
    }

    /// <summary>
    /// Delegate for voice command handlers
    /// </summary>
    public delegate Task VoiceCommandHandler(VoiceCommand command);

    /// <summary>
    /// Voice command system for processing natural language commands
    /// </summary>
    public class VoiceCommandSystem : IDisposable
    {
        private readonly ConcurrentQueue<VoiceCommand> commandQueue = new();
        private readonly Dictionary<string, VoiceCommandHandler> commandHandlers = new();
        private readonly Dictionary<string, ActionType> commandMappings = new();
        private bool isProcessing = false;
        private Thread? processingThread;
        private readonly CancellationTokenSource cancellationToken = new();

        // Statistics
        private int totalCommandsProcessed = 0;
        private int totalCommandsSucceeded = 0;
        private int totalCommandsFailed = 0;

        public VoiceCommandSystem()
        {
            InitializeCommandMappings();
        }

        private void InitializeCommandMappings()
        {
            // Navigation commands
            commandMappings["go to"] = ActionType.Navigate;
            commandMappings["navigate to"] = ActionType.Navigate;
            commandMappings["open"] = ActionType.Navigate;
            commandMappings["visit"] = ActionType.Visit;

            // Interaction commands
            commandMappings["click"] = ActionType.Click;
            commandMappings["double click"] = ActionType.DoubleClick;
            commandMappings["right click"] = ActionType.RightClick;
            commandMappings["press"] = ActionType.Press;
            commandMappings["type"] = ActionType.Type;
            commandMappings["select"] = ActionType.Select;

            // Control commands
            commandMappings["start"] = ActionType.Start;
            commandMappings["stop"] = ActionType.Stop;
            commandMappings["pause"] = ActionType.Pause;
            commandMappings["resume"] = ActionType.Resume;
            commandMappings["quit"] = ActionType.Quit;
            commandMappings["exit"] = ActionType.Exit;
            commandMappings["close"] = ActionType.Close;

            // Query commands
            commandMappings["search"] = ActionType.Search;
            commandMappings["find"] = ActionType.Find;
            commandMappings["identify"] = ActionType.Identify;
            commandMappings["detect"] = ActionType.Detect;

            // Learning commands
            commandMappings["learn"] = ActionType.Learn;
            commandMappings["remember"] = ActionType.Remember;
            commandMappings["observe"] = ActionType.Observe;

            // Communication commands
            commandMappings["say"] = ActionType.Say;
            commandMappings["speak"] = ActionType.Speak;
            commandMappings["reply"] = ActionType.Reply;
            commandMappings["message"] = ActionType.Message;

            // Emergency commands
            commandMappings["abort"] = ActionType.Abort;
            commandMappings["cancel"] = ActionType.Cancel;
            commandMappings["help"] = ActionType.Help;
        }

        /// <summary>
        /// Registers a command handler
        /// </summary>
        public void RegisterHandler(string commandPattern, VoiceCommandHandler handler)
        {
            commandHandlers[commandPattern.ToLower()] = handler;
        }

        /// <summary>
        /// Processes a text command (simulating voice recognition)
        /// </summary>
        public void ProcessTextCommand(string text, double confidence = 1.0)
        {
            var command = new VoiceCommand(text, confidence);
            ParseCommand(command);
            commandQueue.Enqueue(command);
        }

        /// <summary>
        /// Starts processing commands
        /// </summary>
        public void StartProcessing()
        {
            if (isProcessing) return;

            isProcessing = true;
            processingThread = new Thread(ProcessCommandQueueLoop)
            {
                IsBackground = true,
                Name = "VoiceCommandProcessor"
            };
            processingThread.Start();
        }

        /// <summary>
        /// Stops processing commands
        /// </summary>
        public void StopProcessing()
        {
            isProcessing = false;
            cancellationToken.Cancel();
            processingThread?.Join(1000);
        }

        private void ProcessCommandQueueLoop()
        {
            while (isProcessing && !cancellationToken.Token.IsCancellationRequested)
            {
                if (commandQueue.TryDequeue(out var command))
                {
                    Task.Run(async () => await ProcessCommand(command)).Wait();
                }
                else
                {
                    Thread.Sleep(100);
                }
            }
        }

        private async Task ProcessCommand(VoiceCommand command)
        {
            totalCommandsProcessed++;

            try
            {
                // Find matching handler
                VoiceCommandHandler? handler = null;

                foreach (var pattern in commandHandlers.Keys)
                {
                    if (command.Text.ToLower().Contains(pattern))
                    {
                        handler = commandHandlers[pattern];
                        break;
                    }
                }

                if (handler != null)
                {
                    await handler(command);
                    totalCommandsSucceeded++;
                }
                else
                {
                    // No specific handler, use default processing
                    await DefaultCommandHandler(command);
                    totalCommandsSucceeded++;
                }
            }
            catch (Exception ex)
            {
                totalCommandsFailed++;
                Console.WriteLine($"Error processing command: {ex.Message}");
            }
        }

        private async Task DefaultCommandHandler(VoiceCommand command)
        {
            await Task.Delay(10); // Simulate processing

            if (command.DetectedAction.HasValue)
            {
                Console.WriteLine($"Executing {command.DetectedAction.Value}: {command.Text}");
            }
            else
            {
                Console.WriteLine($"Processed command: {command.Text}");
            }
        }

        private void ParseCommand(VoiceCommand command)
        {
            var text = command.Text.ToLower();

            // Detect action type
            foreach (var mapping in commandMappings)
            {
                if (text.Contains(mapping.Key))
                {
                    command.DetectedAction = mapping.Value;

                    // Extract parameters
                    var parts = text.Split(new[] { mapping.Key }, StringSplitOptions.None);
                    if (parts.Length > 1)
                    {
                        command.Parameters["target"] = parts[1].Trim();
                    }
                    break;
                }
            }

            // Extract common parameter patterns
            ExtractParameters(command);
        }

        private void ExtractParameters(VoiceCommand command)
        {
            var text = command.Text.ToLower();

            // Extract coordinates if present
            if (text.Contains("at") || text.Contains("position"))
            {
                // Try to extract coordinates
                var words = text.Split(' ');
                for (int i = 0; i < words.Length - 1; i++)
                {
                    if (int.TryParse(words[i], out var x) && int.TryParse(words[i + 1], out var y))
                    {
                        command.Parameters["x"] = x.ToString();
                        command.Parameters["y"] = y.ToString();
                    }
                }
            }

            // Extract quoted strings as parameters
            var startQuote = text.IndexOf('"');
            var endQuote = text.LastIndexOf('"');
            if (startQuote >= 0 && endQuote > startQuote)
            {
                command.Parameters["quoted_text"] = text[(startQuote + 1)..endQuote];
            }

            // Extract color references
            var colors = new[] { "red", "blue", "green", "yellow", "black", "white", "orange", "purple" };
            foreach (var color in colors)
            {
                if (text.Contains(color))
                {
                    command.Parameters["color"] = color;
                    break;
                }
            }

            // Extract spatial references
            var spatialRefs = new[] { "left", "right", "top", "bottom", "center", "middle", "corner" };
            foreach (var spatial in spatialRefs)
            {
                if (text.Contains(spatial))
                {
                    command.Parameters["spatial"] = spatial;
                    break;
                }
            }
        }

        /// <summary>
        /// Speaks text (simulated)
        /// </summary>
        public void Speak(string text)
        {
            Console.WriteLine($"[Voice Output]: {text}");
        }

        /// <summary>
        /// Gets statistics about command processing
        /// </summary>
        public VoiceCommandStatistics GetStatistics()
        {
            return new VoiceCommandStatistics
            {
                TotalProcessed = totalCommandsProcessed,
                TotalSucceeded = totalCommandsSucceeded,
                TotalFailed = totalCommandsFailed,
                SuccessRate = totalCommandsProcessed > 0 ?
                    totalCommandsSucceeded / (double)totalCommandsProcessed : 0,
                QueuedCommands = commandQueue.Count
            };
        }

        public void Dispose()
        {
            StopProcessing();
            cancellationToken.Dispose();
            GC.SuppressFinalize(this);
        }
    }

    public class VoiceCommandStatistics
    {
        public int TotalProcessed { get; set; }
        public int TotalSucceeded { get; set; }
        public int TotalFailed { get; set; }
        public double SuccessRate { get; set; }
        public int QueuedCommands { get; set; }
    }

    /// <summary>
    /// Simulated speech synthesizer
    /// </summary>
    public class SpeechSynthesizer
    {
        public int Volume { get; set; } = 100;
        public int Rate { get; set; } = 0;

        public void Speak(string text)
        {
            Console.WriteLine($"[TTS Volume:{Volume} Rate:{Rate}]: {text}");
        }

        public void SpeakAsync(string text)
        {
            Task.Run(() => Speak(text));
        }
    }
}
