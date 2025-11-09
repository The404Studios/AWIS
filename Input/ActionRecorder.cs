using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.Versioning;
using System.Threading.Tasks;
using AWIS.Core;

namespace AWIS.Input
{
    /// <summary>
    /// Records and replays user actions for learning
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class ActionRecorder
    {
        private readonly List<RecordedAction> recordedActions;
        private readonly HumanizedInputController inputController;
        private bool isRecording;
        private DateTime recordingStartTime;

        public ActionRecorder(HumanizedInputController inputController)
        {
            this.inputController = inputController;
            recordedActions = new List<RecordedAction>();
            isRecording = false;
        }

        public bool IsRecording => isRecording;
        public int ActionCount => recordedActions.Count;

        /// <summary>
        /// Start recording user actions
        /// </summary>
        public void StartRecording()
        {
            recordedActions.Clear();
            isRecording = true;
            recordingStartTime = DateTime.Now;
            Console.WriteLine("[RECORDER] Started recording actions...");
        }

        /// <summary>
        /// Stop recording actions
        /// </summary>
        public List<RecordedAction> StopRecording()
        {
            isRecording = false;
            Console.WriteLine($"[RECORDER] Stopped recording. Captured {recordedActions.Count} actions.");
            return new List<RecordedAction>(recordedActions);
        }

        /// <summary>
        /// Record a mouse movement
        /// </summary>
        public void RecordMouseMove(int x, int y)
        {
            if (!isRecording) return;

            recordedActions.Add(new RecordedAction
            {
                Type = ActionRecordType.MouseMove,
                X = x,
                Y = y,
                Timestamp = DateTime.Now,
                TimeSinceStart = DateTime.Now - recordingStartTime
            });
        }

        /// <summary>
        /// Record a mouse click
        /// </summary>
        public void RecordMouseClick(MouseButton button, int x, int y)
        {
            if (!isRecording) return;

            recordedActions.Add(new RecordedAction
            {
                Type = ActionRecordType.MouseClick,
                MouseButton = button,
                X = x,
                Y = y,
                Timestamp = DateTime.Now,
                TimeSinceStart = DateTime.Now - recordingStartTime
            });
        }

        /// <summary>
        /// Record a key press
        /// </summary>
        public void RecordKeyPress(byte key)
        {
            if (!isRecording) return;

            recordedActions.Add(new RecordedAction
            {
                Type = ActionRecordType.KeyPress,
                Key = key,
                Timestamp = DateTime.Now,
                TimeSinceStart = DateTime.Now - recordingStartTime
            });
        }

        /// <summary>
        /// Record typed text
        /// </summary>
        public void RecordText(string text)
        {
            if (!isRecording) return;

            recordedActions.Add(new RecordedAction
            {
                Type = ActionRecordType.TypeText,
                Text = text,
                Timestamp = DateTime.Now,
                TimeSinceStart = DateTime.Now - recordingStartTime
            });
        }

        /// <summary>
        /// Replay recorded actions
        /// </summary>
        public async Task ReplayActions(List<RecordedAction>? actions = null, double speedMultiplier = 1.0)
        {
            var actionsToReplay = actions ?? recordedActions;
            if (!actionsToReplay.Any())
            {
                Console.WriteLine("[RECORDER] No actions to replay.");
                return;
            }

            Console.WriteLine($"[RECORDER] Replaying {actionsToReplay.Count} actions at {speedMultiplier}x speed...");

            DateTime lastActionTime = actionsToReplay[0].Timestamp;

            foreach (var action in actionsToReplay)
            {
                // Wait for the time delay between actions
                var delay = (action.Timestamp - lastActionTime).TotalMilliseconds / speedMultiplier;
                if (delay > 0)
                {
                    await Task.Delay((int)delay);
                }

                // Execute the action
                await ExecuteAction(action);
                lastActionTime = action.Timestamp;
            }

            Console.WriteLine("[RECORDER] Replay complete.");
        }

        /// <summary>
        /// Execute a single recorded action
        /// </summary>
        private async Task ExecuteAction(RecordedAction action)
        {
            switch (action.Type)
            {
                case ActionRecordType.MouseMove:
                    if (action.X.HasValue && action.Y.HasValue)
                    {
                        await inputController.MoveMouse(action.X.Value, action.Y.Value);
                    }
                    break;

                case ActionRecordType.MouseClick:
                    if (action.X.HasValue && action.Y.HasValue)
                    {
                        await inputController.MoveMouse(action.X.Value, action.Y.Value);
                        await inputController.Click(action.MouseButton);
                    }
                    break;

                case ActionRecordType.KeyPress:
                    if (action.Key.HasValue)
                    {
                        await inputController.PressKey(action.Key.Value);
                    }
                    break;

                case ActionRecordType.TypeText:
                    if (!string.IsNullOrEmpty(action.Text))
                    {
                        await inputController.TypeText(action.Text);
                    }
                    break;
            }
        }

        /// <summary>
        /// Save recorded actions with a name for later recall
        /// </summary>
        public void SaveRecording(string name, List<RecordedAction>? actions = null)
        {
            var actionsToSave = actions ?? recordedActions;
            SavedRecordings[name] = new List<RecordedAction>(actionsToSave);
            Console.WriteLine($"[RECORDER] Saved recording '{name}' with {actionsToSave.Count} actions.");
        }

        /// <summary>
        /// Load previously saved recording
        /// </summary>
        public List<RecordedAction>? LoadRecording(string name)
        {
            if (SavedRecordings.TryGetValue(name, out var recording))
            {
                Console.WriteLine($"[RECORDER] Loaded recording '{name}' with {recording.Count} actions.");
                return new List<RecordedAction>(recording);
            }

            Console.WriteLine($"[RECORDER] Recording '{name}' not found.");
            return null;
        }

        public Dictionary<string, List<RecordedAction>> SavedRecordings { get; } = new();
    }

    public class RecordedAction
    {
        public ActionRecordType Type { get; set; }
        public int? X { get; set; }
        public int? Y { get; set; }
        public MouseButton MouseButton { get; set; }
        public byte? Key { get; set; }
        public string? Text { get; set; }
        public DateTime Timestamp { get; set; }
        public TimeSpan TimeSinceStart { get; set; }
    }

    public enum ActionRecordType
    {
        MouseMove,
        MouseClick,
        KeyPress,
        TypeText
    }
}
