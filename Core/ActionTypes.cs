using System;

namespace AWIS.Core
{
    /// <summary>
    /// Defines all action types that the AI can perform
    /// </summary>
    public enum ActionType
    {
        // Navigation
        Navigate,
        GoTo,
        Visit,

        // UI Interaction
        Click,
        DoubleClick,
        RightClick,
        Press,
        Type,
        Select,
        Drag,
        Drop,
        Scroll,
        Hover,
        Move,

        // System Control
        Start,
        Stop,
        Pause,
        Resume,
        Restart,
        Quit,
        Exit,
        Close,
        Minimize,
        Maximize,

        // Query and Analysis
        Search,
        Find,
        Identify,
        Recognize,
        Detect,
        Analyze,

        // Learning
        Learn,
        Remember,
        Observe,
        Track,
        Record,

        // Communication
        Say,
        Speak,
        Reply,
        Respond,
        Message,

        // Data Management
        Save,
        Load,
        Export,
        Import,
        Backup,
        Restore,

        // Configuration
        Configure,
        Monitor,

        // Emergency
        Abort,
        Cancel,
        Undo,
        Help
    }

    /// <summary>
    /// Represents an executable action with context
    /// </summary>
    public class AIAction
    {
        public ActionType Type { get; set; }
        public string Description { get; set; } = string.Empty;
        public Dictionary<string, object> Parameters { get; set; } = new();
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public double Confidence { get; set; } = 1.0;

        public AIAction(ActionType type, string description = "")
        {
            Type = type;
            Description = description;
        }

        public void AddParameter(string key, object value)
        {
            Parameters[key] = value;
        }

        public T? GetParameter<T>(string key)
        {
            if (Parameters.TryGetValue(key, out var value) && value is T typedValue)
            {
                return typedValue;
            }
            return default;
        }
    }

    /// <summary>
    /// Result of an action execution
    /// </summary>
    public class ActionResult
    {
        public bool Success { get; set; }
        public string Message { get; set; } = string.Empty;
        public object? Data { get; set; }
        public TimeSpan ExecutionTime { get; set; }
        public Exception? Error { get; set; }

        public static ActionResult SuccessResult(string message = "Action completed successfully", object? data = null)
        {
            return new ActionResult
            {
                Success = true,
                Message = message,
                Data = data
            };
        }

        public static ActionResult FailureResult(string message, Exception? error = null)
        {
            return new ActionResult
            {
                Success = false,
                Message = message,
                Error = error
            };
        }
    }
}
