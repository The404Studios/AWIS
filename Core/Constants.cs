using System;

namespace AWIS.Core
{
    /// <summary>
    /// Global constants for AWIS system
    /// </summary>
    public static class Constants
    {
        // Voice recognition constants
        public const float VOICE_CONFIDENCE_THRESHOLD = 0.7f;
        public const int VOICE_SAMPLE_RATE = 16000;
        public const int VOICE_BUFFER_SIZE = 4096;

        // Memory constants
        public const int SHORT_TERM_MEMORY_CAPACITY = 100;
        public const int LONG_TERM_MEMORY_CAPACITY = 10000;
        public const int WORKING_MEMORY_CAPACITY = 20;

        // Learning constants
        public const double LEARNING_RATE = 0.01;
        public const double DISCOUNT_FACTOR = 0.95;
        public const double EXPLORATION_RATE = 0.1;

        // Performance constants
        public const int MAX_PARALLEL_TASKS = 8;
        public const int BATCH_SIZE = 100;
        public const int QUEUE_CAPACITY = 1000;

        // Vision constants
        public const int DEFAULT_SCREEN_WIDTH = 1920;
        public const int DEFAULT_SCREEN_HEIGHT = 1080;
        public const int COLOR_MATCH_TOLERANCE = 30;

        // Timing constants
        public const int DEFAULT_TIMEOUT_MS = 30000;
        public const int RETRY_DELAY_MS = 1000;
        public const int MAX_RETRIES = 3;

        // File paths
        public const string DATA_DIRECTORY = "data";
        public const string MODELS_DIRECTORY = "models";
        public const string LOGS_DIRECTORY = "logs";
        public const string CACHE_DIRECTORY = "cache";

        // Version info
        public const string VERSION = "8.0";
        public const string BUILD_NUMBER = "20250111";
    }

    /// <summary>
    /// Simple logging utility
    /// </summary>
    public static class Log
    {
        private static readonly object lockObj = new();

        public static void Information(string message)
        {
            lock (lockObj)
            {
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine($"[INFO {DateTime.Now:HH:mm:ss}] {message}");
                Console.ResetColor();
            }
        }

        public static void Warning(string message)
        {
            lock (lockObj)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[WARN {DateTime.Now:HH:mm:ss}] {message}");
                Console.ResetColor();
            }
        }

        public static void Error(Exception ex, string message)
        {
            lock (lockObj)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR {DateTime.Now:HH:mm:ss}] {message}");
                Console.WriteLine($"  Exception: {ex.Message}");
                Console.ResetColor();
            }
        }

        public static void Error(string message)
        {
            lock (lockObj)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR {DateTime.Now:HH:mm:ss}] {message}");
                Console.ResetColor();
            }
        }

        public static void Debug(string message)
        {
            lock (lockObj)
            {
                Console.ForegroundColor = ConsoleColor.Gray;
                Console.WriteLine($"[DEBUG {DateTime.Now:HH:mm:ss}] {message}");
                Console.ResetColor();
            }
        }

        public static void Success(string message)
        {
            lock (lockObj)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"[SUCCESS {DateTime.Now:HH:mm:ss}] {message}");
                Console.ResetColor();
            }
        }
    }

    /// <summary>
    /// Goal for goal-based planning
    /// </summary>
    public class Goal : IComparable<Goal>
    {
        public string Description { get; set; } = string.Empty;
        public double Priority { get; set; } = 0.5;
        public DateTime Deadline { get; set; } = DateTime.MaxValue;
        public bool IsCompleted { get; set; } = false;
        public List<AIAction> RequiredActions { get; set; } = new();

        public Goal(string description, double priority = 0.5)
        {
            Description = description;
            Priority = priority;
        }

        public int CompareTo(Goal? other)
        {
            if (other == null) return 1;

            // Higher priority comes first
            var priorityCompare = other.Priority.CompareTo(Priority);
            if (priorityCompare != 0) return priorityCompare;

            // Earlier deadline comes first
            return Deadline.CompareTo(other.Deadline);
        }
    }

    /// <summary>
    /// Priority queue implementation
    /// </summary>
    public class PriorityQueue<T> where T : IComparable<T>
    {
        private readonly List<T> data = new();
        private readonly object lockObj = new();

        public int Count
        {
            get { lock (lockObj) return data.Count; }
        }

        public void Enqueue(T item)
        {
            lock (lockObj)
            {
                data.Add(item);
                data.Sort();
            }
        }

        public T? Dequeue()
        {
            lock (lockObj)
            {
                if (data.Count == 0) return default;
                var item = data[0];
                data.RemoveAt(0);
                return item;
            }
        }

        public T? Peek()
        {
            lock (lockObj)
            {
                return data.Count > 0 ? data[0] : default;
            }
        }

        public bool TryDequeue(out T? item)
        {
            item = Dequeue();
            return item != null;
        }

        public void Clear()
        {
            lock (lockObj)
            {
                data.Clear();
            }
        }
    }
}
