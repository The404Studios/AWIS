using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace AWIS.AI
{
    /// <summary>
    /// Advanced task execution system with cycle checking, evidence validation, and retry logic
    /// Tracks dynamic hyperlation of objects with timestamps and sequence tracking
    /// </summary>
    public class TaskExecutionCycle
    {
        private readonly Dictionary<string, TaskCycle> activeCycles = new();
        private readonly Dictionary<string, List<ActionSequence>> sequenceHistory = new();
        private readonly Stopwatch globalTimer = Stopwatch.StartNew();
        private double currentFPS = 60.0; // Target FPS
        private double frameTimeMs = 16.67; // 1000/60

        // Backpropagation learning rates
        private double learningRate = 0.01;
        private readonly Dictionary<string, double> taskSuccessGradients = new();

        public TaskExecutionCycle()
        {
            Console.WriteLine("[CYCLE] Task execution cycle system initialized");
            Console.WriteLine("[CYCLE] Evidence-based retry and dynamic hyperlation enabled");
        }

        /// <summary>
        /// Execute a task with cycle checking and evidence validation
        /// </summary>
        public async Task<TaskResult> ExecuteWithCycleCheck(
            string taskId,
            Func<Task<TaskEvidence>> taskFunc,
            int maxRetries = 3,
            int priorityRegister = 6)
        {
            var startTime = globalTimer.Elapsed.TotalMilliseconds;
            var cycle = GetOrCreateCycle(taskId, priorityRegister);

            cycle.Status = TaskStatus.Running;
            cycle.CurrentAttempt++;
            cycle.LastStartTime = startTime;

            Console.WriteLine($"[CYCLE] Starting task '{taskId}' (Priority Register: {priorityRegister}, Attempt: {cycle.CurrentAttempt})");

            try
            {
                // Execute task and collect evidence
                var evidence = await taskFunc();
                var endTime = globalTimer.Elapsed.TotalMilliseconds;
                var duration = endTime - startTime;

                // Record sequence
                RecordActionSequence(taskId, startTime, endTime, evidence);

                // Check evidence for completion
                var isComplete = ValidateEvidence(evidence);

                if (isComplete)
                {
                    cycle.Status = TaskStatus.Completed;
                    cycle.CompletionTime = endTime;
                    cycle.TotalDuration += duration;

                    // Update backpropagation gradients
                    UpdateBackpropagation(taskId, true, duration);

                    Console.WriteLine($"[CYCLE] ✅ Task '{taskId}' completed in {duration:F2}ms (FPS impact: {CalculateFPSImpact(duration):F2})");

                    return new TaskResult
                    {
                        Success = true,
                        Evidence = evidence,
                        Duration = duration,
                        Attempts = cycle.CurrentAttempt
                    };
                }
                else if (cycle.CurrentAttempt < maxRetries)
                {
                    // Evidence not filled - retry
                    Console.WriteLine($"[CYCLE] ⚠️ Evidence incomplete for '{taskId}', retrying... ({cycle.CurrentAttempt}/{maxRetries})");
                    cycle.Status = TaskStatus.Retrying;

                    // Calculate backoff based on priority
                    var backoffMs = CalculateBackoff(priorityRegister, cycle.CurrentAttempt);
                    await Task.Delay((int)backoffMs);

                    // Recursive retry with cycle check
                    return await ExecuteWithCycleCheck(taskId, taskFunc, maxRetries, priorityRegister);
                }
                else
                {
                    // Max retries reached
                    cycle.Status = TaskStatus.Failed;
                    UpdateBackpropagation(taskId, false, duration);

                    Console.WriteLine($"[CYCLE] ❌ Task '{taskId}' failed after {maxRetries} attempts");

                    return new TaskResult
                    {
                        Success = false,
                        Evidence = evidence,
                        Duration = duration,
                        Attempts = cycle.CurrentAttempt
                    };
                }
            }
            catch (Exception ex)
            {
                cycle.Status = TaskStatus.Error;
                Console.WriteLine($"[CYCLE] ❌ Error in task '{taskId}': {ex.Message}");

                UpdateBackpropagation(taskId, false, 0);

                return new TaskResult
                {
                    Success = false,
                    Error = ex.Message,
                    Attempts = cycle.CurrentAttempt
                };
            }
        }

        /// <summary>
        /// Validate evidence to determine if task is complete
        /// </summary>
        private bool ValidateEvidence(TaskEvidence evidence)
        {
            if (evidence == null) return false;

            // Check if all required evidence fields are filled
            var requiredFields = evidence.RequiredFields ?? new List<string>();
            var providedData = evidence.Data ?? new Dictionary<string, object>();

            foreach (var field in requiredFields)
            {
                if (!providedData.ContainsKey(field) || providedData[field] == null)
                {
                    Console.WriteLine($"[CYCLE] Missing required evidence field: {field}");
                    return false;
                }
            }

            // Check confidence threshold
            if (evidence.Confidence < 0.7)
            {
                Console.WriteLine($"[CYCLE] Evidence confidence too low: {evidence.Confidence:P0}");
                return false;
            }

            return true;
        }

        /// <summary>
        /// Record action sequence with timestamps for dynamic hyperlation tracking
        /// </summary>
        private void RecordActionSequence(string taskId, double fromTime, double toTime, TaskEvidence evidence)
        {
            if (!sequenceHistory.ContainsKey(taskId))
            {
                sequenceHistory[taskId] = new List<ActionSequence>();
            }

            var sequence = new ActionSequence
            {
                TaskId = taskId,
                FromTimestamp = fromTime,
                ToTimestamp = toTime,
                DurationMs = toTime - fromTime,
                Evidence = evidence,
                TokenizedInput = TokenizeEvidence(evidence),
                SequenceIndex = sequenceHistory[taskId].Count,
                FrameTimeRatio = (toTime - fromTime) / frameTimeMs
            };

            sequenceHistory[taskId].Add(sequence);

            Console.WriteLine($"[SEQUENCE] Recorded: {taskId} [{fromTime:F2}ms -> {toTime:F2}ms] (Duration: {sequence.DurationMs:F2}ms, Frame Ratio: {sequence.FrameTimeRatio:F2})");
        }

        /// <summary>
        /// Tokenize evidence for integration into next process
        /// </summary>
        private List<string> TokenizeEvidence(TaskEvidence evidence)
        {
            var tokens = new List<string>();

            if (evidence?.Data == null) return tokens;

            foreach (var kvp in evidence.Data)
            {
                tokens.Add($"{kvp.Key}:{kvp.Value}");
            }

            return tokens;
        }

        /// <summary>
        /// Update backpropagation gradients based on task success/failure
        /// </summary>
        private void UpdateBackpropagation(string taskId, bool success, double duration)
        {
            if (!taskSuccessGradients.ContainsKey(taskId))
            {
                taskSuccessGradients[taskId] = 0.5; // Initialize at neutral
            }

            // Calculate gradient based on success and timing
            var timingFactor = Math.Clamp(1.0 - (duration / (frameTimeMs * 10)), 0, 1);
            var gradient = success ? timingFactor : -0.5;

            // Apply learning rate and update
            taskSuccessGradients[taskId] += learningRate * gradient;
            taskSuccessGradients[taskId] = Math.Clamp(taskSuccessGradients[taskId], 0, 1);

            Console.WriteLine($"[BACKPROP] Updated gradient for '{taskId}': {taskSuccessGradients[taskId]:F3} (Δ: {gradient:+0.000;-0.000})");
        }

        /// <summary>
        /// Calculate backoff time based on priority and attempt number
        /// </summary>
        private double CalculateBackoff(int priorityRegister, int attempt)
        {
            // Higher priority (lower number) = shorter backoff
            var basePriority = 13 - priorityRegister; // Invert so 1 = highest
            var backoff = frameTimeMs * basePriority * Math.Pow(1.5, attempt - 1);
            return Math.Min(backoff, 1000); // Cap at 1 second
        }

        /// <summary>
        /// Calculate FPS impact of task duration
        /// </summary>
        private double CalculateFPSImpact(double durationMs)
        {
            return durationMs / frameTimeMs;
        }

        /// <summary>
        /// Update FPS for dynamic frame time calculations
        /// </summary>
        public void UpdateFPS(double fps)
        {
            currentFPS = fps;
            frameTimeMs = 1000.0 / fps;
            Console.WriteLine($"[CYCLE] FPS updated: {fps:F1} (Frame time: {frameTimeMs:F2}ms)");
        }

        /// <summary>
        /// Get cycle information for a task
        /// </summary>
        private TaskCycle GetOrCreateCycle(string taskId, int priority)
        {
            if (!activeCycles.ContainsKey(taskId))
            {
                activeCycles[taskId] = new TaskCycle
                {
                    TaskId = taskId,
                    PriorityRegister = priority,
                    Status = TaskStatus.Pending
                };
            }
            return activeCycles[taskId];
        }

        /// <summary>
        /// Get all active cycles for debugging
        /// </summary>
        public List<TaskCycle> GetActiveCycles()
        {
            return activeCycles.Values.OrderBy(c => c.PriorityRegister).ToList();
        }

        /// <summary>
        /// Get sequence history for a task
        /// </summary>
        public List<ActionSequence> GetSequenceHistory(string taskId)
        {
            return sequenceHistory.GetValueOrDefault(taskId, new List<ActionSequence>());
        }

        /// <summary>
        /// Get backpropagation gradient for a task
        /// </summary>
        public double GetTaskGradient(string taskId)
        {
            return taskSuccessGradients.GetValueOrDefault(taskId, 0.5);
        }
    }

    public class TaskCycle
    {
        public string TaskId { get; set; } = string.Empty;
        public int PriorityRegister { get; set; } // 1-12, 1 = highest priority
        public TaskStatus Status { get; set; }
        public int CurrentAttempt { get; set; }
        public double LastStartTime { get; set; }
        public double CompletionTime { get; set; }
        public double TotalDuration { get; set; }
    }

    public class ActionSequence
    {
        public string TaskId { get; set; } = string.Empty;
        public double FromTimestamp { get; set; }
        public double ToTimestamp { get; set; }
        public double DurationMs { get; set; }
        public TaskEvidence? Evidence { get; set; }
        public List<string> TokenizedInput { get; set; } = new();
        public int SequenceIndex { get; set; }
        public double FrameTimeRatio { get; set; }
    }

    public class TaskEvidence
    {
        public Dictionary<string, object> Data { get; set; } = new();
        public List<string>? RequiredFields { get; set; }
        public double Confidence { get; set; } = 1.0;
        public string Description { get; set; } = string.Empty;
    }

    public class TaskResult
    {
        public bool Success { get; set; }
        public TaskEvidence? Evidence { get; set; }
        public double Duration { get; set; }
        public int Attempts { get; set; }
        public string? Error { get; set; }
    }

    public enum TaskStatus
    {
        Pending,
        Running,
        Retrying,
        Completed,
        Failed,
        Error
    }
}
