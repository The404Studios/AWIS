using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AWIS.AI
{
    /// <summary>
    /// 12-level priority register system for task scheduling
    /// Register 1 = Highest priority, Register 12 = Lowest priority
    /// </summary>
    public class PriorityRegisterSystem
    {
        private const int NUM_REGISTERS = 12;
        private readonly Dictionary<int, Queue<PriorityTask>> registers = new();
        private readonly TaskExecutionCycle executionCycle;
        private readonly Dictionary<string, int> taskPriorities = new();
        private int tasksScheduled = 0;
        private int tasksCompleted = 0;

        public PriorityRegisterSystem(TaskExecutionCycle executionCycle)
        {
            this.executionCycle = executionCycle;

            // Initialize all 12 registers
            for (int i = 1; i <= NUM_REGISTERS; i++)
            {
                registers[i] = new Queue<PriorityTask>();
            }

            Console.WriteLine("[PRIORITY] Initialized 12-level priority register system");
            Console.WriteLine("[PRIORITY] Register 1 = Highest, Register 12 = Lowest");
        }

        /// <summary>
        /// Schedule a task to a specific priority register
        /// </summary>
        public void ScheduleTask(
            string taskId,
            Func<Task<TaskEvidence>> taskFunc,
            int priorityRegister = 6,
            int maxRetries = 3)
        {
            // Validate priority register
            if (priorityRegister < 1 || priorityRegister > NUM_REGISTERS)
            {
                throw new ArgumentException($"Priority register must be between 1 and {NUM_REGISTERS}");
            }

            var task = new PriorityTask
            {
                TaskId = taskId,
                TaskFunc = taskFunc,
                PriorityRegister = priorityRegister,
                MaxRetries = maxRetries,
                ScheduledTime = DateTime.UtcNow
            };

            registers[priorityRegister].Enqueue(task);
            taskPriorities[taskId] = priorityRegister;
            tasksScheduled++;

            Console.WriteLine($"[PRIORITY] Scheduled '{taskId}' to Register {priorityRegister} (Queue depth: {registers[priorityRegister].Count})");
        }

        /// <summary>
        /// Execute next highest priority task from registers
        /// </summary>
        public async Task<TaskResult?> ExecuteNextTask()
        {
            // Scan registers from highest to lowest priority
            for (int register = 1; register <= NUM_REGISTERS; register++)
            {
                if (registers[register].Count > 0)
                {
                    var task = registers[register].Dequeue();

                    Console.WriteLine($"[PRIORITY] Executing task from Register {register}: '{task.TaskId}'");

                    var result = await executionCycle.ExecuteWithCycleCheck(
                        task.TaskId,
                        task.TaskFunc,
                        task.MaxRetries,
                        task.PriorityRegister
                    );

                    if (result.Success)
                    {
                        tasksCompleted++;
                    }
                    else if (result.Attempts < task.MaxRetries)
                    {
                        // Re-queue failed task at lower priority
                        var newPriority = Math.Min(register + 2, NUM_REGISTERS);
                        task.PriorityRegister = newPriority;
                        registers[newPriority].Enqueue(task);

                        Console.WriteLine($"[PRIORITY] Re-queued '{task.TaskId}' to Register {newPriority} (demoted)");
                    }

                    return result;
                }
            }

            // No tasks in any register
            return null;
        }

        /// <summary>
        /// Execute all tasks in priority order
        /// </summary>
        public async Task<List<TaskResult>> ExecuteAllTasks()
        {
            var results = new List<TaskResult>();

            while (GetTotalQueuedTasks() > 0)
            {
                var result = await ExecuteNextTask();
                if (result != null)
                {
                    results.Add(result);
                }
            }

            Console.WriteLine($"[PRIORITY] All tasks completed: {tasksCompleted}/{tasksScheduled} succeeded");

            return results;
        }

        /// <summary>
        /// Dynamically adjust task priority based on performance
        /// </summary>
        public void AdjustPriority(string taskId, int newPriority)
        {
            if (newPriority < 1 || newPriority > NUM_REGISTERS)
            {
                Console.WriteLine($"[PRIORITY] Invalid priority adjustment for '{taskId}': {newPriority}");
                return;
            }

            if (taskPriorities.ContainsKey(taskId))
            {
                var oldPriority = taskPriorities[taskId];
                taskPriorities[taskId] = newPriority;

                Console.WriteLine($"[PRIORITY] Adjusted '{taskId}': Register {oldPriority} -> Register {newPriority}");
            }
        }

        /// <summary>
        /// Promote task to higher priority based on gradient
        /// </summary>
        public void PromoteByGradient(string taskId)
        {
            var gradient = executionCycle.GetTaskGradient(taskId);

            // High gradient (>0.8) = promote
            if (gradient > 0.8 && taskPriorities.TryGetValue(taskId, out var currentPriority))
            {
                var newPriority = Math.Max(1, currentPriority - 1);
                AdjustPriority(taskId, newPriority);
            }
            // Low gradient (<0.3) = demote
            else if (gradient < 0.3 && taskPriorities.TryGetValue(taskId, out currentPriority))
            {
                var newPriority = Math.Min(NUM_REGISTERS, currentPriority + 1);
                AdjustPriority(taskId, newPriority);
            }
        }

        /// <summary>
        /// Get total queued tasks across all registers
        /// </summary>
        public int GetTotalQueuedTasks()
        {
            return registers.Values.Sum(q => q.Count);
        }

        /// <summary>
        /// Get tasks in a specific register
        /// </summary>
        public int GetRegisterDepth(int register)
        {
            return registers.GetValueOrDefault(register, new Queue<PriorityTask>()).Count;
        }

        /// <summary>
        /// Get register statistics for debugging
        /// </summary>
        public RegisterStatistics GetStatistics()
        {
            var stats = new RegisterStatistics
            {
                TotalScheduled = tasksScheduled,
                TotalCompleted = tasksCompleted,
                TotalQueued = GetTotalQueuedTasks(),
                RegisterDepths = new Dictionary<int, int>()
            };

            for (int i = 1; i <= NUM_REGISTERS; i++)
            {
                stats.RegisterDepths[i] = GetRegisterDepth(i);
            }

            return stats;
        }

        /// <summary>
        /// Get visual representation of register state
        /// </summary>
        public string GetRegisterVisualization()
        {
            var viz = "Priority Registers:\n";

            for (int i = 1; i <= NUM_REGISTERS; i++)
            {
                var depth = GetRegisterDepth(i);
                var bar = new string('█', Math.Min(depth, 20));
                var label = i == 1 ? " (HIGHEST)" : i == NUM_REGISTERS ? " (LOWEST)" : "";

                viz += $"R{i,2}: [{bar,-20}] {depth} tasks{label}\n";
            }

            return viz;
        }
    }

    public class PriorityTask
    {
        public string TaskId { get; set; } = string.Empty;
        public Func<Task<TaskEvidence>> TaskFunc { get; set; } = null!;
        public int PriorityRegister { get; set; }
        public int MaxRetries { get; set; }
        public DateTime ScheduledTime { get; set; }
    }

    public class RegisterStatistics
    {
        public int TotalScheduled { get; set; }
        public int TotalCompleted { get; set; }
        public int TotalQueued { get; set; }
        public Dictionary<int, int> RegisterDepths { get; set; } = new();
    }
}
