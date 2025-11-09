using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AWIS.AI;

namespace AWIS.Debug
{
    /// <summary>
    /// Debug overlay system for visualizing task execution, priorities, and performance
    /// Renders on right side of screen with process information and loading indicators
    /// </summary>
    public class DebugOverlay : IDisposable
    {
        private readonly TaskExecutionCycle executionCycle;
        private readonly PriorityRegisterSystem prioritySystem;
        private bool isRunning = false;
        private Task? updateTask;
        private readonly CancellationTokenSource cancellationToken = new();

        // Performance tracking
        private double currentFPS = 0;
        private int frameCount = 0;
        private DateTime lastFPSUpdate = DateTime.UtcNow;

        // Overlay configuration
        private const int OVERLAY_WIDTH = 50;
        private const int UPDATE_INTERVAL_MS = 100;

        public DebugOverlay(TaskExecutionCycle executionCycle, PriorityRegisterSystem prioritySystem)
        {
            this.executionCycle = executionCycle;
            this.prioritySystem = prioritySystem;

            Console.WriteLine("[DEBUG] Debug overlay initialized");
        }

        /// <summary>
        /// Start the debug overlay rendering loop
        /// </summary>
        public void Start()
        {
            if (isRunning) return;

            isRunning = true;
            updateTask = Task.Run(() => UpdateLoop(cancellationToken.Token));

            Console.WriteLine("[DEBUG] Debug overlay started");
        }

        /// <summary>
        /// Stop the debug overlay
        /// </summary>
        public void Stop()
        {
            if (!isRunning) return;

            isRunning = false;
            cancellationToken.Cancel();
            updateTask?.Wait(1000);

            Console.WriteLine("[DEBUG] Debug overlay stopped");
        }

        /// <summary>
        /// Main update loop for rendering debug information
        /// </summary>
        private async Task UpdateLoop(CancellationToken token)
        {
            while (isRunning && !token.IsCancellationRequested)
            {
                try
                {
                    UpdateFPS();
                    RenderOverlay();
                    await Task.Delay(UPDATE_INTERVAL_MS, token);
                }
                catch (TaskCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[DEBUG] Overlay error: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// Update FPS counter
        /// </summary>
        private void UpdateFPS()
        {
            frameCount++;
            var now = DateTime.UtcNow;
            var elapsed = (now - lastFPSUpdate).TotalSeconds;

            if (elapsed >= 1.0)
            {
                currentFPS = frameCount / elapsed;
                frameCount = 0;
                lastFPSUpdate = now;

                // Update execution cycle with current FPS
                executionCycle.UpdateFPS(currentFPS);
            }
        }

        /// <summary>
        /// Render the debug overlay to console
        /// </summary>
        private void RenderOverlay()
        {
            var overlay = BuildOverlay();

            // Clear previous output and render new overlay
            // Note: In a real ImGui implementation, this would render to a window
            // For console, we'll use a bordered box on the right side

            Console.SetCursorPosition(0, 0);
            Console.WriteLine(overlay);
        }

        /// <summary>
        /// Build the overlay content
        /// </summary>
        private string BuildOverlay()
        {
            var sb = new StringBuilder();

            // Header
            sb.AppendLine("╔" + new string('═', OVERLAY_WIDTH - 2) + "╗");
            sb.AppendLine("║" + CenterText("🔍 DEBUG OVERLAY", OVERLAY_WIDTH - 2) + "║");
            sb.AppendLine("╠" + new string('═', OVERLAY_WIDTH - 2) + "╣");

            // FPS and Performance
            sb.AppendLine("║" + PadText($" FPS: {currentFPS:F1}", OVERLAY_WIDTH - 2) + "║");
            sb.AppendLine("║" + PadText($" Frame: {1000.0 / Math.Max(currentFPS, 1):F2}ms", OVERLAY_WIDTH - 2) + "║");
            sb.AppendLine("╠" + new string('═', OVERLAY_WIDTH - 2) + "╣");

            // Active Task Cycles
            sb.AppendLine("║" + CenterText("ACTIVE PROCESSES", OVERLAY_WIDTH - 2) + "║");
            sb.AppendLine("╠" + new string('─', OVERLAY_WIDTH - 2) + "╣");

            var cycles = executionCycle.GetActiveCycles();
            if (cycles.Count == 0)
            {
                sb.AppendLine("║" + PadText(" (No active processes)", OVERLAY_WIDTH - 2) + "║");
            }
            else
            {
                foreach (var cycle in cycles.Take(8)) // Show top 8 processes
                {
                    var statusIcon = GetStatusIcon(cycle.Status);
                    var loadingBar = GetLoadingBar(cycle);
                    var priorityBadge = $"[R{cycle.PriorityRegister,2}]";

                    sb.AppendLine("║" + PadText($" {statusIcon} {priorityBadge} {cycle.TaskId}", OVERLAY_WIDTH - 2) + "║");
                    sb.AppendLine("║" + PadText($"     {loadingBar}", OVERLAY_WIDTH - 2) + "║");
                }

                if (cycles.Count > 8)
                {
                    sb.AppendLine("║" + PadText($" ... and {cycles.Count - 8} more", OVERLAY_WIDTH - 2) + "║");
                }
            }

            sb.AppendLine("╠" + new string('═', OVERLAY_WIDTH - 2) + "╣");

            // Priority Register Status
            sb.AppendLine("║" + CenterText("PRIORITY REGISTERS", OVERLAY_WIDTH - 2) + "║");
            sb.AppendLine("╠" + new string('─', OVERLAY_WIDTH - 2) + "╣");

            var stats = prioritySystem.GetStatistics();
            sb.AppendLine("║" + PadText($" Scheduled: {stats.TotalScheduled}", OVERLAY_WIDTH - 2) + "║");
            sb.AppendLine("║" + PadText($" Completed: {stats.TotalCompleted}", OVERLAY_WIDTH - 2) + "║");
            sb.AppendLine("║" + PadText($" Queued: {stats.TotalQueued}", OVERLAY_WIDTH - 2) + "║");
            sb.AppendLine("╠" + new string('─', OVERLAY_WIDTH - 2) + "╣");

            // Show register visualization (compact)
            for (int i = 1; i <= 6; i++)
            {
                var depth = stats.RegisterDepths.GetValueOrDefault(i, 0);
                var bar = GetPriorityBar(depth);
                sb.AppendLine("║" + PadText($" R{i}: {bar} {depth}", OVERLAY_WIDTH - 2) + "║");
            }

            if (stats.RegisterDepths.Values.Skip(6).Sum() > 0)
            {
                sb.AppendLine("║" + PadText($" R7-12: {stats.RegisterDepths.Values.Skip(6).Sum()} tasks", OVERLAY_WIDTH - 2) + "║");
            }

            sb.AppendLine("╠" + new string('═', OVERLAY_WIDTH - 2) + "╣");

            // Backpropagation Status
            sb.AppendLine("║" + CenterText("LEARNING", OVERLAY_WIDTH - 2) + "║");
            sb.AppendLine("╠" + new string('─', OVERLAY_WIDTH - 2) + "╣");

            var topTasks = cycles.Take(3);
            foreach (var cycle in topTasks)
            {
                var gradient = executionCycle.GetTaskGradient(cycle.TaskId);
                var gradientBar = GetGradientBar(gradient);

                sb.AppendLine("║" + PadText($" {cycle.TaskId.Substring(0, Math.Min(cycle.TaskId.Length, 12))}", OVERLAY_WIDTH - 2) + "║");
                sb.AppendLine("║" + PadText($"   {gradientBar} {gradient:P0}", OVERLAY_WIDTH - 2) + "║");
            }

            sb.AppendLine("╚" + new string('═', OVERLAY_WIDTH - 2) + "╝");

            return sb.ToString();
        }

        /// <summary>
        /// Get status icon for task status
        /// </summary>
        private string GetStatusIcon(TaskStatus status)
        {
            return status switch
            {
                TaskStatus.Running => "⚙️",
                TaskStatus.Retrying => "🔄",
                TaskStatus.Completed => "✅",
                TaskStatus.Failed => "❌",
                TaskStatus.Error => "⚠️",
                _ => "⏸️"
            };
        }

        /// <summary>
        /// Get loading bar for task progress
        /// </summary>
        private string GetLoadingBar(TaskCycle cycle)
        {
            if (cycle.Status == TaskStatus.Completed)
            {
                return "[" + new string('█', 10) + "] DONE";
            }

            // Animate loading bar based on time
            var elapsed = DateTime.UtcNow.Ticks / TimeSpan.TicksPerMillisecond;
            var position = (int)((elapsed / 100) % 10);
            var bar = new string('░', 10).ToCharArray();
            bar[position] = '█';

            return "[" + new string(bar) + "] " + cycle.Status;
        }

        /// <summary>
        /// Get priority bar visualization
        /// </summary>
        private string GetPriorityBar(int depth)
        {
            var barLength = Math.Min(depth, 10);
            return "[" + new string('█', barLength) + new string('░', 10 - barLength) + "]";
        }

        /// <summary>
        /// Get gradient bar visualization
        /// </summary>
        private string GetGradientBar(double gradient)
        {
            var filled = (int)(gradient * 10);
            return "[" + new string('█', filled) + new string('░', 10 - filled) + "]";
        }

        /// <summary>
        /// Center text within width
        /// </summary>
        private string CenterText(string text, int width)
        {
            if (text.Length >= width) return text.Substring(0, width);

            var padding = (width - text.Length) / 2;
            return new string(' ', padding) + text + new string(' ', width - text.Length - padding);
        }

        /// <summary>
        /// Pad text to width
        /// </summary>
        private string PadText(string text, int width)
        {
            if (text.Length >= width) return text.Substring(0, width);
            return text + new string(' ', width - text.Length);
        }

        /// <summary>
        /// Render a nice summary report
        /// </summary>
        public void RenderSummary()
        {
            Console.WriteLine("\n╔════════════════════════════════════════════════╗");
            Console.WriteLine("║        📊 DEBUG OVERLAY SUMMARY REPORT        ║");
            Console.WriteLine("╠════════════════════════════════════════════════╣");

            var stats = prioritySystem.GetStatistics();
            Console.WriteLine($"║ Tasks Scheduled:  {stats.TotalScheduled,30} ║");
            Console.WriteLine($"║ Tasks Completed:  {stats.TotalCompleted,30} ║");
            Console.WriteLine($"║ Tasks Queued:     {stats.TotalQueued,30} ║");
            Console.WriteLine($"║ Current FPS:      {currentFPS,30:F1} ║");

            Console.WriteLine("╠════════════════════════════════════════════════╣");
            Console.WriteLine("║           PRIORITY REGISTER STATUS             ║");
            Console.WriteLine("╠════════════════════════════════════════════════╣");

            Console.WriteLine(prioritySystem.GetRegisterVisualization());

            Console.WriteLine("╚════════════════════════════════════════════════╝\n");
        }

        public void Dispose()
        {
            Stop();
            cancellationToken.Dispose();
        }
    }
}
