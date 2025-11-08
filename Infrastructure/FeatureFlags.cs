using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using AWIS.Core;

namespace AWIS.Infrastructure
{

/// <summary>
/// Feature flags system for runtime toggles without redeployment
/// </summary>
public class FeatureFlagService
{
    private readonly ConcurrentDictionary<string, FeatureFlag> _flags = new ConcurrentDictionary<string, FeatureFlag>();
    private readonly ICorrelatedLogger _logger;
    private readonly string _flagsPath;
    private readonly FileSystemWatcher? _watcher;
    private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim();

    public FeatureFlagService(ICorrelatedLogger logger, string? flagsPath = null)
    {
        _logger = logger;
        _flagsPath = flagsPath ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "AWIS",
            "feature-flags.json");

        // Ensure directory exists
        var dir = Path.GetDirectoryName(_flagsPath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        // Watch for file changes
        if (File.Exists(_flagsPath))
        {
            _watcher = new FileSystemWatcher(dir!, Path.GetFileName(_flagsPath))
            {
                NotifyFilter = NotifyFilters.LastWrite
            };
            _watcher.Changed += async (sender, e) => await ReloadAsync();
            _watcher.EnableRaisingEvents = true;
        }

        // Load default flags
        LoadDefaultFlags();
    }

    /// <summary>
    /// Check if a feature is enabled
    /// </summary>
    public bool IsEnabled(string flagName, Dictionary<string, object>? context = null)
    {
        _lock.EnterReadLock();
        try
        {
            if (!_flags.TryGetValue(flagName, out var flag))
            {
                _logger.LogWithContext(
                    $"Feature flag not found: {flagName}, defaulting to disabled",
                    LogLevel.Debug);
                return false;
            }

            // Check if flag is enabled
            if (!flag.Enabled)
                return false;

            // Check percentage rollout
            if (flag.RolloutPercentage < 100)
            {
                var hash = GetStableHash(flagName + (context?.GetValueOrDefault("userId") ?? ""));
                if ((hash % 100) >= flag.RolloutPercentage)
                    return false;
            }

            // Check conditions
            if (flag.Conditions != null && context != null)
            {
                foreach (var condition in flag.Conditions)
                {
                    if (!EvaluateCondition(condition, context))
                        return false;
                }
            }

            return true;
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Set a feature flag
    /// </summary>
    public void SetFlag(string name, bool enabled)
    {
        _lock.EnterWriteLock();
        try
        {
            if (_flags.TryGetValue(name, out var flag))
            {
                flag.Enabled = enabled;
            }
            else
            {
                _flags[name] = new FeatureFlag
                {
                    Name = name,
                    Enabled = enabled
                };
            }

            _logger.LogWithContext(
                $"Feature flag set: {name} = {enabled}",
                LogLevel.Information);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Set flag with percentage rollout
    /// </summary>
    public void SetFlagWithRollout(string name, bool enabled, int percentageRollout)
    {
        _lock.EnterWriteLock();
        try
        {
            _flags[name] = new FeatureFlag
            {
                Name = name,
                Enabled = enabled,
                RolloutPercentage = percentageRollout
            };

            _logger.LogWithContext(
                $"Feature flag set with rollout: {name} = {enabled} @ {percentageRollout}%",
                LogLevel.Information);
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Get all flags
    /// </summary>
    public Dictionary<string, FeatureFlag> GetAllFlags()
    {
        _lock.EnterReadLock();
        try
        {
            return new Dictionary<string, FeatureFlag>(_flags);
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Save flags to file
    /// </summary>
    public async Task SaveAsync()
    {
        _lock.EnterReadLock();
        try
        {
            var flags = _flags.Values.ToArray();

            var json = JsonSerializer.Serialize(flags, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            // Temporarily disable watcher to avoid triggering reload
            if (_watcher != null)
                _watcher.EnableRaisingEvents = false;

            await File.WriteAllTextAsync(_flagsPath, json);

            if (_watcher != null)
                _watcher.EnableRaisingEvents = true;
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    /// <summary>
    /// Load flags from file
    /// </summary>
    public async Task LoadAsync()
    {
        if (!File.Exists(_flagsPath))
        {
            await SaveAsync(); // Create file with defaults
            return;
        }

        var json = await File.ReadAllTextAsync(_flagsPath);
        var flags = JsonSerializer.Deserialize<FeatureFlag[]>(json);

        if (flags != null)
        {
            _lock.EnterWriteLock();
            try
            {
                _flags.Clear();
                foreach (var flag in flags)
                {
                    _flags[flag.Name] = flag;
                }

                _logger.LogWithContext(
                    $"Loaded {flags.Length} feature flags",
                    LogLevel.Information);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }
    }

    /// <summary>
    /// Reload flags from file
    /// </summary>
    private async Task ReloadAsync()
    {
        try
        {
            await LoadAsync();
            _logger.LogWithContext("Feature flags reloaded", LogLevel.Information);
        }
        catch (Exception ex)
        {
            _logger.LogWithContext(
                $"Failed to reload feature flags: {ex.Message}",
                LogLevel.Error);
        }
    }

    private void LoadDefaultFlags()
    {
        // Experimental features
        SetFlag("experimental.advanced_vision", false);
        SetFlag("experimental.hierarchical_rl", false);
        SetFlag("experimental.curiosity_learning", false);
        SetFlag("experimental.multi_agent", false);

        // Performance features
        SetFlag("performance.simd_optimizations", true);
        SetFlag("performance.object_pooling", true);
        SetFlag("performance.parallel_init", true);

        // Safety features
        SetFlag("safety.policy_enforcement", true);
        SetFlag("safety.capability_tokens", true);
        SetFlag("safety.dry_run_mode", false);

        // Observability
        SetFlag("observability.detailed_metrics", true);
        SetFlag("observability.event_replay", false);

        // Modules
        SetFlag("modules.vision", true);
        SetFlag("modules.voice", true);
        SetFlag("modules.rl", true);
        SetFlag("modules.knowledge_graph", true);
    }

    private bool EvaluateCondition(FlagCondition condition, Dictionary<string, object> context)
    {
        if (!context.TryGetValue(condition.Key, out var value))
            return false;

        return condition.Operator switch
        {
            "equals" => value.ToString() == condition.Value?.ToString(),
            "not_equals" => value.ToString() != condition.Value?.ToString(),
            "contains" => value.ToString()?.Contains(condition.Value?.ToString() ?? "") ?? false,
            "greater_than" => Convert.ToDouble(value) > Convert.ToDouble(condition.Value),
            "less_than" => Convert.ToDouble(value) < Convert.ToDouble(condition.Value),
            _ => false
        };
    }

    private int GetStableHash(string input)
    {
        unchecked
        {
            int hash = 23;
            foreach (var c in input)
            {
                hash = hash * 31 + c;
            }
            return Math.Abs(hash);
        }
    }

    public void Dispose()
    {
        _watcher?.Dispose();
        _lock.Dispose();
    }
}

/// <summary>
/// Feature flag definition
/// </summary>
public class FeatureFlag
{
    public string Name { get; set; } = string.Empty;
    public bool Enabled { get; set; }
    public string? Description { get; set; }
    public int RolloutPercentage { get; set; } = 100;
    public List<FlagCondition>? Conditions { get; set; }
    public Dictionary<string, object>? Metadata { get; set; }
}

/// <summary>
/// Condition for flag evaluation
/// </summary>
public class FlagCondition
{
    public string Key { get; set; } = string.Empty;
    public string Operator { get; set; } = string.Empty; // equals, not_equals, contains, greater_than, less_than
    public object? Value { get; set; }
}

/// <summary>
/// Convenience extension methods for feature flags
/// </summary>
public static class FeatureFlagExtensions
{
    /// <summary>
    /// Execute code only if feature is enabled
    /// </summary>
    public static void IfEnabled(
        this FeatureFlagService flags,
        string flagName,
        Action action,
        Dictionary<string, object>? context = null)
    {
        if (flags.IsEnabled(flagName, context))
        {
            action();
        }
    }

    /// <summary>
    /// Execute async code only if feature is enabled
    /// </summary>
    public static async Task IfEnabledAsync(
        this FeatureFlagService flags,
        string flagName,
        Func<Task> action,
        Dictionary<string, object>? context = null)
    {
        if (flags.IsEnabled(flagName, context))
        {
            await action();
        }
    }

    /// <summary>
    /// Execute one of two actions based on flag
    /// </summary>
    public static T Switch<T>(
        this FeatureFlagService flags,
        string flagName,
        Func<T> enabledAction,
        Func<T> disabledAction,
        Dictionary<string, object>? context = null)
    {
        return flags.IsEnabled(flagName, context) ? enabledAction() : disabledAction();
    }
}

/// <summary>
/// Feature flag middleware for flag-based routing
/// </summary>
public class FeatureFlagMiddleware
{
    private readonly FeatureFlagService _flags;
    private readonly ICorrelatedLogger _logger;

    public FeatureFlagMiddleware(FeatureFlagService flags, ICorrelatedLogger logger)
    {
        _flags = flags;
        _logger = logger;
    }

    /// <summary>
    /// Require a feature flag for operation
    /// </summary>
    public async Task<T> RequireFeatureAsync<T>(
        string flagName,
        Func<Task<T>> operation,
        Dictionary<string, object>? context = null)
    {
        if (!_flags.IsEnabled(flagName, context))
        {
            _logger.LogWithContext(
                $"Feature {flagName} is disabled",
                LogLevel.Warning,
                new Dictionary<string, object>
                {
                    ["FlagName"] = flagName,
                    ["Context"] = context ?? new Dictionary<string, object>()
                });

            throw new FeatureDisabledException($"Feature {flagName} is not enabled");
        }

        return await operation();
    }
}

public class FeatureDisabledException : Exception
{
    public FeatureDisabledException(string message) : base(message) { }
}
}
