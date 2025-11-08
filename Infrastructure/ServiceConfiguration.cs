using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using AWIS.Core;
using AWIS.AI;
using AWIS.Vision;
using AWIS.Data;

namespace AWIS.Infrastructure
{

/// <summary>
/// Dependency injection and service configuration
/// </summary>
public class ServiceConfiguration
{
    private readonly IServiceCollection _services;

    public ServiceConfiguration()
    {
        _services = new ServiceCollection();
    }

    /// <summary>
    /// Configure all AWIS services
    /// </summary>
    public IServiceProvider ConfigureServices(AWISConfig? config = null)
    {
        config ??= AWISConfig.Default();

        // Core infrastructure
        _services.AddSingleton<IEventBus, EventBus>();
        _services.AddSingleton<IMediator>(sp => new Mediator(sp, sp.GetRequiredService<IEventBus>()));
        _services.AddSingleton<IConfigurationManager>(new ConfigurationManager(config));
        _services.AddSingleton<IMetricsCollector, MetricsCollector>();
        _services.AddSingleton<ICorrelatedLogger, CorrelatedLogger>();

        // Data layer
        _services.AddSingleton<IKnowledgeStore>(sp =>
            new KnowledgeGraphService(
                config.Database.Path,
                sp.GetRequiredService<IEventBus>()));

        _services.AddSingleton<IMemorySystem>(sp =>
            new AWIS.Core.MemoryArchitecture());

        // AI services
        _services.AddSingleton(sp =>
            new AutonomousIntelligenceCore());

        _services.AddSingleton<IReinforcementAgent>(sp =>
            new ReinforcementLearningAgent(
                config.RL.StateSize,
                config.RL.ActionSize,
                sp.GetRequiredService<IEventBus>()));

        // Vision services
        _services.AddSingleton<IVisionSystem>(sp =>
            new AdvancedVisionPipeline());

        // Voice services
        _services.AddSingleton<IVoiceSystem>(sp =>
            new AWIS.Voice.VoiceCommandSystem());

        // NLP services
        _services.AddSingleton<INLPProcessor>(sp =>
            new NLPProcessor());

        // Subsystem orchestrator
        _services.AddSingleton<SubsystemOrchestrator>();

        return _services.BuildServiceProvider();
    }

    /// <summary>
    /// Configure with custom service registration
    /// </summary>
    public IServiceProvider ConfigureServices(Action<IServiceCollection> customConfiguration)
    {
        customConfiguration(_services);
        return _services.BuildServiceProvider();
    }
}

/// <summary>
/// Configuration manager for AWIS settings
/// </summary>
public class ConfigurationManager : IConfigurationManager
{
    private readonly Dictionary<string, object> _settings = new();
    private readonly AWISConfig _config;

    public ConfigurationManager(AWISConfig config)
    {
        _config = config;
        LoadFromConfig(config);
    }

    public T GetValue<T>(string key, T defaultValue = default!)
    {
        if (_settings.TryGetValue(key, out var value) && value is T typedValue)
        {
            return typedValue;
        }
        return defaultValue;
    }

    public void SetValue<T>(string key, T value)
    {
        if (value != null)
            _settings[key] = value;
    }

    public async Task LoadAsync(string path)
    {
        if (!File.Exists(path))
            return;

        var json = await File.ReadAllTextAsync(path);
        var config = JsonSerializer.Deserialize<Dictionary<string, object>>(json);

        if (config != null)
        {
            foreach (var kvp in config)
            {
                _settings[kvp.Key] = kvp.Value;
            }
        }
    }

    public async Task SaveAsync(string path)
    {
        var json = JsonSerializer.Serialize(_settings, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        await File.WriteAllTextAsync(path, json);
    }

    private void LoadFromConfig(AWISConfig config)
    {
        _settings["Voice.Enabled"] = config.Voice.Enabled;
        _settings["Voice.ConfidenceThreshold"] = config.Voice.ConfidenceThreshold;
        _settings["Vision.FPS"] = config.Vision.FPS;
        _settings["Vision.Resolution"] = config.Vision.Resolution;
        _settings["Learning.LearningRate"] = config.Learning.LearningRate;
        _settings["Learning.ExplorationRate"] = config.Learning.ExplorationRate;
        _settings["Learning.BatchSize"] = config.Learning.BatchSize;
        _settings["Database.Path"] = config.Database.Path;
        _settings["RL.StateSize"] = config.RL.StateSize;
        _settings["RL.ActionSize"] = config.RL.ActionSize;
    }
}

/// <summary>
/// AWIS configuration model
/// </summary>
public class AWISConfig
{
    public VoiceConfig Voice { get; set; } = new();
    public VisionConfig Vision { get; set; } = new();
    public LearningConfig Learning { get; set; } = new();
    public DatabaseConfig Database { get; set; } = new();
    public RLConfig RL { get; set; } = new();
    public LoggingConfig Logging { get; set; } = new();

    public static AWISConfig Default() => new()
    {
        Voice = new VoiceConfig
        {
            Enabled = true,
            ConfidenceThreshold = 0.7,
            SpeakingRate = 0,
            SpeakingVolume = 100
        },
        Vision = new VisionConfig
        {
            FPS = 30,
            Resolution = "1920x1080",
            ObjectDetectionThreshold = 0.5
        },
        Learning = new LearningConfig
        {
            LearningRate = 0.001,
            ExplorationRate = 0.1,
            BatchSize = 32
        },
        Database = new DatabaseConfig
        {
            Path = "awis.db"
        },
        RL = new RLConfig
        {
            StateSize = 64,
            ActionSize = 10
        },
        Logging = new LoggingConfig
        {
            Level = "Information",
            OutputPath = "./logs"
        }
    };
}

public class VoiceConfig
{
    public bool Enabled { get; set; }
    public double ConfidenceThreshold { get; set; }
    public int SpeakingRate { get; set; }
    public int SpeakingVolume { get; set; }
}

public class VisionConfig
{
    public int FPS { get; set; }
    public string Resolution { get; set; } = string.Empty;
    public double ObjectDetectionThreshold { get; set; }
}

public class LearningConfig
{
    public double LearningRate { get; set; }
    public double ExplorationRate { get; set; }
    public int BatchSize { get; set; }
}

public class DatabaseConfig
{
    public string Path { get; set; } = string.Empty;
}

public class RLConfig
{
    public int StateSize { get; set; }
    public int ActionSize { get; set; }
}

public class LoggingConfig
{
    public string Level { get; set; } = string.Empty;
    public string OutputPath { get; set; } = string.Empty;
}

/// <summary>
/// Metrics collector implementation
/// </summary>
public class MetricsCollector : IMetricsCollector
{
    private readonly Dictionary<string, List<MetricPoint>> _metrics = new();
    private readonly object _lock = new();

    public void RecordMetric(string name, double value, Dictionary<string, string>? tags = null)
    {
        lock (_lock)
        {
            if (!_metrics.ContainsKey(name))
            {
                _metrics[name] = new List<MetricPoint>();
            }

            _metrics[name].Add(new MetricPoint
            {
                Value = value,
                Timestamp = DateTime.UtcNow,
                Tags = tags ?? new Dictionary<string, string>()
            });

            // Keep only last 10000 points per metric
            if (_metrics[name].Count > 10000)
            {
                _metrics[name].RemoveAt(0);
            }
        }
    }

    public void IncrementCounter(string name, Dictionary<string, string>? tags = null)
    {
        RecordMetric(name, 1.0, tags);
    }

    public void RecordHistogram(string name, double value, Dictionary<string, string>? tags = null)
    {
        RecordMetric(name, value, tags);
    }

    public async Task<MetricsSummary> GetSummaryAsync(TimeSpan? window = null)
    {
        return await Task.Run(() =>
        {
            var cutoff = window.HasValue ? DateTime.UtcNow - window.Value : DateTime.MinValue;
            var summary = new MetricsSummary
            {
                StartTime = cutoff,
                EndTime = DateTime.UtcNow
            };

            lock (_lock)
            {
                foreach (var kvp in _metrics)
                {
                    var points = kvp.Value.Where(p => p.Timestamp >= cutoff).Select(p => p.Value).ToArray();

                    if (points.Length > 0)
                    {
                        var mean = points.Average();
                        var variance = points.Select(v => Math.Pow(v - mean, 2)).Average();

                        summary.Metrics[kvp.Key] = new MetricData
                        {
                            Count = points.Length,
                            Sum = points.Sum(),
                            Min = points.Min(),
                            Max = points.Max(),
                            StandardDeviation = Math.Sqrt(variance)
                        };
                    }
                }
            }

            return summary;
        });
    }

    private class MetricPoint
    {
        public double Value { get; set; }
        public DateTime Timestamp { get; set; }
        public Dictionary<string, string> Tags { get; set; } = new();
    }
}

/// <summary>
/// Correlated logger for contextual logging
/// </summary>
public class CorrelatedLogger : ICorrelatedLogger
{
    private readonly Stack<string> _correlationStack = new();

    public void LogWithContext(string message, LogLevel level, Dictionary<string, object>? context = null)
    {
        var correlationId = _correlationStack.Count > 0 ? _correlationStack.Peek() : "default";

        var logEntry = $"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] [{level}] [{correlationId}] {message}";

        if (context != null && context.Count > 0)
        {
            var contextStr = string.Join(", ", context.Select(kvp => $"{kvp.Key}={kvp.Value}"));
            logEntry += $" | Context: {contextStr}";
        }

        Console.WriteLine(logEntry);
    }

    public IDisposable BeginScope(string correlationId)
    {
        _correlationStack.Push(correlationId);
        return new CorrelationScope(() => _correlationStack.Pop());
    }

    private class CorrelationScope : IDisposable
    {
        private readonly Action _onDispose;

        public CorrelationScope(Action onDispose)
        {
            _onDispose = onDispose;
        }

        public void Dispose()
        {
            _onDispose();
        }
    }
}

/// <summary>
/// Subsystem orchestrator for coordinated initialization and health monitoring
/// </summary>
public class SubsystemOrchestrator
{
    private readonly List<ISubsystem> _subsystems = new();
    private readonly IEventBus _eventBus;
    private readonly IMetricsCollector _metrics;
    private readonly ICorrelatedLogger _logger;

    public SubsystemOrchestrator(
        IEventBus eventBus,
        IMetricsCollector metrics,
        ICorrelatedLogger logger)
    {
        _eventBus = eventBus;
        _metrics = metrics;
        _logger = logger;
    }

    public void RegisterSubsystem(ISubsystem subsystem)
    {
        _subsystems.Add(subsystem);
        _logger.LogWithContext($"Registered subsystem: {subsystem.Name}", LogLevel.Information);
    }

    public async Task InitializeAllAsync()
    {
        _logger.LogWithContext("Initializing all subsystems...", LogLevel.Information);

        var tasks = _subsystems.Select(async subsystem =>
        {
            try
            {
                var startTime = DateTime.UtcNow;
                await subsystem.InitializeAsync();
                var duration = (DateTime.UtcNow - startTime).TotalMilliseconds;

                _metrics.RecordMetric($"subsystem.{subsystem.Name}.init_time", duration);
                _logger.LogWithContext(
                    $"Initialized {subsystem.Name}",
                    LogLevel.Information,
                    new Dictionary<string, object> { ["duration_ms"] = duration });

                await _eventBus.PublishAsync(new SystemHealthChangedEvent
                {
                    SubsystemName = subsystem.Name,
                    IsHealthy = true,
                    Status = "Initialized"
                });
            }
            catch (Exception ex)
            {
                _logger.LogWithContext(
                    $"Failed to initialize {subsystem.Name}: {ex.Message}",
                    LogLevel.Error);

                await _eventBus.PublishAsync(new SystemHealthChangedEvent
                {
                    SubsystemName = subsystem.Name,
                    IsHealthy = false,
                    Status = $"Initialization Failed: {ex.Message}"
                });
            }
        });

        await Task.WhenAll(tasks);
    }

    public async Task ShutdownAllAsync()
    {
        _logger.LogWithContext("Shutting down all subsystems...", LogLevel.Information);

        var tasks = _subsystems.Select(async subsystem =>
        {
            try
            {
                await subsystem.ShutdownAsync();
                _logger.LogWithContext($"Shutdown {subsystem.Name}", LogLevel.Information);
            }
            catch (Exception ex)
            {
                _logger.LogWithContext(
                    $"Error shutting down {subsystem.Name}: {ex.Message}",
                    LogLevel.Error);
            }
        });

        await Task.WhenAll(tasks);
    }

    public async Task<Dictionary<string, HealthStatus>> GetHealthStatusAsync()
    {
        var healthStatuses = new Dictionary<string, HealthStatus>();

        var tasks = _subsystems.Select(async subsystem =>
        {
            try
            {
                var health = await subsystem.GetHealthAsync();
                return (subsystem.Name, health);
            }
            catch (Exception ex)
            {
                return (subsystem.Name, new HealthStatus
                {
                    IsHealthy = false,
                    Status = $"Error: {ex.Message}"
                });
            }
        });

        var results = await Task.WhenAll(tasks);

        foreach (var (name, health) in results)
        {
            healthStatuses[name] = health;
        }

        return healthStatuses;
    }

    public async Task MonitorHealthAsync(TimeSpan interval)
    {
        while (true)
        {
            var healthStatuses = await GetHealthStatusAsync();

            foreach (var kvp in healthStatuses)
            {
                _metrics.RecordMetric(
                    $"subsystem.{kvp.Key}.health",
                    kvp.Value.IsHealthy ? 1.0 : 0.0);

                if (!kvp.Value.IsHealthy)
                {
                    _logger.LogWithContext(
                        $"Subsystem {kvp.Key} is unhealthy: {kvp.Value.Status}",
                        LogLevel.Warning);
                }
            }

            await Task.Delay(interval);
        }
    }
}

/// <summary>
/// Simple NLP processor implementation
/// </summary>
public class NLPProcessor : INLPProcessor
{
    public string Name => "NLPProcessor";
    public bool IsInitialized { get; private set; }

    public async Task InitializeAsync()
    {
        IsInitialized = true;
        await Task.CompletedTask;
    }

    public async Task ShutdownAsync()
    {
        IsInitialized = false;
        await Task.CompletedTask;
    }

    public async Task<HealthStatus> GetHealthAsync()
    {
        return await Task.FromResult(new HealthStatus
        {
            IsHealthy = IsInitialized,
            Status = IsInitialized ? "Operational" : "Not Initialized"
        });
    }

    public async Task<string[]> TokenizeAsync(string text)
    {
        // Simple whitespace tokenization
        return await Task.FromResult(text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries));
    }

    public async Task<SentimentResult> AnalyzeSentimentAsync(string text)
    {
        // Simplified sentiment analysis (would use trained model in production)
        var positiveWords = new[] { "good", "great", "excellent", "amazing", "wonderful", "fantastic", "love" };
        var negativeWords = new[] { "bad", "terrible", "awful", "hate", "horrible", "worst", "poor" };

        var tokens = text.ToLower().Split(' ');
        int positiveCount = tokens.Count(t => positiveWords.Contains(t));
        int negativeCount = tokens.Count(t => negativeWords.Contains(t));

        double score = (positiveCount - negativeCount) / (double)Math.Max(tokens.Length, 1);

        return await Task.FromResult(new SentimentResult
        {
            Sentiment = score > 0.1 ? "Positive" : score < -0.1 ? "Negative" : "Neutral",
            Score = Math.Max(-1, Math.Min(1, score)),
            DetailedScores = new Dictionary<string, double>
            {
                ["Positive"] = positiveCount / (double)Math.Max(tokens.Length, 1),
                ["Negative"] = negativeCount / (double)Math.Max(tokens.Length, 1)
            }
        });
    }

    public async Task<IEnumerable<Entity>> ExtractEntitiesAsync(string text)
    {
        // Simplified NER (would use trained model in production)
        var entities = new List<Entity>();

        // Find capitalized words as potential entities
        var words = text.Split(' ');
        for (int i = 0; i < words.Length; i++)
        {
            if (words[i].Length > 0 && char.IsUpper(words[i][0]))
            {
                entities.Add(new Entity
                {
                    Text = words[i],
                    Type = "UNKNOWN",
                    StartIndex = i,
                    EndIndex = i,
                    Confidence = 0.5
                });
            }
        }

        return await Task.FromResult(entities);
    }

    public async Task<Intent> ClassifyIntentAsync(string text)
    {
        // Simplified intent classification
        var lowerText = text.ToLower();

        if (lowerText.Contains("search") || lowerText.Contains("find"))
        {
            return await Task.FromResult(new Intent
            {
                Name = "Search",
                Confidence = 0.9,
                Slots = new Dictionary<string, object>
                {
                    ["query"] = text.Replace("search", "").Replace("find", "").Trim()
                }
            });
        }

        return await Task.FromResult(new Intent
        {
            Name = "Unknown",
            Confidence = 0.1
        });
    }
}
}
