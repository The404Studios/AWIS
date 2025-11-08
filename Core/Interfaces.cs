using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;

namespace AWIS.Core
{

/// <summary>
/// Universal interface for all subsystems
/// </summary>
public interface ISubsystem
{
    string Name { get; }
    bool IsInitialized { get; }
    Task InitializeAsync();
    Task ShutdownAsync();
    Task<HealthStatus> GetHealthAsync();
}

/// <summary>
/// Health status for subsystems
/// </summary>
public class HealthStatus
{
    public bool IsHealthy { get; set; }
    public string Status { get; set; } = string.Empty;
    public Dictionary<string, object> Metrics { get; set; } = new();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Interface for learning capabilities
/// </summary>
public interface ILearnable
{
    Task LearnAsync(object input, object output);
    Task<double> GetConfidenceAsync(object input);
    Task SaveModelAsync(string path);
    Task LoadModelAsync(string path);
}

/// <summary>
/// Interface for perception capabilities
/// </summary>
public interface IPerceptive
{
    Task PerceiveAsync(Bitmap frame);
    Task<IEnumerable<PerceptionResult>> GetPerceptionsAsync();
    Task ClearPerceptionsAsync();
}

/// <summary>
/// Perception result
/// </summary>
public class PerceptionResult
{
    public string Type { get; set; } = string.Empty;
    public object Data { get; set; } = new();
    public double Confidence { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Interface for interactive capabilities
/// </summary>
public interface IInteractive
{
    Task<ActionResult> ExecuteAsync(ActionType action, object? context = null);
    Task<IEnumerable<ActionType>> GetAvailableActionsAsync();
    Task<bool> CanExecuteAsync(ActionType action);
}

/// <summary>
/// Interface for knowledge storage and retrieval
/// </summary>
public interface IKnowledgeStore
{
    Task AddFactAsync(string subject, string predicate, string obj, double confidence = 1.0);
    Task<IEnumerable<KnowledgeFact>> QueryAsync(string subject, string? predicate = null);
    Task<IEnumerable<KnowledgeFact>> InferAsync(string subject, int depth = 2);
    Task<double> GetConfidenceAsync(string subject, string predicate, string obj);
}

/// <summary>
/// Knowledge fact
/// </summary>
public class KnowledgeFact
{
    public string Subject { get; set; } = string.Empty;
    public string Predicate { get; set; } = string.Empty;
    public string Object { get; set; } = string.Empty;
    public double Confidence { get; set; } = 1.0;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Interface for event publishing
/// </summary>
public interface IEventBus
{
    Task PublishAsync<T>(T eventData) where T : class;
    void Subscribe<T>(Func<T, Task> handler) where T : class;
    void Unsubscribe<T>(Func<T, Task> handler) where T : class;
}

/// <summary>
/// Interface for command handling
/// </summary>
public interface ICommandHandler<TCommand, TResult>
{
    Task<TResult> HandleAsync(TCommand command);
}

/// <summary>
/// Interface for query handling
/// </summary>
public interface IQueryHandler<TQuery, TResult>
{
    Task<TResult> HandleAsync(TQuery query);
}

/// <summary>
/// Interface for mediator pattern
/// </summary>
public interface IMediator
{
    Task<TResult> SendAsync<TResult>(ICommand<TResult> command);
    Task<TResult> QueryAsync<TResult>(IQuery<TResult> query);
    Task PublishAsync<TEvent>(TEvent eventData) where TEvent : class;
}

/// <summary>
/// Base command interface
/// </summary>
public interface ICommand<TResult>
{
    string CommandId { get; }
    DateTime Timestamp { get; }
}

/// <summary>
/// Base query interface
/// </summary>
public interface IQuery<TResult>
{
    string QueryId { get; }
    DateTime Timestamp { get; }
}

/// <summary>
/// Interface for neural network operations
/// </summary>
public interface INeuralNetwork : ILearnable
{
    Task<double[]> PredictAsync(double[] input);
    Task TrainAsync(double[][] inputs, double[][] outputs, TrainingConfig config);
    Task<NetworkMetrics> GetMetricsAsync();
}

/// <summary>
/// Training configuration
/// </summary>
public class TrainingConfig
{
    public int Epochs { get; set; } = 100;
    public double LearningRate { get; set; } = 0.001;
    public int BatchSize { get; set; } = 32;
    public double ValidationSplit { get; set; } = 0.2;
    public bool EarlyStopping { get; set; } = true;
    public int EarlyStoppingPatience { get; set; } = 10;
}

/// <summary>
/// Network metrics
/// </summary>
public class NetworkMetrics
{
    public double TrainLoss { get; set; }
    public double ValidationLoss { get; set; }
    public double Accuracy { get; set; }
    public int Epochs { get; set; }
    public TimeSpan TrainingTime { get; set; }
}

/// <summary>
/// Interface for reinforcement learning agents
/// </summary>
public interface IReinforcementAgent : ILearnable
{
    Task<int> SelectActionAsync(double[] state);
    Task UpdateAsync(double[] state, int action, double reward, double[] nextState, bool done);
    Task<AgentMetrics> GetMetricsAsync();
}

/// <summary>
/// Agent metrics
/// </summary>
public class AgentMetrics
{
    public double AverageReward { get; set; }
    public double ExplorationRate { get; set; }
    public int TotalEpisodes { get; set; }
    public int TotalSteps { get; set; }
}

/// <summary>
/// Interface for computer vision operations
/// </summary>
public interface IVisionSystem : IPerceptive, ISubsystem
{
    Task<IEnumerable<DetectedObject>> DetectObjectsAsync(Bitmap image);
    Task<string> ExtractTextAsync(Bitmap image);
    Task<IEnumerable<Face>> DetectFacesAsync(Bitmap image);
    Task<IEnumerable<TrackedObject>> TrackObjectsAsync(Bitmap image);
}

/// <summary>
/// Detected object
/// </summary>
public class DetectedObject
{
    public string Label { get; set; } = string.Empty;
    public Rectangle BoundingBox { get; set; }
    public double Confidence { get; set; }
    public Dictionary<string, object> Properties { get; set; } = new();
}

/// <summary>
/// Face detection result
/// </summary>
public class Face
{
    public Rectangle BoundingBox { get; set; }
    public double Confidence { get; set; }
    public Dictionary<string, double> Emotions { get; set; } = new();
    public int? Age { get; set; }
    public string? Gender { get; set; }
}

/// <summary>
/// Tracked object
/// </summary>
public class TrackedObject
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public DetectedObject Object { get; set; } = new();
    public List<Point> TrajectoryPoints { get; set; } = new();
    public DateTime FirstSeen { get; set; } = DateTime.UtcNow;
    public DateTime LastSeen { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Interface for voice command processing
/// </summary>
public interface IVoiceSystem : ISubsystem
{
    Task<VoiceCommand> ProcessCommandAsync(string text);
    Task RegisterHandlerAsync(string pattern, Func<VoiceCommand, Task> handler);
    Task<IEnumerable<VoiceCommand>> GetCommandHistoryAsync(int limit = 100);
}

/// <summary>
/// Voice command (already defined in Voice namespace, this is the interface version)
/// </summary>
public class VoiceCommandData
{
    public string OriginalText { get; set; } = string.Empty;
    public string Intent { get; set; } = string.Empty;
    public Dictionary<string, object> Parameters { get; set; } = new();
    public double Confidence { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Interface for NLP operations
/// </summary>
public interface INLPProcessor : ISubsystem
{
    Task<string[]> TokenizeAsync(string text);
    Task<SentimentResult> AnalyzeSentimentAsync(string text);
    Task<IEnumerable<Entity>> ExtractEntitiesAsync(string text);
    Task<Intent> ClassifyIntentAsync(string text);
}

/// <summary>
/// Sentiment analysis result
/// </summary>
public class SentimentResult
{
    public string Sentiment { get; set; } = string.Empty; // Positive, Negative, Neutral
    public double Score { get; set; } // -1 to 1
    public Dictionary<string, double> DetailedScores { get; set; } = new();
}

/// <summary>
/// Named entity
/// </summary>
public class Entity
{
    public string Text { get; set; } = string.Empty;
    public string Type { get; set; } = string.Empty; // PERSON, LOCATION, ORGANIZATION, etc.
    public int StartIndex { get; set; }
    public int EndIndex { get; set; }
    public double Confidence { get; set; }
}

/// <summary>
/// Intent classification result
/// </summary>
public class Intent
{
    public string Name { get; set; } = string.Empty;
    public double Confidence { get; set; }
    public Dictionary<string, object> Slots { get; set; } = new();
}

/// <summary>
/// Interface for memory operations
/// </summary>
public interface IMemorySystem : ISubsystem
{
    Task StoreAsync(string content, MemoryType type, double importance = 0.5);
    Task<Memory?> RecallAsync(string query, MemoryType? type = null);
    Task<IEnumerable<Memory>> RecallMultipleAsync(string query, int limit = 10, MemoryType? type = null);
    Task ConsolidateAsync();
}

/// <summary>
/// Memory item
/// </summary>
public class Memory
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Content { get; set; } = string.Empty;
    public MemoryType Type { get; set; }
    public double Importance { get; set; }
    public double Strength { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime LastAccessedAt { get; set; } = DateTime.UtcNow;
    public int AccessCount { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Interface for configuration management
/// </summary>
public interface IConfigurationManager
{
    T GetValue<T>(string key, T defaultValue = default!);
    void SetValue<T>(string key, T value);
    Task LoadAsync(string path);
    Task SaveAsync(string path);
}

/// <summary>
/// Interface for metrics collection
/// </summary>
public interface IMetricsCollector
{
    void RecordMetric(string name, double value, Dictionary<string, string>? tags = null);
    void IncrementCounter(string name, Dictionary<string, string>? tags = null);
    void RecordHistogram(string name, double value, Dictionary<string, string>? tags = null);
    Task<MetricsSummary> GetSummaryAsync(TimeSpan? window = null);
}

/// <summary>
/// Metrics summary
/// </summary>
public class MetricsSummary
{
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public Dictionary<string, MetricData> Metrics { get; set; } = new();
}

/// <summary>
/// Metric data
/// </summary>
public class MetricData
{
    public double Count { get; set; }
    public double Sum { get; set; }
    public double Min { get; set; }
    public double Max { get; set; }
    public double Average => Count > 0 ? Sum / Count : 0;
    public double StandardDeviation { get; set; }
}

/// <summary>
/// Interface for logging with correlation
/// </summary>
public interface ICorrelatedLogger
{
    void LogWithContext(string message, LogLevel level, Dictionary<string, object>? context = null);
    IDisposable BeginScope(string correlationId);
}

/// <summary>
/// Log level
/// </summary>
public enum LogLevel
{
    Trace,
    Debug,
    Information,
    Warning,
    Error,
    Critical
}
}
