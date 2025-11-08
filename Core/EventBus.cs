using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace AWIS.Core;

/// <summary>
/// Thread-safe event bus implementation using channels for async message passing
/// </summary>
public class EventBus : IEventBus
{
    private readonly ConcurrentDictionary<Type, List<Delegate>> _subscriptions = new();
    private readonly ConcurrentDictionary<string, Channel<object>> _channels = new();
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    private readonly CancellationTokenSource _cts = new();

    public async Task PublishAsync<T>(T eventData) where T : class
    {
        var eventType = typeof(T);

        // Get all subscribers for this event type
        if (_subscriptions.TryGetValue(eventType, out var handlers))
        {
            var tasks = handlers
                .OfType<Func<T, Task>>()
                .Select(handler => SafeInvokeHandler(handler, eventData));

            await Task.WhenAll(tasks);
        }

        // Also publish to channel if exists
        if (_channels.TryGetValue(eventType.Name, out var channel))
        {
            await channel.Writer.WriteAsync(eventData!);
        }
    }

    public void Subscribe<T>(Func<T, Task> handler) where T : class
    {
        var eventType = typeof(T);

        _subscriptions.AddOrUpdate(
            eventType,
            _ => new List<Delegate> { handler },
            (_, existing) =>
            {
                existing.Add(handler);
                return existing;
            });
    }

    public void Unsubscribe<T>(Func<T, Task> handler) where T : class
    {
        var eventType = typeof(T);

        if (_subscriptions.TryGetValue(eventType, out var handlers))
        {
            handlers.Remove(handler);
        }
    }

    /// <summary>
    /// Create a channel for specific event type
    /// </summary>
    public Channel<T> CreateChannel<T>(int capacity = 100)
    {
        var eventType = typeof(T);
        var channel = Channel.CreateBounded<T>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait
        });

        var objectChannel = Channel.CreateBounded<object>(capacity);
        _channels[eventType.Name] = objectChannel;

        // Bridge typed channel to object channel
        _ = Task.Run(async () =>
        {
            await foreach (var item in channel.Reader.ReadAllAsync(_cts.Token))
            {
                await objectChannel.Writer.WriteAsync(item!);
            }
        });

        return channel;
    }

    /// <summary>
    /// Get statistics about subscriptions
    /// </summary>
    public Dictionary<string, int> GetSubscriptionStats()
    {
        return _subscriptions.ToDictionary(
            kvp => kvp.Key.Name,
            kvp => kvp.Value.Count);
    }

    private async Task SafeInvokeHandler<T>(Func<T, Task> handler, T eventData)
    {
        try
        {
            await handler(eventData);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[EventBus] Handler failed: {ex.Message}");
            // Log but don't throw - one handler failure shouldn't break others
        }
    }

    public void Dispose()
    {
        _cts.Cancel();
        _semaphore.Dispose();

        foreach (var channel in _channels.Values)
        {
            channel.Writer.Complete();
        }
    }
}

/// <summary>
/// Mediator implementation for CQRS pattern
/// </summary>
public class Mediator : IMediator
{
    private readonly IServiceProvider _serviceProvider;
    private readonly IEventBus _eventBus;

    public Mediator(IServiceProvider serviceProvider, IEventBus eventBus)
    {
        _serviceProvider = serviceProvider;
        _eventBus = eventBus;
    }

    public async Task<TResult> SendAsync<TResult>(ICommand<TResult> command)
    {
        var commandType = command.GetType();
        var handlerType = typeof(ICommandHandler<,>).MakeGenericType(commandType, typeof(TResult));

        var handler = _serviceProvider.GetService(handlerType);
        if (handler == null)
        {
            throw new InvalidOperationException($"No handler registered for command {commandType.Name}");
        }

        var method = handlerType.GetMethod("HandleAsync");
        var task = (Task<TResult>)method!.Invoke(handler, new object[] { command })!;

        return await task;
    }

    public async Task<TResult> QueryAsync<TResult>(IQuery<TResult> query)
    {
        var queryType = query.GetType();
        var handlerType = typeof(IQueryHandler<,>).MakeGenericType(queryType, typeof(TResult));

        var handler = _serviceProvider.GetService(handlerType);
        if (handler == null)
        {
            throw new InvalidOperationException($"No handler registered for query {queryType.Name}");
        }

        var method = handlerType.GetMethod("HandleAsync");
        var task = (Task<TResult>)method!.Invoke(handler, new object[] { query })!;

        return await task;
    }

    public async Task PublishAsync<TEvent>(TEvent eventData) where TEvent : class
    {
        await _eventBus.PublishAsync(eventData);
    }
}

/// <summary>
/// Domain events
/// </summary>
public class VoiceCommandRecognizedEvent
{
    public string Command { get; set; } = string.Empty;
    public Dictionary<string, object> Parameters { get; set; } = new();
    public double Confidence { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public class ObjectDetectedEvent
{
    public DetectedObject Object { get; set; } = new();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public class LearningCompletedEvent
{
    public string ModelName { get; set; } = string.Empty;
    public double FinalLoss { get; set; }
    public double Accuracy { get; set; }
    public int Epochs { get; set; }
    public TimeSpan Duration { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public class ActionExecutedEvent
{
    public ActionType ActionType { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public double ExecutionTime { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public class MemoryStoredEvent
{
    public string MemoryId { get; set; } = string.Empty;
    public MemoryType Type { get; set; }
    public double Importance { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public class KnowledgeLearnedEvent
{
    public string Subject { get; set; } = string.Empty;
    public string Predicate { get; set; } = string.Empty;
    public string Object { get; set; } = string.Empty;
    public double Confidence { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public class EmotionalStateChangedEvent
{
    public Dictionary<string, double> EmotionalVector { get; set; } = new();
    public double Valence { get; set; }
    public double Arousal { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public class DecisionMadeEvent
{
    public string Context { get; set; } = string.Empty;
    public ActionType RecommendedAction { get; set; }
    public double Confidence { get; set; }
    public string Rationale { get; set; } = string.Empty;
    public List<ActionType> Alternatives { get; set; } = new();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public class SystemHealthChangedEvent
{
    public string SubsystemName { get; set; } = string.Empty;
    public bool IsHealthy { get; set; }
    public string Status { get; set; } = string.Empty;
    public Dictionary<string, object> Metrics { get; set; } = new();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

public class PerformanceMetricEvent
{
    public string MetricName { get; set; } = string.Empty;
    public double Value { get; set; }
    public Dictionary<string, string> Tags { get; set; } = new();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Event aggregator for complex event processing
/// </summary>
public class EventAggregator
{
    private readonly IEventBus _eventBus;
    private readonly ConcurrentDictionary<string, List<object>> _eventBuffer = new();
    private readonly Timer _flushTimer;

    public EventAggregator(IEventBus eventBus, TimeSpan? flushInterval = null)
    {
        _eventBus = eventBus;
        var interval = flushInterval ?? TimeSpan.FromSeconds(1);
        _flushTimer = new Timer(_ => FlushEvents(), null, interval, interval);
    }

    /// <summary>
    /// Buffer events for aggregation
    /// </summary>
    public void BufferEvent<T>(T eventData, string aggregationKey) where T : class
    {
        _eventBuffer.AddOrUpdate(
            aggregationKey,
            _ => new List<object> { eventData! },
            (_, existing) =>
            {
                existing.Add(eventData!);
                return existing;
            });
    }

    /// <summary>
    /// Flush aggregated events
    /// </summary>
    private void FlushEvents()
    {
        foreach (var kvp in _eventBuffer.ToArray())
        {
            if (_eventBuffer.TryRemove(kvp.Key, out var events) && events.Count > 0)
            {
                _ = PublishAggregatedEvents(kvp.Key, events);
            }
        }
    }

    private async Task PublishAggregatedEvents(string key, List<object> events)
    {
        // Create aggregated event based on key pattern
        if (key.StartsWith("performance_"))
        {
            var perfEvents = events.OfType<PerformanceMetricEvent>().ToList();
            if (perfEvents.Any())
            {
                var aggregated = new AggregatedPerformanceEvent
                {
                    MetricName = perfEvents.First().MetricName,
                    Count = perfEvents.Count,
                    Average = perfEvents.Average(e => e.Value),
                    Min = perfEvents.Min(e => e.Value),
                    Max = perfEvents.Max(e => e.Value),
                    Sum = perfEvents.Sum(e => e.Value),
                    WindowStart = perfEvents.Min(e => e.Timestamp),
                    WindowEnd = perfEvents.Max(e => e.Timestamp)
                };

                await _eventBus.PublishAsync(aggregated);
            }
        }
    }

    public void Dispose()
    {
        _flushTimer?.Dispose();
    }
}

/// <summary>
/// Aggregated performance event
/// </summary>
public class AggregatedPerformanceEvent
{
    public string MetricName { get; set; } = string.Empty;
    public int Count { get; set; }
    public double Average { get; set; }
    public double Min { get; set; }
    public double Max { get; set; }
    public double Sum { get; set; }
    public DateTime WindowStart { get; set; }
    public DateTime WindowEnd { get; set; }
}

/// <summary>
/// Event filter for complex event processing
/// </summary>
public class EventFilter<T> where T : class
{
    private readonly Func<T, bool> _predicate;
    private readonly IEventBus _eventBus;

    public EventFilter(IEventBus eventBus, Func<T, bool> predicate)
    {
        _eventBus = eventBus;
        _predicate = predicate;
    }

    public void Subscribe(Func<T, Task> handler)
    {
        _eventBus.Subscribe<T>(async e =>
        {
            if (_predicate(e))
            {
                await handler(e);
            }
        });
    }
}

/// <summary>
/// Event replay for debugging and testing
/// </summary>
public class EventReplayer
{
    private readonly List<(Type EventType, object Event, DateTime Timestamp)> _recordedEvents = new();
    private bool _isRecording;

    public void StartRecording()
    {
        _isRecording = true;
        _recordedEvents.Clear();
    }

    public void StopRecording()
    {
        _isRecording = false;
    }

    public void RecordEvent<T>(T eventData) where T : class
    {
        if (_isRecording)
        {
            _recordedEvents.Add((typeof(T), eventData, DateTime.UtcNow));
        }
    }

    public async Task ReplayAsync(IEventBus eventBus, double speedMultiplier = 1.0)
    {
        DateTime? lastTimestamp = null;

        foreach (var (eventType, eventData, timestamp) in _recordedEvents)
        {
            if (lastTimestamp.HasValue)
            {
                var delay = timestamp - lastTimestamp.Value;
                var adjustedDelay = TimeSpan.FromMilliseconds(delay.TotalMilliseconds / speedMultiplier);
                if (adjustedDelay.TotalMilliseconds > 0)
                {
                    await Task.Delay(adjustedDelay);
                }
            }

            // Use reflection to call PublishAsync with correct type
            var method = typeof(IEventBus).GetMethod("PublishAsync")!.MakeGenericMethod(eventType);
            await (Task)method.Invoke(eventBus, new[] { eventData })!;

            lastTimestamp = timestamp;
        }
    }

    public IEnumerable<(Type EventType, object Event, DateTime Timestamp)> GetRecordedEvents()
    {
        return _recordedEvents.ToList();
    }
}
