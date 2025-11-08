using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using AWIS.Core;

namespace AWIS.Infrastructure;

/// <summary>
/// Health check endpoints for liveness and readiness
/// </summary>
public class HealthCheckService
{
    private readonly SubsystemOrchestrator _orchestrator;
    private readonly IMetricsCollector _metrics;
    private readonly ICorrelatedLogger _logger;
    private readonly ConcurrentDictionary<string, HealthCheckResult> _cachedResults = new();
    private readonly TimeSpan _cacheTimeout = TimeSpan.FromSeconds(5);

    public HealthCheckService(
        SubsystemOrchestrator orchestrator,
        IMetricsCollector metrics,
        ICorrelatedLogger logger)
    {
        _orchestrator = orchestrator;
        _metrics = metrics;
        _logger = logger;
    }

    /// <summary>
    /// Liveness check - is the application alive (not hung)?
    /// </summary>
    public async Task<LivenessResult> CheckLivenessAsync()
    {
        var stopwatch = Stopwatch.StartNew();

        try
        {
            // Simple liveness: can we respond?
            var result = new LivenessResult
            {
                IsAlive = true,
                Timestamp = DateTime.UtcNow,
                ResponseTimeMs = stopwatch.ElapsedMilliseconds
            };

            _metrics.RecordMetric("health.liveness.check", 1.0);
            _metrics.RecordMetric("health.liveness.response_time", result.ResponseTimeMs);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogWithContext(
                $"Liveness check failed: {ex.Message}",
                LogLevel.Critical);

            return new LivenessResult
            {
                IsAlive = false,
                Timestamp = DateTime.UtcNow,
                Error = ex.Message,
                ResponseTimeMs = stopwatch.ElapsedMilliseconds
            };
        }
    }

    /// <summary>
    /// Readiness check - is the application ready to serve requests?
    /// </summary>
    public async Task<ReadinessResult> CheckReadinessAsync()
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new ReadinessResult
        {
            Timestamp = DateTime.UtcNow
        };

        try
        {
            // Check all subsystems
            var healthStatuses = await _orchestrator.GetHealthStatusAsync();

            result.SubsystemChecks = healthStatuses.ToDictionary(
                kvp => kvp.Key,
                kvp => new SubsystemHealthCheck
                {
                    IsHealthy = kvp.Value.IsHealthy,
                    Status = kvp.Value.Status,
                    Metrics = kvp.Value.Metrics
                });

            // Overall readiness: all subsystems must be healthy
            result.IsReady = result.SubsystemChecks.Values.All(c => c.IsHealthy);
            result.ResponseTimeMs = stopwatch.ElapsedMilliseconds;

            _metrics.RecordMetric("health.readiness.check", result.IsReady ? 1.0 : 0.0);
            _metrics.RecordMetric("health.readiness.response_time", result.ResponseTimeMs);

            // Cache result
            _cachedResults["readiness"] = new HealthCheckResult
            {
                Timestamp = DateTime.UtcNow,
                Data = result
            };

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogWithContext(
                $"Readiness check failed: {ex.Message}",
                LogLevel.Error);

            result.IsReady = false;
            result.Error = ex.Message;
            result.ResponseTimeMs = stopwatch.ElapsedMilliseconds;

            return result;
        }
    }

    /// <summary>
    /// Detailed health check with diagnostics
    /// </summary>
    public async Task<DetailedHealthResult> CheckDetailedHealthAsync()
    {
        var result = new DetailedHealthResult
        {
            Timestamp = DateTime.UtcNow
        };

        // Check readiness
        var readiness = await CheckReadinessAsync();
        result.IsHealthy = readiness.IsReady;
        result.SubsystemChecks = readiness.SubsystemChecks;

        // Collect metrics summary
        var metricsSummary = await _metrics.GetSummaryAsync(TimeSpan.FromMinutes(5));
        result.Metrics = metricsSummary.Metrics.ToDictionary(
            kvp => kvp.Key,
            kvp => new Dictionary<string, object>
            {
                ["Count"] = kvp.Value.Count,
                ["Average"] = kvp.Value.Average,
                ["Min"] = kvp.Value.Min,
                ["Max"] = kvp.Value.Max
            });

        // Add system info
        var process = Process.GetCurrentProcess();
        result.SystemInfo = new Dictionary<string, object>
        {
            ["ProcessId"] = process.Id,
            ["WorkingSet_MB"] = process.WorkingSet64 / 1024.0 / 1024.0,
            ["PrivateMemory_MB"] = process.PrivateMemorySize64 / 1024.0 / 1024.0,
            ["TotalProcessorTime"] = process.TotalProcessorTime.TotalSeconds,
            ["Threads"] = process.Threads.Count,
            ["Uptime"] = (DateTime.UtcNow - Process.GetCurrentProcess().StartTime.ToUniversalTime()).TotalSeconds
        };

        return result;
    }
}

public class LivenessResult
{
    public bool IsAlive { get; set; }
    public DateTime Timestamp { get; set; }
    public long ResponseTimeMs { get; set; }
    public string? Error { get; set; }
}

public class ReadinessResult
{
    public bool IsReady { get; set; }
    public DateTime Timestamp { get; set; }
    public long ResponseTimeMs { get; set; }
    public Dictionary<string, SubsystemHealthCheck> SubsystemChecks { get; set; } = new();
    public string? Error { get; set; }
}

public class SubsystemHealthCheck
{
    public bool IsHealthy { get; set; }
    public string Status { get; set; } = string.Empty;
    public Dictionary<string, object> Metrics { get; set; } = new();
}

public class DetailedHealthResult
{
    public bool IsHealthy { get; set; }
    public DateTime Timestamp { get; set; }
    public Dictionary<string, SubsystemHealthCheck> SubsystemChecks { get; set; } = new();
    public Dictionary<string, Dictionary<string, object>> Metrics { get; set; } = new();
    public Dictionary<string, object> SystemInfo { get; set; } = new();
}

public class HealthCheckResult
{
    public DateTime Timestamp { get; set; }
    public object? Data { get; set; }
}

/// <summary>
/// Watchdog to restart hung subsystems
/// </summary>
public class SubsystemWatchdog
{
    private readonly SubsystemOrchestrator _orchestrator;
    private readonly ICorrelatedLogger _logger;
    private readonly IMetricsCollector _metrics;
    private readonly Dictionary<string, WatchdogState> _states = new();
    private readonly TimeSpan _checkInterval = TimeSpan.FromSeconds(30);
    private readonly TimeSpan _hungThreshold = TimeSpan.FromMinutes(2);
    private CancellationTokenSource? _cts;

    public SubsystemWatchdog(
        SubsystemOrchestrator orchestrator,
        ICorrelatedLogger logger,
        IMetricsCollector metrics)
    {
        _orchestrator = orchestrator;
        _logger = logger;
        _metrics = metrics;
    }

    /// <summary>
    /// Start watchdog monitoring
    /// </summary>
    public void Start(CancellationToken cancellationToken = default)
    {
        _cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

        Task.Run(async () =>
        {
            while (!_cts.Token.IsCancellationRequested)
            {
                try
                {
                    await CheckAndRestartHungSubsystemsAsync();
                }
                catch (Exception ex)
                {
                    _logger.LogWithContext(
                        $"Watchdog check failed: {ex.Message}",
                        LogLevel.Error);
                }

                await Task.Delay(_checkInterval, _cts.Token);
            }
        }, _cts.Token);

        _logger.LogWithContext("Watchdog started", LogLevel.Information);
    }

    /// <summary>
    /// Stop watchdog
    /// </summary>
    public void Stop()
    {
        _cts?.Cancel();
        _logger.LogWithContext("Watchdog stopped", LogLevel.Information);
    }

    private async Task CheckAndRestartHungSubsystemsAsync()
    {
        var healthStatuses = await _orchestrator.GetHealthStatusAsync();

        foreach (var kvp in healthStatuses)
        {
            var subsystemName = kvp.Key;
            var health = kvp.Value;

            if (!_states.ContainsKey(subsystemName))
            {
                _states[subsystemName] = new WatchdogState
                {
                    LastHealthyAt = DateTime.UtcNow
                };
            }

            var state = _states[subsystemName];

            if (health.IsHealthy)
            {
                state.LastHealthyAt = DateTime.UtcNow;
                state.ConsecutiveFailures = 0;
            }
            else
            {
                state.ConsecutiveFailures++;

                var timeSinceHealthy = DateTime.UtcNow - state.LastHealthyAt;

                if (timeSinceHealthy > _hungThreshold && state.ConsecutiveFailures >= 3)
                {
                    _logger.LogWithContext(
                        $"Subsystem {subsystemName} appears hung, attempting restart",
                        LogLevel.Warning,
                        new Dictionary<string, object>
                        {
                            ["TimeSinceHealthy"] = timeSinceHealthy.TotalSeconds,
                            ["ConsecutiveFailures"] = state.ConsecutiveFailures
                        });

                    _metrics.IncrementCounter("watchdog.restart", new Dictionary<string, string>
                    {
                        ["subsystem"] = subsystemName
                    });

                    // Attempt restart (would need subsystem restart capability)
                    // For now, just log
                    state.RestartAttempts++;
                }
            }
        }
    }

    private class WatchdogState
    {
        public DateTime LastHealthyAt { get; set; }
        public int ConsecutiveFailures { get; set; }
        public int RestartAttempts { get; set; }
    }
}

/// <summary>
/// Circuit breaker for network operations
/// </summary>
public class CircuitBreaker
{
    private readonly string _name;
    private readonly int _failureThreshold;
    private readonly TimeSpan _timeout;
    private readonly TimeSpan _resetTimeout;
    private readonly IMetricsCollector _metrics;
    private readonly ICorrelatedLogger _logger;

    private CircuitBreakerState _state = CircuitBreakerState.Closed;
    private int _failureCount = 0;
    private DateTime _lastFailureTime;
    private DateTime _openedAt;

    public CircuitBreaker(
        string name,
        int failureThreshold,
        TimeSpan timeout,
        TimeSpan resetTimeout,
        IMetricsCollector metrics,
        ICorrelatedLogger logger)
    {
        _name = name;
        _failureThreshold = failureThreshold;
        _timeout = timeout;
        _resetTimeout = resetTimeout;
        _metrics = metrics;
        _logger = logger;
    }

    public CircuitBreakerState State => _state;

    /// <summary>
    /// Execute an operation through the circuit breaker
    /// </summary>
    public async Task<T> ExecuteAsync<T>(Func<Task<T>> operation, CancellationToken cancellationToken = default)
    {
        // Check if circuit is open
        if (_state == CircuitBreakerState.Open)
        {
            if (DateTime.UtcNow - _openedAt > _resetTimeout)
            {
                // Try half-open
                _state = CircuitBreakerState.HalfOpen;
                _logger.LogWithContext(
                    $"Circuit breaker {_name} entering half-open state",
                    LogLevel.Information);
            }
            else
            {
                _metrics.IncrementCounter($"circuit_breaker.{_name}.rejected");
                throw new CircuitBreakerOpenException($"Circuit breaker {_name} is open");
            }
        }

        try
        {
            // Execute with timeout
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            cts.CancelAfter(_timeout);

            var result = await operation();

            // Success
            if (_state == CircuitBreakerState.HalfOpen)
            {
                // Reset to closed
                _state = CircuitBreakerState.Closed;
                _failureCount = 0;
                _logger.LogWithContext(
                    $"Circuit breaker {_name} reset to closed",
                    LogLevel.Information);
            }

            _metrics.IncrementCounter($"circuit_breaker.{_name}.success");

            return result;
        }
        catch (Exception ex)
        {
            RecordFailure(ex);
            throw;
        }
    }

    private void RecordFailure(Exception ex)
    {
        _failureCount++;
        _lastFailureTime = DateTime.UtcNow;

        _metrics.IncrementCounter($"circuit_breaker.{_name}.failure");

        if (_state == CircuitBreakerState.HalfOpen)
        {
            // Immediately open
            _state = CircuitBreakerState.Open;
            _openedAt = DateTime.UtcNow;

            _logger.LogWithContext(
                $"Circuit breaker {_name} re-opened from half-open",
                LogLevel.Warning,
                new Dictionary<string, object> { ["Exception"] = ex.Message });
        }
        else if (_failureCount >= _failureThreshold)
        {
            // Trip the breaker
            _state = CircuitBreakerState.Open;
            _openedAt = DateTime.UtcNow;

            _logger.LogWithContext(
                $"Circuit breaker {_name} opened due to {_failureCount} failures",
                LogLevel.Warning,
                new Dictionary<string, object>
                {
                    ["FailureCount"] = _failureCount,
                    ["LastException"] = ex.Message
                });

            _metrics.IncrementCounter($"circuit_breaker.{_name}.opened");
        }
    }

    /// <summary>
    /// Manually reset the circuit breaker
    /// </summary>
    public void Reset()
    {
        _state = CircuitBreakerState.Closed;
        _failureCount = 0;
        _logger.LogWithContext($"Circuit breaker {_name} manually reset", LogLevel.Information);
    }
}

public enum CircuitBreakerState
{
    Closed,   // Normal operation
    Open,     // Failing, rejecting requests
    HalfOpen  // Testing if system recovered
}

public class CircuitBreakerOpenException : Exception
{
    public CircuitBreakerOpenException(string message) : base(message) { }
}

/// <summary>
/// HTTP client with circuit breaker and retry
/// </summary>
public class ResilientHttpClient
{
    private readonly HttpClient _httpClient;
    private readonly CircuitBreaker _circuitBreaker;
    private readonly ICorrelatedLogger _logger;
    private readonly int _maxRetries = 3;

    public ResilientHttpClient(
        HttpClient httpClient,
        IMetricsCollector metrics,
        ICorrelatedLogger logger,
        string name = "http")
    {
        _httpClient = httpClient;
        _logger = logger;
        _circuitBreaker = new CircuitBreaker(
            name,
            failureThreshold: 5,
            timeout: TimeSpan.FromSeconds(30),
            resetTimeout: TimeSpan.FromSeconds(60),
            metrics,
            logger);
    }

    /// <summary>
    /// GET with exponential backoff and circuit breaker
    /// </summary>
    public async Task<HttpResponseMessage> GetAsync(string url, CancellationToken cancellationToken = default)
    {
        return await _circuitBreaker.ExecuteAsync(async () =>
        {
            return await RetryWithBackoffAsync(
                async () => await _httpClient.GetAsync(url, cancellationToken),
                cancellationToken);
        }, cancellationToken);
    }

    /// <summary>
    /// POST with exponential backoff and circuit breaker
    /// </summary>
    public async Task<HttpResponseMessage> PostAsync(
        string url,
        HttpContent content,
        CancellationToken cancellationToken = default)
    {
        return await _circuitBreaker.ExecuteAsync(async () =>
        {
            return await RetryWithBackoffAsync(
                async () => await _httpClient.PostAsync(url, content, cancellationToken),
                cancellationToken);
        }, cancellationToken);
    }

    private async Task<HttpResponseMessage> RetryWithBackoffAsync(
        Func<Task<HttpResponseMessage>> operation,
        CancellationToken cancellationToken)
    {
        int attempt = 0;
        Exception? lastException = null;

        while (attempt < _maxRetries)
        {
            try
            {
                var response = await operation();

                // Retry on 5xx
                if ((int)response.StatusCode >= 500 && attempt < _maxRetries - 1)
                {
                    attempt++;
                    var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt)); // Exponential backoff
                    _logger.LogWithContext(
                        $"HTTP request failed with {response.StatusCode}, retrying in {delay.TotalSeconds}s (attempt {attempt})",
                        LogLevel.Warning);

                    await Task.Delay(delay, cancellationToken);
                    continue;
                }

                return response;
            }
            catch (Exception ex) when (ex is HttpRequestException || ex is TaskCanceledException)
            {
                lastException = ex;
                attempt++;

                if (attempt >= _maxRetries)
                    break;

                var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt));
                _logger.LogWithContext(
                    $"HTTP request failed: {ex.Message}, retrying in {delay.TotalSeconds}s (attempt {attempt})",
                    LogLevel.Warning);

                await Task.Delay(delay, cancellationToken);
            }
        }

        throw lastException ?? new HttpRequestException("All retry attempts failed");
    }
}

/// <summary>
/// Graceful shutdown coordinator
/// </summary>
public class GracefulShutdownCoordinator
{
    private readonly CancellationTokenSource _globalCts = new();
    private readonly List<Func<CancellationToken, Task>> _shutdownCallbacks = new();
    private readonly ICorrelatedLogger _logger;
    private readonly TimeSpan _shutdownTimeout = TimeSpan.FromSeconds(30);

    public CancellationToken ShutdownToken => _globalCts.Token;

    public GracefulShutdownCoordinator(ICorrelatedLogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Register a shutdown callback
    /// </summary>
    public void RegisterShutdownCallback(Func<CancellationToken, Task> callback)
    {
        _shutdownCallbacks.Add(callback);
    }

    /// <summary>
    /// Initiate graceful shutdown
    /// </summary>
    public async Task ShutdownAsync()
    {
        _logger.LogWithContext("Initiating graceful shutdown", LogLevel.Information);

        // Signal all loops to stop
        _globalCts.Cancel();

        // Execute shutdown callbacks with timeout
        var shutdownTasks = _shutdownCallbacks.Select(callback =>
        {
            return Task.Run(async () =>
            {
                try
                {
                    using var cts = new CancellationTokenSource(_shutdownTimeout);
                    await callback(cts.Token);
                }
                catch (Exception ex)
                {
                    _logger.LogWithContext(
                        $"Shutdown callback failed: {ex.Message}",
                        LogLevel.Error);
                }
            });
        });

        await Task.WhenAll(shutdownTasks);

        _logger.LogWithContext("Graceful shutdown completed", LogLevel.Information);
    }
}
