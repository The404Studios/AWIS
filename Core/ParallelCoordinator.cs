using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;

namespace AWIS.Core
{
    /// <summary>
    /// Coordinates parallel execution of AI/ML systems
    /// </summary>
    public class ParallelSystemCoordinator
    {
        private readonly int maxDegreeOfParallelism;
        private readonly ConcurrentDictionary<string, object> sharedState;
        private readonly SemaphoreSlim semaphore;

        public ParallelSystemCoordinator(int maxDegreeOfParallelism = -1)
        {
            this.maxDegreeOfParallelism = maxDegreeOfParallelism > 0
                ? maxDegreeOfParallelism
                : Environment.ProcessorCount;
            this.sharedState = new ConcurrentDictionary<string, object>();
            this.semaphore = new SemaphoreSlim(this.maxDegreeOfParallelism);
        }

        public async Task<List<TResult>> ExecuteParallelAsync<TInput, TResult>(
            IEnumerable<TInput> inputs,
            Func<TInput, Task<TResult>> operation,
            CancellationToken cancellationToken = default)
        {
            var results = new ConcurrentBag<TResult>();
            var tasks = new List<Task>();

            foreach (var input in inputs)
            {
                if (cancellationToken.IsCancellationRequested)
                    break;

                await semaphore.WaitAsync(cancellationToken);

                var task = Task.Run(async () =>
                {
                    try
                    {
                        var result = await operation(input);
                        results.Add(result);
                    }
                    finally
                    {
                        semaphore.Release();
                    }
                }, cancellationToken);

                tasks.Add(task);
            }

            await Task.WhenAll(tasks);
            return results.ToList();
        }

        public List<TResult> ExecuteParallel<TInput, TResult>(
            IEnumerable<TInput> inputs,
            Func<TInput, TResult> operation)
        {
            var results = new ConcurrentBag<TResult>();

            Parallel.ForEach(inputs, new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDegreeOfParallelism
            }, input =>
            {
                var result = operation(input);
                results.Add(result);
            });

            return results.ToList();
        }

        public void SetSharedState(string key, object value)
        {
            sharedState[key] = value;
        }

        public T GetSharedState<T>(string key)
        {
            if (sharedState.TryGetValue(key, out var value) && value is T typedValue)
            {
                return typedValue;
            }
            return default;
        }

        public async Task<Dictionary<string, TResult>> ExecuteNamedTasksAsync<TResult>(
            Dictionary<string, Func<Task<TResult>>> namedTasks,
            CancellationToken cancellationToken = default)
        {
            var results = new ConcurrentDictionary<string, TResult>();
            var tasks = namedTasks.Select(async kvp =>
            {
                await semaphore.WaitAsync(cancellationToken);
                try
                {
                    var result = await kvp.Value();
                    results[kvp.Key] = result;
                }
                finally
                {
                    semaphore.Release();
                }
            });

            await Task.WhenAll(tasks);
            return results.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }
    }

    /// <summary>
    /// Pipeline for sequential processing with parallel stages
    /// </summary>
    public class ParallelPipeline<TInput, TOutput>
    {
        private readonly List<Func<object, object>> stages;
        private readonly int maxDegreeOfParallelism;

        public ParallelPipeline(int maxDegreeOfParallelism = -1)
        {
            this.stages = new List<Func<object, object>>();
            this.maxDegreeOfParallelism = maxDegreeOfParallelism > 0
                ? maxDegreeOfParallelism
                : Environment.ProcessorCount;
        }

        public ParallelPipeline<TInput, TOutput> AddStage<TStageInput, TStageOutput>(
            Func<TStageInput, TStageOutput> transform)
        {
            stages.Add(input => transform((TStageInput)input));
            return this;
        }

        public List<TOutput> Execute(IEnumerable<TInput> inputs)
        {
            object currentData = inputs;

            foreach (var stage in stages)
            {
                if (currentData is IEnumerable<object> enumerable)
                {
                    currentData = enumerable.AsParallel()
                        .WithDegreeOfParallelism(maxDegreeOfParallelism)
                        .Select(stage)
                        .ToList();
                }
            }

            return ((IEnumerable<object>)currentData).Cast<TOutput>().ToList();
        }
    }

    /// <summary>
    /// Manages distributed task execution across multiple workers
    /// </summary>
    public class DistributedTaskExecutor
    {
        private readonly int numWorkers;
        private readonly ConcurrentQueue<Action> taskQueue;
        private readonly List<Task> workers;
        private CancellationTokenSource cancellationTokenSource;

        public DistributedTaskExecutor(int numWorkers = -1)
        {
            this.numWorkers = numWorkers > 0 ? numWorkers : Environment.ProcessorCount;
            this.taskQueue = new ConcurrentQueue<Action>();
            this.workers = new List<Task>();
        }

        public void Start()
        {
            cancellationTokenSource = new CancellationTokenSource();
            var token = cancellationTokenSource.Token;

            for (int i = 0; i < numWorkers; i++)
            {
                var worker = Task.Run(() => WorkerLoop(token), token);
                workers.Add(worker);
            }
        }

        private void WorkerLoop(CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                if (taskQueue.TryDequeue(out var task))
                {
                    try
                    {
                        task();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Task failed: {ex.Message}");
                    }
                }
                else
                {
                    Thread.Sleep(10);
                }
            }
        }

        public void EnqueueTask(Action task)
        {
            taskQueue.Enqueue(task);
        }

        public async Task StopAsync()
        {
            cancellationTokenSource?.Cancel();
            if (workers.Count > 0)
            {
                await Task.WhenAll(workers);
            }
            workers.Clear();
        }

        public int QueuedTaskCount => taskQueue.Count;
    }

    /// <summary>
    /// Batch processor for efficient parallel processing
    /// </summary>
    public class BatchProcessor<TInput, TOutput>
    {
        private readonly int batchSize;
        private readonly Func<List<TInput>, List<TOutput>> batchOperation;
        private readonly int maxDegreeOfParallelism;

        public BatchProcessor(
            int batchSize,
            Func<List<TInput>, List<TOutput>> batchOperation,
            int maxDegreeOfParallelism = -1)
        {
            this.batchSize = batchSize;
            this.batchOperation = batchOperation;
            this.maxDegreeOfParallelism = maxDegreeOfParallelism > 0
                ? maxDegreeOfParallelism
                : Environment.ProcessorCount;
        }

        public List<TOutput> Process(IEnumerable<TInput> inputs)
        {
            var batches = inputs
                .Select((item, index) => new { item, index })
                .GroupBy(x => x.index / batchSize)
                .Select(group => group.Select(x => x.item).ToList())
                .ToList();

            var results = new ConcurrentBag<TOutput>();

            Parallel.ForEach(batches, new ParallelOptions
            {
                MaxDegreeOfParallelism = maxDegreeOfParallelism
            }, batch =>
            {
                var batchResults = batchOperation(batch);
                foreach (var result in batchResults)
                {
                    results.Add(result);
                }
            });

            return results.ToList();
        }
    }

    /// <summary>
    /// Performance monitor for parallel operations
    /// </summary>
    public class ParallelPerformanceMonitor
    {
        private readonly ConcurrentDictionary<string, List<long>> operationTimes;
        private readonly ConcurrentDictionary<string, int> operationCounts;
        private readonly Stopwatch globalStopwatch;

        public ParallelPerformanceMonitor()
        {
            operationTimes = new ConcurrentDictionary<string, List<long>>();
            operationCounts = new ConcurrentDictionary<string, int>();
            globalStopwatch = Stopwatch.StartNew();
        }

        public void RecordOperation(string operationName, long elapsedMilliseconds)
        {
            operationTimes.AddOrUpdate(
                operationName,
                new List<long> { elapsedMilliseconds },
                (key, list) =>
                {
                    list.Add(elapsedMilliseconds);
                    return list;
                });

            operationCounts.AddOrUpdate(operationName, 1, (key, count) => count + 1);
        }

        public T MeasureOperation<T>(string operationName, Func<T> operation)
        {
            var sw = Stopwatch.StartNew();
            try
            {
                return operation();
            }
            finally
            {
                sw.Stop();
                RecordOperation(operationName, sw.ElapsedMilliseconds);
            }
        }

        public Dictionary<string, (int count, double avgMs, double totalMs)> GetStatistics()
        {
            return operationTimes.ToDictionary(
                kvp => kvp.Key,
                kvp =>
                {
                    var times = kvp.Value;
                    return (
                        count: times.Count,
                        avgMs: times.Average(),
                        totalMs: times.Sum()
                    );
                });
        }

        public void PrintStatistics()
        {
            Console.WriteLine("\n=== Parallel Performance Statistics ===");
            Console.WriteLine($"Total Runtime: {globalStopwatch.Elapsed.TotalSeconds:F2}s");
            Console.WriteLine();

            var stats = GetStatistics();
            foreach (var kvp in stats.OrderByDescending(x => x.Value.totalMs))
            {
                Console.WriteLine($"{kvp.Key}:");
                Console.WriteLine($"  Count: {kvp.Value.count}");
                Console.WriteLine($"  Avg Time: {kvp.Value.avgMs:F2}ms");
                Console.WriteLine($"  Total Time: {kvp.Value.totalMs:F2}ms");
                Console.WriteLine();
            }
        }
    }

    /// <summary>
    /// Thread-safe result aggregator
    /// </summary>
    public class ResultAggregator<T>
    {
        private readonly ConcurrentBag<T> results;
        private readonly object lockObject = new object();
        private int completedTasks;
        private readonly int totalTasks;

        public ResultAggregator(int totalTasks)
        {
            this.totalTasks = totalTasks;
            this.results = new ConcurrentBag<T>();
            this.completedTasks = 0;
        }

        public void AddResult(T result)
        {
            results.Add(result);
            Interlocked.Increment(ref completedTasks);
        }

        public List<T> GetResults()
        {
            return results.ToList();
        }

        public int CompletedTasks => completedTasks;
        public int TotalTasks => totalTasks;
        public double Progress => (double)completedTasks / totalTasks * 100;
        public bool IsComplete => completedTasks >= totalTasks;
    }
}
