using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Drawing;
using System.Runtime.CompilerServices;
using AWIS.Core;

namespace AWIS.Performance;

/// <summary>
/// Central object pool manager for high-performance reuse
/// </summary>
public class ObjectPoolManager
{
    private static readonly Lazy<ObjectPoolManager> _instance = new(() => new ObjectPoolManager());
    public static ObjectPoolManager Instance => _instance.Value;

    public BitmapPool Bitmaps { get; } = new();
    public ByteArrayPool ByteArrays { get; } = new();
    public DoubleArrayPool DoubleArrays { get; } = new();
    public NeuralNetworkBufferPool NeuralNetworkBuffers { get; } = new();

    private readonly IMetricsCollector? _metrics;

    public ObjectPoolManager(IMetricsCollector? metrics = null)
    {
        _metrics = metrics;
    }

    /// <summary>
    /// Get pooling statistics
    /// </summary>
    public PoolingStatistics GetStatistics()
    {
        return new PoolingStatistics
        {
            BitmapPoolSize = Bitmaps.PooledCount,
            ByteArrayPoolSize = ByteArrays.PooledCount,
            DoubleArrayPoolSize = DoubleArrays.PooledCount,
            NNBufferPoolSize = NeuralNetworkBuffers.PooledCount,
            BitmapRentCount = Bitmaps.RentCount,
            ByteArrayRentCount = ByteArrays.RentCount,
            DoubleArrayRentCount = DoubleArrays.RentCount,
            NNBufferRentCount = NeuralNetworkBuffers.RentCount
        };
    }
}

public class PoolingStatistics
{
    public int BitmapPoolSize { get; set; }
    public int ByteArrayPoolSize { get; set; }
    public int DoubleArrayPoolSize { get; set; }
    public int NNBufferPoolSize { get; set; }
    public long BitmapRentCount { get; set; }
    public long ByteArrayRentCount { get; set; }
    public long DoubleArrayRentCount { get; set; }
    public long NNBufferRentCount { get; set; }
}

/// <summary>
/// Pool for Bitmap objects (expensive to allocate)
/// </summary>
public class BitmapPool
{
    private readonly ConcurrentBag<PooledBitmap> _pool = new();
    private long _rentCount = 0;
    private long _returnCount = 0;
    private readonly int _maxPoolSize = 100;

    public int PooledCount => _pool.Count;
    public long RentCount => _rentCount;
    public long ReturnCount => _returnCount;

    /// <summary>
    /// Rent a bitmap from the pool
    /// </summary>
    public PooledBitmap Rent(int width, int height, System.Drawing.Imaging.PixelFormat format = System.Drawing.Imaging.PixelFormat.Format24bppRgb)
    {
        Interlocked.Increment(ref _rentCount);

        // Try to get from pool
        while (_pool.TryTake(out var pooled))
        {
            if (pooled.Bitmap.Width == width &&
                pooled.Bitmap.Height == height &&
                pooled.Bitmap.PixelFormat == format &&
                !pooled.IsDisposed)
            {
                pooled.Reset();
                return pooled;
            }
            else
            {
                // Size mismatch or disposed, discard
                pooled.Bitmap.Dispose();
            }
        }

        // Create new
        var bitmap = new Bitmap(width, height, format);
        return new PooledBitmap(bitmap, this);
    }

    /// <summary>
    /// Return a bitmap to the pool
    /// </summary>
    internal void Return(PooledBitmap pooled)
    {
        if (pooled.IsDisposed)
            return;

        Interlocked.Increment(ref _returnCount);

        // Only pool if under capacity
        if (_pool.Count < _maxPoolSize)
        {
            _pool.Add(pooled);
        }
        else
        {
            pooled.Bitmap.Dispose();
        }
    }
}

/// <summary>
/// Pooled bitmap wrapper that returns to pool on dispose
/// </summary>
public class PooledBitmap : IDisposable
{
    public Bitmap Bitmap { get; }
    private readonly BitmapPool _pool;
    internal bool IsDisposed { get; private set; }

    public PooledBitmap(Bitmap bitmap, BitmapPool pool)
    {
        Bitmap = bitmap;
        _pool = pool;
    }

    internal void Reset()
    {
        IsDisposed = false;
    }

    public void Dispose()
    {
        if (!IsDisposed)
        {
            IsDisposed = true;
            _pool.Return(this);
        }
    }
}

/// <summary>
/// Pool for byte arrays using ArrayPool<T>
/// </summary>
public class ByteArrayPool
{
    private readonly ArrayPool<byte> _pool = ArrayPool<byte>.Shared;
    private long _rentCount = 0;

    public int PooledCount => 0; // ArrayPool doesn't expose count
    public long RentCount => _rentCount;

    /// <summary>
    /// Rent a byte array
    /// </summary>
    public byte[] Rent(int minimumLength)
    {
        Interlocked.Increment(ref _rentCount);
        return _pool.Rent(minimumLength);
    }

    /// <summary>
    /// Return a byte array
    /// </summary>
    public void Return(byte[] array, bool clearArray = false)
    {
        _pool.Return(array, clearArray);
    }

    /// <summary>
    /// Rent with automatic return on dispose
    /// </summary>
    public PooledArray<byte> RentScoped(int minimumLength)
    {
        return new PooledArray<byte>(Rent(minimumLength), _pool);
    }
}

/// <summary>
/// Pool for double arrays (for ML math)
/// </summary>
public class DoubleArrayPool
{
    private readonly ArrayPool<double> _pool = ArrayPool<double>.Shared;
    private long _rentCount = 0;

    public int PooledCount => 0;
    public long RentCount => _rentCount;

    public double[] Rent(int minimumLength)
    {
        Interlocked.Increment(ref _rentCount);
        return _pool.Rent(minimumLength);
    }

    public void Return(double[] array, bool clearArray = false)
    {
        _pool.Return(array, clearArray);
    }

    public PooledArray<double> RentScoped(int minimumLength)
    {
        return new PooledArray<double>(Rent(minimumLength), _pool);
    }
}

/// <summary>
/// RAII wrapper for pooled arrays
/// </summary>
public struct PooledArray<T> : IDisposable
{
    public T[] Array { get; }
    private readonly ArrayPool<T> _pool;

    public PooledArray(T[] array, ArrayPool<T> pool)
    {
        Array = array;
        _pool = pool;
    }

    public Span<T> AsSpan() => Array.AsSpan();
    public Memory<T> AsMemory() => Array.AsMemory();

    public void Dispose()
    {
        _pool.Return(Array);
    }
}

/// <summary>
/// Pool for neural network computation buffers
/// </summary>
public class NeuralNetworkBufferPool
{
    private readonly ConcurrentDictionary<int, ConcurrentBag<double[]>> _pools = new();
    private long _rentCount = 0;
    private readonly int _maxPooledSize = 10000; // Don't pool arrays larger than this

    public int PooledCount => _pools.Values.Sum(bag => bag.Count);
    public long RentCount => _rentCount;

    /// <summary>
    /// Rent a buffer for layer computation
    /// </summary>
    public double[] RentLayerBuffer(int size)
    {
        Interlocked.Increment(ref _rentCount);

        if (size > _maxPooledSize)
        {
            // Too large to pool
            return new double[size];
        }

        var pool = _pools.GetOrAdd(size, _ => new ConcurrentBag<double[]>());

        if (pool.TryTake(out var buffer))
        {
            Array.Clear(buffer, 0, buffer.Length);
            return buffer;
        }

        return new double[size];
    }

    /// <summary>
    /// Return a buffer
    /// </summary>
    public void ReturnLayerBuffer(double[] buffer)
    {
        if (buffer.Length > _maxPooledSize)
        {
            // Too large, don't pool
            return;
        }

        var pool = _pools.GetOrAdd(buffer.Length, _ => new ConcurrentBag<double[]>());

        // Limit pool size
        if (pool.Count < 10)
        {
            pool.Add(buffer);
        }
    }

    /// <summary>
    /// Rent with automatic return
    /// </summary>
    public PooledNNBuffer RentScoped(int size)
    {
        return new PooledNNBuffer(RentLayerBuffer(size), this);
    }
}

/// <summary>
/// RAII wrapper for NN buffers
/// </summary>
public struct PooledNNBuffer : IDisposable
{
    public double[] Buffer { get; }
    private readonly NeuralNetworkBufferPool _pool;

    public PooledNNBuffer(double[] buffer, NeuralNetworkBufferPool pool)
    {
        Buffer = buffer;
        _pool = pool;
    }

    public Span<double> AsSpan() => Buffer.AsSpan();

    public void Dispose()
    {
        _pool.ReturnLayerBuffer(Buffer);
    }
}

/// <summary>
/// Zero-allocation fast paths using value types and spans
/// </summary>
public static class ZeroAllocHelpers
{
    /// <summary>
    /// Softmax using Span<T> (zero-alloc)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SoftmaxInPlace(Span<double> values)
    {
        if (values.Length == 0) return;

        // Find max for numerical stability
        double max = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > max)
                max = values[i];
        }

        // Compute exp and sum
        double sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Math.Exp(values[i] - max);
            sum += values[i];
        }

        // Normalize
        for (int i = 0; i < values.Length; i++)
        {
            values[i] /= sum;
        }
    }

    /// <summary>
    /// ReLU activation in-place
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ReLUInPlace(Span<double> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] < 0)
                values[i] = 0;
        }
    }

    /// <summary>
    /// Sigmoid activation in-place
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SigmoidInPlace(Span<double> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = 1.0 / (1.0 + Math.Exp(-values[i]));
        }
    }

    /// <summary>
    /// Dot product using Span<T>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double DotProduct(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /// <summary>
    /// Matrix-vector multiplication (zero-alloc)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void MatrixVectorMultiply(
        ReadOnlySpan<double> matrix,
        int rows,
        int cols,
        ReadOnlySpan<double> vector,
        Span<double> result)
    {
        for (int i = 0; i < rows; i++)
        {
            double sum = 0;
            var rowStart = i * cols;
            for (int j = 0; j < cols; j++)
            {
                sum += matrix[rowStart + j] * vector[j];
            }
            result[i] = sum;
        }
    }

    /// <summary>
    /// Add bias in-place
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddBias(Span<double> values, ReadOnlySpan<double> bias)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] += bias[i];
        }
    }
}

/// <summary>
/// Readonly struct for configuration (zero heap allocation)
/// </summary>
public readonly struct NeuralNetworkConfig
{
    public readonly int InputSize;
    public readonly int HiddenSize;
    public readonly int OutputSize;
    public readonly double LearningRate;
    public readonly int BatchSize;

    public NeuralNetworkConfig(
        int inputSize,
        int hiddenSize,
        int outputSize,
        double learningRate = 0.001,
        int batchSize = 32)
    {
        InputSize = inputSize;
        HiddenSize = hiddenSize;
        OutputSize = outputSize;
        LearningRate = learningRate;
        BatchSize = batchSize;
    }
}

/// <summary>
/// Benchmarking utilities for pooling verification
/// </summary>
public class PoolingBenchmark
{
    public static void VerifyReuse()
    {
        var poolMgr = ObjectPoolManager.Instance;

        Console.WriteLine("=== Pooling Benchmark ===");

        // Test bitmap pooling
        {
            var stats = poolMgr.GetStatistics();
            var initialRent = stats.BitmapRentCount;

            using (var bitmap1 = poolMgr.Bitmaps.Rent(640, 480))
            using (var bitmap2 = poolMgr.Bitmaps.Rent(640, 480))
            {
                // Use bitmaps
            }

            using (var bitmap3 = poolMgr.Bitmaps.Rent(640, 480))
            {
                // Should reuse from pool
            }

            stats = poolMgr.GetStatistics();
            Console.WriteLine($"Bitmap pool: {stats.BitmapRentCount - initialRent} rents, {stats.BitmapPoolSize} in pool");
        }

        // Test array pooling
        {
            var stats = poolMgr.GetStatistics();
            var initialRent = stats.DoubleArrayRentCount;

            using (var arr1 = poolMgr.DoubleArrays.RentScoped(1000))
            using (var arr2 = poolMgr.DoubleArrays.RentScoped(1000))
            {
                // Use arrays
            }

            stats = poolMgr.GetStatistics();
            Console.WriteLine($"DoubleArray pool: {stats.DoubleArrayRentCount - initialRent} rents");
        }

        // Test NN buffer pooling
        {
            var stats = poolMgr.GetStatistics();
            var initialRent = stats.NNBufferRentCount;

            for (int i = 0; i < 100; i++)
            {
                using (var buffer = poolMgr.NeuralNetworkBuffers.RentScoped(256))
                {
                    // Simulate NN computation
                    ZeroAllocHelpers.ReLUInPlace(buffer.AsSpan());
                }
            }

            stats = poolMgr.GetStatistics();
            Console.WriteLine($"NN buffer pool: {stats.NNBufferRentCount - initialRent} rents, {stats.NNBufferPoolSize} in pool");
        }
    }
}
