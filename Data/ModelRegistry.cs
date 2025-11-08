using System;
using System.Collections.Generic;
using Microsoft.Data.Sqlite;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading.Tasks;
using Dapper;
using AWIS.Core;

namespace AWIS.Data
{

/// <summary>
/// Model registry for ML model governance and versioning
/// </summary>
public class ModelRegistry
{
    private readonly string _connectionString;
    private readonly IEventBus _eventBus;
    private readonly ICorrelatedLogger _logger;

    public ModelRegistry(string databasePath, IEventBus eventBus, ICorrelatedLogger logger)
    {
        _connectionString = $"Data Source={databasePath};Version=3;";
        _eventBus = eventBus;
        _logger = logger;
    }

    public async Task InitializeAsync()
    {
        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync();

        await connection.ExecuteAsync(@"
            CREATE TABLE IF NOT EXISTS model_registry (
                model_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                created_at TEXT NOT NULL,
                created_by TEXT,
                description TEXT,
                weights_hash TEXT NOT NULL,
                weights_path TEXT NOT NULL,
                dataset_hash TEXT,
                dataset_path TEXT,
                training_config TEXT,
                metrics TEXT,
                tags TEXT,
                status TEXT DEFAULT 'active',
                UNIQUE(model_name, version)
            );

            CREATE INDEX IF NOT EXISTS idx_model_name ON model_registry(model_name);
            CREATE INDEX IF NOT EXISTS idx_status ON model_registry(status);
            CREATE INDEX IF NOT EXISTS idx_created_at ON model_registry(created_at);

            CREATE TABLE IF NOT EXISTS model_lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                parent_model_id TEXT,
                relationship TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES model_registry(model_id),
                FOREIGN KEY (parent_model_id) REFERENCES model_registry(model_id)
            );

            CREATE TABLE IF NOT EXISTS model_deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                environment TEXT NOT NULL,
                deployed_at TEXT NOT NULL,
                deployed_by TEXT,
                status TEXT DEFAULT 'active',
                endpoint TEXT,
                FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
            );

            CREATE TABLE IF NOT EXISTS model_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_value REAL NOT NULL,
                measured_at TEXT NOT NULL,
                dataset_name TEXT,
                FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
            );
        ");
    }

    /// <summary>
    /// Register a new model
    /// </summary>
    public async Task<string> RegisterModelAsync(ModelRegistration registration)
    {
        var modelId = Guid.NewGuid().ToString();

        // Compute hash of weights
        var weightsHash = await ComputeFileHashAsync(registration.WeightsPath);

        // Compute dataset hash if provided
        string? datasetHash = null;
        if (!string.IsNullOrEmpty(registration.DatasetPath))
        {
            datasetHash = await ComputeFileHashAsync(registration.DatasetPath);
        }

        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync();

        await connection.ExecuteAsync(@"
            INSERT INTO model_registry (
                model_id, model_name, version, algorithm, created_at, created_by,
                description, weights_hash, weights_path, dataset_hash, dataset_path,
                training_config, metrics, tags
            ) VALUES (
                @ModelId, @ModelName, @Version, @Algorithm, @CreatedAt, @CreatedBy,
                @Description, @WeightsHash, @WeightsPath, @DatasetHash, @DatasetPath,
                @TrainingConfig, @Metrics, @Tags
            )",
            new
            {
                ModelId = modelId,
                registration.ModelName,
                registration.Version,
                registration.Algorithm,
                CreatedAt = DateTime.UtcNow.ToString("O"),
                registration.CreatedBy,
                registration.Description,
                WeightsHash = weightsHash,
                registration.WeightsPath,
                DatasetHash = datasetHash,
                registration.DatasetPath,
                TrainingConfig = JsonSerializer.Serialize(registration.TrainingConfig),
                Metrics = JsonSerializer.Serialize(registration.Metrics),
                Tags = JsonSerializer.Serialize(registration.Tags)
            });

        // Record lineage if parent specified
        if (!string.IsNullOrEmpty(registration.ParentModelId))
        {
            await connection.ExecuteAsync(@"
                INSERT INTO model_lineage (model_id, parent_model_id, relationship, created_at)
                VALUES (@ModelId, @ParentModelId, @Relationship, @CreatedAt)",
                new
                {
                    ModelId = modelId,
                    ParentModelId = registration.ParentModelId,
                    Relationship = "fine_tuned_from",
                    CreatedAt = DateTime.UtcNow.ToString("O")
                });
        }

        _logger.LogWithContext(
            $"Registered model: {registration.ModelName} v{registration.Version}",
            LogLevel.Information,
            new Dictionary<string, object>
            {
                ["ModelId"] = modelId,
                ["Algorithm"] = registration.Algorithm,
                ["WeightsHash"] = weightsHash
            });

        await _eventBus.PublishAsync(new ModelRegisteredEvent
        {
            ModelId = modelId,
            ModelName = registration.ModelName,
            Version = registration.Version,
            Algorithm = registration.Algorithm
        });

        return modelId;
    }

    /// <summary>
    /// Get model by ID
    /// </summary>
    public async Task<ModelRecord?> GetModelAsync(string modelId)
    {
        using var connection = new SqliteConnection(_connectionString);

        var record = await connection.QueryFirstOrDefaultAsync<ModelRecordDto>(@"
            SELECT * FROM model_registry WHERE model_id = @ModelId",
            new { ModelId = modelId });

        return record != null ? MapToModel(record) : null;
    }

    /// <summary>
    /// Get latest version of a model
    /// </summary>
    public async Task<ModelRecord?> GetLatestModelAsync(string modelName)
    {
        using var connection = new SqliteConnection(_connectionString);

        var record = await connection.QueryFirstOrDefaultAsync<ModelRecordDto>(@"
            SELECT * FROM model_registry
            WHERE model_name = @ModelName AND status = 'active'
            ORDER BY created_at DESC
            LIMIT 1",
            new { ModelName = modelName });

        return record != null ? MapToModel(record) : null;
    }

    /// <summary>
    /// List all models
    /// </summary>
    public async Task<IEnumerable<ModelRecord>> ListModelsAsync(string? nameFilter = null)
    {
        using var connection = new SqliteConnection(_connectionString);

        var query = "SELECT * FROM model_registry WHERE status = 'active'";
        if (!string.IsNullOrEmpty(nameFilter))
        {
            query += " AND model_name LIKE @Filter";
        }
        query += " ORDER BY created_at DESC";

        var records = await connection.QueryAsync<ModelRecordDto>(query, new { Filter = $"%{nameFilter}%" });

        return records.Select(MapToModel);
    }

    /// <summary>
    /// Record model quality metric
    /// </summary>
    public async Task RecordQualityMetricAsync(
        string modelId,
        string metricType,
        double metricValue,
        string? datasetName = null)
    {
        using var connection = new SqliteConnection(_connectionString);

        await connection.ExecuteAsync(@"
            INSERT INTO model_quality_metrics (model_id, metric_type, metric_value, measured_at, dataset_name)
            VALUES (@ModelId, @MetricType, @MetricValue, @MeasuredAt, @DatasetName)",
            new
            {
                ModelId = modelId,
                MetricType = metricType,
                MetricValue = metricValue,
                MeasuredAt = DateTime.UtcNow.ToString("O"),
                DatasetName = datasetName
            });
    }

    /// <summary>
    /// Detect drift - check if accuracy dropped
    /// </summary>
    public async Task<bool> DetectDriftAsync(string modelId, string metricType, double threshold, int days = 7)
    {
        using var connection = new SqliteConnection(_connectionString);

        var cutoff = DateTime.UtcNow.AddDays(-days).ToString("O");

        var recentMetrics = await connection.QueryAsync<double>(@"
            SELECT metric_value FROM model_quality_metrics
            WHERE model_id = @ModelId AND metric_type = @MetricType AND measured_at >= @Cutoff
            ORDER BY measured_at DESC",
            new { ModelId = modelId, MetricType = metricType, Cutoff = cutoff });

        if (!recentMetrics.Any())
            return false;

        var values = recentMetrics.ToList();
        if (values.Count < 2)
            return false;

        var latest = values.First();
        var baseline = values.Last();

        var drop = baseline - latest;
        var dropPercentage = (drop / baseline) * 100;

        if (dropPercentage > threshold)
        {
            _logger.LogWithContext(
                $"Drift detected for model {modelId}",
                LogLevel.Warning,
                new Dictionary<string, object>
                {
                    ["MetricType"] = metricType,
                    ["Baseline"] = baseline,
                    ["Latest"] = latest,
                    ["DropPercentage"] = dropPercentage
                });

            return true;
        }

        return false;
    }

    /// <summary>
    /// Export model card (model metadata as JSON)
    /// </summary>
    public async Task<string> ExportModelCardAsync(string modelId)
    {
        var model = await GetModelAsync(modelId);
        if (model == null)
            throw new ArgumentException($"Model not found: {modelId}");

        var card = new
        {
            model.ModelId,
            model.ModelName,
            model.Version,
            model.Algorithm,
            model.CreatedAt,
            model.CreatedBy,
            model.Description,
            model.WeightsHash,
            model.WeightsPath,
            model.DatasetHash,
            model.DatasetPath,
            model.TrainingConfig,
            model.Metrics,
            model.Tags,
            ExportedAt = DateTime.UtcNow
        };

        return JsonSerializer.Serialize(card, new JsonSerializerOptions { WriteIndented = true });
    }

    private async Task<string> ComputeFileHashAsync(string path)
    {
        if (!System.IO.File.Exists(path))
            throw new System.IO.FileNotFoundException($"File not found: {path}");

        using var sha256 = SHA256.Create();
        using var stream = System.IO.File.OpenRead(path);
        var hash = await sha256.ComputeHashAsync(stream);
        return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
    }

    private ModelRecord MapToModel(ModelRecordDto dto)
    {
        return new ModelRecord
        {
            ModelId = dto.model_id,
            ModelName = dto.model_name,
            Version = dto.version,
            Algorithm = dto.algorithm,
            CreatedAt = DateTime.Parse(dto.created_at),
            CreatedBy = dto.created_by,
            Description = dto.description,
            WeightsHash = dto.weights_hash,
            WeightsPath = dto.weights_path,
            DatasetHash = dto.dataset_hash,
            DatasetPath = dto.dataset_path,
            TrainingConfig = string.IsNullOrEmpty(dto.training_config)
                ? new Dictionary<string, object>()
                : JsonSerializer.Deserialize<Dictionary<string, object>>(dto.training_config) ?? new(),
            Metrics = string.IsNullOrEmpty(dto.metrics)
                ? new Dictionary<string, double>()
                : JsonSerializer.Deserialize<Dictionary<string, double>>(dto.metrics) ?? new(),
            Tags = string.IsNullOrEmpty(dto.tags)
                ? new List<string>()
                : JsonSerializer.Deserialize<List<string>>(dto.tags) ?? new(),
            Status = dto.status
        };
    }

    private class ModelRecordDto
    {
        public string model_id { get; set; } = string.Empty;
        public string model_name { get; set; } = string.Empty;
        public string version { get; set; } = string.Empty;
        public string algorithm { get; set; } = string.Empty;
        public string created_at { get; set; } = string.Empty;
        public string? created_by { get; set; }
        public string? description { get; set; }
        public string weights_hash { get; set; } = string.Empty;
        public string weights_path { get; set; } = string.Empty;
        public string? dataset_hash { get; set; }
        public string? dataset_path { get; set; }
        public string? training_config { get; set; }
        public string? metrics { get; set; }
        public string? tags { get; set; }
        public string status { get; set; } = string.Empty;
    }
}

public class ModelRegistration
{
    public string ModelName { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty;
    public string? CreatedBy { get; set; }
    public string? Description { get; set; }
    public string WeightsPath { get; set; } = string.Empty;
    public string? DatasetPath { get; set; }
    public Dictionary<string, object> TrainingConfig { get; set; } = new Dictionary<string, object>();
    public Dictionary<string, double> Metrics { get; set; } = new Dictionary<string, double>();
    public List<string> Tags { get; set; } = new List<string>();
    public string? ParentModelId { get; set; } // For fine-tuning lineage
}

public class ModelRecord
{
    public string ModelId { get; set; } = string.Empty;
    public string ModelName { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public string? CreatedBy { get; set; }
    public string? Description { get; set; }
    public string WeightsHash { get; set; } = string.Empty;
    public string WeightsPath { get; set; } = string.Empty;
    public string? DatasetHash { get; set; }
    public string? DatasetPath { get; set; }
    public Dictionary<string, object> TrainingConfig { get; set; } = new Dictionary<string, object>();
    public Dictionary<string, double> Metrics { get; set; } = new Dictionary<string, double>();
    public List<string> Tags { get; set; } = new List<string>();
    public string Status { get; set; } = string.Empty;
}

public class ModelRegisteredEvent
{
    public string ModelId { get; set; } = string.Empty;
    public string ModelName { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}
}
