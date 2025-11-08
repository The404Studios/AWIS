using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using AWIS.Core;

namespace AWIS.Security;

/// <summary>
/// Secrets manager using Windows DPAPI for credential storage
/// </summary>
public class SecretsManager
{
    private readonly ConcurrentDictionary<string, SecretEntry> _cache = new();
    private readonly ICorrelatedLogger _logger;
    private readonly string _storePath;
    private readonly SecretRotationScheduler _rotationScheduler;

    public SecretsManager(ICorrelatedLogger logger, string? storePath = null)
    {
        _logger = logger;
        _storePath = storePath ?? System.IO.Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "AWIS",
            "secrets.dat");

        _rotationScheduler = new SecretRotationScheduler(this, logger);

        // Ensure directory exists
        var dir = System.IO.Path.GetDirectoryName(_storePath);
        if (!string.IsNullOrEmpty(dir) && !System.IO.Directory.Exists(dir))
        {
            System.IO.Directory.CreateDirectory(dir);
        }
    }

    /// <summary>
    /// Store a secret using DPAPI
    /// </summary>
    public async Task SetSecretAsync(string key, string value, TimeSpan? rotationPeriod = null)
    {
        if (string.IsNullOrEmpty(key))
            throw new ArgumentException("Key cannot be empty", nameof(key));

        // Encrypt using DPAPI (Windows only)
        byte[] encrypted;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var plainBytes = Encoding.UTF8.GetBytes(value);
            encrypted = ProtectedData.Protect(plainBytes, null, DataProtectionScope.CurrentUser);
        }
        else
        {
            // Fallback for non-Windows: use AES with machine key
            encrypted = EncryptWithMachineKey(value);
        }

        var entry = new SecretEntry
        {
            Key = key,
            EncryptedValue = encrypted,
            CreatedAt = DateTime.UtcNow,
            LastAccessedAt = DateTime.UtcNow,
            RotationPeriod = rotationPeriod
        };

        _cache[key] = entry;

        // Persist to disk
        await PersistAsync();

        // Schedule rotation if specified
        if (rotationPeriod.HasValue)
        {
            _rotationScheduler.ScheduleRotation(key, rotationPeriod.Value);
        }

        _logger.LogWithContext(
            $"Secret stored: {key}",
            LogLevel.Information,
            new Dictionary<string, object>
            {
                ["RotationPeriod"] = rotationPeriod?.ToString() ?? "None"
            });
    }

    /// <summary>
    /// Retrieve a secret
    /// </summary>
    public async Task<string> GetSecretAsync(string key)
    {
        if (!_cache.TryGetValue(key, out var entry))
        {
            // Try to load from disk
            await LoadAsync();

            if (!_cache.TryGetValue(key, out entry))
            {
                throw new KeyNotFoundException($"Secret not found: {key}");
            }
        }

        // Decrypt
        string value;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var plainBytes = ProtectedData.Unprotect(entry.EncryptedValue, null, DataProtectionScope.CurrentUser);
            value = Encoding.UTF8.GetString(plainBytes);
        }
        else
        {
            value = DecryptWithMachineKey(entry.EncryptedValue);
        }

        // Update last accessed
        entry.LastAccessedAt = DateTime.UtcNow;
        entry.AccessCount++;

        return value;
    }

    /// <summary>
    /// Retrieve a secret as SecureString (for safer handling)
    /// </summary>
    public async Task<SecureString> GetSecureStringAsync(string key)
    {
        var value = await GetSecretAsync(key);

        var secure = new SecureString();
        foreach (var c in value)
        {
            secure.AppendChar(c);
        }

        secure.MakeReadOnly();

        // Clear the string from memory
        value = null!;
        GC.Collect();

        return secure;
    }

    /// <summary>
    /// Delete a secret
    /// </summary>
    public async Task DeleteSecretAsync(string key)
    {
        if (_cache.TryRemove(key, out var entry))
        {
            // Zero out encrypted data
            Array.Clear(entry.EncryptedValue, 0, entry.EncryptedValue.Length);

            await PersistAsync();

            _logger.LogWithContext(
                $"Secret deleted: {key}",
                LogLevel.Information);
        }
    }

    /// <summary>
    /// List all secret keys (not values)
    /// </summary>
    public IEnumerable<string> ListKeys()
    {
        return _cache.Keys;
    }

    /// <summary>
    /// Rotate a secret (generate new value)
    /// </summary>
    public async Task<string> RotateSecretAsync(string key, Func<string>? generator = null)
    {
        var newValue = generator?.Invoke() ?? GenerateRandomSecret();

        if (_cache.TryGetValue(key, out var existing))
        {
            await SetSecretAsync(key, newValue, existing.RotationPeriod);
        }
        else
        {
            await SetSecretAsync(key, newValue);
        }

        _logger.LogWithContext(
            $"Secret rotated: {key}",
            LogLevel.Information);

        return newValue;
    }

    /// <summary>
    /// Persist secrets to disk (encrypted)
    /// </summary>
    private async Task PersistAsync()
    {
        var entries = _cache.Values.ToArray();

        // Serialize to JSON
        var json = JsonSerializer.Serialize(entries, new JsonSerializerOptions
        {
            WriteIndented = false
        });

        // Encrypt entire file
        byte[] encrypted;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var bytes = Encoding.UTF8.GetBytes(json);
            encrypted = ProtectedData.Protect(bytes, null, DataProtectionScope.CurrentUser);
        }
        else
        {
            encrypted = EncryptWithMachineKey(json);
        }

        await System.IO.File.WriteAllBytesAsync(_storePath, encrypted);
    }

    /// <summary>
    /// Load secrets from disk
    /// </summary>
    private async Task LoadAsync()
    {
        if (!System.IO.File.Exists(_storePath))
            return;

        var encrypted = await System.IO.File.ReadAllBytesAsync(_storePath);

        // Decrypt
        string json;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var bytes = ProtectedData.Unprotect(encrypted, null, DataProtectionScope.CurrentUser);
            json = Encoding.UTF8.GetString(bytes);
        }
        else
        {
            json = DecryptWithMachineKey(encrypted);
        }

        var entries = JsonSerializer.Deserialize<SecretEntry[]>(json);

        if (entries != null)
        {
            foreach (var entry in entries)
            {
                _cache[entry.Key] = entry;
            }
        }
    }

    /// <summary>
    /// Generate a cryptographically strong random secret
    /// </summary>
    private string GenerateRandomSecret()
    {
        var bytes = new byte[32];
        using var rng = RandomNumberGenerator.Create();
        rng.GetBytes(bytes);
        return Convert.ToBase64String(bytes);
    }

    /// <summary>
    /// Fallback encryption for non-Windows platforms
    /// </summary>
    private byte[] EncryptWithMachineKey(string plaintext)
    {
        using var aes = Aes.Create();
        aes.Key = GetMachineKey();
        aes.GenerateIV();

        using var encryptor = aes.CreateEncryptor();
        var plainBytes = Encoding.UTF8.GetBytes(plaintext);
        var cipherBytes = encryptor.TransformFinalBlock(plainBytes, 0, plainBytes.Length);

        // Prepend IV
        var result = new byte[aes.IV.Length + cipherBytes.Length];
        Buffer.BlockCopy(aes.IV, 0, result, 0, aes.IV.Length);
        Buffer.BlockCopy(cipherBytes, 0, result, aes.IV.Length, cipherBytes.Length);

        return result;
    }

    private string DecryptWithMachineKey(byte[] encrypted)
    {
        using var aes = Aes.Create();
        aes.Key = GetMachineKey();

        // Extract IV
        var iv = new byte[16];
        Buffer.BlockCopy(encrypted, 0, iv, 0, 16);
        aes.IV = iv;

        // Decrypt
        using var decryptor = aes.CreateDecryptor();
        var cipherBytes = new byte[encrypted.Length - 16];
        Buffer.BlockCopy(encrypted, 16, cipherBytes, 0, cipherBytes.Length);

        var plainBytes = decryptor.TransformFinalBlock(cipherBytes, 0, cipherBytes.Length);
        return Encoding.UTF8.GetString(plainBytes);
    }

    private byte[] GetMachineKey()
    {
        // Derive key from machine ID (stable across reboots)
        var machineId = Environment.MachineName + Environment.UserName;
        using var sha = SHA256.Create();
        return sha.ComputeHash(Encoding.UTF8.GetBytes(machineId));
    }
}

/// <summary>
/// Secret entry metadata
/// </summary>
public class SecretEntry
{
    public string Key { get; set; } = string.Empty;
    public byte[] EncryptedValue { get; set; } = Array.Empty<byte>();
    public DateTime CreatedAt { get; set; }
    public DateTime LastAccessedAt { get; set; }
    public int AccessCount { get; set; }
    public TimeSpan? RotationPeriod { get; set; }
    public DateTime? LastRotatedAt { get; set; }
}

/// <summary>
/// Automatic secret rotation scheduler
/// </summary>
public class SecretRotationScheduler
{
    private readonly SecretsManager _secretsManager;
    private readonly ICorrelatedLogger _logger;
    private readonly Dictionary<string, System.Threading.Timer> _timers = new();

    public SecretRotationScheduler(SecretsManager secretsManager, ICorrelatedLogger logger)
    {
        _secretsManager = secretsManager;
        _logger = logger;
    }

    public void ScheduleRotation(string key, TimeSpan period)
    {
        if (_timers.ContainsKey(key))
        {
            _timers[key].Dispose();
        }

        var timer = new System.Threading.Timer(
            async _ =>
            {
                try
                {
                    await _secretsManager.RotateSecretAsync(key);
                    _logger.LogWithContext(
                        $"Auto-rotated secret: {key}",
                        LogLevel.Information);
                }
                catch (Exception ex)
                {
                    _logger.LogWithContext(
                        $"Failed to rotate secret {key}: {ex.Message}",
                        LogLevel.Error);
                }
            },
            null,
            period,
            period);

        _timers[key] = timer;
    }

    public void Dispose()
    {
        foreach (var timer in _timers.Values)
        {
            timer.Dispose();
        }
        _timers.Clear();
    }
}

/// <summary>
/// Serilog secret filter to prevent secrets from being logged
/// </summary>
public class SecretFilter
{
    private readonly HashSet<string> _secretPatterns = new();

    public SecretFilter()
    {
        // Common secret patterns
        _secretPatterns.Add(@"password");
        _secretPatterns.Add(@"apikey");
        _secretPatterns.Add(@"api_key");
        _secretPatterns.Add(@"secret");
        _secretPatterns.Add(@"token");
        _secretPatterns.Add(@"bearer");
        _secretPatterns.Add(@"authorization");
        _secretPatterns.Add(@"private_key");
        _secretPatterns.Add(@"client_secret");
    }

    /// <summary>
    /// Sanitize log message to remove potential secrets
    /// </summary>
    public string Sanitize(string message)
    {
        foreach (var pattern in _secretPatterns)
        {
            // Replace secret values with [REDACTED]
            var regex = new System.Text.RegularExpressions.Regex(
                $@"{pattern}\s*[=:]\s*['""]?([^'""\\s]+)",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);

            message = regex.Replace(message, $"{pattern}=[REDACTED]");
        }

        return message;
    }

    /// <summary>
    /// Check if a string looks like a secret
    /// </summary>
    public bool LooksLikeSecret(string value)
    {
        // High entropy check
        var entropy = ComputeEntropy(value);
        if (entropy > 4.5) // Typical for random secrets
            return true;

        // Base64 pattern check
        if (System.Text.RegularExpressions.Regex.IsMatch(value, @"^[A-Za-z0-9+/]{20,}={0,2}$"))
            return true;

        // Hex pattern check
        if (System.Text.RegularExpressions.Regex.IsMatch(value, @"^[a-fA-F0-9]{32,}$"))
            return true;

        return false;
    }

    private double ComputeEntropy(string value)
    {
        var frequencies = new Dictionary<char, int>();

        foreach (var c in value)
        {
            frequencies[c] = frequencies.GetValueOrDefault(c, 0) + 1;
        }

        double entropy = 0;
        int length = value.Length;

        foreach (var freq in frequencies.Values)
        {
            double p = (double)freq / length;
            entropy -= p * Math.Log2(p);
        }

        return entropy;
    }
}

/// <summary>
/// Integration with Windows Credential Manager
/// </summary>
public class WindowsCredentialStore
{
    private readonly ICorrelatedLogger _logger;

    public WindowsCredentialStore(ICorrelatedLogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Store credential in Windows Credential Manager
    /// </summary>
    public void StoreCredential(string target, string username, string password)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            _logger.LogWithContext(
                "Windows Credential Manager not available on this platform",
                LogLevel.Warning);
            return;
        }

        try
        {
            // Use Windows Credential Manager via P/Invoke
            var credential = new NativeMethods.CREDENTIAL
            {
                Type = NativeMethods.CRED_TYPE_GENERIC,
                TargetName = target,
                UserName = username,
                CredentialBlob = password,
                Persist = NativeMethods.CRED_PERSIST_LOCAL_MACHINE
            };

            if (!NativeMethods.CredWrite(ref credential, 0))
            {
                throw new InvalidOperationException("Failed to write credential");
            }

            _logger.LogWithContext(
                $"Credential stored in Windows Credential Manager: {target}",
                LogLevel.Information);
        }
        catch (Exception ex)
        {
            _logger.LogWithContext(
                $"Failed to store credential: {ex.Message}",
                LogLevel.Error);
        }
    }

    /// <summary>
    /// Retrieve credential from Windows Credential Manager
    /// </summary>
    public (string username, string password)? GetCredential(string target)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return null;
        }

        try
        {
            if (NativeMethods.CredRead(target, NativeMethods.CRED_TYPE_GENERIC, 0, out var credPtr))
            {
                var credential = Marshal.PtrToStructure<NativeMethods.CREDENTIAL>(credPtr);
                NativeMethods.CredFree(credPtr);

                return (credential.UserName, credential.CredentialBlob);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWithContext(
                $"Failed to read credential: {ex.Message}",
                LogLevel.Error);
        }

        return null;
    }

    /// <summary>
    /// Delete credential from Windows Credential Manager
    /// </summary>
    public void DeleteCredential(string target)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return;
        }

        try
        {
            NativeMethods.CredDelete(target, NativeMethods.CRED_TYPE_GENERIC, 0);

            _logger.LogWithContext(
                $"Credential deleted from Windows Credential Manager: {target}",
                LogLevel.Information);
        }
        catch (Exception ex)
        {
            _logger.LogWithContext(
                $"Failed to delete credential: {ex.Message}",
                LogLevel.Error);
        }
    }

    /// <summary>
    /// Native Windows API methods
    /// </summary>
    private static class NativeMethods
    {
        public const int CRED_TYPE_GENERIC = 1;
        public const int CRED_PERSIST_LOCAL_MACHINE = 2;

        [StructLayout(LayoutKind.Sequential)]
        public struct CREDENTIAL
        {
            public int Flags;
            public int Type;
            [MarshalAs(UnmanagedType.LPWStr)]
            public string TargetName;
            [MarshalAs(UnmanagedType.LPWStr)]
            public string Comment;
            public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
            public int CredentialBlobSize;
            [MarshalAs(UnmanagedType.LPWStr)]
            public string CredentialBlob;
            public int Persist;
            public int AttributeCount;
            public IntPtr Attributes;
            [MarshalAs(UnmanagedType.LPWStr)]
            public string TargetAlias;
            [MarshalAs(UnmanagedType.LPWStr)]
            public string UserName;
        }

        [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        public static extern bool CredWrite([In] ref CREDENTIAL userCredential, [In] uint flags);

        [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        public static extern bool CredRead(string target, int type, int reservedFlag, out IntPtr credentialPtr);

        [DllImport("advapi32.dll", SetLastError = true)]
        public static extern bool CredFree([In] IntPtr cred);

        [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        public static extern bool CredDelete(string target, int type, int flags);
    }
}
