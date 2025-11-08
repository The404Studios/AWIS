using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using AWIS.Core;

namespace AWIS.Security
{
    /// <summary>
    /// Scoped capability tokens for fine-grained permission control
    /// </summary>
    public class CapabilityToken
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string Principal { get; set; } = string.Empty; // User or service identity
        public HashSet<string> Capabilities { get; set; } = new HashSet<string>();
        public DateTime IssuedAt { get; set; } = DateTime.UtcNow;
        public DateTime ExpiresAt { get; set; } = DateTime.UtcNow.AddHours(24);
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
        public string Signature { get; set; } = string.Empty;

    public bool IsExpired => DateTime.UtcNow > ExpiresAt;

    public bool HasCapability(string capability)
    {
        // Support wildcards: "vision.*" matches "vision.capture", "vision.track", etc.
        return Capabilities.Any(cap =>
        {
            if (cap == capability) return true;
            if (cap.EndsWith(".*"))
            {
                var prefix = cap[..^2];
                return capability.StartsWith(prefix + ".");
            }
            return false;
        });
    }
}

/// <summary>
/// Manages capability tokens and enforces permissions
/// </summary>
public class CapabilityManager
{
    private readonly ConcurrentDictionary<string, CapabilityToken> _tokens = new ConcurrentDictionary<string, CapabilityToken>();
    private readonly byte[] _signingKey;
    private readonly ICorrelatedLogger _logger;

    public CapabilityManager(ICorrelatedLogger logger, byte[]? signingKey = null)
    {
        _logger = logger;
        _signingKey = signingKey ?? GenerateKey();
    }

    /// <summary>
    /// Issue a new capability token
    /// </summary>
    public CapabilityToken IssueToken(string principal, IEnumerable<string> capabilities, TimeSpan? lifetime = null)
    {
        var token = new CapabilityToken
        {
            Principal = principal,
            Capabilities = new HashSet<string>(capabilities),
            ExpiresAt = DateTime.UtcNow.Add(lifetime ?? TimeSpan.FromHours(24))
        };

        // Sign the token
        token.Signature = ComputeSignature(token);

        _tokens[token.Id] = token;

        _logger.LogWithContext(
            $"Issued capability token for {principal}",
            LogLevel.Information,
            new Dictionary<string, object>
            {
                ["TokenId"] = token.Id,
                ["Capabilities"] = string.Join(", ", capabilities),
                ["ExpiresAt"] = token.ExpiresAt
            });

        return token;
    }

    /// <summary>
    /// Verify a token and check capability
    /// </summary>
    public bool VerifyCapability(string tokenId, string capability)
    {
        if (!_tokens.TryGetValue(tokenId, out var token))
        {
            _logger.LogWithContext($"Token not found: {tokenId}", LogLevel.Warning);
            return false;
        }

        if (token.IsExpired)
        {
            _logger.LogWithContext($"Token expired: {tokenId}", LogLevel.Warning);
            _tokens.TryRemove(tokenId, out _);
            return false;
        }

        if (!VerifySignature(token))
        {
            _logger.LogWithContext($"Invalid token signature: {tokenId}", LogLevel.Error);
            return false;
        }

        var hasCapability = token.HasCapability(capability);

        if (!hasCapability)
        {
            _logger.LogWithContext(
                $"Capability denied: {capability}",
                LogLevel.Warning,
                new Dictionary<string, object>
                {
                    ["TokenId"] = tokenId,
                    ["Principal"] = token.Principal,
                    ["RequestedCapability"] = capability
                });
        }

        return hasCapability;
    }

    /// <summary>
    /// Revoke a token
    /// </summary>
    public void RevokeToken(string tokenId)
    {
        if (_tokens.TryRemove(tokenId, out var token))
        {
            _logger.LogWithContext(
                $"Revoked token for {token.Principal}",
                LogLevel.Information,
                new Dictionary<string, object> { ["TokenId"] = tokenId });
        }
    }

    /// <summary>
    /// Cleanup expired tokens
    /// </summary>
    public int CleanupExpiredTokens()
    {
        var expired = _tokens.Where(kvp => kvp.Value.IsExpired).Select(kvp => kvp.Key).ToList();

        foreach (var tokenId in expired)
        {
            _tokens.TryRemove(tokenId, out _);
        }

        return expired.Count;
    }

    private string ComputeSignature(CapabilityToken token)
    {
        var data = $"{token.Id}|{token.Principal}|{string.Join(",", token.Capabilities.OrderBy(c => c))}|{token.IssuedAt:O}|{token.ExpiresAt:O}";
        using var hmac = new HMACSHA256(_signingKey);
        var hash = hmac.ComputeHash(Encoding.UTF8.GetBytes(data));
        return Convert.ToBase64String(hash);
    }

    private bool VerifySignature(CapabilityToken token)
    {
        var expected = ComputeSignature(token);
        return token.Signature == expected;
    }

    private static byte[] GenerateKey()
    {
        var key = new byte[32];
        using var rng = RandomNumberGenerator.Create();
        rng.GetBytes(key);
        return key;
    }
}

/// <summary>
/// Policy engine for declarative action control
/// </summary>
public class PolicyEngine
{
    private readonly List<Policy> _policies = new List<Policy>();
    private readonly ICorrelatedLogger _logger;
    private readonly IMetricsCollector _metrics;

    public PolicyEngine(ICorrelatedLogger logger, IMetricsCollector metrics)
    {
        _logger = logger;
        _metrics = metrics;
        LoadDefaultPolicies();
    }

    /// <summary>
    /// Add a policy
    /// </summary>
    public void AddPolicy(Policy policy)
    {
        _policies.Add(policy);
        _logger.LogWithContext(
            $"Added policy: {policy.Name}",
            LogLevel.Information,
            new Dictionary<string, object>
            {
                ["Effect"] = policy.Effect,
                ["Actions"] = string.Join(", ", policy.Actions)
            });
    }

    /// <summary>
    /// Evaluate if an action is allowed
    /// </summary>
    public PolicyResult Evaluate(PolicyRequest request)
    {
        var applicablePolicies = _policies
            .Where(p => p.Matches(request))
            .OrderByDescending(p => p.Priority)
            .ToList();

        // Default deny
        var result = new PolicyResult
        {
            Allowed = false,
            Reason = "No policy grants this action (default deny)",
            Request = request
        };

        foreach (var policy in applicablePolicies)
        {
            if (policy.Effect == PolicyEffect.Deny)
            {
                result.Allowed = false;
                result.Reason = $"Denied by policy: {policy.Name}";
                result.MatchedPolicy = policy;

                _metrics.IncrementCounter("policy.deny", new Dictionary<string, string>
                {
                    ["policy"] = policy.Name,
                    ["action"] = request.Action
                });

                break; // Explicit deny always wins
            }

            if (policy.Effect == PolicyEffect.Allow)
            {
                result.Allowed = true;
                result.Reason = $"Allowed by policy: {policy.Name}";
                result.MatchedPolicy = policy;

                _metrics.IncrementCounter("policy.allow", new Dictionary<string, string>
                {
                    ["policy"] = policy.Name,
                    ["action"] = request.Action
                });
            }
        }

        // Log policy violations
        if (!result.Allowed)
        {
            _logger.LogWithContext(
                $"Policy violation: {request.Action}",
                LogLevel.Warning,
                new Dictionary<string, object>
                {
                    ["Action"] = request.Action,
                    ["Principal"] = request.Principal,
                    ["Reason"] = result.Reason,
                    ["Resource"] = request.Resource ?? "N/A"
                });
        }

        return result;
    }

    /// <summary>
    /// Load default safety policies
    /// </summary>
    private void LoadDefaultPolicies()
    {
        // Never allow posting to social media
        AddPolicy(new Policy
        {
            Name = "BlockSocialMediaPosts",
            Effect = PolicyEffect.Deny,
            Actions = new[] { "web.post", "web.submit", "web.tweet", "web.dm" },
            Conditions = new Dictionary<string, object>
            {
                ["resource_pattern"] = "^(twitter|facebook|instagram|linkedin)\\.com.*"
            },
            Priority = 1000 // High priority
        });

        // Never allow purchases
        AddPolicy(new Policy
        {
            Name = "BlockPurchases",
            Effect = PolicyEffect.Deny,
            Actions = new[] { "web.purchase", "web.checkout", "web.buy" },
            Priority = 1000
        });

        // Require explicit policy for file system writes
        AddPolicy(new Policy
        {
            Name = "RequireExplicitFileWrite",
            Effect = PolicyEffect.Deny,
            Actions = new[] { "fs.write", "fs.delete", "fs.move" },
            Conditions = new Dictionary<string, object>
            {
                ["requires_signed_policy"] = true
            },
            Priority = 900
        });

        // Allow read-only operations by default
        AddPolicy(new Policy
        {
            Name = "AllowReadOperations",
            Effect = PolicyEffect.Allow,
            Actions = new[] { "vision.capture", "vision.detect", "voice.listen", "fs.read", "web.navigate", "web.read" },
            Priority = 100
        });

        // Allow internal AI operations
        AddPolicy(new Policy
        {
            Name = "AllowAIOperations",
            Effect = PolicyEffect.Allow,
            Actions = new[] { "ml.train", "ml.predict", "memory.store", "knowledge.add", "tts.speak" },
            Priority = 100
        });
    }

    /// <summary>
    /// Load policies from JSON file
    /// </summary>
    public async Task LoadPoliciesFromFileAsync(string path)
    {
        if (!System.IO.File.Exists(path))
            return;

        var json = await System.IO.File.ReadAllTextAsync(path);
        var policies = JsonSerializer.Deserialize<List<Policy>>(json);

        if (policies != null)
        {
            foreach (var policy in policies)
            {
                AddPolicy(policy);
            }
        }
    }

    /// <summary>
    /// Export policies to JSON
    /// </summary>
    public async Task ExportPoliciesToFileAsync(string path)
    {
        var json = JsonSerializer.Serialize(_policies, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        await System.IO.File.WriteAllTextAsync(path, json);
    }
}

/// <summary>
/// Policy definition
/// </summary>
public class Policy
{
    public string Name { get; set; } = string.Empty;
    public PolicyEffect Effect { get; set; }
    public string[] Actions { get; set; } = Array.Empty<string>();
    public string[]? Principals { get; set; }
    public string[]? Resources { get; set; }
    public Dictionary<string, object> Conditions { get; set; } = new Dictionary<string, object>();
    public int Priority { get; set; } = 0;

    public bool Matches(PolicyRequest request)
    {
        // Check action
        if (!Actions.Contains(request.Action) && !Actions.Contains("*"))
            return false;

        // Check principal
        if (Principals != null && !Principals.Contains(request.Principal) && !Principals.Contains("*"))
            return false;

        // Check resource
        if (Resources != null && request.Resource != null)
        {
            var matches = Resources.Any(r =>
            {
                if (r.EndsWith("*"))
                {
                    return request.Resource.StartsWith(r[..^1]);
                }
                return r == request.Resource;
            });

            if (!matches)
                return false;
        }

        // Check conditions
        foreach (var condition in Conditions)
        {
            if (condition.Key == "resource_pattern" && request.Resource != null)
            {
                var pattern = condition.Value.ToString();
                if (!System.Text.RegularExpressions.Regex.IsMatch(request.Resource, pattern ?? ""))
                {
                    return false;
                }
            }
        }

        return true;
    }
}

public enum PolicyEffect
{
    Allow,
    Deny
}

/// <summary>
/// Policy evaluation request
/// </summary>
public class PolicyRequest
{
    public string Action { get; set; } = string.Empty;
    public string Principal { get; set; } = string.Empty;
    public string? Resource { get; set; }
    public Dictionary<string, object> Context { get; set; } = new Dictionary<string, object>();
}

/// <summary>
/// Policy evaluation result
/// </summary>
public class PolicyResult
{
    public bool Allowed { get; set; }
    public string Reason { get; set; } = string.Empty;
    public Policy? MatchedPolicy { get; set; }
    public PolicyRequest? Request { get; set; }
}

/// <summary>
/// Web automation guardrails
/// </summary>
public class WebAutomationGuardrails
{
    private readonly HashSet<string> _allowedDomains = new HashSet<string>();
    private readonly Dictionary<string, RateLimiter> _rateLimiters = new Dictionary<string, RateLimiter>();
    private readonly ICorrelatedLogger _logger;
    private readonly PolicyEngine _policyEngine;

    public WebAutomationGuardrails(ICorrelatedLogger logger, PolicyEngine policyEngine)
    {
        _logger = logger;
        _policyEngine = policyEngine;
        LoadDefaultSafelist();
    }

    /// <summary>
    /// Add domain to safelist
    /// </summary>
    public void AllowDomain(string domain)
    {
        _allowedDomains.Add(domain.ToLowerInvariant());
    }

    /// <summary>
    /// Check if navigation is allowed
    /// </summary>
    public bool CanNavigate(string url, string principal)
    {
        var uri = new Uri(url);
        var domain = uri.Host.ToLowerInvariant();

        // Check safelist
        if (!_allowedDomains.Contains(domain) && !_allowedDomains.Contains("*"))
        {
            _logger.LogWithContext(
                $"Blocked navigation to non-safelisted domain: {domain}",
                LogLevel.Warning,
                new Dictionary<string, object>
                {
                    ["Url"] = url,
                    ["Principal"] = principal
                });

            return false;
        }

        // Check policy
        var policyResult = _policyEngine.Evaluate(new PolicyRequest
        {
            Action = "web.navigate",
            Principal = principal,
            Resource = url
        });

        return policyResult.Allowed;
    }

    /// <summary>
    /// Check if click is allowed (with rate limiting)
    /// </summary>
    public bool CanClick(string url, string principal)
    {
        // Rate limiting: max 10 clicks per minute per domain
        var uri = new Uri(url);
        var domain = uri.Host;

        if (!_rateLimiters.TryGetValue(domain, out var limiter))
        {
            limiter = new RateLimiter(maxRequests: 10, window: TimeSpan.FromMinutes(1));
            _rateLimiters[domain] = limiter;
        }

        if (!limiter.TryAcquire())
        {
            _logger.LogWithContext(
                $"Click rate limit exceeded for {domain}",
                LogLevel.Warning,
                new Dictionary<string, object>
                {
                    ["Domain"] = domain,
                    ["Principal"] = principal
                });

            return false;
        }

        // Check policy
        var policyResult = _policyEngine.Evaluate(new PolicyRequest
        {
            Action = "web.click",
            Principal = principal,
            Resource = url
        });

        return policyResult.Allowed;
    }

    /// <summary>
    /// Check if form submission is allowed (CSRF check)
    /// </summary>
    public bool CanSubmitForm(string url, Dictionary<string, string> formData, string principal)
    {
        // Check for dangerous actions
        var dangerousPatterns = new[] { "post", "tweet", "share", "buy", "purchase", "checkout", "delete", "remove" };

        foreach (var key in formData.Keys)
        {
            if (dangerousPatterns.Any(p => key.ToLowerInvariant().Contains(p)))
            {
                _logger.LogWithContext(
                    $"Blocked form submission with dangerous field: {key}",
                    LogLevel.Warning,
                    new Dictionary<string, object>
                    {
                        ["Url"] = url,
                        ["Field"] = key,
                        ["Principal"] = principal
                    });

                return false;
            }
        }

        // Check policy
        var policyResult = _policyEngine.Evaluate(new PolicyRequest
        {
            Action = "web.submit",
            Principal = principal,
            Resource = url,
            Context = new Dictionary<string, object> { ["form_data"] = formData }
        });

        return policyResult.Allowed;
    }

    private void LoadDefaultSafelist()
    {
        // Add common safe domains
        _allowedDomains.Add("google.com");
        _allowedDomains.Add("wikipedia.org");
        _allowedDomains.Add("github.com");
        _allowedDomains.Add("stackoverflow.com");
        _allowedDomains.Add("localhost");
    }
}

/// <summary>
/// Simple token bucket rate limiter
/// </summary>
public class RateLimiter
{
    private readonly int _maxRequests;
    private readonly TimeSpan _window;
    private readonly Queue<DateTime> _timestamps = new Queue<DateTime>();
    private readonly object _lock = new object();

    public RateLimiter(int maxRequests, TimeSpan window)
    {
        _maxRequests = maxRequests;
        _window = window;
    }

    public bool TryAcquire()
    {
        lock (_lock)
        {
            var now = DateTime.UtcNow;
            var cutoff = now - _window;

            // Remove old timestamps
            while (_timestamps.Count > 0 && _timestamps.Peek() < cutoff)
            {
                _timestamps.Dequeue();
            }

            if (_timestamps.Count < _maxRequests)
            {
                _timestamps.Enqueue(now);
                return true;
            }

            return false;
        }
    }
}

/// <summary>
/// Mediator with capability verification
/// </summary>
public class SecureMediator : IMediator
{
    private readonly IMediator _innerMediator;
    private readonly CapabilityManager _capabilityManager;
    private readonly PolicyEngine _policyEngine;
    private readonly ICorrelatedLogger _logger;

    public SecureMediator(
        IMediator innerMediator,
        CapabilityManager capabilityManager,
        PolicyEngine policyEngine,
        ICorrelatedLogger logger)
    {
        _innerMediator = innerMediator;
        _capabilityManager = capabilityManager;
        _policyEngine = policyEngine;
        _logger = logger;
    }

    public async Task<TResult> SendAsync<TResult>(ICommand<TResult> command)
    {
        // Extract capability requirement from command
        var capability = ExtractCapability(command);

        // Get token from command context
        var tokenId = GetTokenFromCommand(command);

        // Verify capability
        if (!string.IsNullOrEmpty(capability))
        {
            if (string.IsNullOrEmpty(tokenId) || !_capabilityManager.VerifyCapability(tokenId, capability))
            {
                _logger.LogWithContext(
                    $"Command rejected: insufficient capability",
                    LogLevel.Error,
                    new Dictionary<string, object>
                    {
                        ["Command"] = command.GetType().Name,
                        ["RequiredCapability"] = capability
                    });

                throw new UnauthorizedAccessException($"Insufficient capability: {capability}");
            }
        }

        // Evaluate policy
        var policyResult = _policyEngine.Evaluate(new PolicyRequest
        {
            Action = ExtractAction(command),
            Principal = GetPrincipalFromCommand(command),
            Resource = ExtractResource(command)
        });

        if (!policyResult.Allowed)
        {
            _logger.LogWithContext(
                $"Command rejected by policy",
                LogLevel.Error,
                new Dictionary<string, object>
                {
                    ["Command"] = command.GetType().Name,
                    ["Reason"] = policyResult.Reason
                });

            throw new UnauthorizedAccessException($"Policy denied: {policyResult.Reason}");
        }

        return await _innerMediator.SendAsync(command);
    }

    public async Task<TResult> QueryAsync<TResult>(IQuery<TResult> query)
    {
        // Queries typically don't need capability checks (read-only)
        return await _innerMediator.QueryAsync(query);
    }

    public async Task PublishAsync<TEvent>(TEvent eventData) where TEvent : class
    {
        await _innerMediator.PublishAsync(eventData);
    }

    private string ExtractCapability(object command)
    {
        // Try to get from attribute or interface
        var type = command.GetType();
        var attr = type.GetCustomAttributes(typeof(RequiresCapabilityAttribute), true).FirstOrDefault();

        if (attr is RequiresCapabilityAttribute capAttr)
        {
            return capAttr.Capability;
        }

        return string.Empty;
    }

    private string GetTokenFromCommand(object command)
    {
        // Get token from command property if available
        var prop = command.GetType().GetProperty("TokenId");
        return prop?.GetValue(command)?.ToString() ?? string.Empty;
    }

    private string ExtractAction(object command)
    {
        return command.GetType().Name.Replace("Command", "").ToLowerInvariant();
    }

    private string GetPrincipalFromCommand(object command)
    {
        var prop = command.GetType().GetProperty("Principal");
        return prop?.GetValue(command)?.ToString() ?? "system";
    }

    private string? ExtractResource(object command)
    {
        var prop = command.GetType().GetProperty("Resource");
        return prop?.GetValue(command)?.ToString();
    }
}

    /// <summary>
    /// Attribute to mark commands with required capabilities
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class RequiresCapabilityAttribute : Attribute
    {
        public string Capability { get; }

        public RequiresCapabilityAttribute(string capability)
        {
            Capability = capability;
        }
    }
}
