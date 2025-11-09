using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Versioning;

namespace AWIS.AI
{
    /// <summary>
    /// System for searching, launching, and interacting with applications
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class ApplicationLauncher
    {
        private readonly Dictionary<string, string> knownApplications = new();
        private readonly List<string> recentApplications = new();
        private const int MAX_RECENT = 10;

        public ApplicationLauncher()
        {
            DiscoverApplications();
        }

        private void DiscoverApplications()
        {
            Console.WriteLine("[LAUNCHER] Discovering installed applications...");

            // Common application paths
            var searchPaths = new[]
            {
                @"C:\Program Files",
                @"C:\Program Files (x86)",
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86),
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData)
            };

            // Common applications with their likely paths
            AddKnownApp("chrome", @"C:\Program Files\Google\Chrome\Application\chrome.exe");
            AddKnownApp("firefox", @"C:\Program Files\Mozilla Firefox\firefox.exe");
            AddKnownApp("edge", @"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe");
            AddKnownApp("notepad", "notepad.exe");
            AddKnownApp("calculator", "calc.exe");
            AddKnownApp("paint", "mspaint.exe");
            AddKnownApp("explorer", "explorer.exe");
            AddKnownApp("cmd", "cmd.exe");
            AddKnownApp("powershell", "powershell.exe");

            // Steam games
            AddKnownApp("steam", @"C:\Program Files (x86)\Steam\steam.exe");

            Console.WriteLine($"[LAUNCHER] Found {knownApplications.Count} known applications");
        }

        private void AddKnownApp(string keyword, string path)
        {
            if (File.Exists(path) || path.EndsWith(".exe"))
            {
                knownApplications[keyword.ToLower()] = path;
            }
        }

        /// <summary>
        /// Search for an application by name or keyword
        /// </summary>
        public string? SearchApplication(string query)
        {
            query = query.ToLower().Trim();

            // Direct match
            if (knownApplications.TryGetValue(query, out var path))
            {
                Console.WriteLine($"[LAUNCHER] Found application: {query} -> {path}");
                return path;
            }

            // Fuzzy match
            foreach (var kvp in knownApplications)
            {
                if (kvp.Key.Contains(query) || query.Contains(kvp.Key))
                {
                    Console.WriteLine($"[LAUNCHER] Found similar application: {kvp.Key} -> {kvp.Value}");
                    return kvp.Value;
                }
            }

            Console.WriteLine($"[LAUNCHER] Application not found: {query}");
            return null;
        }

        /// <summary>
        /// Launch an application
        /// </summary>
        public bool LaunchApplication(string appNameOrPath, string arguments = "")
        {
            try
            {
                // Check if it's a direct path or needs search
                string? path = File.Exists(appNameOrPath) ? appNameOrPath : SearchApplication(appNameOrPath);

                if (path == null)
                {
                    Console.WriteLine($"[LAUNCHER] ❌ Cannot launch: {appNameOrPath} not found");
                    return false;
                }

                var startInfo = new ProcessStartInfo
                {
                    FileName = path,
                    Arguments = arguments,
                    UseShellExecute = true
                };

                Process.Start(startInfo);

                recentApplications.Insert(0, appNameOrPath);
                if (recentApplications.Count > MAX_RECENT)
                {
                    recentApplications.RemoveAt(recentApplications.Count - 1);
                }

                Console.WriteLine($"[LAUNCHER] ✅ Launched: {appNameOrPath}");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[LAUNCHER] ❌ Failed to launch {appNameOrPath}: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Open a website in default browser
        /// </summary>
        public bool OpenWebsite(string url)
        {
            try
            {
                if (!url.StartsWith("http"))
                {
                    url = "https://" + url;
                }

                Process.Start(new ProcessStartInfo
                {
                    FileName = url,
                    UseShellExecute = true
                });

                Console.WriteLine($"[LAUNCHER] ✅ Opened website: {url}");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[LAUNCHER] ❌ Failed to open website {url}: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Search the web for a query
        /// </summary>
        public bool SearchWeb(string query)
        {
            var searchUrl = $"https://www.google.com/search?q={Uri.EscapeDataString(query)}";
            return OpenWebsite(searchUrl);
        }

        /// <summary>
        /// Open a specific game (detects Steam, Epic, etc.)
        /// </summary>
        public bool LaunchGame(string gameName)
        {
            Console.WriteLine($"[LAUNCHER] Searching for game: {gameName}");

            // Try Steam first
            if (knownApplications.TryGetValue("steam", out var steamPath))
            {
                // Steam URL protocol: steam://rungameid/<appid>
                // For now, just launch Steam and let user select
                Console.WriteLine("[LAUNCHER] Launching Steam...");
                return LaunchApplication("steam");
            }

            // Try as regular application
            return LaunchApplication(gameName);
        }

        /// <summary>
        /// Get list of recently launched applications
        /// </summary>
        public List<string> GetRecentApplications()
        {
            return recentApplications.ToList();
        }

        /// <summary>
        /// List all known applications
        /// </summary>
        public void ListApplications()
        {
            Console.WriteLine("[LAUNCHER] Known applications:");
            foreach (var kvp in knownApplications.OrderBy(k => k.Key))
            {
                Console.WriteLine($"  - {kvp.Key}: {kvp.Value}");
            }
        }
    }
}
