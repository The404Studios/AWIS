using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.Versioning;
using System.Threading;
using System.Threading.Tasks;
using Tesseract;

namespace AWIS.Vision
{
    /// <summary>
    /// Perceived screen state from vision processing
    /// </summary>
    public class ScreenState
    {
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public List<DetectedObject> Objects { get; set; } = new();
        public List<TextRegion> TextRegions { get; set; } = new();
        public Bitmap? Screenshot { get; set; }
        public double ProcessingTimeMs { get; set; }
        public int FrameNumber { get; set; }
    }

    /// <summary>
    /// Text detected via OCR
    /// </summary>
    public class TextRegion
    {
        public string Text { get; set; } = string.Empty;
        public Rectangle BoundingBox { get; set; }
        public double Confidence { get; set; }
        public string Type { get; set; } = "Unknown"; // Button, Label, Input, etc.
    }

    /// <summary>
    /// Continuous 60fps vision processing loop using real Tesseract OCR
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class VisionLoop : IDisposable
    {
        private readonly AdvancedComputerVision vision;
        private TesseractEngine? tesseract;
        private bool isRunning;
        private Task? visionTask;
        private readonly CancellationTokenSource cancellationToken;

        // Latest screen state (thread-safe)
        private ScreenState currentState;
        private readonly object stateLock = new();

        // Performance tracking
        private int frameCount = 0;
        private DateTime lastFpsUpdate = DateTime.UtcNow;
        private double currentFps = 0.0;

        // Target 60 FPS = 16.67ms per frame
        private const int TARGET_FRAME_TIME_MS = 16;

        public VisionLoop()
        {
            vision = new AdvancedComputerVision();
            currentState = new ScreenState();
            cancellationToken = new CancellationTokenSource();

            InitializeTesseract();
        }

        private void InitializeTesseract()
        {
            try
            {
                // Try to initialize Tesseract with trained data
                var tessDataPath = "./tessdata";
                if (!System.IO.Directory.Exists(tessDataPath))
                {
                    tessDataPath = "../tessdata";
                }

                tesseract = new TesseractEngine(tessDataPath, "eng", EngineMode.Default);
                tesseract.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-_:;");

                Console.WriteLine("[VISION] ✅ Tesseract OCR initialized successfully");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[VISION] ⚠️ Failed to initialize Tesseract: {ex.Message}");
                Console.WriteLine("[VISION] OCR will be disabled. Please ensure tessdata folder exists.");
                tesseract = null;
            }
        }

        /// <summary>
        /// Start the 60fps vision loop
        /// </summary>
        public void Start()
        {
            if (isRunning) return;

            isRunning = true;
            visionTask = Task.Run(() => VisionProcessingLoop(cancellationToken.Token));

            Console.WriteLine("[VISION] 🎥 60fps vision loop started");
        }

        /// <summary>
        /// Stop the vision loop
        /// </summary>
        public void Stop()
        {
            isRunning = false;
            cancellationToken.Cancel();

            Console.WriteLine("[VISION] 🛑 Vision loop stopped");
        }

        /// <summary>
        /// Main 60fps vision processing loop
        /// </summary>
        private async Task VisionProcessingLoop(CancellationToken token)
        {
            var frameTimer = System.Diagnostics.Stopwatch.StartNew();

            while (isRunning && !token.IsCancellationRequested)
            {
                frameTimer.Restart();

                try
                {
                    // Capture and process frame
                    var newState = await ProcessFrameAsync();

                    // Update current state (thread-safe)
                    lock (stateLock)
                    {
                        currentState = newState;
                        frameCount++;
                    }

                    // Calculate FPS every second
                    var now = DateTime.UtcNow;
                    if ((now - lastFpsUpdate).TotalSeconds >= 1.0)
                    {
                        currentFps = frameCount / (now - lastFpsUpdate).TotalSeconds;
                        frameCount = 0;
                        lastFpsUpdate = now;
                    }

                    // Maintain 60 FPS timing
                    var elapsed = frameTimer.ElapsedMilliseconds;
                    var sleepTime = TARGET_FRAME_TIME_MS - (int)elapsed;

                    if (sleepTime > 0)
                    {
                        await Task.Delay(sleepTime, token);
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[VISION] Error in vision loop: {ex.Message}");
                    await Task.Delay(100); // Prevent spam on errors
                }
            }
        }

        /// <summary>
        /// Process a single frame: screenshot, OCR, object detection
        /// </summary>
        private async Task<ScreenState> ProcessFrameAsync()
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var state = new ScreenState
            {
                Timestamp = DateTime.UtcNow,
                FrameNumber = frameCount
            };

            // Capture screenshot
            state.Screenshot = vision.CaptureScreen();

            // Run OCR and object detection in parallel
            var ocrTask = Task.Run(() => PerformOCR(state.Screenshot));
            var objectTask = Task.Run(() => vision.DetectObjects(state.Screenshot, 0.3));

            await Task.WhenAll(ocrTask, objectTask);

            state.TextRegions = ocrTask.Result;
            state.Objects = objectTask.Result;
            state.ProcessingTimeMs = sw.Elapsed.TotalMilliseconds;

            return state;
        }

        /// <summary>
        /// Perform OCR on screenshot using Tesseract
        /// </summary>
        private List<TextRegion> PerformOCR(Bitmap screenshot)
        {
            var regions = new List<TextRegion>();

            if (tesseract == null)
                return regions;

            try
            {
                using var pix = PixConverter.ToPix(screenshot);
                using var page = tesseract.Process(pix);

                using var iter = page.GetIterator();
                iter.Begin();

                do
                {
                    var text = iter.GetText(PageIteratorLevel.Word);
                    if (string.IsNullOrWhiteSpace(text))
                        continue;

                    var confidence = iter.GetConfidence(PageIteratorLevel.Word);
                    if (confidence < 30) // Skip low confidence
                        continue;

                    if (iter.TryGetBoundingBox(PageIteratorLevel.Word, out var bounds))
                    {
                        var region = new TextRegion
                        {
                            Text = text.Trim(),
                            BoundingBox = new Rectangle(bounds.X1, bounds.Y1,
                                                       bounds.X2 - bounds.X1,
                                                       bounds.Y2 - bounds.Y1),
                            Confidence = confidence / 100.0,
                            Type = ClassifyTextType(text, bounds)
                        };

                        regions.Add(region);
                    }
                } while (iter.Next(PageIteratorLevel.Word));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[VISION] OCR error: {ex.Message}");
            }

            return regions;
        }

        /// <summary>
        /// Classify text type based on content and position
        /// </summary>
        private string ClassifyTextType(string text, Rect bounds)
        {
            // Simple heuristics for classification
            if (text.Length == 1 || text.All(char.IsDigit))
                return "Number";

            if (text.EndsWith(":"))
                return "Label";

            if (bounds.Width < 100 && bounds.Height < 40)
                return "Button";

            if (bounds.Width > 300)
                return "Paragraph";

            return "Text";
        }

        /// <summary>
        /// Get current screen state (thread-safe)
        /// </summary>
        public ScreenState GetCurrentState()
        {
            lock (stateLock)
            {
                return currentState;
            }
        }

        /// <summary>
        /// Get current FPS
        /// </summary>
        public double GetFPS()
        {
            return currentFps;
        }

        /// <summary>
        /// Find text on screen (case-insensitive partial match)
        /// </summary>
        public TextRegion? FindText(string searchText)
        {
            lock (stateLock)
            {
                return currentState.TextRegions.Find(r =>
                    r.Text.Contains(searchText, StringComparison.OrdinalIgnoreCase));
            }
        }

        /// <summary>
        /// Find all text matching pattern
        /// </summary>
        public List<TextRegion> FindAllText(string searchText)
        {
            lock (stateLock)
            {
                return currentState.TextRegions.FindAll(r =>
                    r.Text.Contains(searchText, StringComparison.OrdinalIgnoreCase));
            }
        }

        /// <summary>
        /// Get all buttons visible on screen
        /// </summary>
        public List<TextRegion> GetButtons()
        {
            lock (stateLock)
            {
                return currentState.TextRegions.FindAll(r => r.Type == "Button");
            }
        }

        /// <summary>
        /// Get statistics
        /// </summary>
        public string GetStatistics()
        {
            lock (stateLock)
            {
                return $"FPS: {currentFps:F1} | Frame: {currentState.FrameNumber} | " +
                       $"Text: {currentState.TextRegions.Count} | Objects: {currentState.Objects.Count} | " +
                       $"Processing: {currentState.ProcessingTimeMs:F1}ms";
            }
        }

        public void Dispose()
        {
            Stop();
            visionTask?.Wait(1000);
            tesseract?.Dispose();

            lock (stateLock)
            {
                currentState.Screenshot?.Dispose();
            }
        }
    }
}
