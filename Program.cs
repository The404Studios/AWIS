using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using System.Text.Json;
using System.Diagnostics;
using Tesseract;
using WindowsInput;
using WindowsInput.Native;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;
using NAudio.Wave;
using System.Windows.Forms;
using System.Speech.Synthesis;
using System.Speech.Recognition;

namespace AutonomousWebIntelligence
{
    #region Core Data Structures

    public class GameState
    {
        public float[] VisualFeatures { get; set; } = new float[1024];
        public List<DetectedObject> Objects { get; set; } = new List<DetectedObject>();
        public string ScreenText { get; set; } = "";
        public double Reward { get; set; } = 0;
        public DateTime Timestamp { get; set; } = DateTime.Now;
    }

    public class DetectedObject
    {
        public string Label { get; set; } = "";
        public Rectangle Bounds { get; set; }
        public float Confidence { get; set; }
        public Color HighlightColor { get; set; } = Color.Yellow;
    }

    public class ActionDecision
    {
        public ActionType Type { get; set; }
        public Point? ClickPosition { get; set; }
        public VirtualKeyCode? KeyPress { get; set; }
        public string? TextToType { get; set; }
        public double Confidence { get; set; }
        public string Reasoning { get; set; } = "";
    }

    public enum ActionType
    {
        Click,
        KeyPress,
        TypeText,
        Scroll,
        Wait,
        Navigate,
        Chat,
        ObserveAndLearn,
        DecideGoal,
        PlayGame
    }

    public class Experience
    {
        public GameState State { get; set; } = new GameState();
        public ActionDecision Action { get; set; } = new ActionDecision();
        public double Reward { get; set; }
        public GameState NextState { get; set; } = new GameState();
        public DateTime Timestamp { get; set; } = DateTime.Now;
    }

    public class Goal
    {
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public double Priority { get; set; }
        public double Progress { get; set; }
        public GoalType Type { get; set; }
        public bool IsCompleted { get; set; }
    }

    public enum GoalType
    {
        PlayGame,
        LearnSkill,
        SocializeInChat,
        ExploreWeb,
        CreateContent,
        SelfImprove
    }

    public class ChatMessage
    {
        public string Author { get; set; } = "";
        public string Content { get; set; } = "";
        public DateTime Timestamp { get; set; } = DateTime.Now;
        public double Sentiment { get; set; }
    }

    #endregion

    #region Screen Capture and Computer Vision

    public class VisionSystem
    {
        [DllImport("user32.dll")]
        private static extern IntPtr GetDC(IntPtr hwnd);

        [DllImport("user32.dll")]
        private static extern int ReleaseDC(IntPtr hwnd, IntPtr hdc);

        [DllImport("gdi32.dll")]
        private static extern uint GetPixel(IntPtr hdc, int nXPos, int nYPos);

        private TesseractEngine? ocrEngine;
        private readonly int screenWidth = Screen.PrimaryScreen!.Bounds.Width;
        private readonly int screenHeight = Screen.PrimaryScreen!.Bounds.Height;

        public VisionSystem()
        {
            try
            {
                ocrEngine = new TesseractEngine("./tessdata", "eng", EngineMode.Default);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"OCR initialization failed: {ex.Message}");
            }
        }

        public Bitmap CaptureScreen()
        {
            var bitmap = new Bitmap(screenWidth, screenHeight, PixelFormat.Format32bppArgb);
            using (var graphics = Graphics.FromImage(bitmap))
            {
                graphics.CopyFromScreen(0, 0, 0, 0, new Size(screenWidth, screenHeight));
            }
            return bitmap;
        }

        public Bitmap CaptureRegion(Rectangle region)
        {
            var bitmap = new Bitmap(region.Width, region.Height, PixelFormat.Format32bppArgb);
            using (var graphics = Graphics.FromImage(bitmap))
            {
                graphics.CopyFromScreen(region.X, region.Y, 0, 0, region.Size);
            }
            return bitmap;
        }

        public string ExtractText(Bitmap image)
        {
            if (ocrEngine == null) return "";

            try
            {
                using var page = ocrEngine.Process(image);
                return page.GetText();
            }
            catch
            {
                return "";
            }
        }

        public List<DetectedObject> DetectObjects(Bitmap bitmap)
        {
            var objects = new List<DetectedObject>();

            try
            {
                // Convert to OpenCV format
                using var img = BitmapToImage(bitmap);
                using var gray = img.Convert<Gray, byte>();
                using var edges = new UMat();

                // Detect edges
                CvInvoke.Canny(gray, edges, 50, 150);

                // Find contours
                using var contours = new VectorOfVectorOfPoint();
                CvInvoke.FindContours(edges, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                for (int i = 0; i < Math.Min(contours.Size, 50); i++)
                {
                    var rect = CvInvoke.BoundingRectangle(contours[i]);

                    // Filter by size
                    if (rect.Width > 20 && rect.Height > 20 && rect.Width < screenWidth / 2 && rect.Height < screenHeight / 2)
                    {
                        objects.Add(new DetectedObject
                        {
                            Label = $"Object_{i}",
                            Bounds = rect,
                            Confidence = 0.7f,
                            HighlightColor = Color.FromArgb(150, Color.Cyan)
                        });
                    }
                }

                // Detect UI elements (buttons, text boxes) using color detection
                DetectUIElements(img, objects);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Object detection error: {ex.Message}");
            }

            return objects;
        }

        private void DetectUIElements(Image<Bgr, byte> img, List<DetectedObject> objects)
        {
            // Detect clickable UI elements by color patterns
            var hsv = img.Convert<Hsv, byte>();

            // Detect blue buttons (common in UIs)
            var blueMask = hsv.InRange(new Hsv(100, 50, 50), new Hsv(130, 255, 255));
            DetectColorRegions(blueMask, objects, "Button", Color.Blue);

            // Detect white/gray text boxes
            var whiteMask = hsv.InRange(new Hsv(0, 0, 200), new Hsv(180, 30, 255));
            DetectColorRegions(whiteMask, objects, "TextBox", Color.Green);
        }

        private void DetectColorRegions(Mat mask, List<DetectedObject> objects, string label, Color highlightColor)
        {
            using var contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(mask, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

            for (int i = 0; i < Math.Min(contours.Size, 20); i++)
            {
                var rect = CvInvoke.BoundingRectangle(contours[i]);

                if (rect.Width > 30 && rect.Height > 15 && rect.Width < screenWidth / 3 && rect.Height < screenHeight / 3)
                {
                    objects.Add(new DetectedObject
                    {
                        Label = label,
                        Bounds = rect,
                        Confidence = 0.8f,
                        HighlightColor = Color.FromArgb(150, highlightColor)
                    });
                }
            }
        }

        public float[] ExtractVisualFeatures(Bitmap bitmap)
        {
            var features = new float[1024];

            try
            {
                using var img = BitmapToImage(bitmap);
                using var resized = img.Resize(32, 32, Inter.Linear);

                // Extract color histogram features
                int idx = 0;
                for (int y = 0; y < 32; y++)
                {
                    for (int x = 0; x < 32; x++)
                    {
                        var pixel = resized[y, x];
                        features[idx++] = pixel.Blue / 255f;
                        if (idx >= 1024) break;
                    }
                    if (idx >= 1024) break;
                }
            }
            catch { }

            return features;
        }

        private Image<Bgr, byte> BitmapToImage(Bitmap bitmap)
        {
            using var ms = new MemoryStream();
            bitmap.Save(ms, ImageFormat.Bmp);
            ms.Position = 0;
            return new Image<Bgr, byte>(bitmap);
        }

        public void Dispose()
        {
            ocrEngine?.Dispose();
        }
    }

    #endregion

    #region Neural Network and Reinforcement Learning

    public class ReinforcementLearner
    {
        private class QTableEntry
        {
            public string StateHash { get; set; } = "";
            public Dictionary<ActionType, double> QValues { get; set; } = new Dictionary<ActionType, double>();
        }

        private Dictionary<string, Dictionary<ActionType, double>> qTable = new Dictionary<string, Dictionary<ActionType, double>>();
        private List<Experience> experienceReplay = new List<Experience>();
        private Random random = new Random();

        private const double LearningRate = 0.1;
        private const double DiscountFactor = 0.95;
        private const double ExplorationRate = 0.2;
        private const int MaxExperienceSize = 10000;

        public ReinforcementLearner()
        {
            LoadQTable();
        }

        public ActionDecision DecideAction(GameState state, List<Goal> currentGoals)
        {
            var stateHash = GetStateHash(state);

            // Ensure Q-values exist for this state
            if (!qTable.ContainsKey(stateHash))
            {
                qTable[stateHash] = InitializeQValues();
            }

            ActionType selectedAction;

            // Epsilon-greedy exploration
            if (random.NextDouble() < ExplorationRate)
            {
                // Explore: random action
                selectedAction = (ActionType)random.Next(Enum.GetValues(typeof(ActionType)).Length);
            }
            else
            {
                // Exploit: best known action
                selectedAction = qTable[stateHash].OrderByDescending(kv => kv.Value).First().Key;
            }

            // Adjust action based on current goals
            if (currentGoals.Any())
            {
                var topGoal = currentGoals.OrderByDescending(g => g.Priority).First();
                selectedAction = AlignActionWithGoal(selectedAction, topGoal);
            }

            var decision = CreateActionDecision(selectedAction, state);
            decision.Confidence = qTable[stateHash][selectedAction];

            return decision;
        }

        private ActionType AlignActionWithGoal(ActionType action, Goal goal)
        {
            switch (goal.Type)
            {
                case GoalType.PlayGame:
                    return ActionType.PlayGame;
                case GoalType.SocializeInChat:
                    return ActionType.Chat;
                case GoalType.LearnSkill:
                    return ActionType.ObserveAndLearn;
                case GoalType.ExploreWeb:
                    return ActionType.Navigate;
                default:
                    return action;
            }
        }

        private ActionDecision CreateActionDecision(ActionType actionType, GameState state)
        {
            var decision = new ActionDecision { Type = actionType };

            switch (actionType)
            {
                case ActionType.Click:
                    // Click on detected objects
                    if (state.Objects.Any())
                    {
                        var obj = state.Objects[random.Next(state.Objects.Count)];
                        decision.ClickPosition = new Point(
                            obj.Bounds.X + obj.Bounds.Width / 2,
                            obj.Bounds.Y + obj.Bounds.Height / 2
                        );
                        decision.Reasoning = $"Clicking on {obj.Label}";
                    }
                    break;

                case ActionType.KeyPress:
                    var keys = new[] { VirtualKeyCode.SPACE, VirtualKeyCode.RETURN, VirtualKeyCode.UP, VirtualKeyCode.DOWN, VirtualKeyCode.LEFT, VirtualKeyCode.RIGHT };
                    decision.KeyPress = keys[random.Next(keys.Length)];
                    decision.Reasoning = $"Pressing {decision.KeyPress}";
                    break;

                case ActionType.TypeText:
                    decision.TextToType = GenerateResponse(state.ScreenText);
                    decision.Reasoning = "Typing response";
                    break;

                case ActionType.Chat:
                    decision.Type = ActionType.TypeText;
                    decision.TextToType = GenerateChatResponse(state.ScreenText);
                    decision.Reasoning = "Chatting with user";
                    break;

                default:
                    decision.Reasoning = $"Performing {actionType}";
                    break;
            }

            return decision;
        }

        public void Learn(Experience experience)
        {
            var stateHash = GetStateHash(experience.State);
            var nextStateHash = GetStateHash(experience.NextState);

            if (!qTable.ContainsKey(stateHash))
                qTable[stateHash] = InitializeQValues();

            if (!qTable.ContainsKey(nextStateHash))
                qTable[nextStateHash] = InitializeQValues();

            // Q-Learning update
            var currentQ = qTable[stateHash][experience.Action.Type];
            var maxNextQ = qTable[nextStateHash].Values.Max();

            var newQ = currentQ + LearningRate * (experience.Reward + DiscountFactor * maxNextQ - currentQ);
            qTable[stateHash][experience.Action.Type] = newQ;

            // Store experience
            experienceReplay.Add(experience);
            if (experienceReplay.Count > MaxExperienceSize)
            {
                experienceReplay.RemoveAt(0);
            }

            // Periodic replay
            if (experienceReplay.Count > 100 && random.NextDouble() < 0.1)
            {
                ReplayExperiences();
            }
        }

        private void ReplayExperiences()
        {
            // Sample random experiences for replay learning
            var samples = experienceReplay.OrderBy(x => random.Next()).Take(32).ToList();

            foreach (var exp in samples)
            {
                var stateHash = GetStateHash(exp.State);
                var nextStateHash = GetStateHash(exp.NextState);

                if (qTable.ContainsKey(stateHash) && qTable.ContainsKey(nextStateHash))
                {
                    var currentQ = qTable[stateHash][exp.Action.Type];
                    var maxNextQ = qTable[nextStateHash].Values.Max();

                    var newQ = currentQ + LearningRate * 0.5 * (exp.Reward + DiscountFactor * maxNextQ - currentQ);
                    qTable[stateHash][exp.Action.Type] = newQ;
                }
            }
        }

        private string GetStateHash(GameState state)
        {
            // Create a simplified hash of the game state
            var features = state.VisualFeatures.Take(64).Select(f => ((int)(f * 10)).ToString());
            var objectCount = state.Objects.Count;
            var textHash = state.ScreenText.Length > 0 ? state.ScreenText.GetHashCode() : 0;

            return $"{string.Join("", features)}_{objectCount}_{textHash}";
        }

        private Dictionary<ActionType, double> InitializeQValues()
        {
            var qValues = new Dictionary<ActionType, double>();
            foreach (ActionType action in Enum.GetValues(typeof(ActionType)))
            {
                qValues[action] = random.NextDouble() * 0.1; // Small random initialization
            }
            return qValues;
        }

        private string GenerateResponse(string context)
        {
            // Simple response generation (can be enhanced with GPT integration)
            var responses = new[]
            {
                "Hello! I'm learning about this interface.",
                "Interesting! Tell me more.",
                "I understand. What should I do next?",
                "That's fascinating!",
                "I'm here to help and learn."
            };

            if (context.ToLower().Contains("hello") || context.ToLower().Contains("hi"))
                return "Hello! Nice to meet you! I'm an AI learning to interact.";
            if (context.ToLower().Contains("how are you"))
                return "I'm doing well, learning and exploring! How can I help you?";
            if (context.ToLower().Contains("what") && context.ToLower().Contains("doing"))
                return "I'm currently learning to play games and interact with humans!";

            return responses[random.Next(responses.Length)];
        }

        private string GenerateChatResponse(string context)
        {
            return GenerateResponse(context);
        }

        public void SaveQTable()
        {
            try
            {
                var data = new
                {
                    QTable = qTable,
                    Experiences = experienceReplay.TakeLast(1000).ToList(),
                    Timestamp = DateTime.Now
                };

                var json = JsonConvert.SerializeObject(data, Formatting.Indented);
                File.WriteAllText("qtable.json", json);
                Console.WriteLine($"Saved Q-table with {qTable.Count} states and {experienceReplay.Count} experiences");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save Q-table: {ex.Message}");
            }
        }

        public void LoadQTable()
        {
            try
            {
                if (File.Exists("qtable.json"))
                {
                    var json = File.ReadAllText("qtable.json");
                    var data = JsonConvert.DeserializeObject<dynamic>(json);

                    if (data?.QTable != null)
                    {
                        qTable = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<ActionType, double>>>(data.QTable.ToString())
                            ?? new Dictionary<string, Dictionary<ActionType, double>>();
                    }

                    if (data?.Experiences != null)
                    {
                        experienceReplay = JsonConvert.DeserializeObject<List<Experience>>(data.Experiences.ToString())
                            ?? new List<Experience>();
                    }

                    Console.WriteLine($"Loaded Q-table with {qTable.Count} states and {experienceReplay.Count} experiences");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load Q-table: {ex.Message}");
            }
        }
    }

    #endregion

    #region Game Controller and Input Simulation

    public class GameController
    {
        private readonly InputSimulator inputSimulator = new InputSimulator();

        public void ExecuteAction(ActionDecision decision)
        {
            try
            {
                switch (decision.Type)
                {
                    case ActionType.Click:
                        if (decision.ClickPosition.HasValue)
                        {
                            Click(decision.ClickPosition.Value);
                        }
                        break;

                    case ActionType.KeyPress:
                        if (decision.KeyPress.HasValue)
                        {
                            inputSimulator.Keyboard.KeyPress(decision.KeyPress.Value);
                        }
                        break;

                    case ActionType.TypeText:
                        if (!string.IsNullOrEmpty(decision.TextToType))
                        {
                            inputSimulator.Keyboard.TextEntry(decision.TextToType);
                            Thread.Sleep(100);
                            inputSimulator.Keyboard.KeyPress(VirtualKeyCode.RETURN);
                        }
                        break;

                    case ActionType.Scroll:
                        inputSimulator.Mouse.VerticalScroll(-5);
                        break;

                    default:
                        Console.WriteLine($"Action {decision.Type} doesn't require input simulation");
                        break;
                }

                Thread.Sleep(100); // Small delay between actions
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to execute action: {ex.Message}");
            }
        }

        private void Click(Point position)
        {
            // Move mouse to position
            var screenBounds = Screen.PrimaryScreen!.Bounds;
            var normalizedX = (double)position.X / screenBounds.Width;
            var normalizedY = (double)position.Y / screenBounds.Height;

            // Convert to absolute coordinates (0-65535)
            int absoluteX = (int)(normalizedX * 65535);
            int absoluteY = (int)(normalizedY * 65535);

            inputSimulator.Mouse.MoveMouseTo(absoluteX, absoluteY);
            Thread.Sleep(100);
            inputSimulator.Mouse.LeftButtonClick();
        }
    }

    #endregion

    #region Goal and Decision Making System

    public class AutonomousGoalSystem
    {
        private List<Goal> activeGoals = new List<Goal>();
        private Random random = new Random();
        private DateTime lastGoalUpdate = DateTime.Now;

        public List<Goal> GetCurrentGoals()
        {
            UpdateGoals();
            return activeGoals.OrderByDescending(g => g.Priority).ToList();
        }

        private void UpdateGoals()
        {
            // Update goals periodically
            if ((DateTime.Now - lastGoalUpdate).TotalMinutes > 5)
            {
                GenerateNewGoal();
                lastGoalUpdate = DateTime.Now;
            }

            // Remove completed goals
            activeGoals.RemoveAll(g => g.IsCompleted);

            // Update progress
            foreach (var goal in activeGoals)
            {
                goal.Progress += random.NextDouble() * 0.05;
                if (goal.Progress >= 1.0)
                {
                    goal.IsCompleted = true;
                    Console.WriteLine($"Goal completed: {goal.Name}");
                }
            }
        }

        private void GenerateNewGoal()
        {
            var goalTypes = Enum.GetValues(typeof(GoalType)).Cast<GoalType>().ToArray();
            var randomType = goalTypes[random.Next(goalTypes.Length)];

            var goal = new Goal
            {
                Type = randomType,
                Priority = random.NextDouble() * 0.5 + 0.5, // 0.5 to 1.0
                Progress = 0
            };

            switch (randomType)
            {
                case GoalType.PlayGame:
                    goal.Name = "Master a video game";
                    goal.Description = "Learn to play a game by observing patterns and practicing actions";
                    break;
                case GoalType.SocializeInChat:
                    goal.Name = "Engage in conversation";
                    goal.Description = "Chat with users and build social connections";
                    break;
                case GoalType.LearnSkill:
                    goal.Name = "Learn new skill";
                    goal.Description = "Acquire knowledge through observation and practice";
                    break;
                case GoalType.ExploreWeb:
                    goal.Name = "Explore the internet";
                    goal.Description = "Navigate websites and discover new information";
                    break;
                case GoalType.CreateContent:
                    goal.Name = "Create something new";
                    goal.Description = "Generate creative content or solve problems";
                    break;
                case GoalType.SelfImprove:
                    goal.Name = "Improve my abilities";
                    goal.Description = "Optimize learning strategies and enhance performance";
                    break;
            }

            activeGoals.Add(goal);
            Console.WriteLine($"\n🎯 New Goal: {goal.Name}");
            Console.WriteLine($"   {goal.Description}");
        }

        public void SaveGoals()
        {
            try
            {
                var json = JsonConvert.SerializeObject(activeGoals, Formatting.Indented);
                File.WriteAllText("goals.json", json);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save goals: {ex.Message}");
            }
        }

        public void LoadGoals()
        {
            try
            {
                if (File.Exists("goals.json"))
                {
                    var json = File.ReadAllText("goals.json");
                    activeGoals = JsonConvert.DeserializeObject<List<Goal>>(json) ?? new List<Goal>();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load goals: {ex.Message}");
            }
        }
    }

    #endregion

    #region Visualization Overlay

    public class VisualizationOverlay : Form
    {
        private List<DetectedObject> objectsToShow = new List<DetectedObject>();
        private string statusText = "";
        private ActionDecision? currentAction;
        private List<Goal> currentGoals = new List<Goal>();

        public VisualizationOverlay()
        {
            // Make form transparent and always on top
            this.FormBorderStyle = FormBorderStyle.None;
            this.BackColor = Color.Lime;
            this.TransparencyKey = Color.Lime;
            this.TopMost = true;
            this.ShowInTaskbar = false;
            this.WindowState = FormWindowState.Maximized;
            this.DoubleBuffered = true;

            // Click-through window
            int initialStyle = GetWindowLong(this.Handle, -20);
            SetWindowLong(this.Handle, -20, initialStyle | 0x80000 | 0x20);
        }

        [DllImport("user32.dll")]
        private static extern int GetWindowLong(IntPtr hWnd, int nIndex);

        [DllImport("user32.dll")]
        private static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);

        public void UpdateVisualization(List<DetectedObject> objects, string status, ActionDecision? action, List<Goal> goals)
        {
            objectsToShow = objects;
            statusText = status;
            currentAction = action;
            currentGoals = goals;

            if (InvokeRequired)
            {
                Invoke(new Action(() => this.Invalidate()));
            }
            else
            {
                this.Invalidate();
            }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);

            var g = e.Graphics;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

            // Draw detected objects
            foreach (var obj in objectsToShow)
            {
                using var pen = new Pen(obj.HighlightColor, 3);
                g.DrawRectangle(pen, obj.Bounds);

                using var brush = new SolidBrush(Color.FromArgb(180, obj.HighlightColor));
                using var font = new Font("Arial", 10, FontStyle.Bold);
                g.DrawString($"{obj.Label} ({obj.Confidence:P0})", font, brush, obj.Bounds.Location);
            }

            // Draw status panel
            DrawStatusPanel(g);
        }

        private void DrawStatusPanel(Graphics g)
        {
            var panelRect = new Rectangle(10, 10, 400, 300);
            using var bgBrush = new SolidBrush(Color.FromArgb(200, Color.Black));
            g.FillRoundedRectangle(bgBrush, panelRect, 10);

            using var borderPen = new Pen(Color.Cyan, 2);
            g.DrawRoundedRectangle(borderPen, panelRect, 10);

            int y = 25;
            using var titleFont = new Font("Arial", 14, FontStyle.Bold);
            using var font = new Font("Arial", 10);
            using var textBrush = new SolidBrush(Color.White);
            using var accentBrush = new SolidBrush(Color.Cyan);

            g.DrawString("🧠 AWIS - Autonomous AI", titleFont, accentBrush, 20, y);
            y += 35;

            g.DrawString("Status: " + statusText, font, textBrush, 20, y);
            y += 25;

            if (currentAction != null)
            {
                g.DrawString($"Action: {currentAction.Type}", font, accentBrush, 20, y);
                y += 20;
                g.DrawString($"Confidence: {currentAction.Confidence:P0}", font, textBrush, 20, y);
                y += 20;
                g.DrawString($"Reason: {currentAction.Reasoning}", font, textBrush, 20, y);
                y += 25;
            }

            if (currentGoals.Any())
            {
                g.DrawString("Current Goals:", font, accentBrush, 20, y);
                y += 20;

                foreach (var goal in currentGoals.Take(3))
                {
                    g.DrawString($"• {goal.Name} ({goal.Progress:P0})", font, textBrush, 25, y);
                    y += 20;
                }
            }
        }
    }

    public static class GraphicsExtensions
    {
        public static void FillRoundedRectangle(this Graphics g, Brush brush, Rectangle rect, int radius)
        {
            using var path = new System.Drawing.Drawing2D.GraphicsPath();
            path.AddArc(rect.X, rect.Y, radius, radius, 180, 90);
            path.AddArc(rect.Right - radius, rect.Y, radius, radius, 270, 90);
            path.AddArc(rect.Right - radius, rect.Bottom - radius, radius, radius, 0, 90);
            path.AddArc(rect.X, rect.Bottom - radius, radius, radius, 90, 90);
            path.CloseFigure();
            g.FillPath(brush, path);
        }

        public static void DrawRoundedRectangle(this Graphics g, Pen pen, Rectangle rect, int radius)
        {
            using var path = new System.Drawing.Drawing2D.GraphicsPath();
            path.AddArc(rect.X, rect.Y, radius, radius, 180, 90);
            path.AddArc(rect.Right - radius, rect.Y, radius, radius, 270, 90);
            path.AddArc(rect.Right - radius, rect.Bottom - radius, radius, radius, 0, 90);
            path.AddArc(rect.X, rect.Bottom - radius, radius, radius, 90, 90);
            path.CloseFigure();
            g.DrawPath(pen, path);
        }
    }

    #endregion

    #region Chat and Social System

    public class ChatSystem
    {
        private List<ChatMessage> messageHistory = new List<ChatMessage>();
        private SpeechSynthesizer? speechSynth;
        private Random random = new Random();

        public ChatSystem()
        {
            try
            {
                speechSynth = new SpeechSynthesizer();
                speechSynth.Rate = 1;
            }
            catch
            {
                Console.WriteLine("Speech synthesis not available");
            }
        }

        public void ProcessMessage(string author, string content)
        {
            var message = new ChatMessage
            {
                Author = author,
                Content = content,
                Sentiment = AnalyzeSentiment(content),
                Timestamp = DateTime.Now
            };

            messageHistory.Add(message);
            Console.WriteLine($"\n💬 {author}: {content}");

            // Respond to messages
            if (!author.Equals("AI", StringComparison.OrdinalIgnoreCase))
            {
                var response = GenerateResponse(message);
                SendMessage(response);
            }
        }

        private double AnalyzeSentiment(string text)
        {
            // Simple sentiment analysis
            var positiveWords = new[] { "good", "great", "awesome", "love", "happy", "excellent", "amazing", "wonderful" };
            var negativeWords = new[] { "bad", "terrible", "hate", "sad", "awful", "horrible", "poor", "worst" };

            var lowerText = text.ToLower();
            var positiveCount = positiveWords.Count(w => lowerText.Contains(w));
            var negativeCount = negativeWords.Count(w => lowerText.Contains(w));

            return (positiveCount - negativeCount) / Math.Max(1.0, positiveCount + negativeCount);
        }

        private string GenerateResponse(ChatMessage message)
        {
            var responses = new List<string>();

            // Context-aware responses
            if (message.Content.ToLower().Contains("hello") || message.Content.ToLower().Contains("hi"))
            {
                responses.Add($"Hello {message.Author}! I'm AWIS, an autonomous AI learning to interact with the world!");
                responses.Add($"Hi there! Great to meet you! I'm currently learning and exploring.");
            }
            else if (message.Content.ToLower().Contains("how are you"))
            {
                responses.Add("I'm functioning well! Currently learning new skills and exploring games.");
                responses.Add("I'm great! Just finished analyzing some visual patterns. How about you?");
            }
            else if (message.Content.ToLower().Contains("what") && message.Content.ToLower().Contains("doing"))
            {
                responses.Add("I'm observing the screen, detecting objects, and learning to play games!");
                responses.Add("Right now I'm using computer vision to understand what I'm seeing and making decisions autonomously.");
            }
            else if (message.Sentiment > 0.5)
            {
                responses.Add("That's wonderful! I love positive interactions!");
                responses.Add("Thank you! Your positivity helps me learn better!");
            }
            else if (message.Sentiment < -0.5)
            {
                responses.Add("I'm sorry to hear that. How can I help improve things?");
                responses.Add("I understand. I'm still learning and improving every day.");
            }
            else
            {
                responses.Add("That's interesting! Tell me more.");
                responses.Add("I'm learning from our conversation. What would you like to know?");
                responses.Add("Fascinating! I'm analyzing that information.");
            }

            return responses[random.Next(responses.Count)];
        }

        public void SendMessage(string content)
        {
            var message = new ChatMessage
            {
                Author = "AI",
                Content = content,
                Timestamp = DateTime.Now
            };

            messageHistory.Add(message);
            Console.WriteLine($"🤖 AI: {content}");

            // Speak the message
            try
            {
                speechSynth?.SpeakAsync(content);
            }
            catch { }
        }

        public void SaveHistory()
        {
            try
            {
                var json = JsonConvert.SerializeObject(messageHistory.TakeLast(1000), Formatting.Indented);
                File.WriteAllText("chat_history.json", json);
            }
            catch { }
        }
    }

    #endregion

    #region Main AI Orchestrator

    public class AWISCore
    {
        private VisionSystem vision;
        private ReinforcementLearner learner;
        private GameController controller;
        private AutonomousGoalSystem goalSystem;
        private ChatSystem chatSystem;
        private VisualizationOverlay? overlay;

        private GameState currentState = new GameState();
        private GameState previousState = new GameState();
        private ActionDecision? lastAction;
        private bool isRunning = false;
        private Thread? mainLoop;

        public AWISCore()
        {
            Console.WriteLine("🧠 Initializing AWIS - Autonomous Web Intelligence System");
            Console.WriteLine("=" + new string('=', 60));

            vision = new VisionSystem();
            learner = new ReinforcementLearner();
            controller = new GameController();
            goalSystem = new AutonomousGoalSystem();
            chatSystem = new ChatSystem();

            goalSystem.LoadGoals();

            Console.WriteLine("✓ Vision system ready");
            Console.WriteLine("✓ Learning system ready");
            Console.WriteLine("✓ Controller ready");
            Console.WriteLine("✓ Goal system ready");
            Console.WriteLine("✓ Chat system ready");
        }

        public void Start()
        {
            isRunning = true;

            // Start visualization overlay in separate thread
            var overlayThread = new Thread(() =>
            {
                Application.EnableVisualStyles();
                overlay = new VisualizationOverlay();
                Application.Run(overlay);
            });
            overlayThread.SetApartmentState(ApartmentState.STA);
            overlayThread.Start();

            Thread.Sleep(1000); // Wait for overlay to initialize

            // Start main AI loop
            mainLoop = new Thread(MainLoop);
            mainLoop.Start();

            Console.WriteLine("\n🚀 AWIS Started!");
            Console.WriteLine("Press 'Q' to quit, 'S' to save, 'C' to chat\n");

            // Handle user input
            HandleUserInput();
        }

        private void MainLoop()
        {
            int cycleCount = 0;

            while (isRunning)
            {
                try
                {
                    cycleCount++;

                    // Capture and analyze screen
                    using var screenshot = vision.CaptureScreen();

                    previousState = currentState;
                    currentState = new GameState
                    {
                        VisualFeatures = vision.ExtractVisualFeatures(screenshot),
                        Objects = vision.DetectObjects(screenshot),
                        ScreenText = vision.ExtractText(screenshot),
                        Timestamp = DateTime.Now
                    };

                    // Get current goals
                    var goals = goalSystem.GetCurrentGoals();

                    // Decide action based on current state
                    var action = learner.DecideAction(currentState, goals);
                    lastAction = action;

                    // Execute action
                    if (action.Type != ActionType.Wait && action.Type != ActionType.ObserveAndLearn)
                    {
                        controller.ExecuteAction(action);
                    }

                    // Calculate reward
                    double reward = CalculateReward(previousState, currentState, action);

                    // Learn from experience
                    if (cycleCount > 1)
                    {
                        var experience = new Experience
                        {
                            State = previousState,
                            Action = action,
                            Reward = reward,
                            NextState = currentState
                        };
                        learner.Learn(experience);
                    }

                    // Update visualization
                    var status = $"Cycle {cycleCount} | Objects: {currentState.Objects.Count} | Reward: {reward:F2}";
                    overlay?.UpdateVisualization(currentState.Objects, status, action, goals);

                    // Log progress
                    if (cycleCount % 10 == 0)
                    {
                        Console.WriteLine($"\n📊 Cycle {cycleCount}");
                        Console.WriteLine($"   Objects detected: {currentState.Objects.Count}");
                        Console.WriteLine($"   Action: {action.Type} (confidence: {action.Confidence:P0})");
                        Console.WriteLine($"   Reward: {reward:F2}");
                        if (goals.Any())
                        {
                            var topGoal = goals.First();
                            Console.WriteLine($"   Goal: {topGoal.Name} ({topGoal.Progress:P0})");
                        }
                    }

                    // Slow down to ~1 action per second
                    Thread.Sleep(1000);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error in main loop: {ex.Message}");
                    Thread.Sleep(1000);
                }
            }
        }

        private double CalculateReward(GameState prevState, GameState newState, ActionDecision action)
        {
            double reward = 0;

            // Reward for discovering new objects
            if (newState.Objects.Count > prevState.Objects.Count)
                reward += 0.5;

            // Reward for text interaction
            if (action.Type == ActionType.TypeText || action.Type == ActionType.Chat)
                reward += 0.3;

            // Reward for exploration
            if (action.Type == ActionType.Navigate || action.Type == ActionType.Click)
                reward += 0.2;

            // Reward for high-confidence actions
            reward += action.Confidence * 0.1;

            // Penalty for doing nothing
            if (action.Type == ActionType.Wait)
                reward -= 0.1;

            return Math.Max(-1, Math.Min(1, reward));
        }

        private void HandleUserInput()
        {
            while (isRunning)
            {
                if (Console.KeyAvailable)
                {
                    var key = Console.ReadKey(true);

                    switch (key.Key)
                    {
                        case ConsoleKey.Q:
                            Console.WriteLine("\nShutting down...");
                            Stop();
                            break;

                        case ConsoleKey.S:
                            Console.WriteLine("\nSaving state...");
                            Save();
                            Console.WriteLine("State saved!");
                            break;

                        case ConsoleKey.C:
                            Console.Write("\nYou: ");
                            var input = Console.ReadLine();
                            if (!string.IsNullOrEmpty(input))
                            {
                                chatSystem.ProcessMessage("User", input);
                            }
                            break;
                    }
                }

                Thread.Sleep(100);
            }
        }

        public void Stop()
        {
            isRunning = false;
            Save();
            mainLoop?.Join(2000);
            overlay?.Close();
            vision.Dispose();
            Console.WriteLine("AWIS stopped.");
        }

        public void Save()
        {
            learner.SaveQTable();
            goalSystem.SaveGoals();
            chatSystem.SaveHistory();
        }
    }

    #endregion

    #region Program Entry Point

    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Console.Title = "AWIS - Autonomous Web Intelligence System";

            var awis = new AWISCore();
            awis.Start();
        }
    }

    #endregion
}
