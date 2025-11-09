using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.Versioning;
using System.Text.RegularExpressions;
using AWIS.Core;

namespace AWIS.NLP
{
    #region Contextual Voice Command System

    /// <summary>
    /// Contextual command parser that understands spatial references, colors, and chained actions
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class ContextualVoiceCommandSystem
    {
        private readonly ScreenContextAnalyzer screenAnalyzer;
        private readonly ActionMapper actionMapper;
        private readonly SpatialReferenceParser spatialParser;
        private readonly ColorDetector colorDetector;
        private readonly Dictionary<string, ScreenRegion> recentRegions;

        public ContextualVoiceCommandSystem()
        {
            screenAnalyzer = new ScreenContextAnalyzer();
            actionMapper = new ActionMapper();
            spatialParser = new SpatialReferenceParser();
            colorDetector = new ColorDetector();
            recentRegions = new Dictionary<string, ScreenRegion>();
        }

        /// <summary>
        /// Parse a contextual voice command
        /// Example: "on the left side of the screen click on the red apple"
        /// </summary>
        public List<ContextualAction> ParseCommand(string voiceCommand, Bitmap screenCapture = null)
        {
            var actions = new List<ContextualAction>();

            // Normalize command
            voiceCommand = voiceCommand.ToLower().Trim();

            // Extract spatial references
            var spatialContext = spatialParser.ExtractSpatialContext(voiceCommand);

            // Extract color references
            var colorContext = colorDetector.ExtractColorReferences(voiceCommand);

            // Extract actions
            var actionChain = actionMapper.ExtractActionChain(voiceCommand);

            // Combine context with actions
            foreach (var action in actionChain)
            {
                var contextualAction = new ContextualAction
                {
                    Action = action,
                    SpatialContext = spatialContext,
                    ColorContext = colorContext,
                    OriginalCommand = voiceCommand
                };

                // If screen capture provided, find target coordinates
                if (screenCapture != null)
                {
                    var targetRegion = screenAnalyzer.FindTarget(
                        screenCapture,
                        spatialContext,
                        colorContext,
                        action.TargetObject
                    );

                    if (targetRegion != null)
                    {
                        contextualAction.TargetX = targetRegion.CenterX;
                        contextualAction.TargetY = targetRegion.CenterY;
                        contextualAction.TargetRegion = targetRegion;
                    }
                }

                actions.Add(contextualAction);
            }

            return actions;
        }

        /// <summary>
        /// Execute a contextual command
        /// </summary>
        public void ExecuteCommand(string voiceCommand, Bitmap screenCapture = null)
        {
            var actions = ParseCommand(voiceCommand, screenCapture);

            foreach (var action in actions)
            {
                Console.WriteLine($"Executing: {action.Action.ActionType} - {action.Action.Description}");

                if (action.TargetX.HasValue && action.TargetY.HasValue)
                {
                    Console.WriteLine($"  Target: ({action.TargetX}, {action.TargetY})");
                }

                action.Execute();
            }
        }
    }

    /// <summary>
    /// Contextual action with spatial and color context
    /// </summary>
    public class ContextualAction
    {
        public required ActionDefinition Action { get; set; }
        public required SpatialContext SpatialContext { get; set; }
        public required ColorContext ColorContext { get; set; }
        public required string OriginalCommand { get; set; }
        public int? TargetX { get; set; }
        public int? TargetY { get; set; }
        public ScreenRegion? TargetRegion { get; set; }

        public void Execute()
        {
            switch (Action.ActionType)
            {
                case ActionType.Click:
                    ExecuteClick();
                    break;
                case ActionType.Type:
                    ExecuteType();
                    break;
                case ActionType.Press:
                    ExecuteKeyPress();
                    break;
                case ActionType.Move:
                    ExecuteMovement();
                    break;
                case ActionType.Scroll:
                    ExecuteScroll();
                    break;
            }
        }

        private void ExecuteClick()
        {
            if (TargetX.HasValue && TargetY.HasValue)
            {
                // Simulate mouse click at coordinates
                Console.WriteLine($"[SIMULATED] Click at ({TargetX}, {TargetY})");
                // In real implementation: Use InputSimulator or Win32 API
                // Cursor.Position = new Point(TargetX.Value, TargetY.Value);
                // mouse_event(MOUSEEVENTF_LEFTDOWN | MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
            }
        }

        private void ExecuteType()
        {
            Console.WriteLine($"[SIMULATED] Type: {Action.Parameter}");
            // In real implementation: Use InputSimulator
            // var simulator = new InputSimulator();
            // simulator.Keyboard.TextEntry(Action.Parameter);
        }

        private void ExecuteKeyPress()
        {
            Console.WriteLine($"[SIMULATED] Press key: {Action.Parameter}");
            // In real implementation: Use InputSimulator
            // var simulator = new InputSimulator();
            // simulator.Keyboard.KeyPress(ParseKey(Action.Parameter));
        }

        private void ExecuteMovement()
        {
            var key = MapMovementToKey(Action.Parameter ?? "forward");
            Console.WriteLine($"[SIMULATED] Movement: {Action.Parameter} → Key: {key}");
            // In real implementation: Press the mapped key
        }

        private void ExecuteScroll()
        {
            Console.WriteLine($"[SIMULATED] Scroll: {Action.Parameter}");
            // In real implementation: Use mouse_event for scrolling
        }

        private static string MapMovementToKey(string movement)
        {
            var mappings = new Dictionary<string, string>
            {
                ["forward"] = "W",
                ["backward"] = "S",
                ["back"] = "S",
                ["left"] = "A",
                ["right"] = "D",
                ["up"] = "Space",
                ["down"] = "Ctrl",
                ["jump"] = "Space",
                ["crouch"] = "C"
            };

            return mappings.TryGetValue(movement.ToLower(), out var key)
                ? key
                : movement;
        }
    }

    #endregion

    #region Spatial Reference Parser

    /// <summary>
    /// Parses spatial references like "left side", "top right corner", "center"
    /// </summary>
    public class SpatialReferenceParser
    {
        private readonly Dictionary<string, ScreenZone> zoneKeywords;

        public SpatialReferenceParser()
        {
            zoneKeywords = new Dictionary<string, ScreenZone>
            {
                ["left"] = ScreenZone.Left,
                ["right"] = ScreenZone.Right,
                ["top"] = ScreenZone.Top,
                ["bottom"] = ScreenZone.Bottom,
                ["center"] = ScreenZone.Center,
                ["middle"] = ScreenZone.Center,
                ["upper"] = ScreenZone.Top,
                ["lower"] = ScreenZone.Bottom,
                ["top left"] = ScreenZone.TopLeft,
                ["top right"] = ScreenZone.TopRight,
                ["bottom left"] = ScreenZone.BottomLeft,
                ["bottom right"] = ScreenZone.BottomRight,
                ["top-left"] = ScreenZone.TopLeft,
                ["top-right"] = ScreenZone.TopRight,
                ["bottom-left"] = ScreenZone.BottomLeft,
                ["bottom-right"] = ScreenZone.BottomRight
            };
        }

        public SpatialContext ExtractSpatialContext(string command)
        {
            var context = new SpatialContext();

            // Check for "side of the screen" pattern
            var sidePattern = @"(left|right|top|bottom)\s+side\s+of\s+the\s+screen";
            var sideMatch = Regex.Match(command, sidePattern);
            if (sideMatch.Success)
            {
                var side = sideMatch.Groups[1].Value;
                context.Zone = zoneKeywords.ContainsKey(side) ? zoneKeywords[side] : ScreenZone.Entire;
                context.HasSpatialReference = true;
                return context;
            }

            // Check for corner patterns
            var cornerPattern = @"(top|bottom)\s+(left|right)\s+corner";
            var cornerMatch = Regex.Match(command, cornerPattern);
            if (cornerMatch.Success)
            {
                var corner = $"{cornerMatch.Groups[1].Value} {cornerMatch.Groups[2].Value}";
                context.Zone = zoneKeywords.ContainsKey(corner) ? zoneKeywords[corner] : ScreenZone.Entire;
                context.HasSpatialReference = true;
                return context;
            }

            // Check for simple directional keywords
            foreach (var kvp in zoneKeywords.OrderByDescending(k => k.Key.Length))
            {
                if (command.Contains(kvp.Key))
                {
                    context.Zone = kvp.Value;
                    context.HasSpatialReference = true;
                    break;
                }
            }

            // Check for "that" reference (refers to previously mentioned region)
            if (command.Contains("that ") || command.Contains("this "))
            {
                context.UsesPreviousReference = true;
            }

            return context;
        }
    }

    public class SpatialContext
    {
        public ScreenZone Zone { get; set; } = ScreenZone.Entire;
        public bool HasSpatialReference { get; set; }
        public bool UsesPreviousReference { get; set; }
        public int? CustomX { get; set; }
        public int? CustomY { get; set; }
        public int? CustomWidth { get; set; }
        public int? CustomHeight { get; set; }
    }

    public enum ScreenZone
    {
        Entire,
        Left,
        Right,
        Top,
        Bottom,
        Center,
        TopLeft,
        TopRight,
        BottomLeft,
        BottomRight
    }

    #endregion

    #region Color Detector

    /// <summary>
    /// Detects and extracts color references from commands
    /// </summary>
    public class ColorDetector
    {
        private readonly Dictionary<string, Color> colorNames;

        public ColorDetector()
        {
            colorNames = new Dictionary<string, Color>
            {
                ["red"] = Color.Red,
                ["blue"] = Color.Blue,
                ["green"] = Color.Green,
                ["yellow"] = Color.Yellow,
                ["orange"] = Color.Orange,
                ["purple"] = Color.Purple,
                ["pink"] = Color.Pink,
                ["white"] = Color.White,
                ["black"] = Color.Black,
                ["gray"] = Color.Gray,
                ["grey"] = Color.Gray,
                ["brown"] = Color.Brown,
                ["cyan"] = Color.Cyan,
                ["magenta"] = Color.Magenta,
                ["lime"] = Color.Lime,
                ["navy"] = Color.Navy,
                ["teal"] = Color.Teal,
                ["violet"] = Color.Violet,
                ["gold"] = Color.Gold,
                ["silver"] = Color.Silver
            };
        }

        public ColorContext ExtractColorReferences(string command)
        {
            var context = new ColorContext();

            // Find all color mentions
            foreach (var kvp in colorNames)
            {
                if (command.Contains(kvp.Key))
                {
                    context.Colors.Add(kvp.Value);
                    context.ColorNames.Add(kvp.Key);
                }
            }

            // Extract target object with color
            var colorObjectPattern = @"(red|blue|green|yellow|orange|purple|pink|white|black|gray|grey|brown)\s+(\w+)";
            var match = Regex.Match(command, colorObjectPattern);
            if (match.Success)
            {
                context.TargetObject = match.Groups[2].Value;
                context.TargetColor = match.Groups[1].Value;
            }

            return context;
        }

        public static bool IsColorMatch(Color pixelColor, Color targetColor, int tolerance = 30)
        {
            int rDiff = Math.Abs(pixelColor.R - targetColor.R);
            int gDiff = Math.Abs(pixelColor.G - targetColor.G);
            int bDiff = Math.Abs(pixelColor.B - targetColor.B);

            return rDiff <= tolerance && gDiff <= tolerance && bDiff <= tolerance;
        }
    }

    public class ColorContext
    {
        public List<Color> Colors { get; set; } = new List<Color>();
        public List<string> ColorNames { get; set; } = new List<string>();
        public string? TargetObject { get; set; }
        public string? TargetColor { get; set; }
    }

    #endregion

    #region Action Mapper

    /// <summary>
    /// Maps natural language commands to executable actions
    /// </summary>
    public class ActionMapper
    {
        private readonly Dictionary<string, ActionType> actionKeywords;
        private readonly Dictionary<string, string> movementKeywords;

        public ActionMapper()
        {
            actionKeywords = new Dictionary<string, ActionType>
            {
                ["click"] = ActionType.Click,
                ["press"] = ActionType.Press,
                ["tap"] = ActionType.Click,
                ["select"] = ActionType.Click,
                ["type"] = ActionType.Type,
                ["write"] = ActionType.Type,
                ["enter"] = ActionType.Type,
                ["move"] = ActionType.Move,
                ["go"] = ActionType.Move,
                ["walk"] = ActionType.Move,
                ["run"] = ActionType.Move,
                ["scroll"] = ActionType.Scroll,
                ["drag"] = ActionType.Drag,
                ["open"] = ActionType.Click,
                ["close"] = ActionType.Click
            };

            movementKeywords = new Dictionary<string, string>
            {
                ["forward"] = "W",
                ["backward"] = "S",
                ["back"] = "S",
                ["left"] = "A",
                ["right"] = "D",
                ["up"] = "Space",
                ["down"] = "Ctrl",
                ["jump"] = "Space",
                ["crouch"] = "C",
                ["sprint"] = "Shift"
            };
        }

        public List<ActionDefinition> ExtractActionChain(string command)
        {
            var actions = new List<ActionDefinition>();

            // Split on conjunctions to find action chains
            var segments = SplitActionChain(command);

            foreach (var segment in segments)
            {
                var action = ParseAction(segment);
                if (action != null)
                {
                    actions.Add(action);
                }
            }

            return actions;
        }

        private List<string> SplitActionChain(string command)
        {
            var conjunctions = new[] { " and ", " then ", " after that " };
            var segments = new List<string> { command };

            foreach (var conjunction in conjunctions)
            {
                var newSegments = new List<string>();
                foreach (var segment in segments)
                {
                    newSegments.AddRange(segment.Split(new[] { conjunction }, StringSplitOptions.None));
                }
                segments = newSegments;
            }

            return segments.Select(s => s.Trim()).ToList();
        }

        private ActionDefinition? ParseAction(string segment)
        {
            // Check for movement commands
            foreach (var kvp in movementKeywords)
            {
                if (segment.Contains(kvp.Key, StringComparison.OrdinalIgnoreCase))
                {
                    return new ActionDefinition
                    {
                        ActionType = ActionType.Move,
                        Parameter = kvp.Key,
                        KeyMapping = kvp.Value,
                        Description = $"Move {kvp.Key}"
                    };
                }
            }

            // Check for action keywords
            foreach (var kvp in actionKeywords.OrderByDescending(k => k.Key.Length))
            {
                if (segment.Contains(kvp.Key, StringComparison.OrdinalIgnoreCase))
                {
                    var action = new ActionDefinition
                    {
                        ActionType = kvp.Value,
                        Description = segment
                    };

                    // Extract target object
                    var clickPattern = $@"{kvp.Key}\s+(?:on\s+)?(?:the\s+)?(.+)";
                    var match = Regex.Match(segment, clickPattern);
                    if (match.Success)
                    {
                        action.TargetObject = match.Groups[1].Value.Trim();
                    }

                    // Extract text to type
                    if (kvp.Value == ActionType.Type)
                    {
                        var typePattern = $@"{kvp.Key}\s+(.+)";
                        var typeMatch = Regex.Match(segment, typePattern);
                        if (typeMatch.Success)
                        {
                            action.Parameter = typeMatch.Groups[1].Value.Trim();
                        }
                    }

                    return action;
                }
            }

            return null;
        }
    }

    public class ActionDefinition
    {
        public ActionType ActionType { get; set; }
        public string? Parameter { get; set; }
        public string? TargetObject { get; set; }
        public string? KeyMapping { get; set; }
        public required string Description { get; set; }
    }

    #endregion

    #region Screen Context Analyzer

    /// <summary>
    /// Analyzes screen content to find targets based on spatial and color context
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class ScreenContextAnalyzer
    {
        private readonly ColorDetector colorDetector;

        public ScreenContextAnalyzer()
        {
            colorDetector = new ColorDetector();
        }

        public ScreenRegion? FindTarget(Bitmap screen, SpatialContext spatial, ColorContext color, string? targetObject)
        {
            // Get the search region based on spatial context
            var searchRegion = GetSearchRegion(screen.Width, screen.Height, spatial);

            // Find colored objects in the region
            var coloredRegions = FindColoredRegions(screen, searchRegion, color);

            // Filter by target object if specified
            if (!string.IsNullOrEmpty(targetObject))
            {
                // Use OCR or object recognition here to identify specific objects
                // For now, return the first colored region
                return coloredRegions.FirstOrDefault();
            }

            return coloredRegions.FirstOrDefault();
        }

        private static Rectangle GetSearchRegion(int screenWidth, int screenHeight, SpatialContext spatial)
        {
            switch (spatial.Zone)
            {
                case ScreenZone.Left:
                    return new Rectangle(0, 0, screenWidth / 2, screenHeight);

                case ScreenZone.Right:
                    return new Rectangle(screenWidth / 2, 0, screenWidth / 2, screenHeight);

                case ScreenZone.Top:
                    return new Rectangle(0, 0, screenWidth, screenHeight / 2);

                case ScreenZone.Bottom:
                    return new Rectangle(0, screenHeight / 2, screenWidth, screenHeight / 2);

                case ScreenZone.Center:
                    int centerX = screenWidth / 4;
                    int centerY = screenHeight / 4;
                    return new Rectangle(centerX, centerY, screenWidth / 2, screenHeight / 2);

                case ScreenZone.TopLeft:
                    return new Rectangle(0, 0, screenWidth / 2, screenHeight / 2);

                case ScreenZone.TopRight:
                    return new Rectangle(screenWidth / 2, 0, screenWidth / 2, screenHeight / 2);

                case ScreenZone.BottomLeft:
                    return new Rectangle(0, screenHeight / 2, screenWidth / 2, screenHeight / 2);

                case ScreenZone.BottomRight:
                    return new Rectangle(screenWidth / 2, screenHeight / 2, screenWidth / 2, screenHeight / 2);

                default:
                    return new Rectangle(0, 0, screenWidth, screenHeight);
            }
        }

        private List<ScreenRegion> FindColoredRegions(Bitmap screen, Rectangle searchArea, ColorContext color)
        {
            var regions = new List<ScreenRegion>();

            if (color.Colors.Count == 0)
            {
                // No color specified, return center of search area
                regions.Add(new ScreenRegion
                {
                    X = searchArea.X + searchArea.Width / 2,
                    Y = searchArea.Y + searchArea.Height / 2,
                    Width = 50,
                    Height = 50,
                    CenterX = searchArea.X + searchArea.Width / 2,
                    CenterY = searchArea.Y + searchArea.Height / 2
                });
                return regions;
            }

            // Scan for colored regions (simplified - in production, use blob detection)
            var targetColor = color.Colors[0];
            var coloredPixels = new List<Point>();

            for (int x = searchArea.Left; x < searchArea.Right; x += 5) // Skip pixels for performance
            {
                for (int y = searchArea.Top; y < searchArea.Bottom; y += 5)
                {
                    if (x >= 0 && x < screen.Width && y >= 0 && y < screen.Height)
                    {
                        var pixelColor = screen.GetPixel(x, y);
                        if (ColorDetector.IsColorMatch(pixelColor, targetColor, 50))
                        {
                            coloredPixels.Add(new Point(x, y));
                        }
                    }
                }
            }

            // Cluster nearby pixels into regions
            if (coloredPixels.Count > 0)
            {
                var avgX = (int)coloredPixels.Average(p => p.X);
                var avgY = (int)coloredPixels.Average(p => p.Y);

                regions.Add(new ScreenRegion
                {
                    X = avgX - 25,
                    Y = avgY - 25,
                    Width = 50,
                    Height = 50,
                    CenterX = avgX,
                    CenterY = avgY,
                    PixelCount = coloredPixels.Count
                });
            }

            return regions;
        }
    }

    public class ScreenRegion
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public int CenterX { get; set; }
        public int CenterY { get; set; }
        public int PixelCount { get; set; }
        public string? Label { get; set; }
    }

    #endregion
}
