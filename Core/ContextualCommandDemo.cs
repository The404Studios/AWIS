using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.Versioning;
using AWIS.NLP;

namespace AWIS.Core
{
    /// <summary>
    /// Demonstration of contextual voice command system
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class ContextualCommandDemo
    {
        private readonly ContextualVoiceCommandSystem commandSystem;

        public ContextualCommandDemo()
        {
            commandSystem = new ContextualVoiceCommandSystem();
        }

        public void RunDemo()
        {
            Console.Clear();
            PrintHeader();

            // Demo 1: Spatial References
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n=== Demo 1: Spatial References ===\n");
            Console.ResetColor();

            DemoSpatialReferences();

            // Demo 2: Color Detection
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n=== Demo 2: Color-Based Targeting ===\n");
            Console.ResetColor();

            DemoColorDetection();

            // Demo 3: Action Chaining
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n=== Demo 3: Action Chaining ===\n");
            Console.ResetColor();

            DemoActionChaining();

            // Demo 4: Game Controls
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n=== Demo 4: Game Movement Controls ===\n");
            Console.ResetColor();

            DemoGameControls();

            // Demo 5: Complex Commands
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n=== Demo 5: Complex Contextual Commands ===\n");
            Console.ResetColor();

            DemoComplexCommands();

            PrintSummary();
        }

        private void PrintHeader()
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║         Contextual Voice Command System Demo                  ║");
            Console.WriteLine("║     Spatial References • Color Detection • Action Chaining     ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════════╝");
            Console.ResetColor();
        }

        private void DemoSpatialReferences()
        {
            var commands = new List<string>
            {
                "on the left side of the screen click the button",
                "click in the top right corner",
                "select the item in the center",
                "click on the bottom of the screen"
            };

            foreach (var command in commands)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("Voice Command: ");
                Console.ResetColor();
                Console.WriteLine($"\"{command}\"");

                var actions = commandSystem.ParseCommand(command);

                foreach (var action in actions)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"  ✓ Parsed Action:");
                    Console.ResetColor();
                    Console.WriteLine($"    Type: {action.Action.ActionType}");
                    Console.WriteLine($"    Spatial Zone: {action.SpatialContext.Zone}");
                    if (!string.IsNullOrEmpty(action.Action.TargetObject))
                    {
                        Console.WriteLine($"    Target: {action.Action.TargetObject}");
                    }
                }
                Console.WriteLine();
            }
        }

        private void DemoColorDetection()
        {
            var commands = new List<string>
            {
                "click on the red apple",
                "select the blue button",
                "on the left side click the green icon",
                "find the yellow star and click it"
            };

            foreach (var command in commands)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("Voice Command: ");
                Console.ResetColor();
                Console.WriteLine($"\"{command}\"");

                var actions = commandSystem.ParseCommand(command);

                foreach (var action in actions)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"  ✓ Parsed Action:");
                    Console.ResetColor();
                    Console.WriteLine($"    Type: {action.Action.ActionType}");

                    if (action.ColorContext.Colors.Count > 0)
                    {
                        Console.Write($"    Target Color: ");
                        PrintColoredText(action.ColorContext.ColorNames[0],
                                       action.ColorContext.Colors[0]);
                        Console.WriteLine();
                    }

                    if (!string.IsNullOrEmpty(action.ColorContext.TargetObject))
                    {
                        Console.WriteLine($"    Target Object: {action.ColorContext.TargetObject}");
                    }

                    if (action.SpatialContext.HasSpatialReference)
                    {
                        Console.WriteLine($"    Spatial Zone: {action.SpatialContext.Zone}");
                    }
                }
                Console.WriteLine();
            }
        }

        private void DemoActionChaining()
        {
            var commands = new List<string>
            {
                "click on that reply and tell them what you think",
                "open the menu and select settings",
                "find the file and then open it",
                "scroll down and click the submit button"
            };

            foreach (var command in commands)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("Voice Command: ");
                Console.ResetColor();
                Console.WriteLine($"\"{command}\"");

                var actions = commandSystem.ParseCommand(command);

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"  ✓ Action Chain ({actions.Count} actions):");
                Console.ResetColor();

                for (int i = 0; i < actions.Count; i++)
                {
                    var action = actions[i];
                    Console.WriteLine($"    {i + 1}. {action.Action.ActionType}");

                    if (!string.IsNullOrEmpty(action.Action.TargetObject))
                    {
                        Console.WriteLine($"       Target: {action.Action.TargetObject}");
                    }

                    if (!string.IsNullOrEmpty(action.Action.Parameter))
                    {
                        Console.WriteLine($"       Parameter: {action.Action.Parameter}");
                    }
                }
                Console.WriteLine();
            }
        }

        private void DemoGameControls()
        {
            var commands = new List<string>
            {
                "move forward",
                "move backward",
                "hey, move backward!",
                "turn left",
                "go right",
                "jump",
                "crouch down",
                "move forward and jump"
            };

            Console.WriteLine("Game Movement Mappings:");
            Console.WriteLine("  forward  → W key");
            Console.WriteLine("  backward → S key");
            Console.WriteLine("  left     → A key");
            Console.WriteLine("  right    → D key");
            Console.WriteLine("  jump     → Space key");
            Console.WriteLine("  crouch   → C key\n");

            foreach (var command in commands)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("Voice Command: ");
                Console.ResetColor();
                Console.WriteLine($"\"{command}\"");

                var actions = commandSystem.ParseCommand(command);

                foreach (var action in actions)
                {
                    if (action.Action.ActionType == ActionType.Move)
                    {
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.WriteLine($"  ✓ Movement Detected:");
                        Console.ResetColor();
                        Console.WriteLine($"    Direction: {action.Action.Parameter}");
                        Console.ForegroundColor = ConsoleColor.Cyan;
                        Console.WriteLine($"    Key Press: [{action.Action.KeyMapping}]");
                        Console.ResetColor();
                    }
                }
                Console.WriteLine();
            }
        }

        private void DemoComplexCommands()
        {
            var commands = new List<string>
            {
                "on the left side of the screen click on the red apple and then type hello",
                "find the blue button in the top right corner and click it",
                "click on that reply and tell them I agree",
                "in the center click the green start button"
            };

            foreach (var command in commands)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.Write("Complex Command: ");
                Console.ResetColor();
                Console.WriteLine($"\"{command}\"");

                var actions = commandSystem.ParseCommand(command);

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"  ✓ Complete Parse:");
                Console.ResetColor();

                foreach (var action in actions)
                {
                    Console.WriteLine($"    • Action: {action.Action.ActionType}");

                    if (action.SpatialContext.HasSpatialReference)
                    {
                        Console.WriteLine($"      Zone: {action.SpatialContext.Zone}");
                    }

                    if (action.ColorContext.Colors.Count > 0)
                    {
                        Console.Write($"      Color: ");
                        PrintColoredText(action.ColorContext.ColorNames[0],
                                       action.ColorContext.Colors[0]);
                        Console.WriteLine();
                    }

                    if (!string.IsNullOrEmpty(action.ColorContext.TargetObject))
                    {
                        Console.WriteLine($"      Object: {action.ColorContext.TargetObject}");
                    }

                    if (!string.IsNullOrEmpty(action.Action.Parameter))
                    {
                        Console.WriteLine($"      Parameter: {action.Action.Parameter}");
                    }
                }
                Console.WriteLine();
            }
        }

        private void PrintColoredText(string text, Color color)
        {
            // Map Color to ConsoleColor
            var consoleColor = MapToConsoleColor(color);
            Console.ForegroundColor = consoleColor;
            Console.Write(text);
            Console.ResetColor();
        }

        private ConsoleColor MapToConsoleColor(Color color)
        {
            if (color.R > 200 && color.G < 100 && color.B < 100) return ConsoleColor.Red;
            if (color.R < 100 && color.G < 100 && color.B > 200) return ConsoleColor.Blue;
            if (color.R < 100 && color.G > 200 && color.B < 100) return ConsoleColor.Green;
            if (color.R > 200 && color.G > 200 && color.B < 100) return ConsoleColor.Yellow;
            if (color.R > 200 && color.G < 100 && color.B > 200) return ConsoleColor.Magenta;
            if (color.R < 100 && color.G > 200 && color.B > 200) return ConsoleColor.Cyan;
            if (color.R > 200 && color.G > 200 && color.B > 200) return ConsoleColor.White;
            return ConsoleColor.Gray;
        }

        private void PrintSummary()
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n╔════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║                     Demo Summary                               ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════════╝");
            Console.ResetColor();

            Console.WriteLine("\n✅ Supported Features:");
            Console.WriteLine("   • Spatial references (left, right, top, bottom, center, corners)");
            Console.WriteLine("   • Color-based targeting (20+ colors)");
            Console.WriteLine("   • Action chaining with conjunctions (and, then, after that)");
            Console.WriteLine("   • Game movement controls (WASD + Space/Ctrl)");
            Console.WriteLine("   • Complex contextual understanding");
            Console.WriteLine("   • Natural language processing");

            Console.WriteLine("\n📍 Example Commands:");
            Console.WriteLine("   \"On the left side of the screen click on the red apple\"");
            Console.WriteLine("   \"Move forward and jump\"");
            Console.WriteLine("   \"Click on that reply and tell them what you think\"");
            Console.WriteLine("   \"Find the blue button in the top right corner\"");

            Console.WriteLine("\n🎮 Game Controls:");
            Console.WriteLine("   forward/backward → W/S keys");
            Console.WriteLine("   left/right → A/D keys");
            Console.WriteLine("   jump → Space");
            Console.WriteLine("   crouch → C");

            Console.WriteLine("\n🎯 Integration:");
            Console.WriteLine("   • Works with speech recognition systems");
            Console.WriteLine("   • Screen capture for visual context");
            Console.WriteLine("   • Input simulation for actions");
            Console.WriteLine("   • Extensible command grammar");
            Console.WriteLine();
        }

        /// <summary>
        /// Test with simulated screen capture
        /// </summary>
        public void TestWithScreenCapture()
        {
            Console.WriteLine("\n=== Testing with Simulated Screen ===\n");

            // Create a fake screen with colored regions
            var fakeScreen = CreateTestScreen(1920, 1080);

            var testCommands = new List<string>
            {
                "on the left side click on the red button",
                "find the blue icon in the top right",
                "click the green button in the center"
            };

            foreach (var command in testCommands)
            {
                Console.WriteLine($"Command: \"{command}\"");

                var actions = commandSystem.ParseCommand(command, fakeScreen);

                foreach (var action in actions)
                {
                    if (action.TargetX.HasValue && action.TargetY.HasValue)
                    {
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.WriteLine($"  ✓ Target found at ({action.TargetX}, {action.TargetY})");
                        Console.ResetColor();
                    }
                }
                Console.WriteLine();
            }

            fakeScreen.Dispose();
        }

        private Bitmap CreateTestScreen(int width, int height)
        {
            var bitmap = new Bitmap(width, height);
            using (var g = Graphics.FromImage(bitmap))
            {
                // Fill background
                g.Clear(Color.White);

                // Add colored regions
                // Red button on left
                g.FillRectangle(new SolidBrush(Color.Red), new Rectangle(100, 500, 100, 50));

                // Blue icon top right
                g.FillEllipse(new SolidBrush(Color.Blue), new Rectangle(1500, 100, 80, 80));

                // Green button center
                g.FillRectangle(new SolidBrush(Color.Green), new Rectangle(900, 500, 120, 60));
            }

            return bitmap;
        }
    }
}
