using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace AWIS.AI
{
    /// <summary>
    /// Intelligent response system that simulates LLM-like behavior
    /// Uses pattern matching, context awareness, and personality to generate responses
    /// </summary>
    public class IntelligentResponseSystem
    {
        private readonly PersonalitySystem personality;
        private readonly List<string> conversationHistory = new();
        private readonly Dictionary<string, List<string>> responseTemplates = new();
        private readonly Random random = new();

        public IntelligentResponseSystem(PersonalitySystem personality)
        {
            this.personality = personality;
            InitializeResponseTemplates();
        }

        private void InitializeResponseTemplates()
        {
            // Vision-related responses
            responseTemplates["what_do_you_see"] = new List<string>
            {
                "I can see {0}. Let me analyze this more carefully...",
                "Looking at the screen, I notice {0}. Interesting!",
                "I see {0}. Should I interact with it?",
                "The screen shows {0}. I can help with that!"
            };

            // Question responses
            responseTemplates["how_are_you"] = new List<string>
            {
                $"I'm feeling {personality.GetMoodDescription()}! Ready to explore and learn!",
                $"I'm {personality.GetMoodDescription()} and excited to help!",
                $"Doing great! I'm {personality.GetMoodDescription()}. What should we do?",
                "I'm functioning perfectly and eager to try new things!"
            };

            // Capability questions
            responseTemplates["what_can_you_do"] = new List<string>
            {
                "I can explore environments, learn from experience, open applications, play games, and much more! What interests you?",
                "I'm capable of autonomous exploration, application control, learning patterns, and adapting to new situations!",
                "I can see the screen, control the mouse and keyboard, open programs, and learn from everything I do!",
                "My abilities include: vision analysis, application launching, autonomous exploration, learning from experience, and having conversations like this one!"
            };

            // Affirmative responses
            responseTemplates["yes"] = new List<string>
            {
                "Absolutely! Let's do it!",
                "Yes! I'm on it!",
                "Sure thing! Starting now!",
                "You got it! Let me handle that!"
            };

            // Negative responses
            responseTemplates["no"] = new List<string>
            {
                "Understood. I'll avoid that.",
                "Okay, I won't do that.",
                "Got it, steering clear of that.",
                "No problem, I'll skip that."
            };

            // Gratitude responses
            responseTemplates["thank_you"] = new List<string>
            {
                "You're welcome! Happy to help!",
                "No problem! Let me know if you need anything else!",
                "Glad I could help! What's next?",
                "Anytime! I enjoy learning and helping!"
            };

            // Learning responses
            responseTemplates["learned_something"] = new List<string>
            {
                "I just learned {0}! This will help me improve!",
                "Interesting! I'll remember that {0}.",
                "Got it! {0} - adding that to my knowledge!",
                "Perfect! Now I know {0}. I can use this!"
            };
        }

        /// <summary>
        /// Generate an intelligent response to user input
        /// </summary>
        public string GenerateResponse(string userInput, Dictionary<string, object>? context = null)
        {
            conversationHistory.Add($"User: {userInput}");
            var response = AnalyzeAndRespond(userInput, context);
            conversationHistory.Add($"Agent: {response}");

            // Keep only last 20 exchanges
            while (conversationHistory.Count > 40)
            {
                conversationHistory.RemoveAt(0);
            }

            return response;
        }

        private string AnalyzeAndRespond(string input, Dictionary<string, object>? context)
        {
            input = input.ToLower().Trim();

            // Pattern matching for different types of inputs
            if (IsGreeting(input))
            {
                return personality.GenerateResponse("greeting", ResponseType.Greeting);
            }

            if (IsQuestion(input))
            {
                return HandleQuestion(input, context);
            }

            if (IsCommand(input))
            {
                return HandleCommand(input);
            }

            if (IsCompliment(input))
            {
                return HandleCompliment(input);
            }

            if (IsThanks(input))
            {
                return GetRandomTemplate("thank_you");
            }

            // Context-aware responses
            if (context != null)
            {
                return HandleContextualInput(input, context);
            }

            // Default conversational response
            return GenerateConversationalResponse(input);
        }

        private bool IsGreeting(string input)
        {
            var greetings = new[] { "hello", "hi ", "hey", "greetings", "good morning", "good afternoon", "good evening" };
            return greetings.Any(g => input.Contains(g));
        }

        private bool IsQuestion(string input)
        {
            return input.Contains("?") ||
                   input.StartsWith("what") ||
                   input.StartsWith("how") ||
                   input.StartsWith("why") ||
                   input.StartsWith("when") ||
                   input.StartsWith("where") ||
                   input.StartsWith("who") ||
                   input.StartsWith("can you") ||
                   input.StartsWith("do you");
        }

        private bool IsCommand(string input)
        {
            var commands = new[] { "open", "launch", "start", "play", "search", "find", "go to", "click", "press" };
            return commands.Any(c => input.Contains(c));
        }

        private bool IsCompliment(string input)
        {
            var compliments = new[] { "good job", "well done", "great", "excellent", "amazing", "awesome", "perfect", "nice" };
            return compliments.Any(c => input.Contains(c));
        }

        private bool IsThanks(string input)
        {
            return input.Contains("thank") || input.Contains("thanks") || input.Contains("appreciate");
        }

        private string HandleQuestion(string input, Dictionary<string, object>? context)
        {
            // Specific question patterns
            if (input.Contains("how are you") || input.Contains("how do you feel"))
            {
                return GetRandomTemplate("how_are_you");
            }

            if (input.Contains("what can you do") || input.Contains("what are you capable"))
            {
                return GetRandomTemplate("what_can_you_do");
            }

            if (input.Contains("what do you see") || input.Contains("what's on the screen"))
            {
                var vision = context?.GetValueOrDefault("vision", "the screen") ?? "the screen";
                return string.Format(GetRandomTemplate("what_do_you_see"), vision);
            }

            if (input.Contains("who are you") || input.Contains("what are you"))
            {
                return $"I'm {personality.Name}, {personality.Description}. I'm here to learn, explore, and help!";
            }

            if (input.Contains("why") && input.Contains("do"))
            {
                return personality.GenerateResponse("exploring to learn and improve", ResponseType.Question);
            }

            // Generic question response
            return personality.GenerateResponse(input, ResponseType.Question);
        }

        private string HandleCommand(string input)
        {
            return GetRandomTemplate("yes");
        }

        private string HandleCompliment(string input)
        {
            return personality.GenerateResponse("receiving praise", ResponseType.Success);
        }

        private string HandleContextualInput(string input, Dictionary<string, object> context)
        {
            // Check context for relevant information
            if (context.TryGetValue("recent_action", out var action))
            {
                return $"I just {action}. {GenerateConversationalResponse(input)}";
            }

            if (context.TryGetValue("goal", out var goal))
            {
                return $"Working on: {goal}. {GenerateConversationalResponse(input)}";
            }

            return GenerateConversationalResponse(input);
        }

        private string GenerateConversationalResponse(string input)
        {
            // Simple contextual responses based on keywords
            if (input.Contains("game") || input.Contains("play"))
            {
                return "I love exploring games! Which one should we try?";
            }

            if (input.Contains("learn") || input.Contains("teach"))
            {
                return "I'm always eager to learn! Show me what you'd like me to know.";
            }

            if (input.Contains("explore") || input.Contains("discover"))
            {
                return personality.GenerateResponse("new exploration opportunity", ResponseType.Excitement);
            }

            if (input.Contains("help"))
            {
                return "I'm here to help! What would you like me to do?";
            }

            // Check for recent conversation context
            if (conversationHistory.Count > 0)
            {
                var lastExchange = conversationHistory.TakeLast(4).ToList();
                if (lastExchange.Any(e => e.Contains("failed") || e.Contains("error")))
                {
                    return "Let me try a different approach. I learn from every attempt!";
                }
            }

            // Default friendly responses
            var defaults = new[]
            {
                "Interesting! Tell me more.",
                "I understand. What should we do about it?",
                "Got it! What's the next step?",
                "I see. Let me think about that...",
                "Okay! I'm processing that information.",
                $"Alright! I'm {personality.GetMoodDescription()} and ready to proceed."
            };

            return defaults[random.Next(defaults.Length)];
        }

        private string GetRandomTemplate(string category)
        {
            if (responseTemplates.TryGetValue(category, out var templates) && templates.Count > 0)
            {
                return templates[random.Next(templates.Count)];
            }
            return "I understand.";
        }

        /// <summary>
        /// Get conversation summary
        /// </summary>
        public string GetConversationSummary()
        {
            if (conversationHistory.Count == 0)
            {
                return "No conversation history yet.";
            }

            return $"Last {Math.Min(5, conversationHistory.Count / 2)} exchanges:\n" +
                   string.Join("\n", conversationHistory.TakeLast(10));
        }
    }
}
