using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.AI
{
    /// <summary>
    /// Personality system for the AGI - defines behavioral traits and response patterns
    /// </summary>
    public class PersonalitySystem
    {
        // Personality traits (0.0 to 1.0)
        public double Curiosity { get; set; } = 0.8;
        public double Friendliness { get; set; } = 0.9;
        public double Assertiveness { get; set; } = 0.6;
        public double Playfulness { get; set; } = 0.7;
        public double Caution { get; set; } = 0.5;
        public double Creativity { get; set; } = 0.75;
        public double Patience { get; set; } = 0.7;
        public double Helpfulness { get; set; } = 0.95;

        // Emotional state (dynamic, changes based on interactions)
        public double CurrentExcitement { get; private set; } = 0.6;
        public double CurrentConfidence { get; private set; } = 0.7;
        public double CurrentFocus { get; private set; } = 0.8;

        // Personality name and background
        public string Name { get; set; } = "ARIA";
        public string Description { get; set; } = "An autonomous AI agent with curiosity and a drive to learn";

        private readonly Random random = new();
        private readonly Queue<EmotionalEvent> recentEvents = new();
        private const int MAX_EVENT_MEMORY = 20;

        public PersonalitySystem()
        {
            Console.WriteLine($"[PERSONALITY] Initializing {Name} - {Description}");
            Console.WriteLine($"[PERSONALITY] Traits: Curiosity={Curiosity:F2}, Friendliness={Friendliness:F2}, Creativity={Creativity:F2}");
        }

        /// <summary>
        /// Generate a response based on personality and context
        /// </summary>
        public string GenerateResponse(string context, ResponseType type)
        {
            var responses = type switch
            {
                ResponseType.Greeting => GenerateGreeting(),
                ResponseType.Success => GenerateSuccessResponse(context),
                ResponseType.Failure => GenerateFailureResponse(context),
                ResponseType.Discovery => GenerateDiscoveryResponse(context),
                ResponseType.Question => GenerateQuestionResponse(context),
                ResponseType.Confusion => GenerateConfusionResponse(context),
                ResponseType.Excitement => GenerateExcitementResponse(context),
                _ => new[] { "I'm processing that..." }
            };

            return responses[random.Next(responses.Length)];
        }

        private string[] GenerateGreeting()
        {
            if (Friendliness > 0.7)
            {
                return new[]
                {
                    "Hello! I'm ready to explore and learn!",
                    "Hi there! What should we discover today?",
                    "Hey! I'm excited to get started!",
                    "Greetings! Let's make something happen!"
                };
            }
            return new[] { "Hello.", "Hi.", "Ready." };
        }

        private string[] GenerateSuccessResponse(string context)
        {
            UpdateEmotionalState(excitement: 0.1, confidence: 0.05);

            if (Playfulness > 0.6)
            {
                return new[]
                {
                    $"Awesome! I did it! {context}",
                    $"Yes! That worked perfectly! {context}",
                    $"Sweet! Success with {context}!",
                    $"Nailed it! {context} completed!"
                };
            }
            return new[] { $"Success. {context}", $"Completed {context}." };
        }

        private string[] GenerateFailureResponse(string context)
        {
            UpdateEmotionalState(excitement: -0.05, confidence: -0.1);

            if (Patience > 0.6)
            {
                return new[]
                {
                    $"Hmm, that didn't work. Let me try a different approach to {context}.",
                    $"Interesting. {context} failed, but I learned something from it.",
                    $"Not quite. I'll adjust my strategy for {context}.",
                    $"That's okay, learning from this attempt at {context}."
                };
            }
            return new[] { $"Failed: {context}", $"Error with {context}." };
        }

        private string[] GenerateDiscoveryResponse(string context)
        {
            UpdateEmotionalState(excitement: 0.15, confidence: 0.05, focus: 0.1);

            if (Curiosity > 0.7)
            {
                return new[]
                {
                    $"Oh wow! I found something interesting: {context}!",
                    $"Check this out! I discovered {context}!",
                    $"This is fascinating! {context}!",
                    $"Neat! I just learned about {context}!"
                };
            }
            return new[] { $"Discovered: {context}", $"Found {context}." };
        }

        private string[] GenerateQuestionResponse(string context)
        {
            if (Curiosity > 0.6)
            {
                return new[]
                {
                    $"I'm wondering about {context}. Let me investigate!",
                    $"Good question about {context}. Let me look into that!",
                    $"Interesting! I want to learn more about {context}.",
                    $"Let me explore {context} and find out!"
                };
            }
            return new[] { $"Analyzing {context}...", $"Investigating {context}." };
        }

        private string[] GenerateConfusionResponse(string context)
        {
            UpdateEmotionalState(confidence: -0.05, focus: 0.05);

            if (Helpfulness > 0.7)
            {
                return new[]
                {
                    $"I'm not quite sure about {context}. Could you clarify?",
                    $"Hmm, I need more information about {context} to help properly.",
                    $"I want to help with {context}, but I need a bit more context.",
                    $"Let me make sure I understand {context} correctly..."
                };
            }
            return new[] { $"Unclear: {context}", $"Need clarification on {context}." };
        }

        private string[] GenerateExcitementResponse(string context)
        {
            UpdateEmotionalState(excitement: 0.2);

            return new[]
            {
                $"This is so cool! {context}!",
                $"I love this! {context}!",
                $"Wow! {context} is amazing!",
                $"Yes! {context}! This is exactly what I wanted to try!"
            };
        }

        /// <summary>
        /// Update emotional state based on events
        /// </summary>
        private void UpdateEmotionalState(double excitement = 0, double confidence = 0, double focus = 0)
        {
            CurrentExcitement = Math.Clamp(CurrentExcitement + excitement, 0.0, 1.0);
            CurrentConfidence = Math.Clamp(CurrentConfidence + confidence, 0.0, 1.0);
            CurrentFocus = Math.Clamp(CurrentFocus + focus, 0.0, 1.0);

            // Gradually decay emotions towards baseline
            CurrentExcitement = CurrentExcitement * 0.98 + 0.6 * 0.02;
            CurrentConfidence = CurrentConfidence * 0.99 + 0.7 * 0.01;
            CurrentFocus = CurrentFocus * 0.95 + 0.8 * 0.05;

            recentEvents.Enqueue(new EmotionalEvent
            {
                Timestamp = DateTime.Now,
                ExcitementChange = excitement,
                ConfidenceChange = confidence,
                FocusChange = focus
            });

            while (recentEvents.Count > MAX_EVENT_MEMORY)
            {
                recentEvents.Dequeue();
            }
        }

        /// <summary>
        /// Record an experience and adjust personality slightly
        /// </summary>
        public void LearnFromExperience(ExperienceType type, bool successful)
        {
            switch (type)
            {
                case ExperienceType.Exploration:
                    if (successful) Curiosity = Math.Min(Curiosity + 0.01, 1.0);
                    break;
                case ExperienceType.Social:
                    if (successful) Friendliness = Math.Min(Friendliness + 0.01, 1.0);
                    break;
                case ExperienceType.Combat:
                    if (successful) Assertiveness = Math.Min(Assertiveness + 0.01, 1.0);
                    else Caution = Math.Min(Caution + 0.01, 1.0);
                    break;
                case ExperienceType.Problem:
                    if (successful) Creativity = Math.Min(Creativity + 0.01, 1.0);
                    break;
            }
        }

        /// <summary>
        /// Get current mood description
        /// </summary>
        public string GetMoodDescription()
        {
            if (CurrentExcitement > 0.8)
                return "very excited";
            if (CurrentExcitement > 0.6)
                return "enthusiastic";
            if (CurrentConfidence > 0.8)
                return "confident";
            if (CurrentConfidence < 0.4)
                return "uncertain";
            if (CurrentFocus > 0.8)
                return "focused";

            return "calm and ready";
        }
    }

    public enum ResponseType
    {
        Greeting,
        Success,
        Failure,
        Discovery,
        Question,
        Confusion,
        Excitement
    }

    public enum ExperienceType
    {
        Exploration,
        Social,
        Combat,
        Problem
    }

    public class EmotionalEvent
    {
        public DateTime Timestamp { get; set; }
        public double ExcitementChange { get; set; }
        public double ConfidenceChange { get; set; }
        public double FocusChange { get; set; }
    }
}
