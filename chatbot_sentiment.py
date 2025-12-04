# import nltk
# nltk.download('vader_lexicon')



"""
Chatbot with Hybrid Sentiment + Groq LLaMA-3.1-8B – Tier 1 + Tier 2
(Full Debugged Version — Persistent Topics Fix — Stability Improved)
"""

import os
import re
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from nltk.sentiment import SentimentIntensityAnalyzer
from groq import Groq

# ---------- Persistent storage paths ----------

MEMORY_DIR = "memory"
USER_MEMORY_FILE = os.path.join(MEMORY_DIR, "user_memory.json")
CHAT_HISTORY_FILE = os.path.join(MEMORY_DIR, "chat_sessions.json")
CHAT_LOG_FILE = "chat_log.txt"


# ======================= HYBRID SENTIMENT ENGINE =======================

class HybridSentimentAnalyzer:
    """
    Hybrid sentiment analyzer combining:
      - VADER sentiment
      - Custom rule-based emotional cue detection
    """

    def __init__(self) -> None:
        self.vader = SentimentIntensityAnalyzer()

        self.failure_patterns = [
            "got failed", "failed in", "i failed", "did not pass",
            "didn't pass", "could not submit", "couldn't submit",
            "missed submission", "submission failed", "failed the exam",
            "failed in submission"
        ]

        self.extra_positive_words = {
            "confident", "confidence", "motivated", "strong",
            "hopeful", "optimistic", "excited", "thrilled",
            "grateful", "satisfied", "content"
        }

        self.extra_negative_words = {
            "pathetic", "trash", "useless", "disgusting", "horrendous",
            "devastated", "miserable", "hopeless", "annoyed", "irritated",
            "suicidal", "blue"
        }

        self.slang_negative = [
            "jaa yrr", "ja yrr", "get lost",
            "go away", "leave me alone", "shut up"
        ]

    def _rule_score(self, text: str) -> float:
        t = text.lower()
        score = 0.0

        if any(p in t for p in self.failure_patterns):
            score -= 0.6

        emo = re.search(r"\b(i feel|i'm feeling|im feeling)\s+([a-z]+)", t)
        if emo:
            feeling = emo.group(2)
            if feeling in self.extra_positive_words:
                score += 0.7
            elif feeling in self.extra_negative_words:
                score -= 0.7

        iam = re.search(r"\bi am\s+([a-z]+)", t)
        if iam:
            feeling = iam.group(1)
            if feeling in self.extra_positive_words:
                score += 0.6
            elif feeling in self.extra_negative_words:
                score -= 0.6

        if any(p in t for p in self.slang_negative):
            score -= 0.6

        if "feeling blue" in t or "feel blue" in t:
            score -= 0.6

        if re.search(r"(.)\1\1+", t):
            score *= 1.2

        ex = t.count("!")
        if ex > 1:
            score *= (1 + min(ex, 5)*0.05)

        return max(-1.0, min(1.0, score))

    def analyze(self, text: str) -> Tuple[str, float]:
        if not text.strip():
            return "Neutral", 0.0

        vs = self.vader.polarity_scores(text)["compound"]
        rule = self._rule_score(text)
        score = 0.8 * vs + 0.4 * rule

        score = max(-1.0, min(1.0, score))

        if score > 0.05:
            label = "Positive"
        elif score < -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return label, score


# ======================= GROQ LLaMA-3.1-8B RESPONDER =======================

os.environ["GROQ_API_KEY"] = "your api key"

class GroqLLMResponder:
    def __init__(self, model: str = "llama-3.1-8b-instant") -> None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None
        self.model = model

    def is_available(self):
        return self.client is not None

    def build_system_prompt(self, tone: str, sentiment: str):
        base = (
            "You are a helpful conversational assistant. "
            "Respond in 1–3 sentences, warm and human but not overly emotional.\n"
        )

        if tone == "casual":
            base += "Tone: friendly casual. Emojis allowed but not excessive.\n"
        elif tone == "formal":
            base += "Tone: polite and formal. No emojis.\n"
        else:
            base += "Tone: neutral-warm professional.\n"

        base += f"Detected user sentiment: {sentiment}. Adjust empathy accordingly.\n"
        return base

    def build_user_prompt(self, user_message: str, short_context: str):
        txt = ""
        if short_context:
            txt += f"Recent context: {short_context}\n"

        txt += f"User message: {user_message}\n"
        txt += "Reply directly. Do not mention sentiment or being an AI.\n"
        return txt

    def generate(self, user_message, sentiment, tone, history):
        if not self.is_available():
            raise RuntimeError("Groq not available")

        recent = []
        for m in history[-4:]:
            recent.append(f"{m['speaker']}: {m['text']}")
        ctx = " | ".join(recent)

        system = self.build_system_prompt(tone, sentiment)
        user = self.build_user_prompt(user_message, ctx)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return completion.choices[0].message.content.strip()


# ======================= CHATBOT CLASS (PART 1) =======================

class Chatbot:

    def __init__(self) -> None:
        self.analyzer = HybridSentimentAnalyzer()
        self.history = []

        # FIXED: topics MUST be a set permanently
        self.memory = {
            "name": None,
            "tone": "neutral",
            "topics": set(),   # ← always a set
            "negative_streak": 0,
            "positive_streak": 0,
            "neg_count": 0,
            "pos_count": 0,
            "last_sentiment": "Neutral",
        }

        self.last_bot_reply = None
        self.past_chats = []

        self.llm = GroqLLMResponder()

        self._load_persistent_memory()

    # ---------- FIXED: Persistent Memory Loader ----------
    def _load_persistent_memory(self):
        if not os.path.exists(MEMORY_DIR):
            os.makedirs(MEMORY_DIR)

        if os.path.exists(USER_MEMORY_FILE):
            try:
                with open(USER_MEMORY_FILE, "r") as f:
                    data = json.load(f)

                for key in self.memory:
                    if key in data:
                        # FIX: convert topics list → set
                        if key == "topics":
                            self.memory["topics"] = set(data["topics"])
                        else:
                            self.memory[key] = data[key]

            except Exception:
                pass

        if os.path.exists(CHAT_HISTORY_FILE):
            try:
                with open(CHAT_HISTORY_FILE, "r") as f:
                    self.past_chats = json.load(f)
            except:
                self.past_chats = []
        else:
            self.past_chats = []
    # ---------- Save persistent memory ----------
    def save_persistent_memory(self):
        data = self.memory.copy()
        # convert set → list for JSON
        if isinstance(data.get("topics"), set):
            data["topics"] = list(data["topics"])

        with open(USER_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        session = {
            "timestamp": str(datetime.now()),
            "history": self.history
        }
        self.past_chats.append(session)

        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.past_chats, f, indent=4)

    # ---------- History helpers ----------
    def add_user(self, text, label, score):
        self.history.append({
            "speaker": "user",
            "text": text,
            "sentiment_label": label,
            "sentiment_score": score,
        })

    def add_bot(self, text):
        self.history.append({"speaker": "bot", "text": text})

    # ---------- Name Extraction ----------
    def _extract_name(self, text):
        negative_words = {
            "upset", "sad", "angry", "tired", "depressed", "worried",
            "frustrated", "annoyed", "sick", "bad", "good", "fine",
            "confident", "happy"
        }

        # “my name is X”
        m = re.search(r"\bmy name is\s+([A-Za-z]+)\b", text, flags=re.I)
        if m:
            name = m.group(1)
            if name.lower() not in negative_words:
                return name

        # “call me X”
        m = re.search(r"\bcall me\s+([A-Za-z]+)\b", text, flags=re.I)
        if m:
            name = m.group(1)
            if name.lower() not in negative_words:
                return name

        # “I am X” (ensure X starts with capital)
        m = re.search(r"\bi am\s+([A-Za-z]+)\b", text)
        if m:
            name = m.group(1)
            if name and name[0].isupper() and name.lower() not in negative_words:
                return name

        # “I'm X”
        m = re.search(r"\bi'm\s+([A-Za-z]+)\b", text)
        if m:
            name = m.group(1)
            if name and name[0].isupper() and name.lower() not in negative_words:
                return name

        return None

    # ---------- Tone Detection ----------
    def _update_tone_from_text(self, text):
        t = text.lower()
        casual = {"bro", "dude", "lol", "lmao", "yrr", "u ", "omg", "yaar", "babe"}
        formal = {"please", "kindly", "could you", "would you"}

        if any(c in t for c in casual):
            self.memory["tone"] = "casual"
        elif any(f in t for f in formal):
            self.memory["tone"] = "formal"

    # ---------- Topic Extraction (fixed set usage) ----------
    def _update_topics(self, text):
        t = text.lower()
        topics = self.memory["topics"]   # ALWAYS a set

        if any(w in t for w in ["exam", "test", "study", "assignment", "submission"]):
            topics.add("study")

        if any(w in t for w in ["job", "work", "office", "career"]):
            topics.add("work")

        if any(w in t for w in [
            "sad", "upset", "happy", "depressed", "angry", "worried",
            "confident", "blue", "suicidal"
        ]):
            topics.add("feelings")

        if any(w in t for w in ["issue", "problem", "error", "bug", "failed"]):
            topics.add("problems")

    # ---------- Memory Update ----------
    def update_memory_from_text(self, text):
        name = self._extract_name(text)
        if name:
            self.memory["name"] = name

        self._update_tone_from_text(text)
        self._update_topics(text)

    # ---------- Sentiment Stats ----------
    def update_sentiment_stats(self, label):
        self.memory["last_sentiment"] = label

        if label == "Negative":
            self.memory["neg_count"] += 1
            self.memory["negative_streak"] += 1
            self.memory["positive_streak"] = 0
        elif label == "Positive":
            self.memory["pos_count"] += 1
            self.memory["positive_streak"] += 1
            self.memory["negative_streak"] = 0
        else:
            self.memory["negative_streak"] = max(0, self.memory["negative_streak"] - 1)
            self.memory["positive_streak"] = max(0, self.memory["positive_streak"] - 1)

    # ---------- Crisis Detection ----------
    def detect_crisis(self, text):
        t = text.lower()
        crisis = [
            "suicidal", "kill myself", "i want to die", "i wanna die",
            "don't want to live", "life is meaningless",
            "hurt myself", "self-harm", "cut myself",
        ]
        if any(p in t for p in crisis):
            return (
                "I'm really sorry you're feeling this way. "
                "I might not be able to provide the help you need right now. "
                "Please reach out to someone you trust or a mental health professional. "
                "You matter, and you're not alone."
            )
        return None

    # ---------- Intent Handler (same but cleaned & stable) ----------
    def detect_special_cases(self, text):
        t = text.lower().strip()

        # crisis first
        crisis = self.detect_crisis(text)
        if crisis:
            return crisis

        # show last chat
        if "previous chat" in t or "last conversation" in t:
            if not self.past_chats:
                return "You don't have any stored conversations yet."

            last = self.past_chats[-1]["history"][-10:]
            lines = [f"{m['speaker']}: {m['text']}" for m in last]
            return "Here are the last messages from your previous chat:\n" + "\n".join(lines)

        # greetings
        if t in {"hi", "hii", "hello", "hey"} or re.fullmatch(r"h+i+", t):
            name = f" {self.memory['name']}" if self.memory['name'] else ""
            return f"Hello{name}! It’s great connecting with you. How may I assist you today?"

        # how are you
        if "how are you" in t:
            return "I'm functioning well, thank you for asking. How are you doing today?"

        # who am I
        if "who am i" in t:
            if self.memory["name"]:
                return f"You told me earlier your name is {self.memory['name']}."
            return "You haven’t told me your name yet."

        # romantic
        if "love" in t:
            self.memory["tone"] = "casual"
            return "That's sweet of you. I'm always here for a good conversation. 💙"

        # jokes
        if "joke" in t:
            import random
            jokes = [
                "Why do programmers hate nature? Too many bugs! 😄",
                "Why do computers get cold? They forgot to close their Windows! 😂"
            ]
            return random.choice(jokes)

        # bye
        if "bye" in t:
            return "Goodbye! It was a pleasure talking with you."

        return None

    # ---------- Tone Adaptation ----------
    def _apply_tone(self, reply, sentiment):
        tone = self.memory["tone"]
        if tone == "casual" and sentiment == "Positive":
            if "😊" not in reply:
                reply += " 😊"
        return reply

    # ---------- Rule-Based Fallback ----------
    def generate_rule_based_reply(self, sentiment, user_text):
        lower = user_text.lower()

        if sentiment == "Negative":
            reply = "I'm really sorry to hear that. Could you tell me what bothered you the most?"
            return reply

        if sentiment == "Positive":
            return "That's good to hear! What else is on your mind?"

        return "I understand. Tell me more so I can assist you better."

    # ---------- LLM Reply Generator ----------
    def generate_reply(self, sentiment, user_text):
        tone = self.memory["tone"]
        if not self.llm.is_available():
            return self.generate_rule_based_reply(sentiment, user_text)

        try:
            reply = self.llm.generate(
                user_message=user_text,
                sentiment=sentiment,
                tone=tone,
                history=self.history
            )
            return self._apply_tone(reply, sentiment)
        except Exception as e:
            print("⚠️ LLM ERROR:", e)
            return self.generate_rule_based_reply(sentiment, user_text)

    # ---------- Main Handler ----------
    def handle(self, msg):
        label, score = self.analyzer.analyze(msg)
        self.add_user(msg, label, score)

        self.update_memory_from_text(msg)
        self.update_sentiment_stats(label)

        intent = self.detect_special_cases(msg)
        if intent:
            self.add_bot(intent)
            return intent, label, score

        reply = self.generate_reply(label, msg)
        self.add_bot(reply)
        return reply, label, score

    # ---------- Tier 1 Summary ----------
    def summary_sentiment(self):
        user_msgs = [m for m in self.history if m["speaker"] == "user"]
        if not user_msgs:
            return "Neutral", 0.0, "No user messages found."

        pos = sum(1 for m in user_msgs if m["sentiment_label"] == "Positive")
        neg = sum(1 for m in user_msgs if m["sentiment_label"] == "Negative")
        neu = sum(1 for m in user_msgs if m["sentiment_label"] == "Neutral")

        # weighted average
        wsum = sum((i+1)*m["sentiment_score"] for i, m in enumerate(user_msgs))
        wtotal = sum(i+1 for i in range(len(user_msgs)))
        avg = wsum / wtotal

        if avg > 0.05:
            overall = "Positive"
        elif avg < -0.05:
            overall = "Negative"
        else:
            overall = "Neutral"

        explanation = (
            f"You expressed negative feelings {neg} time(s).\n"
            f"You expressed positive feelings {pos} time(s).\n"
            f"You expressed neutral or unclear feelings {neu} time(s).\n"
        )

        if overall == "Negative":
            explanation += "Recent messages leaned negative."
        elif overall == "Positive":
            explanation += "Overall tone leaned positive."
        else:
            explanation += "Your emotions were mixed or balanced."

        return overall, avg, explanation

    # ---------- Tier 2 Report ----------
    def print_tier2_report(self):
        print("=== 📌 Statement-Level Sentiment (Tier 2) ===")
        user_msgs = [m for m in self.history if m["speaker"] == "user"]

        if not user_msgs:
            print("No messages.")
            return

        for i, m in enumerate(user_msgs, start=1):
            print(f"{i}. \"{m['text']}\" → {m['sentiment_label']} ({m['sentiment_score']:+.3f})")

    # ---------- Trend Analysis ----------
    def summarize_trend(self):
        scores = [m["sentiment_score"] for m in self.history if m["speaker"] == "user"]

        if len(scores) < 2:
            return "Not enough data for a trend."

        start = scores[0]
        end = scores[-1]
        change = end - start

        if change > 0.25:
            return "Your emotional tone improved over the conversation."
        elif change < -0.25:
            return "Your emotional tone declined over time."
        return "Your emotional tone stayed relatively stable."

    # ---------- ASCII Graph ----------
    def print_ascii_trend(self):
        print("=== 📈 ASCII Sentiment Trend ===")
        scores = [m["sentiment_score"] for m in self.history if m["speaker"] == "user"]

        if not scores:
            print("No data.")
            return

        for i, s in enumerate(scores, start=1):
            bar = "█" * int(abs(s)*10)
            print(f"{i}: {s:+.3f} {bar}")

    # ---------- Save to Text Log ----------
    def save_to_log_txt(self):
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n=== New Chat ===\n")
            for m in self.history:
                f.write(f"{m['speaker'].upper()}: {m['text']}\n")


# ======================= MAIN =======================

def main():
    bot = Chatbot()

    print("=== Chatbot with Hybrid Sentiment + Groq LLaMA-3.1-8B ===")
    print("Type 'exit' or 'quit' to finish.\n")

    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"exit", "quit"}:
            break

        reply, label, score = bot.handle(msg)

        # clean Tier 2 output (no duplicate lines)
        print(f"→ Sentiment: {label} ({score:+.3f})")
        print("Bot:", reply)

    print("\n=========== SENTIMENT SUMMARY ===========")

    bot.print_tier2_report()
    print("\nTrend:", bot.summarize_trend())
    print()
    bot.print_ascii_trend()

    overall, avg, explanation = bot.summary_sentiment()

    print("===== Final Output =====")
    print(f"Overall conversation sentiment: {overall} – {explanation.splitlines()[-1]}")

    print(f"Average Score: {avg:+.3f}")

    # Save logs
    bot.save_to_log_txt()
    bot.save_persistent_memory()
    print("\n💾 Chat saved.")


if __name__ == "__main__":
    main()

