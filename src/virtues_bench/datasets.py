"""Deterministic synthetic inputs for the bench workloads.

The exact text doesn't matter — the model burns the same FLOPs on gibberish as
on real prose, as long as the token count after BPE matches what we'd see in
production. Seeded RNGs guarantee bit-identical inputs across boards and runs.

If we ever want "real-looking" inputs for marketing, swap this module without
touching workloads: they only depend on `embed_inputs()` and `rerank_inputs()`
returning deterministic lists of strings.
"""

from __future__ import annotations

import random

# ~256 common English words. BPE-friendly (most are 1 token under bge's
# tokenizer), so target_tokens ~= word count.
WORDLIST: tuple[str, ...] = (
    "about", "above", "across", "after", "again", "against", "all", "almost",
    "also", "always", "among", "and", "another", "any", "anyone", "anything",
    "around", "as", "at", "back", "be", "because", "become", "been", "before",
    "begin", "behind", "being", "below", "best", "better", "between", "big",
    "book", "both", "but", "by", "call", "can", "case", "change", "child",
    "city", "come", "company", "could", "country", "course", "day", "different",
    "do", "does", "down", "during", "each", "early", "end", "enough", "even",
    "ever", "every", "example", "eye", "face", "fact", "family", "far", "feel",
    "few", "find", "first", "follow", "for", "form", "from", "get", "give",
    "go", "good", "government", "great", "group", "hand", "happen", "have",
    "he", "head", "hear", "help", "here", "high", "him", "his", "home", "house",
    "how", "however", "if", "important", "in", "include", "into", "issue", "it",
    "its", "just", "keep", "kind", "know", "large", "last", "late", "lead",
    "learn", "leave", "left", "less", "let", "life", "like", "line", "little",
    "live", "local", "long", "look", "lot", "love", "low", "made", "make",
    "man", "many", "may", "mean", "meet", "might", "mind", "minute", "moment",
    "money", "more", "most", "mother", "move", "much", "must", "my", "name",
    "national", "need", "never", "new", "next", "night", "no", "not", "nothing",
    "now", "number", "of", "off", "offer", "office", "often", "on", "once",
    "one", "only", "open", "or", "order", "other", "our", "out", "over", "own",
    "page", "part", "party", "people", "perhaps", "person", "place", "plan",
    "play", "point", "policy", "political", "possible", "power", "present",
    "problem", "process", "program", "provide", "public", "put", "question",
    "quite", "rather", "reach", "read", "real", "really", "reason", "remember",
    "report", "result", "right", "room", "run", "same", "say", "school",
    "second", "see", "seem", "send", "service", "set", "should", "show",
    "side", "since", "small", "social", "some", "something", "sometimes",
    "soon", "sort", "speak", "special", "start", "state", "stay", "still",
    "stop", "story", "such", "suggest", "support", "sure", "system", "take",
    "talk", "tell", "than", "thank", "that", "their", "them", "then", "there",
    "these", "they", "thing", "think", "this", "those", "though", "thought",
    "three", "through", "time", "to", "today", "together", "too", "toward",
    "try", "turn", "two", "under", "until", "up", "upon", "us", "use", "used",
    "user", "very", "want", "way", "we", "week", "well", "what", "when",
    "where", "whether", "which", "while", "who", "why", "will", "with",
    "within", "without", "work", "world", "would", "year", "yes", "you",
    "young", "your",
)

EMBED_SEED = 0xBEEF
RERANK_SEED = 0xCAFE
K8_VECTOR_SEED = 0xFEED


def embed_inputs(n: int = 1000, target_words: int = 15, seed: int = EMBED_SEED) -> list[str]:
    """Short strings for K3 (throughput) and K4 (latency)."""
    rng = random.Random(seed)
    return [" ".join(rng.choices(WORDLIST, k=target_words)) for _ in range(n)]


def rerank_inputs(
    n_candidates: int = 50,
    candidate_words: int = 200,
    query_words: int = 10,
    seed: int = RERANK_SEED,
) -> tuple[str, list[str]]:
    """One query + N candidates for K1, K2."""
    rng = random.Random(seed)
    query = " ".join(rng.choices(WORDLIST, k=query_words))
    candidates = [
        " ".join(rng.choices(WORDLIST, k=candidate_words)) for _ in range(n_candidates)
    ]
    return query, candidates


def k8_corpus(n: int = 10_000, target_words: int = 200, seed: int = K8_VECTOR_SEED) -> list[str]:
    """Strings to embed for K8's pgvector population."""
    rng = random.Random(seed)
    return [" ".join(rng.choices(WORDLIST, k=target_words)) for _ in range(n)]
