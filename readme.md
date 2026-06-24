# Pro-Align Module Overview

This folder contains the anonymized reinforcement-learning alignment workflow used after supervised initialization.
The released scripts preserve the public interfaces and method-level flow:

1. Build SFT-style dialogue examples from annotated training data.
2. Train a reward model over multiple educational-quality dimensions.
3. Use PPO/GRPO-style policy optimization with reward-model feedback.
4. Evaluate with and without profile-aware retrieval.

Exact prompt templates, reward aggregation constants, local checkpoint paths, and copy-ready optimization internals
are redacted in this review-facing package.
