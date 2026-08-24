---
# Listener profile — drives the personal-match pass in `rank`.
#
# Schema:
#   topics:      [strings] research directions / methods / tools you're working
#                       on this quarter. Promote items matching these.
#   anti_topics: [strings] topics you want less of. Demote items matching these.
#
# The prose body below the frontmatter is passed verbatim to the personal-match
# prompt as context. Edit freely; this is the only content you tune.
#
# The personal-match prompt itself is fixed (see PERSONAL_PROMPT in rank.py) —
# tune the profile, not the prompt.
topics:
  - character consistency in image generation
  - scene consistency across panels
  - style consistency for comics
  - diffusion models (ControlNet, IP-Adapter, reference-based generation)
  - low-cost / mass-production techniques for image pipelines
  - story-to-image / script-to-comic generation
  - multimodal LLMs for visual reasoning
anti_topics:
  - pure scaling laws without methodological insight
  - niche benchmarks with no path to production
  - RLHF for safety / alignment (not relevant to comics generation)
---

## What I'm working on this quarter

I'm building a system that generates comics from text scripts. The hard
problems are character consistency (the same character must look identical
across every panel), scene consistency (backgrounds and objects must stay
coherent across panels), and style consistency (one art style per comic, not a
mix). I care about methods that work at production scale: low per-image cost,
high throughput, and minimal per-panel human correction. Diffusion-based
approaches (ControlNet, IP-Adapter, reference-image conditioning, LoRA for
character locking) and any multimodal-LLM technique that improves visual
coherence are directly relevant. I'm less interested in pure scaling results
or alignment/safety work that doesn't touch the consistency problem.
