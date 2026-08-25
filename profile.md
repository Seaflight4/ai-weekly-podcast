---
# Listener profile — drives NotebookLM audio style.
#
# All fields are optional; defaults are shown. The pipeline ranks items
# purely by general importance (no personal topic preference).
#
#   knowledge_level: researcher | undergrad
#   pace:            deep_dive | brief
#   format:          deep_dive | brief | critique | debate  (-> AudioFormat)
#   target_length:   default | short | long                (-> AudioLength)
#
# Source selection: items with final_score >= 0.8 are "must include".
# pace controls the floor/cap when fewer/more than the threshold qualify:
#   deep_dive: 5-10 items
#   brief:     10-20 items
knowledge_level: undergrad
pace: deep_dive
format: deep_dive
target_length: default
---
