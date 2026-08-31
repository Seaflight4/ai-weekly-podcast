"""
Conversation Configuration Module

This module provides a default conversation configuration and a loader that
returns a conversation config dict. The minimal bundled version does not
require a YAML file; any provided config is merged over sensible defaults.
"""

import os
from typing import Any, Dict, Optional
import yaml


DEFAULT_CONVERSATION_CONFIG: Dict[str, Any] = {
    "podcast_name": "AI News Weekly",
    "podcast_tagline": "Latest AI research and news",
    "conversation_style": ["informative", "engaging"],
    "roles_person1": "AI researcher",
    "roles_person2": "AI researcher",
    "dialogue_structure": ["conversation", "exchange"],
    "output_language": "English",
    "engagement_techniques": ["examples", "analogies"],
    "text_to_speech": {
        "audio_format": "mp3",
        "ending_message": "",
        "temp_audio_dir": "data/audio/tmp/",
        # Concurrency for per-piece audio generation. Set to 1 for sequential.
        "max_workers": 4,
        "output_directories": {
            "transcripts": "output/transcripts",
            "audio": "output/audio",
        },
        "tng": {
            "model": "qwen3",
            "instructions": "Speak at a natural, unhurried pace. Keep the delivery lively and conversational.",
            "speed": "1.0",
            "temperature": "0.5",
            "reference_voices": {
                "person1": "reference/host_m.wav",
                "person2": "reference/host_f.wav",
            },
        },
    },
}


def get_conversation_config_path(config_file: str = "config_conversation.yaml") -> Optional[str]:
    """Locate a conversation config YAML file in the package root or CWD."""
    try:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_path, config_file)
        if os.path.exists(config_path):
            return config_path
        config_path = os.path.join(os.getcwd(), config_file)
        if os.path.exists(config_path):
            return config_path
    except Exception:
        pass
    return None


def load_config_from_file(config_file: str = "config_conversation.yaml") -> Dict[str, Any]:
    """Load conversation config from a YAML file if present, else return defaults."""
    config_path = get_conversation_config_path(config_file)
    if config_path:
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_conversation_config(conversation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load conversation configuration merging defaults, file-based config and
    any explicitly provided dict.

    Args:
        conversation_config (Optional[Dict]): Extra config to override defaults.

    Returns:
        Dict[str, Any]: Merged conversation configuration.
    """
    config: Dict[str, Any] = {}
    config.update(DEFAULT_CONVERSATION_CONFIG)
    config.update(load_config_from_file())
    if conversation_config:
        for k, v in conversation_config.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                merged = config[k].copy()
                merged.update(v)
                config[k] = merged
            else:
                config[k] = v
    return config
