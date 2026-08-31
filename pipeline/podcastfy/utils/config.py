"""
Configuration Module

This module handles the loading and management of configuration settings for the Podcastfy application.
It uses environment variables to securely store and access API keys and other sensitive information,
and a YAML file for non-sensitive configuration settings.
"""

import os
from dotenv import load_dotenv, find_dotenv
from typing import Any, Dict, Optional
import yaml

def get_config_path(config_file: str = 'config.yaml'):
	"""
	Get the path to the config.yaml file.
	
	Returns:
		str: The path to the config.yaml file, or None if not found.
	"""
	try:
		base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		
		# Look for config.yaml in the package root (pipeline/podcastfy/)
		config_path = os.path.join(base_path, config_file)
		if os.path.exists(config_path):
			return config_path
		
		# If not found, look in the current working directory
		config_path = os.path.join(os.getcwd(), config_file)
		if os.path.exists(config_path):
			return config_path
		
		raise FileNotFoundError(f"{config_file} not found")
	
	except Exception as e:
		print(f"Error locating {config_file}: {str(e)}")
		return None

class Config:
	def __init__(self, config_file: str = 'config.yaml'):
		"""
		Initialize the Config class by loading environment variables and YAML configuration.

		Args:
			config_file (str): Path to the YAML configuration file. Defaults to 'config.yaml'.
		"""
		# Try to find .env file (harmless if absent; the pipeline orchestrator
		# already populates env vars before this runs).
		dotenv_path = find_dotenv(usecwd=True)
		if dotenv_path:
			load_dotenv(dotenv_path)
		
		# Load API keys from environment variables
		# TNG TTS reuses the shared SkaiNet bearer token; alias it so the
		# {MODEL}_API_KEY lookup in TextToSpeech resolves for model="tng".
		self.TNG_API_KEY: str = os.getenv("SKAINET_API_KEY", "")
		
		config_path = get_config_path(config_file)
		if config_path:
			with open(config_path, 'r') as file:
				self.config: Dict[str, Any] = yaml.safe_load(file)
		else:
			print("Could not locate config.yaml")
			self.config = {}
		
		# Set attributes based on YAML config
		self._set_attributes()

	def _set_attributes(self):
		"""Set attributes based on the current configuration."""
		for key, value in self.config.items():
			setattr(self, key.upper(), value)

		# Ensure output directories exist
		if 'output_directories' in self.config:
			for dir_type, dir_path in self.config['output_directories'].items():
				if dir_path:
					os.makedirs(dir_path, exist_ok=True)

	def configure(self, **kwargs):
		"""
		Configure the settings by updating the config dictionary and relevant attributes.

		Args:
			**kwargs: Keyword arguments representing configuration keys and values to update.
		"""
		for key, value in kwargs.items():
			if key in self.config:
				self.config[key] = value
			elif key in ['JINA_API_KEY', 'TNG_API_KEY']:
				setattr(self, key, value)
			else:
				raise ValueError(f"Unknown configuration key: {key}")

		# Update attributes based on the new configuration
		self._set_attributes()

	def get(self, key: str, default: Optional[Any] = None) -> Any:
		"""
		Get a configuration value by key.

		Args:
			key (str): The configuration key to retrieve.
			default (Optional[Any]): The default value if the key is not found.

		Returns:
			Any: The value associated with the key, or the default value if not found.
		"""
		return self.config.get(key, default)

def load_config() -> Config:
	"""
	Load and return a Config instance.

	Returns:
		Config: An instance of the Config class.
	"""
	return Config()
