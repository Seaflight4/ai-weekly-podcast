"""Factory for creating TTS providers."""

from typing import Optional, Type

from .base import TTSProvider


class TTSProviderFactory:
    """Factory class for creating TTS providers."""

    _providers: dict[str, Type[TTSProvider]] = {}
    _module_map = {
        "tng": ".providers.tng",
    }

    @classmethod
    def _get_provider_class(cls, provider_name: str) -> Type[TTSProvider]:
        """Lazily import and return the provider class for a name."""
        provider_name = provider_name.lower()
        if provider_name in cls._providers:
            return cls._providers[provider_name]

        module_path = cls._module_map.get(provider_name)
        if not module_path:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Choose from: {', '.join(list(cls._module_map.keys()) + list(cls._providers.keys()))}"
            )

        try:
            if provider_name == "tng":
                from .providers.tng import TNGTTS as provider_class
            else:
                raise ValueError(f"Unsupported provider: {provider_name}")
        except ImportError as e:
            raise ImportError(
                f"Provider '{provider_name}' is missing its implementation module: {e}"
            ) from e

        cls._providers[provider_name] = provider_class
        return provider_class

    @classmethod
    def create(cls, provider_name: str, api_key: Optional[str] = None, model: Optional[str] = None) -> TTSProvider:
        """
        Create a TTS provider instance.

        Args:
            provider_name: Name of the provider to create
            api_key: Optional API key for the provider
            model: Optional model name for the provider

        Returns:
            TTSProvider instance

        Raises:
            ValueError: If provider_name is not supported
        """
        provider_class = cls._get_provider_class(provider_name)
        return provider_class(api_key, model) if api_key else provider_class(model=model)

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[TTSProvider]) -> None:
        """Register a new provider class."""
        cls._providers[name.lower()] = provider_class
