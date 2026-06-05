from app.providers.base import LLMProvider, ProviderError


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self._model_index: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        if provider.name in self._providers:
            msg = f"Provider {provider.name!r} is already registered"
            raise ProviderError(msg)
        self._providers[provider.name] = provider
        for model in provider.models:
            if model in self._model_index:
                existing = self._model_index[model].name
                msg = f"Model {model!r} already registered to provider {existing!r}"
                raise ProviderError(msg)
            self._model_index[model] = provider

    def get_provider(self, name: str) -> LLMProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            msg = f"Provider {name!r} is not registered"
            raise ProviderError(msg) from exc

    def get_provider_for_model(self, model: str) -> LLMProvider:
        try:
            return self._model_index[model]
        except KeyError as exc:
            msg = f"No provider registered for model {model!r}"
            raise ProviderError(msg) from exc

    def list_providers(self) -> list[str]:
        return sorted(self._providers)

    def list_models(self) -> list[str]:
        return sorted(self._model_index)
