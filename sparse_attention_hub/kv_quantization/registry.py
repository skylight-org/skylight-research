"""Independent config-type registries for replaceable components."""

from typing import Any, Callable, Dict, Type


class ComponentRegistry:
    _implementations: Dict[Type[Any], Type[Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._implementations = {}

    @classmethod
    def register(cls, config_type: Type[Any]) -> Callable[[Type[Any]], Type[Any]]:
        def decorator(implementation: Type[Any]) -> Type[Any]:
            if config_type in cls._implementations:
                raise ValueError(f"{config_type.__name__} is already registered")
            cls._implementations[config_type] = implementation
            return implementation

        return decorator

    @classmethod
    def get_class(cls, config_type: Type[Any]) -> Type[Any]:
        try:
            return cls._implementations[config_type]
        except KeyError as exc:
            raise ValueError(
                f"Unregistered configuration: {config_type.__name__}"
            ) from exc

    @classmethod
    def create(cls, config: Any) -> Any:
        return cls.get_class(type(config))(config)


class KVQuantizationRegistry(ComponentRegistry):
    pass


class KVTransformRegistry(ComponentRegistry):
    pass


class KVQuantizerRegistry(ComponentRegistry):
    pass


class PrecisionPolicyRegistry(ComponentRegistry):
    pass


class CachePolicyRegistry(ComponentRegistry):
    pass
