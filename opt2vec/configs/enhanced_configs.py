"""
Configuration system for enhanced Opt2Vec experiments.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class EnhancedNetworkConfig:
    """Configuration for enhanced network architecture."""

    # Embedding dimensions to test
    embedding_dim: int = 64  # [32, 64, 128]

    # History lengths to test
    history_length: int = 8  # [8, 16, 32]

    # Activation functions to test
    activation: str = 'gelu'  # ['gelu', 'swish', 'relu']

    # Network architecture
    hidden_dim: int = 128
    num_residual_blocks: int = 3
    dropout: float = 0.1
    use_layer_norm: bool = True
    use_attention: bool = True
    use_positional_encoding: bool = True
    num_attention_heads: int = 4

    # Feature configuration
    use_extended_features: bool = True
    normalize_features: bool = True

    # Optimizer configuration
    base_lr: float = 0.01
    max_grad_norm: float = 1.0
    lr_bounds: tuple = (1e-6, 1e2)
    momentum_bounds: tuple = (0.0, 0.99)


class EnhancedExperimentConfigs:
    """
    Predefined configurations for systematic experiments.
    """

    @staticmethod
    def get_embedding_dim_experiments() -> List[EnhancedNetworkConfig]:
        """Get configurations for embedding dimension experiments."""
        configs = []

        for embedding_dim in [32, 64, 128]:
            config = EnhancedNetworkConfig(
                embedding_dim=embedding_dim,
                history_length=16,
                activation='gelu',
                hidden_dim=max(64, embedding_dim * 2),
                use_extended_features=True,
                use_attention=True,
                use_positional_encoding=True
            )
            configs.append(config)

        return configs

    @staticmethod
    def get_history_length_experiments() -> List[EnhancedNetworkConfig]:
        """Get configurations for history length experiments."""
        configs = []

        for history_length in [8, 16, 32]:
            config = EnhancedNetworkConfig(
                embedding_dim=64,
                history_length=history_length,
                activation='gelu',
                use_extended_features=True,
                use_attention=True,
                use_positional_encoding=True
            )
            configs.append(config)

        return configs

    @staticmethod
    def get_activation_experiments() -> List[EnhancedNetworkConfig]:
        """Get configurations for activation function experiments."""
        configs = []

        for activation in ['gelu', 'swish', 'relu']:
            config = EnhancedNetworkConfig(
                embedding_dim=64,
                history_length=16,
                activation=activation,
                use_extended_features=True,
                use_attention=True,
                use_positional_encoding=True
            )
            configs.append(config)

        return configs

    @staticmethod
    def get_architecture_experiments() -> List[EnhancedNetworkConfig]:
        """Get configurations for architectural feature experiments."""
        configs = []

        # Base configuration
        base_config = EnhancedNetworkConfig(
            embedding_dim=64,
            history_length=16,
            activation='gelu',
            use_extended_features=True
        )

        # Test with and without attention
        config1 = EnhancedNetworkConfig(**base_config.__dict__)
        config1.use_attention = True
        config1.use_positional_encoding = True
        configs.append(config1)

        config2 = EnhancedNetworkConfig(**base_config.__dict__)
        config2.use_attention = False
        config2.use_positional_encoding = False
        configs.append(config2)

        # Test with and without LayerNorm
        config3 = EnhancedNetworkConfig(**base_config.__dict__)
        config3.use_layer_norm = True
        configs.append(config3)

        config4 = EnhancedNetworkConfig(**base_config.__dict__)
        config4.use_layer_norm = False
        configs.append(config4)

        # Test different dropout rates
        for dropout in [0.0, 0.1, 0.2]:
            config = EnhancedNetworkConfig(**base_config.__dict__)
            config.dropout = dropout
            configs.append(config)

        return configs

    @staticmethod
    def get_feature_experiments() -> List[EnhancedNetworkConfig]:
        """Get configurations for feature set experiments."""
        configs = []

        # Basic features only
        config1 = EnhancedNetworkConfig(
            embedding_dim=64,
            history_length=16,
            activation='gelu',
            use_extended_features=False,
            use_attention=True,
            use_positional_encoding=True
        )
        configs.append(config1)

        # Extended features
        config2 = EnhancedNetworkConfig(
            embedding_dim=64,
            history_length=16,
            activation='gelu',
            use_extended_features=True,
            use_attention=True,
            use_positional_encoding=True
        )
        configs.append(config2)

        # Extended features without normalization
        config3 = EnhancedNetworkConfig(
            embedding_dim=64,
            history_length=16,
            activation='gelu',
            use_extended_features=True,
            normalize_features=False,
            use_attention=True,
            use_positional_encoding=True
        )
        configs.append(config3)

        return configs

    @staticmethod
    def get_comprehensive_experiments() -> List[EnhancedNetworkConfig]:
        """Get comprehensive set of experiments covering all dimensions."""
        configs = []

        # Embedding dimension experiments
        configs.extend(EnhancedExperimentConfigs.get_embedding_dim_experiments())

        # History length experiments
        configs.extend(EnhancedExperimentConfigs.get_history_length_experiments())

        # Activation function experiments
        configs.extend(EnhancedExperimentConfigs.get_activation_experiments())

        # Architecture experiments
        configs.extend(EnhancedExperimentConfigs.get_architecture_experiments())

        # Feature experiments
        configs.extend(EnhancedExperimentConfigs.get_feature_experiments())

        return configs

    @staticmethod
    def get_best_practices_config() -> EnhancedNetworkConfig:
        """Get recommended configuration based on best practices."""
        return EnhancedNetworkConfig(
            embedding_dim=64,
            history_length=8,
            activation='gelu',
            hidden_dim=128,
            num_residual_blocks=3,
            dropout=0.1,
            use_layer_norm=True,
            use_attention=True,
            use_positional_encoding=True,
            num_attention_heads=4,
            use_extended_features=True,
            normalize_features=True,
            base_lr=0.01,
            max_grad_norm=1.0
        )


def config_to_dict(config: EnhancedNetworkConfig) -> Dict[str, Any]:
    """Convert configuration to dictionary for logging."""
    return {
        'embedding_dim': config.embedding_dim,
        'history_length': config.history_length,
        'activation': config.activation,
        'hidden_dim': config.hidden_dim,
        'num_residual_blocks': config.num_residual_blocks,
        'dropout': config.dropout,
        'use_layer_norm': config.use_layer_norm,
        'use_attention': config.use_attention,
        'use_positional_encoding': config.use_positional_encoding,
        'num_attention_heads': config.num_attention_heads,
        'use_extended_features': config.use_extended_features,
        'normalize_features': config.normalize_features,
        'base_lr': config.base_lr,
        'max_grad_norm': config.max_grad_norm
    }


def dict_to_config(config_dict: Dict[str, Any]) -> EnhancedNetworkConfig:
    """Convert dictionary to configuration."""
    return EnhancedNetworkConfig(**config_dict)
