import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
from typing import Dict, Any, List, Optional

class AdvancedModelArchitectures:
    @staticmethod
    def create_pytorch_model(model_config: Dict[str, Any]) -> nn.Module:
        """
        Create advanced PyTorch model architectures
        """
        model_type = model_config.get('type', 'mlp')
        input_size = model_config.get('input_size', 10)
        output_size = model_config.get('output_size', 1)
        hidden_layers = model_config.get('hidden_layers', [64, 32])

        if model_type == 'mlp':
            layers = []
            layer_sizes = [input_size] + hidden_layers + [output_size]
            
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                if i < len(layer_sizes) - 2:  # Don't add activation to the last layer
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.2))  # Add dropout for regularization
            
            return nn.Sequential(*layers)

        elif model_type == 'lstm':
            return LSTMModel(input_size, hidden_layers[0], output_size)

        elif model_type == 'transformer':
            return TransformerModel(input_size, output_size, 
                                    n_heads=model_config.get('n_heads', 4),
                                    n_layers=model_config.get('n_layers', 2))

        else:
            raise ValueError(f"Unsupported PyTorch model type: {model_type}")

    @staticmethod
    def create_tensorflow_model(model_config: Dict[str, Any]) -> tf.keras.Model:
        """
        Create advanced TensorFlow model architectures
        """
        model_type = model_config.get('type', 'sequential')
        input_size = model_config.get('input_size', 10)
        output_size = model_config.get('output_size', 1)
        hidden_layers = model_config.get('hidden_layers', [64, 32])

        if model_type == 'sequential':
            model = tf.keras.Sequential()
            
            # Input layer
            model.add(tf.keras.layers.InputLayer(input_shape=(input_size,)))
            
            # Hidden layers
            for units in hidden_layers:
                model.add(tf.keras.layers.Dense(units, activation='relu'))
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dropout(0.2))
            
            # Output layer
            model.add(tf.keras.layers.Dense(output_size))
            
            return model

        elif model_type == 'cnn':
            return CNNModel(input_size, output_size)

        elif model_type == 'rnn':
            return RNNModel(input_size, output_size)

        else:
            raise ValueError(f"Unsupported TensorFlow model type: {model_type}")

class LSTMModel(nn.Module):
    """
    LSTM model for sequence processing
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM expects input (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

class TransformerModel(nn.Module):
    """
    Transformer model for advanced sequence processing
    """
    def __init__(self, input_size, output_size, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_size, input_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

class CNNModel(tf.keras.Model):
    """
    Convolutional Neural Network for TensorFlow
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.reshape = tf.keras.layers.Reshape((input_size, 1))
        self.conv1 = tf.keras.layers.Conv1D(32, 3, activation='relu')
        self.pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dense1(x)
        return self.dense2(x)

class RNNModel(tf.keras.Model):
    """
    Recurrent Neural Network for TensorFlow
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.reshape = tf.keras.layers.Reshape((input_size, 1))
        self.rnn = tf.keras.layers.LSTM(64)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        x = self.reshape(inputs)
        x = self.rnn(x)
        x = self.dense1(x)
        return self.dense2(x)

class ModelRegistry:
    """
    Registry for managing and discovering available model architectures
    """
    _pytorch_models = {
        'mlp': AdvancedModelArchitectures.create_pytorch_model,
        'lstm': AdvancedModelArchitectures.create_pytorch_model,
        'transformer': AdvancedModelArchitectures.create_pytorch_model
    }

    _tensorflow_models = {
        'sequential': AdvancedModelArchitectures.create_tensorflow_model,
        'cnn': AdvancedModelArchitectures.create_tensorflow_model,
        'rnn': AdvancedModelArchitectures.create_tensorflow_model
    }

    @classmethod
    def get_available_pytorch_models(cls) -> List[str]:
        """
        Get list of available PyTorch model architectures
        """
        return list(cls._pytorch_models.keys())

    @classmethod
    def get_available_tensorflow_models(cls) -> List[str]:
        """
        Get list of available TensorFlow model architectures
        """
        return list(cls._tensorflow_models.keys())

    @classmethod
    def create_model(
        cls, 
        framework: str, 
        model_config: Dict[str, Any]
    ) -> Optional[nn.Module]:
        """
        Create a model based on framework and configuration
        """
        if framework == 'pytorch':
            return cls._pytorch_models[model_config.get('type', 'mlp')](model_config)
        elif framework == 'tensorflow':
            return cls._tensorflow_models[model_config.get('type', 'sequential')](model_config)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
