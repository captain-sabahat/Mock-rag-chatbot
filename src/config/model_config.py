"""
================================================================================
FILE: src/config/model_config.py
================================================================================

PURPOSE:
    Model-specific configuration parameters. Separate from general settings
    to keep model logic organized. Defines parameters for LLM, SLM, embeddings.

WORKFLOW:
    1. Define model-specific parameters (temperature, max tokens, etc.)
    2. Organize by model type
    3. Use in model initialization and inference
    4. Enable easy tuning (parameters in one place)

IMPORTS:
    - Pydantic for validation
    - Typing for type hints

INPUTS:
    - None (loaded from module)

OUTPUTS:
    - Model-specific configuration dicts

MODEL CONFIGURATION SECTIONS:
    1. Fine-Tuned LLM Configuration
       - temperature: 0-1 (higher = more creative)
       - max_tokens: max output length
       - top_p: nucleus sampling
       - frequency_penalty: penalize repetition
    
    2. SLM Configuration
       - max_length: max summary length
       - min_length: min summary length
       - num_beams: beam search width
       - length_penalty: prefer longer/shorter output
    
    3. Embeddings Configuration
       - normalize_embeddings: L2 normalization
       - show_progress_bar: show progress during encoding
       - batch_size: inference batch size

KEY FACTS:
    - Model-specific, not general config
    - Used during model initialization
    - Can be tuned for better results
    - Pydantic validation ensures correctness

TUNING GUIDE:
    - Temperature: 0 (deterministic) to 1 (random)
    - max_tokens: Larger = longer responses but more latency
    - beam_search: Larger = better quality but slower
    - num_beams: 1 (greedy) to 5 (beam search)

FUTURE SCOPE (Phase 2+):
    - Add model-specific hyperparameters
    - Add A/B testing variants
    - Add per-user configuration overrides
    - Add dynamic configuration adjustment
    - Add model-specific monitoring

TESTING ENVIRONMENT:
    - Use smaller values for faster tests
    - Disable progress bars in tests

PRODUCTION DEPLOYMENT:
    - Use carefully tuned values
    - Monitor quality metrics
    - A/B test parameter changes
"""

# ================================================================================
# IMPORTS
# ================================================================================

from pydantic import BaseModel, Field
from typing import Optional

# ================================================================================
# FINE-TUNED LLM CONFIGURATION
# ================================================================================

class FineTunedLLMConfig(BaseModel):
    """
    Configuration for fine-tuned LLM (answer generation).
    
    These parameters control generation quality and latency.
    Tune based on use case and performance requirements.
    """
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Generation temperature (0=deterministic, 1=random)"
    )
    
    max_tokens: int = Field(
        default=500,
        ge=10,
        le=2000,
        description="Maximum tokens in response"
    )
    
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling (top_p)"
    )
    
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalize repeated tokens"
    )
    
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalize new tokens"
    )
    
    # ========================================================================
    # TUNING RECOMMENDATIONS
    # ========================================================================
    
    # For factual/technical content: temperature=0.3-0.5, top_p=0.9
    # For creative content: temperature=0.7-0.9, top_p=0.95
    # For consistent output: temperature=0.0 (greedy)
    # For diverse output: temperature=0.9, top_p=1.0
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "Factual (Low Temperature)",
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 500
                },
                {
                    "name": "Balanced",
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                },
                {
                    "name": "Creative (High Temperature)",
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "max_tokens": 500
                }
            ]
        }


LLM_CONFIG_DEFAULT = FineTunedLLMConfig()

# ================================================================================
# SLM CONFIGURATION
# ================================================================================

class SLMConfig(BaseModel):
    """
    Configuration for Small Language Model (document summarization).
    
    These parameters control summary quality and latency.
    Tuning affects both quality and speed.
    """
    
    max_length: int = Field(
        default=150,
        ge=50,
        le=500,
        description="Maximum summary length"
    )
    
    min_length: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Minimum summary length"
    )
    
    num_beams: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Beam search width (1=greedy, >1=beam search)"
    )
    
    length_penalty: float = Field(
        default=2.0,
        ge=0.0,
        le=5.0,
        description="Penalty for long/short summaries"
    )
    
    early_stopping: bool = Field(
        default=True,
        description="Stop generation when all beams finish"
    )
    
    # ========================================================================
    # TUNING RECOMMENDATIONS
    # ========================================================================
    
    # For speed: num_beams=1, max_length=100
    # For quality: num_beams=4, max_length=150
    # For very short summary: max_length=50, length_penalty=3.0
    # For longer summary: max_length=300, length_penalty=1.0
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "Fast",
                    "num_beams": 1,
                    "max_length": 100,
                    "length_penalty": 2.0
                },
                {
                    "name": "Balanced",
                    "num_beams": 4,
                    "max_length": 150,
                    "length_penalty": 2.0
                },
                {
                    "name": "High Quality",
                    "num_beams": 8,
                    "max_length": 200,
                    "length_penalty": 2.0
                }
            ]
        }


SLM_CONFIG_DEFAULT = SLMConfig()

# ================================================================================
# EMBEDDINGS CONFIGURATION
# ================================================================================

class EmbeddingsConfig(BaseModel):
    """
    Configuration for embeddings model (query/document embeddings).
    
    These parameters control embedding generation behavior.
    Affects both latency and quality.
    """
    
    normalize_embeddings: bool = Field(
        default=True,
        description="L2 normalize embeddings"
    )
    
    show_progress_bar: bool = Field(
        default=False,
        description="Show progress bar during encoding"
    )
    
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1000,
        description="Batch size for encoding"
    )
    
    convert_to_tensor: bool = Field(
        default=True,
        description="Convert embeddings to tensor"
    )
    
    # ========================================================================
    # TUNING RECOMMENDATIONS
    # ========================================================================
    
    # For speed: batch_size=64, normalize_embeddings=False
    # For quality: batch_size=32, normalize_embeddings=True
    # For CPU: batch_size=16
    # For GPU: batch_size=64-128
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "CPU (Small batch)",
                    "batch_size": 16,
                    "normalize_embeddings": True
                },
                {
                    "name": "Balanced",
                    "batch_size": 32,
                    "normalize_embeddings": True
                },
                {
                    "name": "GPU (Large batch)",
                    "batch_size": 128,
                    "normalize_embeddings": True
                }
            ]
        }


EMBEDDINGS_CONFIG_DEFAULT = EmbeddingsConfig()

# ================================================================================
# INFERENCE CONFIGURATION
# ================================================================================

class InferenceConfig(BaseModel):
    """
    Global inference configuration for all models.
    
    Settings that apply to all model inference.
    """
    
    enable_gpu_acceleration: bool = Field(
        default=True,
        description="Use GPU if available"
    )
    
    enable_mixed_precision: bool = Field(
        default=True,
        description="Use float16 on GPU (faster but less precise)"
    )
    
    enable_model_caching: bool = Field(
        default=True,
        description="Cache models in memory"
    )
    
    max_model_instances: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Max model instances (for pooling in Phase 2)"
    )


INFERENCE_CONFIG_DEFAULT = InferenceConfig()

# ================================================================================
# CONSOLIDATE MODEL CONFIGS
# ================================================================================

MODEL_CONFIGS = {
    "llm": LLM_CONFIG_DEFAULT.dict(),
    "slm": SLM_CONFIG_DEFAULT.dict(),
    "embeddings": EMBEDDINGS_CONFIG_DEFAULT.dict(),
    "inference": INFERENCE_CONFIG_DEFAULT.dict(),
}
