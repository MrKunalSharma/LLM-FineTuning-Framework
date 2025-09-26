"""Test advanced features."""
import pytest
import asyncio
from src.api.streaming import StreamingInference
from src.api.model_registry import ModelRegistry
from src.api.cache import InferenceCache

class TestAdvancedFeatures:
    """Test suite for advanced features."""
    
    @pytest.mark.asyncio
    async def test_streaming_inference(self):
        """Test streaming response generation."""
        streamer = StreamingInference(None, None)  # Mock objects
        
        tokens = []
        async for chunk in streamer.stream_generate("Test prompt", max_tokens=5):
            tokens.append(chunk)
        
        assert len(tokens) > 0
        assert 'finished' in tokens[-1]
    
    def test_model_registry(self):
        """Test model registry and A/B testing."""
        registry = ModelRegistry()
        
        # Register models
        registry.register_model("phi-2", "1.0", "path1", {"bleu": 0.45})
        registry.register_model("phi-2", "1.1", "path2", {"bleu": 0.48})
        
        # Set traffic split
        registry.set_traffic_split({
            "phi-2_v1.0": 30.0,
            "phi-2_v1.1": 70.0
        })
        
        # Test selection distribution
        selections = [registry.select_model() for _ in range(1000)]
        v11_count = selections.count("phi-2_v1.1")
        
        # Should be approximately 70%
        assert 650 < v11_count < 750
    
    def test_cache_operations(self):
        """Test caching functionality."""
        cache = InferenceCache()
        
        # Test set and get
        cache.set(
            "Translate", 
            "Hello",
            {"temperature": 0.7},
            "Bonjour",
            {"model": "test"}
        )
        
        result = cache.get("Translate", "Hello", {"temperature": 0.7})
        assert result is not None
        assert result["response"] == "Bonjour"

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
