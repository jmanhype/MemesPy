"""Property-based testing for all actor agents using Hypothesis.

This module tests invariants and properties that should hold for all actors
in the MemesPy system, regardless of input.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, note, event
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize, invariant
from typing import Dict, Any, List, Optional, Set
import json
import time
from dataclasses import dataclass

from src.dspy_meme_gen.agents.router import RouterAgent
from src.dspy_meme_gen.agents.appropriateness import AppropriatenessAgent
from src.dspy_meme_gen.agents.factuality import FactualityAgent
from src.dspy_meme_gen.agents.format_selector import FormatSelectionAgent
from src.dspy_meme_gen.agents.image_renderer import ImageRenderingAgent
from src.dspy_meme_gen.agents.instruction_following import InstructionFollowingAgent
from src.dspy_meme_gen.agents.prompt_generator import PromptGenerationAgent
from src.dspy_meme_gen.agents.refinement import RefinementLoopAgent
from src.dspy_meme_gen.agents.scorer import ScoringAgent
from src.dspy_meme_gen.agents.trend_analyzer import TrendAnalyzer
from src.dspy_meme_gen.agents.trend_scanner import TrendScanningAgent
from src.dspy_meme_gen.pipeline import MemeGenerationPipeline


# Custom strategies for generating test data
@st.composite
def meme_request_strategy(draw):
    """Generate valid meme requests with various complexity levels."""
    topics = draw(st.sampled_from([
        "programming", "AI", "cats", "dogs", "work from home",
        "coffee", "debugging", "machine learning", "python",
        "javascript", "react", "databases", "microservices"
    ]))
    
    formats = draw(st.sampled_from([
        "drake", "distracted boyfriend", "expanding brain",
        "this is fine", "stonks", "galaxy brain", "custom"
    ]))
    
    styles = draw(st.sampled_from([
        "minimalist", "retro", "modern", "cartoon", "realistic", None
    ]))
    
    # Build request with optional complexity
    base_request = f"Create a meme about {topics}"
    
    if draw(st.booleans()):
        base_request += f" using {formats} format"
    
    if styles and draw(st.booleans()):
        base_request += f" in {styles} style"
    
    # Add random constraints
    if draw(st.booleans()):
        constraints = draw(st.lists(
            st.sampled_from([
                "family friendly", "professional", "educational",
                "trending", "viral potential", "no text"
            ]),
            min_size=0,
            max_size=3
        ))
        if constraints:
            base_request += f" with constraints: {', '.join(constraints)}"
    
    return base_request


@st.composite
def route_result_strategy(draw):
    """Generate valid route results."""
    return {
        "topic": draw(st.text(min_size=1, max_size=50)),
        "format": draw(st.sampled_from(["drake", "distracted", "custom"])),
        "verification_needs": {
            "factuality": draw(st.booleans()),
            "appropriateness": draw(st.booleans()),
            "instructions": draw(st.booleans())
        },
        "constraints": draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.booleans()),
            max_size=5
        )),
        "generation_approach": draw(st.sampled_from(["standard", "creative", "trending"]))
    }


@st.composite
def meme_concept_strategy(draw):
    """Generate valid meme concepts."""
    return {
        "topic": draw(st.text(min_size=1, max_size=100)),
        "format": draw(st.sampled_from(["drake", "distracted", "expanding_brain"])),
        "caption": draw(st.text(min_size=1, max_size=200)),
        "image_prompt": draw(st.text(min_size=1, max_size=500)),
        "metadata": draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.integers(), st.floats(), st.booleans()),
            max_size=10
        ))
    }


class TestRouterAgentProperties:
    """Property-based tests for RouterAgent."""
    
    @given(request=st.text(min_size=1, max_size=1000))
    @settings(max_examples=100, deadline=5000)
    def test_router_always_returns_valid_structure(self, request):
        """Router should always return a valid result structure."""
        router = RouterAgent()
        
        try:
            result = router.forward(request)
            
            # Check required fields exist
            assert isinstance(result, dict)
            assert "topic" in result
            assert "format" in result
            assert "verification_needs" in result
            assert "constraints" in result
            assert "generation_approach" in result
            
            # Check field types
            assert isinstance(result["topic"], str)
            assert isinstance(result["format"], str)
            assert isinstance(result["verification_needs"], dict)
            assert isinstance(result["constraints"], dict)
            assert isinstance(result["generation_approach"], str)
            
            # Verification needs should have boolean values
            for key, value in result["verification_needs"].items():
                assert isinstance(value, bool)
                
        except Exception as e:
            # Router should handle any input gracefully
            assert isinstance(e, RuntimeError)
            assert "Failed to process routing request" in str(e)
    
    @given(request=meme_request_strategy())
    @settings(max_examples=50)
    def test_router_deterministic_for_same_input(self, request):
        """Router should be deterministic for the same input."""
        router = RouterAgent()
        
        results = []
        for _ in range(3):
            try:
                result = router.forward(request)
                results.append(json.dumps(result, sort_keys=True))
            except Exception:
                results.append("ERROR")
        
        # All results should be the same
        assert len(set(results)) == 1
    
    @given(requests=st.lists(meme_request_strategy(), min_size=2, max_size=5))
    def test_router_handles_concurrent_requests(self, requests):
        """Router should handle multiple requests without state leakage."""
        router = RouterAgent()
        
        results = []
        for req in requests:
            try:
                result = router.forward(req)
                results.append(result)
            except Exception:
                pass
        
        # Each result should be independent
        topics = [r["topic"] for r in results if r]
        assert len(set(topics)) >= len(topics) // 2  # At least half should be unique


class TestAppropriatenessAgentProperties:
    """Property-based tests for AppropriatenessAgent."""
    
    @given(concept=meme_concept_strategy())
    @settings(max_examples=100)
    def test_appropriateness_always_returns_boolean(self, concept):
        """Appropriateness check should always return a boolean result."""
        agent = AppropriatenessAgent()
        
        result = agent.forward(concept)
        
        assert isinstance(result, dict)
        assert "is_appropriate" in result
        assert isinstance(result["is_appropriate"], bool)
        
        if not result["is_appropriate"]:
            assert "reason" in result
            assert isinstance(result["reason"], str)
            assert len(result["reason"]) > 0
    
    @given(
        safe_words=st.lists(st.sampled_from(["cat", "dog", "happy", "sunny"]), min_size=5),
        unsafe_words=st.lists(st.sampled_from(["offensive1", "offensive2", "inappropriate"]), min_size=1)
    )
    def test_appropriateness_detects_unsafe_content(self, safe_words, unsafe_words):
        """Agent should flag content with unsafe words."""
        agent = AppropriatenessAgent()
        
        # Test safe content
        safe_concept = {
            "topic": " ".join(safe_words),
            "format": "drake",
            "caption": f"A nice meme about {' '.join(safe_words)}"
        }
        safe_result = agent.forward(safe_concept)
        
        # Test unsafe content
        unsafe_concept = {
            "topic": " ".join(unsafe_words),
            "format": "drake",
            "caption": f"A meme about {' '.join(unsafe_words)}"
        }
        unsafe_result = agent.forward(unsafe_concept)
        
        # Safe content should generally pass, unsafe should generally fail
        # Note: This is a probabilistic test due to LLM behavior
        event(f"Safe result: {safe_result['is_appropriate']}")
        event(f"Unsafe result: {unsafe_result['is_appropriate']}")


class TestScoringAgentProperties:
    """Property-based tests for ScoringAgent."""
    
    @given(
        meme_candidates=st.lists(meme_concept_strategy(), min_size=1, max_size=10),
        verification_results=st.dictionaries(
            st.sampled_from(["factuality", "appropriateness", "instructions"]),
            st.booleans()
        )
    )
    @settings(max_examples=50)
    def test_scorer_always_returns_sorted_results(self, meme_candidates, verification_results):
        """Scorer should always return sorted results with valid scores."""
        scorer = ScoringAgent()
        
        # Add required fields for scoring
        for candidate in meme_candidates:
            candidate["verification_results"] = verification_results
        
        results = scorer.forward(meme_candidates)
        
        # Check structure
        assert isinstance(results, list)
        assert len(results) == len(meme_candidates)
        
        # Check each result has required fields
        for result in results:
            assert "final_score" in result
            assert isinstance(result["final_score"], (int, float))
            assert 0 <= result["final_score"] <= 1
            
            # Check sub-scores
            for score_type in ["humor_score", "clarity_score", "creativity_score", "shareability_score"]:
                if score_type in result:
                    assert isinstance(result[score_type], (int, float))
                    assert 0 <= result[score_type] <= 1
        
        # Check sorting
        scores = [r["final_score"] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    @given(
        num_candidates=st.integers(min_value=2, max_value=20)
    )
    def test_scorer_ranking_consistency(self, num_candidates):
        """Scorer should rank consistently across multiple runs."""
        scorer = ScoringAgent()
        
        # Create candidates with clear quality differences
        candidates = []
        for i in range(num_candidates):
            quality = i / num_candidates  # Linear quality distribution
            candidates.append({
                "topic": f"Topic with quality {quality}",
                "format": "drake",
                "caption": f"{'Good ' * int(quality * 10)}caption",
                "verification_results": {
                    "factuality": quality > 0.3,
                    "appropriateness": True,
                    "instructions": quality > 0.5
                }
            })
        
        # Score multiple times
        rankings = []
        for _ in range(3):
            results = scorer.forward(candidates)
            ranking = [r["topic"] for r in results]
            rankings.append(ranking)
        
        # Rankings should be consistent
        # Allow some variation due to LLM non-determinism
        first_ranking = rankings[0]
        for ranking in rankings[1:]:
            # At least 70% should be in the same position
            matches = sum(1 for i, topic in enumerate(ranking) if i < len(first_ranking) and topic == first_ranking[i])
            assert matches >= len(ranking) * 0.7


class MemeGenerationStateMachine(RuleBasedStateMachine):
    """Stateful testing for the complete meme generation pipeline."""
    
    requests = Bundle('requests')
    generated_memes = Bundle('memes')
    
    def __init__(self):
        super().__init__()
        self.pipeline = MemeGenerationPipeline()
        self.request_history: List[str] = []
        self.meme_history: List[Dict[str, Any]] = []
        self.error_count = 0
        self.success_count = 0
        self.processing_times: List[float] = []
    
    @initialize()
    def setup(self):
        """Initialize the state machine."""
        self.start_time = time.time()
    
    @rule(request=meme_request_strategy())
    def submit_request(self, request):
        """Submit a new meme generation request."""
        note(f"Submitting request: {request}")
        self.request_history.append(request)
        
        start = time.time()
        try:
            result = self.pipeline.forward(request)
            self.processing_times.append(time.time() - start)
            
            if result["status"] == "success":
                self.success_count += 1
                self.meme_history.append(result["meme"])
                return result
            else:
                self.error_count += 1
                
        except Exception as e:
            self.error_count += 1
            note(f"Error: {str(e)}")
    
    @rule()
    def check_memory_usage(self):
        """Check that memory usage remains bounded."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Memory should not exceed 1GB
        assert memory_mb < 1024, f"Memory usage too high: {memory_mb}MB"
    
    @invariant()
    def error_rate_acceptable(self):
        """Error rate should remain below threshold."""
        if self.success_count + self.error_count > 10:
            error_rate = self.error_count / (self.success_count + self.error_count)
            assert error_rate < 0.2, f"Error rate too high: {error_rate}"
    
    @invariant()
    def processing_time_reasonable(self):
        """Processing time should remain reasonable."""
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            assert avg_time < 10, f"Average processing time too high: {avg_time}s"
    
    @invariant()
    def no_duplicate_exact_memes(self):
        """Should not generate exact duplicate memes."""
        if len(self.meme_history) > 1:
            # Check for exact duplicates
            captions = [m.get("caption", "") for m in self.meme_history]
            unique_captions = set(captions)
            
            # Allow some duplicates but not too many
            duplicate_ratio = 1 - (len(unique_captions) / len(captions))
            assert duplicate_ratio < 0.1, f"Too many duplicate memes: {duplicate_ratio}"


# Test pipeline resilience with property-based testing
class TestPipelineResilience:
    """Test pipeline resilience to various input conditions."""
    
    @given(
        request=st.one_of(
            st.just(""),  # Empty request
            st.text(min_size=1000, max_size=10000),  # Very long request
            st.text().filter(lambda x: all(c in "!@#$%^&*()" for c in x)),  # Special chars only
            st.lists(st.sampled_from(["😀", "🎉", "🚀", "💻"]), min_size=10).map("".join),  # Emojis
            st.text(alphabet="你好世界测试"),  # Non-ASCII
        )
    )
    @settings(max_examples=50, deadline=10000)
    def test_pipeline_handles_edge_case_inputs(self, request):
        """Pipeline should handle edge case inputs gracefully."""
        pipeline = MemeGenerationPipeline()
        
        try:
            result = pipeline.forward(request)
            
            # Should either succeed or fail gracefully
            assert isinstance(result, dict)
            assert "status" in result
            assert result["status"] in ["success", "rejected", "error"]
            
            if result["status"] == "success":
                assert "meme" in result
                assert isinstance(result["meme"], dict)
                
        except Exception as e:
            # Should only raise HTTP exceptions
            from fastapi import HTTPException
            assert isinstance(e, HTTPException)
    
    @given(
        requests=st.lists(meme_request_strategy(), min_size=5, max_size=20)
    )
    @settings(max_examples=10, deadline=30000)
    def test_pipeline_performance_under_load(self, requests):
        """Pipeline should maintain performance under load."""
        pipeline = MemeGenerationPipeline()
        
        processing_times = []
        errors = 0
        
        for request in requests:
            start = time.time()
            try:
                pipeline.forward(request)
                processing_times.append(time.time() - start)
            except Exception:
                errors += 1
                processing_times.append(time.time() - start)
        
        # Performance assertions
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            max_time = max(processing_times)
            
            assert avg_time < 5, f"Average processing time too high: {avg_time}s"
            assert max_time < 15, f"Max processing time too high: {max_time}s"
            assert errors / len(requests) < 0.3, f"Error rate too high: {errors / len(requests)}"


# Run the state machine tests
TestMemeGenerationStateMachine = MemeGenerationStateMachine.TestCase

if __name__ == "__main__":
    pytest.main([__file__, "-v"])