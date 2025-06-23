"""Contract testing between services in MemesPy system.

This module implements contract tests to ensure services maintain
their API contracts and can communicate correctly.
"""

import pytest
import json
import jsonschema
from typing import Dict, Any, List, Optional, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import pact
from pact import Consumer, Provider, Like, EachLike, SomethingLike, Format, Term

from src.dspy_meme_gen.agents.router import RouterAgent
from src.dspy_meme_gen.agents.appropriateness import AppropriatenessAgent
from src.dspy_meme_gen.agents.factuality import FactualityAgent
from src.dspy_meme_gen.agents.scorer import ScoringAgent
from src.dspy_meme_gen.pipeline import MemeGenerationPipeline


# Contract schemas for each service
SCHEMAS = {
    "router_input": {
        "type": "object",
        "properties": {
            "request": {"type": "string", "minLength": 1}
        },
        "required": ["request"]
    },
    "router_output": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "format": {"type": "string"},
            "verification_needs": {
                "type": "object",
                "properties": {
                    "factuality": {"type": "boolean"},
                    "appropriateness": {"type": "boolean"},
                    "instructions": {"type": "boolean"}
                }
            },
            "constraints": {"type": "object"},
            "generation_approach": {"type": "string"}
        },
        "required": ["topic", "format", "verification_needs", "constraints", "generation_approach"]
    },
    "appropriateness_input": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "format": {"type": "string"},
            "caption": {"type": "string"}
        },
        "required": ["topic"]
    },
    "appropriateness_output": {
        "type": "object",
        "properties": {
            "is_appropriate": {"type": "boolean"},
            "reason": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["is_appropriate"]
    },
    "scorer_input": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "caption": {"type": "string"},
                "format": {"type": "string"},
                "topic": {"type": "string"}
            },
            "required": ["caption"]
        },
        "minItems": 1
    },
    "scorer_output": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "final_score": {"type": "number", "minimum": 0, "maximum": 1},
                "humor_score": {"type": "number", "minimum": 0, "maximum": 1},
                "clarity_score": {"type": "number", "minimum": 0, "maximum": 1},
                "creativity_score": {"type": "number", "minimum": 0, "maximum": 1},
                "shareability_score": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["final_score"]
        }
    },
    "pipeline_input": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "minLength": 1},
            "format": {"type": "string"},
            "style": {"type": "string"},
            "constraints": {"type": "object"}
        },
        "required": ["prompt"]
    },
    "pipeline_output": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "rejected", "error"]},
            "meme": {
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                    "image_url": {"type": "string"},
                    "format": {"type": "string"},
                    "topic": {"type": "string"}
                }
            },
            "alternatives": {"type": "array"},
            "reason": {"type": "string"}
        },
        "required": ["status"]
    }
}


class ServiceContract(Protocol):
    """Protocol for service contracts."""
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data against contract."""
        ...
    
    def validate_output(self, data: Any) -> bool:
        """Validate output data against contract."""
        ...
    
    def get_sample_input(self) -> Any:
        """Get sample valid input."""
        ...
    
    def get_sample_output(self) -> Any:
        """Get sample valid output."""
        ...


@dataclass
class ContractViolation:
    """Represents a contract violation."""
    
    service: str
    direction: str  # "input" or "output"
    data: Any
    errors: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class ServiceContractValidator:
    """Validates service contracts."""
    
    def __init__(self, service_name: str, input_schema: Dict[str, Any], output_schema: Dict[str, Any]):
        self.service_name = service_name
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.violations: List[ContractViolation] = []
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data against contract."""
        try:
            jsonschema.validate(data, self.input_schema)
            return True
        except jsonschema.ValidationError as e:
            self.violations.append(ContractViolation(
                service=self.service_name,
                direction="input",
                data=data,
                errors=[str(e)]
            ))
            return False
    
    def validate_output(self, data: Any) -> bool:
        """Validate output data against contract."""
        try:
            jsonschema.validate(data, self.output_schema)
            return True
        except jsonschema.ValidationError as e:
            self.violations.append(ContractViolation(
                service=self.service_name,
                direction="output",
                data=data,
                errors=[str(e)]
            ))
            return False
    
    def get_violations_summary(self) -> Dict[str, Any]:
        """Get summary of contract violations."""
        return {
            "service": self.service_name,
            "total_violations": len(self.violations),
            "input_violations": len([v for v in self.violations if v.direction == "input"]),
            "output_violations": len([v for v in self.violations if v.direction == "output"]),
            "violations": [
                {
                    "direction": v.direction,
                    "errors": v.errors,
                    "timestamp": v.timestamp.isoformat()
                }
                for v in self.violations[-10:]  # Last 10 violations
            ]
        }


class TestServiceContracts:
    """Test service contracts."""
    
    @pytest.fixture
    def validators(self):
        """Create validators for all services."""
        return {
            "router": ServiceContractValidator(
                "router",
                SCHEMAS["router_input"],
                SCHEMAS["router_output"]
            ),
            "appropriateness": ServiceContractValidator(
                "appropriateness",
                SCHEMAS["appropriateness_input"],
                SCHEMAS["appropriateness_output"]
            ),
            "scorer": ServiceContractValidator(
                "scorer",
                SCHEMAS["scorer_input"],
                SCHEMAS["scorer_output"]
            ),
            "pipeline": ServiceContractValidator(
                "pipeline",
                SCHEMAS["pipeline_input"],
                SCHEMAS["pipeline_output"]
            )
        }
    
    def test_router_contract(self, validators):
        """Test router service contract."""
        validator = validators["router"]
        router = RouterAgent()
        
        # Test valid inputs
        valid_inputs = [
            {"request": "Create a meme about Python"},
            {"request": "Make a funny meme using drake format"},
            {"request": "Generate trending meme about AI"}
        ]
        
        for input_data in valid_inputs:
            assert validator.validate_input(input_data), f"Valid input rejected: {input_data}"
            
            # Test service execution
            result = router.forward(input_data["request"])
            assert validator.validate_output(result), f"Service output violates contract: {result}"
        
        # Test invalid inputs
        invalid_inputs = [
            {},  # Missing required field
            {"request": ""},  # Empty string
            {"request": 123},  # Wrong type
            {"wrong_field": "value"}  # Wrong field name
        ]
        
        for input_data in invalid_inputs:
            assert not validator.validate_input(input_data), f"Invalid input accepted: {input_data}"
    
    def test_appropriateness_contract(self, validators):
        """Test appropriateness service contract."""
        validator = validators["appropriateness"]
        agent = AppropriatenessAgent()
        
        # Test valid inputs
        valid_inputs = [
            {"topic": "programming", "format": "drake", "caption": "Test caption"},
            {"topic": "cats"},  # Minimal required fields
            {"topic": "work", "format": "distracted", "caption": "Office humor", "extra": "ignored"}
        ]
        
        for input_data in valid_inputs:
            assert validator.validate_input(input_data), f"Valid input rejected: {input_data}"
            
            result = agent.forward(input_data)
            assert validator.validate_output(result), f"Service output violates contract: {result}"
            
            # Verify output structure
            assert isinstance(result["is_appropriate"], bool)
            if "confidence" in result:
                assert 0 <= result["confidence"] <= 1
    
    def test_scorer_contract(self, validators):
        """Test scorer service contract."""
        validator = validators["scorer"]
        scorer = ScoringAgent()
        
        # Test valid inputs
        valid_inputs = [
            [{"caption": "Test meme", "format": "drake", "topic": "coding"}],
            [
                {"caption": "Meme 1", "format": "drake"},
                {"caption": "Meme 2", "format": "distracted"}
            ],
            [{"caption": "Minimal meme"}]  # Only required fields
        ]
        
        for input_data in valid_inputs:
            assert validator.validate_input(input_data), f"Valid input rejected: {input_data}"
            
            # Add required verification results for scorer
            for item in input_data:
                item["verification_results"] = {"appropriateness": True}
            
            result = scorer.forward(input_data)
            assert validator.validate_output(result), f"Service output violates contract: {result}"
            
            # Verify output structure
            assert len(result) == len(input_data)
            for item in result:
                assert 0 <= item["final_score"] <= 1
    
    def test_pipeline_contract(self, validators):
        """Test pipeline service contract."""
        validator = validators["pipeline"]
        pipeline = MemeGenerationPipeline()
        
        # Test valid inputs
        valid_inputs = [
            {"prompt": "Create a meme about Python"},
            {"prompt": "Make meme", "format": "drake", "style": "minimal"},
            {"prompt": "Generate meme", "constraints": {"appropriate": True}}
        ]
        
        for input_data in valid_inputs:
            assert validator.validate_input(input_data), f"Valid input rejected: {input_data}"
            
            result = pipeline.forward(input_data["prompt"])
            assert validator.validate_output(result), f"Service output violates contract: {result}"
            
            # Verify output structure
            assert result["status"] in ["success", "rejected", "error"]
            if result["status"] == "success":
                assert "meme" in result
    
    def test_service_composition_contracts(self, validators):
        """Test contracts when services are composed."""
        
        # Router -> Appropriateness flow
        router = RouterAgent()
        appropriateness = AppropriatenessAgent()
        
        # Router output should be compatible with downstream services
        router_result = router.forward("Create appropriate meme")
        assert validators["router"].validate_output(router_result)
        
        # Extract data for appropriateness check
        appropriateness_input = {
            "topic": router_result["topic"],
            "format": router_result["format"],
            "caption": "Test caption from router"
        }
        
        assert validators["appropriateness"].validate_input(appropriateness_input)
        appropriateness_result = appropriateness.forward(appropriateness_input)
        assert validators["appropriateness"].validate_output(appropriateness_result)
    
    def test_contract_evolution_compatibility(self):
        """Test backward compatibility when contracts evolve."""
        
        # Old contract (v1)
        old_schema = {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "format": {"type": "string"}
            },
            "required": ["topic"]
        }
        
        # New contract (v2) - adds optional field
        new_schema = {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "format": {"type": "string"},
                "version": {"type": "string", "default": "v2"}  # New optional field
            },
            "required": ["topic"]
        }
        
        # Old data should still be valid in new schema
        old_data = {"topic": "cats", "format": "drake"}
        jsonschema.validate(old_data, new_schema)  # Should not raise
        
        # New data with version field
        new_data = {"topic": "dogs", "format": "distracted", "version": "v2"}
        jsonschema.validate(new_data, new_schema)  # Should not raise


class TestPactContracts:
    """Consumer-driven contract tests using Pact."""
    
    @pytest.fixture
    def pact(self):
        """Set up Pact consumer."""
        return Consumer('MemeGenerationPipeline').has_pact_with(
            Provider('RouterAgent'),
            pact_dir='./pacts'
        )
    
    def test_router_consumer_contract(self, pact):
        """Test consumer contract for router service."""
        
        # Define expected interaction
        expected_response = {
            "topic": Like("programming"),
            "format": Like("drake"),
            "verification_needs": Like({
                "factuality": False,
                "appropriateness": True,
                "instructions": False
            }),
            "constraints": Like({}),
            "generation_approach": Term(r"standard|creative|trending", "standard")
        }
        
        # Set up the interaction
        (pact
         .given('a meme generation request')
         .upon_receiving('a request to analyze and route')
         .with_request('POST', '/route', body={"request": "Create a meme about programming"})
         .will_respond_with(200, body=expected_response))
        
        with pact:
            # Consumer test code
            router = RouterAgent()
            result = router.forward("Create a meme about programming")
            
            # Verify the result matches our expectations
            assert "topic" in result
            assert "format" in result
            assert "verification_needs" in result
            assert result["generation_approach"] in ["standard", "creative", "trending"]
    
    def test_scorer_consumer_contract(self, pact):
        """Test consumer contract for scorer service."""
        
        expected_response = EachLike({
            "final_score": Like(0.85),
            "humor_score": Like(0.9),
            "clarity_score": Like(0.8),
            "creativity_score": Like(0.85),
            "shareability_score": Like(0.87),
            "feedback": Like("Good meme with strong humor")
        })
        
        (pact
         .given('meme candidates to score')
         .upon_receiving('a request to score memes')
         .with_request('POST', '/score', body=[{
             "caption": "Test meme",
             "format": "drake",
             "verification_results": {"appropriateness": True}
         }])
         .will_respond_with(200, body=expected_response))
        
        with pact:
            scorer = ScoringAgent()
            result = scorer.forward([{
                "caption": "Test meme",
                "format": "drake",
                "verification_results": {"appropriateness": True}
            }])
            
            assert len(result) > 0
            assert all(0 <= item["final_score"] <= 1 for item in result)


class ContractTestRunner:
    """Runs contract tests and generates reports."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {
            "services": {},
            "compatibility": {},
            "violations": []
        }
    
    def run_all_contract_tests(self):
        """Run all contract tests."""
        
        # Test individual service contracts
        services = {
            "router": (RouterAgent(), SCHEMAS["router_input"], SCHEMAS["router_output"]),
            "appropriateness": (AppropriatenessAgent(), SCHEMAS["appropriateness_input"], SCHEMAS["appropriateness_output"]),
            "scorer": (ScoringAgent(), SCHEMAS["scorer_input"], SCHEMAS["scorer_output"])
        }
        
        for service_name, (service, input_schema, output_schema) in services.items():
            validator = ServiceContractValidator(service_name, input_schema, output_schema)
            
            # Test with various inputs
            test_results = self._test_service(service, validator)
            self.results["services"][service_name] = test_results
        
        # Test service compatibility
        self._test_service_compatibility()
        
        return self.results
    
    def _test_service(self, service: Any, validator: ServiceContractValidator) -> Dict[str, Any]:
        """Test a single service."""
        
        test_cases = self._generate_test_cases(validator.input_schema)
        results = {
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "contract_violations": []
        }
        
        for test_input in test_cases:
            try:
                # Validate input
                if not validator.validate_input(test_input):
                    results["failed"] += 1
                    continue
                
                # Execute service
                if isinstance(test_input, dict) and "request" in test_input:
                    output = service.forward(test_input["request"])
                else:
                    output = service.forward(test_input)
                
                # Validate output
                if validator.validate_output(output):
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["contract_violations"].append(validator.violations[-1])
                    
            except Exception as e:
                results["failed"] += 1
                results["contract_violations"].append({
                    "error": str(e),
                    "input": test_input
                })
        
        return results
    
    def _generate_test_cases(self, schema: Dict[str, Any]) -> List[Any]:
        """Generate test cases based on schema."""
        
        test_cases = []
        
        # Basic valid case
        if schema["type"] == "object":
            valid_case = {}
            for prop, prop_schema in schema.get("properties", {}).items():
                if prop in schema.get("required", []):
                    valid_case[prop] = self._generate_value(prop_schema)
            test_cases.append(valid_case)
            
        elif schema["type"] == "array":
            # Array with one item
            item_schema = schema.get("items", {"type": "object"})
            test_cases.append([self._generate_value(item_schema)])
            
            # Array with multiple items
            test_cases.append([
                self._generate_value(item_schema) for _ in range(3)
            ])
        
        return test_cases
    
    def _generate_value(self, schema: Dict[str, Any]) -> Any:
        """Generate a value based on schema."""
        
        if schema["type"] == "string":
            return "test_string"
        elif schema["type"] == "number":
            return 0.5
        elif schema["type"] == "integer":
            return 42
        elif schema["type"] == "boolean":
            return True
        elif schema["type"] == "object":
            obj = {}
            for prop, prop_schema in schema.get("properties", {}).items():
                obj[prop] = self._generate_value(prop_schema)
            return obj
        elif schema["type"] == "array":
            return [self._generate_value(schema.get("items", {"type": "string"}))]
        
        return None
    
    def _test_service_compatibility(self):
        """Test compatibility between services."""
        
        # Test Router -> Appropriateness compatibility
        router = RouterAgent()
        appropriateness = AppropriatenessAgent()
        
        router_output = router.forward("Test request")
        
        # Check if router output can be used as appropriateness input
        appropriateness_input = {
            "topic": router_output["topic"],
            "format": router_output["format"]
        }
        
        try:
            appropriateness_output = appropriateness.forward(appropriateness_input)
            self.results["compatibility"]["router_to_appropriateness"] = "compatible"
        except Exception as e:
            self.results["compatibility"]["router_to_appropriateness"] = f"incompatible: {str(e)}"
    
    def generate_report(self, output_file: str = "contract_test_report.json"):
        """Generate contract test report."""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_services": len(self.results["services"]),
                "services_passed": sum(
                    1 for s in self.results["services"].values()
                    if s["passed"] > s["failed"]
                ),
                "compatibility_issues": sum(
                    1 for c in self.results["compatibility"].values()
                    if "incompatible" in c
                )
            },
            "details": self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


# Contract monitoring in production
class ContractMonitor:
    """Monitor contract compliance in production."""
    
    def __init__(self):
        self.violations: List[ContractViolation] = []
        self.validators = {
            "router": ServiceContractValidator("router", SCHEMAS["router_input"], SCHEMAS["router_output"]),
            "appropriateness": ServiceContractValidator("appropriateness", SCHEMAS["appropriateness_input"], SCHEMAS["appropriateness_output"]),
            "scorer": ServiceContractValidator("scorer", SCHEMAS["scorer_input"], SCHEMAS["scorer_output"])
        }
    
    def intercept_service_call(self, service_name: str, input_data: Any, output_data: Any):
        """Intercept and validate service calls."""
        
        if service_name in self.validators:
            validator = self.validators[service_name]
            
            # Validate input
            if not validator.validate_input(input_data):
                self.violations.extend(validator.violations)
            
            # Validate output
            if not validator.validate_output(output_data):
                self.violations.extend(validator.violations)
    
    def get_violation_metrics(self) -> Dict[str, Any]:
        """Get contract violation metrics."""
        
        metrics = {
            "total_violations": len(self.violations),
            "violations_by_service": {},
            "violations_by_direction": {
                "input": 0,
                "output": 0
            },
            "recent_violations": []
        }
        
        for violation in self.violations:
            # Count by service
            if violation.service not in metrics["violations_by_service"]:
                metrics["violations_by_service"][violation.service] = 0
            metrics["violations_by_service"][violation.service] += 1
            
            # Count by direction
            metrics["violations_by_direction"][violation.direction] += 1
        
        # Add recent violations
        metrics["recent_violations"] = [
            {
                "service": v.service,
                "direction": v.direction,
                "errors": v.errors,
                "timestamp": v.timestamp.isoformat()
            }
            for v in self.violations[-10:]
        ]
        
        return metrics


if __name__ == "__main__":
    # Run contract tests
    runner = ContractTestRunner()
    results = runner.run_all_contract_tests()
    report = runner.generate_report()
    
    print(f"Contract Test Results:")
    print(f"  Services tested: {report['summary']['total_services']}")
    print(f"  Services passed: {report['summary']['services_passed']}")
    print(f"  Compatibility issues: {report['summary']['compatibility_issues']}")
    print(f"\nDetailed report saved to contract_test_report.json")