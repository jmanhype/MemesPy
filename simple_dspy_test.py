"""Test DSPy with Anthropic Claude model."""

import dspy
import os

# Set up the language model
lm = dspy.LM(
    model="anthropic/claude-3-sonnet-20240229",
    api_key="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"
)

# Configure DSPy to use this language model
dspy.settings.configure(lm=lm)

# Define a simple signature for generating text
class SimpleCompletion(dspy.Signature):
    """Simple text completion signature."""
    
    topic: str = dspy.InputField(desc="Topic to generate text about")
    
    completion: str = dspy.OutputField(desc="Generated text about the topic")

# Create a DSPy module that uses the signature
simple_predictor = dspy.Predict(SimpleCompletion)

# Test the module
topic = "Python programming"
print(f"Generating text about '{topic}'...")
result = simple_predictor(topic=topic)
print(f"\nResult: {result.completion}") 