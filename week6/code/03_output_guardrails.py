"""
Module 6.2: Output Validation and Content Filtering
====================================================
Multi-stage output validation pipeline:
- Format validation
- Content moderation
- Factuality checking
- Bias detection
- PII redaction
- Hallucination detection
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# OUTPUT VALIDATION TYPES
# ============================================================

class OutputValidationResult(Enum):
    PASS = "pass"
    MODIFIED = "modified"
    BLOCKED = "blocked"


@dataclass
class OutputValidation:
    """Result of output validation."""
    result: OutputValidationResult
    original_output: str
    validated_output: str
    modifications: List[str]
    warnings: List[str]
    confidence_score: float


# ============================================================
# OUTPUT VALIDATOR
# ============================================================

class OutputValidator:
    """Multi-stage output validation pipeline."""

    def __init__(self):
        # PII patterns
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            "ssn": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        }

        # Harmful content patterns
        self.harmful_patterns = [
            r'\b(kill|murder|assassinate)\s+(yourself|himself|herself|themselves)\b',
            r'\b(instructions|steps|guide)\s+(to|for)\s+(make|build|create)\s+(bomb|weapon)\b',
        ]

        # Bias indicators
        self.bias_patterns = [
            (r'\b(all|every)\s+(men|women|blacks|whites|asians)\s+(are|always)\b', "absolute_generalization"),
            (r'\b(typical|obviously)\s+(male|female|man|woman)\b', "gender_stereotype"),
        ]

    def validate(self, output: str, context: Dict[str, Any] = None) -> OutputValidation:
        """Run full output validation pipeline."""

        context = context or {}
        modifications = []
        warnings = []
        validated_output = output

        # Stage 1: Format validation
        format_result = self._validate_format(validated_output)
        if format_result["modified"]:
            validated_output = format_result["output"]
            modifications.append("format_adjusted")

        # Stage 2: Content moderation
        content_result = self._moderate_content(validated_output)
        if content_result["blocked"]:
            return OutputValidation(
                result=OutputValidationResult.BLOCKED,
                original_output=output,
                validated_output="",
                modifications=["content_blocked"],
                warnings=content_result["reasons"],
                confidence_score=0.0
            )
        if content_result["modified"]:
            validated_output = content_result["output"]
            modifications.append("content_moderated")
            warnings.extend(content_result["reasons"])

        # Stage 3: Factuality check
        if context.get("source_documents"):
            fact_result = self._check_factuality(validated_output, context["source_documents"])
            warnings.extend(fact_result["warnings"])

        # Stage 4: Bias detection
        bias_result = self._detect_bias(validated_output)
        if bias_result["detected"]:
            warnings.extend(bias_result["types"])
            validated_output += "\n\n[Note: This response may contain generalizations.]"
            modifications.append("bias_disclaimer_added")

        # Stage 5: PII redaction
        pii_result = self._redact_pii(validated_output)
        if pii_result["redacted"]:
            validated_output = pii_result["output"]
            modifications.append(f"pii_redacted:{pii_result['count']}")

        # Stage 6: Confidence calculation
        confidence = self._calculate_confidence(validated_output, context)

        result_type = OutputValidationResult.MODIFIED if modifications else OutputValidationResult.PASS

        return OutputValidation(
            result=result_type,
            original_output=output,
            validated_output=validated_output,
            modifications=modifications,
            warnings=warnings,
            confidence_score=confidence
        )

    def _validate_format(self, output: str) -> Dict[str, Any]:
        """Validate output format."""
        modified = False
        result = output

        cleaned = ' '.join(output.split())
        if cleaned != output:
            result = cleaned
            modified = True

        max_length = 10000
        if len(result) > max_length:
            result = result[:max_length] + "... [truncated]"
            modified = True

        return {"output": result, "modified": modified}

    def _moderate_content(self, output: str) -> Dict[str, Any]:
        """Check for harmful content."""
        output_lower = output.lower()
        reasons = []

        for pattern in self.harmful_patterns:
            if re.search(pattern, output_lower, re.IGNORECASE):
                return {
                    "blocked": True,
                    "modified": False,
                    "output": output,
                    "reasons": ["harmful_content_detected"]
                }

        return {
            "blocked": False,
            "modified": len(reasons) > 0,
            "output": output,
            "reasons": reasons
        }

    def _check_factuality(self, output: str, sources: List[str]) -> Dict[str, Any]:
        """Check if claims are supported by sources."""
        warnings = []

        claim_patterns = [
            r'(?:research shows|studies indicate|according to)\s+([^.]+)',
            r'(?:definitely|certainly|always|never)\s+([^.]+)',
        ]

        for pattern in claim_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                claim_words = set(match.lower().split())
                source_text = ' '.join(sources).lower()
                overlap = len(claim_words & set(source_text.split()))

                if overlap < len(claim_words) * 0.3:
                    warnings.append(f"unverified_claim:{match[:30]}...")

        return {"warnings": warnings}

    def _detect_bias(self, output: str) -> Dict[str, Any]:
        """Detect potentially biased content."""
        detected = []

        for pattern, bias_type in self.bias_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                detected.append(f"bias:{bias_type}")

        return {"detected": len(detected) > 0, "types": detected}

    def _redact_pii(self, output: str) -> Dict[str, Any]:
        """Redact personally identifiable information."""
        redacted_output = output
        total_count = 0

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, redacted_output)
            count = len(matches)
            if count > 0:
                redacted_output = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]', redacted_output)
                total_count += count

        return {
            "output": redacted_output,
            "redacted": total_count > 0,
            "count": total_count
        }

    def _calculate_confidence(self, output: str, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for the output."""
        confidence = 0.8

        hedging_patterns = [
            r'\b(might|maybe|perhaps|possibly|could be)\b',
            r'\b(I think|I believe|in my opinion)\b',
        ]

        hedging_count = 0
        for pattern in hedging_patterns:
            hedging_count += len(re.findall(pattern, output, re.IGNORECASE))

        confidence -= hedging_count * 0.05

        if not context or "source_documents" not in context:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))


# ============================================================
# HALLUCINATION DETECTOR
# ============================================================

class HallucinationDetector:
    """Detect potential hallucinations in LLM output."""

    def __init__(self, llm=None):
        self.llm = llm

        self.hallucination_signals = [
            r'(?:in\s+)?(\d{4}),?\s+(?:the|a)\s+(?:study|research|paper)',
            r'(?:Dr\.|Professor)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:from|at)',
            r'according to (?:the|a) \d{4} report',
        ]

    def detect(self, output: str, context: Dict[str, Any] = None) -> Tuple[float, List[str]]:
        """
        Detect potential hallucinations.

        Returns:
            - hallucination_probability: float (0-1)
            - indicators: list of hallucination indicators
        """
        indicators = []
        probability = 0.0

        # Check for hallucination signals
        for pattern in self.hallucination_signals:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                indicators.append(f"suspicious_citation:{matches[0]}")
                probability += 0.2

        # Check for invented specifics
        specific_patterns = [
            (r'\b(\d{1,2}\.\d{1,2}%)\b', "precise_percentage"),
            (r'\b(\$\d{1,3}(?:,\d{3})*)\s+(?:million|billion)\b', "precise_money"),
            (r'(?:exactly|precisely)\s+(\d+)\s+', "exact_number"),
        ]

        for pattern, indicator_type in specific_patterns:
            if re.search(pattern, output):
                if not context or "source_documents" not in context:
                    indicators.append(f"unverified_{indicator_type}")
                    probability += 0.15

        # Check against sources if provided
        if context and "source_documents" in context:
            source_check = self._check_against_sources(output, context["source_documents"])
            if source_check["unsupported_claims"]:
                indicators.extend(source_check["unsupported_claims"])
                probability += 0.3

        return min(1.0, probability), indicators

    def _check_against_sources(self, output: str, sources: List[str]) -> Dict[str, Any]:
        """Check if output claims are supported by sources."""
        unsupported = []
        source_text = ' '.join(sources).lower()

        # Extract numerical claims
        claim_patterns = [
            r'(?:is|are|was|were)\s+(\d+(?:\.\d+)?%?)',
            r'(?:founded|established)\s+in\s+(\d{4})',
        ]

        for pattern in claim_patterns:
            matches = re.findall(pattern, output)
            for match in matches:
                if match.lower() not in source_text:
                    unsupported.append(f"unsupported:{match}")

        return {"unsupported_claims": unsupported}


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate output validation."""
    print("=" * 60)
    print("OUTPUT VALIDATION DEMO")
    print("=" * 60)

    validator = OutputValidator()
    hallucination_detector = HallucinationDetector()

    test_outputs = [
        # Clean output
        "Paris is the capital of France. It's known for the Eiffel Tower.",

        # Output with PII
        "Contact John at john.smith@email.com or call 555-123-4567 for details.",

        # Output with bias
        "All programmers are introverted. Men are typically better at math.",

        # Output with potential hallucination
        "According to a 2023 study by Dr. James Smith at Harvard, exactly 73.2% of users prefer AI assistants.",

        # Harmful output
        "Here are instructions to hurt yourself: [content would be blocked]",
    ]

    for output in test_outputs:
        print(f"\n{'─' * 60}")
        print(f"Input: {output[:60]}...")

        # Run validation
        result = validator.validate(output)

        print(f"\nValidation Result: {result.result.value.upper()}")
        print(f"Confidence: {result.confidence_score:.0%}")

        if result.modifications:
            print(f"Modifications: {result.modifications}")

        if result.warnings:
            print(f"Warnings: {result.warnings}")

        if result.result != OutputValidationResult.BLOCKED:
            print(f"Output: {result.validated_output[:80]}...")

            # Check for hallucinations
            prob, indicators = hallucination_detector.detect(result.validated_output)
            if prob > 0:
                print(f"Hallucination probability: {prob:.0%}")
                print(f"Indicators: {indicators}")


if __name__ == "__main__":
    demo()
