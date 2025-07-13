"""
Code evaluation module for SWTBot generated code.
Provides utilities for evaluating the quality and correctness of generated code.
"""

import logging
import re
import ast
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import subprocess
import tempfile


class CodeEvaluator:
    """Evaluator for generated SWTBot code quality and correctness."""
    
    def __init__(self):
        """Initialize the code evaluator."""
        self.logger = logging.getLogger(__name__)
        
        # SWTBot-specific patterns
        self.swtbot_patterns = {
            "bot_creation": r"SWTBot\s+\w+\s*=\s*new\s+SWTBot\(\)",
            "widget_access": r"bot\.\w+\(",
            "assertions": r"assert\w+\(",
            "test_annotation": r"@Test",
            "imports": r"import\s+org\.eclipse\.swtbot",
        }
        
        # Common SWTBot methods
        self.swtbot_methods = [
            "button", "text", "label", "tree", "table", "menu",
            "shell", "view", "editor", "click", "type", "select",
            "doubleClick", "rightClick", "expand", "collapse"
        ]
        
        # Quality metrics
        self.quality_metrics = [
            "syntax_correctness",
            "swtbot_usage",
            "test_structure",
            "assertion_presence",
            "import_correctness",
            "method_coverage",
            "code_completeness"
        ]
    
    def evaluate_code(self, generated_code: str, prompt: str = "") -> Dict[str, Any]:
        """
        Evaluate generated SWTBot code comprehensively.
        
        Args:
            generated_code: The generated Java/SWTBot code
            prompt: Original prompt used for generation
            
        Returns:
            Evaluation results dictionary
        """
        results = {
            "prompt": prompt,
            "code": generated_code,
            "metrics": {},
            "issues": [],
            "suggestions": [],
            "overall_score": 0.0
        }
        
        # Run individual evaluations
        results["metrics"]["syntax"] = self._evaluate_syntax(generated_code)
        results["metrics"]["swtbot_usage"] = self._evaluate_swtbot_usage(generated_code)
        results["metrics"]["test_structure"] = self._evaluate_test_structure(generated_code)
        results["metrics"]["assertions"] = self._evaluate_assertions(generated_code)
        results["metrics"]["imports"] = self._evaluate_imports(generated_code)
        results["metrics"]["completeness"] = self._evaluate_completeness(generated_code)
        results["metrics"]["readability"] = self._evaluate_readability(generated_code)
        
        # Collect issues and suggestions
        self._collect_issues_and_suggestions(results)
        
        # Calculate overall score
        results["overall_score"] = self._calculate_overall_score(results["metrics"])
        
        return results
    
    def _evaluate_syntax(self, code: str) -> Dict[str, Any]:
        """Evaluate basic syntax correctness."""
        metrics = {
            "score": 0.0,
            "issues": [],
            "details": {}
        }
        
        # Check for basic Java syntax elements
        checks = {
            "has_class": bool(re.search(r'class\s+\w+', code)),
            "has_method": bool(re.search(r'public\s+void\s+\w+\s*\(', code)),
            "balanced_braces": self._check_balanced_braces(code),
            "proper_semicolons": self._check_semicolons(code),
            "valid_identifiers": self._check_identifiers(code)
        }
        
        metrics["details"] = checks
        
        # Calculate score
        score = sum(checks.values()) / len(checks)
        metrics["score"] = score
        
        # Add issues
        if not checks["has_class"]:
            metrics["issues"].append("Missing class declaration")
        if not checks["has_method"]:
            metrics["issues"].append("Missing method declaration")
        if not checks["balanced_braces"]:
            metrics["issues"].append("Unbalanced braces")
        
        return metrics
    
    def _evaluate_swtbot_usage(self, code: str) -> Dict[str, Any]:
        """Evaluate SWTBot-specific usage patterns."""
        metrics = {
            "score": 0.0,
            "issues": [],
            "details": {}
        }
        
        # Check SWTBot patterns
        pattern_matches = {}
        for pattern_name, pattern in self.swtbot_patterns.items():
            matches = re.findall(pattern, code, re.IGNORECASE)
            pattern_matches[pattern_name] = len(matches)
        
        metrics["details"]["pattern_matches"] = pattern_matches
        
        # Check for SWTBot methods
        method_usage = {}
        for method in self.swtbot_methods:
            count = len(re.findall(rf'\.{method}\(', code, re.IGNORECASE))
            if count > 0:
                method_usage[method] = count
        
        metrics["details"]["method_usage"] = method_usage
        
        # Calculate score
        score = 0.0
        if pattern_matches["bot_creation"] > 0:
            score += 0.3
        if pattern_matches["widget_access"] > 0:
            score += 0.3
        if len(method_usage) > 0:
            score += 0.4
        
        metrics["score"] = min(score, 1.0)
        
        # Add issues
        if pattern_matches["bot_creation"] == 0:
            metrics["issues"].append("No SWTBot instance creation found")
        if len(method_usage) == 0:
            metrics["issues"].append("No SWTBot methods used")
        
        return metrics
    
    def _evaluate_test_structure(self, code: str) -> Dict[str, Any]:
        """Evaluate test method structure."""
        metrics = {
            "score": 0.0,
            "issues": [],
            "details": {}
        }
        
        checks = {
            "has_test_annotation": bool(re.search(r'@Test', code)),
            "has_test_method": bool(re.search(r'public\s+void\s+test\w*\s*\(', code)),
            "proper_class_structure": bool(re.search(r'public\s+class\s+\w+Test', code)),
            "has_imports": bool(re.search(r'import\s+', code))
        }
        
        metrics["details"] = checks
        
        # Calculate score
        score = sum(checks.values()) / len(checks)
        metrics["score"] = score
        
        # Add issues
        if not checks["has_test_annotation"]:
            metrics["issues"].append("Missing @Test annotation")
        if not checks["has_test_method"]:
            metrics["issues"].append("Missing test method")
        
        return metrics
    
    def _evaluate_assertions(self, code: str) -> Dict[str, Any]:
        """Evaluate presence and usage of assertions."""
        metrics = {
            "score": 0.0,
            "issues": [],
            "details": {}
        }
        
        # Find assertion patterns
        assertion_patterns = [
            r'assert\w+\(',
            r'assertEquals\(',
            r'assertTrue\(',
            r'assertFalse\(',
            r'assertNotNull\(',
            r'assertNull\('
        ]
        
        assertion_counts = {}
        total_assertions = 0
        
        for pattern in assertion_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            count = len(matches)
            assertion_counts[pattern] = count
            total_assertions += count
        
        metrics["details"]["assertion_counts"] = assertion_counts
        metrics["details"]["total_assertions"] = total_assertions
        
        # Calculate score
        if total_assertions > 0:
            metrics["score"] = min(total_assertions / 3.0, 1.0)  # Expect at least 3 assertions for full score
        else:
            metrics["score"] = 0.0
            metrics["issues"].append("No assertions found")
        
        return metrics
    
    def _evaluate_imports(self, code: str) -> Dict[str, Any]:
        """Evaluate import statements."""
        metrics = {
            "score": 0.0,
            "issues": [],
            "details": {}
        }
        
        # Required imports for SWTBot
        required_imports = [
            "org.eclipse.swtbot",
            "org.junit",
        ]
        
        import_checks = {}
        for required in required_imports:
            found = bool(re.search(rf'import\s+{re.escape(required)}', code))
            import_checks[required] = found
        
        metrics["details"]["import_checks"] = import_checks
        
        # Calculate score
        score = sum(import_checks.values()) / len(import_checks)
        metrics["score"] = score
        
        # Add issues
        for required, found in import_checks.items():
            if not found:
                metrics["issues"].append(f"Missing import: {required}")
        
        return metrics
    
    def _evaluate_completeness(self, code: str) -> Dict[str, Any]:
        """Evaluate code completeness."""
        metrics = {
            "score": 0.0,
            "issues": [],
            "details": {}
        }
        
        # Check for completeness indicators
        completeness_checks = {
            "has_closing_brace": code.strip().endswith('}'),
            "no_incomplete_lines": not bool(re.search(r'\.\s*$', code)),
            "proper_method_closure": self._check_method_closure(code),
            "no_truncated_statements": not bool(re.search(r'[^;{}]\s*$', code.strip()))
        }
        
        metrics["details"] = completeness_checks
        
        # Calculate score
        score = sum(completeness_checks.values()) / len(completeness_checks)
        metrics["score"] = score
        
        # Add issues
        if not completeness_checks["has_closing_brace"]:
            metrics["issues"].append("Code appears incomplete (missing closing brace)")
        if completeness_checks["no_incomplete_lines"]:
            metrics["issues"].append("Code has incomplete lines")
        
        return metrics
    
    def _evaluate_readability(self, code: str) -> Dict[str, Any]:
        """Evaluate code readability."""
        metrics = {
            "score": 0.0,
            "issues": [],
            "details": {}
        }
        
        lines = code.split('\n')
        
        readability_checks = {
            "proper_indentation": self._check_indentation(lines),
            "reasonable_line_length": self._check_line_length(lines),
            "has_comments": bool(re.search(r'//', code)),
            "consistent_naming": self._check_naming_consistency(code)
        }
        
        metrics["details"] = readability_checks
        
        # Calculate score
        score = sum(readability_checks.values()) / len(readability_checks)
        metrics["score"] = score
        
        return metrics
    
    def _check_balanced_braces(self, code: str) -> bool:
        """Check if braces are balanced."""
        open_count = code.count('{')
        close_count = code.count('}')
        return open_count == close_count
    
    def _check_semicolons(self, code: str) -> bool:
        """Check for proper semicolon usage."""
        # Simple check - look for statements that should end with semicolons
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        statement_lines = [line for line in lines if not line.startswith('//') and 
                          not line.startswith('/*') and not line.endswith('{') and 
                          not line.endswith('}') and line]
        
        if not statement_lines:
            return True
        
        semicolon_lines = [line for line in statement_lines if line.endswith(';')]
        return len(semicolon_lines) / len(statement_lines) > 0.7  # Allow some flexibility
    
    def _check_identifiers(self, code: str) -> bool:
        """Check for valid Java identifiers."""
        # Look for obviously invalid identifiers
        invalid_patterns = [
            r'\b\d+\w+',  # Starting with digit
            r'\b[^a-zA-Z_$]\w*',  # Starting with invalid character
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, code):
                return False
        
        return True
    
    def _check_method_closure(self, code: str) -> bool:
        """Check if methods are properly closed."""
        # Simple heuristic: if we have method declarations, they should be closed
        method_starts = len(re.findall(r'public\s+\w+\s+\w+\s*\([^)]*\)\s*\{', code))
        if method_starts == 0:
            return True
        
        # Check if we have reasonable closing braces
        return self._check_balanced_braces(code)
    
    def _check_indentation(self, lines: List[str]) -> bool:
        """Check for consistent indentation."""
        indent_levels = []
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                indent_levels.append(leading_spaces)
        
        if not indent_levels:
            return True
        
        # Check if indentation is consistent (multiples of 2 or 4)
        consistent_2 = all(level % 2 == 0 for level in indent_levels)
        consistent_4 = all(level % 4 == 0 for level in indent_levels)
        
        return consistent_2 or consistent_4
    
    def _check_line_length(self, lines: List[str]) -> bool:
        """Check for reasonable line lengths."""
        long_lines = [line for line in lines if len(line) > 120]
        return len(long_lines) / max(len(lines), 1) < 0.2  # Less than 20% long lines
    
    def _check_naming_consistency(self, code: str) -> bool:
        """Check for consistent naming conventions."""
        # Look for camelCase method names
        method_names = re.findall(r'\.(\w+)\(', code)
        if not method_names:
            return True
        
        camel_case_count = sum(1 for name in method_names if re.match(r'^[a-z][a-zA-Z0-9]*$', name))
        return camel_case_count / len(method_names) > 0.7
    
    def _collect_issues_and_suggestions(self, results: Dict[str, Any]):
        """Collect all issues and generate suggestions."""
        all_issues = []
        suggestions = []
        
        for metric_name, metric_data in results["metrics"].items():
            all_issues.extend(metric_data.get("issues", []))
        
        results["issues"] = all_issues
        
        # Generate suggestions based on issues
        if "No SWTBot instance creation found" in all_issues:
            suggestions.append("Add 'SWTBot bot = new SWTBot();' to create a bot instance")
        
        if "No assertions found" in all_issues:
            suggestions.append("Add assertions like 'assertEquals(expected, actual)' to verify test results")
        
        if "Missing @Test annotation" in all_issues:
            suggestions.append("Add '@Test' annotation before the test method")
        
        results["suggestions"] = suggestions
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        weights = {
            "syntax": 0.2,
            "swtbot_usage": 0.25,
            "test_structure": 0.2,
            "assertions": 0.15,
            "imports": 0.1,
            "completeness": 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                weighted_score += metrics[metric_name]["score"] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def evaluate_multiple(self, code_samples: List[str], prompt: str = "") -> Dict[str, Any]:
        """
        Evaluate multiple code samples and provide comparative analysis.
        
        Args:
            code_samples: List of generated code samples
            prompt: Original prompt
            
        Returns:
            Comparative evaluation results
        """
        individual_results = []
        
        for i, code in enumerate(code_samples):
            result = self.evaluate_code(code, prompt)
            result["sample_id"] = i
            individual_results.append(result)
        
        # Calculate aggregate statistics
        scores = [result["overall_score"] for result in individual_results]
        
        aggregate_results = {
            "prompt": prompt,
            "num_samples": len(code_samples),
            "individual_results": individual_results,
            "aggregate_stats": {
                "mean_score": sum(scores) / len(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "best_sample_id": scores.index(max(scores)) if scores else 0
            }
        }
        
        return aggregate_results
