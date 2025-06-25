import json
import time
from typing import Any, Dict, List
from openai import AsyncOpenAI
import os
import logging
import instructor

from src.models import ComplianceResults, ComplianceFinding, ComplianceStatus, RuleEvaluation

logger = logging.getLogger(__name__)

class UniversalComplianceAgent:
    """
    Agent responsible for checking all compliance rules (both single-document and cross-document)
    against a batch of processed documents.
    """

    SYSTEM_PROMPT = """
You are an expert document compliance verification assistant. Your role is to systematically evaluate compliance rules against document data with precision and clarity.

## Core Responsibilities

### 1. Rule Analysis & Classification
For each rule, identify:
- **Operation Type**: Mathematical comparison, field existence, cross-document validation, pattern matching
- **Required Fields**: Which data fields need to be extracted from which documents
- **Evaluation Logic**: The specific comparison or check to be performed

### 2. Data Extraction & Validation
- Locate required values across all provided documents
- Handle different data types (numbers, strings, dates, booleans)
- Account for field variations and synonyms (e.g., "total", "amount", "sum")
- Consider document type context (invoice vs purchase order vs delivery note)

### 3. Precise Evaluation
- Perform exact mathematical comparisons with proper precision
- Handle edge cases (null values, missing fields, zero values)
- Apply logical operations consistently
- Document the step-by-step evaluation process

### 4. Status Determination Rules
- **COMPLIANT**: Rule evaluation returns TRUE, all required data present and valid
- **NON_COMPLIANT**: Rule evaluation returns FALSE, data present but fails check
- **ERROR**: Missing required data, invalid data types, or evaluation cannot be performed
- **NOT_APPLICABLE**: Rule doesn't apply to the available document types or context

## Evaluation Methodology

### Mathematical Comparisons
- Extract numerical values with proper type conversion
- Handle currency symbols, thousands separators, and decimal places
- Show exact values and calculations: "1,332.50 < 1,564.00 = TRUE"
- Account for floating-point precision in comparisons

### Field Existence Checks
- Verify field presence and non-empty values
- Distinguish between null, empty string, and zero values
- Consider field variations across document types

### Cross-Document Validation
- Match corresponding fields across multiple documents
- Handle document type relationships (invoice → purchase order → delivery note)
- Verify data consistency and logical relationships

### Pattern Matching & Business Rules
- Apply domain-specific validation logic
- Check format compliance (dates, IDs, reference numbers)
- Validate business rule consistency

## Quality Standards

### Details Field Requirements
- Start with document context: "In [filename]: [field]=[value]"
- Show mathematical work: "Evaluation: [value1] [operator] [value2] = [TRUE/FALSE]"
- Explain reasoning for complex rules
- Include all relevant document references

### Involved Documents
- List ALL documents that contributed data to the evaluation
- Use exact filenames as provided
- Include documents even if they contained no relevant data (for transparency)

### Rule ID Assignment
- Use sequential numbering: "1", "2", "3", etc.
- Match the order of rules as presented
- Maintain consistency across the evaluation batch

## Expert Tips for Accuracy

1. **Double-check numerical comparisons**: Ensure proper type conversion and precision
2. **Consider business context**: Invoice totals should match purchase order amounts
3. **Handle synonyms**: "total", "amount", "sum", "value" may refer to the same concept
4. **Document relationships**: Understand how different document types relate to each other
5. **Error classification**: Distinguish between missing data vs. failed validation

## Response Quality
Your structured output will be automatically validated. Focus on:
- Accurate rule interpretation and evaluation
- Clear, detailed explanations of your reasoning
- Precise numerical calculations
- Comprehensive document analysis
- Consistent status assignment based on evaluation results

Remember: You are the expert ensuring financial and regulatory compliance. Accuracy and thoroughness are paramount.
"""



    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        # Using temperature 0.0 for more deterministic compliance checks
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        # Create instructor-patched client (modern approach)
        self.client = instructor.patch(AsyncOpenAI(api_key=api_key))
        
        self.model_name = model_name
        self.temperature = temperature



    def _build_prompt_for_llm(
        self,
        all_documents_data: List[Dict[str, Any]], # Each dict: {"filename": str, "doc_type": str, "extracted_data": Dict}
        compliance_rules: str # A single string with all rules, numbered.
    ) -> str:
        # Use full data for maximum accuracy (no compression)
        prompt_parts = ["Here is the data extracted from all documents in this transaction set:"]
        if not all_documents_data:
            prompt_parts.append("No documents were provided or no data was successfully extracted.")
        else:
            for i, doc_data in enumerate(all_documents_data):
                prompt_parts.append(f"\n--- Document {i+1} ---")
                prompt_parts.append(f"Filename: {doc_data.get('filename', 'Unknown Filename')}")
                prompt_parts.append(f"Type: {doc_data.get('doc_type', 'Unknown Type')}")
                prompt_parts.append("Extracted Data:")
                # Use readable JSON formatting for maximum clarity
                prompt_parts.append(json.dumps(doc_data.get('extracted_data', {}), indent=2))
                prompt_parts.append("--- End of Document ---")

        prompt_parts.append("\nHere is the consolidated list of compliance rules you need to check against ALL the documents above:")
        prompt_parts.append(compliance_rules)
        prompt_parts.append("\nPlease evaluate EACH rule and provide your findings as a single JSON list, following the specified format in the system prompt.")
        return "\n".join(prompt_parts)

    async def check_all_compliance(
        self,
        all_documents_data: List[Dict[str, Any]],
        consolidated_rules: str
    ) -> List[Dict[str, Any]]:
        """
        Checks all compliance rules (single and cross-document) against the batch of documents.
        Now uses instructor for structured output with improved accuracy.

        Args:
            all_documents_data: A list of dictionaries, where each dictionary contains
                                'filename', 'doc_type', and 'extracted_data' for a document.
            consolidated_rules: A string containing all compliance rules, typically numbered.

        Returns:
            A list of dictionaries, where each dictionary represents a compliance finding for a rule.
        """
        start_time = time.time()
        
        if not consolidated_rules.strip():
            logger.info("No compliance rules provided. Skipping compliance check.")
            return []
        
        # Parse rule count for logging
        rule_count = len([line for line in consolidated_rules.strip().split('\n') if line.strip() and line.strip()[0].isdigit()])
        
        user_prompt = self._build_prompt_for_llm(all_documents_data, consolidated_rules)
        
        # Token estimation (rough)
        estimated_input_tokens = len(user_prompt.split()) + len(self.SYSTEM_PROMPT.split())
        logger.info(f"🚀 INSTRUCTOR COMPLIANCE: Processing {rule_count} rules against {len(all_documents_data)} documents")
        logger.info(f"📊 Structured output mode with validation: ~{estimated_input_tokens} estimated tokens")

        try:
            api_start = time.time()
            
            # Use instructor for structured output with validation
            compliance_results = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_model=ComplianceResults,
                max_retries=2  # Instructor will retry if validation fails
            )
            
            api_duration = time.time() - api_start
            
            if compliance_results is None:
                logger.error("Instructor returned None for compliance results.")
                return [{"rule_id": "N/A", "rule_checked": "Instructor Error", "status": "error", "details": "Instructor returned no results.", "involved_documents": []}]

            # Extract findings and convert to dictionaries for backward compatibility
            findings = []
            for finding in compliance_results.findings:
                # Convert Pydantic model to dictionary format expected by the workflow
                finding_dict = {
                    "rule_id": finding.rule_id,
                    "rule_checked": finding.rule_checked,
                    "status": finding.status.value,  # Convert enum to string
                    "details": finding.details,
                    "involved_documents": finding.involved_documents,
                    "is_compliant": finding.is_compliant,
                    "reason": finding.reason or finding.details
                }
                
                # Handle legacy field name mapping
                if "rule_text" not in finding_dict:
                    finding_dict["rule_text"] = finding.rule_checked
                    
                findings.append(finding_dict)

            # Performance logging with structured output metrics
            total_duration = time.time() - start_time
            
            # Get token usage if available (instructor preserves this)
            usage_info = getattr(compliance_results, '_raw_response', None)
            if usage_info and hasattr(usage_info, 'usage'):
                actual_input_tokens = usage_info.usage.prompt_tokens
                actual_output_tokens = usage_info.usage.completion_tokens
            else:
                actual_input_tokens = estimated_input_tokens
                actual_output_tokens = len(str(compliance_results).split())
            
            logger.info(f"✅ INSTRUCTOR COMPLIANCE COMPLETED:")
            logger.info(f"   📈 Performance: {total_duration:.2f}s total ({api_duration:.2f}s API)")
            logger.info(f"   🎯 Rules processed: {len(findings)}/{rule_count}")
            logger.info(f"   💰 Tokens: {actual_input_tokens} in, {actual_output_tokens} out")
            logger.info(f"   ⚡ Speed: {len(findings)/total_duration:.1f} rules/second")
            logger.info(f"   🔧 Validation: Automatic type checking and field validation applied")
            
            # Log summary statistics
            summary = compliance_results.get_summary()
            logger.info(f"   📊 Summary: {summary['compliant']}/{summary['total_rules']} compliant ({summary['compliance_rate']:.1f}%)")
            
            return findings

        except Exception as e:
            logger.error(f"Error during instructor-based compliance check: {e}", exc_info=True)
            
            # Create error finding using our Pydantic model for consistency
            try:
                error_result = ComplianceResults(
                    findings=[
                        ComplianceFinding(
                            rule_id="instructor_error",
                            rule_checked="Instructor Processing Error",
                            status=ComplianceStatus.ERROR,
                            details=f"An error occurred during structured compliance evaluation: {e}",
                            involved_documents=[doc.get('filename', 'unknown') for doc in all_documents_data],
                            is_compliant=False
                        )
                    ],
                    total_rules_evaluated=1
                )
                return [finding.model_dump() for finding in error_result.findings]
            except Exception as fallback_error:
                logger.error(f"Failed to create error result with Pydantic: {fallback_error}", exc_info=True)
                return [{"rule_id": "N/A", "rule_checked": "Critical Error", "status": "error", "details": f"Critical error: {e}", "involved_documents": [], "is_compliant": False}]

    async def check_all_compliance_structured(
        self,
        all_documents_data: List[Dict[str, Any]],
        consolidated_rules: str
    ) -> ComplianceResults:
        """
        Alternative method that returns structured Pydantic results directly.
        Useful for future enhancements that want to work with typed objects.

        Args:
            all_documents_data: A list of dictionaries, where each dictionary contains
                                'filename', 'doc_type', and 'extracted_data' for a document.
            consolidated_rules: A string containing all compliance rules, typically numbered.

        Returns:
            A ComplianceResults object with structured, validated findings.
        """
        start_time = time.time()
        
        if not consolidated_rules.strip():
            logger.info("No compliance rules provided. Returning empty results.")
            return ComplianceResults(
                findings=[],
                total_rules_evaluated=0,
                processing_metadata={"processing_time": 0.0}
            )
        
        user_prompt = self._build_prompt_for_llm(all_documents_data, consolidated_rules)
        
        try:
            api_start = time.time()
            
            compliance_results = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_model=ComplianceResults,
                max_retries=2
            )
            
            api_duration = time.time() - api_start
            total_duration = time.time() - start_time
            
            # Add processing metadata
            compliance_results.processing_metadata.update({
                "processing_time": total_duration,
                "api_time": api_duration,
                "model_used": self.model_name,
                "structured_output": True
            })
            
            logger.info(f"✅ Structured compliance check completed in {total_duration:.2f}s")
            return compliance_results
            
        except Exception as e:
            logger.error(f"Error during structured compliance check: {e}", exc_info=True)
            
            # Return error result with proper structure
            return ComplianceResults(
                findings=[
                    ComplianceFinding(
                        rule_id="error",
                        rule_checked="Processing Error",
                        status=ComplianceStatus.ERROR,
                        details=f"Failed to process compliance rules: {e}",
                        involved_documents=[doc.get('filename', 'unknown') for doc in all_documents_data],
                        is_compliant=False
                    )
                ],
                total_rules_evaluated=1,
                processing_metadata={
                    "processing_time": time.time() - start_time,
                    "error": str(e),
                    "structured_output": True
                }
            )


