"""
Pydantic models for compliance checking and evaluation results.
These models ensure structured, validated output from the universal compliance agent.
"""

from enum import Enum
from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class ComplianceStatus(str, Enum):
    """Enumeration of possible compliance statuses."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non-compliant"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


class RuleEvaluation(BaseModel):
    """
    Represents the evaluation details of a compliance rule.
    """
    operation: str = Field(
        ..., 
        description="The type of operation performed (e.g., 'comparison', 'existence_check', 'cross_document')"
    )
    values_extracted: dict = Field(
        default_factory=dict,
        description="The actual values extracted from documents for this rule"
    )
    calculation: Optional[str] = Field(
        None,
        description="The mathematical calculation or logical evaluation performed"
    )
    result: bool = Field(
        ...,
        description="The boolean result of the evaluation (True/False)"
    )


class ComplianceFinding(BaseModel):
    """
    Represents a single compliance finding for a rule.
    """
    rule_id: str = Field(
        ..., 
        description="Unique identifier for the rule"
    )
    rule_checked: str = Field(
        ..., 
        description="The actual rule text that was evaluated"
    )
    status: ComplianceStatus = Field(
        ...,
        description="The compliance status result"
    )
    details: str = Field(
        ...,
        description="Detailed explanation of the evaluation including values and calculations"
    )
    involved_documents: List[str] = Field(
        default_factory=list,
        description="List of document filenames involved in this rule evaluation"
    )
    evaluation: Optional[RuleEvaluation] = Field(
        None,
        description="Detailed evaluation breakdown for this rule"
    )
    is_compliant: bool = Field(
        ...,
        description="Boolean flag indicating compliance (True) or non-compliance (False)"
    )
    reason: Optional[str] = Field(
        None,
        description="Brief reason for the compliance status (alternative to details)"
    )

    @model_validator(mode='after')
    def set_computed_fields(self):
        """Set computed fields based on other field values."""
        # Set is_compliant based on status
        if self.status == ComplianceStatus.COMPLIANT:
            self.is_compliant = True
        elif self.status == ComplianceStatus.NON_COMPLIANT:
            self.is_compliant = False
        elif self.status in [ComplianceStatus.NOT_APPLICABLE, ComplianceStatus.ERROR]:
            self.is_compliant = False  # Conservative approach for errors and non-applicable
        
        # Set reason from details if not provided
        if self.reason is None:
            self.reason = self.details or 'No details provided.'
        
        return self


class ComplianceResults(BaseModel):
    """
    Container for all compliance findings from a batch evaluation.
    """
    findings: List[ComplianceFinding] = Field(
        default_factory=list,
        description="List of all compliance findings"
    )
    total_rules_evaluated: int = Field(
        ...,
        description="Total number of rules that were evaluated"
    )
    compliant_count: int = Field(
        default=0,
        description="Number of rules that are compliant"
    )
    non_compliant_count: int = Field(
        default=0,
        description="Number of rules that are non-compliant"
    )
    error_count: int = Field(
        default=0,
        description="Number of rules that had errors during evaluation"
    )
    not_applicable_count: int = Field(
        default=0,
        description="Number of rules that were not applicable"
    )
    processing_metadata: dict = Field(
        default_factory=dict,
        description="Metadata about the processing (timing, tokens, etc.)"
    )

    @model_validator(mode='after')
    def calculate_statistics(self):
        """Calculate counts and statistics based on findings."""
        if not self.findings:
            self.compliant_count = 0
            self.non_compliant_count = 0
            self.error_count = 0
            self.not_applicable_count = 0
            self.total_rules_evaluated = 0
        else:
            self.compliant_count = len([f for f in self.findings if f.status == ComplianceStatus.COMPLIANT])
            self.non_compliant_count = len([f for f in self.findings if f.status == ComplianceStatus.NON_COMPLIANT])
            self.error_count = len([f for f in self.findings if f.status == ComplianceStatus.ERROR])
            self.not_applicable_count = len([f for f in self.findings if f.status == ComplianceStatus.NOT_APPLICABLE])
            self.total_rules_evaluated = len(self.findings)
        
        return self

    def get_summary(self) -> dict:
        """Get a summary of compliance results."""
        return {
            "total_rules": self.total_rules_evaluated,
            "compliant": self.compliant_count,
            "non_compliant": self.non_compliant_count,
            "errors": self.error_count,
            "not_applicable": self.not_applicable_count,
            "compliance_rate": (self.compliant_count / self.total_rules_evaluated * 100) if self.total_rules_evaluated > 0 else 0
        }


# Alternative simplified model for backward compatibility
class LegacyComplianceFinding(BaseModel):
    """
    Legacy format for compliance findings to maintain backward compatibility.
    """
    rule_id: str = Field(default="N/A")
    rule_text: Optional[str] = Field(None, alias="rule_checked")
    rule_checked: str = Field(default="N/A")
    is_compliant: bool = Field(default=False)
    reason: str = Field(default="No details provided.", alias="details")
    details: str = Field(default="No details provided.")
    involved_documents: List[str] = Field(default_factory=list)
    status: str = Field(default="error")  # String version for legacy compatibility

    class Config:
        allow_population_by_field_name = True 