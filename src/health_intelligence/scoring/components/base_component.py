"""
Base Score Component Interface

Abstract base class for health scoring components.
Each component calculates a score from 0-25 points based on specific health metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class ComponentScore:
    """
    Score result from a component calculation.
    
    Attributes:
        score: Calculated score (0-25 points)
        normalized_score: Normalized score (0-1 scale)
        confidence: Confidence in the calculation (0-1)
        details: Detailed breakdown of score calculation
        warnings: Any warnings or data quality issues
    """
    score: float
    normalized_score: float
    confidence: float
    details: Dict[str, Any]
    warnings: list = None
    
    def __post_init__(self):
        """Validate score ranges."""
        if self.warnings is None:
            self.warnings = []
        
        # Ensure score is in valid range
        if not 0 <= self.score <= 25:
            self.warnings.append(f"Score {self.score} outside valid range [0, 25], clamping")
            self.score = max(0, min(25, self.score))
        
        # Ensure normalized score is in valid range
        if not 0 <= self.normalized_score <= 1:
            self.warnings.append(
                f"Normalized score {self.normalized_score} outside valid range [0, 1], clamping"
            )
            self.normalized_score = max(0, min(1, self.normalized_score))
        
        # Ensure confidence is in valid range
        if not 0 <= self.confidence <= 1:
            self.warnings.append(f"Confidence {self.confidence} outside valid range [0, 1], clamping")
            self.confidence = max(0, min(1, self.confidence))


class BaseScoreComponent(ABC):
    """
    Abstract base class for health score components.
    
    Each component is responsible for:
    - Calculating a score from 0-25 points based on specific metrics
    - Providing confidence in the calculation
    - Returning detailed breakdown for transparency
    - Handling missing or incomplete data gracefully
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize component with configuration.
        
        Args:
            config: Component-specific configuration dictionary
        """
        self.config = config or {}
        self.component_name = self.__class__.__name__
    
    @abstractmethod
    def calculate_score(
        self,
        cow_id: str,
        data: pd.DataFrame,
        **kwargs
    ) -> ComponentScore:
        """
        Calculate component score.
        
        Args:
            cow_id: Animal identifier
            data: DataFrame with relevant data for this component
            **kwargs: Additional component-specific parameters
        
        Returns:
            ComponentScore object with score, confidence, and details
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> list:
        """
        Get list of required DataFrame columns for this component.
        
        Returns:
            List of column names required in the data DataFrame
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> tuple:
        """
        Validate that required data is present and sufficient.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Tuple of (is_valid: bool, warnings: list)
        """
        warnings = []
        
        if data is None or data.empty:
            return False, ["No data provided"]
        
        # Check for required columns
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            warnings.append(f"Missing required columns: {missing_cols}")
            return False, warnings
        
        # Check data completeness (at least some non-null values)
        for col in required_cols:
            non_null_count = data[col].notna().sum()
            if non_null_count == 0:
                warnings.append(f"Column '{col}' has no valid data")
        
        if warnings:
            return False, warnings
        
        return True, warnings
    
    def normalize_score(self, score: float, max_score: float = 25.0) -> float:
        """
        Normalize score to 0-1 range.
        
        Args:
            score: Raw score value
            max_score: Maximum possible score (default 25)
        
        Returns:
            Normalized score in [0, 1] range
        """
        return max(0, min(1, score / max_score))
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get component metadata and configuration.
        
        Returns:
            Dictionary with component information
        """
        return {
            'component_name': self.component_name,
            'max_score': 25,
            'required_columns': self.get_required_columns(),
            'config': self.config
        }
