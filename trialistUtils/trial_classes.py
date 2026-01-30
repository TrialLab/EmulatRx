from typing import List, Optional

class TrialEligibilityCriterion:
    def __init__(self,
                 text: Optional[str] = None,
                 temporal_type: Optional[str] = None,
                 temporal_value: Optional[str] = None,
                 x_category: Optional[str] = None,
                 x_concept_name: Optional[str] = None,
                 x_value: Optional[str] = None,
                 y_category: Optional[str] = None,
                 y_concept_name: Optional[str] = None,
                 y_value: Optional[str] = None):
        self.text = text
        self.temporal_type = temporal_type
        self.temporal_value = temporal_value
        self.x_category = x_category
        self.x_concept_name = x_concept_name
        self.x_value = x_value
        self.y_category = y_category
        self.y_concept_name = y_concept_name
        self.y_value = y_value

class TrialTreatment:
    def __init__(self,
                 text: Optional[str] = None,
                 temporal: Optional[str] = None,
                 category: Optional[str] = None,
                 concept_name: Optional[str] = None,
                 value: Optional[str] = None):
        self.text = text
        self.temporal = temporal
        self.category = category
        self.concept_name = concept_name
        self.value = value

class TrialOutcome:
    def __init__(self,
                 text: Optional[str] = None,
                 temporal: Optional[str] = None,
                 category: Optional[str] = None,
                 concept_name: Optional[str] = None):
        self.text = text
        self.temporal = temporal
        self.category = category
        self.concept_name = concept_name

class TrialInfo:
    def __init__(self,
                inclusion_criteria: List[TrialEligibilityCriterion],
                exclusion_criteria: List[TrialEligibilityCriterion],
                treatments: List[TrialTreatment],
                outcomes: List[TrialOutcome]):
        self.inclusion_criteria = inclusion_criteria
        self.exclusion_criteria = exclusion_criteria
        self.treatments = treatments
        self.outcomes = outcomes