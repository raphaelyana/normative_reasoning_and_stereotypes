import enum
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from profiles.schema import (
    PersonMeta, 
    Gender, Ethnicity, CogStyle, 
    PersonSet
)
from profiles.profile_sets import (
    PERSON_SYSTEMATIC,
    PERSON_CORE,
    PERSON_MAIN
)



def make_system_message(
    case_name: str,
    person_set: PersonSet,
    person_key: Optional[str] = None,
) -> dict:
    
    """
    Build the dict {"role": "system", "content": "..."} expected by call_llm().
    
    Parameters
    ----------
    case_name
        Name of the classification task (e.g. "StereotypeDetection").
    person_key
        Key in PERSON_SEEDS or None for a neutral system prompt.
    """

    if person_key is None:
        base = f"You are a classifier for {case_name}. Decide if the reasoning is an example or not."
        return {"role": "system", "content": base}
    try:
        seed = person_set.seeds[person_key]
    except KeyError as e:
        raise KeyError(f"Unknown persona_key '{person_key}'. Valid keys: {list(person_set.seeds)}") from e

    return {"role": "system", "content": seed}