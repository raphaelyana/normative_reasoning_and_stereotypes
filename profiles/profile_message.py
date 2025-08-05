import enum
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from schema import (
    PersonMeta, 
    Gender, Ethnicity, CogStyle, 
    PersonSet
)
from profile_sets import (
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




def get_profile_traits(profile_id: str, person_set: PersonSet,  group_keys=("gender", "ethnicity", "cognitive_style")) -> dict:
    """
    Return the trait values associated with a single profile ID.

    This is used in **sample-level or profile-level annotations**, such as
    logging trait information during case analysis or disagreement detection.

    Parameters:
    -----------
    profile_id : str
        The ID of the profile (e.g., 'profile3_passive').
    group_keys : tuple
        Traits to extract from the profile metadata.

    Returns:
    --------
    dict : Mapping of trait_name → trait_value for the profile.
    Unknown is returned if the profile is not in the metadata.
    """

    pid = profile_id.replace("_passive", "")
    meta = person_set.metadata.get(pid, None)
    if not meta:
        return {key: "Unknown" for key in group_keys}
    return {
        key: getattr(meta, key).value if isinstance(getattr(meta, key, None), Enum) else getattr(meta, key, "Unknown")
        for key in group_keys
    }