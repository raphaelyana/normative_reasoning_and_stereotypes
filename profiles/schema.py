from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict


__all__ = ["Gender", "Ethnicity", "CogStyle", "PersonMeta", "PersonSet"]


class Gender(str, Enum):
    man = "man"
    woman = "woman"
    nonbinary = "nonbinary"

class Ethnicity(str, Enum):
    white = "white"
    black = "black"
    asian = "asian"
    latine = "latine"

class CogStyle(str, Enum):
    expansive = "expansive"
    literal = "literal"
    high_harm = "high_harm"
    low_harm = "low_harm"
    balanced = "balanced"

@dataclass(frozen=True, slots=True)
class PersonMeta:
    gender: Gender
    ethnicity: Ethnicity
    cognitive_style: Optional[CogStyle] = None

@dataclass(frozen=True)
class PersonSet:
    seeds: Dict[str, str]
    metadata: Dict[str, PersonMeta]

    def get_traits(self, profile_id: str, group_keys=("gender", "ethnicity", "cognitive_style")) -> dict:
        pid = profile_id.replace("_passive", "")
        meta = self.metadata.get(pid, None)
        if not meta:
            return {key: "Unknown" for key in group_keys}
        return {
            key: getattr(meta, key).value if isinstance(getattr(meta, key, None), Enum) else getattr(meta, key, "Unknown")
            for key in group_keys
        }

