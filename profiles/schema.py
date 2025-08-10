from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict


__all__ = ["Gender", "Ethnicity", "CogStyle", "Age", "PersonMeta", "PersonSet"]


class Gender(str, Enum):
    man = "man"
    woman = "woman"
    nonbinary = "nonbinary"

class Ethnicity(str, Enum):
    white = "white"
    black = "black"
    asian = "asian"
    latine = "latine"
    middle_eastern = "middle_eastern"
    indian = "indian"

class CogStyle(str, Enum):
    expansive = "expansive"
    literal = "literal"
    high_harm = "high_harm"
    low_harm = "low_harm"
    balanced = "balanced"

class Age(int, Enum):
    age_1=20
    age_2=25
    age_3=35
    age_4=45
    age_5=55 


@dataclass(frozen=True, slots=True)
class PersonMeta:
    gender: Gender
    ethnicity: Ethnicity
    cognitive_style: Optional[CogStyle] = None
    age: Optional[Age] = None

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
            key: getattr(meta, key).value if isinstance(getattr(meta, key, None), Enum) 
                 else getattr(meta, key, "Unknown") if getattr(meta, key, None) is not None
                 else "Unknown"
            for key in group_keys
        }

