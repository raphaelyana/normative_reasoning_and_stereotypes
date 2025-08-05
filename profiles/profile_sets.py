from schema import *
from profile_dict import (
    PERSON_SEEDS_SYSTEMATIC, 
    PROFILE_META_SYSTEMATIC,
    PERSON_SEEDS_CORE,
    PERSON_SEEDS
)

__all__ = ["PERSON_SYSTEMATIC", "PERSON_CORE", "PERSON_MAIN"]

PERSON_SYSTEMATIC = PersonSet(
    seeds=PERSON_SEEDS_SYSTEMATIC,
    metadata=PROFILE_META_SYSTEMATIC,
)

PERSON_CORE = PersonSet(
    seeds=PERSON_SEEDS_CORE,
    metadata={}
)

PERSON_MAIN = PersonSet(
    seeds=PERSON_SEEDS,
    metadata={}
)