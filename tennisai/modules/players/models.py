from typing import Optional
from pydantic import BaseModel


class PartnerRecord(BaseModel):
    name: str
    wins: int = 0
    losses: int = 0


class CourtRecord(BaseModel):
    court_key: str    # e.g. "C1S", "C2D"
    wins: int = 0
    losses: int = 0


class CalibrationNote(BaseModel):
    context: str      # e.g. "Court 1 Singles vs ETC-A Team"
    predicted: str    # "us" or "them"
    actual: str       # "us" or "them"
    note: str


class PlayerFile(BaseModel):
    name: str
    team: str = ""
    tennisrecord_rating: float = 0.0
    ntrp_level: float = 0.0
    profile_url: str = ""
    wtn_singles: Optional[float] = None   # WTN scale: lower = stronger (1=elite, 40=beginner)
    wtn_doubles: Optional[float] = None   # Same scale, specific to doubles
    usta_profile_url: str = ""
    singles_wins: int = 0
    singles_losses: int = 0
    doubles_wins: int = 0
    doubles_losses: int = 0
    matches_played: int = 0
    by_court: list[CourtRecord] = []
    best_partners: list[PartnerRecord] = []
    conclusions: str = ""
    calibration: list[CalibrationNote] = []
    updated_at: str = ""
