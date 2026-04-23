import datetime
from typing import Optional
from pydantic import BaseModel
from tennisai.models import CourtPrediction, CourtResult


class MatchFile(BaseModel):
    id: str
    recorded_at: str
    status: str = "upcoming"          # "upcoming" | "completed" | "backfill-pending"
    match_date: Optional[datetime.date] = None
    location: str = ""
    home_team: str = ""
    away_team: str = ""
    opponent: str = ""
    scorecard_url: str = ""
    our_lineup: dict[str, list[str]] = {}
    opponent_lineup: dict[str, list[str]] = {}
    predictions: list[CourtPrediction] = []
    actual_results: list[CourtResult] = []
    overall_predicted: str = ""
    overall_actual: Optional[str] = None
