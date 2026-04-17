import datetime
from typing import Optional
from pydantic import BaseModel


class Player(BaseModel):
    name: str
    ntrp_level: Optional[float] = None   # official NTRP band, e.g. 3.0 or 3.5
    rating: Optional[float] = None       # tennisrecord estimated rating (2 decimals, e.g. 2.98) — use this as primary predictor
    team: str = ""
    profile_url: str = ""               # tennisrecord.com player profile URL for history lookups


class Team(BaseModel):
    name: str
    url: str
    players: list[Player] = []


class Match(BaseModel):
    date: Optional[datetime.date] = None
    home_team: str
    away_team: str
    location: str = ""


class MatchResult(BaseModel):
    date: Optional[datetime.date] = None
    opponent: str
    location: str = ""
    score: str = ""       # e.g. "3-2" (courts won - courts lost)
    won: Optional[bool] = None


class CourtPrediction(BaseModel):
    court: int
    court_type: str  # "Singles" or "Doubles"
    my_players: list[str]
    opponent_players: list[str]
    predicted_winner: str  # "us" or "them"
    confidence: str  # "high", "medium", "low"
    reasoning: str


class CourtTrend(BaseModel):
    court: int
    court_type: str              # "Singles" or "Doubles"
    wins: int = 0
    losses: int = 0

    @property
    def matches_played(self) -> int:
        return self.wins + self.losses


class PlayerHistory(BaseModel):
    player_name: str
    wins_last_6_months: int = 0
    losses_last_6_months: int = 0
    matches: list[dict] = []     # [{date, opponent, won, score}]


class MatchFormat(BaseModel):
    singles_courts: int = 3      # 1, 2, or 3
    doubles_courts: int = 3      # always 3 per USTA adult league rules


class MatchAnalysis(BaseModel):
    match: Match
    my_team: Team
    opponent_team: Team
    my_recent_results: list[MatchResult] = []
    opponent_recent_results: list[MatchResult] = []
    predictions: list[CourtPrediction] = []
    overall_outlook: str = ""
    lineup_suggestions: list[str] = []
