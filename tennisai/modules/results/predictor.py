"""
Result predictor: court-by-court win/loss predictions using ratings,
player module conclusions, and calibration history.
"""
from typing import Optional
from tennisai.models import MatchAnalysis
import datetime


def predict_match_results(
    my_team_url: str,
    usta_team_url: str,
    lineup: dict[str, list[str]],
    opponent_lineup: Optional[dict[str, list[str]]] = None,
    history_text: str = "",
    singles_courts: int = 2,
    doubles_courts: int = 3,
    match_date: Optional[datetime.date] = None,
    opponent_name: str = "",
) -> MatchAnalysis:
    """
    Predict court-by-court results for a match.

    When opponent_lineup is provided (both lineups known), uses run_analysis_direct —
    a single no-tool-call LLM request. This avoids the Groq 400 error from malformed
    get_player_history XML tool calls.

    When opponent_lineup is absent, falls back to run_analysis (full agent loop with
    web scraping tools to discover opponent data).
    """
    from tennisai.modules.players.store import load_player

    # Enrich history_text with player conclusions and calibration notes
    conclusion_lines: list[str] = []
    calibration_lines: list[str] = []
    for players in lineup.values():
        for name in players:
            pf = load_player(name)
            if not pf:
                continue
            if pf.conclusions:
                conclusion_lines.append(f"  {name}: {pf.conclusions}")
            for note in pf.calibration[-5:]:
                calibration_lines.append(f"  {name} ({note.context}): predicted {note.predicted}, actual {note.actual}")

    if conclusion_lines:
        history_text = (history_text + "\nPlayer intelligence:\n" + "\n".join(conclusion_lines)).strip()
    if calibration_lines:
        history_text = (history_text + "\nCalibration history:\n" + "\n".join(calibration_lines)).strip()

    if opponent_lineup:
        from tennisai.agent import run_analysis_direct
        return run_analysis_direct(
            lineup=lineup,
            opponent_lineup=opponent_lineup,
            history_text=history_text,
            singles_courts=singles_courts,
            doubles_courts=doubles_courts,
            match_date=match_date,
            opponent_name=opponent_name,
        )

    from tennisai.agent import run_analysis
    return run_analysis(
        my_team_url=my_team_url,
        usta_team_url=usta_team_url,
        lineup=lineup,
        opponent_lineup=None,
        history_text=history_text,
        singles_courts=singles_courts,
        doubles_courts=doubles_courts,
    )
