"""
Learner: compares predicted vs actual court results and writes calibration notes
to player files. Called after record-result to improve future predictions.
"""
from tennisai.modules.matches.models import MatchFile
from tennisai.modules.players.models import CalibrationNote
from tennisai.modules.players.store import load_player, save_player


def apply_learnings(match_file: MatchFile) -> list[str]:
    """
    Update calibration notes in player files based on actual vs predicted results.
    Returns a list of summary lines (one per court) for display.
    """
    if not match_file.actual_results:
        return []

    summary: list[str] = []

    for actual in match_file.actual_results:
        pred = next(
            (p for p in match_file.predictions
             if p.court == actual.court and p.court_type == actual.court_type),
            None,
        )
        if not pred:
            continue

        court_label = f"Court {actual.court} {actual.court_type}"
        correct = pred.predicted_winner.lower() == actual.winner.lower()
        result_str = "correct" if correct else f"wrong (predicted {pred.predicted_winner}, got {actual.winner})"
        summary.append(f"  {court_label}: {result_str}")

        note_text = (
            f"Predicted {pred.predicted_winner}, actual {actual.winner} on {court_label} vs {match_file.opponent}."
            + ("" if correct else " Recalibrate confidence for similar matchups.")
        )
        calibration_note = CalibrationNote(
            context=f"{court_label} vs {match_file.opponent}",
            predicted=pred.predicted_winner,
            actual=actual.winner,
            note=note_text,
        )

        for player_name in pred.my_players:
            pf = load_player(player_name)
            if not pf:
                continue
            pf.calibration.append(calibration_note)
            pf.calibration = pf.calibration[-20:]  # Keep last 20
            save_player(pf)

    return summary
