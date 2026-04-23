from tennisai.modules.players.models import PlayerFile
from tennisai.modules.players.store import load_player, save_player

GROQ_MODEL = "llama-3.3-70b-versatile"
CLAUDE_MODEL = "claude-sonnet-4-6"


def generate_conclusions(pf: PlayerFile) -> str:
    """Use LLM to generate human-readable strengths/weaknesses for a player."""
    from tennisai.config import get_ai_provider, get_groq_api_key, get_anthropic_api_key

    court_lines = "\n".join(
        f"  {c.court_key}: {c.wins}W-{c.losses}L" for c in pf.by_court
    ) or "  No court data yet"
    partner_lines = "\n".join(
        f"  {p.name}: {p.wins}W-{p.losses}L" for p in pf.best_partners[:5]
    ) or "  No partner data yet"
    calib_lines = "\n".join(
        f"  {c.context}: predicted={c.predicted}, actual={c.actual}" for c in pf.calibration[-5:]
    ) if pf.calibration else ""

    prompt = (
        f"Player: {pf.name} (NTRP {pf.ntrp_level}, tennisrecord rating {pf.tennisrecord_rating})\n"
        f"Singles: {pf.singles_wins}W-{pf.singles_losses}L\n"
        f"Doubles: {pf.doubles_wins}W-{pf.doubles_losses}L\n"
        f"By court:\n{court_lines}\n"
        f"Best partners:\n{partner_lines}\n"
        + (f"Recent calibration:\n{calib_lines}\n" if calib_lines else "")
        + "\nIn 2-3 sentences, summarize this player's strengths, weaknesses, "
        "optimal court position, and best doubles partner. Be specific and data-driven."
    )

    provider = get_ai_provider()
    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=get_groq_api_key())
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return resp.choices[0].message.content or ""
    if provider == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=get_anthropic_api_key())
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""
    return ""


def refresh_player_conclusions(name: str) -> None:
    """Refresh the conclusions field for a named player."""
    pf = load_player(name)
    if not pf:
        return
    pf.conclusions = generate_conclusions(pf)
    save_player(pf)
