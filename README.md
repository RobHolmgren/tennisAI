# TennisAI

An AI agent that helps USTA adult league team captains prepare for upcoming matches. It scrapes player ratings from [tennisrecord.com](https://www.tennisrecord.com) and schedule/results data from [tennislink.usta.com](https://tennislink.usta.com), then uses an LLM to predict court-by-court outcomes based on your tentative lineup.

---

## Features

- Fetches your team's NTRP player ratings from tennisrecord.com
- Identifies your next match and recent results from USTA TennisLink
- Detects the match format (1, 2, or 3 singles courts + 3 doubles) automatically
- Scrapes your opponent's player ratings
- Pulls per-court win/loss trends from individual match scorecards
- Pulls individual player match history (last 6 months) for form analysis
- Predicts the opponent's most likely lineup based on their ratings
- Interactively collects your tentative lineup (players shown as a numbered roster)
- Predicts each court outcome with confidence level and reasoning
- Suggests lineup changes to maximise your win chances
- Optionally exports predictions and suggestions to CSV

---

## Setup

### 1. Prerequisites

- Python 3.11+
- A [USTA TennisLink](https://tennislink.usta.com) account
- A [Groq API key](https://console.groq.com) (free tier — default provider)

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium  # one-time browser install for USTA TennisLink
```

> Activate the virtual environment (`source .venv/bin/activate`) each time you open a new terminal session.

### 3. Configure credentials and team URLs

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```
# AI provider
GROQ_API_KEY=your_groq_api_key_here
AI_PROVIDER=groq

# USTA TennisLink login
USTA_USERNAME=your_usta_email@example.com
USTA_PASSWORD=your_usta_password_here

# Your team URLs — stored here so you never need to type them on the command line
MY_TEAM_URL=https://www.tennisrecord.com/adult/teamprofile.aspx?teamname=YourTeam&year=2026&s=2
USTA_TEAM_URL=https://tennislink.usta.com/Leagues/Main/StatsAndStandings.aspx?t=R-3#&&s=YOUR_TOKEN
```

> **Security:** `.env` is listed in `.gitignore` and will never be committed to the repository. Your credentials, USTA session token, and team URLs all stay on your local machine only.

---

## Usage

Once your `.env` is configured, no arguments are required:

```bash
python -m tennisai analyze
```

With CSV output:

```bash
python -m tennisai analyze --output-csv predictions.csv
```

The CLI will:
1. Fetch the match format from USTA (1S/2S/3S + 3D)
2. Display your team roster as a numbered list
3. Prompt you to pick players for each court by number
4. Run the AI agent and print a full match analysis

You can override the URLs for a one-off analysis:

```bash
python -m tennisai analyze --team-url "https://..." --usta-url "https://..."
```

### Diagnostic commands

Test tennisrecord.com scraping independently:

```bash
python -m tennisai check-tennisrecord
```

Test USTA TennisLink login and schedule fetch:

```bash
python -m tennisai check-usta
python -m tennisai check-usta --debug   # prints raw page structure
```

---

## What the AI considers

When predicting court outcomes the agent weighs:

1. **Player ratings** — tennisrecord.com estimated ratings (2-decimal precision, e.g. 2.98) are the primary signal; NTRP band is the fallback
2. **Per-court trends** — win/loss record at each court position (S1, S2, D1, D2, D3) pulled from historical match scorecards
3. **Player form** — individual win/loss history over the last 6 months, including matches outside the current team
4. **Predicted opponent lineup** — opponent players ranked strongest-to-weakest and assigned to courts accordingly
5. **Lineup suggestions** — specific player swaps or reassignments to improve your chances

---

## Project Structure

```
tennisAI/
├── .env.example          # Credential template (committed — no real values)
├── .env                  # Your credentials and URLs (gitignored, never committed)
├── requirements.txt
├── tennisai/
│   ├── cli.py            # Click CLI entry point
│   ├── agent.py          # AI agent with tool-use loop (Groq/Claude)
│   ├── config.py         # Environment variable loading
│   ├── models.py         # Pydantic data models
│   └── tools/
│       ├── tennisrecord.py   # tennisrecord.com scraper
│       └── usta.py           # USTA TennisLink Playwright client
```

---

## Changing the AI Provider

The default provider is **Groq** (free tier, `llama-3.3-70b-versatile`). To switch to Claude:

```
AI_PROVIDER=claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Both providers use the same tool-use interface — the agent logic in `agent.py` is shared.

---

## USTA TennisLink Notes

tennislink.usta.com uses Auth0 for authentication and JavaScript to render all content. The `tools/usta.py` module uses a headless Chromium browser (Playwright) to log in, click tabs, and parse the rendered HTML. If USTA updates their page structure, the tab IDs and HTML panel IDs at the top of `usta.py` may need to be updated.

---

## Contributing

Pull requests welcome. Please ensure:
- No credentials, session tokens, team URLs, or `.env` files are ever committed
- New AI providers are implemented as alternatives in `agent.py` behind the `AI_PROVIDER` config
