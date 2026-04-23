# TennisAI

An AI agent that helps USTA adult league team captains prepare for upcoming matches. It scrapes player ratings from [tennisrecord.com](https://www.tennisrecord.com) and schedule/results data from [tennislink.usta.com](https://tennislink.usta.com), then uses an LLM to predict court-by-court outcomes based on your tentative lineup.

> **Recommended setup:** Run the AI model locally using [Ollama](https://ollama.com). It's free, has no rate limits, requires no API key, and keeps all your data on your own machine. See [Choosing an AI Provider](#choosing-an-ai-provider) for setup instructions.

---

## Features

- Fetches your team's NTRP and WTN player ratings from tennisrecord.com and worldtennisnumber.com
- Identifies your next match and recent results from USTA TennisLink
- Detects the match format (1, 2, or 3 singles courts + 3 doubles) automatically
- Scrapes your opponent's player ratings
- Pulls per-court win/loss trends from individual match scorecards
- Pulls individual player match history (last 6 months) for form analysis
- Predicts the opponent's most likely lineup based on their ratings
- Interactively collects your tentative lineup (players shown as a numbered roster)
- Predicts each court outcome with confidence level and reasoning
- Suggests lineup changes to maximise your win chances
- Records actual match results to calibrate future predictions
- Optionally exports predictions and suggestions to CSV
- Supports multiple AI providers — including a fully local, free, unlimited option via Ollama

---

## Setup

### 1. Prerequisites

- Python 3.11+
- A [USTA TennisLink](https://tennislink.usta.com) account
- [Ollama](https://ollama.com) for local AI inference (recommended — free, no API key needed)

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

Edit `.env` with your USTA credentials, team URLs, and chosen AI provider (see below).

> **Security:** `.env` is listed in `.gitignore` and will never be committed. Your credentials and team URLs stay on your local machine only.

---

## Choosing an AI Provider

TennisAI supports four AI providers. **Ollama is the recommended default** — it runs entirely on your machine with no API key, no rate limits, and no cost.

| Provider | Cost | Rate limits | Quality | Setup |
|---|---|---|---|---|
| **Ollama** (local) ✅ **recommended** | Free, unlimited | None | Good–Excellent | Install Ollama + pull a model |
| **Groq** | Free tier | 100k tokens/day | Excellent | API key from console.groq.com |
| **Gemini** | Free tier | ~1,500 req/day | Excellent | API key from aistudio.google.com |
| **Claude** | Paid | None | Excellent | API key from anthropic.com |

### Ollama — local, free, unlimited ✅ recommended

Ollama runs an LLM entirely on your machine. No API key, no rate limits, no cost.

**One-time setup:**

```bash
# Install Ollama
brew install ollama          # macOS
# or download from https://ollama.com for Windows/Linux

# Pull a model (pick one)
ollama pull llama3.1:8b     # recommended — best quality, ~5 GB, supports tool calling
ollama pull llama3.2:3b     # faster, lighter — ~2 GB, good for quick predictions
ollama pull qwen2.5:7b      # alternative with strong instruction following

# Start the Ollama server (runs in the background)
ollama serve
```

**.env settings:**

```
AI_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b        # optional — defaults to llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434/v1   # optional — this is the default
```

> **Model note:** Use a model that supports tool calling — `llama3.1`, `llama3.2`, `qwen2.5`, or `mistral`. Older models like `llama2` do not support tools and will not work with the full agent loop.

### Groq — free cloud API

```
AI_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com). Free tier allows 100,000 tokens per day, which is sufficient for one or two match analyses daily.

### Gemini — free cloud API

```
AI_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Use AI Studio — not the Google Cloud Console — to ensure the free tier quota is applied correctly.

### Claude (Anthropic) — paid

```
AI_PROVIDER=claude
ANTHROPIC_API_KEY=your_key_here
```

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
1. Fetch your match schedule from USTA TennisLink
2. Let you select the upcoming match to analyse
3. Display your team roster as a numbered list
4. Prompt you to enter your lineup by player number
5. Run the AI agent and print a full court-by-court analysis

You can override the URLs for a one-off analysis:

```bash
python -m tennisai analyze --team-url "https://..." --usta-url "https://..."
```

### All commands

| Command | Description |
|---|---|
| `analyze` | Analyse a match and predict court outcomes |
| `suggest-lineup` | AI predicts opponent lineup, then recommends your optimal lineup |
| `record-result` | Enter actual match results to improve future predictions |
| `list-matches` | List all saved matches (upcoming and completed) |
| `view-match <id>` | Show full detail for a saved match |
| `backfill` | Run predictions on past matches to calibrate accuracy |
| `accuracy` | Show prediction accuracy stats |
| `update-players` | Rebuild player files from match history |
| `update-wtn` | Fetch WTN ratings for all players |
| `list-teams` | List all configured teams and show which is active |
| `switch-team <number>` | Switch the active team (persists to `.env`) |
| `check-tennisrecord` | Test tennisrecord.com scraping |
| `check-usta` | Test USTA TennisLink login and schedule fetch |

---

## What the AI considers

When predicting court outcomes the agent weighs:

1. **Player ratings** — WTN singles/doubles (lower = stronger) as the primary signal; tennisrecord.com combined rating as supporting context
2. **Per-court trends** — win/loss record at each court position (S1, S2, D1, D2, D3) pulled from historical match scorecards
3. **Player form** — individual win/loss history over the last 6 months, including matches outside the current team
4. **Predicted opponent lineup** — opponent players ranked strongest-to-weakest and assigned to courts accordingly
5. **Calibration history** — adjustments based on how past predictions compared to actual results
6. **Lineup suggestions** — specific player swaps or reassignments to improve your team's chances

---

## Multiple Teams

TennisAI supports multiple teams from a single installation. Each team keeps its own match history; player files are shared (useful if you play on more than one team with some of the same players).

### Setup

Add team entries to `.env`:

```
TEAM_COUNT=2
ACTIVE_TEAM=1

TEAM_1_NAME=My First Team
TEAM_1_URL=https://www.tennisrecord.com/adult/teamprofile.aspx?teamname=...
TEAM_1_USTA_URL=https://tennislink.usta.com/Leagues/Main/statsandstandings.aspx#&&s=...

TEAM_2_NAME=My Second Team
TEAM_2_URL=https://www.tennisrecord.com/adult/teamprofile.aspx?teamname=...
TEAM_2_USTA_URL=https://tennislink.usta.com/Leagues/Main/statsandstandings.aspx#&&s=...
```

Match files are stored in team-specific subdirectories (`matches/team-1/`, `matches/team-2/`, …).

### Switching teams

```bash
python -m tennisai list-teams          # show all teams (* = active)
python -m tennisai switch-team 2       # switch to team 2 (persists to .env)
python -m tennisai analyze --team 2   # one-off run for team 2 without switching
```

### Existing single-team users — no changes needed

If your `.env` uses the original `MY_TEAM_NAME` / `MY_TEAM_URL` / `USTA_TEAM_URL` keys, everything continues to work without any changes. The multi-team keys are optional; the old keys remain supported as a fallback.

---

## Project Structure

```
tennisAI/
├── .env.example              # Credential template (committed — no real values)
├── .env                      # Your credentials and URLs (gitignored, never committed)
├── requirements.txt
├── matches/                  # Saved match predictions and results (gitignored)
├── players/                  # Per-player rating and calibration files (gitignored)
└── tennisai/
    ├── cli.py                # Click CLI entry point
    ├── agent.py              # AI agent with tool-use loop (all providers)
    ├── config.py             # Environment variable loading
    ├── models.py             # Pydantic data models
    ├── tools/
    │   ├── tennisrecord.py   # tennisrecord.com scraper
    │   ├── usta.py           # USTA TennisLink Playwright client
    │   ├── usta_wtn.py       # World Tennis Number (WTN) fetcher
    │   └── history.py        # Match history helpers
    └── modules/
        ├── matches/          # Match storage and retrieval
        ├── players/          # Player profile store and analyzer
        ├── lineup/           # Lineup optimizer and predictor
        └── results/          # Result predictor and learner
```

---

## USTA TennisLink Notes

tennislink.usta.com uses Auth0 for authentication and JavaScript to render all content. The `tools/usta.py` module uses a headless Chromium browser (Playwright) to log in, click tabs, and parse the rendered HTML. If USTA updates their page structure, the tab IDs and HTML panel IDs at the top of `usta.py` may need to be updated.

---

## Contributing

Pull requests welcome. Please ensure:
- No credentials, session tokens, team URLs, or `.env` files are ever committed
- New AI providers are implemented in `agent.py` and `modules/lineup/predictor.py` behind the `AI_PROVIDER` config
- The `matches/` and `players/` directories remain gitignored
