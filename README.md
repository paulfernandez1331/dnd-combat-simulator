# D&D Encounter Simulator

A Streamlit app that lays out a Dungeons & Dragons 5e fight, resolves attacks and supported spells in one click, and repeats the same layout many times so the results can be summarized.

This is a solo project for Programming for Data Science. The rubric also mentions commits from each teammate. Confirm with the instructor that a solo submission is accepted before you turn it in.

The language model is called through the OpenAI-compatible client in `src/ai.py`. The default endpoint is xAI (`https://api.x.ai/v1`). Action choices use `grok-4.20-non-reasoning`. The written report uses `grok-4.7`. The rubric names OpenAI, Anthropic, Gemini, a small local model, and disclosed use of Claude Code or Codex. xAI is not on that list. The base URL and model are environment variables, so you can point the same client at another provider. Confirm the choice with the instructor before the presentation. The API key stays in `.env` and is not committed.

## Setup

Python 3.14 is what this project was written and tested with.

```powershell
python -m pip install -r requirements.txt
copy .env.example .env
```

Put your key in `.env` if you want the model to pick enemy actions or draft the written analysis. Heuristic fights, the bestiary, and the charts work with the key left blank.

The 2024 SRD catalog is already in `data/cache` (341 monsters and 339 spells from `https://www.dnd5eapi.co/api/2024`). Refresh it with:

```powershell
python -m scripts.refresh_cache
```

## Run

```powershell
python -m streamlit run app.py
```

Pages:

- **Bestiary.** Coverage of what the engine can resolve, and the records it has to skip.
- **Encounter.** Place monsters, or load the two demo layouts. The battle mat labels each token with its name and hit points. Party tokens are teal and enemy tokens are red. Click a token to select it, or an empty square to move it, then start the fight.
- **Combat.** The battle mat is how you aim. A gold ring marks the creature whose turn it is. Click a green square to move. Action and Bonus open beside that token. Action includes Attack, which lists every weapon even before anyone is in reach. Click a creature for a single target. For an area, point at a gold square: the amber template follows the pointer, and the click casts it. End turn passes initiative. On any turn the model or the heuristic can play the whole turn, and the wheel is still there if you play that creature yourself. An armed attack closes the action list so the creature next to you stays visible, and the mat names that target.
- **Simulate.** Repeat the opening layout from 1 to 100 times. The party always uses the heuristic. Enemies use the model or the heuristic. Each finished fight is appended to `data/runs/` immediately. Stop writes a `STOP` file; the current turn finishes, and the fights already saved are kept.
- **Analysis.** Starts on the latest Simulate batch. A toggle pools every saved run. The charts cover who went first, who scored the first knockout, how many turns each side took, concentration, and damage per round. "Write the report" sends that brief to the model and saves `data/runs/latest-report.md`.

The rules the engine actually uses are in the sidebar. A fight that is still going after 15 rounds is a draw.

## The repeatable cohort

Heuristic fights use a stored dice seed, so the same layout and seed produce the same log. Model choices are not seeded. A batch that asks for the model and has no API key stops with an error. It does not silently switch the whole batch to the heuristic.

To regenerate the two demo layouts (30 fights each) and print the analysis brief:

```powershell
python -m scripts.sample_batch
```

That writes `data/runs/four-goblins-heuristic` and `data/runs/bugbear-patrol-heuristic`. Those folders are gitignored. Delete a folder before running the script again, or the old fights stay in place.

Turn **Last batch only** off before comparing layouts. Challenge rating does not vary inside one batch, so a single run cannot show what it does.

The write-up of the 60 heuristic fights is `report/encounter-simulation-report.docx`. That document, and the two run folders it reads, are the earlier 2014 cohort in which one action ended the turn. They were not rerun after the 2024 catalog and the move-plus-action turn. Rebuild the figures and the document after a fresh batch with:

```powershell
python -m scripts.build_report
```

The reflection section is a placeholder. Replace it before you submit.

## Tests

```powershell
python -m pytest -q
```

`tests/test_app.py` drives the Streamlit script with AppTest. It does not open a browser. Battle-mat clicks are `tests/test_battlemap.py`. That file loads `board/index.html` in Edge through Playwright, draws a real `battle_payload`, and clicks the squares and the action wheel. The same helper, `tests/board_browser.py`, is what to use when a mat change needs a browser. Edge has to be installed. Playwright uses that browser and does not download another one.

## What this does not do

It is not a full virtual tabletop. There is no fog of war, no drawing tools, and no multiplayer. Cover, beast-form Wild Shape, and anything that only matters on a rest or a skill check are left out. A placed zone or conjured token from the 2024 SRD stays on the mat while the caster concentrates. Grease is difficult terrain, lasts 1 minute, and does not use concentration. Summons that need a creature stat block the cache does not have are still left out. Legendary actions are omitted. Player characters come from public D&D Beyond sheets saved in data/samples, or from a link pasted on the Encounter page. The importer uses the scores, weapons, mastery, and structured actions on that sheet, including Rage and a bonus heal when the sheet stores the die and the uses. A spell or monster action is resolved when the record has damage or healing, including a few description patterns the 2024 API uses instead of the old fields. Other prose-only effects are counted on the Bestiary page and then left out. If a monster save's field and its description disagree about half damage, combat follows the field and the Bestiary counts the disagreement.
