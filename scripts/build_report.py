"""Build the short report and its figures from the two heuristic demo runs.

Expects data/runs/four-goblins-heuristic and data/runs/bugbear-patrol-heuristic,
which scripts/sample_batch.py writes. The Word file is report/encounter-simulation-report.docx.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from src.api_client import load_catalog
from src.bestiary import coverage
from src.eda import load_runs, outlier_fights
from src.paths import ROOT, RUNS

RUN_IDS = ("four-goblins-heuristic", "bugbear-patrol-heuristic")
FIGURES = ROOT / "report" / "figures"
DOCX = ROOT / "report" / "encounter-simulation-report.docx"


def _kept() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fights, turns = load_runs(RUNS)
    fights = fights[fights["run_id"].isin(RUN_IDS)].copy()
    turns = turns[turns["run_id"].isin(RUN_IDS)].copy()
    if fights.empty:
        raise SystemExit("No demo runs found. Run python -m scripts.sample_batch first.")
    catalog = load_catalog()
    report = coverage(list(catalog.spells.values()), list(catalog.monsters.values()))
    return fights, turns, report


def _layout_rows(fights: pd.DataFrame) -> list[dict]:
    rows = []
    for label, group in fights.groupby("label"):
        winners = group["winner"].value_counts().to_dict()
        rows.append(
            {
                "label": label,
                "fights": len(group),
                "wins": int(winners.get("party", 0)),
                "draws": int(winners.get("draw", 0)),
                "losses": int(winners.get("enemy", 0)),
                "mean_rounds": float(group["rounds"].mean()),
                "median_rounds": float(group["rounds"].median()),
                "party_damage": float(group["damage_by_party"].mean()),
                "enemy_damage": float(group["damage_by_enemy"].mean()),
                "party_hp": float(group["party_hp_remaining"].mean()),
                "cr": float(group["mean_enemy_cr"].mean()),
            }
        )
    return rows


def _save_figures(fights: pd.DataFrame) -> tuple[Path, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    order = ["Four goblins", "Bugbear patrol"]
    colors = {"Four goblins": "#1F4E79", "Bugbear patrol": "#C65911"}
    round_values = sorted(int(value) for value in fights["rounds"].unique())

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    width = 0.38
    positions = list(range(len(round_values)))
    for offset, label in ((-width / 2, order[0]), (width / 2, order[1])):
        counts = fights.loc[fights["label"] == label, "rounds"].value_counts()
        heights = [int(counts.get(value, 0)) for value in round_values]
        axis.bar([pos + offset for pos in positions], heights, width=width, label=label, color=colors[label])
    axis.set_xticks(positions, [str(value) for value in round_values])
    axis.set_xlabel("Rounds until the fight ended")
    axis.set_ylabel("Fights")
    axis.set_title("How long each layout lasted")
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    rounds_path = FIGURES / "rounds_by_layout.png"
    figure.savefig(rounds_path, dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    data = [fights.loc[fights["label"] == label, "party_hp_remaining"].astype(float) for label in order]
    box = axis.boxplot(data, tick_labels=order, patch_artist=True, medianprops={"color": "black"})
    for patch, label in zip(box["boxes"], order):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.85)
    axis.axhline(76, color="#666666", linestyle="--", linewidth=1, label="Party started at 76 HP")
    axis.set_ylabel("Party hit points remaining")
    axis.set_title("Hit points the party had left")
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    hp_path = FIGURES / "party_hp_by_layout.png"
    figure.savefig(hp_path, dpi=140)
    plt.close(figure)
    return rounds_path, hp_path


def _shade(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def _set_run(run, *, bold: bool = False, size: int = 11, color: str | None = None, italic: bool = False) -> None:
    run.bold = bold
    run.italic = italic
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def _paragraph(document: Document, text: str, *, bold: bool = False, italic: bool = False, size: int = 11, space_after: int = 8):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.space_before = Pt(0)
    run = paragraph.add_run(text)
    _set_run(run, bold=bold, italic=italic, size=size)
    return paragraph


def _heading(document: Document, text: str, level: int) -> None:
    paragraph = document.add_heading(text, level=level)
    for run in paragraph.runs:
        run.font.color.rgb = RGBColor.from_string("1F4E79")
        run.font.name = "Calibri"


def _table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(header)
        _set_run(run, bold=True, size=9, color="FFFFFF")
        _shade(cell, "1F4E79")
    for row_index, values in enumerate(rows, start=1):
        for index, value in enumerate(values):
            cell = table.rows[row_index].cells[index]
            cell.text = ""
            run = cell.paragraphs[0].add_run(value)
            _set_run(run, size=9)
            if row_index % 2 == 0:
                _shade(cell, "F2F2F2")
    document.add_paragraph().paragraph_format.space_after = Pt(6)


def build() -> None:
    fights, turns, report = _kept()
    rounds_path, hp_path = _save_figures(fights)
    rows = _layout_rows(fights)
    by_label = {row["label"]: row for row in rows}
    goblins = by_label["Four goblins"]
    patrol = by_label["Bugbear patrol"]
    outliers = outlier_fights(fights)
    numeric_rounds = pd.to_numeric(fights["rounds"])
    numeric_damage = pd.to_numeric(fights["damage_by_party"])
    round_q1, round_q3 = float(numeric_rounds.quantile(0.25)), float(numeric_rounds.quantile(0.75))
    damage_q1, damage_q3 = float(numeric_damage.quantile(0.25)), float(numeric_damage.quantile(0.75))
    round_fence = round_q3 + 1.5 * (round_q3 - round_q1)
    damage_fence = damage_q1 - 1.5 * (damage_q3 - damage_q1)
    rounds_enemy = float(pd.to_numeric(fights["rounds"]).corr(pd.to_numeric(fights["damage_by_enemy"])))
    rounds_party = float(pd.to_numeric(fights["rounds"]).corr(pd.to_numeric(fights["damage_by_party"])))
    cr_count = float(pd.to_numeric(fights["mean_enemy_cr"]).corr(pd.to_numeric(fights["enemy_count"])))

    wizard = turns[turns["actor_name"] == "Wizard"].sort_values(["fight_id", "round"])
    first_wizard = wizard.groupby("fight_id", as_index=False).first()
    goblin_aims = first_wizard[first_wizard["run_id"] == "four-goblins-heuristic"]["action_label"].value_counts().to_dict()
    patrol_first = first_wizard[first_wizard["run_id"] == "bugbear-patrol-heuristic"]["action_kind"].value_counts().to_dict()

    kinds = report["spells_by_kind"]
    reasons = report["exclude_reasons"]
    disagreements = []
    catalog = load_catalog()
    for monster in catalog.monsters.values():
        for name in monster.disagreements:
            disagreements.append(f"{monster.name} {name}")

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.paragraph_format.space_after = Pt(2)
    run = title.add_run("What thirty seeded fights say about two SRD encounters")
    _set_run(run, bold=True, size=18, color="1F4E79")
    _paragraph(
        document,
        "Programming for Data Science. Heuristic cohort, seeds 7–36 (four goblins) and 11–40 (bugbear patrol). November 2026.",
        italic=True,
        size=10,
        space_after=10,
    )

    _heading(document, "Question and data", 1)
    _paragraph(
        document,
        "The question is how a 5th-level fighter and wizard fare against two fixed layouts when both sides use the same damage-first heuristic: four goblins, or a bugbear with two hobgoblins. "
        "Each layout was fought 30 times. Dice for fight number k come from seed plus k, so the cohort can be rerun.",
    )
    _paragraph(
        document,
        f"The records are the 2014 Systems Reference Document on dnd5eapi.co, cached locally: {report['monster_total']} monsters and {report['spell_total']} spells. "
        f"Of those spells, {kinds.get('save', 0)} are saves, {kinds.get('attack', 0)} are spell attacks, {kinds.get('auto', 0)} deal automatic damage, and {kinds.get('heal', 0)} heal. "
        f"{kinds.get('excluded', 0)} are excluded: {reasons.get('no structured damage, healing, attack, or save', 0)} have no structured damage, healing, attack, or save, and {reasons.get('casting time is not 1 action', 0)} are not a single action. "
        f"{report['monsters_with_resolvable_action']} monsters have at least one resolvable action. {report['monsters_without_resolvable_action']} do not. "
        f"{report['save_text_disagreements']} monster saves describe half damage on a success while the structured field says none ({', '.join(disagreements)}). The engine follows the field.",
    )

    _heading(document, "Method", 1)
    _paragraph(
        document,
        "The grid is 12 by 12 squares of 5 feet, and a diagonal costs the same as an orthogonal step. A turn is a move plus one action. The modeled actions are Attack, a supported spell, Dash, Disengage, and Dodge. "
        "Bonus actions and legendary actions are not modeled. A natural 20 doubles dice but not the modifier. Resistance is applied after the save and rounded down. A fight still going after 15 rounds is a draw, and a draw is left out of the win rate.",
    )
    _paragraph(
        document,
        "The heuristic heals an ally at or below 35 percent hit points, otherwise it takes the highest expected damage, otherwise it moves closer, otherwise it dodges. Expected damage ignores advantage. "
        "In this cohort both sides use that heuristic. A language-model enemy is a separate policy: it must return an action id, one bad reply is retried, and the fallback is logged apart from a real model choice. Those calls are not seeded, and they are not in these 60 fights.",
    )

    _heading(document, "Results", 1)
    decided = int((fights["winner"] != "draw").sum())
    wins = int((fights["winner"] == "party").sum())
    _paragraph(
        document,
        f"The party won {wins} of {decided} decided fights. The other fight was a draw, so the party win rate among fights with a winner is {wins / decided:.0%}. "
        f"Across all 60 fights the mean length was {fights['rounds'].mean():.2f} rounds and the median was {fights['rounds'].median():.0f}. "
        f"Mean damage was {fights['damage_by_party'].mean():.1f} from the party and {fights['damage_by_enemy'].mean():.1f} from the enemies. The party started at 76 hit points and finished with {fights['party_hp_remaining'].mean():.1f} on average.",
    )
    _table(
        document,
        ["Layout", "Wins / draws / losses", "Rounds, mean (median)", "Damage, party / enemy", "Party HP left"],
        [
            [
                row["label"],
                f"{row['wins']} / {row['draws']} / {row['losses']}",
                f"{row['mean_rounds']:.2f} ({row['median_rounds']:.0f})",
                f"{row['party_damage']:.1f} / {row['enemy_damage']:.1f}",
                f"{row['party_hp']:.1f}",
            ]
            for row in (goblins, patrol)
        ],
    )
    _paragraph(
        document,
        f"Every goblin fight ended in round 1. The wizard's first action was Fireball in all 30, aimed at (8, 4) in {goblin_aims.get('Fireball L3 at (8, 4)', 0)} fights and at (8, 8) in {goblin_aims.get('Fireball L3 at (8, 8)', 0)}. "
        f"On the bugbear patrol the first wizard action was a spell in {patrol_first.get('spell', 0)} fights and a skipped turn in {patrol_first.get('skip', 0)}, which is the draw. "
        "Mean enemy challenge rating is 0.25 for the goblins and about 0.67 for the patrol. That gap did not produce a loss. It produced longer fights and more damage taken.",
        space_after=4,
    )
    document.add_picture(str(rounds_path), width=Inches(6.3))
    _paragraph(document, "Figure 1. Round counts. The round-15 bar is one fight, not a cluster.", italic=True, size=9, space_after=6)
    document.add_picture(str(hp_path), width=Inches(6.3))
    _paragraph(document, "Figure 2. Party hit points remaining. The low point on the patrol is the draw.", italic=True, size=9)

    _paragraph(
        document,
        f"Pearson correlation between rounds and enemy damage is {rounds_enemy:.2f}. Between rounds and party damage it is {rounds_party:.2f}. "
        "Longer fights are the ones in which the monsters got to attack. Party win has no variation among decided fights, because every decided fight is a win, so correlation with the win column is undefined. "
        f"Enemy count and mean challenge rating correlate at {cr_count:.2f}. With only two layouts, those columns are two labels for the same design choice. They should not be read as a general relationship between count and challenge rating.",
    )
    outlier = outliers.iloc[0]
    draw = fights.loc[fights["fight_id"] == outlier["fight_id"]].iloc[0]
    _paragraph(
        document,
        f"Outliers use a fence of 1.5 times the interquartile range, and a value on the fence is not flagged. "
        f"For rounds the upper fence is {round_fence:.1f} (the quartiles are {round_q1:.0f} and {round_q3:.0f}). For party damage the lower fence is {damage_fence:.0f}. "
        f"One fight is outside both: {outlier['fight_id']}, a {outlier['winner']} in {int(outlier['rounds'])} rounds, with {int(outlier['damage_by_party'])} party damage and {int(outlier['damage_by_enemy'])} enemy damage. "
        f"The wizard was reduced to 0 hit points before acting, by a javelin and two longbow shots, and skipped every turn after that. The party had {int(draw['party_hp_remaining'])} hit points left at the cap, all of them on the fighter, and the enemies had {int(draw['enemy_hp_remaining'])} left. "
        "The other 29 patrol fights ended by round 3. A three-round fight sits inside the fence.",
    )

    _heading(document, "Limits", 1)
    _paragraph(
        document,
        "These numbers describe one heuristic, not a human dungeon master and not the language model. The wizard casts Fireball whenever that is legal, so a layout that clumps monsters on round 1 will look easy. "
        "Changing the party, the opening squares, or the tactic would change the win rate. The model path is implemented and logged separately, but a model cohort is not repeatable from the seed and was not run for this draft. "
        "The in-app report button sends this same style of brief to the model and writes data/runs/latest-report.md. It needs an API key.",
    )
    _paragraph(
        document,
        "Several high correlations are layout identities. A third layout, or a sweep of party level and starting distance, would be required before those columns could be interpreted on their own. "
        "Spells that exist only as prose never enter the damage totals, so the coverage counts above are part of the result, not a footnote.",
    )

    _heading(document, "Reflection", 1)
    _paragraph(
        document,
        "Replace this paragraph before you submit. The rubric wants your own reflection: what you built, what the simulations changed your mind about, and what you would do with another week. Do not leave this placeholder in the file you turn in.",
        italic=True,
    )

    DOCX.parent.mkdir(parents=True, exist_ok=True)
    document.save(DOCX)
    print(f"Wrote {DOCX}")


if __name__ == "__main__":
    build()
