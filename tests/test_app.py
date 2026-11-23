"""Smoke test for the Streamlit pages. This does not click through a browser."""

from streamlit.testing.v1 import AppTest

from src.paths import ROOT


def _button(app: AppTest, label: str):
    matches = [button for button in app.button if button.label == label]
    assert matches, label
    return matches[0]


def test_bestiary_preset_and_one_attack():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=90)
    app.run()
    assert not app.exception
    assert app.metric[0].value == "341"
    assert app.metric[1].value == "339"

    app.sidebar.radio[0].set_value("Encounter").run()
    assert not app.exception
    sample = next(box for box in app.selectbox if box.label == "Sample character")
    sample.set_value("173717514").run()
    _button(app, "Add sample").click().run()
    assert not app.exception
    assert any("Sample Barbarian" in row for row in _table_text(app))

    _button(app, "Load four goblin warriors").click().run()
    assert not app.exception
    assert any("Sample Barbarian" in row for row in _table_text(app))
    chosen = next(box for box in app.selectbox if box.label == "Selected creature")
    chosen.set_value("goblin-warrior-1").run()
    assert not app.exception
    visible = _visible(app)
    assert "SRD stat block" in visible
    assert "Scimitar" in visible

    link = next(box for box in app.text_input if box.label == "Character link or id")
    link.set_value("not-a-character").run()
    _button(app, "Import character").click().run()
    assert not app.exception
    assert "numeric id" in _visible(app).lower()

    _button(app, "Start fight").click().run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Combat"
    _button(app, "Restart with this seed")
    seed_box = next(box for box in app.number_input if box.label == "Dice seed")
    assert int(seed_box.value) == 7
    fight = app.session_state["encounter"]
    assert fight.log[0] == "Dice seed 7."
    _button(app, "Let the model choose")
    _button(app, "Use the heuristic instead").click().run()
    assert not app.exception
    _button(app, "End turn").click().run()
    assert not app.exception
    logged = " ".join(str(block.value) for block in list(app.code) + list(app.markdown))
    assert "ends the turn" in logged
    seed_box = next(box for box in app.number_input if box.label == "Dice seed")
    seed_box.set_value(12).run()
    assert not app.exception
    fight = app.session_state["encounter"]
    note = "Dice seed is now 12. Rolls from here use that seed."
    assert [line for line in fight.log if line.startswith("Dice seed is now")] == [note]
    assert fight.log[0] == "Dice seed 7."
    assert any(line.startswith("Initiative:") for line in fight.log)
    seed_box = next(box for box in app.number_input if box.label == "Dice seed")
    seed_box.set_value(12).run()
    assert not app.exception
    fight = app.session_state["encounter"]
    assert [line for line in fight.log if line.startswith("Dice seed is now")] == [note]

    app.sidebar.radio[0].set_value("Analysis").run()
    assert not app.exception


def _visible(app: AppTest) -> str:
    chunks = [str(block.value) for block in list(app.markdown) + list(app.error)]
    return "\n".join(chunks)


def _table_text(app: AppTest) -> list[str]:
    lines = []
    for frame in list(app.dataframe) + list(app.table):
        lines.append(str(frame.value))
    return lines
