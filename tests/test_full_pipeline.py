import subprocess
import sys


def test_full_pipeline_invokes_fetch_topics(monkeypatch):
    """Ensure the full pipeline script calls fetch_topics.py with correct flags."""

    called_cmd = {}

    def fake_run(cmd, text=True):  # noqa: D401
        """Fake subprocess.run that records the command and returns success."""
        called_cmd["value"] = cmd
        class FakeCompleted:  # noqa: D401
            returncode = 0
        return FakeCompleted()

    # Patch subprocess.run used inside the pipeline script
    monkeypatch.setattr(subprocess, "run", fake_run)

    # Execute the pipeline entrypoint via python -m to get a fresh __main__
    exit_code = subprocess.call([sys.executable, "scripts/full_pipeline.py", "--with-content-analysis"])
    assert exit_code == 0, "Pipeline script should exit cleanly"

    # Verify fetch_topics.py was in the command
    assert called_cmd["value"].pop(0).endswith("python"), "First element should be python executable"
    assert "fetch_topics.py" in called_cmd["value"][0], "Expected fetch_topics.py invocation"
    assert "--with-content-analysis" in called_cmd["value"], "Flag should be forwarded" 