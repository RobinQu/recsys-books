from pathlib import Path

import nbformat
import pytest

from scripts.execute_notebooks import sanitize_notebook_outputs
from scripts.verify_published_notebooks import verify


def test_sanitize_notebook_outputs_removes_workspace_and_staging_paths(tmp_path: Path) -> None:
    project = tmp_path / "workspace"
    artifacts = tmp_path / "stage" / "artifacts"
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell(
            "print('ok')",
            execution_count=1,
            outputs=[nbformat.v4.new_output(
                "stream",
                name="stdout",
                text=f"{project}/data {artifacts}/results /var/folders/a/T/ipykernel_1/2.py:3\n",
            )],
        )
    ])
    sanitize_notebook_outputs(
        notebook,
        project_root=project,
        artifact_root=artifacts,
        execution_cwd=project / "notebooks",
    )
    text = notebook.cells[0].outputs[0].text
    assert str(project) not in text and str(artifacts) not in text
    assert "/var/folders/" not in text
    assert "<PROJECT_ROOT>" in text and "<ARTIFACT_ROOT>" in text and "<KERNEL_TEMP>" in text


def test_publication_verifier_rejects_unexecuted_or_stale_artifacts(tmp_path: Path) -> None:
    notebook_dir = tmp_path / "notebooks"
    preview_dir = tmp_path / "previews"
    notebook_dir.mkdir(); preview_dir.mkdir()
    with pytest.raises(AssertionError, match="notebook slugs differ"):
        verify(notebook_dir, preview_dir)
