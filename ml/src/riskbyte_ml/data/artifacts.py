"""Atomic artifact writes; exclusive creation protects processed outputs."""

import os
import tempfile
from pathlib import Path

from riskbyte_ml.data.errors import DatasetError


def write_artifact(path: Path, content: bytes, *, overwrite: bool = True) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Hard-link publish is atomic and fails when the destination already exists.
            os.link(temporary, path)
    except FileExistsError:
        raise DatasetError(
            "dataset_output_exists", "Output already exists; choose a new path", 409
        ) from None
    except OSError:
        raise DatasetError(
            "dataset_artifact_write_failed", "Dataset artifact cannot be written", 503
        ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
