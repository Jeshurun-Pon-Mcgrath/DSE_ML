"""All data constructed here is synthetic and remains inside temporary test directories."""

from pathlib import Path

import pytest

from riskbyte_ml.data.config import DatasetConfig


@pytest.fixture
def synthetic_csv() -> Path:
    return Path(__file__).parent / "fixtures" / "synthetic_transactions.csv"


@pytest.fixture
def synthetic_config(synthetic_csv) -> DatasetConfig:
    return DatasetConfig(
        name="SYNTHETIC test transactions",
        source_path=synthetic_csv,
        mode="explicit",
        target_column="label",
        positive_class="fraud",
        negative_class="legit",
        timestamp_column="occurred",
        amount_column="value",
        identifier_columns=("event_key", "party_key"),
    )


@pytest.fixture
def csv_factory(tmp_path):
    def create(content: str | bytes, **changes):
        path = tmp_path / "synthetic.csv"
        path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        return DatasetConfig(name="SYNTHETIC test data", source_path=path, **changes)

    return create
