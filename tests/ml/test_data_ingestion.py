import hashlib

import pytest
from pydantic import ValidationError

from riskbyte_ml.data.config import DatasetConfig, load_config
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.ingestion import load_csv
from riskbyte_ml.data.inspection import inspect_schema


def test_binary_codec_is_not_a_csv_text_encoding(synthetic_config):
    config = synthetic_config.model_copy(update={"encoding": "base64_codec"})
    with pytest.raises(DatasetError) as error:
        load_csv(config)
    assert error.value.code == "dataset_encoding_unsupported"


@pytest.mark.parametrize("delimiter", ["\r", "\n", "\x00"])
def test_invalid_delimiter_is_rejected_at_configuration(delimiter):
    with pytest.raises(ValidationError):
        DatasetConfig(delimiter=delimiter)


def test_valid_csv_preserves_schema_strings_and_source(synthetic_config):
    original = synthetic_config.source_path.read_bytes()
    dataset = load_csv(synthetic_config)
    assert dataset.frame.shape == (12, 7)
    assert dataset.frame.loc[0, "event_key"] == "0001"
    assert dataset.frame.loc[7, "channel"] == ""
    assert set(dataset.frame.dtypes.astype(str)) == {"string"}
    assert synthetic_config.source_path.read_bytes() == original


def test_fingerprint_tracks_exact_bytes_and_not_path(synthetic_config, tmp_path):
    original = load_csv(synthetic_config)
    copied = tmp_path / "copy.csv"
    copied.write_bytes(synthetic_config.source_path.read_bytes())
    duplicate = load_csv(synthetic_config.model_copy(update={"source_path": copied}))
    assert original.fingerprint == duplicate.fingerprint
    assert original.fingerprint == "sha256:" + hashlib.sha256(copied.read_bytes()).hexdigest()
    copied.write_bytes(copied.read_bytes().replace(b",10,", b",11,", 1))
    assert (
        load_csv(synthetic_config.model_copy(update={"source_path": copied})).fingerprint
        != original.fingerprint
    )


@pytest.mark.parametrize(
    "content,code",
    [
        ("", "dataset_empty"),
        ("   \n", "dataset_empty"),
        ("a,b\n", "dataset_empty"),
        ("a,a\n1,2\n", "dataset_duplicate_columns"),
        ("a, \n1,2\n", "dataset_columns_invalid"),
        ('a,b\n1,"unterminated', "dataset_csv_malformed"),
        ("a,b\n1,2,3\n", "dataset_csv_malformed"),
        ("a,b\n1\n", "dataset_csv_malformed"),
        ("a,b\n\n", "dataset_csv_malformed"),
        ("a\n\x00\n", "dataset_csv_malformed"),
    ],
)
def test_invalid_csv_is_rejected_without_silent_repair(csv_factory, content, code):
    with pytest.raises(DatasetError) as error:
        load_csv(csv_factory(content))
    assert error.value.code == code


def test_missing_file_is_sanitized(tmp_path):
    with pytest.raises(DatasetError) as error:
        load_csv(DatasetConfig(source_path=tmp_path / "private-financial.csv"))
    assert error.value.code == "dataset_file_missing"
    assert str(tmp_path) not in error.value.message


@pytest.mark.parametrize("changes", [{"max_bytes": 3}, {"max_rows": 1}, {"max_columns": 1}])
def test_resource_limits(csv_factory, changes):
    with pytest.raises(DatasetError, match="limit"):
        load_csv(csv_factory("a,b\n1,2\n3,4\n", **changes))


def test_explicit_encoding_and_delimiter(csv_factory):
    data = load_csv(
        csv_factory("key;city\n001;München\n".encode("latin-1"), encoding="latin-1", delimiter=";")
    )
    assert data.frame.loc[0, "city"] == "München"
    assert data.frame.loc[0, "key"] == "001"


@pytest.mark.parametrize(
    "encoding,content,code",
    [
        ("not-an-encoding", b"a\n1\n", "dataset_encoding_unsupported"),
        ("utf-8", b"a\n\xff\n", "dataset_encoding_invalid"),
    ],
)
def test_bad_encoding(csv_factory, encoding, content, code):
    with pytest.raises(DatasetError) as error:
        load_csv(csv_factory(content, encoding=encoding))
    assert error.value.code == code


def test_quoted_multiline_records_and_bom(csv_factory):
    dataset = load_csv(csv_factory('\ufeffkey,text\n001,"line one\nline two"\n'))
    assert dataset.frame.columns.tolist() == ["key", "text"]
    assert dataset.frame.loc[0, "text"] == "line one\nline two"


def test_missing_selected_target_is_an_error(csv_factory):
    config = csv_factory(
        "x\n1\n",
        mode="explicit",
        target_column="absent",
        positive_class="bad",
        negative_class="good",
    )
    with pytest.raises(DatasetError) as error:
        load_csv(config)
    assert error.value.code == "dataset_target_missing"


@pytest.mark.parametrize(
    "settings",
    [
        {"target_column": "label"},
        {"mode": "explicit", "target_column": "label"},
        {
            "mode": "explicit",
            "target_column": "label",
            "positive_class": "1",
            "negative_class": "1",
        },
        {"positive_class": "1"},
        {"percentiles": (float("nan"),)},
        {"zscore_threshold": 0},
        {"unique_identifier_columns": ("id",)},
        {"timestamp_kind": "elapsed"},
        {"timestamp_unit": "s"},
        {"timezone": "Invalid/Timezone"},
        {"unknown_field": True},
    ],
)
def test_config_requires_explicit_unambiguous_semantics(settings):
    with pytest.raises(ValidationError):
        DatasetConfig(**settings)


def test_config_source_is_relative_to_config_file(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text('{"source_path":"relative.csv"}', encoding="utf-8")
    assert load_config(path).source_path == tmp_path / "relative.csv"


def test_inspection_candidates_never_select_roles(synthetic_csv):
    dataset = load_csv(DatasetConfig(source_path=synthetic_csv))
    schema = inspect_schema(dataset)
    assert "label" in [candidate.column for candidate in schema.potential_targets]
    assert "occurred" in schema.datetime_like_columns
    assert "occurred" in [candidate.column for candidate in schema.potential_timestamps]
    assert schema.target.available is False and schema.target.column is None
    assert schema.timestamp.available is False and schema.timestamp.column is None
    assert all(candidate.inferred for candidate in schema.potential_timestamps)


def test_schema_cardinality_near_constant_boolean_and_no_implicit_na(csv_factory):
    rows = "\n".join(
        f"K{i},same,{str(i % 2 == 0).lower()},{'base' if i < 99 else 'rare'},NA" for i in range(100)
    )
    config = csv_factory(
        "custom_key,fixed,switch,nearly,country\n" + rows + "\n", high_cardinality_threshold=50
    )
    schema = inspect_schema(load_csv(config))
    assert "custom_key" in schema.high_cardinality_categorical_columns
    assert "custom_key" in [candidate.column for candidate in schema.potential_identifiers]
    assert "fixed" in schema.constant_columns
    assert "nearly" in schema.near_constant_columns
    assert "switch" in schema.boolean_columns
    assert (
        next(column for column in schema.columns if column.name == "country").non_missing_count
        == 100
    )
