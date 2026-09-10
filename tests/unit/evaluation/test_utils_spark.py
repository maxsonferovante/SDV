"""Unit tests for Spark-compatible evaluation utils."""
import re
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sdv.evaluation.utils import (
    get_combination_overlap,
    get_pii_overlap,
    print_referential_integrity,
)
from sdv.metadata import Metadata

NO_OVERLAP_MESSAGE = (
    '✅ The synthetic data does not contain any of the same combinations from the real data'
)
FEW_OVERLAP_MESSAGE = (
    '⚠️ The synthetic data contains a few of the same combinations as the real data. '
    'This might be due to random chance.'
)
SIGNIFICANT_OVERLAP_MESSAGE = (
    '❌ The synthetic data contains a significant number of the same combinations as the real '
    'data. This might be due to a small number of possible combinations, a large sample of '
    'synthetic data, or a misconfiguration in your synthesizer.'
)
NO_PII_OVERLAP_MESSAGE = '✅ The synthetic data does not contain any PII values from the real data'
FEW_PII_OVERLAP_MESSAGE = (
    '⚠️ The synthetic data contains a few PII values from the real data. '
    'This might be due to random chance.'
)
SIGNIFICANT_PII_OVERLAP_MESSAGE = (
    '❌ The synthetic data contains a significant number of the same PII values of as the real '
    'data. This might be due to a small number of possible PII values, a large sample of '
    'synthetic data, or a misconfiguration in your synthesizer.'
)


class MockSparkColumn:
    """Mock PySpark Column expression."""

    def __init__(self, name):
        self._name = name

    def cast(self, dtype):
        return self

    def alias(self, name):
        self._name = name
        return self


class MockSparkDataFrame:
    """A mock PySpark DataFrame that simulates Spark SQL operations for testing."""

    def __init__(self, data_dict):
        """Initialize with a dict of column_name -> list of values."""
        self.columns = list(data_dict.keys())
        self._data = data_dict
        self._num_rows = len(next(iter(data_dict.values()))) if data_dict else 0

    def select(self, cols_or_exprs):
        """Mock select - returns a new MockSparkDataFrame with only selected columns."""
        new_data = {}
        if isinstance(cols_or_exprs, list):
            for item in cols_or_exprs:
                if isinstance(item, MockSparkColumn):
                    col_name = item._name
                    if col_name in self._data:
                        new_data[col_name] = [str(v) for v in self._data[col_name]]
                elif isinstance(item, str):
                    if item in self._data:
                        new_data[item] = list(self._data[item])
        elif isinstance(cols_or_exprs, str):
            if cols_or_exprs in self._data:
                new_data[cols_or_exprs] = list(self._data[cols_or_exprs])
        return MockSparkDataFrame(new_data)

    def distinct(self):
        """Mock distinct - returns unique rows."""
        if not self._data or self._num_rows == 0:
            return MockSparkDataFrame(self._data)

        rows = []
        for i in range(self._num_rows):
            row = tuple(self._data[col][i] for col in self.columns)
            rows.append(row)
        unique_rows = list(dict.fromkeys(rows))

        new_data = {col: [] for col in self.columns}
        for row in unique_rows:
            for j, col in enumerate(self.columns):
                new_data[col].append(row[j])
        return MockSparkDataFrame(new_data)

    def intersect(self, other):
        """Mock intersect - returns rows common to both DataFrames."""
        real_rows = set()
        for i in range(self._num_rows):
            row = tuple(self._data[col][i] for col in self.columns)
            real_rows.add(row)

        other_rows = set()
        for i in range(other._num_rows):
            row = tuple(other._data[col][i] for col in other.columns)
            other_rows.add(row)

        common = real_rows & other_rows
        new_data = {col: [] for col in self.columns}
        for row in common:
            for j, col in enumerate(self.columns):
                new_data[col].append(row[j])
        return MockSparkDataFrame(new_data)

    def union(self, other):
        """Mock union - returns all rows from both DataFrames."""
        new_data = {col: list(self._data[col]) for col in self.columns}
        for col in other.columns:
            if col in new_data:
                new_data[col].extend(other._data[col])
        return MockSparkDataFrame(new_data)

    def count(self):
        """Mock count - returns number of rows."""
        return self._num_rows

    def orderBy(self, *args, **kwargs):
        """Mock orderBy - returns self (for chaining with limit)."""
        return self

    def limit(self, n):
        """Mock limit - returns first n rows."""
        new_data = {col: self._data[col][:n] for col in self.columns}
        return MockSparkDataFrame(new_data)

    def collect(self):
        """Mock collect - returns list of Row-like dicts."""
        rows = []
        for i in range(self._num_rows):
            row = {col: self._data[col][i] for col in self.columns}
            rows.append(row)
        return rows


MockSparkDataFrame.__name__ = 'DataFrame'
MockSparkDataFrame.__module__ = 'pyspark.sql.dataframe'


def _install_mock_pyspark():
    """Install a mock pyspark module so F.col() works without a SparkContext."""
    mock_pyspark = ModuleType('pyspark')
    mock_sql = ModuleType('pyspark.sql')
    mock_functions = MagicMock()

    # F.col("name") returns a MockSparkColumn
    mock_functions.col = lambda name: MockSparkColumn(name)

    mock_pyspark.sql = mock_sql
    mock_sql.functions = mock_functions

    sys.modules['pyspark'] = mock_pyspark
    sys.modules['pyspark.sql'] = mock_sql
    sys.modules['pyspark.sql.functions'] = mock_functions

    return mock_functions


# Install mock pyspark before any test imports
_mock_functions = _install_mock_pyspark()


def _make_spark_metadata():
    """Return metadata for a parent-child dataset."""
    return Metadata().load_from_dict({
        'tables': {
            'parent': {
                'columns': {'parent_id': {'sdtype': 'id'}},
                'primary_key': 'parent_id',
            },
            'child': {
                'columns': {
                    'child_id': {'sdtype': 'id'},
                    'parent_id': {'sdtype': 'id'},
                },
                'primary_key': 'child_id',
            },
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'parent_primary_key': 'parent_id',
                'child_table_name': 'child',
                'child_foreign_key': 'parent_id',
            }
        ],
    })


def _make_spark_data():
    """Return Spark DataFrames for parent-child with one broken reference."""
    return {
        'parent': MockSparkDataFrame({'parent_id': [0, 1]}),
        'child': MockSparkDataFrame({'child_id': ['A', 'B'], 'parent_id': [0, 9]}),
    }


def _pandas_to_spark_table_dicts(real_table, synthetic_table):
    """Wrap single-table DataFrames in dicts."""
    return {'table': real_table}, {'table': synthetic_table}


# ===== get_combination_overlap Spark tests =====


@pytest.mark.parametrize(
    ('real_values', 'synthetic_values', 'expected_result', 'expected_summary'),
    [
        (['x', 'y'], ['q', 'r'], 0, ('0 (0.0%)', NO_OVERLAP_MESSAGE)),
        (list(range(51)), list(range(50, 100)), 1, ('1 (1.0%)', FEW_OVERLAP_MESSAGE)),
        (list(range(51)), list(range(49, 100)), 2, ('2 (2.0%)', FEW_OVERLAP_MESSAGE)),
        (['x', 'y', 'z'], ['x', 'q'], 1, ('1 (25.0%)', SIGNIFICANT_OVERLAP_MESSAGE)),
    ],
    ids=['none', 'few', 'two_percent_boundary', 'significant'],
)
@patch('sdv.evaluation.utils.is_spark_dataframe', return_value=True)
def test_get_combination_overlap_spark_reports_the_overlap(
    mock_is_spark, capsys, real_values, synthetic_values, expected_result, expected_summary
):
    """Test Spark path reports correct count, percentage and interpretation."""
    real_data, synthetic_data = _pandas_to_spark_table_dicts(
        MockSparkDataFrame({'a': real_values}),
        MockSparkDataFrame({'a': synthetic_values}),
    )
    counts, message = expected_summary

    result = get_combination_overlap(real_data, synthetic_data, 'table', ['a'])

    captured = capsys.readouterr()
    assert result == expected_result
    assert captured.out == f'Number of common combinations: {counts}\n{message}\n'


@patch('sdv.evaluation.utils.is_spark_dataframe', return_value=True)
def test_get_combination_overlap_spark_verbose_false(mock_is_spark, capsys):
    """Test Spark path with verbose=False prints nothing."""
    real_data, synthetic_data = _pandas_to_spark_table_dicts(
        MockSparkDataFrame({'a': ['x']}),
        MockSparkDataFrame({'a': ['x']}),
    )

    result = get_combination_overlap(real_data, synthetic_data, 'table', ['a'], verbose=False)

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ''


@patch('sdv.evaluation.utils.is_spark_dataframe', return_value=True)
def test_get_combination_overlap_spark_empty_tables(mock_is_spark, capsys):
    """Test Spark path with empty tables does not raise."""
    real_data, synthetic_data = _pandas_to_spark_table_dicts(
        MockSparkDataFrame({'a': [], 'b': []}),
        MockSparkDataFrame({'a': [], 'b': []}),
    )

    result = get_combination_overlap(real_data, synthetic_data, 'table', ['a', 'b'])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == f'Number of common combinations: 0 (0.0%)\n{NO_OVERLAP_MESSAGE}\n'


# ===== get_pii_overlap Spark tests =====


@pytest.mark.parametrize(
    ('real_values', 'synthetic_values', 'expected_result', 'expected_summary'),
    [
        (['a', 'b'], ['y', 'z'], 0, ('0 (0.0%)', NO_PII_OVERLAP_MESSAGE)),
        (list(range(51)), list(range(50, 100)), 1, ('1 (1.0%)', FEW_PII_OVERLAP_MESSAGE)),
        (['a', 'b', 'c'], ['a', 'z'], 1, ('1 (25.0%)', SIGNIFICANT_PII_OVERLAP_MESSAGE)),
    ],
    ids=['none', 'few', 'significant'],
)
@patch('sdv.evaluation.utils.is_spark_dataframe', return_value=True)
def test_get_pii_overlap_spark_reports_the_overlap(
    mock_is_spark, capsys, real_values, synthetic_values, expected_result, expected_summary
):
    """Test Spark path reports correct PII overlap."""
    real_data, synthetic_data = _pandas_to_spark_table_dicts(
        MockSparkDataFrame({'ssn': real_values}),
        MockSparkDataFrame({'ssn': synthetic_values}),
    )
    counts, message = expected_summary

    result = get_pii_overlap(real_data, synthetic_data, 'table', 'ssn')

    captured = capsys.readouterr()
    assert result == expected_result
    assert captured.out == f'Number of common data points: {counts}\n{message}\n'


@patch('sdv.evaluation.utils.is_spark_dataframe', return_value=True)
def test_get_pii_overlap_spark_verbose_false(mock_is_spark, capsys):
    """Test Spark path with verbose=False prints nothing."""
    real_data, synthetic_data = _pandas_to_spark_table_dicts(
        MockSparkDataFrame({'ssn': ['a']}),
        MockSparkDataFrame({'ssn': ['a']}),
    )

    result = get_pii_overlap(real_data, synthetic_data, 'table', 'ssn', verbose=False)

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ''


def test_get_pii_overlap_with_invalid_column_name():
    """Test that a non-string PII column name raises an error."""
    real_data, synthetic_data = _pandas_to_spark_table_dicts(
        MockSparkDataFrame({'ssn': ['a']}),
        MockSparkDataFrame({'ssn': ['a']}),
    )

    expected_message = "'pii_column_name' must be a string, got list."
    with pytest.raises(TypeError, match=expected_message):
        get_pii_overlap(real_data, synthetic_data, 'table', ['ssn'])


# ===== print_referential_integrity Spark tests =====


@patch('sdv.evaluation.utils.is_spark_dataframe', return_value=True)
def test_print_referential_integrity_spark_found_and_missing(mock_is_spark, capsys):
    """Test Spark path: valid reference prints match, broken prints failure."""
    metadata = _make_spark_metadata()
    synthetic_data = _make_spark_data()

    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=2)

    captured = capsys.readouterr().out
    assert 'Picking random child row: A\n✅ Found parent row! parent_id: 0\n' in captured
    assert 'Picking random child row: B\n❌ Unable to find the linked parent row\n' in captured


@patch('sdv.evaluation.utils.is_spark_dataframe', return_value=True)
def test_print_referential_integrity_spark_null_fk(mock_is_spark, capsys):
    """Test Spark path: null foreign key is not reported as broken."""
    metadata = _make_spark_metadata()
    synthetic_data = {
        'parent': MockSparkDataFrame({'parent_id': [0, 1]}),
        'child': MockSparkDataFrame({'child_id': ['A', 'B'], 'parent_id': [None, None]}),
    }

    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=2)

    captured = capsys.readouterr().out
    assert captured.count('✅ Foreign key is null; no linked parent row expected') == 2
    assert '❌' not in captured


@patch('sdv.evaluation.utils.is_spark_dataframe', return_value=True)
def test_print_referential_integrity_spark_limits_rows(mock_is_spark, capsys):
    """Test Spark path: only num_rows rows are checked."""
    metadata = _make_spark_metadata()
    synthetic_data = _make_spark_data()

    print_referential_integrity(metadata, synthetic_data, 'child', 'parent_id', num_rows=1)

    captured = capsys.readouterr().out
    assert captured.count('Picking random child row') == 1


@patch('sdv.evaluation.utils.is_spark_dataframe', return_value=True)
def test_print_referential_integrity_spark_too_few_rows(mock_is_spark, capsys):
    """Test Spark path: num_rows lowered to table size with warning."""
    metadata = _make_spark_metadata()
    synthetic_data = _make_spark_data()

    expected_warning = re.escape(
        "The synthetic data contains '2' rows which is less than num_rows: '5'. "
        "Changing num_rows to '2'."
    )
    with pytest.warns(UserWarning, match=expected_warning):
        print_referential_integrity(
            metadata, synthetic_data, 'child', 'parent_id', num_rows=5
        )

    captured = capsys.readouterr().out
    assert captured.count('Picking random child row') == 2


@patch('sdv.evaluation.utils.is_spark_dataframe', return_value=True)
def test_print_referential_integrity_spark_composite_key(mock_is_spark, capsys):
    """Test Spark path with composite foreign key."""
    metadata = Metadata().load_from_dict({
        'tables': {
            'parent': {
                'columns': {'P': {'sdtype': 'id'}, 'Q': {'sdtype': 'id'}},
                'primary_key': ['P', 'Q'],
            },
            'child': {'columns': {'A': {'sdtype': 'id'}, 'B': {'sdtype': 'id'}}},
        },
        'relationships': [
            {
                'parent_table_name': 'parent',
                'parent_primary_key': ['P', 'Q'],
                'child_table_name': 'child',
                'child_foreign_key': ['A', 'B'],
            }
        ],
    })
    synthetic_data = {
        'parent': MockSparkDataFrame({'P': [1], 'Q': ['X']}),
        'child': MockSparkDataFrame({'A': [1], 'B': ['X']}),
    }

    print_referential_integrity(metadata, synthetic_data, 'child', ('B', 'A'), num_rows=1)

    captured = capsys.readouterr().out
    assert '✅ Found parent row! P: 1, Q: X' in captured


# ===== Validation tests (shared pandas/spark) =====


@pytest.mark.parametrize(
    ('table_name', 'column_names', 'expected_error', 'expected_message'),
    [
        (123, ['a'], TypeError, "'table_name' must be a string, got int."),
        ('table', 'a', TypeError, "'column_names' must be a list of strings."),
        ('table', ['a', 2], TypeError, "'column_names' must be a list of strings."),
        ('table', [], ValueError, "'column_names' must contain at least one column name."),
        ('missing', ['a'], ValueError, "Table 'missing' is not present in 'real_data'."),
        (
            'table',
            ['a', 'b', 'c'],
            ValueError,
            "The columns 'b', 'c' are not present in table 'table' of 'real_data'.",
        ),
    ],
    ids=[
        'table_name_not_a_string',
        'column_names_not_a_list',
        'column_names_not_all_strings',
        'no_columns',
        'missing_table',
        'missing_columns',
    ],
)
def test_get_combination_overlap_with_invalid_input(
    table_name, column_names, expected_error, expected_message
):
    """Test that invalid arguments raise errors (pandas or spark path)."""
    real_data, synthetic_data = _pandas_to_spark_table_dicts(
        MockSparkDataFrame({'a': ['x']}),
        MockSparkDataFrame({'a': ['x']}),
    )

    with pytest.raises(expected_error, match=expected_message):
        get_combination_overlap(real_data, synthetic_data, table_name, column_names)
