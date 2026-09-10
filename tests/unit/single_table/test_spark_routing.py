from unittest.mock import Mock
import pandas as pd
import pytest

from sdv._utils import is_spark_dataframe
from sdv.single_table.base import BaseSingleTableSynthesizer, BaseSynthesizer
from sdv.multi_table.base import BaseMultiTableSynthesizer


class MockSparkDataFrame:
    pass

MockSparkDataFrame.__name__ = 'DataFrame'
MockSparkDataFrame.__module__ = 'pyspark.sql.dataframe'


class TestSparkRouting:

    def test_is_spark_dataframe(self):
        """Test the is_spark_dataframe helper function."""
        # Setup
        spark_df = MockSparkDataFrame()
        pandas_df = pd.DataFrame({'a': [1, 2]})
        not_a_df = "string"

        # Assert
        assert is_spark_dataframe(spark_df) is True
        assert is_spark_dataframe(pandas_df) is False
        assert is_spark_dataframe(not_a_df) is False

    def test_base_synthesizer_fit_routing(self):
        """Test that BaseSynthesizer.fit routes Spark DataFrames to _fit_spark."""
        # Setup
        instance = Mock(spec=BaseSynthesizer)
        instance._spark_mode = False
        spark_df = MockSparkDataFrame()

        # Run
        BaseSynthesizer.fit(instance, spark_df)

        # Assert
        assert instance._spark_mode is True
        instance._fit_spark.assert_called_once_with(spark_df)

    def test_base_synthesizer_preprocess_routing(self):
        """Test that BaseSynthesizer.preprocess routes Spark DataFrames to _preprocess_spark."""
        # Setup
        instance = Mock(spec=BaseSynthesizer)
        spark_df = MockSparkDataFrame()

        # Run
        BaseSynthesizer.preprocess(instance, spark_df)

        # Assert
        instance._preprocess_spark.assert_called_once_with(spark_df)

    def test_base_synthesizer_sample_routing(self):
        """Test that BaseSingleTableSynthesizer.sample routes to _sample_spark when _spark_mode is True."""
        # Setup
        instance = Mock(spec=BaseSingleTableSynthesizer)
        instance._spark_mode = True

        # Run
        BaseSingleTableSynthesizer.sample(instance, num_rows=10)

        # Assert
        instance._sample_spark.assert_called_once_with(10, 100, None, None)

    def test_multi_table_synthesizer_fit_routing(self):
        """Test that BaseMultiTableSynthesizer.fit routes dictionaries containing Spark DataFrames."""
        # Setup
        instance = Mock(spec=BaseMultiTableSynthesizer)
        instance._spark_mode = False
        data = {
            'table1': MockSparkDataFrame(),
            'table2': pd.DataFrame({'a': [1, 2]})
        }

        # Run
        BaseMultiTableSynthesizer.fit(instance, data)

        # Assert
        assert instance._spark_mode is True
        instance._fit_spark.assert_called_once_with(data)

    def test_multi_table_synthesizer_preprocess_routing(self):
        """Test that BaseMultiTableSynthesizer.preprocess routes dictionaries containing Spark DataFrames."""
        # Setup
        instance = Mock(spec=BaseMultiTableSynthesizer)
        data = {
            'table1': MockSparkDataFrame(),
            'table2': pd.DataFrame({'a': [1, 2]})
        }

        # Run
        BaseMultiTableSynthesizer.preprocess(instance, data)

        # Assert
        instance._preprocess_spark.assert_called_once_with(data)

    def test_multi_table_synthesizer_sample_routing(self):
        """Test that BaseMultiTableSynthesizer.sample routes to _sample_spark when _spark_mode is True."""
        # Setup
        instance = Mock(spec=BaseMultiTableSynthesizer)
        instance._spark_mode = True

        # Run
        BaseMultiTableSynthesizer.sample(instance, scale=2.0)

        # Assert
        instance._sample_spark.assert_called_once_with(2.0)
