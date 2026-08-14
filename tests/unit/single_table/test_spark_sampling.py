from unittest.mock import Mock, patch
import pandas as pd
import pytest

from sdv.single_table.base import BaseSingleTableSynthesizer


class TestSparkSampling:

    @patch('pyspark.sql.SparkSession.builder')
    def test_sample_spark(self, mock_builder):
        """Test that _sample_spark constructs Spark range, broadcasts, and calls mapInPandas."""
        # Setup
        synthesizer = Mock()
        synthesizer._model = Mock()
        
        dummy_model_sample = pd.DataFrame({'a': [1]})
        dummy_reversed = pd.DataFrame({'a': [1.0]})
        synthesizer._model.sample.return_value = dummy_model_sample
        
        synthesizer._data_processor._data_processor.reverse_transform.return_value = dummy_reversed
        synthesizer._data_processor._pandas_to_spark_schema.return_value = "dummy_schema"
        
        mock_spark = Mock()
        mock_builder.getOrCreate.return_value = mock_spark
        
        mock_skeleton = Mock()
        mock_spark.range.return_value = mock_skeleton
        
        # Run
        BaseSingleTableSynthesizer._sample_spark(synthesizer, num_rows=150000)
        
        # Assert
        mock_spark.range.assert_called_once_with(0, 150000, numPartitions=2)
        mock_skeleton.mapInPandas.assert_called_once()
        synthesizer._data_processor._pandas_to_spark_schema.assert_called_once_with(dummy_reversed)
