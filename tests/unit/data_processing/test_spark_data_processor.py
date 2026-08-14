from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import pytest

from sdv.data_processing.spark_data_processor import SparkDataProcessor


class TestSparkDataProcessor:

    def test_fit(self):
        """Test that fit limits and converts data to pandas for driver-side fit."""
        # Setup
        metadata = Mock()
        metadata.column_relationships = []
        spark_df = Mock()
        dummy_pdf = pd.DataFrame({'col1': [1, 2]})
        spark_df.limit.return_value.toPandas.return_value = dummy_pdf

        processor = SparkDataProcessor(metadata)

        # Run
        with patch.object(processor._data_processor, 'fit') as mock_fit:
            processor.fit(spark_df)

            # Assert
            spark_df.limit.assert_called_once_with(10000)
            spark_df.limit.return_value.toPandas.assert_called_once_with()
            mock_fit.assert_called_once_with(dummy_pdf)
            assert processor.fitted is True

    def test_transform(self):
        """Test transform maps partitions and returns Spark DataFrame."""
        # Setup
        metadata = Mock()
        metadata.column_relationships = []
        spark_df = Mock()
        dummy_pdf = pd.DataFrame({'col1': [1]})
        dummy_transformed = pd.DataFrame({'col1': [1.0]})

        spark_df.limit.return_value.toPandas.return_value = dummy_pdf
        spark_df.sql_ctx.sparkSession.sparkContext = Mock()

        processor = SparkDataProcessor(metadata)
        processor.fitted = True

        # Run
        with patch.object(processor._data_processor, 'transform', return_value=dummy_transformed):
            processor.transform(spark_df)

            # Assert
            spark_df.limit.assert_called_once_with(1)
            spark_df.mapInPandas.assert_called_once()

    def test_reverse_transform(self):
        """Test reverse_transform maps partitions and returns Spark DataFrame."""
        # Setup
        metadata = Mock()
        metadata.column_relationships = []
        spark_df = Mock()
        dummy_pdf = pd.DataFrame({'col1': [1.0]})
        dummy_reversed = pd.DataFrame({'col1': [1]})

        spark_df.limit.return_value.toPandas.return_value = dummy_pdf
        spark_df.sql_ctx.sparkSession.sparkContext = Mock()

        processor = SparkDataProcessor(metadata)
        processor.fitted = True

        # Run
        with patch.object(processor._data_processor, 'reverse_transform', return_value=dummy_reversed):
            processor.reverse_transform(spark_df)

            # Assert
            spark_df.limit.assert_called_once_with(1)
            spark_df.mapInPandas.assert_called_once()
