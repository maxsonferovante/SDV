from unittest.mock import Mock, patch
import pytest
from sdv.evaluation.evaluation import evaluate_quality, run_diagnostic
from sdv.metadata.metadata import Metadata


class MockSparkDataFrame:
    def __init__(self):
        self.columns = ['col1', 'col2']


MockSparkDataFrame.__name__ = 'DataFrame'
MockSparkDataFrame.__module__ = 'pyspark.sql.dataframe'


class TestSparkEvaluation:

    @patch('sdv.evaluation.spark_quality.evaluate_quality_spark')
    def test_evaluate_quality_spark_single_table(self, mock_evaluate_quality_spark):
        """Test that evaluate_quality wraps single-table Spark DataFrame in a dict and routes to evaluate_quality_spark."""
        # Setup
        real_df = MockSparkDataFrame()
        synth_df = MockSparkDataFrame()
        metadata = Metadata()
        metadata.add_table('table')
        metadata.add_column('col1', 'table', sdtype='numerical')
        metadata.add_column('col2', 'table', sdtype='categorical')

        # Run
        evaluate_quality(real_df, synth_df, metadata, verbose=False)

        # Assert
        mock_evaluate_quality_spark.assert_called_once_with(
            {'table': real_df},
            {'table': synth_df},
            metadata,
            False
        )

    @patch('sdv.evaluation.spark_quality.evaluate_quality_spark')
    def test_evaluate_quality_spark_multi_table(self, mock_evaluate_quality_spark):
        """Test that evaluate_quality directly routes a dict of Spark DataFrames to evaluate_quality_spark."""
        # Setup
        real_data = {'table1': MockSparkDataFrame()}
        synth_data = {'table1': MockSparkDataFrame()}
        metadata = Metadata()
        metadata.add_table('table1')
        metadata.add_column('col1', 'table1', sdtype='numerical')

        # Run
        evaluate_quality(real_data, synth_data, metadata, verbose=True)

        # Assert
        mock_evaluate_quality_spark.assert_called_once_with(
            real_data,
            synth_data,
            metadata,
            True
        )

    def test_run_diagnostic_spark_single_table_raises_not_implemented(self):
        """Test that run_diagnostic raises NotImplementedError for single-table Spark inputs."""
        # Setup
        real_df = MockSparkDataFrame()
        synth_df = MockSparkDataFrame()
        metadata = Mock()

        # Run & Assert
        with pytest.raises(NotImplementedError, match='Diagnostic evaluation is not supported for PySpark DataFrames.'):
            run_diagnostic(real_df, synth_df, metadata)

    def test_run_diagnostic_spark_multi_table_raises_not_implemented(self):
        """Test that run_diagnostic raises NotImplementedError for multi-table Spark inputs."""
        # Setup
        real_data = {'table1': MockSparkDataFrame()}
        synth_data = {'table1': MockSparkDataFrame()}
        metadata = Mock()

        # Run & Assert
        with pytest.raises(NotImplementedError, match='Diagnostic evaluation is not supported for PySpark DataFrames.'):
            run_diagnostic(real_data, synth_data, metadata)
