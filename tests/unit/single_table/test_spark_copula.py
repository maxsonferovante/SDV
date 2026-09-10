from unittest.mock import Mock, patch
import pandas as pd
import pytest

from sdv.single_table.copulas import GaussianCopulaSynthesizer


class MockSparkDataFrame:
    def __init__(self):
        self.count = Mock()
        self.toPandas = Mock()
        self.columns = ['col1']

MockSparkDataFrame.__name__ = 'DataFrame'
MockSparkDataFrame.__module__ = 'pyspark.sql.dataframe'


class TestSparkCopula:

    @patch('sdv.single_table.copulas.warn_missing_numerical_distributions')
    def test_fit_spark(self, mock_warn):
        """Test that _fit handles Spark DataFrames by count and toPandas conversion."""
        # Setup
        metadata = Mock()
        metadata.to_dict.return_value = {
            'METADATA_SPEC_VERSION': 'V1',
            'tables': {
                'table': {
                    'columns': {
                        'col1': {
                            'sdtype': 'numerical'
                        }
                    }
                }
            }
        }
        metadata.column_relationships = []
        
        synthesizer = GaussianCopulaSynthesizer(metadata)
        
        spark_df = MockSparkDataFrame()
        spark_df.count.return_value = 100
        
        pdf = pd.DataFrame({'col1': [1.0, 2.0, 3.0]})
        spark_df.toPandas.return_value = pdf
        
        with patch.object(synthesizer, '_get_numerical_distributions') as mock_get_dist, \
             patch.object(synthesizer, '_initialize_model') as mock_init_model, \
             patch.object(synthesizer, '_fit_model') as mock_fit_model:
             
            mock_get_dist.return_value = {'col1': 'norm'}
            mock_model = Mock()
            mock_init_model.return_value = mock_model
            
            # Run
            synthesizer._fit(spark_df)
            
            # Assert
            assert synthesizer._num_rows == 100
            spark_df.count.assert_called_once()
            spark_df.toPandas.assert_called_once()
            mock_fit_model.assert_called_once_with(pdf)
