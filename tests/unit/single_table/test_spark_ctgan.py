from unittest.mock import Mock, patch
import pandas as pd
import pytest

from sdv.single_table.ctgan import CTGANSynthesizer, _fit_spark_pytorch


class MockSparkDataFrame:
    pass

MockSparkDataFrame.__name__ = 'DataFrame'
MockSparkDataFrame.__module__ = 'pyspark.sql.dataframe'


class TestSparkCTGAN:

    @patch('sdv.single_table.ctgan._fit_spark_pytorch')
    def test_fit_routing_to_spark(self, mock_fit_spark_pytorch):
        """Test that CTGANSynthesizer._fit routes Spark DataFrames to _fit_spark_pytorch."""
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
        with patch('sdv.single_table.ctgan.CTGAN', new=Mock):
            synthesizer = CTGANSynthesizer(metadata)
            spark_df = MockSparkDataFrame()

            # Run
            synthesizer._fit(spark_df)

            # Assert
            mock_fit_spark_pytorch.assert_called_once_with(synthesizer, spark_df, Mock)

    @patch('tempfile.mkdtemp')
    @patch('torch.load')
    @patch('sdv.single_table.ctgan.TorchDistributor')
    def test_fit_spark_pytorch_helper(self, mock_distributor_cls, mock_torch_load, mock_mkdtemp):
        """Test that _fit_spark_pytorch writes parquet, configures distributor, and runs training."""
        # Setup
        synthesizer = Mock()
        synthesizer._model_kwargs = {'enable_gpu': False}
        synthesizer._data_processor._hyper_transformer.field_transformers = {}
        
        spark_df = Mock()
        dummy_pdf = pd.DataFrame({'a': [1]})
        spark_df.limit.return_value.toPandas.return_value = dummy_pdf
        
        mock_mkdtemp.return_value = "/tmp/test"
        
        mock_distributor = Mock()
        mock_distributor_cls.return_value = mock_distributor
        
        # Run
        with patch('sdv.single_table.ctgan.detect_discrete_columns', return_value=[]):
            _fit_spark_pytorch(synthesizer, spark_df, Mock)
            
            # Assert
            spark_df.write.parquet.assert_called_once_with("/tmp/test/processed_data.parquet")
            mock_distributor_cls.assert_called_once_with(num_processes=2, use_gpu=False)
            mock_distributor.run.assert_called_once()
            mock_torch_load.assert_called_once_with("/tmp/test/model.pt")
