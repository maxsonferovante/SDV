from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from sdv.single_table.copulagan import CopulaGANSynthesizer


class MockSparkDataFrame:
    def __init__(self):
        self.count = Mock()
        self.toPandas = Mock()
        self.columns = ['col1', 'col2']

MockSparkDataFrame.__name__ = 'DataFrame'
MockSparkDataFrame.__module__ = 'pyspark.sql.dataframe'


class TestSparkCopulaGAN:

    @patch('sdv.single_table.copulagan.warn_missing_numerical_distributions')
    def test_fit_spark_collects_to_pandas(self, mock_warn):
        """Test that _fit collects a Spark DataFrame to pandas before GaussianNormalizer."""
        # Setup
        metadata = Mock()
        metadata.to_dict.return_value = {
            'METADATA_SPEC_VERSION': 'V1',
            'tables': {
                'table': {
                    'columns': {
                        'col1': {'sdtype': 'numerical'},
                        'col2': {'sdtype': 'categorical'},
                    }
                }
            }
        }
        metadata.column_relationships = []

        synthesizer = CopulaGANSynthesizer(metadata)

        pdf = pd.DataFrame({'col1': [1.0, 2.0, 3.0], 'col2': ['A', 'B', 'C']})
        spark_df = MockSparkDataFrame()
        spark_df.toPandas.return_value = pdf

        with patch.object(synthesizer, '_create_gaussian_normalizer_config', return_value={}) as mock_cfg, \
             patch('rdt.HyperTransformer') as mock_ht_cls, \
             patch.object(synthesizer.__class__.__bases__[0], '_fit') as mock_ctgan_fit:

            mock_ht = Mock()
            mock_ht.fit_transform.return_value = pdf
            mock_ht_cls.return_value = mock_ht

            synthesizer._fit(spark_df)

            # toPandas must be called to collect before GaussianNormalizer
            spark_df.toPandas.assert_called_once()
            # GaussianNormalizer config must receive the collected pandas DF
            mock_cfg.assert_called_once_with(pdf)

    @patch('sdv.single_table.copulagan.SparkSession')
    def test_sample_spark_generates_on_driver(self, mock_spark_cls):
        """_sample_spark generates on driver (PyTorch cannot run in Spark workers) and wraps as Spark DF."""
        synthesizer = MagicMock()
        synthesizer._table_name = 'table'
        synthesizer.metadata.tables.get.return_value = None  # no PK

        pdf = pd.DataFrame({'col1': [1.0, 2.0, 3.0]})
        synthesizer._model.sample.return_value = pdf
        synthesizer._gaussian_normalizer_hyper_transformer.reverse_transform.return_value = pdf
        synthesizer._data_processor._data_processor.reverse_transform.return_value = pdf

        mock_spark = Mock()
        mock_spark_cls.builder.getOrCreate.return_value = mock_spark
        mock_spark_df = Mock()
        mock_spark.createDataFrame.return_value = mock_spark_df

        result = CopulaGANSynthesizer._sample_spark(synthesizer, num_rows=3)

        # Model sample must be called on driver (not inside worker)
        synthesizer._model.sample.assert_called_once_with(3)
        # GN reverse_transform applied driver-side
        synthesizer._gaussian_normalizer_hyper_transformer.reverse_transform.assert_called_once()
        # Result is a Spark DF created from pandas output
        mock_spark.createDataFrame.assert_called_once()
        assert result == mock_spark_df
