from unittest.mock import Mock, patch
import pandas as pd
import pytest

from sdv.metadata import Metadata
from sdv.multi_table.hma import HMASynthesizer


class MockSparkDataFrame:
    pass

MockSparkDataFrame.__name__ = 'DataFrame'
MockSparkDataFrame.__module__ = 'pyspark.sql.dataframe'


class TestSparkHMA:

    @patch('sdv.multi_table.hma.HMASynthesizer._augment_table_spark')
    def test_augment_tables_routing(self, mock_augment_spark):
        """Test that _augment_tables routes Spark DataFrames to _augment_table_spark."""
        # Setup
        metadata_dict = {
            'METADATA_SPEC_VERSION': 'V1',
            'tables': {
                'parent': {
                    'primary_key': 'parent_id',
                    'columns': {
                        'parent_id': {'sdtype': 'id'}
                    }
                },
                'child': {
                    'primary_key': 'child_id',
                    'columns': {
                        'child_id': {'sdtype': 'id'},
                        'parent_id': {'sdtype': 'id'},
                        'col': {'sdtype': 'numerical'}
                    }
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'parent',
                    'parent_primary_key': 'parent_id',
                    'child_table_name': 'child',
                    'child_foreign_key': 'parent_id'
                }
            ]
        }
        metadata = Metadata.load_from_dict(metadata_dict)
        
        synthesizer = HMASynthesizer(metadata)
        synthesizer._spark_mode = True
        
        processed_data = {
            'parent': MockSparkDataFrame()
        }

        # Run
        synthesizer._augment_tables(processed_data)

        # Assert
        mock_augment_spark.assert_called_once()

    @patch('pyspark.sql.SparkSession.builder')
    def test_get_extension_spark(self, mock_builder):
        """Test that _get_extension_spark extracts metadata details and sets up applyInPandas UDF."""
        # Setup
        metadata_dict = {
            'METADATA_SPEC_VERSION': 'V1',
            'tables': {
                'parent': {
                    'primary_key': 'parent_id',
                    'columns': {
                        'parent_id': {'sdtype': 'id'}
                    }
                },
                'child': {
                    'primary_key': 'child_id',
                    'columns': {
                        'child_id': {'sdtype': 'id'},
                        'parent_id': {'sdtype': 'id'},
                        'col': {'sdtype': 'numerical'}
                    }
                }
            },
            'relationships': [
                {
                    'parent_table_name': 'parent',
                    'parent_primary_key': 'parent_id',
                    'child_table_name': 'child',
                    'child_foreign_key': 'parent_id'
                }
            ]
        }
        metadata = Metadata.load_from_dict(metadata_dict)
        
        synthesizer = HMASynthesizer(metadata)
        synthesizer._table_synthesizers = {
            'child': Mock()
        }
        synthesizer._table_parameters = {
            'child': {}
        }
        
        child_table = Mock()
        child_field = Mock()
        child_field.name = 'parent_id'
        from pyspark.sql.types import StringType
        child_field.dataType = StringType()
        child_table.schema.fields = [child_field]
        
        dummy_pdf = pd.DataFrame({'parent_id': ['p1'], 'col': [1.0]})
        child_table.limit.return_value.toPandas.return_value = dummy_pdf
        
        mock_spark = Mock()
        mock_builder.getOrCreate.return_value = mock_spark
        
        mock_synthesizer_inst = Mock()
        mock_synthesizer_inst._get_parameters.return_value = {'mean': 1.0}
        synthesizer._synthesizer = Mock(return_value=mock_synthesizer_inst)
        
        # Run
        synthesizer._get_extension_spark('child', child_table, 'parent_id')
        
        # Assert
        child_table.groupby.assert_called_once_with('parent_id')
        child_table.groupby.return_value.applyInPandas.assert_called_once()
