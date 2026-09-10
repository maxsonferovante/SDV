from unittest.mock import Mock, patch
import pandas as pd
import pytest

from sdv.metadata import Metadata
from sdv.multi_table.hma import HMASynthesizer


class TestSparkHMASampling:

    @patch('sdv.multi_table.hma.HMASynthesizer._sample_children_spark')
    def test_sample_children_routing(self, mock_sample_children_spark):
        """Test that _sample_children routes Spark DataFrames to _sample_children_spark."""
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
        
        sampled_data = {
            'parent': Mock()
        }

        # Run
        synthesizer._sample_children('parent', sampled_data)

        # Assert
        mock_sample_children_spark.assert_called_once_with('parent', sampled_data, 1.0)

    @patch('pyspark.sql.SparkSession.builder')
    def test_sample_children_spark_flow(self, mock_builder):
        """Test that _sample_children_spark executes the Spark pipeline to group and applyInPandas UDF."""
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
        
        parent_table = Mock()
        dummy_parent_pdf = pd.DataFrame({
            'parent_id': ['p1'],
            '__child__parent_id__num_rows': [2.0]
        })
        parent_table.limit.return_value.toPandas.return_value = dummy_parent_pdf
        parent_table.sql_ctx.sparkSession.sparkContext = Mock()
        
        sampled_data = {
            'parent': parent_table
        }
        
        synthesizer._table_sizes = {'child': 10}
        synthesizer._max_child_rows = {'__child__parent_id__num_rows': 5}
        synthesizer._min_child_rows = {'__child__parent_id__num_rows': 0}
        
        mock_spark = Mock()
        mock_builder.getOrCreate.return_value = mock_spark
        
        child_synth = Mock()
        child_synth._data_processor._pandas_to_spark_schema.return_value = "child_schema"
        synthesizer._table_synthesizers = {'child': child_synth}
        
        dummy_child_rows = pd.DataFrame({'child_id': ['c1'], 'col': [1.0]})
        
        # Mock enforce size to avoid join complications in mock tests
        with patch.object(synthesizer, '_enforce_table_size_spark'):
            with patch.object(synthesizer, '_recreate_child_synthesizer', return_value=Mock()) as mock_recreate:
                with patch.object(synthesizer, '_sample_rows', return_value=dummy_child_rows):
                    
                    # Run
                    synthesizer._sample_children_spark('parent', sampled_data)
                    
                    # Assert
                    mock_recreate.assert_called_once()
                    parent_table.groupby.assert_called_once_with('parent_id')
                    parent_table.groupby.return_value.applyInPandas.assert_called_once()
