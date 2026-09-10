import numpy as np
import pandas as pd
from sdv.data_processing.data_processor import DataProcessor
from pyspark.sql.types import StructType, StructField, DoubleType, LongType, StringType, BooleanType, TimestampType


class SparkDataProcessor:
    """Spark-compatible wrapper around DataProcessor to handle PySpark DataFrames."""

    def __init__(self, metadata, enforce_rounding=True, enforce_min_max_values=True, locales=['en_US']):
        self.metadata = metadata
        self.enforce_rounding = enforce_rounding
        self.enforce_min_max_values = enforce_min_max_values
        self.locales = locales
        self._data_processor = DataProcessor(
            metadata=self.metadata,
            enforce_rounding=self.enforce_rounding,
            enforce_min_max_values=self.enforce_min_max_values,
            locales=self.locales,
        )
        self.fitted = False

    def prepare_for_fitting(self, data):
        """Prepare the DataProcessor using a representative sample collected to the driver node."""
        sample_pdf = data.limit(10000).toPandas()
        self._data_processor.prepare_for_fitting(sample_pdf)

    def fit(self, data):
        """Fit the DataProcessor using a representative sample collected to the driver node."""
        # Fit on a driver-side sample (up to 10,000 rows is statistically representative for RDT parameters)
        sample_pdf = data.limit(10000).toPandas()
        self._data_processor.fit(sample_pdf)
        self.fitted = True

    def update_transformers(self, column_name_to_transformer):
        """Update transformers configuration and mark fitted as False."""
        self._data_processor.update_transformers(column_name_to_transformer)
        self.fitted = False

    def _pandas_to_spark_schema(self, pdf):
        """Convert a Pandas DataFrame schema to a PySpark StructType schema."""
        
        fields = []
        for col in pdf.columns:
            dtype = pdf[col].dtype
            if np.issubdtype(dtype, np.integer):
                spark_type = LongType()
            elif np.issubdtype(dtype, np.floating):
                spark_type = DoubleType()
            elif dtype == bool:
                spark_type = BooleanType()
            elif np.issubdtype(dtype, np.datetime64):
                spark_type = TimestampType()
            else:
                spark_type = StringType()
            fields.append(StructField(col, spark_type, True))
        return StructType(fields)

    def transform(self, data, is_condition=False):
        """Transform a PySpark DataFrame using mapInPandas across executors."""
        # 1. Perform a dry-run on driver to determine the Spark output schema
        dummy_pdf = data.limit(1).toPandas()
        dummy_transformed = self._data_processor.transform(dummy_pdf, is_condition=is_condition)
        if dummy_transformed.index.name is not None:
            dummy_transformed = dummy_transformed.reset_index(drop=False)
        schema = self._pandas_to_spark_schema(dummy_transformed)

        # 2. Broadcast the driver-fitted preprocessor to all executors
        sc = data.sparkSession.sparkContext
        dp_broadcast = sc.broadcast(self._data_processor)

        # 3. Apply transformation in parallel
        def transform_partition(iterator):
            local_dp = dp_broadcast.value
            for pdf in iterator:
                transformed = local_dp.transform(pdf, is_condition=is_condition)
                if transformed.index.name is not None:
                    transformed = transformed.reset_index(drop=False)
                yield transformed

        return data.mapInPandas(transform_partition, schema)

    def reverse_transform(self, data, reset_keys=False, conditions=None):
        """Reverse transform data using the inner data processor.

        Accepts both pandas DataFrames (e.g., from child synthesizers during HMA sampling)
        and PySpark DataFrames (distributed path).
        """
        if isinstance(data, pd.DataFrame):
            # Pandas path: used by child synthesizers during HMA hierarchical sampling
            return self._data_processor.reverse_transform(
                data, reset_keys=reset_keys, conditions=conditions
            )

        # Spark path: distributed reverse transform via mapInPandas
        # ponytail: Sequence-based key generation in parallel partitions will collision. Offset coordination or native Spark ID generation (monotonically_increasing_id) is the upgrade path.
        dummy_pdf = data.limit(1).toPandas()
        dummy_reversed = self._data_processor.reverse_transform(
            dummy_pdf, reset_keys=reset_keys, conditions=conditions
        )
        schema = self._pandas_to_spark_schema(dummy_reversed)

        sc = data.sparkSession.sparkContext
        dp_broadcast = sc.broadcast(self._data_processor)

        def reverse_transform_partition(iterator):
            local_dp = dp_broadcast.value
            for pdf in iterator:
                yield local_dp.reverse_transform(
                    pdf, reset_keys=reset_keys, conditions=conditions
                )

        return data.mapInPandas(reverse_transform_partition, schema)

    def reset_sampling(self):
        self._data_processor.reset_sampling()

    def __getattr__(self, name):
        # Guard against infinite recursion during unpickling (e.g., gc.enable() callback)
        # when _data_processor hasn't been set yet on the instance.
        if name == '_data_processor':
            raise AttributeError(name)
        return getattr(self._data_processor, name)
