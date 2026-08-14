"""Hierarchical Samplers."""

import logging
import warnings

import numpy as np
import pandas as pd
import random
import time

try:
    import torch
except ImportError:
    torch = None

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.sql.types import StructType, StructField, LongType, IntegerType
except ImportError:
    SparkSession = None
    F = None
    Window = None

from sdv._utils import is_spark_dataframe

LOGGER = logging.getLogger(__name__)


class BaseHierarchicalSampler:
    """Hierarchical sampler mixin.

    Args:
        metadata (sdv.metadata.multi_table.MultiTableMetadata):
            Multi-table metadata representing the data tables that this sampler will be used for.
        table_synthesizers (dict):
            Dictionary mapping each table to a synthesizer. Should be instantiated and passed to
            sampler by the multi-table synthesizer using this sampler.
        table_sizes (dict):
            Dictionary mapping each table to its size. Should be instantiated and passed to the
            sampler by the multi-table synthesizer using this sampler.
    """

    def __init__(self, metadata, table_synthesizers, table_sizes):
        self.metadata = metadata
        self._null_foreign_key_percentages = {}
        self._table_synthesizers = table_synthesizers
        self._table_sizes = table_sizes

    def _recreate_child_synthesizer(self, child_name, parent_name, parent_row):
        """Recreate a child table's synthesizer based on the parent's row.

        Args:
            child_name (str):
                The name of the child table.
            parent_name (str):
                The name of the parent table.
            parent_row (pd.Series):
                The row from the parent table to use for the child synthesizer.
        """
        raise NotImplementedError()

    def _add_foreign_key_columns(self, child_table, parent_table, child_name, parent_name):
        """Add all the foreign keys that connect the child table to the parent table.

        Args:
            child_table (pd.DataFrame):
                The table containing data sampled for the child.
            parent_table (pd.DataFrame):
                The table containing data sampled for the parent.
            parent_name (str):
                The name of the parent table.
            child_name (str):
                The name of the child table.
        """
        raise NotImplementedError()

    def _sample_rows(self, synthesizer, num_rows=None):
        """Sample ``num_rows`` from ``synthesizer``.

        Args:
            synthesizer (copula.multivariate.base):
                The fitted synthesizer for the table.
            num_rows (int):
                Number of rows to sample.

        Returns:
            pandas.DataFrame:
                Sampled rows, shape (, num_rows)
        """
        if num_rows is None:
            num_rows = synthesizer._num_rows

        if getattr(synthesizer, '_spark_mode', False) is True:
            return synthesizer._sample_spark(round(num_rows), keep_extra_columns=True)

        return synthesizer._sample_batch(round(num_rows), keep_extra_columns=True)

    def _add_child_rows(self, child_name, parent_name, parent_row, sampled_data, num_rows=None):
        """Sample the child rows that reference the parent row.

        Args:
            child_name (str):
                The name of the child table.
            parent_name (str):
                The name of the parent table.
            parent_row (pd.Series):
                The row from the parent table to sample for from the child table.
            sampled_data (dict):
                A dictionary mapping table names to sampled data (pd.DataFrame).
            num_rows (int):
                Number of rows to sample. If None, infers number of child rows to sample
                from the parent row. Defaults to None.
        """
        # A child table is created based on only one foreign key.
        foreign_key = self.metadata._get_foreign_keys(parent_name, child_name)[0]
        if num_rows is None:
            num_rows = parent_row[f'__{child_name}__{foreign_key}__num_rows']

        child_synthesizer = self._recreate_child_synthesizer(child_name, parent_name, parent_row)

        sampled_rows = self._sample_rows(child_synthesizer, num_rows)
        if len(sampled_rows):
            parent_key = self.metadata.tables[parent_name].primary_key
            if foreign_key in sampled_rows:
                row_indices = sampled_rows.index
                sampled_rows.loc[row_indices, foreign_key] = parent_row[parent_key]
            else:
                sampled_rows[foreign_key] = (
                    parent_row[parent_key] if parent_row is not None else np.nan
                )

            previous = sampled_data.get(child_name)
            if previous is None:
                sampled_data[child_name] = sampled_rows
            else:
                sampled_data[child_name] = pd.concat([previous, sampled_rows]).reset_index(
                    drop=True
                )

    def _enforce_table_size(self, child_name, table_name, scale, sampled_data):
        """Ensure the child table has the same size as in the real data times the scale factor.

        This is accomplished by adjusting the number of rows to sample for each parent row.
        If the sum of the values of the `__num_rows` column in the parent table is greater than
        the real data table size * scale, the values are decreased. If the sum is lower, the
        values are increased.

        The values are changed with the following algorithm:

        1. Sort the `__num_rows` column.
        2. If the sum of the values is lower than the target, add 1 to the values from the lowest
           to the highest until the sum is reached, while respecting the maximum values obsverved
           in the real data when possible.
        3. If the sum of the values is higher than the target, subtract 1 from the values from the
           highest to the lowest until the sum is reached, while respecting the minimum values
           observed in the real data when possible.

        Args:
            child_name (str):
                The name of the child table.
            table_name (str):
                The name of the parent table.
            scale (float):
                The scale factor to apply to the table size.
            sampled_data (dict):
                A dictionary mapping table names to sampled data (pd.DataFrame).
        """
        total_num_rows = round(self._table_sizes[child_name] * scale)
        for foreign_key in self.metadata._get_foreign_keys(table_name, child_name):
            null_fk_pctgs = getattr(self, '_null_foreign_key_percentages', {})
            null_fk_pctg = null_fk_pctgs.get(f'__{child_name}__{foreign_key}', 0)
            total_parent_rows = round(total_num_rows * (1 - null_fk_pctg))
            num_rows_key = f'__{child_name}__{foreign_key}__num_rows'
            min_rows = getattr(self, '_min_child_rows', {num_rows_key: 0})[num_rows_key]
            max_rows = self._max_child_rows[num_rows_key]
            key_data = sampled_data[table_name][num_rows_key].fillna(0).round()
            sampled_data[table_name][num_rows_key] = key_data.clip(min_rows, max_rows).astype(int)

            while sum(sampled_data[table_name][num_rows_key]) != total_parent_rows:
                num_rows_column = sampled_data[table_name][num_rows_key].argsort()

                if sum(sampled_data[table_name][num_rows_key]) < total_parent_rows:
                    for i in num_rows_column:
                        # If the number of rows is already at the maximum, skip
                        # The exception is when the smallest value is already at the maximum,
                        # in which case we ignore the boundary
                        if (
                            sampled_data[table_name].loc[i, num_rows_key] >= max_rows
                            and sampled_data[table_name][num_rows_key].min() < max_rows
                        ):
                            break

                        sampled_data[table_name].loc[i, num_rows_key] += 1
                        if sum(sampled_data[table_name][num_rows_key]) == total_parent_rows:
                            break

                else:
                    for i in num_rows_column[::-1]:
                        # If the number of rows is already at the minimum, skip
                        # The exception is when the highest value is already at the minimum,
                        # in which case we ignore the boundary
                        if (
                            sampled_data[table_name].loc[i, num_rows_key] <= min_rows
                            and sampled_data[table_name][num_rows_key].max() > min_rows
                        ):
                            break

                        sampled_data[table_name].loc[i, num_rows_key] -= 1
                        if sum(sampled_data[table_name][num_rows_key]) == total_parent_rows:
                            break

    def _sample_children(self, table_name, sampled_data, scale=1.0):
        """Recursively sample the children of a table.

        This method will loop through the children of a table and sample rows for that child for
        every primary key value in the parent. If the child has already been sampled by another
        parent, this method will skip it.

        Args:
            table_name (string):
                Name of the table (parent) to sample children for.
            sampled_data (dict):
                A dictionary mapping table names to sampled tables (pd.DataFrame).
        """
        if getattr(self, '_spark_mode', False) is True:
            self._sample_children_spark(table_name, sampled_data, scale)
            return

        for child_name in self.metadata._get_child_map()[table_name]:
            self._enforce_table_size(child_name, table_name, scale, sampled_data)

            if child_name not in sampled_data:  # Sample based on only 1 parent
                for _, row in sampled_data[table_name].astype(object).iterrows():
                    self._add_child_rows(
                        child_name=child_name,
                        parent_name=table_name,
                        parent_row=row,
                        sampled_data=sampled_data,
                    )

                foreign_key = self.metadata._get_foreign_keys(table_name, child_name)[0]

                if child_name not in sampled_data:  # No child rows sampled, force row creation
                    num_rows_key = f'__{child_name}__{foreign_key}__num_rows'
                    max_num_child_index = pd.to_numeric(
                        sampled_data[table_name][num_rows_key], errors='coerce'
                    ).idxmax()
                    parent_row = sampled_data[table_name].iloc[max_num_child_index]

                    self._add_child_rows(
                        child_name=child_name,
                        parent_name=table_name,
                        parent_row=parent_row,
                        sampled_data=sampled_data,
                        num_rows=1,
                    )

                total_num_rows = round(self._table_sizes[child_name] * scale)
                null_fk_pctgs = getattr(self, '_null_foreign_key_percentages', {})
                null_fk_pctg = null_fk_pctgs.get(f'__{child_name}__{foreign_key}', 0)
                num_null_rows = round(total_num_rows * null_fk_pctg)
                if num_null_rows > 0:
                    self._add_child_rows(
                        child_name=child_name,
                        parent_name=table_name,
                        parent_row=None,
                        sampled_data=sampled_data,
                        num_rows=num_null_rows,
                    )

                self._sample_children(table_name=child_name, sampled_data=sampled_data, scale=scale)

    def _finalize(self, sampled_data):
        """Remove extra columns from sampled tables and apply finishing touches.

        This method reverts the previous transformations to go back
        to values in the original space.

        Args:
            sampled_data (dict):
                Dictionary mapping table names to sampled tables (pd.DataFrame)

        Returns:
            Dictionary mapping table names to their formatted sampled tables.
        """
        if getattr(self, '_spark_mode', False) is True:
            return self._finalize_spark(sampled_data)

        final_data = {}
        for table_name, table_rows in sampled_data.items():
            synthesizer = self._table_synthesizers.get(table_name)
            column_names = self.get_metadata().get_column_names(table_name)
            if not synthesizer._fitted:
                final_data[table_name] = table_rows[column_names]
                continue

            dtypes = synthesizer._data_processor._dtypes
            for name in column_names:
                dtype = dtypes.get(name)
                if dtype is None:
                    continue

                try:
                    table_rows[name] = table_rows[name].dropna().astype(dtype)
                except Exception:
                    LOGGER.info(
                        "Could not cast back to column's original dtype, keeping original typing."
                    )
                    table_rows[name] = table_rows[name].dropna()

            final_data[table_name] = table_rows[column_names]

        return final_data

    def _sample(self, scale=1.0):
        """Sample the entire dataset.

        Returns a dictionary with all the tables of the dataset. The amount of rows sampled will
        depend from table to table. This is because the children tables are created modeling the
        relation that they have with their parent tables, so its behavior may change from one
        table to another.

        Args:
            scale (float):
                A float representing how much to scale the data by. If scale is set to ``1.0``,
                this does not scale the sizes of the tables. If ``scale`` is greater than ``1.0``
                create more rows than the original data by a factor of ``scale``.
                If ``scale`` is lower than ``1.0`` create fewer rows by the factor of ``scale``
                than the original tables. Defaults to ``1.0``.

        Returns:
            dict:
                A dictionary containing as keys the names of the tables and as values the
                sampled data tables as ``pandas.DataFrame``.
        """
        sampled_data = {}

        # DFS to sample roots and then their children
        non_root_parents = set(self.metadata._get_parent_map().keys())
        root_parents = set(self.metadata.tables.keys()) - non_root_parents
        send_min_sample_warning = False
        for table in root_parents:
            num_rows = round(self._table_sizes[table] * scale)
            if num_rows <= 0:
                send_min_sample_warning = True
                num_rows = 1

            synthesizer = self._table_synthesizers[table]
            LOGGER.info(f'Sampling {num_rows} rows from table {table}')
            sampled_data[table] = self._sample_rows(synthesizer, num_rows)
            self._sample_children(table_name=table, sampled_data=sampled_data, scale=scale)

        if send_min_sample_warning:
            warn_msg = (
                "The 'scale' parameter is too small. Some tables may have 1 row."
                ' For better quality data, please choose a larger scale.'
            )
            warnings.warn(warn_msg)

        added_relationships = set()
        for relationship in self.metadata.relationships:
            parent_name = relationship['parent_table_name']
            child_name = relationship['child_table_name']
            # When more than one relationship exists between two tables, only the first one
            # is used to recreate the child tables, so the rest can be skipped.
            if (parent_name, child_name) not in added_relationships:
                self._add_foreign_key_columns(
                    sampled_data[child_name], sampled_data[parent_name], child_name, parent_name
                )
                added_relationships.add((parent_name, child_name))

        # During this step,  `drop_unknown_references` triggers `metadata.validate(data)`
        # which now warns if the column order in the data does not match the metadata.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='The metadata lists columns in a different order than the data.'
            )
            sampled_data = self._reverse_transform_constraints(sampled_data)

        return self._finalize(sampled_data)

    def _sample_spark(self, scale=1.0):
        """Sample the entire dataset in Spark mode."""
        spark = SparkSession.builder.getOrCreate()
        sampled_data = {}

        # DFS to sample roots and then their children
        non_root_parents = set(self.metadata._get_parent_map().keys())
        root_parents = set(self.metadata.tables.keys()) - non_root_parents
        send_min_sample_warning = False
        for table in root_parents:
            num_rows = round(self._table_sizes[table] * scale)
            if num_rows <= 0:
                send_min_sample_warning = True
                num_rows = 1

            synthesizer = self._table_synthesizers[table]
            LOGGER.info(f'Sampling {num_rows} rows from table {table} in Spark')
            
            root_pdf = self._sample_rows(synthesizer, num_rows)
            # Convert root sample to Spark DataFrame if it isn't one already
            if is_spark_dataframe(root_pdf):
                sampled_data[table] = root_pdf
            else:
                sampled_data[table] = spark.createDataFrame(root_pdf)
            
            # Recursively sample children using Spark
            self._sample_children_spark(table_name=table, sampled_data=sampled_data, scale=scale)

        if send_min_sample_warning:
            warn_msg = (
                "The 'scale' parameter is too small. Some tables may have 1 row."
                ' For better quality data, please choose a larger scale.'
            )
            warnings.warn(warn_msg)

        # Finalize and select columns matching metadata
        return self._finalize_spark(sampled_data)

    def _enforce_table_size_spark(self, child_name, table_name, scale, sampled_data):
        parent_key = self.metadata.tables[table_name].primary_key
        foreign_keys = self.metadata._get_foreign_keys(table_name, child_name)
        for foreign_key in foreign_keys:
            num_rows_key = f'__{child_name}__{foreign_key}__num_rows'
            pdf = sampled_data[table_name].select(parent_key, num_rows_key).toPandas()
            
            total_num_rows = round(self._table_sizes[child_name] * scale)
            null_fk_pctgs = getattr(self, '_null_foreign_key_percentages', {})
            null_fk_pctg = null_fk_pctgs.get(f'__{child_name}__{foreign_key}', 0)
            total_parent_rows = round(total_num_rows * (1 - null_fk_pctg))
            
            min_rows = getattr(self, '_min_child_rows', {num_rows_key: 0})[num_rows_key]
            max_rows = self._max_child_rows[num_rows_key]
            
            key_data = pdf[num_rows_key].fillna(0).round()
            pdf[num_rows_key] = key_data.clip(min_rows, max_rows).astype(int)
            
            current_sum = pdf[num_rows_key].sum()
            if current_sum != total_parent_rows:
                num_rows_column = pdf[num_rows_key].argsort()
                if current_sum < total_parent_rows:
                    for i in num_rows_column:
                        if pdf.loc[i, num_rows_key] >= max_rows and pdf[num_rows_key].min() < max_rows:
                            break
                        pdf.loc[i, num_rows_key] += 1
                        if pdf[num_rows_key].sum() == total_parent_rows:
                            break
                else:
                    for i in num_rows_column[::-1]:
                        if pdf.loc[i, num_rows_key] <= min_rows and pdf[num_rows_key].max() > min_rows:
                            break
                        pdf.loc[i, num_rows_key] -= 1
                        if pdf[num_rows_key].sum() == total_parent_rows:
                            break
            
            spark = sampled_data[table_name].sql_ctx.sparkSession
            parent_df = sampled_data[table_name].drop(num_rows_key)
            # Derive schema from parent DF to avoid type mismatch in join
            parent_key_type = dict(parent_df.dtypes)[parent_key]
            pk_spark_type = LongType() if 'long' in parent_key_type or 'bigint' in parent_key_type else IntegerType()
            update_schema = StructType([
                StructField(parent_key, pk_spark_type, True),
                StructField(num_rows_key, LongType(), True),
            ])
            pdf[parent_key] = pdf[parent_key].astype('int64')
            pdf[num_rows_key] = pdf[num_rows_key].astype('int64')
            updated_df = spark.createDataFrame(pdf[[parent_key, num_rows_key]], schema=update_schema)
            sampled_data[table_name] = parent_df.join(updated_df, on=parent_key, how='left')

    def _sample_children_spark(self, table_name, sampled_data, scale=1.0):
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext
        
        for child_name in self.metadata._get_child_map()[table_name]:
            self._enforce_table_size_spark(child_name, table_name, scale, sampled_data)

            if child_name not in sampled_data:
                foreign_key = self.metadata._get_foreign_keys(table_name, child_name)[0]
                parent_key = self.metadata.tables[table_name].primary_key
                
                dummy_parent_row = sampled_data[table_name].limit(1).toPandas().iloc[0]
                dummy_child_synthesizer = self._recreate_child_synthesizer(child_name, table_name, dummy_parent_row)
                dummy_child_rows = self._sample_rows(dummy_child_synthesizer, 1)
                dummy_child_rows[foreign_key] = dummy_parent_row[parent_key]
                
                schema = self._table_synthesizers[child_name]._data_processor._pandas_to_spark_schema(dummy_child_rows)
                
                self_broadcast = sc.broadcast(self)
                
                def sample_child_rows_group(key_val, pdf):
                    parent_row = pdf.iloc[0]
                    local_self = self_broadcast.value
                    
                    num_rows_key = f'__{child_name}__{foreign_key}__num_rows'
                    raw_num_rows = parent_row.get(num_rows_key, 0)
                    num_rows = 0 if (raw_num_rows is None or pd.isna(raw_num_rows)) else int(raw_num_rows)
                    if num_rows <= 0:
                        return pd.DataFrame(columns=dummy_child_rows.columns)
                        
                    # ponytail: Random seeds inside parallel workers are derived from local system entropy to ensure unique synthetic rows per partition.
                    pass
                    seed = int(time.time() * 1000) % 2**32
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    random.seed(seed)
                    
                    child_synthesizer = local_self._recreate_child_synthesizer(child_name, table_name, parent_row)
                    sampled_rows = local_self._sample_rows(child_synthesizer, num_rows)
                    
                    if len(sampled_rows) > 0:
                        parent_key = local_self.metadata.tables[table_name].primary_key
                        sampled_rows[foreign_key] = parent_row[parent_key]
                        
                    return sampled_rows

                child_df = sampled_data[table_name].groupby(parent_key).applyInPandas(sample_child_rows_group, schema)

                total_num_rows = round(self._table_sizes[child_name] * scale)
                null_fk_pctgs = getattr(self, '_null_foreign_key_percentages', {})
                null_fk_pctg = null_fk_pctgs.get(f'__{child_name}__{foreign_key}', 0)
                num_null_rows = round(total_num_rows * null_fk_pctg)
                
                if num_null_rows > 0:
                    child_synthesizer = self._recreate_child_synthesizer(child_name, table_name, None)
                    null_rows = self._sample_rows(child_synthesizer, num_null_rows)
                    null_rows[foreign_key] = None
                    null_rows_spark = spark.createDataFrame(null_rows, schema)
                    child_df = child_df.union(null_rows_spark)
                    
                sampled_data[child_name] = child_df

                self._sample_children_spark(table_name=child_name, sampled_data=sampled_data, scale=scale)

    def _finalize_spark(self, sampled_data):
        # Add missing FK columns by joining from parent tables (mirrors _add_foreign_key_columns)
        added_relationships = set()
        for relationship in self.metadata.relationships:
            parent_name = relationship['parent_table_name']
            child_name = relationship['child_table_name']
            foreign_key = relationship['child_foreign_key']
            parent_key = self.metadata.tables[parent_name].primary_key

            if (parent_name, child_name) in added_relationships:
                continue
            added_relationships.add((parent_name, child_name))

            if child_name not in sampled_data or parent_name not in sampled_data:
                continue

            child_df = sampled_data[child_name]
            # If FK column is already present, skip
            if foreign_key in child_df.columns:
                continue

            parent_df = sampled_data[parent_name]
            # Assign FK values via round-robin from parent pool.
            # ponytail: round-robin, not relational. Upgrade path: propagate FK in _sample_children_spark.
            parent_keys = parent_df.select(parent_key).toPandas()[parent_key].tolist()
            if not parent_keys:
                continue
            child_count = child_df.count()
            if child_count == 0:
                # Add FK as null column to preserve schema for downstream evaluation
                pk_type = dict(parent_df.dtypes).get(parent_key, 'bigint')
                pk_spark_type = LongType() if 'long' in pk_type or 'bigint' in pk_type else IntegerType()
                sampled_data[child_name] = child_df.withColumn(foreign_key, F.lit(None).cast(pk_spark_type))
                continue
            assigned = (parent_keys * ((child_count // len(parent_keys)) + 1))[:child_count]
            # Use row_number() for sequential 1-based idx, then subtract 1 to match range()
            w = Window.orderBy(F.lit(0))
            child_with_idx = child_df.withColumn('__row_idx__', F.row_number().over(w) - 1)
            spark = child_df.sparkSession
            pdf_map = pd.DataFrame({'__row_idx__': range(child_count), foreign_key: assigned})
            map_df = spark.createDataFrame(pdf_map)
            sampled_data[child_name] = child_with_idx.join(map_df, on='__row_idx__', how='left').drop('__row_idx__')

        final_data = {}
        for table_name, table_rows in sampled_data.items():
            column_names = self.metadata.get_column_names(table_name)
            # Only select columns that actually exist in this Spark DataFrame
            existing = set(table_rows.columns)
            cols_to_select = [c for c in column_names if c in existing]
            final_data[table_name] = table_rows.select(*cols_to_select)
        return final_data
