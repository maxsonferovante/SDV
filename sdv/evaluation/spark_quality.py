"""Spark-native quality evaluation for multi-table synthetic data.

Computes Column Shapes, Column Pair Trends, Cardinality, and Intertable Trends
using Spark aggregations — avoids .toPandas() on the full dataset.
"""

import math

import numpy as np

try:
    from pyspark.sql import functions as F
except ImportError:
    F = None

from sdv._utils import is_spark_dataframe


def _ks_complement_spark(real_col, synth_col, real_df, synth_df, num_bins=100):
    """Compute KS complement for a numeric column using Spark histograms.

    Bins the column into `num_bins` equal-width buckets, computes CDFs from
    the bin counts, and returns 1 - KS_statistic.
    """
    # Get global min/max across both datasets
    stats = real_df.select(
        F.min(real_col).alias('rmin'), F.max(real_col).alias('rmax')
    ).collect()[0]
    s_stats = synth_df.select(
        F.min(synth_col).alias('smin'), F.max(synth_col).alias('smax')
    ).collect()[0]

    col_min = min(stats['rmin'] or 0, s_stats['smin'] or 0)
    col_max = max(stats['rmax'] or 1, s_stats['smax'] or 1)

    if col_min == col_max:
        return 1.0  # identical constant columns

    bin_width = (col_max - col_min) / num_bins

    def _bin_counts(df, col):
        return (
            df.select(F.floor((F.col(col) - col_min) / bin_width).cast('int').alias('bin'))
            .filter(F.col('bin').isNotNull())
            .groupBy('bin')
            .count()
            .collect()
        )

    real_counts = np.zeros(num_bins)
    for row in _bin_counts(real_df, real_col):
        idx = max(0, min(row['bin'], num_bins - 1))
        real_counts[idx] += row['count']

    synth_counts = np.zeros(num_bins)
    for row in _bin_counts(synth_df, synth_col):
        idx = max(0, min(row['bin'], num_bins - 1))
        synth_counts[idx] += row['count']

    # CDFs
    real_total = real_counts.sum()
    synth_total = synth_counts.sum()
    if real_total == 0 or synth_total == 0:
        return 0.0

    real_cdf = np.cumsum(real_counts) / real_total
    synth_cdf = np.cumsum(synth_counts) / synth_total

    ks_stat = np.max(np.abs(real_cdf - synth_cdf))
    return 1.0 - ks_stat


def _tv_complement_spark(real_col, synth_col, real_df, synth_df):
    """Compute TV complement for a categorical column using Spark groupBy counts."""
    real_dist = dict(
        real_df.groupBy(real_col).count()
        .withColumnRenamed('count', 'cnt')
        .select(F.col(real_col).cast('string'), 'cnt')
        .collect()
    )
    synth_dist = dict(
        synth_df.groupBy(synth_col).count()
        .withColumnRenamed('count', 'cnt')
        .select(F.col(synth_col).cast('string'), 'cnt')
        .collect()
    )

    real_total = sum(real_dist.values()) or 1
    synth_total = sum(synth_dist.values()) or 1

    all_keys = set(real_dist.keys()) | set(synth_dist.keys())
    tv = sum(
        abs(real_dist.get(k, 0) / real_total - synth_dist.get(k, 0) / synth_total)
        for k in all_keys
    ) / 2.0

    return 1.0 - tv


def _correlation_similarity_spark(col_a, col_b, real_df, synth_df):
    """Compute correlation similarity between two numeric columns using Spark corr()."""
    real_corr = real_df.select(F.corr(col_a, col_b)).collect()[0][0]
    synth_corr = synth_df.select(F.corr(col_a, col_b)).collect()[0][0]

    if real_corr is None or synth_corr is None:
        return float('nan')
    if math.isnan(real_corr) or math.isnan(synth_corr):
        return float('nan')

    # Score = 1 - |diff| / 2, clamped to [0, 1]
    return max(0.0, 1.0 - abs(real_corr - synth_corr) / 2.0)


def _cardinality_score_spark(real_data, synth_data, metadata):
    """Compute cardinality score for all relationships using Spark countDistinct."""
    scores = []
    for rel in metadata.relationships:
        parent_name = rel['parent_table_name']
        child_name = rel['child_table_name']
        fk = rel['child_foreign_key']
        pk = rel['parent_primary_key']

        if child_name not in real_data or child_name not in synth_data:
            continue

        real_child = real_data[child_name]
        synth_child = synth_data[child_name]
        real_parent = real_data[parent_name]
        synth_parent = synth_data[parent_name]

        # Cardinality = avg children per parent
        real_parent_count = real_parent.select(F.countDistinct(pk)).collect()[0][0] or 1
        synth_parent_count = synth_parent.select(F.countDistinct(pk)).collect()[0][0] or 1
        real_child_count = real_child.count()
        synth_child_count = synth_child.count()

        real_avg = real_child_count / real_parent_count
        synth_avg = synth_child_count / max(synth_parent_count, 1)

        if real_avg == 0 and synth_avg == 0:
            scores.append(1.0)
        elif real_avg == 0 or synth_avg == 0:
            scores.append(0.0)
        else:
            ratio = min(real_avg, synth_avg) / max(real_avg, synth_avg)
            scores.append(ratio)

    return np.mean(scores) * 100 if scores else float('nan')


def evaluate_quality_spark(real_data, synth_data, metadata, verbose=True):
    """Evaluate synthetic data quality using Spark aggregations.

    Computes metrics distributedly without collecting full datasets to driver.
    Compatible score semantics with sdmetrics QualityReport.

    Args:
        real_data: dict of table_name -> pandas or Spark DataFrame (real data).
        synth_data: dict of table_name -> Spark DataFrame (synthetic data).
        metadata: SDV Metadata object.
        verbose: Print scores as they are computed.

    Returns:
        dict with keys 'column_shapes', 'column_pair_trends', 'cardinality',
        'intertable_trends', 'overall'.
    """
    # Convert real_data to Spark if needed
    if any(not is_spark_dataframe(df) for df in real_data.values()):
        # Need a SparkSession to convert pandas → Spark
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        real_spark = {}
        for name, df in real_data.items():
            if is_spark_dataframe(df):
                real_spark[name] = df
            else:
                real_spark[name] = spark.createDataFrame(df)
        real_data = real_spark

    if verbose:
        print("Generating Spark quality report ...\n")

    # 1. Column Shapes
    shape_scores = []
    for table_name in metadata.tables:
        if table_name not in real_data or table_name not in synth_data:
            continue
        real_df = real_data[table_name]
        synth_df = synth_data[table_name]
        table_meta = metadata.tables[table_name]

        for col_name in metadata.get_column_names(table_name):
            if col_name == table_meta.primary_key:
                continue
            if col_name not in real_df.columns or col_name not in synth_df.columns:
                continue

            col_meta = table_meta.columns.get(col_name, {})
            sdtype = col_meta.get('sdtype', 'categorical')

            if sdtype in ('numerical', 'id'):
                score = _ks_complement_spark(col_name, col_name, real_df, synth_df)
            else:
                score = _tv_complement_spark(col_name, col_name, real_df, synth_df)

            if not math.isnan(score):
                shape_scores.append(score)

    column_shapes = np.mean(shape_scores) * 100 if shape_scores else float('nan')
    if verbose:
        print(f"Column Shapes Score: {column_shapes:.2f}%")

    # 2. Column Pair Trends
    pair_scores = []
    for table_name in metadata.tables:
        if table_name not in real_data or table_name not in synth_data:
            continue
        real_df = real_data[table_name]
        synth_df = synth_data[table_name]
        table_meta = metadata.tables[table_name]

        numeric_cols = [
            col for col in metadata.get_column_names(table_name)
            if col != table_meta.primary_key
            and col in real_df.columns and col in synth_df.columns
            and table_meta.columns.get(col, {}).get('sdtype') == 'numerical'
        ]

        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i + 1:]:
                score = _correlation_similarity_spark(col_a, col_b, real_df, synth_df)
                if not math.isnan(score):
                    pair_scores.append(score)

    column_pair_trends = np.mean(pair_scores) * 100 if pair_scores else float('nan')
    if verbose:
        print(f"Column Pair Trends Score: {column_pair_trends:.2f}%")

    # 3. Cardinality
    cardinality = _cardinality_score_spark(real_data, synth_data, metadata)
    if verbose:
        print(f"Cardinality Score: {cardinality:.2f}%")

    # 4. Intertable Trends
    inter_scores = []
    for rel in metadata.relationships:
        parent_name = rel['parent_table_name']
        child_name = rel['child_table_name']
        fk = rel['child_foreign_key']
        pk = rel['parent_primary_key']

        if child_name not in real_data or child_name not in synth_data:
            continue
        if parent_name not in real_data or parent_name not in synth_data:
            continue

        real_child_a = real_data[child_name].alias('child')
        real_parent_a = real_data[parent_name].alias('parent')
        synth_child_a = synth_data[child_name].alias('child')
        synth_parent_a = synth_data[parent_name].alias('parent')

        real_joined = real_child_a.join(real_parent_a, F.col(f'child.{fk}') == F.col(f'parent.{pk}'), 'left')
        synth_joined = synth_child_a.join(synth_parent_a, F.col(f'child.{fk}') == F.col(f'parent.{pk}'), 'left')

        parent_meta = metadata.tables[parent_name]
        child_meta = metadata.tables[child_name]

        # Find numeric column pairs across parent-child
        parent_numeric = [
            c for c in metadata.get_column_names(parent_name)
            if c != parent_meta.primary_key
            and parent_meta.columns.get(c, {}).get('sdtype') == 'numerical'
        ]
        child_numeric = [
            c for c in metadata.get_column_names(child_name)
            if c != child_meta.primary_key and c != fk
            and child_meta.columns.get(c, {}).get('sdtype') == 'numerical'
        ]

        for p_col in parent_numeric:
            for c_col in child_numeric:
                p_ref = f'parent.{p_col}'
                c_ref = f'child.{c_col}'
                try:
                    score = _correlation_similarity_spark(p_ref, c_ref, real_joined, synth_joined)
                    if not math.isnan(score):
                        inter_scores.append(score)
                except Exception:
                    pass  # ponytail: skip columns that don't resolve after join (e.g. dropped by Spark)

    intertable_trends = np.mean(inter_scores) * 100 if inter_scores else float('nan')
    if verbose:
        print(f"Intertable Trends Score: {intertable_trends:.2f}%")

    # Overall
    valid_scores = [s for s in [column_shapes, column_pair_trends, cardinality, intertable_trends] if not math.isnan(s)]
    overall = np.mean(valid_scores) if valid_scores else float('nan')

    if verbose:
        print(f"\nOverall Score (Average): {overall:.2f}%")

    return {
        'column_shapes': column_shapes,
        'column_pair_trends': column_pair_trends,
        'cardinality': cardinality,
        'intertable_trends': intertable_trends,
        'overall': overall,
    }
