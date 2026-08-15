# Product Requirement Document (PRD) - PySpark Integration in SDV

## Problem Statement

Currently, the SDV library operates entirely on **Pandas DataFrames** in memory. While highly effective for small-to-medium datasets, this architecture does not scale to high volumes of enterprise-level data. Attempting to fit generative models or sample large synthetic datasets (millions or billions of rows) on a single node leads to out-of-memory (OOM) errors and extremely long execution times. To make SDV viable for enterprise data lakes and modern cloud architectures (e.g., Databricks, EMR, GCP Dataproc), it must natively support distributed data structures and parallel processing using PySpark.

## Solution

Enable end-to-end distributed execution in SDV by integrating **PySpark DataFrames** (`pyspark.sql.DataFrame`) natively. The solution will scale data validation, preprocessing, model training, and data generation/sampling across a Spark cluster without single-node bottlenecks.

Key capabilities:
1.  Accept and return Spark DataFrames natively at the API boundaries.
2.  Parallelize data preprocessing using a hybrid Spark/RDT approach with `mapInPandas`.
3.  Support distributed training for deep learning models (CTGAN, TVAE) via PySpark `TorchDistributor`.
4.  Distribute relational model training (HMA child-table parameters) using grouped map UDFs (`groupby().applyInPandas()`).
5.  Perform parallel, partition-level synthesis for single-table and relational sampling on Spark workers.

## User Stories

1.  As a data engineer, I want to pass a PySpark DataFrame directly to the synthesizer's `fit` method, so that I don't have to convert large datasets to Pandas and risk OOM crashes.
2.  As a data scientist, I want the `preprocess` method to transform my large raw dataset into a preprocessed PySpark DataFrame, so that I can scale feature engineering across the cluster.
3.  As an enterprise user, I want the `sample` method to generate a large PySpark DataFrame with millions of synthetic rows in parallel, so that the synthesis completes in a reasonable time.
4.  As a machine learning engineer, I want CTGAN/TVAE training to run in a distributed PyTorch environment using multiple nodes/GPUs in the Spark cluster, so that I can train models on massive datasets quickly.
5.  As a database administrator, I want to synthesize a multi-table schema using `HMASynthesizer` directly on Spark DataFrames, so that relational integrity (primary/foreign keys) is maintained across large tables.
6.  As a developer, I want the RDT metadata parameters (e.g. min/max, categories) to be calculated on a representative sample or via optimized Spark SQL queries, so that the metadata fitting phase is fast and lightweight.
7.  As a security specialist, I want PII anonymization and constraint validation to run in parallel on Spark workers during sampling, so that security policies are enforced efficiently on high-volume synthetic outputs.
8.  As a data engineer, I want the final sampled synthetic data to be returned as a Spark DataFrame, so that I can directly save it to S3, HDFS, or Delta Lake using native Spark write formats.

## Implementation Decisions

*   **API Interface Alignment**: Synthesizer boundaries (e.g., `fit`, `preprocess`, `sample`) will check the type of input. If a PySpark DataFrame is provided, they will route logic to Spark-specific execution classes/methods; if a Pandas DataFrame is provided, they will fall back to the existing Pandas pipeline to preserve backward compatibility.
*   **Hybrid Preprocessing (Spark + RDT)**:
    *   Parameters for the `HyperTransformer` will be fitted by taking a statistically significant sample of the Spark DataFrame or via Spark aggregation functions.
    *   The fitted configuration will be broadcasted to all executors.
    *   The `transform` and `reverse_transform` functions will run on Spark workers using `.mapInPandas()` to parallelize computation across partitions.
*   **Distributed Training for PyTorch Synthesizers**:
    *   `CTGANSynthesizer` and `TVAESynthesizer` will use PySpark's native `TorchDistributor` class to spawn PyTorch DDP (Distributed Data Parallel) processes on worker nodes, allowing distributed gradient descent.
*   **Relational Extension Parameter Fitting**:
    *   During HMA relational parameter learning, the children tables will be grouped by their parent foreign key on Spark, and the parameter estimation for each group will be executed in parallel on Spark workers using `groupby().applyInPandas()`.
*   **Partition-level Sampling**:
    *   Sampling will use `spark.range()` to allocate partitions of a skeleton DataFrame.
    *   On each partition, worker processes will use a local random seed to generate latents, run inference through the broadcasted model, apply the reverse transformer, and execute constraint checks.
    *   Child-table generation in HMA will partition the synthesized parent table and generate corresponding child rows on workers.

## Testing Decisions

*   **Seams to Test**:
    *   **Preprocessing Seam**: Test that `DataProcessor.transform()` on a Spark DataFrame returns a Spark DataFrame whose content is identical to running it in Pandas.
    *   **Synthesizer Fit/Sample Seam**: Test that calling `fit()` and `sample()` at the highest API levels (e.g. `GaussianCopulaSynthesizer`, `CTGANSynthesizer`, `HMASynthesizer`) behaves correctly and outputs valid Spark DataFrames.
    *   **Regression Seam**: Verify that all existing unit and integration tests run successfully with Pandas inputs, confirming no regressions.
*   **Test Environment**:
    *   Tests will run on a local unit test suite using a local Spark session (`SparkSession.builder.master("local[*]").getOrCreate()`).
*   **Prior Art**:
    *   Verify existing test structure in `tests/` and mirror the test style for Spark-specific test suites.

## Out of Scope

*   Rewriting the RDT (Reversible Data Transforms) library entirely in Spark ML (the hybrid `mapInPandas` approach is selected for code reuse and rapid migration).
*   Supporting distributed training for multi-table architectures other than HMA (e.g. HSA) in the initial release.
*   Automatic cluster scaling or cloud provider infrastructure setup (users are expected to provide an active, configured `SparkSession`).

## Further Notes

*   Ensure proper warning messages when partitions are too large, which could lead to OOM errors when converted to Pandas inside UDFs. Provide configuration documentation on how to calculate partition size relative to worker RAM.
