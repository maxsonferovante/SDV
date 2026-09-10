"""Integration test: CopulaGANSynthesizer fit + sample on Spark DataFrames."""
import os
import sys
import kagglehub
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CopulaGANSynthesizer
from pyspark.sql import SparkSession


def main():
    print("--- 1. Inicializando Spark Session ---")
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.environ['KAGGLEHUB_CACHE'] = './dataset_real'
    
    path_czech = kagglehub.dataset_download("mariammariamr/1999-czech-financial-dataset")
    print("Caminho local do dataset:", path_czech)
    
    base_dir = os.path.join(path_czech, "lpetrocelli-czech-financial-dataset-real-anonymized-transactions")
    if not os.path.exists(base_dir):
        base_dir = path_czech
    


    account_path = os.path.join(base_dir, "account.csv")

    accounts_df = pd.read_csv(account_path, sep=';')
    
    print(f"  - Contas: {accounts_df.shape[0]} linhas")

    spark = SparkSession.builder \
        .appName("CopulaGANSparkTest") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

    # Pre-clean string/object columns with NaNs to prevent PySpark schema merge issues
    for col in accounts_df.columns:
        if accounts_df[col].dtype == object:
            accounts_df[col] = accounts_df[col].fillna('').astype(str)

    metadata = Metadata.detect_from_dataframe(data=accounts_df)

    print("\n--- 2. Dados originais (Spark DF) ---")
    spark_df = spark.createDataFrame(accounts_df)
    spark_df.limit(5).show()

    print("\n--- 3. Treinando CopulaGANSynthesizer no Spark ---")
    synthesizer = CopulaGANSynthesizer(metadata, epochs=100)
    synthesizer.fit(spark_df)
    print("Fit concluído!")

    print("\n--- 4. Amostrando 30 linhas no Spark (batch_size=3 para testar micro-batching) ---")
    synthetic_df = synthesizer.sample(num_rows=30, batch_size=3)
    print(f"Tipo: {type(synthetic_df).__name__}")
    count = synthetic_df.count()
    print(f"Contagem: {count}")
    synthetic_df.show(truncate=False)

    print("\n--- SUCESSO: CopulaGANSynthesizer Spark end-to-end OK! ---")
    spark.stop()


if __name__ == "__main__":
    main()
