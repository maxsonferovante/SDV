"""Integration test: CopulaGANSynthesizer fit + sample on Spark DataFrames."""
import os
import sys
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CopulaGANSynthesizer
from pyspark.sql import SparkSession


def main():
    print("--- 1. Inicializando Spark Session ---")
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder \
        .appName("CopulaGANSparkTest") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

    data = pd.DataFrame({
        'id':       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'age':      [23.5, 45.0, 12.0, 67.2, 34.5, 55.1, 29.3, 41.8, 37.6, 22.0],
        'income':   [3000, 8000, 1500, 12000, 5000, 9500, 4200, 7100, 6300, 2800],
        'category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
    })

    metadata = Metadata.load_from_dict({
        'METADATA_SPEC_VERSION': 'V1',
        'columns': {
            'id':       {'sdtype': 'id'},
            'age':      {'sdtype': 'numerical'},
            'income':   {'sdtype': 'numerical'},
            'category': {'sdtype': 'categorical'},
        },
        'primary_key': 'id'
    })

    print("\n--- 2. Dados originais (Spark DF) ---")
    spark_df = spark.createDataFrame(data)
    spark_df.show()

    print("\n--- 3. Treinando CopulaGANSynthesizer no Spark ---")
    synthesizer = CopulaGANSynthesizer(metadata, epochs=5)
    synthesizer.fit(spark_df)
    print("Fit concluído!")

    print("\n--- 4. Amostrando 10 linhas no Spark ---")
    synthetic_df = synthesizer.sample(num_rows=10)
    print(f"Tipo: {type(synthetic_df).__name__}")
    count = synthetic_df.count()
    print(f"Contagem: {count}")
    synthetic_df.show(truncate=False)

    assert count == 10, f"Esperado 10, obtido {count}"
    assert set(synthetic_df.columns) == {'id', 'age', 'income', 'category'}, \
        f"Colunas inesperadas: {synthetic_df.columns}"

    print("\n--- SUCESSO: CopulaGANSynthesizer Spark end-to-end OK! ---")
    spark.stop()


if __name__ == "__main__":
    main()
