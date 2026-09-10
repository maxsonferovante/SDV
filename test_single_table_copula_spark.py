import os
import sys
import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from pyspark.sql import SparkSession

def main():
    print("--- 1. Inicializando Spark Session ---")
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    spark = SparkSession.builder \
        .appName("SingleTableCopulaSparkTest") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "1g") \
        .getOrCreate()

    # 2. Criando dados reais de exemplo
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'age': [23.5, 45.0, 12.0, 67.2, 34.5],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    
    metadata_dict = {
        'METADATA_SPEC_VERSION': 'V1',
        'columns': {
            'id': {'sdtype': 'id'},
            'age': {'sdtype': 'numerical'},
            'category': {'sdtype': 'categorical'}
        },
        'primary_key': 'id'
    }
    metadata = Metadata.load_from_dict(metadata_dict)
    
    print("\n--- 3. Convertendo para Spark DataFrame ---")
    spark_df = spark.createDataFrame(data)
    spark_df.show()
    
    print("\n--- 4. Treinando GaussianCopulaSynthesizer no Spark ---")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(spark_df)
    print("Modelo treinado com sucesso no Spark!")
    
    print("\n--- 5. Amostrando Dados no Spark ---")
    synthetic_df = synthesizer.sample(num_rows=10)
    print(f"Tipo de retorno: {type(synthetic_df)}")
    
    count = synthetic_df.count()
    print("Contagem real de linhas:", count)
    print("Todas as linhas:")
    synthetic_df.show(truncate=False)

    assert count == 10, f"Esperado 10 linhas, obtido {count}"
    print("\n--- Sucesso total no teste de integração do GaussianCopulaSynthesizer! ---")
    spark.stop()

if __name__ == "__main__":
    main()
