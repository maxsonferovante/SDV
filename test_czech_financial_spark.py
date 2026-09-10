from kagglehub import colab_cache_resolver
import os
import pandas as pd
import kagglehub
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.evaluation.spark_quality import evaluate_quality_spark

def main():
    print("--- 1. Verificando/Baixando Czech Financial Dataset (Berka) ---")
    # Define o cache do kagglehub para ser uma pasta local no projeto
    import sys
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.environ['KAGGLEHUB_CACHE'] = './dataset_real'
    
    # Faz o download (o kagglehub verifica se ja existe localmente antes de baixar)
    path_czech = kagglehub.dataset_download("mariammariamr/1999-czech-financial-dataset")
    print("Caminho local do dataset:", path_czech)
    
    base_dir = os.path.join(path_czech, "lpetrocelli-czech-financial-dataset-real-anonymized-transactions")
    if not os.path.exists(base_dir):
        base_dir = path_czech
        
    client_path = os.path.join(base_dir, "client.csv")
    account_path = os.path.join(base_dir, "account.csv")
    disp_path = os.path.join(base_dir, "disp.csv")
    card_path = os.path.join(base_dir, "card.csv")
    district_path = os.path.join(base_dir, "district.csv")
    loan_path = os.path.join(base_dir, "loan.csv")
    order_path = os.path.join(base_dir, "order.csv")
    trans_path = os.path.join(base_dir, "trans.csv")
    
    # Carrega todas as 8 tabelas originais, limitando o tamanho para manter o teste extremamente rápido
    clients_df = pd.read_csv(client_path, sep=';')[['client_id', 'birth_number', 'district_id']].head(500)
    accounts_df = pd.read_csv(account_path, sep=';')[['account_id', 'district_id', 'date']].head(500)
    disp_df = pd.read_csv(disp_path, sep=';')[['disp_id', 'client_id', 'account_id', 'type']].head(500)
    card_df = pd.read_csv(card_path, sep=';')[['card_id', 'disp_id', 'type']].head(500)
    districts_df = pd.read_csv(district_path, sep=';')[['A1', 'A2']].rename(columns={'A1': 'district_id'}).head(500)
    loans_df = pd.read_csv(loan_path, sep=';')[['loan_id', 'account_id', 'amount']].head(500)
    orders_df = pd.read_csv(order_path, sep=';')[['order_id', 'account_id', 'amount']].head(500)
    trans_df = pd.read_csv(trans_path, sep=';', low_memory=False)[['trans_id', 'account_id', 'amount']].head(500)
    
    # ponytail: Pre-clean string columns with NaNs to prevent PySpark schema merge issues (e.g. NaN float mixed with string)
    for df in [clients_df, accounts_df, disp_df, card_df, districts_df, loans_df, orders_df, trans_df]:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna('').astype(str)
    
    print(f"Subconjunto carregado para treino:")
    print(f"  - Distritos: {districts_df.shape[0]} linhas")
    print(f"  - Clientes: {clients_df.shape[0]} linhas")
    print(f"  - Contas: {accounts_df.shape[0]} linhas")
    print(f"  - Disposicoes: {disp_df.shape[0]} linhas")
    print(f"  - Cartoes: {card_df.shape[0]} linhas")
    print(f"  - Emprestimos: {loans_df.shape[0]} linhas")
    print(f"  - Ordens: {orders_df.shape[0]} linhas")
    print(f"  - Transacoes: {trans_df.shape[0]} linhas")

    data = {
        "distritos": districts_df,
        "clientes": clients_df,
        "contas": accounts_df,
        "disposicoes": disp_df,
        "cartoes": card_df,
        "emprestimos": loans_df,
        "ordens": orders_df,
        "transacoes": trans_df
    }

    print("\n--- 2. Detectando Metadados Relacionais ---")
    # Usamos o novo Metadata do SDV v1
    metadata = Metadata.detect_from_dataframes(data=data)
    
    # Validar metadados auto-detectados
    metadata.validate()
    print("Metadados relacionais auto-detectados e validados com sucesso!")
    print("Relacionamentos detectados:")
    for rel in metadata.relationships:
        print(f"  {rel['parent_table_name']}.{rel['parent_primary_key']} -> {rel['child_table_name']}.{rel['child_foreign_key']}")

    print("\n--- 3. Iniciando Spark Session e Convertendo para Spark DataFrames ---")
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("CzechFinancialHMASparkTest") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
        
    spark_data = {
        name: spark.createDataFrame(df) for name, df in data.items()
    }
    print("Spark Session iniciada. Tabelas convertidas com sucesso!")

    print("\n--- 4. Treinando o HMASynthesizer no Spark ---")
    synthesizer = HMASynthesizer(metadata, verbose=True)
    synthesizer.fit(spark_data)
    print("Modelo HMASynthesizer treinado com sucesso no Spark!")

    print("\n--- 5. Gerando Dados Sinteticos Relacionais no Spark ---")
    synthetic_spark_data = synthesizer.sample(scale=1.0)
    
    synthetic_data = {
        name: df.cache() for name, df in synthetic_spark_data.items()
    }
    
    print("\nDados Sinteticos Gerados:")
    for name, df in synthetic_data.items():
        print(f"  - {name.capitalize()}: {df.count()} registros")

    print("\nAmostra das Transacoes sinteticas:")
    synthetic_data["transacoes"].show(3)

    print("\n--- 6. Avaliando Qualidade Estatística Completa ---")
    evaluate_quality_spark(
        real_data=spark_data,
        synth_data=synthetic_data,
        metadata=metadata,
        verbose=True
    )
    
    print("\n--- 7. Sucesso! ---")

if __name__ == "__main__":
    main()
