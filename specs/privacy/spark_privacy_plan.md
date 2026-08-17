# Plano de Implementação: Garantia de Não-Vazamento e Privacidade em Dados Sintéticos com Apache Spark

Este plano estabelece a arquitetura de **Garantia de Privacidade em Grande Escala (Big Data)** baseada em uma **Abordagem Híbrida (Privacy-by-Design)**. Todas as operações de varredura de dados, geração de hash, cálculos de distância (DCR) e comparação entre datasets real e sintético são executadas de forma **100% distribuída no Apache Spark (PySpark SQL / ML)**.

---

## 1. Arquitetura Distribuída no Apache Spark

```
[ Dataset Treino Real (Spark DataFrame) ]      [ Dataset Sintético Gerado (Spark DataFrame) ]
                        │                                        │
                        └───────────────────┬────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Filtro de Correspondência Exata em Spark (Exact Match via Hash SHA-256)             │
│    - Compute row hash: sha2(concat_ws('||', colunas...), 256)                          │
│    - Anti-join ou exceptAll distribuído entre Spark DataFrames                         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ (Linhas sem cópia exata)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Filtro de Distância Mínima Distribuído em Spark (DCR com Locality Sensitive Hashing)│
│    - Mapeamento de atributos via VectorAssembler + Normalizer em Spark ML              │
│    - BucketedRandomProjectionLSH / MinHashLSH para busca k-NN escalável O(N)          │
│    - Cálculo de distância Gower/Euclidiana por UDF/vector_distance distribuída         │
│    - Rejeição e reamostragem de linhas com DCR < τ_DCR                                 │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ (Dataset Sintético Aprovado)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Treinamento com Privacidade Diferencial (DP-SGD / DP-Copula)                        │
│    - Orçamento ε = 1.0, δ = 1/N configurável via Spark Config                         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Dashboard Visual de Auditoria de Privacidade em Spark                               │
│    - Coleta de agregados distribuídos (hist_bins, min/mean/max DCR, MIA Score)         │
│    - Geração do Dashboard com distribuições DCR e curva de Fidelidade vs. Privacidade  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detalhes de Implementação em Spark

### 2.1. Varredura e Hash para Exact Match (`SparkExactMatchFilter`)
- **Geração de Hash Distribuída**:
  ```python
  df_with_hash = df.withColumn(
      "__row_hash__",
      F.sha2(F.concat_ws("||", *[F.coalesce(F.col(c).cast("string"), F.lit("__NULL__")) for c in check_cols]), 256)
  )
  ```
- **Anti-Join Distribuído**:
  - Filtra o DataFrame sintético contra o DataFrame real executando um `left_anti` join no Spark sobre a coluna `__row_hash__`.
  - Tratamento de colunas de baixa cardinalidade: O hash e o anti-join utilizam o vetor completo de colunas (ou o conjunto de colunas de alta cardinalidade/contínuas + QIDs), evitando descarte indevido por colisões categóricas triviais.

### 2.2. Cálculo Escala-BigData de DCR (`SparkDCRFilter`)
- **Problema de Escala**: Calcular a distância par a par entre $N_{sintetico}$ e $N_{real}$ é $O(N \times M)$ e não escala em grande massa de dados.
- **Solução Spark ML LSH**:
  1. `VectorAssembler` cria o vetor de features distribuído para colunas numéricas e categorizadas.
  2. `BucketedRandomProjectionLSH` (ou `MinHashLSH` para categóricas) distribui os pontos em buckets.
  3. `lsh.approxSimilarityJoin(df_syn, df_real, threshold=τ_DCR)` calcula distâncias **apenas** entre vizinhos candidatos próximos.
  4. Linhas sintéticas que retornarem pares com distância $< \tau_{DCR}$ são identificadas distribuidamente no Spark e filtradas.

### 2.3. Séries Temporais e Relacional Multi-Tabela em Spark
- **Multi-Tabela**: O DCR é computado por chave primária/estrangeira usando `groupBy(parent_id)` e agregações de contagem/estatísticas filhas via PySpark SQL (`agg`, `collect_list`).
- **Séries Temporais**: Matrizes de janelas deslizantes (rolling windows) e estatísticas de séries temporais computadas via Spark Window Functions (`Window.partitionBy().orderBy()`) para compor o vetor de atributos do DCR.

### 2.4. Dashboard Visual de Auditoria (`SparkPrivacyDashboard`)
- Todas as métricas estatísticas pesadas (percentis de DCR, frequências, histogramas com `width_bucket`) são computadas nativamente pelos executores Spark via `groupBy` e `agg`.
- Somente os resultados agregados leves (histograma binned, contagens de exact match, métricas de MIA) são coletados para renderização dos gráficos no Dashboard.

---

## 3. Componentes a Criar/Modificar no Repositório

#### [NEW] [spark_privacy_guard.py](file:///Users/mferovante/Documents/workspace/SDV/sdv/spark_privacy_guard.py)
Implementação 100% PySpark para:
- `SparkExactMatchFilter`: Hashing SHA-256 e `left_anti` join.
- `SparkDCRFilter`: Locality Sensitive Hashing (LSH) e junção de similaridade de proximidade.
- Integration loop com reamostragem distribuída em Spark `mapInPandas`.

#### [NEW] [spark_privacy_dashboard.py](file:///Users/mferovante/Documents/workspace/SDV/sdv/spark_privacy_dashboard.py)
Módulo que executa agregações distribuídas em Spark SQL e plota o Dashboard de Auditoria Visual (Distribuição DCR, Exact Matches, Risco de MIA).

#### [NEW] [test_spark_privacy_guard.py](file:///Users/mferovante/Documents/workspace/SDV/tests/unit/test_spark_privacy_guard.py)
Suíte de testes de integração com `pyspark.sql.SparkSession` verificando varredura distribuída, hashing, LSH DCR e geração de gráficos.

---

## 4. Plano de Verificação

### Automated Tests (PySpark)
1. **Verificação de Hash & Anti-Join Distribuído**:
   - Criar `SparkDataFrame` real e sintético contendo 1.000.000 de linhas sintéticas com 10 cópias intencionais.
   - Executar `SparkExactMatchFilter` e validar que as 10 cópias exatas são filtradas instantaneamente via hash SHA-256 distribuído no Spark.
2. **Verificação de DCR via LSH**:
   - Rodar `SparkDCRFilter` com `BucketedRandomProjectionLSH` no PySpark ML e verificar a rejeição de registros a uma distância $< \tau_{DCR}$.
3. **Execução de PyTest**:
   - Executar `pytest tests/unit/test_spark_privacy_guard.py` em ambiente local/Spark master.

### Manual / Visual Verification
- Gerar o Dashboard de Auditoria gerado a partir do Spark DataFrame e inspecionar a curva de distribuição DCR e os indicadores de privacidade vs utilidade.
