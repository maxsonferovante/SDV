# Estudo de Migração do SDV para PySpark

Este documento apresenta uma análise de design e um roteiro arquitetural para migrar a biblioteca SDV (Synthetic Data Vault) de uma execução local baseada em **Pandas** para uma arquitetura distribuída e escalável baseada em **PySpark DataFrames** (`pyspark.sql.DataFrame`).

---

## 1. Visão Geral da Arquitetura Distribuída

Para garantir alta volumetria a nível empresarial sem gargalos de memória em nós únicos (Single-Node), dividimos o pipeline do SDV em três fases principais escaladas no Spark:

```mermaid
graph TD
    subgraph Entrada
        RawSparkDF[Raw PySpark DataFrame]
    end

    subgraph Pré-processamento (Híbrido)
        FitDriver[Fit no Driver / Amostra] -->|Gera Parâmetros RDT| BroadcastParams[Broadcast do Config]
        RawSparkDF -->|Transformação Distribuída| MapInPandasTransform[mapInPandas: Transform]
    end

    subgraph Treinamento de Modelos
        MapInPandasTransform -->|Deep Learning| TorchDistributor[PyTorch TorchDistributor DDP]
        MapInPandasTransform -->|HMA Relacional| ApplyInPandasFit[applyInPandas: Treino de Mini-Copulas por Grupo]
    end

    subgraph Amostragem (Sampling)
        BroadcastModel[Broadcast do Modelo Treinado] -->|Paralelização de Partição| MapPartitionsSample[mapPartitions / applyInPandas]
        MapPartitionsSample -->|Reverse Transform Distribuído| SyntheticSparkDF[Synthetic PySpark DataFrame]
    end

    RawSparkDF --> FitDriver
    BroadcastParams --> MapInPandasTransform
    TorchDistributor --> BroadcastModel
    ApplyInPandasFit --> BroadcastModel
```

---

## 2. Detalhes de Implementação por Componente

### A. Interface e Entrada/Saída
*   **Abordagem**: Aceitar nativamente e retornar `pyspark.sql.DataFrame` nos métodos de entrada/saída (`fit`, `preprocess`, `sample`).
*   **Vantagem**: Integração perfeita com pipelines de ETL corporativos (Databricks, EMR, GCP Dataproc).

### B. Pré-processamento e RDT (Reversible Data Transforms)
O RDT é altamente acoplado ao Pandas. Para escalá-lo de forma pragmática, adotamos o padrão **Híbrido**:
1.  **Fit (Driver)**: O cálculo das propriedades (como categorias únicas, valores mínimo/máximo, casas decimais para arredondamento) é realizado no nó Driver a partir de uma amostra estatisticamente representativa ou computado usando consultas de agregação otimizadas no Spark SQL (ex: `df.select(mean(c), stddev(c))`).
2.  **Transform / Reverse Transform (Workers)**: As instâncias dos transformadores do RDT são serializadas e enviadas via `broadcast`. A aplicação real sobre bilhões de linhas é executada em paralelo em cada partição usando `.mapInPandas()` (que converte eficientemente partições do Spark DataFrame em Pandas em memória nos workers).

### C. Treinamento de Modelos Distribuído
O gargalo do treinamento varia conforme o tipo de sintetizador:
1.  **Sintetizadores de Deep Learning (CTGAN / TVAE)**:
    *   Utilização do `pyspark.ml.torch.distributor.TorchDistributor` (nativo do Spark 3.4+) para orquestrar o loop de treinamento do PyTorch de forma distribuída (DDP) através dos workers do cluster Spark, permitindo treinamento com uso de GPUs multi-nós.
2.  **Sintetizadores Relacionais (HMA - Hierarchical Modeling Algorithm)**:
    *   No HMA original, os parâmetros das tabelas filhas são aprendidos iterando sequencialmente sobre cada valor de chave estrangeira (FK) do pai e aplicando um fit local.
    *   **Solução Spark**: Utilizar `child_df.groupby("foreign_key").applyInPandas(...)`. Isso permite paralelizar o treinamento dos milhares de mini-copulas (distribuições filhas) nos workers do Spark em um único passo distribuído.

### D. Amostragem (Synthesis & Sampling)
A geração de dados sintéticos para bilhões de linhas é realizada de forma totalmente distribuída:
1.  **Criação de Partições**: Um DataFrame esqueleto do Spark é criado com o tamanho total desejado e distribuído em partições.
2.  **Geração em Worker**: Cada worker gera dados latentes (ruído aleatório), executa a inferência do modelo gerador (PyTorch/Copulas) e aplica o `reverse_transform` do RDT de forma local e paralela usando `mapPartitions` / `applyInPandas`.
3.  **HMA Relacional**: A amostragem de tabelas filhas (que dependem dinamicamente das contagens geradas na tabela pai) é resolvida particionando o DataFrame da tabela pai gerada. Cada partição lê o modelo broadcasted e gera em lote as linhas filhas correspondentes de forma paralela e independente.

---

## 3. Roteiro Técnico de Desenvolvimento (Roadmap)

### Fase 1: Fundação & Pré-processador Spark (Single-Table)
*   Criar o encapsulamento `SparkDataProcessor` que substitui/envolve o `DataProcessor`.
*   Implementar métodos `fit` (usando amostras ou Spark SQL stats) e `transform` / `reverse_transform` (usando `mapInPandas`).
*   **Validação**: Testar a consistência dos dados transformados comparando Pandas vs Spark.

### Fase 2: Paralelização da Amostragem (Single-Table)
*   Distribuir a geração de dados latentes e a aplicação dos modelos via `mapPartitions`.
*   Integrar o pós-processamento e aplicação de restrições (constraints) diretamente nos workers.
*   **Validação**: Gerar 100 milhões de linhas sintéticas e validar escalabilidade linear.

### Fase 3: Treinamento Distribuído com PyTorch
*   Migrar os loops de treinamento do `CTGANSynthesizer` e `TVAESynthesizer` para usar PyTorch DDP encapsulado em `TorchDistributor`.
*   Ajustar tratamento de hiperparâmetros para contexto distribuído.

### Fase 4: Escalabilidade Relacional (HMA Multi-Table)
*   Reescrever o `HMASynthesizer` para aceitar um dicionário de Spark DataFrames.
*   Implementar a computação das extensões parent-child via `groupby().applyInPandas()` (Fase de Treinamento).
*   Implementar a amostragem relacional distribuída via `mapPartitions` da tabela pai.

---

## 4. Desafios de Engenharia & Mitigações

| Desafio | Impacto | Mitigação |
| :--- | :--- | :--- |
| **Serialização de Modelos PyTorch/RDT** | Falhas na transmissão dos modelos para workers (`PicklingError`). | Utilização do `cloudpickle` para serializar os estados dos modelos e configuração correta do `sparkContext.broadcast`. |
| **Consumo de Memória nos Workers** | OOM (Out Of Memory) se as partições convertidas para Pandas forem muito grandes. | Controle rigoroso do número de partições no Spark. Garantir que o tamanho ideal de partição (ex: 64MB-128MB) caiba com folga na memória RAM do executor do worker. |
| **Dependências de Ambiente** | Necessidade de PyTorch, RDT, e SDV instalados em todos os nós do cluster. | Utilizar imagens Docker para os executores do Spark (Databricks Container Services, EMR Docker, etc.) ou configurar `pip install` via inicialização de cluster (bootstrap). |
