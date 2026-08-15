# PySpark Synthetic Data Privacy Guard — Especificação Técnica

## Declaração do Problema

A geração de dados sintéticos usando modelos generativos e estatísticos (Copulas, CTGAN, TVAE, HMA, PAR) apresenta um risco crítico de **memorização e vazamento de dados de treinamento**. Em cenários corporativos envolvendo dados pessoais ou regulados (LGPD, GDPR, HIPAA, Open Finance), um modelo pode reproduzir linhas reais inteiras ou registros "quase idênticos", resultando em violações de privacidade e sanções legais.

Além disso, em ambientes de Big Data onde os datasets contêm milhões ou bilhões de linhas distribuídas em estruturas tabulares simples, esquemas relacionais multi-tabela ou séries temporais, as verificações de privacidade tradicionais em máquina única (pandas/scikit-learn) falham por estouro de memória no driver (Out-Of-Memory / OOM) e complexidade computacional impraticável ($O(N \times M)$ cross-joins).

Todas as operações de varredura, geração de hash, cálculos de matrizes de distância e validação estatística devem ser executadas nativamente e de forma 100% distribuída no **Apache Spark (PySpark SQL / ML)**.

---

## Análise de Opções Avaliadas

Para resolver o problema de garantia de ausência de dados reais no dataset sintético em escala Big Data, foram avaliadas 4 alternativas técnicas principais:

### Opção 1: Filtro Estrito por Correspondência Exata (Exact Match Filter) em Pós-Processamento

* **Descrição**: Após a geração do dataset sintético, é executada uma comparação de igualdade 100% estrita contra o dataset real de treino para eliminar linhas que sejam exatamente iguais em todas as colunas.
* **Vantagens**:
  * Implementação simples e de altíssimo desempenho no Spark (`sha2` + `left_anti` join).
  * Preserva 100% da fidelidade e utilidade dos dados sintéticos que não forem duplicatas literais.
* **Desvantagens**:
  * **Insuficiente para Privacidade**: Não protege contra memorização parcial ou overfitting. Se o modelo sintetizar uma linha trocando apenas 1 valor por uma diferença ínfima (ex: idade de 40 para 40.001), o registro passa no filtro, mas a privacidade do indivíduo foi violada.
  * Vulnerável a Ataques de Inferência de Membros (Membership Inference Attacks - MIA).

---

### Opção 2: Exact Match + Filtro de Distância Mínima ao Registro Mais Próximo (DCR)

* **Descrição**: Elimina cópias exatas e também calcula a distância de cada registro sintético ao registro real mais próximo no espaço multidimensional de atributos. Linhas sintéticas com distância $DCR < \tau_{DCR}$ são rejeitadas e reamostradas.
* **Vantagens**:
  * Impede a memorização de dados "quase idênticos" e evita o overfitting.
  * Não degrada a capacidade do gerador durante a fase de modelagem/treinamento.
* **Desvantagens**:
  * Se executado com cross-join ingênuo em Big Data, possui complexidade $O(N \times M)$, tornando-se inviável sem aproximação distribuída.
  * Não oferece uma garantia matemática formal como a Privacidade Diferencial (não cobre leakage estatístico global).

---

### Opção 3: Privacidade Diferencial (Differential Privacy - DP) Pura no Treinamento

* **Descrição**: Adiciona ruído matemático controlado durante o treinamento dos modelos (ex: DP-SGD em redes neurais ou perturbando matrizes de covariância em Copulas). Garantia matemática baseada em parâmetros $(\epsilon, \delta)$.
* **Vantagens**:
  * Oferece prova matemática formal de privacidade: a presença ou ausência de qualquer indivíduo no treino não altera a saída do modelo além de um limite $\epsilon$.
  * Protege contra qualquer tipo de ataque estatístico ou inferência de membros.
* **Desvantagens**:
  * **Trade-off de Fidelidade**: Quanto menor o $\epsilon$ (maior privacidade), maior a degradação da qualidade/utilidade dos dados sintéticos.
  * **Não impede duplicatas aleatórias em pós-processamento**: Por ser uma garantia probabilística global, em distribuições densas, o modelo DP ainda pode, por pura amostragem estatística, gerar uma linha idêntica a um dado real.

---

### Opção 4 (Escolhida): Abordagem Híbrida Completa (Privacy-by-Design em Spark)

* **Descrição**: Integração em 4 camadas distribuídas:
  1. Treinamento do gerador com **Privacidade Diferencial** ($\epsilon, \delta$) para garantia estatística formal.
  2. Filtro Spark de **Exact Match** distribuído via hash SHA-256 e `left_anti` join.
  3. Filtro Spark de **Distância Mínima (DCR)** usando **Locality Sensitive Hashing (LSH)** no PySpark ML para escalabilidade $O(N)$.
  4. **Dashboard Visual de Auditoria** alimentado por agregações distribuídas do Spark.
* **Vantagens**:
  * **Defesa em Profundidade**: Garante tanto a prova matemática estatística (DP) quanto a certeza prática (zero cópias exatas e zero registros quase idênticos).
  * **Escalabilidade Big Data**: O uso de LSH no PySpark ML reduz a busca de vizinhos mais próximos de $O(N \times M)$ para $O(N)$ distribuído.
  * **Tratamento Inteligente de Colisões Naturais**: Distingue colisões categóricas triviais de baixa cardinalidade (ex: Sexo=M, UF=SP) de vazamentos em vetores completos de atributos.
* **Desvantagens**:
  * Exige maior complexidade arquitetural e integração com o ecossistema PySpark ML.

---

## Solução Escolhida & Detalhamento das Especificações Técnicas

A **Opção 4 (Abordagem Híbrida Completa)** foi a selecionada. Abaixo está a especificação técnica detalhada da arquitetura e dos seus componentes em PySpark.

```
[ Dataset Treino Real (PySpark DataFrame) ]    [ Dataset Sintético Amostrado (PySpark DataFrame) ]
                        │                                          │
                        └────────────────────┬─────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Treinamento do Modelo com Privacidade Diferencial (DP-SGD / DP-Copula)             │
│    - Orçamento ε = 1.0, δ = 1/N configurável por experimento                           │
└────────────────────────────────────────────┬───────────────────────────────────────────┘
                                             │
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. SparkExactMatchFilter (Geração de Hash SHA-256 & Anti-Join Distribuído)             │
│    - sha2(concat_ws('||', *cols), 256) sobre colunas de alta cardinalidade/QIDs        │
│    - left_anti join no Spark DataFrame para purga instantânea                          │
└────────────────────────────────────────────┬───────────────────────────────────────────┘
                                             │ (Linhas aprovadas)
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. SparkDCRFilter (Distância Mínima via PySpark ML LSH)                                │
│    - Spark VectorAssembler + BucketedRandomProjectionLSH / MinHashLSH                  │
│    - Junção de similaridade LSH (approxSimilarityJoin) com threshold τ_DCR             │
│    - Rejeição e reamostragem distribuída de viés de memorização                       │
└────────────────────────────────────────────┬───────────────────────────────────────────┘
                                             │ (Dataset Sintético Seguro)
                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. SparkPrivacyDashboard (Auditoria e Métricas Visualizadas)                           │
│    - Agregações distribuídas Spark SQL (Histogramas DCR, Contagens, MIA Score)         │
│    - Geração de Dashboard em formato HTML/PNG com gráficos de utilidade vs privacidade │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Histórias de Usuário (User Stories)

1. Como oficial de privacidade, quero garantir que nenhuma cópia exata de linhas de treino originais esteja presente no dataset sintético gerado, para que nossa organização cumpra as regulamentações da LGPD/GDPR.
2. Como engenheiro de dados, quero que toda a geração de hashes e comparações de correspondência exata rodem nativamente em PySpark DataFrames usando anti-joins distribuídos, para que datasets com centenas de milhões de linhas sejam varridos sem erros de memória no driver.
3. Como cientista de dados, quero que registros sintéticos quase idênticos sejam filtrados usando limites de Distância ao Registro Mais Próximo (DCR), para que amostras memorizadas ou superajustadas (overfitted) sejam rejeitadas automaticamente.
4. Como engenheiro de aprendizado de máquina, quero que os cálculos de distância DCR sejam paralelizados entre os nós workers do Spark utilizando Locality Sensitive Hashing (LSH), para que a busca por vizinhos mais próximos escale em tempo $O(N)$ em vez de usar cross-joins caros.
5. Como engenheiro de privacidade, quero configurar os parâmetros de Privacidade Diferencial (orçamento $\epsilon$ e probabilidade de falha $\delta$) durante o treinamento do modelo, para que eu possa controlar a garantia matemática de privacidade por experimento.
6. Como analista de conformidade, quero que combinações categóricas de baixa cardinalidade (ex: Sexo = Masculino, Estado = SP) sejam reconhecidas como colisões estatísticas naturais e não como vazamentos de dados, para que amostras sintéticas válidas não sejam descartadas desnecessariamente.
7. Como engenheiro de dados trabalhando com esquemas multi-tabela, quero que o privacy guard compute o DCR estrutural através de relacionamentos de chave primária e estrangeira no Spark, para que vazamentos em dados relacionais sejam evitados.
8. Como engenheiro de dados trabalhando com dados sequenciais, quero que o privacy guard compute métricas de distância em janelas móveis de séries temporais usando Funções de Janela do Spark (Window Functions), para que a memorização de trajetórias seja detectada.
9. Como cientista de dados, quero um dashboard visual de auditoria exibindo gráficos de distribuição de DCR e curvas de trade-off entre utilidade e privacidade, para que eu possa verificar e documentar a postura de privacidade antes de implantar os dados sintéticos.
10. Como engenheiro de DevOps, quero uma função de asserção automatizada em Python que levante um erro de vazamento de dados durante a execução do pipeline de CI/CD se exact matches ou linhas com baixo DCR forem detectadas, para que datasets sintéticos inseguros sejam bloqueados de irem para produção.
11. Como auditor de segurança, quero uma pontuação de risco de Ataques de Inferência de Membros (Membership Inference Attack / MIA Score) incluída no resumo de auditoria de privacidade, para que eu possa avaliar a probabilidade empírica de reidentificação por um adversário.
12. Como engenheiro de plataforma, quero que todos os módulos de privacidade PySpark executem de forma transparente tanto em sessões Spark locais (`local[*]`) quanto em clusters distribuídos de produção (EMR, Databricks, GCP Dataproc), para que os testes locais espelhem a execução em produção.

---

## Decisões de Implementação

### 1. Módulo `SparkExactMatchFilter`
- **Algoritmo de Hash**: Utiliza a função SQL `sha2` nativa do Spark sobre a concatenação das colunas selecionadas com separador nulo seguro (`||`).
- **Execução**:
  ```python
  # Representação da lógica de hash distribuída no Spark
  hash_expr = F.sha2(F.concat_ws("||", *[F.coalesce(F.col(c).cast("string"), F.lit("__NULL__")) for c in cols]), 256)
  df_real_hashed = df_real.withColumn("__hash__", hash_expr).select("__hash__").distinct()
  df_syn_hashed = df_syn.withColumn("__hash__", hash_expr)
  
  # Purga distribuída via anti-join
  df_syn_clean = df_syn_hashed.join(df_real_hashed, on="__hash__", how="left_anti").drop("__hash__")
  ```

### 2. Módulo `SparkDCRFilter` com PySpark ML LSH
- **Montagem do Vetor**: As colunas numéricas e categóricas (após `StringIndexer` e `OneHotEncoder`) são consolidadas em uma coluna de vetores `features` via `VectorAssembler`.
- **Indexação por LSH**:
  - Para atributos contínuos: `BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=bucket_size)`.
  - Para atributos categóricos/binários: `MinHashLSH(inputCol="features", outputCol="hashes")`.
- **Similarity Join Distribuído**:
  ```python
  # Junção por similaridade em tempo O(N) distribuído
  pairs = lsh_model.approxSimilarityJoin(df_syn_vec, df_real_vec, threshold=tau_dcr, distCol="dcr_distance")
  # Identificação das chaves sintéticas com dcr_distance < tau_dcr para descarte
  violating_ids = pairs.select(F.col("datasetA.__id__")).distinct()
  df_syn_safe = df_syn.join(violating_ids, on="__id__", how="left_anti")
  ```

### 3. Tratamento de Estruturas Relacionais e Séries Temporais
- **Relacional (HMA Multi-Table)**: DCR estrutural computado por entidade pai agregando estatísticas dos filhos via `groupBy(parent_id).agg(...)` antes do cálculo de vetor.
- **Séries Temporais (PAR)**: Extração de vetores de janela deslizante usando `Window.partitionBy(entity_id).orderBy(timestamp)` para garantir que sequências temporais memorizadas sejam barradas.

### 4. Módulo `SparkPrivacyDashboard`
- **Agregações no Spark**: Todas as métricas pesadas (bins de histograma DCR, percentis 1%, 5%, 50%, estatísticas de MIA) são computadas no cluster Spark usando `groupBy` e `expr("percentile_approx(...)")`.
- **Renderização**: Coleta de vetores agregados leves (< 100 linhas de bin) para geração de plots via `matplotlib` / `seaborn` e exportação para HTML/PNG.

---

## Decisões de Teste

### Seams de Teste (Costuras Arquiteturais)
1. **Seam do Exact Match Filter**:
   - Injetar registros idênticos em um `SparkDataFrame` sintético e verificar que a saída de `SparkExactMatchFilter.filter()` não contém nenhum dos registros injetados.
2. **Seam do LSH DCR Filter**:
   - Injetar registros com perturbações pequenas ($10^{-5}$) e verificar se a busca por vizinhos aproximados via PySpark ML LSH identifica e descarta os registros violadores.
3. **Seam do Dashboard de Auditoria**:
   - Validar que o `SparkPrivacyDashboard` calcula métricas corretas de DCR mínimo e médio em uma sessão local do Spark.

### Qualidade dos Bons Testes
- Testar estritamente as entradas e saídas de DataFrames (`pyspark.sql.DataFrame`), sem depender de implementação de baixo nível de RDDs ou variáveis internas de partição.

### Arte Anterior (Prior Art)
- Utilização da fixture `spark_session` com `SparkSession.builder.master("local[*]").getOrCreate()` conforme estabelecido em `test_single_table_copula_spark.py` e `spark_spec.md`.

---

## Fora do Escopo

1. Suporte a filtragem de dados sintéticos em tempo real em pipelines de PySpark Structured Streaming (foco em DataFrames em lote / batch).
2. Fallbacks single-machine em Pandas para datasets que não caibam no driver (toda a suíte exige ambiente PySpark).
3. Busca automatizada exaustiva por força bruta de limites $\tau_{DCR}$ em múltiplos terabytes de dados (os limites são definidos por parâmetros configuráveis).

---

## Notas Adicionais

- **Ajuste de Parâmetros LSH**: O comprimento do bucket (`bucketLength`) do `BucketedRandomProjectionLSH` deve ser calibrado com base na variância dos atributos numéricos normalizados para evitar concentrar muitos pontos no mesmo bucket (evitando desbalanceamento/skew de partição no Spark).
