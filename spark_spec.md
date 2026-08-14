# Integração SDV com PySpark — Especificação Técnica

## Declaração do Problema

A biblioteca SDV (Synthetic Data Vault) gera dados tabulares sintéticos usando modelos generativos profundos (copulas, CTGAN, TVAE). Todo o pipeline — pré-processamento de dados, treinamento de modelos, geração de dados sintéticos e orquestração multi-tabela relacional — opera exclusivamente sobre **pandas DataFrames em um modelo single-process e single-machine**.

Para datasets corporativos com milhões de linhas distribuídas em múltiplas tabelas relacionadas, essa arquitetura esbarra em três limites:

1. **Memória**: o pandas carrega tudo na RAM do driver. Datasets que excedem a memória disponível simplesmente não podem ser processados.
2. **Computação**: o treinamento de modelos (especialmente CTGAN/TVAE com redes neurais) e o loop de fitting de copulas do HMA sobre cada grupo de chave estrangeira rodam sequencialmente. O tempo de execução escala linearmente com o tamanho da tabela.
3. **Throughput de amostragem**: gerar linhas sintéticas para tabelas filhas requer iterar sobre cada linha pai e chamar o modelo sequencialmente — nenhum paralelismo é explorado.

Usuários que já operam em ecossistema Spark (data lakes, Databricks, EMR) não conseguem usar o SDV sem antes extrair seus dados para um formato compatível com pandas, o que anula o propósito de ter uma plataforma de dados distribuída.

---

## Solução

Estender o SDV para **aceitar PySpark DataFrames como entrada de primeira classe** ao longo de todo o pipeline do `HMASynthesizer` (fit → preprocess → augment → train → sample → evaluate), preservando 100% de compatibilidade retroativa com a API baseada em pandas existente. Quando o usuário passa um `dict[str, pyspark.sql.DataFrame]` para `synthesizer.fit()`, a biblioteca automaticamente:

1. Envolve o `DataProcessor` de cada tabela em um novo `SparkDataProcessor` que distribui `transform` / `reverse_transform` pelos executores Spark via `mapInPandas`.
2. Computa extensões de copula do HMA (`_get_extension_spark`) fazendo fitting de copulas por grupo em paralelo via `groupBy().applyInPandas()`.
3. Treina cada modelo single-table (GaussianCopula, CTGAN, TVAE) no driver usando uma amostra representativa coletada via `.toPandas()`.
4. Gera dados sintéticos em paralelo: tabelas raiz via `mapInPandas`, tabelas filhas via `groupBy(parent_key).applyInPandas()` onde cada grupo instancia um sintetizador filho e gera o número apropriado de linhas.
5. Retorna dados sintéticos como `dict[str, pyspark.sql.DataFrame]`, prontos para pipelines Spark downstream.

A mudança na API voltada ao usuário é **zero** — os mesmos métodos `fit()` / `sample()` detectam inputs Spark automaticamente e despacham internamente.

---

## User Stories

1. Como engenheiro de dados, quero passar PySpark DataFrames diretamente para `HMASynthesizer.fit()`, para não precisar materializar todo meu data lake em pandas numa única máquina.

2. Como engenheiro de dados, quero que `synthesizer.sample(scale=1.0)` retorne PySpark DataFrames, para poder escrever dados sintéticos de volta no meu data lake sem uma conversão intermediária para pandas.

3. Como cientista de dados, quero que a integração Spark seja transparente (mesma API do pandas), para não precisar aprender uma API separada ou reescrever meu workflow.

4. Como cientista de dados, quero que o fitting de extensão de copula do HMA rode em paralelo entre grupos de chave estrangeira, para que aumentar uma tabela com 100K chaves únicas de pai não leve horas sequencialmente.

5. Como cientista de dados, quero que a amostragem de tabelas filhas seja paralelizada por linha pai, para que a geração de dados sintéticos relacionais escale com o tamanho do cluster.

6. Como engenheiro de plataforma, quero que o modo Spark do SDV funcione tanto com clusters Spark standalone (master/worker) quanto em modo local (`local[*]`), para poder testar localmente e implantar em clusters de produção sem mudanças de código.

7. Como cientista de dados, quero que `evaluate_quality()` rode nativamente sobre Spark DataFrames sem converter para pandas, para que a validação de datasets sintéticos grandes não esgote a memória do driver.

> [!IMPORTANT]
> **Decisão de design**: adotaremos a abordagem de **wrapper distribuído** — um módulo `SparkQualityEvaluator` que computa estatísticas por partição via agregações Spark (histogramas, contagens, momentos estatísticos, correlações) e agrega os resultados no driver como escalares. Isso evita ao máximo o uso de pandas e permite avaliar datasets com milhões de linhas sem materializar tudo em memória. O `.toPandas()` será restrito apenas a datasets de diagnóstico pequenos (< 10K linhas) ou como fallback explícito.
>
> **Estratégia de implementação do wrapper**:
> - **Column Shapes** (KSComplement, TVComplement): computar distribuições empíricas (histogramas com bins fixos) via `groupBy` + `count` no Spark; comparar distribuições no driver com KS ou TVD sobre os histogramas agregados
> - **Column Pair Trends** (CorrelationSimilarity, ContingencySimilarity): computar momentos (média, variância, covariância) via `agg()` Spark para colunas numéricas; para categóricas, computar tabelas de contingência via `groupBy().count()`
> - **Cardinality**: computar `countDistinct(FK)` por tabela via Spark — puramente distribuído, sem pandas
> - **Intertable Trends**: desnormalizar tabelas filhas com pais via `join` Spark, depois aplicar as mesmas agregações distribuídas acima

7b. Como cientista de dados, quero que o wrapper distribuído produza scores compatíveis com os mesmos scores do sdmetrics (mesmo range, mesma semântica), para poder comparar resultados entre avaliações locais (pandas) e distribuídas (Spark) sem ambiguidade.

8. Como engenheiro de dados, quero que a biblioteca trate tabelas filhas com múltiplas chaves estrangeiras (ex: `dispositions` ligada tanto a `clients` quanto a `accounts`), para que esquemas relacionais complexos funcionem corretamente.

9. Como engenheiro de dados, quero que a biblioteca trate tabelas filhas vazias graciosamente (0 linhas após enforcement), para que o pipeline não quebre em datasets relacionais esparsos.

10. Como cientista de dados, quero que o pré-processamento de dados (codificação, escala, fitting de transformadores) aconteça sobre uma amostra representativa no driver, para que os parâmetros estatísticos sejam consistentes em todas as partições.

11. Como engenheiro de plataforma, quero que o `SparkDataProcessor` seja serializável (picklable) para broadcast aos executores, para que closures do `mapInPandas` possam acessar o preprocessador fitted sem crashar.

12. Como engenheiro de dados, quero testes de integração baseados em Docker com um cluster Spark real (master + worker), para validar execução distribuída antes de implantar em produção.

13. Como cientista de dados, quero que modelos CTGAN e TVAE sejam treináveis via `TorchDistributor` no Spark, para que clusters GPU possam ser aproveitados para treinamento de redes neurais.

14. Como engenheiro de dados, quero que a amostragem de tabelas raiz preserve colunas de extensão do HMA (ex: `__child__fk__num_rows`, `__child__fk__univariates__*`) na saída gerada, para que a amostragem de filhos possa usá-las para determinar quantas linhas gerar por pai.

15. Como cientista de dados, quero que as instâncias de `FloatFormatter` para colunas estendidas sejam pré-fitted no driver antes da augmentação completar, para que `_extract_parameters` durante a amostragem possa clipar valores corretamente.

16. Como engenheiro de dados, quero que colunas de chave estrangeira sejam corretamente tipadas em operações de join Spark, para que incompatibilidades de tipo entre `int32` do pandas e `LongType` do Spark não produzam silenciosamente valores FK nulos.

17. Como engenheiro de plataforma, quero que todos os imports do PySpark sejam opcionais (protegidos com try/except), para que a biblioteca continue funcionando em ambientes onde o PySpark não está instalado.

18. Como engenheiro de dados, quero que a restrição `requires-python` seja `<3.13`, para que a resolução de dependências do `uv`/`pip` não falhe tentando satisfazer `numpy>=2` para Python 3.14+ (que conflita com `pomegranate<1`).

19. Como cientista de dados, quero que seeds aleatórias em workers paralelos sejam derivadas de entropia local do sistema (não uma seed global fixa), para que cada partição gere linhas sintéticas únicas em vez de duplicatas.

20. Como engenheiro de plataforma, quero que o harness de teste suporte configurar o master Spark via variável de ambiente `SPARK_MASTER_URL`, para que o mesmo script de teste funcione em modo local, Docker e clusters remotos.

---

## Decisões de Implementação

### Arquitetura: Despacho de Caminho Duplo

A decisão arquitetural central é **despacho em tempo de execução nos pontos de entrada** (`fit()`, `preprocess()`, `sample()`). Quando dados de entrada são detectados como PySpark DataFrames (via `is_spark_dataframe()`), a execução é roteada para variantes de método `_spark`. Isso evita qualquer refatoração do caminho pandas existente e mantém o conjunto de mudanças mínimo.

A função de detecção `is_spark_dataframe()` usa duck-typing no nome da classe e caminho do módulo (`type(data).__name__ == 'DataFrame' and module.startswith('pyspark.sql')`) para evitar importar PySpark no carregamento do módulo.

### SparkDataProcessor: Padrão Wrapper

Em vez de modificar o `DataProcessor` para tratar tanto pandas quanto Spark, uma nova classe `SparkDataProcessor` envolve uma instância interna do `DataProcessor`. Esta decisão foi tomada porque:

- O `DataProcessor` é profundamente acoplado a internals do pandas (RDT, HyperTransformer)
- O `DataProcessor` interno é fitted numa amostra do lado do driver (até 10K linhas) e broadcastado para executores
- `transform()` e `reverse_transform()` usam `mapInPandas` — cada partição recebe um chunk pandas, processa através do `DataProcessor` interno e retorna o resultado
- `reverse_transform()` bifurca: inputs pandas vão direto para o processador interno (usado por sintetizadores filhos durante amostragem HMA); inputs Spark usam o caminho distribuído
- `__getattr__` delega todos os atributos desconhecidos para o processador interno, com um guard de recursão para `_data_processor` para prevenir recursão infinita durante unpickling de variáveis broadcast

A inferência de schema usa um dry-run em 1 linha coletada no driver, depois mapeia dtypes numpy para tipos Spark (`np.integer → LongType`, `np.floating → DoubleType`, etc.).

### Augmentação HMA no Spark

O método `_augment_tables` detecta Spark DataFrames e despacha para `_augment_table_spark`, que:

1. Processa recursivamente tabelas filhas de baixo para cima (mesmo que o caminho pandas)
2. Para cada tabela filha e chave estrangeira, chama `_get_extension_spark` que usa `groupBy(foreign_key).applyInPandas()` para fazer fitting de uma copula por grupo FK **em paralelo** através dos executores
3. A closure por grupo instancia um sintetizador novo, chama `fit_processed_data` e extrai parâmetros como um DataFrame pandas de uma única linha
4. Resultados de extensão são unidos de volta à tabela pai via `left join` na chave estrangeira
5. Instâncias de `FloatFormatter` para colunas estendidas são fitted numa amostra de 10K linhas do lado do driver, permitindo clipping adequado de valores durante amostragem subsequente
6. Estatísticas (`max_rows`, `min_rows`, `null_fk_percentages`) são computadas via agregações Spark (`spark_max`, `spark_min`, `spark_sum`)

### Amostragem Hierárquica no Spark

O método `_sample_spark` em `BaseHierarchicalSampler`:

1. Amostra tabelas raiz usando `BaseSingleTableSynthesizer._sample_spark` com `keep_extra_columns=True` (preserva colunas `__child__fk__num_rows` e `__child__fk__univariates__*` que o modelo aprendeu)
2. Chama recursivamente `_sample_children_spark` para cada tabela filha
3. Amostragem de filhos usa `groupBy(parent_key).applyInPandas()` — cada grupo recebe uma linha pai, lê `num_rows` da coluna de extensão, cria um sintetizador filho via `_recreate_child_synthesizer` e gera `num_rows` linhas sintéticas filhas
4. Valores NaN em `num_rows` (de left joins onde um pai não tinha filhos no treino) são tratados como 0
5. `_finalize_spark` itera sobre todos os relacionamentos do metadata e adiciona quaisquer colunas FK faltantes via atribuição round-robin do pool de chaves do pai (trata tabelas filhas com múltiplas FKs como `dispositions`)

### Amostragem Spark Single-Table

`BaseSingleTableSynthesizer._sample_spark`:

1. Cria um DataFrame Spark esqueleto (`spark.range(num_rows)`) com particionamento configurável (alvo: 100K linhas por partição)
2. Broadcastia o modelo fitted e o data processor para executores
3. Cada partição gera sua alocação de linhas via `mapInPandas`: chama `model.sample(count)`, depois `reverse_transform` no `DataProcessor` do driver
4. Quando `keep_extra_columns=True`, colunas presentes na saída do modelo mas ausentes da saída do reverse-transform são preservadas (colunas de extensão HMA)
5. Seeds aleatórias em cada partição derivam de `time.time() * 1000` para garantir linhas sintéticas únicas entre partições

### Treinamento Distribuído CTGAN/TVAE

Uma função `_fit_spark_pytorch` a nível de módulo:

1. Escreve dados processados em Parquet via Spark
2. Usa `TorchDistributor(num_processes=2)` para lançar treinamento
3. Dentro da função de treinamento, lê o Parquet de volta como pandas e treina o modelo normalmente
4. Rank 0 salva o modelo via `torch.save`; o driver carrega de volta

> **Nota**: Este é um caminho protótipo. O treinamento em si ainda não é DDP-paralelo — cada processo do TorchDistributor treina independentemente. O caminho de upgrade é envolver CTGAN/TVAE em `DistributedDataParallel`.

### Enforcement de Tamanho de Tabela no Spark

`_enforce_table_size_spark`:

1. Coleta a coluna `num_rows` para o driver como um DataFrame pandas
2. Aplica o mesmo algoritmo de clipping e redistribuição min/max do caminho pandas
3. Cria um DataFrame Spark a partir do DataFrame pandas atualizado com **schema explícito** (compatibilizando o tipo da chave primária do pai com `LongType`/`IntegerType`) para evitar falhas de join por incompatibilidade de tipos
4. Faz join do num_rows atualizado de volta ao DataFrame Spark pai

### População de Colunas FK no Finalize

`_finalize_spark` trata o passo final sobre todos os relacionamentos:

1. Para cada relacionamento, verifica se a coluna FK existe no DF filho
2. Se ausente e filho tem linhas: atribui valores FK via round-robin do pool de chaves do pai, usando window function `row_number()` para ordenação sequencial determinística
3. Se ausente e filho tem 0 linhas: adiciona a coluna FK como literal nulo (`F.lit(None).cast(pk_spark_type)`) para preservar o schema para consumidores downstream
4. Seleção final de colunas filtra apenas colunas declaradas no metadata que realmente existem no DF

### Dependências e Compatibilidade

- Todos os imports PySpark estão em blocos `try/except` com fallbacks `None`
- `requires-python` restrito a `>=3.9,<3.13` para evitar conflito `pomegranate<1` vs `numpy>=2` no Python 3.14+
- Setup Docker: `Dockerfile` baseado em `apache/spark:latest`, instala SDV via `pip install .`
- `docker-compose.yml`: 3 serviços (spark-master, spark-worker, test-runner) com variável de ambiente `SPARK_MASTER_URL`
- **Ignorar Validação Comercial de Complexidade**: O `HMASynthesizer` open-source impõe por padrão um limite rígido de complexidade (`num_tables > 5` ou `schema_depth > 2`) no método `_validate_schema_complexity()`, visando incentivar o upgrade para a versão Enterprise comercial. Como esse bloqueio é puramente comercial (não há limitantes técnicos ou matemáticos de arquitetura), esta especificação formaliza a decisão de **desativar (bypass)** esse limite, permitindo modelar esquemas relacionais complexos inteiros no Spark (como as 8 tabelas do dataset Czech Financial/Berka).

---

## Decisões de Teste

### O Que Torna um Bom Teste

Testes devem validar **comportamento externo** — o pipeline produz um dataset sintético válido com schema correto e integridade referencial? Detalhes internos como contagem de partições, mecânica de broadcast ou schemas intermediários de DataFrame são detalhes de implementação que não devem ser testados diretamente.

### Teste de Integração End-to-End

A validação primária é `test_czech_financial_spark.py`, que:

1. Baixa o dataset Czech Financial (Berka) do Kaggle
2. Carrega 3 tabelas relacionadas (clientes, contas, disposições) com estrutura relacional real
3. Converte para Spark DataFrames
4. Chama `HMASynthesizer.fit(spark_data)` → valida que o fit completa sem erro
5. Chama `synthesizer.sample(scale=1.0)` → valida que todas as 3 tabelas são retornadas como Spark DataFrames
6. Converte para pandas e roda `evaluate_quality()` → valida que o relatório de qualidade do sdmetrics gera com sucesso
7. Imprime scores de qualidade (Column Shapes, Cardinality, etc.)

### Teste em Cluster Docker

O mesmo script de teste roda dentro do Docker via `docker compose up --build`, conectando a um cluster Spark real (master+worker) via `SPARK_MASTER_URL=spark://spark-master:7077`. Isso valida que broadcast, `mapInPandas` e `applyInPandas` funcionam através de fronteiras JVM, não apenas em modo local.

### Precedentes

A suíte de testes existente do SDV usa pytest com fixtures pandas. O teste de integração Spark segue um padrão similar mas roda como script standalone devido à necessidade de uma SparkSession e infraestrutura Docker opcional.

---

## Fora de Escopo

- **Treinamento DDP-paralelo de CTGAN/TVAE**: A integração com `TorchDistributor` é um protótipo que treina em um único processo. Treinamento verdadeiramente distribuído com data-parallel requer envolver o modelo em `DistributedDataParallel` e particionar batches de treinamento entre workers.
- **Spark Connect / Databricks Connect**: A implementação atual usa `SparkSession` clássico com variáveis broadcast. Spark Connect (serverless) tem restrições de serialização diferentes.
- **Streaming / fit incremental**: O pipeline assume processamento batch. Integração com Structured Streaming não é abordada.
- **Enforcement de unicidade de chave primária**: Tabelas raiz amostradas podem produzir chaves primárias duplicadas. O caminho de upgrade é usar `monotonically_increasing_id()` ou um gerador de chaves customizado.
- **Colisão de chaves baseadas em sequência**: `reverse_transform` com `reset_keys=True` usa geração de ID baseada em sequência que pode colidir entre partições. O caminho de upgrade é coordenação de offset ou geração de ID nativa do Spark.
- **Testes unitários para métodos Spark individuais**: Apenas testes de integração end-to-end são fornecidos. Testes unitários isolados para `SparkDataProcessor`, `_get_extension_spark`, etc. são adiados.
- **Integração com catálogo Spark SQL**: Tabelas sintéticas são retornadas como DataFrames em memória, não registradas em Hive metastore ou Unity Catalog.

---

## Notas Adicionais

### Características de Performance

| Estágio | Modelo de Paralelismo |
|---------|----------------------|
| Pré-processamento (`transform`) | `mapInPandas` — nível de partição |
| Augmentação HMA (fitting de copula) | `groupBy(FK).applyInPandas` — nível de grupo FK |
| Treinamento de modelo single-table | Apenas driver (coletado para pandas) |
| Amostragem de tabela raiz | `mapInPandas` — nível de partição |
| Amostragem de tabela filha | `groupBy(parent_key).applyInPandas` — nível de linha pai |
| Reverse transform | `mapInPandas` — nível de partição |

### Arquivos Alterados (8 arquivos, +649 / -8 linhas)

| Arquivo | Tipo de Mudança | Propósito |
|---------|----------------|-----------|
| `sdv/_utils.py` | Modificado | Adicionou helper `is_spark_dataframe()` com duck-typing |
| `sdv/data_processing/spark_data_processor.py` | **Novo** | Wrapper `SparkDataProcessor` com transform/reverse_transform distribuído via `mapInPandas` |
| `sdv/data_processing/data_processor.py` | Modificado | Ampliou tratamento de exceções no `transform` (auxílio de debug) |
| `sdv/multi_table/base.py` | Modificado | Adicionou despacho `_fit_spark`, `_preprocess_spark`; `sample()` com awareness Spark |
| `sdv/multi_table/hma.py` | Modificado | Adicionou `_augment_table_spark`, `_get_extension_spark`; `_augment_tables` e `_pop_foreign_keys` com awareness Spark |
| `sdv/sampling/hierarchical_sampler.py` | Modificado | Adicionou `_sample_spark`, `_sample_children_spark`, `_enforce_table_size_spark`, `_finalize_spark` |
| `sdv/single_table/base.py` | Modificado | Adicionou `_fit_spark`, `_preprocess_spark`, `_sample_spark` com suporte a `keep_extra_columns` |
| `sdv/single_table/ctgan.py` | Modificado | Adicionou `_fit_spark_pytorch` para CTGAN/TVAE via `TorchDistributor` |

### Tetos Conhecidos (notas ponytail no código)

- **Atribuição round-robin de FK**: `_finalize_spark` atribui FKs ciclando pelas chaves do pai, sem preservar semântica relacional. Upgrade: propagar FK dentro de `_sample_children_spark` para todos os relacionamentos.
- **`row_number()` com `orderBy(lit(0))`**: Ordenação não-determinística dentro de partições. Upgrade: usar uma chave de ordenação determinística.
- **Seed aleatória de `time.time()`**: Adequada para unicidade mas não reprodutível. Upgrade: derivação de seed com awareness de partição a partir de uma master seed fornecida pelo usuário.
- **Amostra do driver para pré-processamento**: 10K linhas podem não capturar todos os valores categóricos em colunas com alta cardinalidade. Upgrade: coletar valores distintos por coluna antes do fitting.
