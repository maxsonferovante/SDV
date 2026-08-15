# Documento de Requisitos de Produto (PRD) - PySpark Synthetic Data Privacy Guard & Proteção Contra Vazamento de Dados

## Declaração do Problema

Modelos generativos e técnicas estatísticas utilizadas para criar datasets sintéticos (como Copulas, CTGAN, TVAE, HMA e PAR) correm o risco de memorizar ou reproduzir registros exatos ou quase idênticos do dataset original de treinamento. Quando dados sintéticos são gerados para domínios sensíveis (como setor bancário, saúde ou dados pessoais protegidos por LGPD/GDPR), qualquer vazamento de registros originais ou linhas quase duplicadas compromete a privacidade individual e viola a conformidade regulatória.

Além disso, em ambientes de dados corporativos onde os datasets escalam para milhões ou bilhões de linhas em esquemas de tabela única, relacionais multi-tabela e séries temporais, as checagens de privacidade tradicionais em máquina única falham devido a erros de falta de memória (Out-Of-Memory / OOM) e tempos de execução inviáveis. Toda a varredura de privacidade, geração de hashes, cálculos de matrizes de distância e comparações entre datasets devem ser executadas de forma 100% distribuída no Apache Spark.

## Solução

Implementar uma solução de classe enterprise **Spark Synthetic Data Privacy Guard** utilizando uma estratégia de **Abordagem Híbrida Completa (Privacy-by-Design)**. A solução garante que nenhum dado do dataset original esteja presente ou exposto no dataset sintético gerado através da integração de quatro camadas distribuídas:

1. **Privacidade Diferencial (DP) no Treinamento**: Treinar os modelos generativos com parâmetros de privacidade matematicamente limitados ($\epsilon, \delta$) para restringir a influência de qualquer registro individual.
2. **Filtro Spark de Correspondência Exata (Exact Match via SHA-256)**: Realizar varredura distribuída de hash de linha e junções `left_anti` em PySpark DataFrames para eliminar instantaneamente duplicatas exatas em todos os atributos.
3. **Filtro Spark de Distância ao Registro Mais Próximo (DCR via LSH)**: Calcular métricas de distância escaladas de Gower, Euclidiana e DTW entre registros sintéticos e reais de forma distribuída usando Locality Sensitive Hashing (LSH) no PySpark ML, rejeitando linhas que fiquem abaixo de um limite mínimo de distância ($\tau_{DCR}$).
4. **Dashboard Visual de Auditoria de Privacidade no Spark**: Computar métricas distribuídas de privacidade e fidelidade (distribuição de DCR, contagem de Exact Match, Risco de Membership Inference) nos executores Spark e renderizar um dashboard estatístico e visual.

Colisões naturais em atributos categóricos de baixa cardinalidade (ex: Sexo, Estado) são permitidas, enquanto unicidade estrita e limites de distância são impostos sobre o vetor completo de atributos e atributos contínuos/de alta cardinalidade.

## Histórias de Usuário

1. Como oficial de privacidade, eu quero garantir que nenhuma cópia exata de linhas de treino originais esteja presente no dataset sintético gerado, para que nossa organização cumpra as regulamentações da LGPD/GDPR.
2. Como engenheiro de dados, eu quero que toda a geração de hashes e comparações de correspondência exata rodem nativamente em PySpark DataFrames usando anti-joins distribuídos, para que datasets com centenas de milhões de linhas sejam varridos sem erros de memória no driver.
3. Como cientista de dados, eu quero que registros sintéticos quase idênticos sejam filtrados usando limites de Distância ao Registro Mais Próximo (DCR), para que amostras memorizadas ou superajustadas (overfitted) sejam rejeitadas automaticamente.
4. Como engenheiro de aprendizado de máquina, eu quero que os cálculos de distância DCR sejam paralelizados entre os nós workers do Spark utilizando Locality Sensitive Hashing (LSH), para que a busca por vizinhos mais próximos escale em tempo $O(N)$ em vez de usar cross-joins caros.
5. Como engenheiro de privacidade, eu quero configurar os parâmetros de Privacidade Diferencial (orçamento $\epsilon$ e probabilidade de falha $\delta$) durante o treinamento do modelo, para que eu possa controlar a garantia matemática de privacidade por experimento.
6. Como analista de conformidade, eu quero que combinações categóricas de baixa cardinalidade (ex: Sexo = Masculino, Estado = SP) sejam reconhecidas como colisões estatísticas naturais e não como vazamentos de dados, para que amostras sintéticas válidas não sejam descartadas desnecessariamente.
7. Como engenheiro de dados trabalhando com esquemas multi-tabela, eu quero que o privacy guard compute o DCR estrutural através de relacionamentos de chave primária e estrangeira no Spark, para que vazamentos em dados relacionais sejam evitados.
8. Como engenheiro de dados trabalhando com dados sequenciais, eu quero que o privacy guard compute métricas de distância em janelas móveis de séries temporais usando Funções de Janela do Spark (Window Functions), para que a memorização de trajetórias seja detectada.
9. Como cientista de dados, eu quero um dashboard visual de auditoria exibindo gráficos de distribuição de DCR e curvas de trade-off entre utilidade e privacidade, para que eu possa verificar e documentar a postura de privacidade antes de implantar os dados sintéticos.
10. Como engenheiro de DevOps, eu quero uma função de asserção automatizada em Python que levante um erro de vazamento de dados durante a execução do pipeline de CI/CD se exact matches ou linhas com baixo DCR forem detectadas, para que datasets sintéticos inseguros sejam bloqueados de irem para produção.
11. Como auditor de segurança, eu quero uma pontuação de risco de Ataques de Inferência de Membros (Membership Inference Attack / MIA Score) incluída no resumo de auditoria de privacidade, para que eu possa avaliar a probabilidade empírica de reidentificação por um adversário.
12. Como engenheiro de plataforma, eu quero que todos os módulos de privacidade PySpark executem de forma transparente tanto em sessões Spark locais (`local[*]`) quanto em clusters distribuídos de produção (EMR, Databricks, GCP Dataproc), para que os testes locais espelhem a execução em produção.

## Decisões de Implementação

* **Arquitetura Híbrida Privacy-by-Design**: Combinar Privacidade Diferencial no treinamento ($\epsilon = 1.0, \delta = 1/N$) com filtragem distribuída de pós-processamento (Exact Match + DCR) para fornecer defesa em profundidade contra vazamento de dados.
* **Geração de Hash e Anti-Join em PySpark**: Gerar hashes SHA-256 de 256 bits sobre valores de colunas concatenadas usando funções nativas do PySpark SQL. Filtrar registros sintéticos usando junções `left_anti` do Spark contra o índice de hash do conjunto de treinamento.
* **Locality Sensitive Hashing (LSH) para DCR em Grande Escala**: Utilizar `VectorAssembler` e `BucketedRandomProjectionLSH` / `MinHashLSH` do PySpark ML para agrupar vetores de atributos em buckets. Executar junções de similaridade para computar distâncias de vizinhos mais próximos em paralelo, sem materializar cross-joins de ordem $O(N \times M)$.
* **Padronização de Métricas de Distância**: Aplicar normalização da distância de Gower para dados tabulares mistos, distância Euclidiana/Mahalanobis para dados numéricos contínuos e métricas baseadas em janelas de DTW para séries temporais sequenciais.
* **Tratamento de Colisões Naturais**: Excluir colunas categóricas de baixa cardinalidade isoladas de gatilhos de duplicação individual, mantendo a checagem de hash e DCR sobre o vetor de linha completo e sobre colunas contínuas e de alta cardinalidade.
* **Agregação Distribuída para o Dashboard Visual**: Computar contagens de bins de histogramas, quantis de DCR, métricas de exact match e indicadores de utilidade diretamente nos executores Spark via chamadas `groupBy` e `agg` do PySpark SQL, coletando apenas resumos agregados para o driver renderizar os gráficos.

## Decisões de Teste

* **Costuras de Teste (Testing Seams)**:
  * **Costura de Hash e Exact Match**: Verificar se a geração de hash SHA-256 e a junção `left_anti` no PySpark identificam e eliminam com precisão 100% de linhas duplicadas exatas injetadas em um DataFrame sintético do Spark.
  * **Costura de DCR via LSH de Vizinhos Próximos**: Testar se o pipeline de LSH do PySpark ML calcula corretamente as distâncias mínimas e rejeita linhas sintéticas criadas com micro-perturbações abaixo do limite de DCR ($\tau_{DCR}$).
  * **Costura de Pipeline Ponta a Ponta**: Validar o fluxo completo (Treinamento com DP $\to$ Amostragem $\to$ Filtro de Exact Match $\to$ Filtro de DCR $\to$ Dashboard de Auditoria) sobre um PySpark DataFrame sintético multi-coluna.
* **Qualidade dos Bons Testes**: Os testes devem assertar comportamento externo (ex: DataFrame de entrada contendo linha com vazamento $\to$ DataFrame de saída com a linha removida) em vez de inspecionar partições internas de RDD ou estado interno de classes.
* **Arte Anterior**: Espelhar os padrões de teste PySpark existentes no diretório `tests/`, utilizando uma fixture compartilhada de `SparkSession` local (`local[*]`).

## Fora do Escopo

* Fallback de execução em memória única com Pandas para datasets que excedam a RAM do driver (todas as checagens neste escopo visam PySpark DataFrames).
* Filtragem de privacidade em tempo real para dados em streaming (o escopo inicial abrange PySpark DataFrames em lote/batch).
* Ajuste automático de thresholds via busca em grade por força bruta sobre volumes massivos de dados (os limites são configuráveis pelo usuário com padrões recomendados).

## Notas Adicionais

* Ao configurar o número de buckets do LSH, garantir que a contagem de buckets escale adequadamente com os nós executores do cluster para evitar desbalanceamento de partições (skew) durante as junções de similaridade em grande massa de dados.
