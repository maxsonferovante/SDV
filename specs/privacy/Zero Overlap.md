# Spec: Garantia de 100% de Zero Overlap

## Problem Statement

Ao utilizar o **SDV (Synthetic Data Vault)** para geração de dados sintéticos, embora os modelos probabilísticos e geradores aleatórios minimizem colisões, certas aplicações rígidas de privacidade exigem uma **garantia absoluta de 100% de Zero Overlap** (nenhum registro sintético gerado pode ser idêntico a um registro real da base de treinamento).

Fazer esse controle de forma ad-hoc ou manual por aplicação não é escalável em ambiente de produção. É necessária uma solução arquitetural padronizada para garantir a não-repetição de dados originais.

## Solution

Oferecer duas abordagens arquiteturais escaláveis para garantir 100% de Zero Overlap entre os dados sintéticos amostrados e o dataset original de treino:

1. **Arquitetura A (`ZeroOverlapSynthesizer` - Wrapper Rejection Sampling):** Envelopamento do método `.sample()` do sintetizador que realiza filtragem em lote com busca $O(1)$ contra o conjunto de tuplas do dataset original de treino, amostrando dados adicionais (*rejection sampling*) até atingir a quantidade desejada de registros 100% inéditos.
2. **Arquitetura B (`UniqueFromOriginalDataConstraint` - CAG Nativo SDV):** Restrição programável estendendo `SingleTableProgrammableConstraint` que se integra diretamente ao fluxo de *Constraint-Augmented Generation* (CAG) do SDV, validando e rejeitando duplicações automaticamente durante o processo de geração.

---

## User Stories

1. **Como Engenheiro de Privacidade**, quero garantir 100% de Zero Overlap entre os dados sintéticos gerados e a base de treino real, para evitar qualquer risco de vazamento de dados sensíveis (*data leakage*).
2. **Como Arquiteto de Soluções**, quero reutilizar o `ZeroOverlapSynthesizer` (Arquitetura A) de forma transparente em qualquer sintetizador do SDV, para aplicar a garantia de não-repetição em uma única linha de código.
3. **Como Engenheiro de Machine Learning**, quero integrar a restrição de Zero Overlap nativamente ao pipeline de restrições do SDV via `SingleTableProgrammableConstraint` (Arquitetura B), para aproveitar os mecanismos internos de reamostragem do framework.
4. **Como Administrador de Banco de Dados**, quero uma verificação de sobreposição com complexidade de tempo $O(1)$, para que o processo de amostragem em grande escala permaneça veloz e eficiente.

---

## Implementation Decisions

- **Arquitetura A (`ZeroOverlapSynthesizer`):**
  - Implementação de classe wrapper que encapsula uma instância de `BaseSynthesizer`.
  - Construção de um `set` de tuplas contendo o hash/valores dos registros reais da tabela de treino para buscas instantâneas $O(1)$.
  - Loop de reamostragem (*rejection sampling*) com oversampling de 20% até satisfazer o número exato de linhas limpas requisitado no `.sample()`.

- **Arquitetura B (`UniqueFromOriginalDataConstraint`):**
  - Implementação de subclasse estendendo `SingleTableProgrammableConstraint`.
  - Método `is_valid(table_data)` avaliando cada linha gerada contra o conjunto de tuplas originais e retornando uma `pd.Series` booleana.
  - Suporte à especificação de subconjuntos de colunas ou à tabela inteira para checagem de unicidade.

---

## Testing Decisions

- **Seam de Teste:**
  - Interface externa do método `.sample()` no `ZeroOverlapSynthesizer`.
  - Método `is_valid()` da classe `UniqueFromOriginalDataConstraint`.
- **Boas Práticas de Teste:**
  - Testar apenas o comportamento observável externo e garantias de contrato, sem acoplamento com estruturas internas do RDT.
  - Testar casos de borda: dataset de treino pequeno vs. amostragem grande, 100% de colisão (deve reamostrar adequadamente), 0% de colisão.
- **Artefatos de Referência:**
  - Testes existentes em `tests/unit/cag/test_programmable_constraint.py`.

---

## Out of Scope

- Modificações nas rotinas internas de treinamento dos modelos CTGAN, TVAE ou Copula.
- Adição de ruído de Privacidade Diferencial matemática (DP-SGD/e-differential privacy) durante o gradiente.
- Alterações em esquemas de bancos de dados relacionais externos.

---

## Further Notes

- **Desempenho:** Para colunas de alta entropia (como UUIDv4 e Regexes aleatórias), o número de descartes no *rejection sampling* é virtualmente 0. Para colunas categóricas de baixo domínio de valores, a taxa de descarte pode ser mais alta, exigindo dimensionamento adequado de `max_attempts`.
