Searched for "pii"
Viewed single_table.py:1-100
Viewed data_processor.py:260-300
Viewed data_processor.py:640-750
Searched for "_transformers_by_sdtype"
Viewed single_table.py:470-500

No **SDV (Synthetic Data Vault)**, a marcação de colunas com **PII** (*Personally Identifiable Information*) e a definição do **`sdtype`** (*Semantic Data Type*) são fundamentais para garantir tanto a qualidade dos dados sintéticos quanto a **privacidade** de informações sensíveis.

O processo é dividido entre a **camada de metadados do SDV** (que classifica e define o significado semântico) e a biblioteca **RDT (Reversible Data Transforms)** (que realiza a transformação matemática e a geração de dados fakes).

---

### 1. O que é `sdtype` e a marcação de PII no SDV

O **`sdtype`** indica o tipo semântico do dado contido na coluna. Existem dois grandes grupos de `sdtype`:

1. **Tipos Estatísticos / Modeláveis (Não-PII):** `numerical`, `categorical`, `datetime`, `boolean`, `id`.
   - *Objetivo:* O modelo sintético (ex: CTGAN, TVAE, GaussianCopula) aprende a distribuição estatística e as correlações desses dados para gerar novas amostras parecidas com as reais.
2. **Tipos de PII (Anonimização):** `email`, `phone_number`, `first_name`, `last_name`, `ssn`, `address`, `credit_card_number`, entre outros.
   - *Objetivo:* Proteger a privacidade. O modelo de aprendizado **não deve memorizar nem aprender distribuições sobre dados pessoais reais**.

#### Como a marcação é feita no Metadata (`SingleTableMetadata`)

A definição/marcação pode ocorrer de duas formas:

* **Detecção Automática (`detect_from_dataframe`):**
  Ao chamar `metadata.detect_from_dataframe(data)`, o SDV analisa os nomes das colunas usando heurísticas e tokenização (por regex e camelCase). Se o nome da coluna coincidir com padrões conhecidos de PII (ex: `user_email`, `first_name`, `cpf`), o SDV automaticamente atribui o `sdtype` específico e marca a propriedade `pii = True`.

* **Configuração Manual (`update_column`):**
  Você pode definir ou alterar o `sdtype` e a flag `pii` explicitamente:
  ```python
  from sdv.metadata import SingleTableMetadata

  metadata = SingleTableMetadata()
  metadata.detect_from_dataframe(data)

  # Atualizando uma coluna para ser tratada como PII
  metadata.update_column(
      column_name='email_usuario',
      sdtype='email',
      pii=True
  )
  ```

---

### 2. Como o SDV conecta o `sdtype` com o RDT (`DataProcessor`)

Quando você inicializa um sintetizador no SDV e chama o método `.fit(data)`, o SDV utiliza uma camada interna chamada **`DataProcessor`** para configurar o **`HyperTransformer`** da biblioteca **RDT**.

O RDT é a engine responsável por transformar cada coluna do Pandas DataFrame em algo que o modelo sintético consiga processar e, depois, reverter na fase de amostragem (`sample`).

```
                              ┌────────────────────────────────────────┐
                              │         SingleTableMetadata            │
                              │ (coluna='email', sdtype='email', pii=True)│
                              └───────────────────┬────────────────────┘
                                                  │
                                                  ▼
                              ┌────────────────────────────────────────┐
                              │             DataProcessor              │
                              └───────────────────┬────────────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
   Coluna PII (pii=True)                                          Coluna Normal (pii=False)
 ┌──────────────────────────────────────┐                       ┌──────────────────────────────────────┐
 │ Transformer RDT:                     │                       │ Transformer RDT:                     │
 │ rdt.transformers.pii.AnonymizedFaker │                       │ FloatFormatter / FrequencyEncoder... │
 └──────────────────────────────────────┘                       └──────────────────────────────────────┘
```

#### Para Colunas PII (`pii=True`)
Para qualquer coluna onde `pii=True` (ou cujo `sdtype` seja reconhecido como PII), o `DataProcessor` atribui o transformador **`rdt.transformers.pii.AnonymizedFaker`** do RDT.
- O `AnonymizedFaker` utiliza a biblioteca **Faker** por baixo dos panos.
- Parâmetros da coluna no metadado (como `locales` para definir o idioma dos dados gerados, ou regras de unicidade se for uma chave) são repassados diretamente para esse transformador do RDT.

#### Para Colunas Padrão (`pii=False`)
Para colunas que não são PII, o RDT seleciona transformadores matemáticos/estatísticos adequados:
- **`numerical`** $\rightarrow$ `FloatFormatter` ou `GaussianNormalizer` (converte números em distribuições Gaussianas normalizadas).
- **`categorical`** $\rightarrow$ `FrequencyEncoder` ou `OneHotEncoder` (converte categorias em vetores numéricos).
- **`datetime`** $\rightarrow$ `UnixTimestampEncoder` (converte datas para inteiros Unix de alta precisão).
- **`id`** $\rightarrow$ `RegexGenerator` (preserva o padrão de IDs sequenciais/formato regex).

---

### 3. O Fluxo de Execução no RDT: `fit` e `sample`

O comportamento das colunas PII difere drasticamente do comportamento das colunas convencionais durante o ciclo de vida da síntese:

#### Fase 1: Treinamento (`synthesizer.fit(data)`)
1. **Colunas PII:** Durante a transformação do RDT para alimentar o modelo de rede neural / estatístico, as colunas marcadas como PII são **descartadas ou ignoradas** pelo modelo sintético. Isso impede qualquer vazamento (*data leakage*) dos dados reais dos usuários.
2. **Colunas Não-PII:** O RDT aplica as transformações numéricas e o modelo sintético aprende a matriz de correlação estatística entre elas.

#### Fase 2: Amostragem (`synthesizer.sample(num_rows)`)
1. O modelo sintético gera novos vetores numéricos/categóricos para as colunas não-PII.
2. O RDT executa o método `reverse_transform()`:
   - Para colunas normais, reconverte os números gerados de volta para seus tipos originais (floats, categorias, datas).
   - Para colunas **PII**, o `AnonymizedFaker` do RDT entra em ação e **gera valores sintéticos inéditos e realistas** usando o `Faker` (ex: gera e-mails no formato `nome.sobrenome@dominio.com`, números de telefone válidos ou nomes fictícios aleatórios), garantindo que nenhum dado PII real apareça na tabela sintetizada.

---

### Resumo dos Benefícios

1. **Privacidade Garantida:** Nenhuma informação pessoal identificável real é passada para o modelo de Inteligência Artificial.
2. **Realismo Visual:** A tabela final sintetizada terá e-mails, nomes e telefones com formato perfeitamente válido gerados pelo RDT / Faker.
3. **Eficiência no Modelo:** Reduz a complexidade do modelo sintético, que foca seu poder computacional em aprender correlações estatísticas relevantes entre as variáveis não-sensíveis.