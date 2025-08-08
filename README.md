
## RAG-система для ответов на вопросы 

В файле `RAG.ipynb` продемонстрирована работа **RAG (Retrieval-Augmented Generation)** системы. Я использовал датасет `enelpol/rag-mini-bioasq`, чтобы показать, как можно комбинировать поиск по документам и генерацию текста для получения  ответов на вопросы, которых нет в LLM или же для повышения точности ответов.

### Структура работы

  * **Загрузка данных:** Я загружаю датасеты с вопросами и корпусом текстов и делю тексты на фрагменты:
    ```bash
    splitter = RecursiveCharacterTextSplitter(
           chunk_size=512,   #фрагменты будут не длиннее chunk_size СИМВОЛОВ
           chunk_overlap=100)
    ```
    `Всего Чанков: 160190`
    
  * **Векторное хранилище:** Я создаю векторное хранилище из корпуса текстов с помощью **langchain_huggingface.HuggingFaceEmbeddings** и **langchain.vectorstores.Chroma**.\
    На роль **ретривера** -  модели, создающей векторные представления фрагментов и запросов, выбрал `sentence-transformers/all-mpnet-base-v2`

  * **Проверка длины фрагментов в токенах**
    <img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/92cad50f-a725-4493-af0d-05a7b2e71039" />

  * **Выбор генератора и его квантищация** В качестве генератора была выбрана LLM :  `Qwen/Qwen3-8B`. Она была квантизирована в 4 бита с использованием специальньного 4-битного типа данных  **NormalFloat4 (NF4)**
```bash
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # включаем 4-битное квантование
    bnb_4bit_quant_type="nf4",                  # тип квантования NF4
    bnb_4bit_use_double_quant=True,             # двойное квантование для уменьшения объёма
    bnb_4bit_compute_dtype=torch.float16        # вычисления в float16
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,  # передаём явно тип
    trust_remote_code=True      
    
```
  * **Сборка RAG-пайплайна:** Я использую **LangChain** для объединения поиска по документам и генерации ответа языковой моделью. Генератору на вход подаётся 3 самых релевантных фрагмента в качестве контекста

  * **Тестирование:** В конце я запускаю цикл для ответов на вопросы из тестового набора, чтобы показать, как работает система. Если из трёх самых релевантных документов, найденных ретривером, ни одного из них нет в списке актуальных по мнению авторов данного датасета, то я пишу, что `контект НЕ актуален`. Вот пример парочки вопросов и ответов на них выбранной LLM:
     
    
 ```bash
Контекст  актуален

Вопрос: What is the function of Neu5Gc (N-Glycolylneuraminic acid)?
Ожидаемый ответ: N-glycolylneuraminic acid (Neu5Gc) is an immunogenic sugar of dietary origin that metabolically incorporates into diverse native glycoconjugates in humans.  Humans lack a functional cytidine monophosphate-N-acetylneuraminic acid hydroxylase (CMAH) protein and cannot synthesize the sugar Neu5Gc, an innate mammalian signal of self. N-Glycolylneuraminic acid (Neu5Gc) can be incorporated in human cells and can trigger immune response, a response that is diverse and polyclonal. As dietary Neu5Gc is primarily found in red meat and milk products, it is suggested that this ongoing antigen-antibody reaction may generate chronic inflammation, possibly contributing to the high frequency of diet-related carcinomas and other diseases in humans.
Ответ модели:  The function of Neu5Gc is to serve as a receptor for certain pathogens like IAVs. It is also involved in cell-cell interactions and immune responses due to its presence on cell surfaces. Neu5Gc can be incorporated from diet and may influence infection by pathogens that recognize it. Neu5Gc is a sialic acid that terminates sugar chains on cell surfaces, playing a role in various biological processes. The absence of Neu5Gc in humans leads to immune responses against it when it is ingested.

Контекст  актуален

Вопрос: What is the mechanism of action of Inclisiran?
Ожидаемый ответ: Inclisiran, a novel, synthetic, siRNA molecule, inhibits PCSK9 synthesis in hepatocytes. Inclisiran targets intracellular PCSK9 synthesis specifically, resulting in a dose-dependent, long-term, significant reduction in LDL-C.
Ответ модели:  Inhibits PCSK9

Контекст НЕ актуален

Вопрос: What is F105-P?
Ожидаемый ответ: F105-P is a protamine-antibody fusion protein designed to deliver siRNA to HIV-infected or envelope-transfected cells. In specific, it was designed with the protamine coding sequence linked to the C terminus of the heavy chain Fab fragment of an HIV-1 envelope antibody.
Ответ модели:  F105-P is a variant in the ACVR1 gene associated with FOP, specifically the p.G356D and p.G328E mutations. The most common recurrent allele is p.R206H, which is prevalent in Indian patients with FOP. The variant is linked to the development of pyogenic granulomas, and its management typically involves conservative approaches with no significant complications.

Контекст  актуален

Вопрос: Is tubulin acetylation involved in cell motility?
Ожидаемый ответ: Yes, induction of alpha-tubulin acetylation correlates with inhibition of cell motility, while it is involved in additional cellular processes, e.g. cell cycle progression, differentiation, intracellular trafficking, and signalling. Dynamic microtubule (MT) acetylation/deacetylation mediating cell motility and adhesion is controlled by enzymes such as HDAC6, a major cytoplasmic α-tubulin deacetylase. While its overexpression and activation is capable to enhance cell motility, HDAC6 activity can also be negatively regulated by a number of cellular inhibitors, thus decreasing the ability of cells for migration.
Ответ модели:  Yes. Based on the context provided, tubulin acetylation is involved in cell motility. The context mentions that tubulin acetylation plays a role in various cellular functions, including cell motility. Additionally, it states that tubulin acetylation is part of a 'language' that regulates microtubule structure and function, which is crucial for processes like cell migration. Therefore, the answer is yes. Yes. The context explicitly states that tubulin acetylation is involved in multiple cellular functions, including cell motility. The text also mentions that tubulin acetylation is part of a regulatory 'language' for microtubule structure and function, which is essential for cell migration. Therefore, the answer is yes. Yes. According to the context, tubulin acetylation is involved in multiple cellular functions, including cell motility. The text states that tubulin acetylation is part of a regulatory 'language' that influences microtubule structure and function, which is crucial for cell migration. Therefore, the answer is yes. Yes. The context indicates that tubulin acetylation is involved in cell motility as it is part of the regulatory mechanisms that control microtubule structure and function, which are essential for cell migration

```

    

  * **Результаты:** Было достигнуто качество в 84% **Hit Rate 3** нашим ретривером, но даже при случаях, когда ретривер не находил актуальные статьи, LLM могла отвечать правильно.







