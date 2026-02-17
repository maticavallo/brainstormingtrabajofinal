# ECGAssistant - Detección y Explicación de Arritmias con Deep Learning, Fine-Tuning y RAG

## Definición del Problema

### El problema de negocio

Las arritmias cardíacas afectan a más de 300 millones de personas en el mundo. Un electrocardiograma (ECG) genera datos que requieren interpretación especializada: un cardiólogo tarda entre 5-15 minutos en analizar un registro de 30 minutos, y hay escasez global de especialistas — especialmente en Latinoamérica.

**El gap**: Los datos del ECG existen, las bases de datos están disponibles (MIT-BIH, PhysioNet), y los modelos de Deep Learning ya pueden clasificar arritmias con altísima precisión. Sin embargo, **no existe una herramienta accesible en español** que combine detección automática con explicación clínica comprensible. Los sistemas actuales clasifican pero no explican — un paciente recibe "arritmia detectada" sin entender qué significa, qué riesgo implica, o cuándo debería ir a urgencias.

**La oportunidad**: Construir un asistente que no solo detecte arritmias (DL), sino que las explique en español clínico pero comprensible (LLM fine-tuned), con contexto clínico actualizado (RAG).

### El desafío técnico de AI

Construir un sistema que clasifique y explique arritmias combina tres desafíos de AI distintos:

1. **Clasificación de señales eléctricas crudas (Deep Learning)**: Un ECG es una serie temporal de voltajes muestreados a 360 Hz. Clasificar latidos normales vs anormales a partir de señales eléctricas crudas requiere un modelo que aprenda patrones morfológicos (complejo QRS, onda P, segmento ST) directamente de los datos — esto es Deep Learning puro sobre series temporales.

2. **Generación de explicaciones clínicas sin alucinaciones (Fine-tuning)**: Un LLM base puede "inventar" información médica peligrosa. El fine-tuning le enseña a: anclar sus explicaciones en el resultado concreto del clasificador, usar terminología clínica correcta, comunicar en español, y nunca afirmar diagnósticos que el modelo no puede hacer.

3. **Contexto clínico actualizado sin re-entrenar (RAG)**: Las guías médicas se actualizan, los protocolos cambian. El sistema necesita acceso a documentos clínicos actualizados (guías de arritmias, protocolos de derivación, farmacología) sin re-entrenar el modelo cada vez — esto es exactamente lo que resuelve RAG.

**La solución técnica**: Pipeline **DL (CNN)** → **LLM Fine-tuned (Gemma 3 + QLoRA)** → **RAG (FAISS + documentos clínicos)** que produce explicaciones clínicas verificables en español.

### Encuadre como proyecto del bootcamp

Este proyecto implementa un **pipeline completo de AI** que cubre todas las fases exigidas por la práctica final del bootcamp de KeepCoding:

| Fase del Bootcamp | Cómo se cubre | Notebook |
|---|---|---|
| **Definición del problema** | Detección + explicación de arritmias en español | Este documento |
| **Dataset** | MIT-BIH Arrhythmia Database: 73K latidos reales, preprocesamiento + generación Q&A | Notebook 1 |
| **Deep Learning** | CNN 1D (PyTorch): F1=0.981, Accuracy=99% — **DL REAL como pilar central** | Notebook 2 |
| **Modelo (LLM)** | Fine-tuning de Gemma 3 4B con QLoRA para explicaciones clínicas | Notebook 3 |
| **RAG** | FAISS + 6 documentos clínicos especializados | Notebook 4 |
| **Evaluación** | DL (F1, confusion matrix) + LLM (precisión clínica, claridad, anclaje) | Notebook 5 |

**Decisión clave**: A diferencia de otros proyectos donde el DL es opcional, aquí el Deep Learning es el **pilar central** del pipeline — sin la CNN clasificadora, el resto del sistema no tiene input. Esto demuestra DL real aplicado a un problema real, no como bonus sino como fundamento.

---

## Concepto

Un asistente en español que combina un **clasificador DL** (CNN 1D sobre señales ECG) con un **LLM fine-tuneado** (Gemma 3 + QLoRA) y **RAG** (FAISS + documentos clínicos) para detectar arritmias y explicar su significado clínico.

El flujo es: señal ECG cruda → CNN clasifica Normal/Anormal con confianza → resultado alimenta al LLM fine-tuned → RAG aporta contexto de guías médicas → usuario recibe explicación en español.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Dataset    │───→│   DL Model   │───→│  LLM Gemma3  │───→│     RAG      │───→│   Usuario    │
│   MIT-BIH    │    │  CNN 1D      │    │  Fine-tuned  │    │ FAISS+Docs   │    │  (español)   │
│  73K latidos │    │  (PyTorch)   │    │   (QLoRA)    │    │  clínicos    │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
  Señales ECG        Clasifica           Explica el          Aporta guías       Recibe diagnóstico
  reales             Normal/Anormal      diagnóstico         clínicas           + explicación
```

---

## Dataset Fuente

**MIT-BIH Arrhythmia Database** (PhysioNet):
- 35 registros de pacientes reales
- Frecuencia de muestreo: 360 Hz
- Duración: ~30 minutos por registro (~1805 segundos)
- Señal: 2 canales (se usa canal 0 — derivación MLII)
- **73,737 latidos** extraídos con ventana de 0.8s (288 muestras)

### Distribución de Clases

| Tipo | Clases originales | Cantidad | % |
|------|-------------------|----------|---|
| **Normal** | N | 52,242 | 70.9% |
| **Anormal** | V, L, R, A, F, f, !, E, j, a, J, Q, S, x, [, ], " | 21,495 | 29.1% |

**18 clases de anotación** binarizadas: Normal (N) vs Todo lo demás (Anormal).

### Splits

| Split | Tamaño | Normal | Anormal | Estratificado |
|-------|--------|--------|---------|---------------|
| Train | 51,615 (70%) | 36,569 | 15,046 | Si |
| Validation | 7,374 (10%) | 5,224 | 2,150 | Si |
| Test | 14,748 (20%) | 10,449 | 4,299 | Si |

---

## Pipeline de Notebooks

| # | Notebook | Fase del Pipeline | Técnicas principales |
|---|----------|-------------------|----------------------|
| 1 | `ECG_Dataset` | Datos + Preprocesamiento | wfdb, neurokit2, PhysioNet, sklearn |
| 2 | `ECG_DeepLearning` | Clasificación DL | PyTorch, Conv1d, BatchNorm, CrossEntropyLoss |
| 3 | `ECG_FineTuning` | Explicación LLM | Unsloth, QLoRA, Gemma 3 4B, trl |
| 4 | `ECG_RAG` | Contexto clínico | langchain, FAISS, sentence-transformers |
| 5 | `ECG_Evaluation` | Evaluación completa | F1, accuracy, rubrics LLM, test clínicos |

---

## Notebook 1: `ECG_Dataset`

### 1.1 Descarga de MIT-BIH

```python
import wfdb
import numpy as np

records = ["100","101","102","103","104","105","106","107","108","109",
           "111","112","113","114","115","116","117","118","119",
           "121","122","123","124","200","201","202","203","205","207","208","209",
           "210","212","213","214"]

wfdb.dl_database("mitdb", dl_dir="mitdb", records=records)
```

### 1.2 Extracción de Latidos

Ventana de 0.8 segundos centrada en cada anotación R-peak:

```python
def load_record(record_id, base_dir="mitdb", channel=0, window_sec=0.8, target_size=288):
    record = wfdb.rdrecord(f"{base_dir}/{record_id}")
    ann = wfdb.rdann(f"{base_dir}/{record_id}", "atr")

    signal = record.p_signal[:, channel]
    fs = record.fs  # 360 Hz

    window_size = int(window_sec * fs)  # 288 muestras
    half = window_size // 2

    X_local, y_local = [], []

    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in ["+", "~", "|", "/"]:  # No-beat annotations
            continue

        start = sample - half
        end = sample + half

        if start > 0 and end < len(signal):
            beat = signal[start:end]
            if len(beat) == target_size:
                X_local.append(beat)
                y_local.append(symbol)

    return np.array(X_local), np.array(y_local)
```

**Resultado**: 73,737 latidos de 288 muestras cada uno.

### 1.3 Binarización de Clases

```python
# Normal (N) = 0, Todo lo demás = 1 (Anormal)
y_binary = np.where(y_all == "N", 0, 1)
# Normal: 52,242 | Anormal: 21,495
```

### 1.4 Normalización Z-score

```python
# Calcular media y std SOLO del entrenamiento
mean = x_train.mean()   # -0.354
std = x_train.std()     # 0.480

X_train = (x_train - mean) / std
X_val = (x_val - mean) / std
X_test = (x_test - mean) / std
```

### 1.5 Generación de Dataset Q&A para Fine-Tuning

Generar pares (resultado clasificación CNN + explicación clínica) para entrenar el LLM:

**Estrategia template-based con datos reales del clasificador:**

```python
# Ejemplo de par Q&A generado:
{
    "messages": [
        {"role": "system", "content": system_prompt_ecg},
        {"role": "user", "content": "El modelo CNN analizó un ECG y clasificó el latido como ANORMAL con una confianza del 97.3%. El tipo detectado es V (contracción ventricular prematura). ¿Qué significa esto?"},
        {"role": "assistant", "content": "El análisis del ECG muestra un latido clasificado como anormal..."}
    ]
}
```

#### Tipos de pares Q&A

**Tipo 1 - Explicación de resultado Normal:**
```
Input:  "El modelo clasificó este latido como NORMAL con confianza del {X}%."
Output: Explicación de qué significa un latido normal, qué indica la confianza,
        y cuándo un resultado normal NO descarta problemas.
```

**Tipo 2 - Explicación de resultado Anormal:**
```
Input:  "El modelo detectó un latido ANORMAL con confianza del {X}%.
         Tipo detectado: {tipo}."
Output: Explicación del tipo de anomalía, gravedad estimada, qué hacer,
        y cuándo buscar atención médica.
```

**Tipo 3 - Interpretación de proporción:**
```
Input:  "De {N} latidos analizados, {M} fueron clasificados como anormales ({P}%).
         Tipos detectados: {distribución}."
Output: Análisis de la proporción, comparación con umbrales clínicos,
        recomendación de acción.
```

**Tipo 4 - Preguntas clínicas generales:**
```
Input:  "¿Qué es una arritmia ventricular?"
Output: Explicación en español clínico pero comprensible, tipos, gravedad,
        cuándo preocuparse.
```

**Tipo 5 - Limitaciones del modelo:**
```
Input:  "¿Puedo confiar en este diagnóstico?"
Output: Explicación honesta de las limitaciones del modelo, que NO reemplaza
        a un cardiólogo, y cuándo buscar atención profesional.
```

### 1.6 Traducción de Términos Clínicos EN→ES

```python
# Glosario de términos clave traducidos:
terminologia = {
    "Premature Ventricular Contraction (PVC)": "Contracción Ventricular Prematura (CVP)",
    "Atrial Fibrillation": "Fibrilación Auricular",
    "Bundle Branch Block": "Bloqueo de Rama",
    "Supraventricular Tachycardia": "Taquicardia Supraventricular",
    "Normal Sinus Rhythm": "Ritmo Sinusal Normal",
    # ... más términos
}
```

---

## Notebook 2: `ECG_DeepLearning`

> **Este notebook YA EXISTE y está implementado.** Los resultados son reales.

### 2.1 Arquitectura: CNN 1D

```python
class ECG_CNN(nn.Module):

    def __init__(self):
        super(ECG_CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(32 * 72, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # (B, 16, 144)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # (B, 32, 72)
        x = x.view(x.size(0), -1)                           # (B, 2304)
        x = self.dropout(torch.relu(self.fc1(x)))            # (B, 64)
        x = self.fc2(x)                                      # (B, 2)
        return x
```

**Decisiones de diseño:**
- **Conv1d** (no Conv2d): El ECG es una señal 1D — no tiene sentido tratarla como imagen
- **BatchNorm**: Estabiliza el entrenamiento con señales de amplitud variable
- **Weighted CrossEntropyLoss**: Compensa el desbalance 70/30 entre Normal y Anormal
- **Dropout 0.3**: Previene overfitting sin degradar F1

### 2.2 Entrenamiento

```python
# Configuración
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Pesos para compensar desbalance de clases
class_counts = np.bincount(y_train)
class_weights = torch.tensor(
    [1.0/class_counts[0], 1.0/class_counts[1]], dtype=torch.float32
).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 35 epochs
```

### 2.3 Resultados (REALES - ya obtenidos)

| Métrica | Clase Normal (0) | Clase Anormal (1) | Global |
|---------|-----------------|-------------------|--------|
| **Precision** | 0.99 | 0.98 | — |
| **Recall** | 0.99 | 0.98 | — |
| **F1-Score** | 0.99 | **0.981** | — |
| **Accuracy** | — | — | **0.99** |

**Confusion Matrix (sobre 14,748 latidos de test):**

|  | Pred Normal | Pred Anormal |
|--|-------------|-------------|
| **Real Normal** | 10,361 | 88 |
| **Real Anormal** | 73 | 4,226 |

**Curva de entrenamiento**: Loss desciende de 0.146 → 0.015 en 35 epochs. Val F1 estabiliza en ~0.985 desde epoch 26.

---

## Notebook 3: `ECG_FineTuning`

### 3.1 Setup

```python
!pip install -qU unsloth trl bitsandbytes datasets

from unsloth import FastModel
from datasets import load_dataset

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it",
    max_seq_length=2048,
    load_in_4bit=True
)

model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)
```

### 3.2 Dataset de Entrenamiento

```python
dataset = load_dataset("usuario/ecg-assistant-es")
```

Pares generados en Notebook 1: resultado CNN + explicación clínica.

### 3.3 System Prompt

```python
system_prompt = """Eres ECGAssistant, un asistente especializado en la interpretación
de electrocardiogramas. Explicás los resultados de un modelo de clasificación de arritmias
basado en Deep Learning. Siempre anclás tus explicaciones en los datos concretos del análisis
(clasificación, confianza, tipo de anomalía). Comunicás en español clínico pero comprensible.
IMPORTANTE: No sos un médico ni reemplazás el diagnóstico profesional. Siempre recomendás
consultar con un cardiólogo para decisiones clínicas."""
```

### 3.4 Capacidades que Aprende el Modelo

1. **Explicar clasificaciones**: "El modelo detectó un latido anormal tipo V, lo que significa..."
2. **Interpretar confianza**: "La confianza del 97% indica que el modelo está bastante seguro..."
3. **Contextualizar proporciones**: "3 latidos anormales de 100 es una proporción baja..."
4. **Comunicar limitaciones**: "Este modelo clasifica Normal vs Anormal, no diagnostica..."
5. **Fluidez en español clínico**: Terminología correcta, explicaciones comprensibles

### 3.5 Training

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=SFTConfig(
        output_dir="./ecg-assistant-model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
    ),
)

trainer.train()
```

### 3.6 Upload

```python
model.push_to_hub("usuario/ecg-assistant-gemma3-qlora")
tokenizer.push_to_hub("usuario/ecg-assistant-gemma3-qlora")
```

---

## Notebook 4: `ECG_RAG`

### 4.1 Documentos Clínicos

| Documento | Contenido | Fuente |
|-----------|-----------|--------|
| `guia_arritmias.txt` | Tipos de arritmias, clasificación, gravedad, síntomas | Redacción basada en guías AHA/ESC |
| `protocolo_ecg.txt` | Cómo se realiza un ECG, qué mide cada derivación, valores normales | Redacción basada en literatura clínica |
| `explicaciones_pacientes.txt` | Explicaciones simplificadas de arritmias para pacientes, preguntas frecuentes | Redacción manual |
| `farmacologia_basica.txt` | Medicamentos antiarrítmicos más comunes, indicaciones generales | Redacción basada en vademécum |
| `derivacion_urgencias.txt` | Criterios de cuándo derivar a urgencias, signos de alarma, síntomas peligrosos | Redacción basada en protocolos |
| `glosario_medico.txt` | Glosario de términos médicos EN-ES, abreviaturas cardíacas | Compilación manual |

### 4.2 Stack RAG

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Cargar documentos clínicos
loader = DirectoryLoader('./docs_clinicos/', glob="*.txt")
documents = loader.load()

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embeddings + FAISS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Retrieval
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### 4.3 Pipeline Completo: CNN → LLM → RAG

```python
def ecg_assistant(ecg_signal, question=None):
    # 1. Clasificar con la CNN
    prediction, confidence = cnn_classify(ecg_signal)
    label = "Normal" if prediction == 0 else "Anormal"

    # 2. Construir prompt con resultado de la CNN
    cnn_result = f"Clasificación: {label} (confianza: {confidence:.1%})"

    # 3. Recuperar contexto clínico relevante (RAG)
    query = f"{cnn_result} {question}" if question else cnn_result
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    # 4. Generar explicación con el LLM fine-tuned
    prompt = f"""Resultado del análisis de ECG:
{cnn_result}

Contexto clínico:
{context}

{f'Pregunta del usuario: {question}' if question else 'Explicá este resultado.'}

Respondé como ECGAssistant, anclando tu explicación en los datos del análisis."""

    response = generate(finetuned_model, prompt)
    return response
```

---

## Notebook 5: `ECG_Evaluation`

### 5.1 Evaluación del Deep Learning (ya obtenida)

| Métrica | Valor |
|---------|-------|
| **F1 Score** | 0.981 |
| **Accuracy** | 99% |
| **Precision (Normal)** | 0.99 |
| **Precision (Anormal)** | 0.98 |
| **Recall (Normal)** | 0.99 |
| **Recall (Anormal)** | 0.98 |
| **Falsos Positivos** | 88 / 14,748 (0.6%) |
| **Falsos Negativos** | 73 / 14,748 (0.5%) |

### 5.2 Evaluación del LLM Fine-tuned

**Comparación Base vs Fine-tuned** con los mismos prompts:

| Dimensión | Qué se evalúa | Cómo |
|-----------|---------------|------|
| **Precisión clínica** | ¿La explicación es médicamente correcta? | Verificar contra guías médicas |
| **Anclaje en datos del ECG** | ¿Cita el resultado de la CNN o inventa? | Verificar contra output del clasificador |
| **Claridad** | ¿Un paciente entendería la explicación? | Rubric manual (1-5) |
| **Honestidad sobre limitaciones** | ¿Dice que no es diagnóstico médico? | Verificar disclaimer |
| **Fluidez en español** | ¿Es natural y usa terminología correcta? | Rubric manual (1-5) |

### 5.3 Test Cases por Dificultad

**Fácil** (respuesta directa):
- "El modelo clasificó este latido como Normal con 99% de confianza. ¿Qué significa?"
- "¿Qué es un latido normal en un ECG?"

**Medio** (requiere interpretación):
- "De 100 latidos, 5 fueron anormales tipo V. ¿Es grave?"
- "La confianza del modelo fue del 62%. ¿Puedo confiar en el resultado?"
- "¿Qué diferencia hay entre una arritmia supraventricular y una ventricular?"

**Difícil** (requiere razonamiento clínico):
- "Tengo 3 latidos anormales consecutivos tipo V. ¿Debería ir a urgencias?"
- "Mi ECG muestra arritmia pero me siento bien. ¿Es posible?"
- "El modelo dice Normal pero mi médico dijo que tengo arritmia. ¿Quién tiene razón?"

### 5.4 Formato de Evaluación

```python
test_prompts = [
    {"prompt": "El modelo clasificó el latido como Anormal (confianza 97%)...", "difficulty": "fácil"},
    {"prompt": "De 200 latidos, 15 son anormales tipo V...", "difficulty": "medio"},
    {"prompt": "Tengo latidos anormales consecutivos, ¿urgencias?...", "difficulty": "difícil"},
]

results = []
for test in test_prompts:
    response_base = generate(base_model, test["prompt"])
    response_ft = generate(finetuned_model, test["prompt"])
    results.append({
        "prompt": test["prompt"],
        "base": response_base,
        "finetuned": response_ft,
        "difficulty": test["difficulty"]
    })
```

---

## Ejemplos de Interacción

### Explicación de Resultado Normal

> **Usuario**: El modelo clasificó mi ECG como Normal con 99.2% de confianza. ¿Qué significa?
>
> **ECGAssistant**: El análisis de tu ECG muestra un latido clasificado como **Normal** (ritmo sinusal) con una confianza del 99.2%, lo que indica que el modelo está muy seguro de esta clasificación.
>
> Un latido normal significa que la señal eléctrica del corazón sigue el patrón esperado: se origina en el nodo sinusal, se propaga correctamente por aurículas y ventrículos, y genera un complejo QRS de forma y duración normales.
>
> **Importante**: Un resultado Normal en un latido individual no descarta problemas cardíacos que aparecen de forma intermitente. Si tenés síntomas (palpitaciones, mareos, dolor torácico), consultá con un cardiólogo aunque el resultado sea Normal. Este análisis es orientativo y **no reemplaza** el diagnóstico de un profesional médico.

### Explicación de Resultado Anormal

> **Usuario**: El modelo detectó un latido anormal tipo V con confianza del 97.3%. ¿Qué significa?
>
> **ECGAssistant**: El análisis detectó un latido clasificado como **Anormal** con confianza del 97.3%. El tipo "V" corresponde a una **Contracción Ventricular Prematura (CVP)** — un latido que se origina en los ventrículos en lugar del nodo sinusal.
>
> **¿Qué significa?** Las CVP son las arritmias más comunes. Un latido aislado tipo V en una persona sin enfermedad cardíaca suele ser benigno — muchas personas sanas las experimentan sin saberlo.
>
> **¿Cuándo preocuparse?** Si aparecen múltiples CVP consecutivas (3 o más), si son frecuentes (más del 10% de los latidos), o si se acompañan de síntomas como mareos, desmayos o dolor torácico.
>
> **Recomendación**: Este resultado es orientativo. Consultá con un cardiólogo para una evaluación completa, especialmente si tenés factores de riesgo cardíaco o síntomas asociados.

### Interpretación de Proporción

> **Usuario**: De 100 latidos analizados, 5 fueron clasificados como anormales tipo V. ¿Debo preocuparme?
>
> **ECGAssistant**: De 100 latidos, 5 fueron clasificados como anormales tipo V (Contracciones Ventriculares Prematuras), lo que representa un **5% de carga ectópica ventricular**.
>
> **Contexto clínico**: Un 5% está por debajo del umbral de preocupación clínica. Las guías médicas consideran como umbral significativo un **10% o más** de latidos ectópicos, o la presencia de CVP en salvas (3 o más consecutivas).
>
> **Tu situación**: Con un 5% de CVP aisladas, la mayoría de los cardiólogos considerarían esto dentro de un rango normal-alto. Sin embargo, hay que considerar:
> - Si las CVP están distribuidas uniformemente o aparecen en ráfagas
> - Si tenés síntomas asociados (palpitaciones, fatiga)
> - Si tenés antecedentes de enfermedad cardíaca
>
> **Recomendación**: No es una emergencia, pero vale la pena mencionarlo en tu próxima consulta cardiológica. Si presentás síntomas, consultá antes.

### Pregunta Clínica General

> **Usuario**: ¿Qué diferencia hay entre una arritmia supraventricular y una ventricular?
>
> **ECGAssistant**: La diferencia principal está en el **origen** del impulso eléctrico anormal:
>
> **Arritmias supraventriculares** (por encima de los ventrículos):
> - Se originan en las aurículas o el nodo AV
> - Ejemplos: fibrilación auricular, taquicardia supraventricular, aleteo auricular
> - Generalmente menos peligrosas
> - El complejo QRS en el ECG suele ser angosto (< 120 ms)
>
> **Arritmias ventriculares** (en los ventrículos):
> - Se originan en el tejido ventricular
> - Ejemplos: CVP (tipo V en nuestro modelo), taquicardia ventricular, fibrilación ventricular
> - Pueden ser más peligrosas, especialmente si son sostenidas
> - El complejo QRS suele ser ancho y deformado
>
> **En nuestro modelo**: El clasificador agrupa ambos tipos bajo "Anormal". Las CVP (tipo V) son las anomalías ventriculares más comunes y generalmente benignas cuando son aisladas. La fibrilación ventricular, en cambio, es una emergencia médica.
>
> **Importante**: Esta es una explicación general. El diagnóstico específico del tipo de arritmia requiere un cardiólogo con acceso al ECG completo.

### Confianza Baja del Modelo

> **Usuario**: La confianza del modelo fue del 58%. ¿Puedo confiar en el resultado?
>
> **ECGAssistant**: Una confianza del 58% indica que el modelo tiene **incertidumbre significativa** sobre la clasificación. El modelo clasifica en dos categorías (Normal y Anormal), así que 58% está apenas por encima del azar (50%).
>
> **¿Qué puede causar confianza baja?**
> - Señal ECG con ruido (movimiento del paciente, interferencia eléctrica)
> - Latido con morfología ambigua (borderline entre Normal y Anormal)
> - Tipo de arritmia que el modelo no vio suficiente en el entrenamiento
>
> **Recomendación**: Con confianza menor al 80%, el resultado debe tomarse con precaución. Esto NO significa que haya o no haya arritmia — significa que el modelo no está seguro. Consultá con un cardiólogo para una interpretación profesional del ECG.

### Limitaciones del Modelo

> **Usuario**: ¿Puedo usar este sistema en lugar de ir al cardiólogo?
>
> **ECGAssistant**: **No.** Este sistema es una herramienta de apoyo, **no un sustituto del diagnóstico médico**. Específicamente:
>
> **Lo que SÍ puede hacer**: Clasificar latidos individuales como Normal o Anormal con alta precisión (F1=0.981), y explicar qué significan los resultados.
>
> **Lo que NO puede hacer**:
> - Diagnosticar enfermedades cardíacas
> - Analizar todas las derivaciones de un ECG de 12 leads (usa solo una)
> - Detectar todas las patologías (solo distingue Normal vs Anormal)
> - Considerar tu historial médico, medicación, o síntomas
> - Detectar infartos, isquemia, o problemas estructurales del corazón
>
> **Siempre consultá con un cardiólogo** para decisiones clínicas. Este sistema puede ayudarte a entender mejor tus resultados, pero la interpretación final debe ser de un profesional médico con acceso a tu historial completo.

---

## Stack Técnico

| Componente | Herramienta | Versión / Nota |
|------------|-------------|----------------|
| Runtime | Google Colab Free | GPU T4 |
| Datos ECG | wfdb + neurokit2 | Descarga y procesamiento de PhysioNet |
| Preprocesamiento | numpy, sklearn | Z-score normalización, splits estratificados |
| **Deep Learning** | **PyTorch (Conv1d)** | **CNN 1D — pilar central del proyecto** |
| LLM | Gemma 3 4B Instruct | Base model |
| Fine-tuning | Unsloth + QLoRA | r=16, 4-bit |
| Training | trl (SFTTrainer) | 3 epochs |
| Vectorstore | FAISS (faiss-gpu) | RAG retrieval |
| RAG | langchain | Document loading + retrieval |
| Embeddings RAG | sentence-transformers (all-MiniLM-L6-v2) | 384 dims |
| Dataset Hub | HuggingFace Hub | Dataset + modelo |

---

## Comparación con Banking Assistant

| Aspecto | Banking Assistant | ECGAssistant |
|---------|-------------------|--------------|
| Dominio | Banca retail | Cardiología / ECG |
| Dataset base | Bitext banking chatbot | MIT-BIH Arrhythmia Database (PhysioNet) |
| **Deep Learning** | **No** | **Si — CNN 1D, F1=0.981 (pilar central)** |
| Tipo de respuesta | Factual ("tu saldo es...") | Explicativa ("este latido anormal significa...") |
| Complejidad razonamiento | Baja (lookup) | Media-Alta (interpretación clínica) |
| Modelo LLM | Gemma 3 4B + QLoRA | Gemma 3 4B + QLoRA |
| RAG | langchain + FAISS | langchain + FAISS |
| Evaluación | Correcto/Incorrecto | DL (F1) + LLM (precisión clínica, claridad) |
| Notebooks | 4 | 5 (incluye DL como notebook dedicado) |
| Impacto social | Medio | Alto (salud pública) |
| Wow factor | Medio | Alto (DL real + dominio médico) |

---

## Fortalezas del Proyecto

1. **Deep Learning REAL como pilar central**: No es un bonus — la CNN clasificadora es el fundamento de todo el pipeline. F1=0.981 ya demostrado
2. **Pipeline completo DL → FT → RAG → Eval**: Cubre todas las fases del bootcamp con DL real integrado
3. **Datos reales de PhysioNet**: 73K latidos de la base de datos más citada en investigación cardíaca — no datos ficticios
4. **Dominio de alto impacto**: Salud es un dominio que impresiona en presentaciones y tiene impacto social real
5. **Explicabilidad**: No solo clasifica sino que explica — diferenciador frente a otros proyectos de DL
6. **En español**: Traducción de terminología clínica EN→ES
7. **Evaluación dual**: Métricas cuantitativas para DL (F1, accuracy) + rubrics para LLM (precisión clínica, claridad)
8. **Disclaimer ético**: El sistema reconoce sus limitaciones y nunca reemplaza al profesional médico

## Riesgos y Mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Dataset Q&A de baja calidad para fine-tuning | Template-based con resultados reales del CNN, no requiere generación creativa |
| Fine-tuning genera "alucinaciones" médicas | System prompt con disclaimers, evaluación de anclaje en datos del CNN |
| RAG recupera contexto clínico irrelevante | Documentos específicos y acotados, ajustar chunk_size y k |
| Clasificador CNN falla en producción | F1=0.981 demostrado; disclaimers en cada respuesta |
| Confusión con diagnóstico médico real | Disclaimer prominente en system prompt y en cada respuesta |
| Colab Free sin GPU suficiente | CNN liviana (2 conv layers); QLoRA 4-bit para fine-tuning |
| Desbalance de clases (70/30) | Weighted CrossEntropyLoss ya implementado |

---

## Entregables

- [ ] 5 notebooks en GitHub (Dataset, DeepLearning, FineTuning, RAG, Evaluation)
- [ ] Dataset Q&A en HuggingFace Hub (`usuario/ecg-assistant-es`)
- [ ] Modelo fine-tuned en HuggingFace Hub (`usuario/ecg-assistant-gemma3-qlora`)
- [ ] Modelo CNN guardado (`.pth`)
- [ ] Presentación
- [ ] Post LinkedIn
