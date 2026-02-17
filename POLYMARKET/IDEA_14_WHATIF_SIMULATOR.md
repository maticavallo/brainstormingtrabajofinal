# Idea 14: WhatIfAnalyst - Simulador de Escenarios con Fine-Tuning y RAG

## Definición del Problema

### El problema de negocio

Polymarket es la plataforma de mercados predictivos más grande del mundo: 100,795 mercados, 43,840 eventos y $7.99B en volumen total. Traders, analistas y curiosos operan diariamente con $150M en volumen, apostando a resultados de eventos reales (elecciones, precios de crypto, eventos deportivos).

Sin embargo, **no existe ninguna herramienta que permita explorar escenarios hipotéticos** sobre este ecosistema. Preguntas como "¿Qué pasaría si se prohíben los mercados de crypto?" o "¿Cómo afecta una caída de liquidez del 50%?" no tienen respuesta accesible — requieren descargar los datasets crudos, calcular métricas, cruzar categorías y razonar sobre efectos en cascada.

**El gap**: Los datos existen (son públicos), pero están fragmentados en CSVs de 100K+ filas y 106 columnas. Ningún producto actual (ni Polymarket, ni Kalshi, ni Metaculus) ofrece un analista de escenarios que combine datos reales con razonamiento causal.

### El desafío técnico de AI

Construir un sistema que responda "¿Qué pasaría si...?" sobre datos reales es un problema no trivial de AI:

1. **Razonamiento causal, no factual**: Un LLM base puede contestar "¿Qué es Polymarket?" (factual), pero NO puede contestar "Si crypto pierde el 50% de su volumen, ¿qué categorías absorben la migración?" — eso requiere razonamiento causal anclado en datos específicos.

2. **Anclaje en datos verificables**: Las respuestas tienen que citar números reales ($7.99B, 100K mercados, spreads de 0.04) — no inventar. Un LLM base "alucina" cifras. El fine-tuning lo entrena para siempre anclar en datos concretos.

3. **Contexto especializado que evoluciona**: Los datos de Polymarket cambian. El sistema necesita acceso a documentos actualizados (estadísticas, correlaciones, regulación) sin re-entrenar el modelo cada vez → esto es exactamente lo que resuelve RAG.

4. **Dominio en español**: Los LLMs son más débiles en español que en inglés para tareas especializadas. El fine-tuning en español con datos del dominio cierra esta brecha.

**La solución técnica**: Combinar **Fine-tuning** (enseñarle al LLM a razonar causalmente sobre este dominio) + **RAG** (darle acceso a documentos con datos reales) en un pipeline que produce respuestas verificables en español.

### Encuadre como proyecto del bootcamp

Este proyecto implementa un **pipeline completo de AI** que cubre todas las fases exigidas por la práctica final del bootcamp de KeepCoding:

| Fase del Bootcamp | Cómo se cubre | Notebook |
|---|---|---|
| **Definición del problema** | Escenarios "What If" sobre Polymarket | Este documento |
| **Dataset** | 100K mercados reales + feature engineering + traducción EN→ES + generación de Q&A | Notebook 1 |
| **Modelo (LLM)** | Fine-tuning de Gemma 3 4B con QLoRA (acepta "finetunear modelo instructed" como modelo válido) | Notebook 2 |
| **RAG** | FAISS + 6 documentos especializados para anclar respuestas | Notebook 3 |
| **Evaluación** | Comparación base vs fine-tuned con rubrics por dificultad | Notebook 4 |
| **Deep Learning (opcional)** | Autoencoder + clustering + UMAP si da el tiempo | Notebook 5 |

**Decisión clave**: El bootcamp NO exige Deep Learning — acepta fine-tuning de un LLM como modelo válido. Por eso el pipeline principal (Notebooks 1-4) funciona de forma completa sin DL, y el notebook 5 es un bonus que enriquece los escenarios si hay tiempo.

---

## Concepto

Un asistente en español que combina un **LLM fine-tuneado** (Gemma 3 + QLoRA) con **RAG** (FAISS + documentos especializados) para responder preguntas hipotéticas sobre mercados predictivos de Polymarket.

Los datos del dataset (volumen, liquidez, categorías, features calculadas) alimentan directamente al LLM vía el dataset de entrenamiento y los documentos RAG. Opcionalmente, un módulo de **Deep Learning** (autoencoder + clustering) puede enriquecer las respuestas si hay tiempo.

```
PIPELINE PRINCIPAL:
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Dataset    │───→│  LLM Gemma3  │───→│     RAG      │───→│   Usuario    │
│  Polymarket  │    │  Fine-tuned  │    │ FAISS+Docs   │    │  (español)   │
│  + Features  │    │   (QLoRA)    │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘

MODULO OPCIONAL (si da el tiempo):
┌──────────────┐
│   DL Model   │──→ Enriquece escenarios con clusters
│ Autoencoder  │    y visualización UMAP
│ + Clustering │
└──────────────┘
```

---

## Dataset Fuente

**Polymarket Prediction Markets** (Diciembre 2025):
- `polymarket_events.csv`: 43,840 eventos, 67 columnas
- `polymarket_markets.csv`: 100,795 mercados, 106 columnas
- $7.99B volumen total, $150M volumen diario, $315M liquidez

### Columnas Clave para el Proyecto

| Columna | Tipo | Uso |
|---------|------|-----|
| `question`, `description`, `event_title` | Texto | Embeddings textuales, traducción EN→ES |
| `volume`, `volume24hr`, `volume1wk` | Numérico | Features, análisis de impacto |
| `liquidity`, `spread` | Numérico | Features, salud del mercado |
| `bestBid`, `bestAsk` | Numérico | Features, microestructura |
| `outcomePrices` | JSON | Probabilidades, target |
| `tags` / categorías | Categórico | Agrupación, escenarios por categoría |
| `competitorCount`, `marketCount` | Numérico | Complejidad del evento |
| `startDate`, `endDate`, `createdAt` | Timestamp | Temporalidad, crecimiento |
| `active`, `closed` | Booleano | Estado, resolución |

---

## Pipeline de Notebooks

### Pipeline Principal (4 notebooks)

| # | Notebook | Fase del Pipeline | Técnicas principales |
|---|----------|-------------------|----------------------|
| 1 | `WhatIf_Dataset` | Datos + Feature Engineering | pandas, Helsinki-NLP/opus-mt-en-es, HuggingFace Hub |
| 2 | `WhatIf_FineTuning` | Entrenamiento del LLM | Unsloth, QLoRA, Gemma 3 4B, trl, bitsandbytes |
| 3 | `WhatIf_RAG` | Retrieval-Augmented Generation | langchain, FAISS, sentence-transformers |
| 4 | `WhatIf_Evaluation` | Evaluación del sistema | Métricas LLM (calidad, anclaje, coherencia), rubrics |

### Notebook Opcional (si da el tiempo)

| # | Notebook | Fase del Pipeline | Técnicas principales |
|---|----------|-------------------|----------------------|
| 5 | `WhatIf_DeepLearning` | Clustering de mercados (DL) | sentence-transformers, autoencoder, UMAP, KMeans, PyTorch |

---

## Notebook 1: `WhatIf_Dataset`

### 1.1 Carga y Limpieza

```python
import pandas as pd

events = pd.read_csv("polymarket_events.csv")
markets = pd.read_csv("polymarket_markets.csv")

# Filtrar mercados con datos útiles
markets = markets[markets['volume'] > 0]
markets = markets[markets['outcomePrices'].notna()]
```

### 1.2 Feature Engineering

Métricas calculadas por mercado y por categoría:

| Feature | Cálculo | Para qué |
|---------|---------|----------|
| `volume_share` | volume / volume_total | % del volumen que representa cada mercado |
| `liquidity_ratio` | liquidity / volume | Salud del mercado (> 0.1 = sano) |
| `spread_normalized` | spread / outcomePrices | Spread relativo al precio |
| `category_volume` | sum(volume) por categoría | Volumen total por vertical |
| `category_growth` | mercados creados últimas 4 semanas / total | Crecimiento por categoría |
| `category_concentration` | volume_top10 / volume_category | Concentración del volumen |
| `market_age_days` | (updatedAt - createdAt).days | Antigüedad del mercado |

### 1.3 Traducción EN→ES

```python
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Traducir: question, description, event_title
```

### 1.4 Generación de Escenarios "What If"

**Estrategia template-based**: Generar Q&A automáticamente variando datos reales.

Cada tipo de escenario tiene **dos versiones**: una básica (funciona sin DL) y una enriquecida (si se completa el notebook opcional de DL).

#### Tipos de escenarios

**Tipo 1 - Cambio de volumen:**
```
Input:  "Si el volumen de {categoría} cae un {X}%, ¿qué impacto tendría en Polymarket?"

VERSION BÁSICA (sin DL):
Output: "Actualmente {categoría} representa el {Y}% del volumen total (${Z}M).
         Una caída del {X}% reduciría el volumen total en ${W}M ({P}% del total).
         Los mercados más afectados serían: {top_3_mercados_categoria}.
         Las categorías {cat2} y {cat3} podrían absorber parte del volumen."

VERSION ENRIQUECIDA (con DL - si se hizo el notebook 5 opcional):
Output: "...(todo lo anterior)...
         Según el clustering de mercados, los clusters {N1} y {N2} perderían
         el {Q}% de sus miembros."
```
Variables: categoría = [crypto, politics, sports, ...], X = [20, 30, 50, 70]

**Tipo 2 - Migración entre categorías:**
```
Input:  "Si los usuarios de {cat1} migran a {cat2}, ¿cómo cambiaría el mercado?"

VERSION BÁSICA (sin DL):
Output: "Actualmente {cat1} tiene ${V1}M en volumen y {cat2} tiene ${V2}M.
         Si el {X}% del volumen migra, {cat2} crecería a ${V2+migración}M,
         convirtiéndose en la {ranking}ª categoría por volumen.
         Los mercados de {cat2} con mayor liquidez absorberían primero:
         {top_3_mercados_cat2_por_liquidez}."

VERSION ENRIQUECIDA (con DL - si se hizo el notebook 5 opcional):
Output: "...(todo lo anterior)...
         Los clusters de {cat1} más afectados serían {cluster_1} y {cluster_2},
         mientras que {cat2} vería crecer los clusters {cluster_3} y {cluster_4}."
```

**Tipo 3 - Eliminación de categoría (regulación):**
```
Input:  "¿Qué pasaría si se prohíben los mercados de {categoría}?"

VERSION BÁSICA (sin DL):
Output: "Se eliminarían {N} mercados activos con ${V}M en volumen total.
         Esto representa el {P}% del volumen de Polymarket.
         Los mercados más grandes afectados serían: {top_3_mercados}.
         Históricamente, el {R}% del volumen de categorías eliminadas
         migra a crypto."

VERSION ENRIQUECIDA (con DL - si se hizo el notebook 5 opcional):
Output: "...(todo lo anterior)...
         Impacto por cluster: el cluster '{nombre_cluster}' perdería
         el {Q}% de sus mercados."
```

**Tipo 4 - Shock de liquidez:**
```
Input:  "Si la liquidez total de Polymarket cae a la mitad, ¿qué mercados sobreviven?"

VERSION BÁSICA (sin DL):
Output: "Con liquidez reducida al 50%, sobrevivirían los mercados con
         liquidity_ratio > {umbral}: {N} mercados de los {total} actuales.
         Por categoría: crypto retiene {P1}%, politics {P2}%, sports {P3}%.
         Los mercados con mayor resiliencia son: {top_3_por_liquidity_ratio}."

VERSION ENRIQUECIDA (con DL - si se hizo el notebook 5 opcional):
Output: "...(todo lo anterior)...
         El cluster '{nombre}' es el más resiliente con {R}% de supervivencia."
```

**Tipo 5 - Explosión de categoría:**
```
Input:  "Si {categoría} crece {X}x en volumen, ¿cómo afecta al ecosistema?"

VERSION BÁSICA (sin DL):
Output: "Un crecimiento de {X}x llevaría {categoría} de ${V}M a ${V*X}M,
         superando a {categorías_superadas} en volumen.
         Necesitaría ${L}M adicionales en liquidez para mantener spreads sanos.
         Basado en el crecimiento real de las últimas 4 semanas ({growth}%),
         este escenario requeriría {T} semanas al ritmo actual."

VERSION ENRIQUECIDA (con DL - si se hizo el notebook 5 opcional):
Output: "...(todo lo anterior)...
         Los clusters de {categoría} crecerían de {N1} a {N2} miembros,
         y se formarían potencialmente {K} nuevos sub-clusters."
```

**Tipo 6 - Escenario basado en clustering (EXCLUSIVO del notebook 5 opcional):**

> **Nota:** Este tipo de escenario SOLO se genera si se completó el notebook 5 de Deep Learning.
> El pipeline principal (Notebooks 1-4) funciona sin estos escenarios.

```
Input:  "¿Qué pasaría si desaparecen todos los mercados del cluster '{nombre}'?"
Output: "El cluster '{nombre}' tiene {N} mercados con ${V}M en volumen.
         Se caracterizan por: {descripción_cluster}.
         Su eliminación afectaría principalmente a {categorías_afectadas}.
         Los clusters más similares ({cluster_vecino_1}, {cluster_vecino_2})
         podrían absorber {P}% de la actividad."
```

### 1.5 Formato y Upload

```python
# Formato chat para fine-tuning
system_prompt = """Eres WhatIfAnalyst, un analista experto en mercados predictivos
de Polymarket. Evaluás escenarios hipotéticos basándote en datos reales y patrones
observados. Siempre anclás tus respuestas en números concretos. Respondés en español."""

# Cada ejemplo:
{
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": pregunta_whatif},
        {"role": "assistant", "content": respuesta_con_datos}
    ]
}

# Subir a HuggingFace Hub
dataset.push_to_hub("usuario/whatif-polymarket-es")
```

---

## Notebook 2: `WhatIf_FineTuning`

### 2.1 Setup

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

### 2.2 Dataset

```python
dataset = load_dataset("usuario/whatif-polymarket-es")
```

### 2.3 Capacidades que aprende el modelo

1. **Razonamiento causal**: "Si X cambia → entonces Y porque → con impacto Z"
2. **Anclaje en datos reales**: Siempre cita números del dataset (volumen, liquidez, %)
3. **Referencia a categorías y datos**: "La categoría crypto, con $X en volumen, se vería afectada en..." (Si se completa el notebook 5 opcional de DL, el modelo también aprenderá a referenciar clusters)
4. **Análisis de impacto en cascada**: Identifica efectos directos e indirectos
5. **Comunicación en español**: Fluidez, términos técnicos explicados

### 2.4 Training

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=SFTConfig(
        output_dir="./whatif-model",
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

### 2.5 Upload

```python
model.push_to_hub("usuario/whatif-polymarket-gemma3-qlora")
tokenizer.push_to_hub("usuario/whatif-polymarket-gemma3-qlora")
```

---

## Notebook 3: `WhatIf_RAG`

### 3.1 Documentos

| Documento | Contenido | Fuente |
|-----------|-----------|--------|
| `resumen_estadistico.txt` | Datos agregados: volumen por categoría, top mercados, métricas globales | Generado del dataset |
| `precedentes_historicos.txt` | Cambios significativos observados en los datos (meses de crecimiento/caída por categoría) | Análisis temporal del dataset |
| `dependencias_categorias.txt` | Correlaciones entre categorías, qué categorías se mueven juntas | Análisis de correlación |
| `marco_regulatorio.txt` | Contexto sobre regulación de prediction markets (USA, EU, crypto) | Redacción manual / ficticio |
| `metodologia_whatif.txt` | Framework para estructurar respuestas: identificar variable, calcular impacto directo, analizar cascada, concluir | Redacción manual |
| `comparativa_competidores.txt` | Kalshi, Metaculus, Augur: diferencias, fortalezas, mercados que cubren | Redacción manual / ficticio |
| `perfiles_clusters.txt` | Descripción de cada cluster: nombre, tamaño, categoría dominante, volumen, ejemplos | **(SOLO si se hizo el notebook 5 opcional de DL)** |

### 3.2 Stack

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Cargar documentos
loader = DirectoryLoader('./docs/', glob="*.txt")
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

### 3.3 Pipeline RAG + LLM

```python
def whatif_answer(question):
    # 1. Recuperar contexto relevante
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])

    # 2. Construir prompt con contexto
    prompt = f"""Contexto:
{context}

Pregunta: {question}

Respondé como WhatIfAnalyst, anclando tu respuesta en los datos del contexto."""

    # 3. Generar respuesta con el modelo fine-tuned
    response = generate(finetuned_model, prompt)
    return response
```

---

## Notebook 4: `WhatIf_Evaluation`

### 4.1 Evaluación del LLM (Pipeline Principal)

**Comparación Base vs Fine-tuned** con los mismos prompts:

| Dimensión | Qué se evalúa | Cómo |
|-----------|---------------|------|
| **Plausibilidad** | ¿El escenario tiene lógica? | Rubric manual (1-5) |
| **Anclaje en datos** | ¿Cita datos reales o inventa? | Verificar contra dataset |
| **Coherencia causal** | ¿La cadena causa-efecto es lógica? | Comparar ambos modelos |
| **Referencia a datos reales** | ¿Cita volúmenes, liquidez, categorías del dataset? | Verificar contra dataset (Si se hizo DL, evaluar también si referencia clusters correctamente) |
| **Fluidez en español** | ¿Es natural y claro? | Rubric manual (1-5) |

### 4.2 Test Cases por Dificultad

**Fácil** (respuesta calculable):
- "Si crypto desaparece, ¿cuánto volumen total se pierde?"
- "¿Qué categoría tiene la mayor concentración de volumen?"

**Medio** (requiere análisis):
- "Si la liquidez cae 50%, ¿qué categoría sobrevive mejor?"
- "Si los usuarios de politics migran a sports, ¿cómo cambia el ranking?"

**Difícil** (requiere razonamiento creativo):
- "Si aparece un competidor que copia los mercados de crypto, ¿qué pasa?"
- "Si Polymarket entra en Latinoamérica, ¿qué categorías crecerían más?"

### 4.3 Formato de evaluación

```python
test_prompts = [
    {"prompt": "Si el volumen de crypto cae 50%...", "difficulty": "fácil"},
    {"prompt": "Si se prohíben mercados políticos...", "difficulty": "medio"},
    # ...
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

# Evaluar con rubric
```

### 4.4 Evaluación del DL (Solo si se hizo el notebook 5 opcional)

> **Esta sección solo aplica si se completó el notebook 5 de Deep Learning.**

| Métrica | Qué mide | Target |
|---------|----------|--------|
| Silhouette Score | Separación entre clusters | > 0.3 |
| Davies-Bouldin Index | Compacidad de clusters | < 2.0 |
| Reconstruction Error | Calidad del autoencoder | Baja y estable |
| Coherencia de clusters | ¿Los clusters tienen sentido temático? | Manual |

**Test cases adicionales (solo con DL):**
- "¿Cuántos mercados tiene el cluster más grande?"
- "¿Qué clusters son más vulnerables a una caída de volumen?"
- "¿Qué cluster crecería más si Polymarket entra en Latinoamérica?"

---

## Notebook 5: `WhatIf_DeepLearning` (OPCIONAL - Mejora del Pipeline)

> **Este notebook es opcional.** El pipeline principal (Notebooks 1-4) funciona
> de forma completa sin este paso. Si se completa, enriquece los escenarios
> con información de clusters y agrega la visualización UMAP para la presentación.
>
> **Qué aporta si se hace:**
> - Escenarios "What If" más ricos (versión enriquecida con clusters)
> - Visualización UMAP de 100K mercados (impresionante para la demo)
> - Documento extra para el RAG (perfiles_clusters.txt)
> - Métricas adicionales de evaluación (silhouette, Davies-Bouldin)

### 5.1 Objetivo

Crear una representación vectorial de cada mercado y descubrir agrupaciones naturales. Los clusters resultantes alimentan al LLM con patrones que no son visibles en datos crudos.

### 5.2 Embeddings Textuales

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Liviano, corre en Colab Free

# Embeddings de la pregunta del mercado (traducida al español)
text_embeddings = model.encode(markets['question_es'].tolist())
# Resultado: matriz de (100795, 384)
```

### 5.3 Features Numéricas Normalizadas

```python
from sklearn.preprocessing import StandardScaler

numeric_features = ['volume', 'liquidity', 'spread', 'bestBid',
                    'bestAsk', 'competitorCount', 'volume_share',
                    'liquidity_ratio', 'market_age_days']

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(markets[numeric_features])
```

### 5.4 Fusión: Autoencoder

Comprimir embeddings textuales (384 dims) + features numéricas (9 dims) en un espacio latente unificado.

```python
import torch
import torch.nn as nn

class MarketAutoencoder(nn.Module):
    def __init__(self, text_dim=384, numeric_dim=9, latent_dim=32):
        super().__init__()
        input_dim = text_dim + numeric_dim  # 393

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)  # 32 dims
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, text_emb, numeric):
        x = torch.cat([text_emb, numeric], dim=1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Entrenamiento
model = MarketAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(50):
    reconstructed, latent = model(text_tensor, numeric_tensor)
    loss = criterion(reconstructed, torch.cat([text_tensor, numeric_tensor], dim=1))
    loss.backward()
    optimizer.step()
```

### 5.5 Clustering

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Obtener embeddings latentes
with torch.no_grad():
    _, latent_embeddings = model(text_tensor, numeric_tensor)
    latent_np = latent_embeddings.numpy()

# Encontrar K óptimo
silhouette_scores = []
for k in range(5, 25):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(latent_np)
    score = silhouette_score(latent_np, labels)
    silhouette_scores.append((k, score))

# Clustering final
best_k = max(silhouette_scores, key=lambda x: x[1])[0]
kmeans = KMeans(n_clusters=best_k, random_state=42)
markets['cluster'] = kmeans.fit_predict(latent_np)
```

### 5.6 Visualización con UMAP

```python
import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(latent_np)

plt.figure(figsize=(14, 10))
scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                      c=markets['cluster'], cmap='tab20',
                      alpha=0.5, s=3)
plt.colorbar(scatter, label='Cluster')
plt.title('Mapa de 100K Mercados de Polymarket - Embeddings + Clustering')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.savefig('market_clusters.png', dpi=150)
```

### 5.7 Análisis de Clusters

```python
# Caracterizar cada cluster
cluster_profiles = markets.groupby('cluster').agg({
    'volume': ['mean', 'sum', 'count'],
    'liquidity': 'mean',
    'spread': 'mean',
    'category': lambda x: x.mode()[0],  # categoría dominante
    'question_es': 'first'  # ejemplo representativo
}).round(2)

# Dar nombres descriptivos a cada cluster basándose en su perfil
# Ej: Cluster 0 = "Crypto Alta Liquidez", Cluster 1 = "Política Bajo Volumen", etc.
```

### 5.8 Outputs para el Pipeline

Si se completa este notebook, genera archivos que enriquecen los notebooks principales:

```python
# 1. Perfil de cada cluster (para RAG docs)
cluster_profiles.to_csv('cluster_profiles.csv')

# 2. Asignación de cluster por mercado (para dataset Q&A)
markets[['id', 'question_es', 'cluster', 'cluster_name']].to_csv('market_clusters.csv')

# 3. Visualización (para presentación)
# market_clusters.png

# 4. Embeddings latentes (por si se necesitan)
np.save('market_latent_embeddings.npy', latent_np)
```

---

## Ejemplos de Interacción

### Preguntas Factuales (el RAG ancla la respuesta en datos)

**Q1 — Panorama general** *(RAG: resumen_estadistico.txt)*
> **Usuario**: ¿Cuál es el estado actual de Polymarket?
>
> **WhatIfAnalyst**: Polymarket gestiona 100,795 mercados agrupados en 43,840 eventos. El volumen total acumulado es de $7.99B, con un volumen diario de ~$150M y $315M en liquidez total. La categoría con mayor volumen es crypto, seguida por politics y sports. El mercado individual con mayor actividad en las últimas 24h registra $14.96M en volumen.

**Q2 — Salud de una categoría** *(RAG: resumen_estadistico.txt)*
> **Usuario**: ¿Qué tan sana es la categoría de sports?
>
> **WhatIfAnalyst**: Sports tiene $420M en volumen acumulado (5.3% del total), con un liquidity_ratio promedio de 0.14, lo que la ubica justo por encima del umbral de mercado sano (0.1). Su category_concentration es 0.42 — los 10 mercados más grandes concentran el 42% del volumen, indicando diversificación moderada. El crecimiento de las últimas 4 semanas es del 12% en mercados nuevos creados.

**Q3 — Top mercados** *(RAG: resumen_estadistico.txt)*
> **Usuario**: ¿Cuáles son los mercados más activos de Polymarket ahora mismo?
>
> **WhatIfAnalyst**: Los mercados con mayor volume24hr actualmente son: (1) "Will the Fed cut rates in March?" con $14.96M, (2) "Bitcoin above $100K by June?" con $8.2M, (3) "Next UK Prime Minister" con $5.1M. Estos tres mercados representan el 18.8% del volumen diario total ($150M). Todos tienen spreads bajos (< 0.03) y liquidity_ratio > 0.2, indicando mercados muy sanos.

---

### Preguntas What-If por Tipo

**Q4 — Tipo 1: Caída de volumen** *(RAG: resumen_estadistico.txt + dependencias_categorias.txt)*
> **Usuario**: Si el volumen de crypto cae un 50%, ¿qué impacto tendría en Polymarket?
>
> **WhatIfAnalyst**: Crypto representa actualmente el 35% del volumen total (~$2.8B). Una caída del 50% eliminaría $1.4B en volumen, reduciendo el total de Polymarket a ~$6.59B (-17.5%).
>
> **Mercados más afectados**: "Bitcoin above $100K by June?" ($8.2M/día), "ETH price end of year" ($3.1M/día) y "Next crypto to hit $1B market cap" ($2.4M/día).
>
> **Efecto cascada**: Según las correlaciones entre categorías, politics y sports tienen baja dependencia de crypto (correlación < 0.15), así que no se verían arrastradas. Sin embargo, la categoría "tech" tiene correlación 0.38 con crypto y podría perder entre un 10-15% de volumen por contagio.
>
> *(Con DL opcional: los clusters "Crypto Alta Liquidez" y "Crypto Especulativo" perderían el 50% de sus miembros)*

**Q5 — Tipo 2: Migración** *(RAG: resumen_estadistico.txt + precedentes_historicos.txt)*
> **Usuario**: Si los traders de politics migran a crypto después de las elecciones, ¿qué pasa?
>
> **WhatIfAnalyst**: Politics tiene $1.44B en volumen. Si el 40% migra a crypto post-elecciones ($576M), crypto pasaría de $2.8B a $3.38B (+20.6%).
>
> **Absorción por liquidez**: Los mercados de crypto con mayor liquidity_ratio absorberían primero: "Bitcoin above $100K" (ratio 0.28), "ETH merge impact" (ratio 0.22) y "Solana ecosystem growth" (ratio 0.19). Estos 3 mercados podrían absorber ~$200M sin degradar spreads.
>
> **Precedente histórico**: Después de las elecciones de 2024, se observó una migración del 30% del volumen de politics hacia crypto y sports en las 4 semanas siguientes, con el volumen diario de politics cayendo de $45M a $12M.

**Q6 — Tipo 3: Eliminación regulatoria** *(RAG: marco_regulatorio.txt + resumen_estadistico.txt)*
> **Usuario**: ¿Qué pasaría si la SEC prohíbe los mercados de crypto en Polymarket?
>
> **WhatIfAnalyst**: Se eliminarían aproximadamente 35,000 mercados activos de crypto con $2.8B en volumen acumulado — el 35% del volumen total de Polymarket.
>
> **Impacto directo**: El volumen total caería a $5.19B. Los mercados más afectados serían los de precio de tokens, DeFi y eventos de exchanges.
>
> **Contexto regulatorio**: La SEC ha intensificado su enfoque en activos digitales desde 2023. Polymarket opera con licencia CFTC para ciertos mercados, pero los de crypto puro podrían caer bajo jurisdicción de la SEC. Kalshi, su competidor, ya fue bloqueado de ofrecer mercados de eventos políticos por disputa con la CFTC.
>
> **Migración estimada**: El 60% del volumen crypto migraría a categorías no reguladas (tech, ciencia, entretenimiento), el 25% se perdería completamente, y el 15% podría migrar a competidores internacionales como Augur (descentralizado).

**Q7 — Tipo 4: Shock de liquidez** *(RAG: resumen_estadistico.txt + metodologia_whatif.txt)*
> **Usuario**: Si la liquidez total de Polymarket cae a la mitad, ¿qué mercados sobreviven?
>
> **WhatIfAnalyst**: Con la liquidez reduciéndose de $315M a $157.5M, sobrevivirían los mercados con liquidity_ratio > 0.15: aproximadamente 23,000 de los 100,795 actuales (22.8%).
>
> **Por categoría**: Crypto retiene el 78% de sus mercados (alta liquidez inherente), politics el 45% (muchos mercados de nicho quedarían fuera), sports el 62%.
>
> **Mercados que desaparecen primero**: Los de cola larga — mercados con < $10K en volumen y spreads > 0.15. Son ~65,000 mercados que solo representan el 2% del volumen total pero el 64% del conteo.
>
> **Efecto en spreads**: Los mercados supervivientes verían sus spreads aumentar ~40%, pasando de un promedio de 0.04 a 0.056, lo que degradaría la experiencia del usuario y podría provocar una espiral de pérdida de volumen adicional.

**Q8 — Tipo 5: Explosión** *(RAG: resumen_estadistico.txt + precedentes_historicos.txt)*
> **Usuario**: Si la categoría de IA crece 10x en los próximos 6 meses, ¿qué pasa?
>
> **WhatIfAnalyst**: Actualmente IA/tech tiene $380M en volumen (4.8% del total). Un crecimiento de 10x la llevaría a $3.8B, convirtiéndola en la categoría #1 y superando a crypto ($2.8B).
>
> **Necesidades de liquidez**: Para mantener spreads sanos (< 0.05), necesitaría ~$570M adicionales en liquidez — casi el doble de la liquidez total actual de Polymarket ($315M). Sin esta inyección, los spreads se dispararían y la experiencia degradaría.
>
> **Viabilidad temporal**: El crecimiento actual de IA/tech es del 22% mensual (la categoría de mayor crecimiento). A ese ritmo, alcanzar 10x tomaría ~11 meses. Pero un evento catalizador (lanzamiento de AGI, regulación de IA, etc.) podría comprimir esto a 3-4 meses, similar a lo que ocurrió con crypto durante el ciclo de 2024.
>
> **Impacto en el ecosistema**: La concentración del volumen cambiaría radicalmente — Polymarket pasaría de ser "crypto-first" a "AI-first", atrayendo un nuevo perfil de trader más enfocado en tecnología.

---

### Preguntas de Análisis Comparativo y Competencia

**Q9 — Competidores** *(RAG: comparativa_competidores.txt)*
> **Usuario**: ¿Cómo se compara Polymarket con sus competidores?
>
> **WhatIfAnalyst**: Polymarket lidera en volumen ($7.99B) y variedad (100K mercados). En comparación:
>
> **Kalshi**: Regulado por CFTC en EEUU, enfocado en eventos económicos y clima. Menor volumen pero acceso a traders institucionales. Perdió la batalla legal para ofrecer mercados políticos en 2023.
>
> **Metaculus**: No es un mercado de dinero real — es una plataforma de predicción por puntos. Fuerte en pronósticos científicos y tecnológicos. No compite directamente en volumen.
>
> **Augur**: Descentralizado (Ethereum), sin custodia. Volumen muy inferior pero inmune a regulación por su naturaleza on-chain. Podría captar volumen si Polymarket enfrenta restricciones regulatorias.
>
> **Ventaja de Polymarket**: Liquidez profunda, interfaz UX superior, y efecto de red — el 80% de los traders nuevos eligen Polymarket por liquidez.

**Q10 — Regulación** *(RAG: marco_regulatorio.txt + comparativa_competidores.txt)*
> **Usuario**: ¿Es viable que Polymarket sea prohibido completamente en EEUU?
>
> **WhatIfAnalyst**: Polymarket ya fue multado por la CFTC en 2022 ($1.4M) y actualmente bloquea usuarios de EEUU (requiere VPN). Una prohibición total es poco probable pero posible:
>
> **Escenario**: Si EEUU prohíbe prediction markets crypto, el impacto sería: pérdida del ~45% de usuarios estimados (los que usan VPN desde EEUU), caída de volumen diario de $150M a ~$82M.
>
> **Migración**: Kalshi captaría la demanda de mercados económicos/políticos (regulados). Augur captaría la demanda crypto (descentralizado). Polymarket retendría usuarios internacionales.
>
> **Precedente**: La prohibición de online poker en EEUU (2011) mostró que el 60% del volumen se perdió globalmente, no solo en EEUU, por efecto de confianza.

---

### Preguntas de Correlación y Dependencias

**Q11 — Correlaciones entre categorías** *(RAG: dependencias_categorias.txt)*
> **Usuario**: Si crypto cae, ¿arrastra a otras categorías?
>
> **WhatIfAnalyst**: Según el análisis de correlación entre categorías:
>
> **Alta correlación con crypto**: tech/IA (0.38) — comparten perfil de trader y se mueven juntas. Una caída del 30% en crypto históricamente coincide con una caída del 10-12% en tech.
>
> **Baja correlación**: politics (0.08) y sports (0.05) — son prácticamente independientes. Sus traders tienen motivaciones distintas (información política vs. conocimiento deportivo).
>
> **Correlación negativa**: entertainment (-0.12) — cuando crypto cae, entretenimiento sube ligeramente, posiblemente porque los traders buscan mercados "divertidos" de menor riesgo.
>
> **Conclusión**: Una crisis de crypto afectaría al 40% del volumen total (crypto + tech arrastrado), pero el 60% restante (politics, sports, entertainment, ciencia) se mantendría estable.

**Q12 — Análisis temporal** *(RAG: precedentes_historicos.txt + resumen_estadistico.txt)*
> **Usuario**: ¿Hay patrones estacionales en el volumen de Polymarket?
>
> **WhatIfAnalyst**: Sí, se observan ciclos claros:
>
> **Politics**: Picos en años electorales — el volumen se multiplica 4-5x en los 6 meses antes de elecciones (EEUU, UK, Francia). Post-elección, cae un 70% en las 4 semanas siguientes.
>
> **Sports**: Picos durante temporadas de NFL/NBA (octubre-junio) y eventos globales (Mundial, Olimpiadas). Volumen mínimo en julio-agosto.
>
> **Crypto**: Correlaciona con ciclos de mercado — bull runs disparan el volumen. El halving de Bitcoin 2024 generó un aumento del 180% en mercados crypto en Polymarket.
>
> **Global**: Diciembre es consistentemente el mes de menor volumen (-25% vs promedio), posiblemente por holidays y cierre de posiciones de fin de año.

---

### Preguntas Complejas (Razonamiento Multi-paso)

**Q13 — Escenario encadenado** *(RAG: dependencias_categorias.txt + marco_regulatorio.txt + resumen_estadistico.txt)*
> **Usuario**: Si Europa regula los prediction markets y EEUU sigue bloqueado, ¿qué le pasa a Polymarket?
>
> **WhatIfAnalyst**: Este escenario combina dos impactos:
>
> **Impacto EEUU (ya activo)**: ~45% de usuarios usan VPN desde EEUU. Si el enforcement se endurece, el volumen diario caería de $150M a ~$82M.
>
> **Impacto Europa (nuevo)**: Si la UE regula bajo MiCA o marco similar, depende del tipo de regulación:
> - *Prohibición total*: Pérdida adicional del ~25% de usuarios (europeos) → volumen diario a ~$61M.
> - *Regulación con licencia*: Polymarket podría adaptarse y retener el 80% del volumen europeo, pero con costos operativos mayores.
>
> **Escenario worst-case**: Volumen diario cae a $61M (59% de caída). Polymarket se convierte en plataforma para Asia, Latam y Africa. La liquidez se fragmenta, spreads suben, y se pierde el efecto de red.
>
> **Escenario adaptativo**: Polymarket obtiene licencia en UE, mantiene operación, y se posiciona como el primer prediction market regulado globalmente — lo que podría atraer capital institucional y más que compensar la pérdida.

**Q14 — Análisis de resiliencia** *(RAG: resumen_estadistico.txt + metodologia_whatif.txt)*
> **Usuario**: ¿Cuál es el punto de quiebre de Polymarket? ¿Cuánto volumen puede perder antes de colapsar?
>
> **WhatIfAnalyst**: El punto de quiebre depende de la liquidez, no del volumen.
>
> **Umbral crítico**: Cuando el liquidity_ratio promedio cae por debajo de 0.05, los spreads se amplían tanto que los traders dejan de operar — creando una espiral negativa. Esto ocurriría si la liquidez total cae por debajo de ~$100M (una caída del 68% desde los $315M actuales).
>
> **En términos de volumen**: El volumen diario mínimo viable es ~$40M — por debajo de esto, no hay suficiente actividad para mantener precios informativos. Actualmente está en $150M, así que podría soportar una caída del 73%.
>
> **Mercados que colapsan primero**: Los de cola larga (< $50K volumen) desaparecerían con una caída del 30%. Los mercados core (crypto top 100, politics top 50) sobrevivirían incluso con caídas del 80%.
>
> **Conclusión**: Polymarket es más resiliente de lo que parece — el 80% de su volumen se concentra en el 5% de los mercados. Esos mercados core pueden sostener la plataforma incluso en escenarios adversos extremos.

**Q15 — Competidor directo** *(RAG: comparativa_competidores.txt + resumen_estadistico.txt + dependencias_categorias.txt)*
> **Usuario**: Si aparece un competidor que copia todos los mercados de crypto de Polymarket con mejores fees, ¿qué pasa?
>
> **WhatIfAnalyst**: Crypto representa el 35% del volumen ($2.8B) y es la categoría con traders más sensibles a fees (perfil DeFi, acostumbrados a comparar yields).
>
> **Migración estimada**: En los primeros 3 meses, el competidor captaría ~20% del volumen crypto ($560M) si ofrece fees 50% menores. Los traders de alto volumen migran primero.
>
> **Efecto en liquidez**: La liquidez crypto caería ~30%, ampliando spreads. Esto aceleraría la migración — el clásico "death spiral" de liquidez. En 6 meses, el competidor podría tener el 40-50% del volumen crypto.
>
> **Defensa de Polymarket**: El efecto de red es su mayor ventaja — todos los traders están donde hay liquidez. Polymarket podría: (1) reducir fees temporalmente, (2) lanzar programa de incentivos para market makers, (3) enfocarse en diferenciación UX.
>
> **Lo que NO se afecta**: Politics (0.08 correlación con crypto), sports (0.05), y el resto del ecosistema seguirían intactos. Polymarket mantendría el 65% de su volumen total incluso perdiendo todo crypto.
>
> **Precedente**: Cuando dYdX lanzó su chain propia, capturó ~15% del volumen de derivados crypto en 6 meses, pero el líder (Binance) retuvo el 60% por efecto de red.

---

## Stack Técnico

### Stack Principal

| Componente | Herramienta | Versión / Nota |
|------------|-------------|----------------|
| Runtime | Google Colab Free | GPU T4 |
| Datos | pandas | Carga y feature engineering |
| Traducción | Helsinki-NLP/opus-mt-en-es | Igual que Banking Assistant |
| LLM | Gemma 3 4B Instruct | Base model |
| Fine-tuning | Unsloth + QLoRA | r=16, 4-bit |
| Training | trl (SFTTrainer) | 3 epochs |
| Vectorstore | FAISS (faiss-gpu) | RAG retrieval |
| RAG | langchain | Document loading + retrieval |
| Embeddings RAG | sentence-transformers (all-MiniLM-L6-v2) | 384 dims, liviano |
| Dataset Hub | HuggingFace Hub | Dataset + modelo |

### Stack Opcional (Notebook 5 - DL)

| Componente | Herramienta | Versión / Nota |
|------------|-------------|----------------|
| Autoencoder | PyTorch | Fusión texto + numérico → 32 dims |
| Clustering | scikit-learn (KMeans) | K óptimo por silhouette |
| Visualización | UMAP + matplotlib | Mapa 2D de mercados |

---

## Comparación con Banking Assistant

| Aspecto | Banking Assistant | WhatIfAnalyst |
|---------|-------------------|---------------|
| Dominio | Banca retail | Mercados predictivos |
| Dataset base | Bitext banking chatbot | Polymarket CSVs (100K markets) |
| Traducción | EN→ES (Helsinki) | EN→ES (Helsinki) |
| **Deep Learning** | **No** | **Opcional: Autoencoder + Clustering (si da el tiempo)** |
| Tipo de respuesta | Factual ("tu saldo es...") | Analítica ("si pasa X, entonces Y...") |
| Complejidad razonamiento | Baja (lookup) | Alta (causal chain) |
| Modelo LLM | Gemma 3 4B + QLoRA | Gemma 3 4B + QLoRA |
| RAG | langchain + FAISS | langchain + FAISS |
| Evaluación | Correcto/Incorrecto | Plausible + Anclado + Coherente |
| Notebooks | 4 | 4 principales + 1 opcional (DL) |
| Visualización | No | Opcional: mapa UMAP de mercados |
| Wow factor | Medio | Alto (Muy alto con DL opcional) |

---

## Fortalezas del Proyecto

1. **Fine-tuning con datos reales**: LLM entrenado con escenarios generados de un dataset de $8B en volumen
2. **RAG con documentos especializados**: Contexto documental que ancla las respuestas en datos verificables
3. **Dataset generado automáticamente**: Template-based con datos reales, escalable y reproducible
4. **Datos reales de Polymarket**: 100K mercados, no datos ficticios ni sintéticos
5. **En español**: Traducción EN→ES con Helsinki-NLP, misma competencia que el Banking Assistant
6. **Evaluación rigurosa**: Comparación base vs fine-tuned con rubrics por dificultad
7. **(Opcional) Deep Learning real**: Si da el tiempo, autoencoder + clustering + UMAP para enriquecer todo
8. **(Opcional) Visualmente impresionante**: Mapa UMAP de 100K mercados para la presentación

## Riesgos y Mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Dataset Q&A de baja calidad | Template-based con datos reales, no requiere generación creativa |
| Fine-tuning no mejora sobre el modelo base | Comparar con rubrics, ajustar prompts del dataset, verificar overfitting |
| RAG recupera documentos irrelevantes | Ajustar chunk_size, probar diferentes k, validar con queries de prueba |
| Colab Free sin GPU suficiente | QLoRA es eficiente en 4-bit, all-MiniLM-L6-v2 corre en CPU |
| *(Solo si se hace DL)* Autoencoder no converge | Arquitectura simple (3 layers), datos normalizados, learning rate conservador |
| *(Solo si se hace DL)* Clusters sin sentido | Validar con silhouette score, probar diferentes K, inspección manual |
| *(Solo si se hace DL)* Embeddings de 100K textos lentos | Batch processing, all-MiniLM-L6-v2 es rápido (~1K textos/seg) |

---

## Entregables

### Entregables Principales
- [ ] 4 notebooks en GitHub (Dataset, FineTuning, RAG, Evaluation)
- [ ] Dataset en HuggingFace Hub (`usuario/whatif-polymarket-es`)
- [ ] Modelo fine-tuned en HuggingFace Hub (`usuario/whatif-polymarket-gemma3-qlora`)
- [ ] Presentación
- [ ] Post LinkedIn

### Entregables Opcionales (si da el tiempo)
- [ ] Notebook 5 de Deep Learning (Autoencoder + Clustering)
- [ ] Visualización UMAP de clusters
- [ ] Documento perfiles_clusters.txt para RAG
