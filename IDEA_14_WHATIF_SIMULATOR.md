# Idea 14: WhatIfAnalyst - Simulador de Escenarios con Deep Learning

## Concepto

Un asistente en español que combina **Deep Learning** (embeddings + clustering de mercados) con un **LLM fine-tuneado** (Gemma 3 + QLoRA) para responder preguntas hipotéticas sobre mercados predictivos de Polymarket.

El DL descubre patrones y agrupa mercados → el LLM interpreta y explica en lenguaje natural → el RAG aporta contexto documental.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   DL Model   │───→│  LLM Gemma3  │───→│   Usuario    │
│ Embeddings + │    │  Fine-tuned  │    │  (español)   │
│  Clustering  │    │   + RAG      │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
  Descubre            Interpreta          Pregunta
  patrones            y explica           "¿Qué pasaría si...?"
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

| # | Notebook | Fase Lifecycle | Técnicas |
|---|----------|---------------|----------|
| 1 | `WhatIf_Dataset` | Definición + Dataset | pandas, Helsinki-NLP/opus-mt-en-es, feature engineering, HuggingFace Hub |
| 2 | `WhatIf_DeepLearning` | Modelo DL | sentence-transformers, autoencoder, UMAP, KMeans, PyTorch |
| 3 | `WhatIf_FineTuning` | Modelo LLM | Unsloth, QLoRA, Gemma 3 4B, trl, bitsandbytes |
| 4 | `WhatIf_Evaluation` | Evaluación | Métricas DL (silhouette, Davies-Bouldin) + métricas LLM (calidad, anclaje) |
| 5 | `WhatIf_RAG` | RAG + Integración | langchain, FAISS, sentence-transformers |

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

#### Tipos de escenarios

**Tipo 1 - Cambio de volumen:**
```
Input:  "Si el volumen de {categoría} cae un {X}%, ¿qué impacto tendría en Polymarket?"
Output: "Actualmente {categoría} representa el {Y}% del volumen total (${Z}M).
         Una caída del {X}% reduciría el volumen total en ${W}M ({P}% del total).
         Los mercados más afectados serían: {top_3_mercados_categoria}.
         Según el clustering de mercados, los clusters {N1} y {N2} perderían
         el {Q}% de sus miembros. Las categorías {cat2} y {cat3} podrían
         absorber parte del volumen."
```
Variables: categoría = [crypto, politics, sports, ...], X = [20, 30, 50, 70]

**Tipo 2 - Migración entre categorías:**
```
Input:  "Si los usuarios de {cat1} migran a {cat2}, ¿cómo cambiaría el mercado?"
Output: "Actualmente {cat1} tiene ${V1}M en volumen y {cat2} tiene ${V2}M.
         Si el {X}% del volumen migra, {cat2} crecería a ${V2+migración}M,
         convirtiéndose en la {ranking}ª categoría por volumen.
         Los mercados de {cat2} con mayor liquidez absorberían primero:
         {top_3_mercados_cat2_por_liquidez}."
```

**Tipo 3 - Eliminación de categoría (regulación):**
```
Input:  "¿Qué pasaría si se prohíben los mercados de {categoría}?"
Output: "Se eliminarían {N} mercados activos con ${V}M en volumen total.
         Esto representa el {P}% del volumen de Polymarket.
         Impacto por cluster: el cluster '{nombre_cluster}' perdería
         el {Q}% de sus mercados. Históricamente, el {R}% del volumen
         de categorías eliminadas migra a crypto."
```

**Tipo 4 - Shock de liquidez:**
```
Input:  "Si la liquidez total de Polymarket cae a la mitad, ¿qué mercados sobreviven?"
Output: "Con liquidez reducida al 50%, sobrevivirían los mercados con
         liquidity_ratio > {umbral}: {N} mercados de los {total} actuales.
         Por categoría: crypto retiene {P1}%, politics {P2}%, sports {P3}%.
         El cluster '{nombre}' es el más resiliente con {R}% de supervivencia."
```

**Tipo 5 - Explosión de categoría:**
```
Input:  "Si {categoría} crece {X}x en volumen, ¿cómo afecta al ecosistema?"
Output: "Un crecimiento de {X}x llevaría {categoría} de ${V}M a ${V*X}M,
         superando a {categorías_superadas} en volumen.
         Necesitaría ${L}M adicionales en liquidez para mantener spreads sanos.
         Basado en el crecimiento real de las últimas 4 semanas ({growth}%),
         este escenario requeriría {T} semanas al ritmo actual."
```

**Tipo 6 - Escenario basado en clustering:**
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

## Notebook 2: `WhatIf_DeepLearning`

### 2.1 Objetivo

Crear una representación vectorial de cada mercado y descubrir agrupaciones naturales. Los clusters resultantes alimentan al LLM con patrones que no son visibles en datos crudos.

### 2.2 Embeddings Textuales

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Liviano, corre en Colab Free

# Embeddings de la pregunta del mercado (traducida al español)
text_embeddings = model.encode(markets['question_es'].tolist())
# Resultado: matriz de (100795, 384)
```

### 2.3 Features Numéricas Normalizadas

```python
from sklearn.preprocessing import StandardScaler

numeric_features = ['volume', 'liquidity', 'spread', 'bestBid',
                    'bestAsk', 'competitorCount', 'volume_share',
                    'liquidity_ratio', 'market_age_days']

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(markets[numeric_features])
```

### 2.4 Fusión: Autoencoder

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

### 2.5 Clustering

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

### 2.6 Visualización con UMAP

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

### 2.7 Análisis de Clusters

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

### 2.8 Outputs para el LLM

El notebook genera archivos que se usan en los notebooks siguientes:

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

## Notebook 3: `WhatIf_FineTuning`

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

### 3.2 Dataset

```python
dataset = load_dataset("usuario/whatif-polymarket-es")
```

### 3.3 Capacidades que aprende el modelo

1. **Razonamiento causal**: "Si X cambia → entonces Y porque → con impacto Z"
2. **Anclaje en datos reales**: Siempre cita números del dataset (volumen, liquidez, %)
3. **Referencia a clusters**: "El cluster 'Crypto Alta Liquidez' se vería afectado en..."
4. **Análisis de impacto en cascada**: Identifica efectos directos e indirectos
5. **Comunicación en español**: Fluidez, términos técnicos explicados

### 3.4 Training

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

### 3.5 Upload

```python
model.push_to_hub("usuario/whatif-polymarket-gemma3-qlora")
tokenizer.push_to_hub("usuario/whatif-polymarket-gemma3-qlora")
```

---

## Notebook 4: `WhatIf_Evaluation`

### 4.1 Evaluación del DL (Notebook 2)

| Métrica | Qué mide | Target |
|---------|----------|--------|
| Silhouette Score | Separación entre clusters | > 0.3 |
| Davies-Bouldin Index | Compacidad de clusters | < 2.0 |
| Reconstruction Error | Calidad del autoencoder | Baja y estable |
| Coherencia de clusters | ¿Los clusters tienen sentido temático? | Manual |

### 4.2 Evaluación del LLM (Notebook 3)

**Comparación Base vs Fine-tuned** con los mismos prompts:

| Dimensión | Qué se evalúa | Cómo |
|-----------|---------------|------|
| **Plausibilidad** | ¿El escenario tiene lógica? | Rubric manual (1-5) |
| **Anclaje en datos** | ¿Cita datos reales o inventa? | Verificar contra dataset |
| **Coherencia causal** | ¿La cadena causa-efecto es lógica? | Comparar ambos modelos |
| **Referencia a clusters** | ¿Usa los clusters del DL? | Solo el fine-tuned debería |
| **Fluidez en español** | ¿Es natural y claro? | Rubric manual (1-5) |

### 4.3 Test Cases por Dificultad

**Fácil** (respuesta calculable):
- "Si crypto desaparece, ¿cuánto volumen total se pierde?"
- "¿Cuántos mercados tiene el cluster más grande?"

**Medio** (requiere análisis):
- "Si la liquidez cae 50%, ¿qué categoría sobrevive mejor?"
- "¿Qué clusters son más vulnerables a una caída de volumen?"

**Difícil** (requiere razonamiento creativo):
- "Si aparece un competidor que copia los mercados de crypto, ¿qué pasa?"
- "¿Qué cluster crecería más si Polymarket entra en Latinoamérica?"

### 4.4 Formato de evaluación

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

---

## Notebook 5: `WhatIf_RAG`

### 5.1 Documentos

| Documento | Contenido | Fuente |
|-----------|-----------|--------|
| `resumen_estadistico.txt` | Datos agregados: volumen por categoría, top mercados, métricas globales | Generado del dataset |
| `perfiles_clusters.txt` | Descripción de cada cluster: nombre, tamaño, categoría dominante, volumen, ejemplos | Output del Notebook 2 (DL) |
| `precedentes_historicos.txt` | Cambios significativos observados en los datos (meses de crecimiento/caída por categoría) | Análisis temporal del dataset |
| `dependencias_categorias.txt` | Correlaciones entre categorías, qué categorías se mueven juntas | Análisis de correlación |
| `marco_regulatorio.txt` | Contexto sobre regulación de prediction markets (USA, EU, crypto) | Redacción manual / ficticio |
| `metodologia_whatif.txt` | Framework para estructurar respuestas: identificar variable, calcular impacto directo, analizar cascada, concluir | Redacción manual |
| `comparativa_competidores.txt` | Kalshi, Metaculus, Augur: diferencias, fortalezas, mercados que cubren | Redacción manual / ficticio |

### 5.2 Stack

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

### 5.3 Pipeline RAG + LLM

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

## Ejemplos de Interacción

### Ejemplo 1: Eliminación de categoría
> **Usuario**: ¿Qué pasaría si se prohíben los mercados de política en Polymarket?
>
> **WhatIfAnalyst**: Según los datos actuales, los mercados políticos representan aproximadamente el 18% del volumen total y generan algunos de los eventos con mayor volumen individual (como la decisión de la Fed con $14.96M en 24h).
>
> **Impacto directo**: Pérdida estimada de ~$1.44B en volumen histórico. Los 5 mercados políticos más activos concentran el 35% de ese volumen.
>
> **Impacto en clusters**: El cluster "Política Alta Visibilidad" (42 mercados, $890M volumen) desaparecería completamente. El cluster "Geopolítica y Regulación" perdería el 65% de sus miembros.
>
> **Efecto cascada**: Históricamente, cuando una categoría pierde volumen, crypto absorbe ~60% de la migración. Deportes absorbería ~25% y el 15% restante se perdería.
>
> **Liquidez**: La liquidez total caería ~12%, pero se concentraría en menos mercados, potencialmente mejorando la profundidad en crypto y deportes.

### Ejemplo 2: Escenario basado en clusters
> **Usuario**: ¿Qué pasaría si desaparece el cluster de mercados de baja liquidez?
>
> **WhatIfAnalyst**: El cluster "Baja Liquidez" contiene 12,340 mercados (12% del total) pero solo $23M en volumen (0.3% del total). Son mercados de nicho con spreads altos (promedio 0.18).
>
> **Si desaparecen**: El impacto en volumen sería mínimo (-0.3%), pero se eliminaría el 12% de la diversidad de mercados. Los mercados más afectados son de categorías minoritarias como cultura y ciencia.
>
> **Paradoja**: Aunque individualmente son insignificantes, estos mercados de cola larga son los que dan diversidad temática a Polymarket. Sin ellos, la plataforma se concentraría aún más en crypto y política.

### Ejemplo 3: Crecimiento
> **Usuario**: Si deportes crece 3x en volumen, ¿cómo cambia el ecosistema?
>
> **WhatIfAnalyst**: Actualmente deportes tiene $420M en volumen (5.3% del total). Un crecimiento de 3x lo llevaría a $1,260M, superando a política y acercándose a crypto.
>
> **Necesidades**: Para mantener spreads sanos (< 0.05), necesitaría ~$180M adicionales en liquidez. El cluster "Deportes Competitivos" pasaría de 2,100 a ~6,300 mercados.
>
> **Proyección**: Al ritmo de crecimiento actual de deportes (12% mensual), alcanzar 3x tomaría ~10 meses. Pero con un evento catalizador (Mundial, Olimpiadas), podría acelerarse a 3-4 meses.

---

## Stack Técnico Completo

| Componente | Herramienta | Versión / Nota |
|------------|-------------|----------------|
| Runtime | Google Colab Free | GPU T4 |
| Datos | pandas | Carga y feature engineering |
| Traducción | Helsinki-NLP/opus-mt-en-es | Igual que Banking Assistant |
| Embeddings texto | sentence-transformers (all-MiniLM-L6-v2) | 384 dims, liviano |
| Autoencoder | PyTorch | Fusión texto + numérico → 32 dims |
| Clustering | scikit-learn (KMeans) | K óptimo por silhouette |
| Visualización | UMAP + matplotlib | Mapa 2D de mercados |
| LLM | Gemma 3 4B Instruct | Base model |
| Fine-tuning | Unsloth + QLoRA | r=16, 4-bit |
| Training | trl (SFTTrainer) | 3 epochs |
| Vectorstore | FAISS (faiss-gpu) | RAG retrieval |
| RAG | langchain | Document loading + retrieval |
| Dataset Hub | HuggingFace Hub | Dataset + modelo |

---

## Comparación con Banking Assistant

| Aspecto | Banking Assistant | WhatIfAnalyst |
|---------|-------------------|---------------|
| Dominio | Banca retail | Mercados predictivos |
| Dataset base | Bitext banking chatbot | Polymarket CSVs (100K markets) |
| Traducción | EN→ES (Helsinki) | EN→ES (Helsinki) |
| **Deep Learning** | **No** | **Sí: Autoencoder + Clustering** |
| Tipo de respuesta | Factual ("tu saldo es...") | Analítica ("si pasa X, entonces Y...") |
| Complejidad razonamiento | Baja (lookup) | Alta (causal chain) |
| Modelo LLM | Gemma 3 4B + QLoRA | Gemma 3 4B + QLoRA |
| RAG | langchain + FAISS | langchain + FAISS |
| Evaluación | Correcto/Incorrecto | Plausible + Anclado + Coherente |
| Notebooks | 4 | 5 (+ notebook DL) |
| Visualización | No | Sí: mapa UMAP de mercados |
| Wow factor | Medio | Muy alto |

---

## Fortalezas del Proyecto

1. **Deep Learning real**: No solo fine-tuning de LLM, también autoencoder + clustering
2. **Pipeline DL → LLM**: El DL descubre patrones, el LLM los explica (producción real)
3. **Dataset generado automáticamente**: Template-based, sin depender de trabajo manual
4. **Datos reales**: $8B en volumen, 100K mercados (no datos ficticios)
5. **Visualmente impresionante**: Mapa UMAP de 100K mercados para presentación
6. **Multimodal**: Combina texto + datos numéricos en un solo modelo
7. **En español**: Misma competencia de traducción que el Banking Assistant
8. **5 notebooks**: Un notebook más que el Banking Assistant, demostrando mayor ambición

## Riesgos y Mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Autoencoder no converge | Arquitectura simple (3 layers), datos normalizados, learning rate conservador |
| Clusters sin sentido | Validar con silhouette score, probar diferentes K, inspección manual |
| Dataset Q&A de baja calidad | Template-based con datos reales, no requiere generación creativa |
| Colab Free sin GPU suficiente | all-MiniLM-L6-v2 corre en CPU, autoencoder es liviano, QLoRA es eficiente |
| Embeddings de 100K textos lentos | Batch processing, all-MiniLM-L6-v2 es rápido (~1K textos/seg) |

---

## Entregables

- [ ] 5 notebooks en GitHub
- [ ] Dataset en HuggingFace Hub (`usuario/whatif-polymarket-es`)
- [ ] Modelo fine-tuned en HuggingFace Hub (`usuario/whatif-polymarket-gemma3-qlora`)
- [ ] Visualización UMAP de clusters
- [ ] Presentación
- [ ] Post LinkedIn
