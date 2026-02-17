# Comparativa: WhatIfAnalyst (Polymarket) vs ECGAssistant (ECG)

## Resumen Ejecutivo

Dos candidatos para la práctica final del bootcamp de KeepCoding. Ambos comparten el mismo stack de LLM (Gemma 3 4B + QLoRA + FAISS + langchain), pero difieren fundamentalmente en dominio, estructura del pipeline, y rol del Deep Learning.

---

## Tabla Comparativa General

| Dimensión | WhatIfAnalyst (Polymarket) | ECGAssistant (ECG) |
|-----------|---------------------------|---------------------|
| **Dominio** | Mercados predictivos | Cardiología / ECG |
| **Dataset** | Polymarket CSVs (100K markets, datos tabulares) | MIT-BIH PhysioNet (73K latidos, señales eléctricas) |
| **Deep Learning** | Opcional (Autoencoder + Clustering) | **Central (CNN 1D, F1=0.981)** |
| **DL ya implementado** | No | **Si — notebook completo con resultados** |
| **Fine-tuning** | Razonamiento causal "What If" | Explicación clínica de clasificaciones |
| **RAG** | 6-7 docs (estadísticos + regulatorios) | 6 docs (guías médicas + protocolos) |
| **Tipo de razonamiento** | Causal / hipotético | Interpretativo / explicativo |
| **Notebooks** | 4 principales + 1 opcional (DL) | 5 (DL integrado como paso obligatorio) |
| **Evaluación DL** | Silhouette, Davies-Bouldin (si se hace) | F1=0.981, Accuracy=99% (ya hecho) |
| **Evaluación LLM** | Plausibilidad, anclaje, coherencia causal | Precisión clínica, claridad, anclaje |
| **Impacto social** | Medio (finanzas/predicción) | Alto (salud pública) |
| **Wow factor presentación** | Alto (escenarios hipotéticos son llamativos) | Alto (DL médico + explicaciones) |

---

## Análisis por Criterio del Bootcamp

### 1. Deep Learning

| Criterio | Polymarket | ECG | Ventaja |
|----------|------------|-----|---------|
| ¿Tiene DL? | Opcional (bonus) | **Central (pilar del pipeline)** | **ECG** |
| ¿Está implementado? | No | **Si, F1=0.981** | **ECG** |
| Tipo de DL | Autoencoder + Clustering (no supervisado) | CNN 1D clasificadora (supervisado) | **ECG** |
| Complejidad DL | Media (fusión embeddings + numérico) | Media (Conv1d + BatchNorm + FC) | Empate |
| DL aplicado a problema real | Enriquece escenarios (mejora) | **Clasifica arritmias (fundamental)** | **ECG** |

**Veredicto**: ECG gana claramente. El bootcamp valora DL, y en ECG no es un bonus sino el pilar central. Además ya está implementado y tiene resultados demostrables (F1=0.981).

### 2. Dataset

| Criterio | Polymarket | ECG | Ventaja |
|----------|------------|-----|---------|
| Tamaño | 100K mercados, 106 columnas | 73K latidos, 288 muestras | Polymarket (más grande) |
| Tipo de dato | Tabular (CSV) | Serie temporal (señal eléctrica) | ECG (más sofisticado) |
| Fuente | Polymarket public data | PhysioNet (estándar de investigación) | ECG (más prestigioso) |
| Preprocesamiento | Feature engineering + traducción | Extracción de latidos + normalización Z-score | Empate |
| Generación Q&A | Templates con datos calculados | Templates con resultados del CNN | Empate |

**Veredicto**: Empate. Polymarket tiene más volumen; ECG tiene datos más sofisticados (series temporales vs tabular) y de una fuente de investigación reconocida.

### 3. Fine-tuning (LLM)

| Criterio | Polymarket | ECG | Ventaja |
|----------|------------|-----|---------|
| Modelo base | Gemma 3 4B + QLoRA | Gemma 3 4B + QLoRA | Empate |
| Tipo de tarea | Razonamiento causal hipotético | Explicación de clasificaciones DL | Empate |
| Dificultad de la tarea | Alta (razonar causalmente) | Media (explicar resultados) | Polymarket |
| Verificabilidad | Media (escenarios son hipotéticos) | Alta (anclado en output de CNN) | **ECG** |
| Riesgo de alucinación | Alto (dominio especulativo) | Medio (anclado en datos concretos) | **ECG** |

**Veredicto**: Empate con matices. Polymarket es más ambicioso en razonamiento; ECG es más verificable y seguro clínicamente.

### 4. RAG

| Criterio | Polymarket | ECG | Ventaja |
|----------|------------|-----|---------|
| Stack | langchain + FAISS + MiniLM | langchain + FAISS + MiniLM | Empate |
| Documentos | 6-7 (estadísticos + regulatorios) | 6 (guías médicas + protocolos) | Empate |
| Calidad de documentos | Mixta (datos reales + ficticios) | Basada en guías AHA/ESC | **ECG** |
| Utilidad del RAG | Ancla datos reales | Aporta contexto clínico | Empate |

**Veredicto**: Prácticamente empate. La implementación es idéntica. ECG tiene ligera ventaja por la calidad de las fuentes médicas.

### 5. Evaluación

| Criterio | Polymarket | ECG | Ventaja |
|----------|------------|-----|---------|
| Métricas DL | Solo si se hace el bonus | **F1, Accuracy, Confusion Matrix (ya hechas)** | **ECG** |
| Métricas LLM | Rubrics (plausibilidad, anclaje, coherencia) | Rubrics (precisión clínica, claridad, anclaje) | Empate |
| Test cases | 3 niveles de dificultad | 3 niveles de dificultad | Empate |
| Comparación base vs FT | Si | Si | Empate |

**Veredicto**: ECG tiene ventaja por tener métricas DL ya demostradas.

---

## Análisis Cualitativo

### Fortalezas de Polymarket

1. **Originalidad del dominio**: Nadie más va a presentar un proyecto sobre mercados predictivos. Es un nicho único que destaca.
2. **Razonamiento causal**: La tarea de "What If" es intelectualmente más ambiciosa que explicar clasificaciones.
3. **Volumen de datos**: 100K mercados, $8B en volumen — los números impresionan.
4. **Potencial narrativo**: Los escenarios hipotéticos son inherentemente interesantes para una presentación.
5. **Dataset público masivo**: El feature engineering sobre datos reales de Polymarket es valioso.

### Debilidades de Polymarket

1. **DL es opcional**: Si no da el tiempo, el proyecto no tiene Deep Learning. El bootcamp puede no exigirlo, pero tenerlo siempre suma.
2. **Respuestas no verificables**: Los escenarios "What If" son hipotéticos por naturaleza — no hay ground truth para validar si "la respuesta correcta es X".
3. **Complejidad vs resultado**: Mucho esfuerzo en feature engineering y generación de escenarios, pero el output final es texto especulativo.
4. **Riesgo de alucinación alto**: El dominio especulativo invita a que el LLM invente datos.

### Fortalezas de ECG

1. **DL real y central**: La CNN es el pilar del proyecto, no un bonus. F1=0.981 ya demostrado.
2. **Pipeline DL → LLM → RAG integrado**: El output de un modelo alimenta al siguiente — esto es un pipeline de AI real.
3. **Verificabilidad**: Cada respuesta se ancla en el output concreto del clasificador (Normal/Anormal + confianza).
4. **Impacto social**: Salud es un dominio que impresiona a evaluadores y tiene relevancia real.
5. **Ya tiene trabajo avanzado**: El notebook de DL está completo con resultados.
6. **Fuente prestigiosa**: MIT-BIH es el dataset de referencia en investigación de arritmias.

### Debilidades de ECG

1. **Dominio sensible**: Salud implica responsabilidad extra. Los disclaimers son necesarios pero pueden complicar la evaluación.
2. **Binario vs multiclase**: La clasificación es Normal vs Anormal — no distingue tipos específicos de arritmia (aunque esto es una decisión válida de simplificación).
3. **Explicaciones menos "espectaculares"**: "Este latido es anormal" es menos llamativo que "Si crypto cae 50%..."
4. **Menos narrativa para la presentación**: Los escenarios "What If" de Polymarket son inherentemente más entretenidos de presentar.

---

## Opinión y Recomendación

### Mi recomendación: **ECGAssistant**

Por las siguientes razones, ordenadas de mayor a menor peso:

**1. Deep Learning es el diferenciador decisivo.**
El bootcamp evalúa un pipeline de AI. ECG tiene un modelo de Deep Learning REAL que es el pilar central del proyecto, ya implementado, con F1=0.981 demostrado. Polymarket tiene DL como bonus opcional. Si el evaluador valora DL (y la mayoría lo hace), ECG gana automáticamente.

**2. El pipeline DL → LLM → RAG es más coherente y técnicamente impresionante.**
En ECG, cada componente depende del anterior: sin la CNN no hay clasificación, sin clasificación no hay qué explicar, sin RAG no hay contexto clínico. En Polymarket, el pipeline es LLM → RAG (lineal, sin DL obligatorio). La integración de modelos es lo que distingue un proyecto avanzado de uno estándar.

**3. El trabajo ya está avanzado.**
El notebook de DL está completo. No es una promesa — son resultados. Esto reduce el riesgo de no terminar a tiempo y da una base sólida sobre la que construir.

**4. Verificabilidad > Especulación.**
Las respuestas de ECG se anclan en datos concretos (output del clasificador). Las de Polymarket son hipotéticas por naturaleza. Para un evaluador técnico, poder verificar que el sistema no alucina es un punto fuerte.

**5. Impacto social.**
Salud pública > mercados predictivos para una audiencia de bootcamp. No es lo más importante técnicamente, pero en la presentación final cuenta.

### ¿Cuándo elegiría Polymarket?

Elegiría Polymarket si:
- El bootcamp explícitamente NO valora DL (solo fine-tuning + RAG)
- La originalidad del dominio es el factor #1 de evaluación
- La presentación es más importante que la profundidad técnica
- Hay mucho tiempo disponible y se puede completar el notebook 5 de DL

### Escenario ideal

Si fuera posible, el proyecto más fuerte sería **ECGAssistant con la profundidad de documentación de Polymarket**. El .md de Polymarket tiene ejemplos de interacción más desarrollados y escenarios más variados — esa calidad de documentación aplicada al proyecto ECG lo haría imbatible.

---

## Tabla Resumen de Scoring

| Criterio (peso) | Polymarket | ECG |
|------------------|-----------|-----|
| Deep Learning (30%) | 3/10 (opcional, no hecho) | **10/10** (central, hecho, F1=0.981) |
| Pipeline integrado (20%) | 6/10 (LLM→RAG, lineal) | **9/10** (DL→LLM→RAG, cascada) |
| Dataset (15%) | 8/10 (masivo, real) | 8/10 (real, prestigioso) |
| Fine-tuning (15%) | 8/10 (ambicioso, causal) | 7/10 (sólido, verificable) |
| RAG (10%) | 7/10 | 7/10 |
| Presentación/Wow (10%) | 8/10 (escenarios llamativos) | 7/10 (dominio impactante) |
| **Total ponderado** | **5.95/10** | **8.45/10** |

**ECGAssistant gana por el peso decisivo del Deep Learning real e integrado.**
