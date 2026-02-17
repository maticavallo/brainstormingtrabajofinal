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

**1. Fine-tuning + RAG anclado en datos concretos > especulación.**
El fine-tuning de ECG tiene una ventaja estructural: cada respuesta del LLM se ancla en el output concreto del clasificador CNN (Normal/Anormal + confianza). Esto hace que las alucinaciones sean detectables y medibles. En Polymarket, los escenarios "What If" son hipotéticos por naturaleza — no hay ground truth para validar si la respuesta es correcta. Desde una perspectiva de AI engineering, un LLM que explica datos verificables es más robusto que uno que especula.

**2. El pipeline DL → LLM → RAG demuestra integración de modelos.**
En ECG, cada componente depende del anterior: sin la CNN no hay clasificación, sin clasificación no hay qué explicar, sin RAG no hay contexto clínico. Es un pipeline de AI real donde modelos distintos se encadenan. En Polymarket, el pipeline es LLM → RAG (lineal, sin integración multi-modelo). La orquestación de modelos heterogéneos (DL + LLM + retrieval) es lo que la industria demanda hoy.

**3. Deep Learning ya implementado reduce riesgo.**
El notebook de DL está completo con F1=0.981 demostrado. No es una promesa — son resultados. Esto da una base sólida y reduce el riesgo de no terminar a tiempo. Aunque un CNN para ECG es un problema académicamente resuelto, tenerlo funcionando es mejor que no tener DL.

**4. RAG con documentos clínicos tiene decisiones no triviales.**
Ambos proyectos usan el mismo stack de RAG (FAISS + langchain), pero el dominio médico impone restricciones más exigentes: un chunk mal recuperado podría generar una explicación clínica incorrecta. Esto obliga a diseñar el retrieval con más cuidado y demuestra madurez en la implementación.

**5. Impacto social.**
Salud pública > mercados predictivos para una audiencia de bootcamp. No es lo más importante técnicamente, pero en la presentación final cuenta.

### Matiz importante: qué vale más en la industria vs en el bootcamp

Desde **AI engineering**, las skills más valiosas del pipeline son fine-tuning y RAG — no el CNN. Un CNN 1D para clasificación binaria es un problema resuelto desde ~2017. El verdadero desafío está en: cómo diseñar el dataset de fine-tuning para que el LLM no alucine información médica, cómo evaluar la calidad de las explicaciones generadas, y cómo hacer que el RAG recupere contexto relevante sin ruido.

Sin embargo, para el **bootcamp**, tener DL implementado y funcionando sigue siendo un diferenciador — demuestra que se domina el stack completo. La clave es no confundir "el CNN es impresionante" con "el CNN es lo más difícil del proyecto".

### ¿Cuándo elegiría Polymarket?

Elegiría Polymarket si:
- El bootcamp explícitamente NO valora DL (solo fine-tuning + RAG)
- La originalidad del dominio es el factor #1 de evaluación
- La presentación es más importante que la profundidad técnica
- Hay mucho tiempo disponible y se puede completar el notebook 5 de DL
- Se quiere demostrar razonamiento causal con LLMs (tarea más ambiciosa que explicar clasificaciones)

### Escenario ideal

Si fuera posible, el proyecto más fuerte sería **ECGAssistant con la ambición de razonamiento de Polymarket**. El fine-tuning de Polymarket es más ambicioso (razonamiento causal) — esa exigencia aplicada al dominio médico de ECG lo haría imbatible.

---

## Tabla Resumen de Scoring

Los pesos reflejan la realidad de AI engineering moderna, donde fine-tuning y RAG son las skills más demandadas y complejas, por encima de DL clásico.

| Criterio (peso) | Polymarket | ECG |
|------------------|-----------|-----|
| Fine-tuning LLM (25%) | 8/10 (ambicioso, razonamiento causal) | 7/10 (sólido, verificable, anclado en CNN) |
| RAG (20%) | 7/10 (docs mixtos: datos reales + ficticios) | 8/10 (docs clínicos, fuentes AHA/ESC) |
| Deep Learning (20%) | 3/10 (opcional, no implementado) | **9/10** (central, implementado, F1=0.981) |
| Pipeline integrado (15%) | 6/10 (LLM→RAG, lineal) | **9/10** (DL→LLM→RAG, cascada multi-modelo) |
| Dataset (10%) | 8/10 (masivo, 100K markets) | 8/10 (prestigioso, PhysioNet, series temporales) |
| Presentación/Wow (10%) | 8/10 (escenarios llamativos) | 7/10 (dominio impactante) |
| **Total ponderado** | **6.35/10** | **7.90/10** |

**ECGAssistant gana por la combinación de pipeline integrado multi-modelo + verificabilidad del fine-tuning + DL real implementado.** La ventaja no es solo "tiene DL" sino que el DL alimenta al LLM, que se enriquece con RAG — cada pieza tiene un rol claro en el pipeline.
