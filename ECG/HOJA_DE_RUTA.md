# Hoja de Ruta — ECGAssistant

> **Equipo**: 4 personas (Roger, Matias, Matteo, Raul)
> **Periodo**: 17 febrero - 12 marzo 2026 (23 dias, 3 sprints)
> **Metodologia**: SCRUM (sprints de ~8 dias, dailies, Trello)
> **Repo**: GitHub — rama `main`

---

## 1. Estado Actual y Entregables

### Lo que ya existe

| Componente | Estado | Detalle |
|---|---|---|
| `ECG.ipynb` | Completo | CNN 1D funcionando: descarga MIT-BIH, preprocesamiento, entrenamiento, evaluacion |
| `IDEA_ECG_ASSISTANT.md` | Completo | Spec tecnica completa del proyecto (pipeline, arquitectura, stack) |
| Modelo CNN | Entrenado | F1=0.981, Accuracy=99%, 35 epochs, 73,737 latidos |
| Dataset MIT-BIH | Descargado | 34 registros, 73K latidos, binarizado Normal/Anormal |

### Lo que falta

| Componente | Prioridad | Complejidad |
|---|---|---|
| Separar `ECG.ipynb` en Notebook 1 (Dataset) y Notebook 2 (DL) | Alta | Baja |
| Generar dataset Q&A para fine-tuning | Alta | Media |
| Notebook 3: Fine-tuning (Gemma 3 + QLoRA) | Alta | Alta |
| Escribir 6 documentos clinicos para RAG | Alta | Media |
| Notebook 4: RAG (FAISS + langchain) | Alta | Media |
| Notebook 5: Evaluacion completa (DL + LLM) | Alta | Media |
| Presentacion (15 slides) | Alta | Baja |
| Post LinkedIn | Baja | Baja |
| Subir dataset y modelo a HuggingFace Hub | Media | Baja |

### Entregables finales

- [ ] 5 notebooks en GitHub: `ECG_Dataset`, `ECG_DeepLearning`, `ECG_FineTuning`, `ECG_RAG`, `ECG_Evaluation`
- [ ] Dataset Q&A en HuggingFace Hub
- [ ] Modelo fine-tuned en HuggingFace Hub
- [ ] Modelo CNN guardado (`.pth`)
- [ ] 6 documentos clinicos en `/docs_clinicos/`
- [ ] Presentacion (15 slides, max 15 min)
- [ ] Post LinkedIn
- [ ] Tablero Trello con historias de usuario y burndown chart

---

## 2. Historias de Usuario para Trello

### Formato: "Como [rol], quiero [accion] para [beneficio]"

| ID | Historia de Usuario | Story Points | Sprint | Prioridad |
|---|---|---|---|---|
| US-01 | Como equipo, quiero separar el notebook monolitico en Notebook 1 (Dataset) y Notebook 2 (DL) para tener un pipeline modular y presentable | 3 | 1 | Alta |
| US-02 | Como data scientist, quiero preprocesar y documentar el dataset MIT-BIH con EDA completo para demostrar dominio de los datos | 3 | 1 | Alta |
| US-03 | Como ML engineer, quiero documentar la arquitectura CNN y sus resultados (F1=0.981) para que el notebook DL sea autocontenido | 3 | 1 | Alta |
| US-04 | Como NLP engineer, quiero generar un dataset Q&A de al menos 200 pares (resultado CNN + explicacion clinica) para entrenar el LLM | 5 | 1 | Alta |
| US-05 | Como equipo, quiero redactar 6 documentos clinicos en espanol para alimentar el sistema RAG | 5 | 1 | Alta |
| US-06 | Como ML engineer, quiero hacer fine-tuning de Gemma 3 4B con QLoRA usando el dataset Q&A para que el LLM explique arritmias | 8 | 2 | Alta |
| US-07 | Como ML engineer, quiero implementar el pipeline RAG con FAISS y langchain para aportar contexto de guias medicas | 5 | 2 | Alta |
| US-08 | Como developer, quiero integrar CNN + LLM fine-tuned + RAG en un pipeline end-to-end funcional | 5 | 2 | Alta |
| US-09 | Como developer, quiero subir el dataset Q&A y el modelo fine-tuned a HuggingFace Hub para que sean accesibles | 2 | 2 | Media |
| US-10 | Como QA, quiero evaluar el modelo DL con metricas cuantitativas (F1, confusion matrix, curvas) para el notebook de evaluacion | 3 | 3 | Alta |
| US-11 | Como QA, quiero evaluar el LLM fine-tuned con rubrics de precision clinica, claridad y anclaje para comparar base vs fine-tuned | 5 | 3 | Alta |
| US-12 | Como equipo, quiero compilar el notebook de Evaluacion con resultados DL + LLM + RAG para tener el entregable completo | 3 | 3 | Alta |
| US-13 | Como equipo, quiero crear la presentacion de 15 slides con timing y speaker asignados para la defensa del proyecto | 3 | 3 | Alta |
| US-14 | Como equipo, quiero preparar un banco de 15 preguntas probables con estrategias de respuesta para el Q&A post-presentacion | 2 | 3 | Alta |
| US-15 | Como equipo, quiero publicar un post en LinkedIn con resultados y aprendizajes del proyecto | 2 | 3 | Baja |

**Total story points: 57**

### Distribucion por sprint

| Sprint | Story Points | % del total |
|---|---|---|
| Sprint 1 | 19 | 33% |
| Sprint 2 | 20 | 35% |
| Sprint 3 | 18 | 32% |

---

## 3. Sprints con Desglose Diario

### Sprint 1: Fundamentos (Feb 17 - Feb 24) — 19 SP

**Objetivo**: Tener los notebooks 1 y 2 listos, el dataset Q&A generado, y los documentos clinicos escritos.

| Dia | Fecha | Roger | Matias | Matteo | Raul |
|---|---|---|---|---|---|
| 1 | Lun 17 | Setup Trello + historias de usuario | Separar ECG.ipynb: extraer celdas de Dataset | Investigar templates Q&A para fine-tuning | Investigar fuentes para docs clinicos |
| 2 | Mar 18 | Revisar separacion + crear Notebook 1 (Dataset) con EDA | Crear Notebook 2 (DL) con arquitectura + entrenamiento | Disenar 5 tipos de pares Q&A (templates) | Redactar `guia_arritmias.txt` |
| 3 | Mie 19 | Documentar EDA en Notebook 1 (graficos, distribuciones) | Documentar resultados DL en Notebook 2 (metricas, confusion matrix) | Implementar generador de pares Q&A tipo 1 y 2 | Redactar `protocolo_ecg.txt` |
| 4 | Jue 20 | Agregar visualizaciones de senales ECG al Notebook 1 | Agregar curvas de entrenamiento + guardar modelo `.pth` | Implementar generador de pares Q&A tipo 3 y 4 | Redactar `explicaciones_pacientes.txt` |
| 5 | Vie 21 | QA Notebook 1: revisar que corre end-to-end en Colab | QA Notebook 2: revisar que corre end-to-end en Colab | Implementar generador pares Q&A tipo 5 + validacion | Redactar `farmacologia_basica.txt` |
| 6 | Sab 22 | Revisar calidad del dataset Q&A generado | Investigar Unsloth + QLoRA para Sprint 2 | Generar dataset completo (~200+ pares Q&A) | Redactar `derivacion_urgencias.txt` |
| 7 | Dom 23 | Buffer / deuda tecnica | Buffer / deuda tecnica | Buffer / deuda tecnica | Redactar `glosario_medico.txt` |
| 8 | Lun 24 | **Sprint Review 1** + Retrospectiva | **Sprint Review 1** | Subir dataset Q&A a HuggingFace | QA documentos clinicos: revision cruzada |

**Definition of Done Sprint 1** — ver Seccion 8.

---

### Sprint 2: Modelos LLM + Integracion (Feb 25 - Mar 5) — 20 SP

**Objetivo**: Fine-tuning del LLM, RAG funcionando, pipeline end-to-end integrado.

| Dia | Fecha | Roger | Matias | Matteo | Raul |
|---|---|---|---|---|---|
| 9 | Mar 25 | Sprint Planning 2 + setup Notebook 3 (FineTuning) | Configurar entorno Colab para Unsloth | Disenar chunking strategy para docs clinicos | Investigar FAISS + sentence-transformers |
| 10 | Mie 26 | Implementar carga de modelo Gemma 3 + QLoRA config | Preparar dataset Q&A en formato SFTTrainer | Implementar carga de documentos con langchain | Setup embeddings all-MiniLM-L6-v2 |
| 11 | Jue 27 | Ejecutar fine-tuning (3 epochs) en Colab | Monitorear training + ajustar hiperparametros si necesario | Implementar FAISS vectorstore + retriever | Testear retrieval con queries de ejemplo |
| 12 | Vie 28 | Evaluar modelo fine-tuned con prompts de prueba | Guardar modelo + documentar Notebook 3 | Completar Notebook 4 (RAG) con pipeline retrieval | Testear RAG con diferentes chunk_size y k |
| 13 | Sab 1 | Integrar CNN + LLM: funcion `ecg_assistant()` | Testear pipeline con senales de test reales | QA Notebook 3: revisar que corre en Colab | QA Notebook 4: revisar que corre en Colab |
| 14 | Dom 2 | Ajustar system prompt + mejorar calidad de respuestas | Buffer / debugging integracion | Buffer / debugging | Buffer / debugging |
| 15 | Lun 3 | Subir modelo fine-tuned a HuggingFace Hub | Probar pipeline completo end-to-end | Documentar pipeline RAG en Notebook 4 | Preparar test cases para evaluacion |
| 16 | Mar 4 | Buffer / deuda tecnica | Buffer / deuda tecnica | Buffer / deuda tecnica | Buffer / deuda tecnica |
| 17 | Mie 5 | **Sprint Review 2** + Retrospectiva | **Sprint Review 2** | **Sprint Review 2** | **Sprint Review 2** |

**Definition of Done Sprint 2** — ver Seccion 8.

---

### Sprint 3: Evaluacion + Presentacion (Mar 6 - Mar 12) — 18 SP

**Objetivo**: Notebook de evaluacion completo, presentacion lista, ensayo general.

| Dia | Fecha | Roger | Matias | Matteo | Raul |
|---|---|---|---|---|---|
| 18 | Jue 6 | Sprint Planning 3 + setup Notebook 5 (Evaluation) | Ejecutar evaluacion DL: F1, precision, recall, confusion matrix | Ejecutar evaluacion LLM: base vs fine-tuned con test cases | Preparar esqueleto de presentacion (15 slides) |
| 19 | Vie 7 | Compilar resultados DL en Notebook 5 | Compilar resultados LLM en Notebook 5 | Ejecutar evaluacion RAG: relevancia de retrieval | Slides 1-5: Problema + Dataset + Mercado |
| 20 | Sab 8 | Ejecutar evaluacion pipeline completo (end-to-end) | Generar graficos comparativos (base vs FT) | Documentar limitaciones + analisis de errores | Slides 6-10: DL + FT + RAG |
| 21 | Dom 9 | QA Notebook 5: revisar que corre en Colab | Revisar todos los notebooks (QA final) | Preparar banco de preguntas Q&A (15 preguntas) | Slides 11-15: Evaluacion + Conclusiones + Demo |
| 22 | Lun 10 | Ensayo presentacion (dry run 1) | Ensayo presentacion (dry run 1) | Ensayo presentacion (dry run 1) | Ensayo presentacion (dry run 1) |
| 23 | Mar 11 | Ajustes finales post-ensayo | Redactar post LinkedIn | Ensayo final (dry run 2) con cronometro | Ajustes finales slides + speaker notes |
| — | Mie 12 | **PRESENTACION FINAL** | **PRESENTACION FINAL** | **PRESENTACION FINAL** | **PRESENTACION FINAL** |

**Definition of Done Sprint 3** — ver Seccion 8.

---

## 4. Rotacion de Roles

El bootcamp exige rotacion de roles SCRUM entre sprints. Cada persona asume un rol diferente cada sprint.

| Rol | Sprint 1 (Feb 17-24) | Sprint 2 (Feb 25 - Mar 5) | Sprint 3 (Mar 6-12) |
|---|---|---|---|
| **Scrum Master (SM)** | Roger | Matteo | Matias |
| **Product Owner (PO)** | Matias | Raul | Roger |
| **Tech Lead (TL)** | Matteo | Roger | Raul |
| **QA Lead** | Raul | Matias | Matteo |

### Responsabilidades por rol

| Rol | Responsabilidad |
|---|---|
| **SM** | Facilita dailies (15 min), actualiza Trello y burndown chart, remueve blockers, organiza sprint review/retro |
| **PO** | Prioriza backlog, acepta/rechaza entregables en sprint review, define criterios de aceptacion |
| **TL** | Revisa codigo/notebooks, decide trade-offs tecnicos, lidera pair programming si hay bloqueos |
| **QA** | Ejecuta checklists de Definition of Done, verifica que notebooks corran en Colab, revisa calidad de outputs |

---

## 5. Estructura de Presentacion

**Duracion total**: 14 minutos de presentacion + 1 minuto de margen (total ≤ 15 min).

| # | Slide | Contenido | Tiempo | Speaker |
|---|---|---|---|---|
| 1 | Portada | Titulo, nombres del equipo, bootcamp KeepCoding | 0:20 | Roger |
| 2 | El Problema | 300M ECGs/ano, gap deteccion-explicacion, LATAM sin cardiologos | 1:00 | Roger |
| 3 | Competidores | Tabla KardiaMobile/Apple Watch/Cardiologs: clasifican pero no explican | 0:50 | Roger |
| 4 | Nuestra Solucion | Pipeline DL → FT → RAG, diagrama de arquitectura | 1:00 | Matias |
| 5 | Dataset MIT-BIH | 73K latidos, distribucion de clases, splits, grafico de senal ECG | 1:00 | Matias |
| 6 | Deep Learning — CNN | Arquitectura Conv1d, decisiones de diseno, entrenamiento 35 epochs | 1:00 | Matteo |
| 7 | Resultados DL | F1=0.981, confusion matrix, curvas de training | 1:00 | Matteo |
| 8 | Fine-Tuning — Gemma 3 | QLoRA config, dataset Q&A, system prompt, training | 1:00 | Matteo |
| 9 | RAG — Documentos Clinicos | 6 docs, FAISS, chunking, retrieval, pipeline completo | 1:00 | Raul |
| 10 | Demo en Vivo | Input ECG real → CNN clasifica → LLM explica → RAG contextualiza | 1:30 | Raul |
| 11 | Evaluacion DL | Metricas cuantitativas: F1, precision, recall, AUC | 0:50 | Matias |
| 12 | Evaluacion LLM | Comparacion base vs FT: precision clinica, claridad, anclaje | 1:00 | Matias |
| 13 | Viabilidad Economica | Costos desarrollo vs produccion, value proposition, mercado | 0:50 | Roger |
| 14 | Conclusiones | Aprendizajes, limitaciones honestas, trabajo futuro | 0:50 | Roger |
| 15 | Preguntas | Slide de cierre + Q&A | 0:40 | Todos |
| | | **TOTAL** | **13:50** | |

**Margen disponible**: 1 minuto 10 segundos de buffer.

### Tips para la presentacion

- Cada speaker practica SU bloque al menos 3 veces con cronometro
- La demo (slide 10) debe tener un backup grabado en video por si falla en vivo
- Slides visuales: graficos > texto, maximo 5 bullet points por slide
- El Q&A lo responde quien mejor domine el tema de la pregunta

---

## 6. Banco de Preguntas Q&A

15 preguntas probables de evaluadores con estrategia de respuesta.

| # | Pregunta | Quien responde | Estrategia |
|---|---|---|---|
| 1 | "Por que eligieron ECG y no otro dominio?" | Roger (PO) | Mercado $1.34B, gap explicacion, datos reales de PhysioNet, impacto social en LATAM |
| 2 | "Por que CNN 1D y no un Transformer o RNN?" | Matteo (TL) | ECG es serie temporal 1D, CNN captura patrones morfologicos locales (QRS), menor complejidad computacional, F1=0.981 valida la eleccion |
| 3 | "Que pasa con el 1.1% de error del modelo?" | Matteo | 88 falsos positivos + 73 falsos negativos sobre 14K. En produccion: siempre derivar a profesional, el modelo es apoyo no diagnostico |
| 4 | "Por que Gemma 3 y no GPT/LLaMA/Qwen?" | Matteo | Open source, 4B parametros (corre en Colab Free con QLoRA), formato instruccion, buen rendimiento en espanol |
| 5 | "Como evitan alucinaciones medicas?" | Matias | System prompt con disclaimers, anclaje obligatorio en output CNN, RAG con docs verificados, evaluacion de precision clinica |
| 6 | "Que limitaciones tiene el sistema?" | Roger | Solo 1 derivacion (MLII), clasificacion binaria (Normal/Anormal), no detecta infartos/isquemia, no reemplaza cardiologo |
| 7 | "Como midieron la calidad del fine-tuning?" | Matias (QA) | Comparacion base vs FT con mismos prompts, rubrics de precision clinica (1-5), claridad (1-5), anclaje en datos CNN |
| 8 | "El RAG realmente mejora las respuestas?" | Raul | Comparar respuesta sin RAG vs con RAG: la version RAG cita guias, protocolos y umbrales clinicos concretos |
| 9 | "Podrian escalar esto a produccion?" | Roger | Si: API con FastAPI, CNN en ONNX, LLM via Groq/Together, RAG en Pinecone. Costo estimado: $200-500/mes |
| 10 | "Como manejaron el desbalance de clases?" | Matteo | Weighted CrossEntropyLoss con pesos inversamente proporcionales al conteo de cada clase. Split estratificado |
| 11 | "Que metrica priorizaron y por que?" | Matias | F1 Score: balanza precision y recall. En contexto medico, tanto falsos positivos (alarma innecesaria) como falsos negativos (arritmia no detectada) son criticos |
| 12 | "Por que FAISS y no ChromaDB u otro?" | Raul | FAISS es el estandar de facto para vectorstore, funciona con faiss-gpu en Colab, altamente eficiente para nuestro volumen de documentos |
| 13 | "Como organizaron el trabajo en equipo?" | SM del sprint actual | SCRUM con sprints de 8 dias, Trello con historias de usuario, dailies de 15 min, rotacion de roles, burndown chart |
| 14 | "Cual fue el mayor desafio tecnico?" | TL del sprint actual | Fine-tuning en Colab Free con memoria limitada: resuelto con QLoRA 4-bit + Unsloth. Segundo: calidad del dataset Q&A |
| 15 | "Que harian diferente si empezaran de cero?" | Todos | Empezar con multi-clase (no binario), usar ECG de 12 derivaciones, considerar modelos mas grandes si hay presupuesto GPU |

---

## 7. Matriz de Riesgos

| # | Riesgo | Prob. | Impacto | Mitigacion | Contingencia |
|---|---|---|---|---|---|
| 1 | Colab Free sin GPU disponible durante fine-tuning | Media | Alto | Ejecutar en horarios de baja demanda (noche/madrugada). QLoRA 4-bit minimiza uso de VRAM | Usar Kaggle Notebooks (30h GPU/semana) como backup |
| 2 | Dataset Q&A de baja calidad | Media | Alto | Templates estructurados con datos reales del CNN. Revision cruzada entre Matteo y Raul | Reducir a 150 pares de alta calidad en vez de 200 mediocres |
| 3 | Fine-tuning genera alucinaciones medicas | Media | Alto | System prompt con restricciones estrictas. Evaluacion de anclaje en datos CNN | Agregar post-procesamiento que filtre respuestas sin disclaimer |
| 4 | RAG recupera contexto irrelevante | Baja | Medio | Documentos acotados y especificos. Ajustar `chunk_size` (500) y `k` (3) | Reducir k a 2, aumentar chunk_size a 800 |
| 5 | No llegar a completar los 5 notebooks | Baja | Alto | Sprint planning con buffer de 1-2 dias por sprint. Priorizar notebooks 1-4 | Fusionar Notebook 5 (Eval) como seccion final de Notebook 4 |
| 6 | Demo falla durante la presentacion | Media | Alto | Ensayo con la demo 3+ veces. Preparar video backup grabado | Mostrar el video backup. Tener screenshots de outputs como ultimo recurso |
| 7 | Un miembro del equipo se ausenta | Baja | Medio | Documentacion clara en cada notebook. Pair programming en tareas criticas | Redistribuir tareas entre los 3 restantes. Reducir scope de US-15 |
| 8 | Conflictos de merge en GitHub | Baja | Bajo | Cada persona trabaja en un notebook diferente. Feature branches por tarea | Resolver conflictos en pareja (TL + quien tiene el conflicto) |
| 9 | Modelo CNN no reproduce resultados al separar notebook | Baja | Alto | Guardar modelo `.pth` + semillas (`random_state=42`). Verificar reproducibilidad antes de avanzar | Usar el modelo ya entrenado del notebook original |
| 10 | Exceder los 15 minutos de presentacion | Media | Medio | Ensayar con cronometro 3+ veces. Timing por slide estricto | El SM corta al speaker si supera su tiempo. Slides de backup no se muestran |

---

## 8. Definition of Done por Sprint

### Sprint 1 — Fundamentos

- [ ] Notebook 1 (`ECG_Dataset`) corre end-to-end en Colab sin errores
- [ ] Notebook 1 incluye: descarga MIT-BIH, extraccion de latidos, binarizacion, EDA con graficos, normalizacion, splits
- [ ] Notebook 2 (`ECG_DeepLearning`) corre end-to-end en Colab sin errores
- [ ] Notebook 2 incluye: arquitectura CNN, entrenamiento, metricas (F1, confusion matrix), modelo guardado como `.pth`
- [ ] Dataset Q&A generado con al menos 200 pares en formato `messages` (system/user/assistant)
- [ ] 6 documentos clinicos escritos en `/docs_clinicos/` y revisados por al menos 2 personas
- [ ] Trello actualizado con todas las historias y el burndown chart del Sprint 1
- [ ] Sprint Review 1 completada con demo de Notebooks 1 y 2

### Sprint 2 — Modelos LLM + Integracion

- [ ] Notebook 3 (`ECG_FineTuning`) corre end-to-end en Colab sin errores
- [ ] Fine-tuning ejecutado con al menos 3 epochs, loss decreciente documentada
- [ ] Modelo fine-tuned genera respuestas coherentes en espanol con anclaje en datos CNN
- [ ] Modelo y dataset subidos a HuggingFace Hub
- [ ] Notebook 4 (`ECG_RAG`) corre end-to-end en Colab sin errores
- [ ] RAG recupera contexto relevante para al menos 5 queries de prueba
- [ ] Pipeline `ecg_assistant()` integra CNN + LLM + RAG y produce respuestas completas
- [ ] Trello actualizado + burndown chart Sprint 2
- [ ] Sprint Review 2 completada con demo del pipeline end-to-end

### Sprint 3 — Evaluacion + Presentacion

- [ ] Notebook 5 (`ECG_Evaluation`) corre end-to-end en Colab sin errores
- [ ] Evaluacion DL documentada: F1, precision, recall, confusion matrix, curvas
- [ ] Evaluacion LLM documentada: comparacion base vs FT con al menos 10 test cases y rubrics
- [ ] Presentacion lista con 15 slides, speaker y timing asignado por slide
- [ ] Al menos 2 ensayos completos con cronometro realizados (total ≤ 15 min)
- [ ] Banco de 15 preguntas Q&A preparado con estrategias de respuesta
- [ ] Post LinkedIn redactado (publicar despues de la presentacion)
- [ ] Todos los notebooks revisados y verificados en Colab
- [ ] Burndown chart final actualizado en Trello

---

## 9. Burndown Chart

### Story Points por Historia

| Historia | SP | Sprint | Acumulado |
|---|---|---|---|
| US-01: Separar notebooks | 3 | 1 | 3 |
| US-02: Dataset + EDA | 3 | 1 | 6 |
| US-03: Documentar DL | 3 | 1 | 9 |
| US-04: Dataset Q&A | 5 | 1 | 14 |
| US-05: Docs clinicos | 5 | 1 | 19 |
| US-06: Fine-tuning | 8 | 2 | 27 |
| US-07: RAG | 5 | 2 | 32 |
| US-08: Integracion | 5 | 2 | 37 |
| US-09: HuggingFace Hub | 2 | 2 | 39 |
| US-10: Eval DL | 3 | 3 | 42 |
| US-11: Eval LLM | 5 | 3 | 47 |
| US-12: Notebook Eval | 3 | 3 | 50 |
| US-13: Presentacion | 3 | 3 | 53 |
| US-14: Banco Q&A | 2 | 3 | 55 |
| US-15: Post LinkedIn | 2 | 3 | 57 |

### Burn Rate Esperado

```
SP Restantes
57 |*
   |  *
   |    *
   |      *
   |        *
39 |──────────*                          ← Fin Sprint 1 (dia 8)
   |            *
   |              *
   |                *
   |                  *
19 |────────────────────*                ← Fin Sprint 2 (dia 17)
   |                      *
   |                        *
   |                          *
 0 |────────────────────────────*        ← Fin Sprint 3 (dia 23)
   └──────────────────────────────
   D1    D5    D8   D12   D17  D23
```

**Velocidad promedio**: ~19 SP/sprint (57 SP / 3 sprints)

### Como mantener el burndown chart en Trello

1. Crear un tablero con columnas: `Backlog` | `Sprint Backlog` | `In Progress` | `In Review` | `Done`
2. Cada tarjeta = 1 user story con su etiqueta de SP
3. Al final de cada dia laboral, el SM mueve las tarjetas completadas a `Done`
4. Registrar en una hoja de calculo (o Power-Up de Trello) los SP completados acumulados
5. Graficar SP restantes vs dias transcurridos

---

## 10. Viabilidad Economica

### Costos de Desarrollo (Fase Bootcamp)

| Recurso | Costo |
|---|---|
| Google Colab Free (GPU T4) | 0 EUR |
| Dataset MIT-BIH (PhysioNet) | 0 EUR (acceso publico) |
| Modelo base Gemma 3 4B (HuggingFace) | 0 EUR (open source) |
| Unsloth + QLoRA | 0 EUR (open source) |
| FAISS + langchain | 0 EUR (open source) |
| GitHub (repositorio) | 0 EUR |
| HuggingFace Hub (hosting modelo) | 0 EUR (free tier) |
| **Total desarrollo** | **0 EUR** |

### Costos de Produccion Estimados (Hipotetico)

| Recurso | Costo/mes | Nota |
|---|---|---|
| GPU Cloud (RunPod/Lambda) | ~100-200 EUR | Para inferencia LLM |
| API Groq o Together (alternativa serverless) | ~50-100 EUR | Pay-per-token, sin GPU propia |
| Almacenamiento FAISS (vectorstore) | ~10 EUR | Cloud storage para indices |
| Hosting API (FastAPI en Railway/Render) | ~20-50 EUR | Backend para servir el pipeline |
| Dominio + SSL | ~15 EUR/ano | Presencia web |
| **Total produccion (mensual)** | **~200-400 EUR/mes** | |

### Value Proposition

| Aspecto | Valor |
|---|---|
| **Mercado objetivo** | Clinicas de atencion primaria en LATAM, usuarios de wearables ECG |
| **TAM** (Total Addressable Market) | $3.34B para 2029 (monitoreo ECG con IA) |
| **Diferenciador** | Unico sistema que detecta + explica en espanol con contexto de guias medicas |
| **Modelo de negocio potencial** | SaaS B2B para clinicas ($50-200/mes por clinica) o B2C freemium para usuarios de wearables |
| **Ventaja competitiva** | Open source (confianza), espanol nativo, RAG actualizable sin re-entrenar |

### Comparativa con Competidores

| Producto | Costo para el usuario | Explica en espanol | Open source | RAG actualizable |
|---|---|---|---|---|
| AliveCor KardiaMobile | $99 dispositivo + $9.99/mes | No | No | No |
| Apple Watch ECG | $399+ (Watch) | No | No | No |
| Cardiologs (Philips) | Enterprise pricing | No | No | No |
| **ECGAssistant** | **Gratuito (bootcamp) / $50-200/mes B2B** | **Si** | **Si** | **Si** |

---

## 11. Dependencias Criticas

### Diagrama de Dependencias

```
US-01 (Separar notebooks)
  ├──→ US-02 (Dataset + EDA)           ─┐
  └──→ US-03 (Documentar DL)            │
                                         │
US-04 (Dataset Q&A) ────────────────────→ US-06 (Fine-tuning)
                                         │         │
US-05 (Docs clinicos) ──────────────────→ US-07 (RAG)
                                         │         │
                                         └────┬────┘
                                              │
                                              ▼
                                      US-08 (Integracion)
                                              │
                                     ┌────────┼────────┐
                                     │        │        │
                                     ▼        ▼        ▼
                              US-10 (Eval DL) US-11 (Eval LLM)
                                     │        │
                                     └────┬───┘
                                          │
                                          ▼
                                   US-12 (Notebook Eval)
                                          │
                                          ▼
                                   US-13 (Presentacion)
                                          │
                                          ▼
                                   US-14 (Banco Q&A)
```

### Bloqueos Criticos

| Si se atrasa... | Se bloquea... | Impacto |
|---|---|---|
| US-04 (Dataset Q&A) | US-06 (Fine-tuning) → US-08 (Integracion) → Todo Sprint 3 | **Critico**: sin dataset no hay fine-tuning |
| US-05 (Docs clinicos) | US-07 (RAG) → US-08 (Integracion) | **Alto**: sin docs no hay RAG |
| US-06 (Fine-tuning) | US-08, US-11, US-12, US-13 | **Critico**: sin LLM no hay pipeline |
| US-01 (Separar notebooks) | US-02, US-03 | **Medio**: se puede trabajar en paralelo si se coordina |

### Ruta Critica

```
US-04 → US-06 → US-08 → US-11 → US-12 → US-13
```

**Esta es la secuencia que NO puede atrasarse.** Si cualquier tarea en esta ruta se atrasa 1 dia, toda la entrega se atrasa 1 dia.

---

## 12. Recomendaciones Operativas

### Git Workflow

```
main
  └── feature/notebook-1-dataset
  └── feature/notebook-2-dl
  └── feature/notebook-3-finetuning
  └── feature/notebook-4-rag
  └── feature/notebook-5-evaluation
  └── feature/docs-clinicos
  └── feature/presentacion
```

**Reglas**:
- Cada persona trabaja en su propia branch
- PR con al menos 1 review antes de mergear a `main`
- No pushear directamente a `main`
- Commit messages en espanol, formato: `[NB-X] Descripcion breve` (ej: `[NB-2] Agregar curvas de entrenamiento`)

### Dailies (Stand-up)

- **Duracion**: 15 minutos max
- **Horario fijo**: Acordar 1 horario (ej: 10:00 AM)
- **Formato**: Cada persona responde 3 preguntas:
  1. Que hice ayer?
  2. Que voy a hacer hoy?
  3. Tengo algun bloqueo?
- **Herramienta**: WhatsApp/Slack/Discord (lo que ya usen)
- **El SM toma nota** y actualiza Trello si hay cambios

### Naming Conventions

| Tipo | Convencion | Ejemplo |
|---|---|---|
| Notebooks | `ECG_{Fase}.ipynb` | `ECG_Dataset.ipynb` |
| Docs clinicos | `snake_case.txt` | `guia_arritmias.txt` |
| Branches | `feature/{descripcion}` | `feature/notebook-3-finetuning` |
| Commits | `[NB-X] Descripcion` | `[NB-3] Implementar QLoRA config` |
| Carpeta docs | `docs_clinicos/` | — |
| Modelo guardado | `ecg_cnn_model.pth` | — |

### Critical Path Checklist

Revisar cada 2 dias (el SM es responsable):

- [ ] El dataset Q&A (US-04) esta avanzando segun lo planeado?
- [ ] Los docs clinicos (US-05) van a estar listos para cuando empiece RAG?
- [ ] El fine-tuning (US-06) se puede ejecutar en Colab Free o necesitamos backup?
- [ ] La integracion (US-08) tiene todas sus dependencias listas?
- [ ] Quedan al menos 2 dias para ensayar la presentacion?

### Herramientas del Equipo

| Proposito | Herramienta |
|---|---|
| Tablero SCRUM | Trello (obligatorio por bootcamp) |
| Repositorio | GitHub |
| Notebooks | Google Colab |
| Hosting modelos | HuggingFace Hub |
| Comunicacion | WhatsApp/Discord/Slack |
| Presentacion | Google Slides / Canva |
| Documentacion | Markdown en el repo |

---

## Apendice: Cobertura de Fases del Bootcamp

Verificacion de que el proyecto cubre las 10 fases exigidas por el PDF de la practica final:

| # | Fase del Bootcamp | Donde se cubre | Notebook |
|---|---|---|---|
| 1 | Definicion del problema | `IDEA_ECG_ASSISTANT.md` + Slide 2-3 | Documento |
| 2 | Dataset | Notebook 1: descarga, EDA, preprocesamiento | `ECG_Dataset` |
| 3 | Deep Learning | Notebook 2: CNN 1D, F1=0.981 | `ECG_DeepLearning` |
| 4 | Modelo LLM | Notebook 3: Fine-tuning Gemma 3 + QLoRA | `ECG_FineTuning` |
| 5 | RAG | Notebook 4: FAISS + langchain + docs clinicos | `ECG_RAG` |
| 6 | Evaluacion | Notebook 5: DL metrics + LLM rubrics | `ECG_Evaluation` |
| 7 | Metodologia SCRUM | Trello + burndown chart + rotacion roles | Este documento |
| 8 | Viabilidad economica | Seccion 10 de este documento + Slide 13 | Este documento |
| 9 | Presentacion | 15 slides, ≤15 min | Google Slides |
| 10 | Post LinkedIn | US-15 | LinkedIn |

**Resultado**: Las 10 fases estan cubiertas.
