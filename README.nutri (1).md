# 🥗 NutriGuía — Agente experto con RAG condicional

Agente conversacional experto en **nutrición y alimentación saludable**, construido con **Gemini** (LLM \+ embeddings), **ChromaDB**, **LangChain** y **LangGraph**. Decide por sí mismo cuándo consultar su base de conocimiento y cuándo responder directamente, y recuerda la conversación.

> Dominio de ejemplo: se puede cambiar a cualquier otro editando `DOMAIN_*`, `SYSTEM_PROMPT`, `ROUTER_PROMPT` (en `rag_agent.py`) y los documentos de `docs/`.

## 1\. Dominio elegido y por qué

Nutrición, porque (a) existen documentos públicos y fiables (guías alimentarias oficiales, informes de organismos de salud), (b) las preguntas son factuales y verificables —encaja con RAG— y (c) en una conversación real mezclan preguntas técnicas con charla general, lo que justifica un **RAG condicional**.

**Base de conocimiento:** ≥ 3 documentos (\~20 páginas) en `docs/`&nbsp;

1. **Guía de la alimentación saludable para Atención Primaria y colectivos ciudadanos (SENC)**: la guía de la Sociedad Española de Nutrición Comunitaria.

**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**[https\://www\.fesnad.org/resources/files/guiaSENC.pdf fesnad](https://www.fesnad.org/resources/files/guiaSENC.pdf)

&nbsp;

2. **Informe del Comité Científico de la AESAN sobre recomendaciones dietéticas sostenibles y de actividad física**: es el informe técnico que está detrás de la campaña oficial, enlazado desde el reportaje de Agencia SINC. [https\://www\.aesan.gob.es/AECOSAN/docs/documentos/seguridad\_alimentaria/evaluacion\_riesgos/informes\_comite/INFORME\_RECOMENDACIONES\_DIETETICAS.pdf](https://www.aesan.gob.es/AECOSAN/docs/documentos/seguridad_alimentaria/evaluacion_riesgos/informes_comite/INFORME_RECOMENDACIONES_DIETETICAS.pdf) [agenciasinc](https://cdn.agenciasinc.es/print/134185)

&nbsp;

3. **“Come sano, muévete y cuida tu planeta” (AESAN, versión divulgativa)**: las recomendaciones para la población española, con más productos vegetales, cereales integrales, aceite de oliva y menos carnes procesadas, azúcar y sal.  
   [https\://www\.consejodietistasnutricionistas.com/wp-content/uploads/2024/11/6.-RECOMENDACIONES\_DIETETICAS\_compressed.pdf](https://www.consejodietistasnutricionistas.com/wp-content/uploads/2024/11/6.-RECOMENDACIONES_DIETETICAS_compressed.pdf)

## 2\. Requisitos

- Python 3.10+  
- API key de Gemini (Google AI Studio)  
- Dependencias: `pip install -r requirements.txt`

## 3\. Instalación y ejecución

```sh
git clone <tu-repo> && cd proyecto_agente_rag
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # y pon tu GOOGLE_API_KEY
# Copia tus PDFs/CSVs/TXT en ./docs
```

**Notebook:** abre `NutriGuia_agente_RAG.ipynb` (VS Code o Jupyter) y ejecuta las celdas en orden. En **Colab**: sube el notebook y la carpeta `docs/`, y guarda la clave en *Secretos* como `GOOGLE_API_KEY`.

**Streamlit (recomendado):**

```sh
streamlit run app.py
```

**Despliegue en Streamlit Community Cloud:** sube el repo a GitHub (con `docs/`, sin `.env`), crea la app apuntando a `app.py` y añade en *Secrets*: `GOOGLE_API_KEY = "..."`.&nbsp;

## 4\. System prompt y justificación

El prompt completo está en `SYSTEM_PROMPT` (`rag_agent.py`). Estructura y razones:

| Bloque | Qué dice | Por qué |
| :---- | :---- | :---- |
| **Rol** | Experto divulgador, no médico ni dietista | Delimita responsabilidad y evita que parezca asesoramiento sanitario. |
| **Tono** | Cercano, claro, breve, en el idioma del usuario | El público no es técnico. |
| **Uso de la información** | Basarse en el contexto, citar `[doc, p.X]`, no inventar | Reduce alucinaciones y hace las respuestas verificables. |
| **Sin información suficiente** | Admitirlo y separar claramente lo que no viene de los documentos | Un RAG fiable debe saber decir «no lo sé». |
| **Límites** | Derivar a profesionales en síntomas, embarazo o TCA; reconducir lo off-topic | Seguridad y coherencia con el enfoque del agente. |

&nbsp;

## 5\. Justificación del RAG condicional

**Condición:** un nodo *router* (LLM con salida estructurada `RouteDecision`) decide.

- **Consulta la base** si la pregunta pide información técnica del dominio o es un seguimiento que la necesita.  
- **No la consulta** en saludos, agradecimientos, preguntas sobre la conversación o sobre el agente, y reformulaciones de la respuesta anterior.

**Por qué así y no «siempre RAG»:** ahorra latencia y coste, y evita meter contexto irrelevante que confunda al modelo (p. ej. recuperar fragmentos para un «gracias»). **Por qué un LLM y no reglas:** las reglas (palabras clave) fallan con el lenguaje natural y con las preguntas de seguimiento.&nbsp;

## 6\. Otras decisiones técnicas

- **Chunking** 1000/150 con `RecursiveCharacterTextSplitter`: fragmentos con sentido completo sin perder ideas en los cortes.  
- **MIN\_RELEVANCE:** Si el agente pasa al LLM cualquier fragmento recuperado, puede acabar respondiendo con contexto que no tiene relación con la pregunta. Hace falta un umbral de similitud por debajo del cual se descarte el contexto y el agente admita que no sabe. Se decidió que fuese 0.35. Sin embargo vimos que no descartaba nada. Con la pregunta *¿cómo se arregla el motor de un coche?*, que está fuera de dominio, la puntuación fue de 0.60-0.62. Por tanto el umbral debería rondar esos valores. Aun así mantuvimos el 0.35 debido a los pocos valores o datos que teníamos para justificar un umbral en torno al 0.60, ya que pueden haber preguntas mal formuladas o errores. Por ello el system prompt es la segunda barrera: si llega contexto flojo, el agente tiene instrucciones de decir «no encuentro esa información» en vez de inventar.  
- **Indexación idempotente y persistente:** los embeddings se calculan una vez.  
- **Memoria acotada** a los últimos 10 mensajes, para controlar tokens.

## 7\. Limitaciones

Calidad dependiente de los documentos y del umbral; el router puede equivocarse en casos límite; `MemorySaver` es en RAM (en producción, un checkpointer persistente); no sustituye a un profesional sanitario.

&nbsp;

## 8\. Estructura

```
├── NutriGuia_agente_RAG.ipynb   # entregable principal
├── rag_agent.py                 # lógica del agente (reutilizada por notebook y app)
├── app.py                       # interfaz Streamlit
├── docs/                        # documentos de la base de conocimiento
├── requirements.txt · .env.example · .gitignore
```

&nbsp;