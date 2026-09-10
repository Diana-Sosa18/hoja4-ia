# Hoja de Trabajo #4 - Herramientas (CC3116)

Agente de preguntas frecuentes para Parachute S.A. Implementa una base de datos
vectorial en PostgreSQL con `pgvector` y un agente de LLM (Google Gemini, con
tool use / function calling) que consulta esa base de datos como única fuente
de verdad para responder preguntas sobre el evento.

## Arquitectura

- **Base de datos vectorial:** PostgreSQL + extensión [pgvector](https://github.com/pgvector/pgvector),
  corriendo en un contenedor Docker.
- **Embeddings:** `sentence-transformers` con el modelo `all-MiniLM-L6-v2` (384
  dimensiones). Se embebe únicamente el texto de la `PREGUNTA` de cada FAQ (no
  la respuesta), porque las respuestas del corpus comparten una plantilla casi
  idéntica entre sí y, si se incluyeran, todos los embeddings quedarían
  demasiado parecidos entre ellos.
- **Script de carga** (`src/load_data.py`): parsea `data/Corpus_FAQs_Parachute_SA_2026.txt`,
  genera un embedding por cada FAQ y lo inserta/actualiza en la tabla `faqs`.
- **Agente** (`src/agent.py`): CLI interactiva que usa el SDK de Google Gemini
  (`google-genai`) con una herramienta (`buscar_faqs`) configurada vía function
  calling. El modelo decide cuándo llamar a la herramienta, esta hace la
  búsqueda semántica en pgvector (top 5) y el modelo redacta la respuesta
  final basándose únicamente en esos resultados.

  > **Nota sobre el umbral de relevancia:** `all-MiniLM-L6-v2` es un modelo
  > entrenado mayormente en inglés, así que en español su puntaje de similitud
  > coseno no siempre distingue de forma confiable una pregunta relacionada de
  > una totalmente ajena (se observaron casos donde una pregunta sin relación
  > alguna puntuó más alto que un parafraseo válido). Por eso el agente no
  > filtra resultados por un umbral numérico fijo: la herramienta siempre
  > devuelve los 5 resultados más cercanos con su puntaje, y es Gemini quien,
  > leyendo el contenido de cada uno, decide si realmente responden la
  > pregunta o si debe admitir que no tiene esa información.

  > **¿Por qué Gemini y no Anthropic/OpenAI?** La API de Gemini tiene un nivel
  > gratuito real (sin registrar tarjeta de crédito) suficiente para este
  > proyecto, a diferencia de la API de Anthropic o de OpenAI que requieren
  > cargar saldo desde la primera llamada.

  > **Sobre el corpus de FAQs:** de las 120 entradas, unas 106 (~88%) tienen
  > una respuesta genérica de plantilla ("consulte a soporte para más
  > detalles") y solo ~14 tienen un dato concreto (peso máximo, altura del
  > salto, edad mínima, etc.). El agente trata la respuesta genérica como
  > información oficial válida siempre que el TEMA de la pregunta coincida
  > (no la descarta por "poco detallada"), y solo admite que no sabe cuando
  > ninguna entrada de la base trata el tema preguntado. Para un video más
  > vistoso, prueba con preguntas que sí tengan dato concreto, por ejemplo:
  > *"¿Cuál es el peso máximo permitido para saltar?"*,
  > *"¿A qué altura se realiza el salto tándem?"* o
  > *"¿Cuál es la edad mínima para saltar?"*.

## Requisitos previos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo.
- Python 3.11 - 3.13 (recomendado; `sentence-transformers` puede no tener wheels
  disponibles todavía para versiones más nuevas).
- Una API key de Google Gemini ([Google AI Studio](https://aistudio.google.com/apikey),
  nivel gratuito, no requiere tarjeta de crédito).

## 1. Inicializar la infraestructura

### 1.1 Variables de entorno

Copia el archivo de ejemplo y completa tu API key:

```bash
cp .env.example .env
```

Edita `.env` y coloca tu `GEMINI_API_KEY` (consíguela gratis en
[Google AI Studio](https://aistudio.google.com/apikey) con tu cuenta de
Google, sin tarjeta de crédito). Los valores de Postgres ya tienen defaults
funcionales para desarrollo local.

### 1.2 Levantar PostgreSQL + pgvector con Docker Compose

Con Docker Desktop abierto, desde la raíz del repositorio:

```bash
docker compose up -d
```

Esto descarga la imagen [`pgvector/pgvector:pg16`](https://hub.docker.com/r/pgvector/pgvector)
y levanta un contenedor `hoja4_pgvector` escuchando en `localhost:5433`, con un
volumen persistente (`pgvector_data`) para no perder los datos entre reinicios.

> **Nota:** se usa el puerto `5433` (no el `5432` por defecto de PostgreSQL) para
> evitar conflictos si ya tienes una instalación nativa de PostgreSQL corriendo
> en el equipo. Si tu máquina tiene el puerto `5432` libre, puedes cambiarlo en
> `.env` (`POSTGRES_PORT`).

Verifica que el contenedor esté saludable:

```bash
docker compose ps
```

Para detener el contenedor (sin borrar datos):

```bash
docker compose down
```

Para borrar también el volumen de datos (reinicio completo):

```bash
docker compose down -v
```

> ¿Prefieres Podman? El mismo archivo funciona con `podman compose up -d`
> (o `podman-compose up -d`), ya que la sintaxis de Compose es compatible.

### 1.3 Entorno de Python

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

La extensión `vector` y la tabla `faqs` se crean automáticamente la primera
vez que corres el script de carga (`db/schema.sql`), no requieren pasos manuales.

## 2. Cargar la base de conocimientos

Con el contenedor corriendo y el entorno activado:

```bash
python src/load_data.py
```

Esto parsea las 120 FAQs de `data/Corpus_FAQs_Parachute_SA_2026.txt`, genera
sus embeddings con `all-MiniLM-L6-v2` y las inserta (o actualiza, si ya existen)
en la tabla `faqs` de PostgreSQL.

## 3. Ejecutar el agente

```bash
python src/agent.py
```

El agente responde preguntas en una sesión interactiva por terminal, apoyándose
siempre en la herramienta `buscar_faqs` para consultar la base de datos
vectorial antes de responder. Si la pregunta no tiene relación con información
disponible en la base de conocimientos, el agente lo indica explícitamente en
lugar de inventar una respuesta.

Para salir de la sesión, escribe `Bye` o presiona `Ctrl-C`.

## Solución de problemas

- **`FATAL: la autentificación password falló` / errores raros de codificación
  al conectar con psycopg en Windows:** se debe a un bug conocido de
  `psycopg2` con la configuración regional en español de Windows. Este proyecto
  ya usa `psycopg` (v3), que no tiene ese problema.
- **La conexión falla o llega a un Postgres con credenciales distintas:**
  verifica que no tengas otro PostgreSQL (nativo, no en Docker) escuchando en
  el mismo puerto (`netstat -ano | findstr 5432`). Por eso este proyecto usa
  `5433` por defecto.
- **`El servicio de Gemini no está disponible` / error 503:** el nivel
  gratuito de Gemini a veces satura ("high demand"); el agente ya reintenta
  automáticamente con backoff. Si persiste, espera un minuto e intenta de nuevo.
- **`Se agotó la cuota gratuita` / error 429 `RESOURCE_EXHAUSTED`:** cada
  modelo de Gemini tiene su propio límite de solicitudes gratis por día/minuto
  (algunos modelos de vista previa, como los que resuelve el alias
  `gemini-flash-latest`, pueden tener límites muy bajos, ej. 20/día). Cambia
  `GEMINI_MODEL` en tu `.env` a otro modelo (por ejemplo `gemini-2.5-flash-lite`)
  o espera a que se reinicie la cuota.

## Estructura del repositorio

```
.
├── data/
│   └── Corpus_FAQs_Parachute_SA_2026.txt   # Corpus de FAQs entregado por Parachute S.A.
├── db/
│   └── schema.sql                          # Definición de la tabla faqs + índice ivfflat
├── src/
│   ├── db.py                               # Conexión a PostgreSQL/pgvector
│   ├── load_data.py                        # Script de carga (parseo + embeddings + insert)
│   └── agent.py                            # Agente CLI con tool use (Google Gemini)
├── docker-compose.yml                      # Contenedor de PostgreSQL + pgvector
├── requirements.txt
├── .env.example
└── README.md
```
