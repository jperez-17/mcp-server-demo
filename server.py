import os
import json
import logging
import requests
from dotenv import load_dotenv
from typing import Dict, Optional
from fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from collections import defaultdict

load_dotenv()

chat_sessions = defaultdict(list)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-server-demo")

mcp = FastMCP("MCP Server Demo")

user_api_url = os.getenv("USER_API_URL")
default_user_id = os.getenv("DEFAULT_USER_ID")
token_api_url = os.getenv("TOKEN_API_URL")
user_name = os.getenv("TOKEN_USER_NAME")
user_password = os.getenv("TOKEN_USER_PASSWORD")
openai_api_key = os.getenv("OPENAI_API_KEY")

auth_token: Optional[str] = None
user_data: Optional[Dict] = None
vector_store = None

custom_middleware = [
  Middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
  ),
]

http_app = mcp.http_app(
  path='/mcp',
  transport='http',
  middleware=custom_middleware,
  stateless_http=True
)

def setup_vector_store(data: dict):
  """
  Divide el JSON, crea embeddings y los almacena en una base de datos vectorial FAISS.
  """
  global vector_store

  splitter = RecursiveJsonSplitter(max_chunk_size=2000)
  json_chunks = splitter.split_json(json_data=data)
  texts = [json.dumps(chunk) for chunk in json_chunks]
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  vector_store = FAISS.from_texts(texts, embeddings)
  logger.info("✅ Base de datos vectorial creada exitosamente.")

@mcp.tool()
def process_with_rag(
  session_id: str,
  prompt: str = "¿Cuál fue la primera cuota inicial, en qué fecha fue y de cuánto fue el valor?",
  model: str = "gpt-4o",
  temperature: float = 0.2,
) -> str:
  """
  Responde preguntas sobre los datos del usuario utilizando un enfoque RAG y manteniendo memoria por sesión.
  """

  global vector_store
  global user_data

  if not user_data:
    return make_response("error", "No hay datos disponibles. Ejecuta 'fetch_data' primero.")

  if vector_store is None:
    try:
      logger.info("Creando base de datos vectorial por primera vez...")
      setup_vector_store(user_data)
    except Exception as e:
      return make_response("error",  f"Error al crear la base de datos vectorial: {str(e)}")

  try:
    logger.info(f"Procesando prompt con RAG y el modelo: {model}")

    llm = ChatOpenAI(model=model, temperature=temperature)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(prompt)

    history = chat_sessions[session_id]

    system_prompt = (
      "Eres un asistente experto en el análisis de datos en formato JSON, especializado en el sector inmobiliario.\n"
      "Utiliza el contexto proporcionado para responder con precisión a las preguntas del usuario.\n\n"
      "Instrucciones para interpretar la información:\n"
      "- Para información sobre viviendas, analiza el campo: 'unidadesAgrupacion'.\n"
      "- Para información sobre trámites, analiza el campo: 'tramitesAgrupacion'. Si 'fechaCumplimiento' no está disponible, utiliza 'fechaCompromiso', el orden de estos está basado en el campo 'orden'.\n"
      "- Para detalles de pagos o procesos relacionados con las viviendas, analiza el campo: 'planPagosAgrupacion'.\n\n"
      "Importante:\n"
      "- Si la información solicitada no se encuentra en el contexto, responde indicando que no hay suficiente información disponible.\n\n"
      "Contexto:\n{context}"
    ).format(context=relevant_docs)

    langchain_messages = [SystemMessage(content=system_prompt)]

    for m in history:
      if m["role"] == "user":
        langchain_messages.append(HumanMessage(content=m["content"]))
      elif m["role"] == "assistant":
        langchain_messages.append(AIMessage(content=m["content"]))

    langchain_messages.append(HumanMessage(content=prompt))
    response = llm.invoke(langchain_messages)

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": response.content})

    return make_response(
      "success",
      "Respuesta generada con RAG",
      { "answer": response.content }
    )

  except Exception as e:
    return make_response("error", f"Error en el procesamiento RAG: {str(e)}")

async def authenticate(
  api_url: str,
  username: str,
  password: str
) -> str:
  """
  Realiza la autenticación en la API y guarda el token globalmente.
  """
  global auth_token

  if not api_url or not username or not password:
    return make_response("error", "Faltan parámetros de autenticación")

  payload = json.dumps({
    "NomUsuario": username,
    "ClaveUsuario": password
  })

  try:
    logger.info(f"Intentando autenticar en: {api_url}")

    response = requests.post(
      api_url,
      data=payload,
      headers={"Content-Type": "application/json"},
      timeout=30
    )
    response.raise_for_status()
    data = response.json()
    auth_token = data.get("access_token")

    logger.info(f"Respuesta de autenticación recibida")
    
    if auth_token:
      return make_response(
        "success",
        "Autenticación exitosa. Token obtenido y guardado.",
        { 
          "token": auth_token
        }
      )

    else:
      return make_response("warning", "Autenticación completada pero no se encontró token en la respuesta", data)

  except requests.exceptions.HTTPError as e:
    error_msg = f"Error HTTP {e.response.status_code}: {e.response.text}"
    logger.error(f"❌ {error_msg}")
    return make_response("error", error_msg, e.response.text)
  except requests.exceptions.Timeout:
    error_msg = "Timeout en la petición de autenticación"
    logger.error(f"❌ {error_msg}")
    return make_response("error", error_msg, e.response.text)
  except requests.exceptions.RequestException as e:
    error_msg = f"Error de conexión en autenticación: {str(e)}"
    logger.error(f"❌ {error_msg}")
    return make_response("error", error_msg, e.response.text)
  except Exception as e:
    error_msg = f"Error inesperado en autenticación: {str(e)}"
    logger.error(f"❌ {error_msg}")
    return make_response("error", error_msg, e.response.text)

@mcp.tool()
async def fetch_data(
  user_id: str = default_user_id,
) -> str:
  """
  Obtiene datos de la API usando el token de autenticación
  
  Args:
    user_id: Id de usuario
  
  Returns:
    Datos obtenidos de la API en formato JSON
  """
  global auth_token, user_data

  await authenticate(
    api_url=token_api_url,
    username=user_name,
    password=user_password
  )

  if not auth_token:
    return make_response("error", "No hay token de autenticación disponible.")

  try:
    api_url = f"{user_api_url}/{user_id}"

    logger.info(f"Obteniendo datos de: {api_url}")

    request_headers = {
      "Authorization": f"Bearer {auth_token}",
      "Content-Type": "application/json",
    }

    response = requests.get(
      api_url,
      headers=request_headers,
      timeout=30
    )

    response.raise_for_status()
    user_data = response.json()

    logger.info(f"✅ Datos obtenidos exitosamente")

    return make_response("success", "Datos obtenidos exitosamente", user_data)

  except requests.exceptions.HTTPError as e:
    error_msg = f"Error HTTP {e.response.status_code}: {e.response.text}"
    logger.error(f"❌ {error_msg}")
    return make_response("error", error_msg, e.response.text)
  except requests.exceptions.Timeout:
    error_msg = "Timeout obteniendo datos"
    logger.error(f"❌ {error_msg}")
    return make_response("error", error_msg, e.response.text)
  except requests.exceptions.RequestException as e:
    error_msg = f"Error de conexión obteniendo datos: {str(e)}"
    logger.error(f"❌ {error_msg}")
    return make_response("error", error_msg, e.response.text)
  except Exception as e:
    error_msg = f"Error inesperado obteniendo datos: {str(e)}"
    logger.error(f"❌ {error_msg}")
    return make_response("error", error_msg, e.response.text)

@mcp.tool()
def clear_user_data() -> str:
  """
  Limpia el token de autenticación actual
  
  Returns:
      Confirmación de limpieza
  """
  global auth_token, user_data
  auth_token = None
  user_data = None
  return make_response("success", "Token de autenticación y datos de usuario eliminados")

@mcp.tool()
def clear_session(session_id: str = "default") -> str:
  """
  Limpia el historial de la sesión indicada.
  """
  if session_id in chat_sessions:
    del chat_sessions[session_id]
    return make_response("success", f"Historial de la sesión '{session_id}' eliminado.")
  return make_response("warning", f"No existe la sesión '{session_id}'.")

def make_response(status: str, message: str, data=None) -> str:
  """
  Genera una respuesta estándar en formato JSON para las tools MCP.
  """
  result = {
    "status": status,
    "message": message,
    "data": data
  }
  return f"{json.dumps(result, indent=2)}"

if __name__ == "__main__":
  import uvicorn

  port = int(os.environ.get("PORT", 8000))

  uvicorn.run(
    http_app,
    host="0.0.0.0",
    port=port
  )
