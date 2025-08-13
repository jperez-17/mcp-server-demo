import os
import json
import logging
import requests
from dotenv import load_dotenv
from typing import Any, Dict, Optional
from fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

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
  print("✅ Base de datos vectorial creada exitosamente.")

@mcp.tool()
def process_with_rag(
  prompt: str = "¿Cuál fue la primera cuota inicial, en qué fecha fue y de cuánto fue el valor?",
  model: str = "gpt-4o",
  temperature: float = 0.2,
) -> str:
  """
  Responde preguntas sobre los datos del usuario utilizando un enfoque RAG.
  """
  global vector_store
  global user_data

  if not user_data:
    return "❌ No hay datos disponibles. Ejecuta 'fetch_data' primero."

  if vector_store is None:
    try:
      logger.info("Creando base de datos vectorial por primera vez...")
      setup_vector_store(user_data)
    except Exception as e:
      return f"❌ Error al crear la base de datos vectorial: {str(e)}"

  try:
    logger.info(f"Procesando prompt con RAG y el modelo: {model}")
    llm = ChatOpenAI(model=model, temperature=temperature)

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(prompt)

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
    )

    prompt_template = ChatPromptTemplate.from_messages(
      [
        ("system", system_prompt),
        ("human", "{input}"),
      ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    response = question_answer_chain.invoke({"input": prompt, "context": relevant_docs})
    return f"🤖 Respuesta RAG:\n{response}"

  except Exception as e:
    return f"❌ Error en el procesamiento RAG: {str(e)}"

@mcp.tool()
async def authenticate(
  api_url: str = token_api_url,
  username: str = user_name,
  password: str = user_password,
) -> str:
  """
  Obtiene token de autenticación de la API
  
  Args:
    api_url: URL del endpoint de autenticación
    username: nombre de usuario para autenticación
    password: contraseña de usuario para autenticación
  
  Returns:
    Mensaje de confirmación de autenticación
  """
  global auth_token

  if not api_url or not username or not password:
    return "❌ Faltan parámetros de autenticación."

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
      return f"✅ Autenticación exitosa. Token obtenido y guardado."
    else:
      return f"⚠️ Autenticación completada pero no se encontró token en la respuesta:\n{json.dumps(data, indent=2)}"
            
  except requests.exceptions.HTTPError as e:
    error_msg = f"❌ Error HTTP {e.response.status_code}: {e.response.text}"
    logger.error(error_msg)
    return error_msg
  except requests.exceptions.Timeout:
    error_msg = "❌ Timeout en la petición de autenticación"
    logger.error(error_msg)
    return error_msg
  except requests.exceptions.RequestException as e:
    error_msg = f"❌ Error de conexión en autenticación: {str(e)}"
    logger.error(error_msg)
    return error_msg
  except Exception as e:
    error_msg = f"❌ Error inesperado en autenticación: {str(e)}"
    logger.error(error_msg)
    return error_msg

@mcp.tool()
async def fetch_data(
  endpoint: str = user_api_url,
  user_id: str = default_user_id,
) -> str:
  """
  Obtiene datos de la API usando el token de autenticación
  
  Args:
    endpoint: URL del endpoint para obtener datos
    user_id: Id de usuario
  
  Returns:
    Datos obtenidos de la API en formato JSON
  """
  global auth_token, user_data
  
  if not auth_token:
    return "❌ No hay token de autenticación disponible. Ejecuta 'authenticate' primero."
    
  try:
    api_url = f"{endpoint}/{user_id}"

    logger.info(f"Obteniendo datos de: {endpoint}")

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

    try:
      user_data = response.json()
      logger.info(f"✅ Datos obtenidos exitosamente")
      return f"✅ Datos obtenidos exitosamente:\n{json.dumps(user_data, indent=2, ensure_ascii=False)}"

    except requests.exceptions.JSONDecodeError:
      return f"✅ Datos obtenidos (no JSON):\n{response.text}"

  except requests.exceptions.HTTPError as e:
    error_msg = f"❌ Error HTTP {e.response.status_code}: {e.response.text}"
    logger.error(error_msg)
    return error_msg
  except requests.exceptions.Timeout:
    error_msg = "❌ Timeout obteniendo datos"
    logger.error(error_msg)
    return error_msg
  except requests.exceptions.RequestException as e:
    error_msg = f"❌ Error de conexión obteniendo datos: {str(e)}"
    logger.error(error_msg)
    return error_msg
  except Exception as e:
    error_msg = f"❌ Error inesperado obteniendo datos: {str(e)}"
    logger.error(error_msg)
    return error_msg

@mcp.tool()
def clear_auth() -> str:
  """
  Limpia el token de autenticación actual
  
  Returns:
      Confirmación de limpieza
  """
  global auth_token
  auth_token = None
  return "✅ Token de autenticación eliminado"

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8000))

  mcp.run(
    transport="http",
    host="0.0.0.0",
    port=port,
    path="/mcp",
    log_level="info"
  )