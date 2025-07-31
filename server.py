import os
import asyncio
import json
import logging
import requests
from dotenv import load_dotenv
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-server-demo")

mcp = FastMCP("MCP Server Demo")

auth_token: Optional[str] = None

@mcp.tool()
async def authenticate(
    api_url: str,
    username: str,
    password: str,
) -> str:
    global auth_token

    if not api_url:
      raise ValueError("Falta el parámetro api_url")
    if not username:
        raise ValueError("Falta el parámetro username")
    if not password:
        raise ValueError("Falta el parámetro password")

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
        logger.info(f"Respuesta de autenticación recibida")
        
        auth_token = data.get("access_token")
        
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
def fetch_data(
    endpoint: str,
    user_id: str,
) -> str:
    global auth_token
    
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
            data = response.json()
            logger.info(f"Datos obtenidos exitosamente")
            return f"✅ Datos obtenidos exitosamente:\n{json.dumps(data, indent=2, ensure_ascii=False)}"
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
    global auth_token
    auth_token = None
    return "✅ Token de autenticación eliminado"


