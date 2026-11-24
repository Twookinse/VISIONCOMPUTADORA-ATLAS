"""
Control de acceso al servidor.

Reglas:
- Desde la propia PC (127.0.0.1 / ::1) el acceso es libre, para que el lanzador
  de 1 clic siga funcionando igual.
- Desde otro equipo de la red (ej. un celular conectado al WiFi del local) se
  exige usuario y contrasena (HTTP Basic, el navegador muestra una ventana de login).
  Se configuran con NEX_USER / NEX_PASSWORD en el archivo .env (ver .env.example).
- Si NEX_PASSWORD no esta configurada, el acceso remoto queda BLOQUEADO.

Se implementa como middleware ASGI para cubrir tambien el WebSocket y el
streaming MJPEG, no solo los endpoints REST.
"""

import base64
import json
import secrets

from src.config.settings import auth_password, auth_user

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def is_loopback(host):
    host = str(host or "")
    return host in LOOPBACK_HOSTS or host.startswith("127.") or host == "::ffff:127.0.0.1"


def _check_basic_auth(header_value, user, password):
    if not header_value or not header_value.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header_value[6:].strip()).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    given_user, _, given_pass = decoded.partition(":")
    user_ok = secrets.compare_digest(given_user.encode(), user.encode())
    pass_ok = secrets.compare_digest(given_pass.encode(), password.encode())
    return user_ok and pass_ok


class AccessControlMiddleware:
    """Permite la propia PC sin login y exige contrasena a los demas equipos."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        client = scope.get("client") or ("", 0)
        if is_loopback(client[0]):
            return await self.app(scope, receive, send)

        password = auth_password()
        if password:
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
            if _check_basic_auth(headers.get("authorization"), auth_user(), password):
                return await self.app(scope, receive, send)
            return await self._reject(scope, receive, send, 401,
                                      "Usuario o contrasena incorrectos", ask_login=True)

        return await self._reject(
            scope, receive, send, 403,
            "Acceso remoto deshabilitado. Configura NEX_PASSWORD en el archivo .env "
            "del servidor para permitir el acceso desde otros equipos.",
        )

    @staticmethod
    async def _reject(scope, receive, send, status, message, ask_login=False):
        if scope["type"] == "websocket":
            # Cerrar el handshake sin aceptar la conexion
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"detail": message}, ensure_ascii=False).encode("utf-8")
        headers = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(body)).encode()),
        ]
        if ask_login:
            headers.append((b"www-authenticate", b'Basic realm="Nex Vision", charset="UTF-8"'))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
