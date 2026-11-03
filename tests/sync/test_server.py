import dataclasses
import hmac
import http
import logging
import pathlib
import socket
import threading
import time
import unittest
from unittest.mock import patch

from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidHandshake,
    InvalidStatus,
    NegotiationError,
)
from websockets.http11 import Request, Response
from websockets.sync.client import connect, unix_connect
from websockets.sync.server import *

from ..utils import (
    CLIENT_CONTEXT,
    MS,
    SERVER_CONTEXT,
    DeprecationTestCase,
    LoggingTestCase,
    temp_unix_socket_path,
)
from .server import (
    EvalShellMixin,
    get_host_port,
    get_uri,
    handler,
    run_server,
    run_unix_server,
)


class ServerTests(EvalShellMixin, LoggingTestCase, unittest.TestCase):
    def test_connection(self):
        """Server receives connection from client and the handshake succeeds."""
        with run_server() as server:
            with connect(get_uri(server)) as client:
                self.assertEval(client, "ws.protocol.state.name", "OPEN")

    def test_connection_handler_returns(self):
        """Connection handler returns."""
        with run_server() as server:
            with connect(get_uri(server) + "/no-op") as client:
                with self.assertRaises(ConnectionClosedOK) as raised:
                    client.recv()

        self.assertEqual(
            str(raised.exception),
            "received 1000 (OK); then sent 1000 (OK)",
        )

    def test_connection_handler_raises_exception(self):
        """Connection handler raises an exception."""
        with run_server() as server:
            with connect(get_uri(server) + "/crash") as client:
                with self.assertRaises(ConnectionClosedError) as raised:
                    client.recv()

        self.assertEqual(
            str(raised.exception),
            "received 1011 (internal error); then sent 1011 (internal error)",
        )

    def test_existing_socket(self):
        """Server receives connection using a pre-existing socket."""
        with socket.create_server(("localhost", 0)) as sock:
            host, port = sock.getsockname()
            # Unset the default values of host and port set by run_server.
            with run_server(host=None, port=None, sock=sock):
                with connect(f"ws://{host}:{port}/") as client:
                    self.assertEval(client, "ws.protocol.state.name", "OPEN")

    def test_select_subprotocol(self):
        """Server selects a subprotocol with the select_subprotocol callable."""

        def select_subprotocol(ws, subprotocols):
            ws.select_subprotocol_ran = True
            assert "chat" in subprotocols
            return "chat"

        with run_server(
            subprotocols=["chat"],
            select_subprotocol=select_subprotocol,
        ) as server:
            with connect(get_uri(server), subprotocols=["chat"]) as client:
                self.assertEval(client, "ws.select_subprotocol_ran", "True")
                self.assertEval(client, "ws.subprotocol", "chat")

    def test_select_subprotocol_rejects_handshake(self):
        """Server rejects handshake if select_subprotocol raises NegotiationError."""

        def select_subprotocol(ws, subprotocols):
            raise NegotiationError

        with run_server(select_subprotocol=select_subprotocol) as server:
            with self.assertRaises(InvalidStatus) as raised:
                with connect(get_uri(server)):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 400",
        )

    def test_select_subprotocol_raises_exception(self):
        """Server returns an error if select_subprotocol raises an exception."""

        def select_subprotocol(ws, subprotocols):
            raise RuntimeError

        with run_server(select_subprotocol=select_subprotocol) as server:
            with self.assertRaises(InvalidStatus) as raised:
                with connect(get_uri(server)):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 500",
        )

    def test_compression_is_enabled(self):
        """Server enables compression by default."""
        with run_server() as server:
            with connect(get_uri(server)) as client:
                self.assertEval(
                    client,
                    "[type(ext).__name__ for ext in ws.protocol.extensions]",
                    "['PerMessageDeflate']",
                )

    def test_disable_compression(self):
        """Server disables compression."""
        with run_server(compression=None) as server:
            with connect(get_uri(server)) as client:
                self.assertEval(client, "ws.protocol.extensions", "[]")

    def test_process_request_returns_none(self):
        """Server runs process_request and continues the handshake."""

        def process_request(ws, request):
            self.assertIsInstance(request, Request)
            ws.process_request_ran = True

        with run_server(process_request=process_request) as server:
            with connect(get_uri(server)) as client:
                self.assertEval(client, "ws.process_request_ran", "True")

    def test_process_request_returns_response(self):
        """Server aborts handshake if process_request returns a response."""

        def process_request(ws, request):
            return ws.respond(http.HTTPStatus.FORBIDDEN, "Forbidden")

        def handler(ws):
            self.fail("handler must not run")

        with self.assertNoLogs("websockets", logging.ERROR):
            with run_server(handler, process_request=process_request) as server:
                with self.assertRaises(InvalidStatus) as raised:
                    with connect(get_uri(server)):
                        self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 403",
        )

    def test_process_request_raises_exception(self):
        """Server returns an error if process_request raises an exception."""

        def process_request(ws, request):
            raise RuntimeError("BOOM")

        with self.assertLogs("websockets", logging.ERROR) as logs:
            with run_server(process_request=process_request) as server:
                with self.assertRaises(InvalidStatus) as raised:
                    with connect(get_uri(server)):
                        self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 500",
        )
        self.assertExceptionLogged(logs, "process_request failed", "BOOM")

    def test_process_response_returns_none(self):
        """Server runs process_response but keeps the handshake response."""

        def process_response(ws, request, response):
            self.assertIsInstance(request, Request)
            self.assertIsInstance(response, Response)
            ws.process_response_ran = True

        with run_server(process_response=process_response) as server:
            with connect(get_uri(server)) as client:
                self.assertEval(client, "ws.process_response_ran", "True")

    def test_process_response_modifies_response(self):
        """Server runs process_response and modifies the handshake response."""

        def process_response(ws, request, response):
            response.headers["X-ProcessResponse"] = "OK"

        with run_server(process_response=process_response) as server:
            with connect(get_uri(server)) as client:
                self.assertEqual(client.response.headers["X-ProcessResponse"], "OK")

    def test_process_response_replaces_response(self):
        """Server runs process_response and replaces the handshake response."""

        def process_response(ws, request, response):
            headers = response.headers.copy()
            headers["X-ProcessResponse"] = "OK"
            return dataclasses.replace(response, headers=headers)

        with run_server(process_response=process_response) as server:
            with connect(get_uri(server)) as client:
                self.assertEqual(client.response.headers["X-ProcessResponse"], "OK")

    def test_process_response_raises_exception(self):
        """Server returns an error if process_response raises an exception."""

        def process_response(ws, request, response):
            raise RuntimeError("BOOM")

        with self.assertLogs("websockets", logging.ERROR) as logs:
            with run_server(process_response=process_response) as server:
                with self.assertRaises(InvalidStatus) as raised:
                    with connect(get_uri(server)):
                        self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 500",
        )
        self.assertExceptionLogged(logs, "process_response failed", "BOOM")

    def test_override_server_header(self):
        """Server can override Server header with server_header."""
        with run_server(server_header="Neo") as server:
            with connect(get_uri(server)) as client:
                self.assertEval(client, "ws.response.headers['Server']", "Neo")

    def test_remove_server_header(self):
        """Server can remove Server header with server_header."""
        with run_server(server_header=None) as server:
            with connect(get_uri(server)) as client:
                self.assertEval(client, "'Server' in ws.response.headers", "False")

    def test_keepalive_is_enabled(self):
        """Server enables keepalive and measures latency."""
        with run_server(ping_interval=MS) as server:
            with connect(get_uri(server)) as client:
                client.send("ws.latency")
                latency = eval(client.recv())
                self.assertEqual(latency, 0)
                time.sleep(2 * MS)
                client.send("ws.latency")
                latency = eval(client.recv())
                self.assertGreater(latency, 0)

    def test_disable_keepalive(self):
        """Server disables keepalive."""
        with run_server(ping_interval=None) as server:
            with connect(get_uri(server)) as client:
                time.sleep(2 * MS)
                client.send("ws.latency")
                latency = eval(client.recv())
                self.assertEqual(latency, 0)

    def test_logger(self):
        """Server accepts a logger argument."""
        logger = logging.getLogger("test")
        with run_server(logger=logger) as server:
            self.assertEqual(server.logger.name, logger.name)

    def test_custom_connection_factory(self):
        """Server runs ServerConnection factory provided in create_connection."""

        def create_connection(*args, **kwargs):
            server = ServerConnection(*args, **kwargs)
            server.create_connection_ran = True
            return server

        with run_server(create_connection=create_connection) as server:
            with connect(get_uri(server)) as client:
                self.assertEval(client, "ws.create_connection_ran", "True")

    def test_connections(self):
        """Server provides a connections property."""
        connected = threading.Barrier(2)

        def connection_handler(connection):
            connected.wait()
            handler(connection)

        with run_server(connection_handler) as server:
            self.assertEqual(server.connections, set())
            with (
                connect(get_uri(server)) as client1,
                connect(get_uri(server)) as client2,
            ):
                client1.send("ws.id")
                client2.send("ws.id")
                self.assertEqual(
                    {str(connection.id) for connection in server.connections},
                    {client1.recv(), client2.recv()},
                )
            self.assertEqual(server.connections, set())

    def test_handshake_fails(self):
        """Server receives connection from client but the handshake fails."""

        def remove_key_header(self, request):
            del request.headers["Sec-WebSocket-Key"]

        with run_server(process_request=remove_key_header) as server:
            with self.assertRaises(InvalidStatus) as raised:
                with connect(get_uri(server)):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 400",
        )

    def test_timeout_before_handshake_request(self):
        """Server times out before receiving handshake request from client."""
        with self.assertLogs("websockets", logging.DEBUG) as logs:
            with run_server(open_timeout=MS) as server:
                with socket.create_connection(get_host_port(server)) as sock:
                    # Wait for the server to close the connection.
                    self.assertEqual(sock.recv(4096), b"")

        self.assertExceptionLogged(
            logs,
            "! no valid HTTP request",
            "connection closed while reading HTTP request line",
        )

    def test_connection_closed_before_handshake_request(self):
        """Server reads EOF before receiving handshake request from client."""
        with self.assertLogs("websockets", logging.DEBUG) as logs:
            with run_server() as server:
                with socket.create_connection(get_host_port(server)):
                    # Wait for the server to receive the connection, then close it.
                    time.sleep(MS)

        self.assertExceptionLogged(
            logs,
            "! no valid HTTP request",
            "connection closed while reading HTTP request line",
        )

    def test_connection_closed_before_handshake_response_sync(self):
        """Server reads EOF before sending handshake response to client."""

        def process_request(ws, _request):
            ws.close_socket()

        with self.assertNoLogs("websockets", logging.ERROR):
            with run_server(process_request=process_request) as server:
                with self.assertRaises(InvalidHandshake):
                    with connect(get_uri(server)):
                        self.fail("did not raise")

    def test_junk_handshake_request(self):
        """Server closes the connection when receiving non-HTTP request from client."""
        with self.assertLogs("websockets.server", logging.DEBUG) as logs:
            with run_server() as server:
                with socket.create_connection(get_host_port(server)) as sock:
                    sock.send(b"HELO relay.invalid\r\n")
                    # Wait for the server to close the connection.
                    self.assertEqual(sock.recv(4096), b"")

        self.assertExceptionLogged(
            logs,
            "! no valid HTTP request",
            "invalid HTTP request line: HELO relay.invalid",
        )

    def test_shutdown_stops_accepting_connections(self):
        """Server stops accepting connections."""
        with run_server() as server:
            # Connect a client and disable close_connections to delay closing.
            uri = get_uri(server)
            with connect(uri):
                shutdown_thread = threading.Thread(
                    target=server.shutdown,
                    kwargs={"close_connections": False},
                )
                shutdown_thread.start()
                time.sleep(MS)

                # Server cannot receive new connections.
                with self.assertRaises(OSError):
                    with connect(uri):
                        self.fail("did not raise")

            shutdown_thread.join()

    def test_shutdown_rejects_connecting_connections(self):
        """Server rejects connecting connections with HTTP 503."""

        def process_request(ws, _request):
            ws.server.socket_closed.wait()

        def shutdown_server(server):
            time.sleep(2 * MS)
            server.shutdown()

        with run_server(process_request=process_request) as server:
            shutdown_thread = threading.Thread(target=shutdown_server, args=(server,))
            shutdown_thread.start()
            with self.assertRaises(InvalidStatus) as raised:
                with connect(get_uri(server)):
                    self.fail("did not raise")
            shutdown_thread.join(MS)

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 503",
        )

    def test_shutdown_closes_open_connections(self):
        """Server closes open connections with code 1001."""
        with run_server() as server:
            with connect(get_uri(server)) as client:
                server.shutdown()
                with self.assertRaises(ConnectionClosedOK) as raised:
                    client.recv()

        self.assertEqual(
            str(raised.exception),
            "received 1001 (going away); then sent 1001 (going away)",
        )

    def test_shutdown_closes_open_connections_with_code_and_reason(self):
        """Server closes open connections with a custom code and reason."""
        with run_server() as server:
            with connect(get_uri(server)) as client:
                server.shutdown(code=1012, reason="restarting")
                with self.assertRaises(ConnectionClosedError) as raised:
                    client.recv()

        self.assertEqual(
            str(raised.exception),
            "received 1012 (service restart) restarting; "
            "then sent 1012 (service restart) restarting",
        )

    def test_shutdown_keeps_connections_open(self):
        """Server waits for the client to close open connections."""
        with run_server() as server:
            with connect(get_uri(server)):
                shutdown_thread = threading.Thread(
                    target=server.shutdown,
                    kwargs={"close_connections": False},
                )
                shutdown_thread.start()

                # The server waits for the client to close the connection.
                shutdown_thread.join(MS)
                self.assertTrue(shutdown_thread.is_alive())

            # Once the client closes the connection, the server terminates.
            shutdown_thread.join(MS)
            self.assertFalse(shutdown_thread.is_alive())

    @patch("websockets.sync.server.Server.SHUTDOWN_POLLING_INTERVAL", MS)
    def test_shutdown_keeps_handlers_running(self):
        """Server waits for connection handlers to terminate."""
        with run_server() as server:
            with connect(get_uri(server) + "/delay") as client:
                # Delay termination of the connection handler.
                client.send(str(3 * MS))

                shutdown_thread = threading.Thread(target=server.shutdown)
                shutdown_thread.start()

                # The server waits for the connection handler to terminate.
                shutdown_thread.join(2 * MS)
                self.assertTrue(shutdown_thread.is_alive())

                # Set a longer timeout here, else the test becomes flaky.
                shutdown_thread.join(5 * MS)
                self.assertFalse(shutdown_thread.is_alive())

    def test_fileno(self):
        """Server provides a fileno method."""
        with run_server() as server:
            self.assertIsInstance(server.fileno(), int)

    def test_context_manager_closes_server(self):
        """Server closes when exiting context manager."""
        with serve(handler, "localhost", 0) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            self.addCleanup(thread.join)
            self.assertFalse(server.socket_closed.is_set())

        self.assertTrue(server.socket_closed.is_set())


class SecureServerTests(EvalShellMixin, unittest.TestCase):
    def test_connection(self):
        """Server receives secure connection from client."""
        with run_server(ssl=SERVER_CONTEXT) as server:
            with connect(get_uri(server), ssl=CLIENT_CONTEXT) as client:
                self.assertEval(client, "ws.protocol.state.name", "OPEN")
                self.assertEval(client, "ws.socket.version()[:3]", "TLS")

    def test_timeout_during_tls_handshake(self):
        """Server times out before receiving TLS handshake request from client."""
        with run_server(ssl=SERVER_CONTEXT, open_timeout=MS) as server:
            with socket.create_connection(get_host_port(server)) as sock:
                # Wait for the server to close the connection.
                self.assertEqual(sock.recv(4096), b"")

    def test_connection_closed_during_tls_handshake(self):
        """Server reads EOF before receiving TLS handshake request from client."""
        with run_server(ssl=SERVER_CONTEXT) as server:
            with socket.create_connection(get_host_port(server)):
                # Wait for the server to receive the connection, then close it.
                time.sleep(MS)


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "this test requires Unix sockets")
class UnixServerTests(EvalShellMixin, unittest.TestCase):
    def test_connection(self):
        """Server receives connection from client over a Unix socket."""
        with temp_unix_socket_path() as path:
            with run_unix_server(path):
                with unix_connect(path) as client:
                    self.assertEval(client, "ws.protocol.state.name", "OPEN")

    def test_pathlib_path(self):
        """Server accepts a pathlib.Path object as the path argument."""
        with temp_unix_socket_path() as path:
            with run_unix_server(pathlib.Path(path)):
                with unix_connect(path) as client:
                    self.assertEval(client, "ws.protocol.state.name", "OPEN")


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "this test requires Unix sockets")
class SecureUnixServerTests(EvalShellMixin, unittest.TestCase):
    def test_connection(self):
        """Server receives secure connection from client over a Unix socket."""
        with temp_unix_socket_path() as path:
            with run_unix_server(path, ssl=SERVER_CONTEXT):
                with unix_connect(path, ssl=CLIENT_CONTEXT) as client:
                    self.assertEval(client, "ws.protocol.state.name", "OPEN")
                    self.assertEval(client, "ws.socket.version()[:3]", "TLS")


class ServerUsageErrorsTests(unittest.TestCase):
    def test_host_and_sock(self):
        """Server rejects host when sock is provided."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(sock.close)
        with self.assertRaises(ValueError) as raised:
            serve(handler, host="localhost", sock=sock)

        self.assertEqual(
            str(raised.exception),
            "host is incompatible with sock",
        )

    def test_port_and_sock(self):
        """Server rejects port when sock is provided."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(sock.close)
        with self.assertRaises(ValueError) as raised:
            serve(handler, port=0, sock=sock)

        self.assertEqual(
            str(raised.exception),
            "port is incompatible with sock",
        )

    def test_unix_without_path_or_sock(self):
        """Unix server requires path when sock isn't provided."""
        with self.assertRaises(ValueError) as raised:
            unix_serve(handler)

        self.assertEqual(
            str(raised.exception),
            "missing path argument",
        )

    def test_unix_with_path_and_sock(self):
        """Unix server rejects path when sock is provided."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(sock.close)
        with self.assertRaises(ValueError) as raised:
            unix_serve(handler, path="/", sock=sock)

        self.assertEqual(
            str(raised.exception),
            "path is incompatible with sock",
        )

    def test_invalid_subprotocol(self):
        """Server rejects single value of subprotocols."""
        with self.assertRaises(TypeError) as raised:
            serve(handler, subprotocols="chat")

        self.assertEqual(
            str(raised.exception),
            "subprotocols must be a list, not a str",
        )

    def test_unsupported_compression(self):
        """Server rejects incorrect value of compression."""
        with self.assertRaises(ValueError) as raised:
            serve(handler, compression=False)

        self.assertEqual(
            str(raised.exception),
            "unsupported compression: False",
        )


class BasicAuthTests(EvalShellMixin, unittest.IsolatedAsyncioTestCase):
    def test_valid_authorization(self):
        """basic_auth authenticates client."""
        with run_server(
            process_request=basic_auth(credentials=("hello", "iloveyou")),
        ) as server:
            with connect(
                get_uri(server),
                additional_headers={"Authorization": "Basic aGVsbG86aWxvdmV5b3U="},
            ) as client:
                self.assertEval(client, "ws.username", "hello")

    def test_valid_authorization_non_ascii_password(self):
        """basic_auth authenticates client with non-ASCII password."""
        with run_server(
            process_request=basic_auth(credentials=("test", "123£")),
        ) as server:
            with connect(
                get_uri(server),
                additional_headers={"Authorization": "Basic dGVzdDoxMjPCow=="},
            ) as client:
                self.assertEval(client, "ws.username", "test")

    def test_valid_authorization_non_utf8_password(self):
        """basic_auth authenticates client with non-UTF-8 password."""
        password = b"\xca\xfe".decode(errors="surrogateescape")
        with run_server(
            process_request=basic_auth(credentials=("drink", password)),
        ) as server:
            with connect(
                get_uri(server),
                additional_headers={"Authorization": "Basic ZHJpbms6yv4="},
            ) as client:
                self.assertEval(client, "ws.username", "drink")

    def test_missing_authorization(self):
        """basic_auth rejects client without credentials."""
        with run_server(
            process_request=basic_auth(credentials=("hello", "iloveyou")),
        ) as server:
            with self.assertRaises(InvalidStatus) as raised:
                with connect(get_uri(server)):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 401",
        )

    def test_unsupported_authorization(self):
        """basic_auth rejects client with unsupported credentials."""
        with run_server(
            process_request=basic_auth(credentials=("hello", "iloveyou")),
        ) as server:
            with self.assertRaises(InvalidStatus) as raised:
                with connect(
                    get_uri(server),
                    additional_headers={"Authorization": "Negotiate ..."},
                ):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 401",
        )

    def test_authorization_with_unknown_username(self):
        """basic_auth rejects client with unknown username."""
        with run_server(
            process_request=basic_auth(credentials=("hello", "iloveyou")),
        ) as server:
            with self.assertRaises(InvalidStatus) as raised:
                with connect(
                    get_uri(server),
                    additional_headers={"Authorization": "Basic YnllOnlvdWxvdmVtZQ=="},
                ):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 401",
        )

    def test_authorization_with_incorrect_password(self):
        """basic_auth rejects client with incorrect password."""
        with run_server(
            process_request=basic_auth(credentials=("hello", "changeme")),
        ) as server:
            with self.assertRaises(InvalidStatus) as raised:
                with connect(
                    get_uri(server),
                    additional_headers={"Authorization": "Basic aGVsbG86aWxvdmV5b3U="},
                ):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 401",
        )

    def test_list_of_credentials(self):
        """basic_auth accepts a list of hard coded credentials."""
        with run_server(
            process_request=basic_auth(
                credentials=[
                    ("hello", "iloveyou"),
                    ("bye", "youloveme"),
                ]
            ),
        ) as server:
            with connect(
                get_uri(server),
                additional_headers={"Authorization": "Basic YnllOnlvdWxvdmVtZQ=="},
            ) as client:
                self.assertEval(client, "ws.username", "bye")

    def test_check_credentials_function(self):
        """basic_auth accepts a check_credentials function."""

        def check_credentials(username, password):
            self.assertEqual(username, "hello")
            return hmac.compare_digest(password, "iloveyou")

        with run_server(
            process_request=basic_auth(check_credentials=check_credentials),
        ) as server:
            with connect(
                get_uri(server),
                additional_headers={"Authorization": "Basic aGVsbG86aWxvdmV5b3U="},
            ) as client:
                self.assertEval(client, "ws.username", "hello")

    def test_without_credentials_or_check_credentials(self):
        """basic_auth requires either credentials or check_credentials."""
        with self.assertRaises(ValueError) as raised:
            basic_auth()

        self.assertEqual(
            str(raised.exception),
            "provide either credentials or check_credentials",
        )

    def test_with_credentials_and_check_credentials(self):
        """basic_auth requires only one of credentials and check_credentials."""
        with self.assertRaises(ValueError) as raised:
            basic_auth(
                credentials=("hello", "iloveyou"),
                check_credentials=lambda: False,  # pragma: no cover
            )

        self.assertEqual(
            str(raised.exception),
            "provide either credentials or check_credentials",
        )

    def test_bad_credentials(self):
        """basic_auth receives an unsupported credentials argument."""
        with self.assertRaises(TypeError) as raised:
            basic_auth(credentials=42)

        self.assertEqual(
            str(raised.exception),
            "invalid credentials argument: 42",
        )

    def test_bad_list_of_credentials(self):
        """basic_auth receives an unsupported credentials argument."""
        with self.assertRaises(TypeError) as raised:
            basic_auth(credentials=[42])

        self.assertEqual(
            str(raised.exception),
            "invalid credentials argument: [42]",
        )


class BackwardsCompatibilityTests(DeprecationTestCase):
    def test_ssl_context_argument(self):
        """Server supports the deprecated ssl_context argument."""
        with self.assertDeprecationWarning("ssl_context was renamed to ssl"):
            with run_server(ssl_context=SERVER_CONTEXT) as server:
                with connect(get_uri(server), ssl=CLIENT_CONTEXT):
                    pass

    def test_web_socket_server_class(self):
        with self.assertDeprecationWarning("WebSocketServer was renamed to Server"):
            from websockets.sync.server import WebSocketServer
        self.assertIs(WebSocketServer, Server)


class HandshakeGenerationTests(LoggingTestCase, unittest.TestCase):
    """
    Opening-handshake interception/rewrite generations in the threading
    integration.

    These tests observe causality across connections of ONE server, with
    handshakes running in separate threads: an intent bound to generation N
    stops driving new CONNECTING handshakes once N+1 is pushed; pre-accept
    interception, accept() (including rejects), response rewriting, and the
    shutdown 503 gate remain separate outlets.

    """

    def reject_response(self, uri, **kwargs):
        """Connect, expect a rejection, and return the parsed response."""
        with self.assertRaises(InvalidStatus) as raised:
            with connect(uri, open_timeout=5, **kwargs):
                self.fail("did not reject")
        return raised.exception.response

    def connect_in_background(self, uri, box):
        """Run a rejected connection in a thread and record its status."""

        def run():
            try:
                with connect(uri, open_timeout=5):
                    box["status"] = 101
            except InvalidStatus as exc:
                box["status"] = exc.response.status_code

        thread = threading.Thread(target=run)
        thread.start()
        return thread

    def test_generation_advance_cuts_off_new_handshakes(self):
        """Intents of generation N don't drive new handshakes after advancing."""
        hold_entered = threading.Event()
        release_hold = threading.Event()
        gen1_ids = []

        def intercept(connection, request):
            gen1_ids.append(connection.id)
            if request.path == "/hold":
                hold_entered.set()
                release_hold.wait()
            return connection.respond(http.HTTPStatus.PAYMENT_REQUIRED, "gen1\n")

        with run_server() as server:
            with server.generation() as gen1:
                gen1.process_request(intercept)

                # C1 starts its handshake in gen1 and stays CONNECTING there.
                held = {}
                thread = self.connect_in_background(get_uri(server) + "/hold", held)
                self.assertTrue(hold_entered.wait(5))
                self.assertEqual(gen1.booked, {gen1_ids[0]})

                # Advance to gen2: fresh ledger, no inherited interception.
                with server.generation() as gen2:
                    gen2_ids = []

                    def record(connection, request):
                        gen2_ids.append(connection.id)

                    gen2.process_request(record)
                    with connect(get_uri(server) + "/no-op", open_timeout=5):
                        pass
                    while gen2.booked:  # unbook follows the response.
                        time.sleep(MS)

                    # The new handshake went through accept() on gen2.
                    self.assertEqual(gen2.accepted, {gen2_ids[0]})
                    self.assertEqual(gen2.intercepted, set())
                    self.assertNotIn(gen2_ids[0], gen1_ids)
                    self.assertEqual(gen1.booked, {gen1_ids[0]})

                # gen2 popped: gen1 resumes for new handshakes.
                self.assertTrue(gen2.sealed)
                self.assertIs(server.handshakes.current, gen1)
                response = self.reject_response(get_uri(server) + "/resumed")
                self.assertEqual(response.status_code, 402)

                # The held connection completes with gen1's intent.
                release_hold.set()
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(held["status"], 402)

        self.assertEqual(gen1.intercepted, set(gen1_ids))
        self.assertNotIn(gen1_ids[0], gen2.intercepted)

    def test_intercept_and_accept_reject_stay_separate(self):
        """Forced response without accept() vs reject after accept() fork."""
        seen = []

        def record(connection, request):
            seen.append((connection, request.path))

        def force(connection, request):
            if request.path == "/forced":
                return connection.respond(http.HTTPStatus.PAYMENT_REQUIRED, "forced\n")
            return None

        with self.assertLogs("websockets", logging.INFO) as logs:
            with run_server(origins=[]) as server:
                with server.generation() as generation:
                    generation.process_request(record)
                    generation.process_request(force)

                    # Host interception forces 402 before accept().
                    response = self.reject_response(get_uri(server) + "/forced")
                    self.assertEqual(response.status_code, 402)
                    # No acceptable origin: interception returns None, then
                    # accept() rejects with 403.
                    response = self.reject_response(get_uri(server) + "/")
                    self.assertEqual(response.status_code, 403)

        forced_ws, _ = seen[0]
        rejected_ws, _ = seen[1]
        self.assertFalse(forced_ws.protocol.accept_called)
        self.assertTrue(rejected_ws.protocol.accept_called)
        self.assertEqual(generation.intercepted, {forced_ws.id})
        self.assertEqual(generation.accepted, {rejected_ws.id})
        messages = "".join(record.getMessage() for record in logs.records)
        self.assertIn("HTTP response sent (402 Payment Required)", messages)
        self.assertIn("connection rejected (403 Forbidden)", messages)

    def test_process_response_rewrite_forks_from_interception(self):
        """Response rewrite is a separate outlet from pre-accept interception."""
        forced_ids = []
        rewritten_ids = []

        def force(connection, request):
            if request.path == "/teapot":
                forced_ids.append(connection.id)
                return connection.respond(http.HTTPStatus.IM_A_TEAPOT, "teapot\n")
            return None

        def rewrite(connection, _request, response):
            rewritten_ids.append(connection.id)
            headers = response.headers.copy()
            headers["X-Generation"] = "1"
            return dataclasses.replace(response, headers=headers)

        with run_server() as server:
            with server.generation() as generation:
                generation.process_request(force)
                generation.process_response(rewrite)

                # Forced before accept(): 418, then rewritten.
                teapot = self.reject_response(get_uri(server) + "/teapot")
                self.assertEqual(teapot.status_code, 418)
                self.assertEqual(teapot.headers["X-Generation"], "1")

                # Normal accept() path: 101, then rewritten.
                with connect(get_uri(server) + "/no-op", open_timeout=5) as client:
                    self.assertEqual(client.response.headers["X-Generation"], "1")

        # The two outlets land in different ledgers; rewrite spans both.
        self.assertEqual(generation.intercepted, set(forced_ids))
        self.assertEqual(generation.rewritten, set(rewritten_ids))
        self.assertEqual(generation.accepted, set(rewritten_ids) - set(forced_ids))

    def test_shutdown_gate_and_callback_reject_stay_separate(self):
        """503 shutdown gate doesn't collapse with an active callback reject."""
        hold_entered = threading.Event()
        release_hold = threading.Event()
        held_ids = []
        rejected_ids = []

        def intercept(connection, request):
            if request.path == "/hold":
                held_ids.append(connection.id)
                hold_entered.set()
                release_hold.wait()
                return None
            if request.path == "/reject":
                rejected_ids.append(connection.id)
                return connection.respond(http.HTTPStatus.FORBIDDEN, "no\n")
            return None

        with run_server() as server:
            with server.generation() as generation:
                generation.process_request(intercept)

                # C1 is CONNECTING, parked inside the interception intent.
                held = {}
                thread = self.connect_in_background(get_uri(server) + "/hold", held)
                self.assertTrue(hold_entered.wait(5))

                # C2 is rejected explicitly by the callback while serving.
                response = self.reject_response(get_uri(server) + "/reject")
                self.assertEqual(response.status_code, 403)

                # Closing flips only the pending 101 of C1 to 503.
                shutdown_thread = threading.Thread(target=server.shutdown)
                shutdown_thread.start()
                self.assertTrue(server.socket_closed.wait(5))
                release_hold.set()
                thread.join(5)
                shutdown_thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertFalse(shutdown_thread.is_alive())

        self.assertEqual(held["status"], 503)
        # C1 ran accept() (101) before the gate rewrote it to 503; C2 never did.
        self.assertEqual(generation.accepted, set(held_ids))
        self.assertEqual(generation.intercepted, set(rejected_ids))

    def test_nested_generations_do_not_cross_on_same_server(self):
        """Inner generation never touches outer-ledger pending intents."""
        outer_entered = threading.Event()
        release_outer = threading.Event()
        outer_ids = []
        inner_ids = []

        def outer_intercept(connection, request):
            outer_ids.append(connection.id)
            if request.path == "/slow":
                outer_entered.set()
                release_outer.wait()
            return connection.respond(http.HTTPStatus.PAYMENT_REQUIRED, "outer\n")

        with run_server() as server:
            with server.generation() as outer:
                outer.process_request(outer_intercept)

                # An outer connection hangs CONNECTING with its intent pending.
                held = {}
                thread = self.connect_in_background(get_uri(server) + "/slow", held)
                self.assertTrue(outer_entered.wait(5))

                # Exiting the inner generation with an exception must still
                # pop it cleanly.
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    with server.generation() as inner:

                        def inner_intercept(connection, _request):
                            inner_ids.append(connection.id)
                            return connection.respond(
                                http.HTTPStatus.IM_A_TEAPOT, "inner\n"
                            )

                        inner.process_request(inner_intercept)

                        # A new handshake is driven by inner, never by outer.
                        response = self.reject_response(get_uri(server) + "/")
                        self.assertEqual(response.status_code, 418)
                        while inner.booked:  # unbook follows the response.
                            time.sleep(MS)

                        # The outer pending connection isn't on inner's ledger.
                        self.assertEqual(inner.booked, set())
                        self.assertEqual(outer.booked, {outer_ids[0]})

                        raise RuntimeError("boom")

                self.assertTrue(inner.sealed)
                self.assertNotIn(inner, server.handshakes.generations)
                self.assertIs(server.handshakes.current, outer)

                # Outer resumes for new handshakes.
                response = self.reject_response(get_uri(server) + "/")
                self.assertEqual(response.status_code, 402)

                # The held outer intent completes untouched by inner ledger.
                release_outer.set()
                thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertEqual(held["status"], 402)

        self.assertEqual(inner.intercepted, set(inner_ids))
        self.assertEqual(outer.intercepted, set(outer_ids))

    def test_root_generation_binds_serve_level_callbacks(self):
        """serve()-level callbacks live on the root generation and keep working."""

        def root_intercept(connection, _request):
            return connection.respond(http.HTTPStatus.IM_A_TEAPOT, "root\n")

        with run_server(process_request=root_intercept) as server:
            root = server.handshakes.current
            self.assertFalse(root.sealed)
            response = self.reject_response(get_uri(server) + "/")
            self.assertEqual(response.status_code, 418)

            # A fresh generation shadows the root ledger.
            with server.generation() as generation:
                with connect(get_uri(server) + "/no-op", open_timeout=5):
                    pass

            self.assertTrue(generation.sealed)
            with self.assertRaisesRegex(RuntimeError, "sealed generation"):
                generation.process_request(root_intercept)
            with self.assertRaisesRegex(RuntimeError, "sealed generation"):
                generation.process_response(lambda connection, request, r: None)

            # Root ledger resumes.
            response = self.reject_response(get_uri(server) + "/")
            self.assertEqual(response.status_code, 418)
            self.assertIs(server.handshakes.current, root)

    def test_server_defaults_to_empty_root_generation(self):
        """Server creates an empty root generation when none is provided."""

        def sock_handler(sock, addr):
            pass

        with socket.create_server(("localhost", 0)) as sock:
            server = Server(sock, sock_handler)
            self.assertEqual(server.handshakes.current.request_interceptors, [])
            self.assertEqual(server.handshakes.current.response_rewriters, [])
            self.assertFalse(server.handshakes.current.sealed)
