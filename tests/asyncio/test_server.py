import asyncio
import dataclasses
import hmac
import http
import logging
import pathlib
import socket
import unittest

from websockets.asyncio.client import connect, unix_connect
from websockets.asyncio.server import *
from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidHandshake,
    InvalidStatus,
    NegotiationError,
)
from websockets.http11 import Request, Response

from ..utils import (
    CLIENT_CONTEXT,
    MS,
    SERVER_CONTEXT,
    LoggingTestCase,
    temp_unix_socket_path,
)
from .server import (
    EvalShellMixin,
    args,
    get_host_port,
    get_uri,
    handler,
)
from .utils import UVLoopTestCase, requires_accurate_clock


class ServerTests(EvalShellMixin, LoggingTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_connection(self):
        """Server receives connection from client and the handshake succeeds."""
        async with serve(*args) as server:
            async with connect(get_uri(server)) as client:
                await self.assertEval(client, "ws.protocol.state.name", "OPEN")

    async def test_connection_handler_returns(self):
        """Connection handler returns."""
        async with serve(*args) as server:
            async with connect(get_uri(server) + "/no-op") as client:
                with self.assertRaises(ConnectionClosedOK) as raised:
                    await client.recv()

        self.assertEqual(
            str(raised.exception),
            "received 1000 (OK); then sent 1000 (OK)",
        )

    async def test_connection_handler_raises_exception(self):
        """Connection handler raises an exception."""
        async with serve(*args) as server:
            async with connect(get_uri(server) + "/crash") as client:
                with self.assertRaises(ConnectionClosedError) as raised:
                    await client.recv()

        self.assertEqual(
            str(raised.exception),
            "received 1011 (internal error); then sent 1011 (internal error)",
        )

    async def test_existing_socket(self):
        """Server receives connection using a pre-existing socket."""
        with socket.create_server(("localhost", 0)) as sock:
            host, port = sock.getsockname()
            async with serve(handler, sock=sock):
                async with connect(f"ws://{host}:{port}/") as client:
                    await self.assertEval(client, "ws.protocol.state.name", "OPEN")

    async def test_select_subprotocol(self):
        """Server selects a subprotocol with the select_subprotocol callable."""

        def select_subprotocol(ws, subprotocols):
            ws.select_subprotocol_ran = True
            assert "chat" in subprotocols
            return "chat"

        async with serve(
            *args,
            subprotocols=["chat"],
            select_subprotocol=select_subprotocol,
        ) as server:
            async with connect(get_uri(server), subprotocols=["chat"]) as client:
                await self.assertEval(client, "ws.select_subprotocol_ran", "True")
                await self.assertEval(client, "ws.subprotocol", "chat")

    async def test_select_subprotocol_rejects_handshake(self):
        """Server rejects handshake if select_subprotocol raises NegotiationError."""

        def select_subprotocol(ws, subprotocols):
            raise NegotiationError

        async with serve(*args, select_subprotocol=select_subprotocol) as server:
            with self.assertRaises(InvalidStatus) as raised:
                async with connect(get_uri(server)):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 400",
        )

    async def test_select_subprotocol_raises_exception(self):
        """Server returns an error if select_subprotocol raises an exception."""

        def select_subprotocol(ws, subprotocols):
            raise RuntimeError

        async with serve(*args, select_subprotocol=select_subprotocol) as server:
            with self.assertRaises(InvalidStatus) as raised:
                async with connect(get_uri(server)):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 500",
        )

    async def test_compression_is_enabled(self):
        """Server enables compression by default."""
        async with serve(*args) as server:
            async with connect(get_uri(server)) as client:
                await self.assertEval(
                    client,
                    "[type(ext).__name__ for ext in ws.protocol.extensions]",
                    "['PerMessageDeflate']",
                )

    async def test_disable_compression(self):
        """Server disables compression."""
        async with serve(*args, compression=None) as server:
            async with connect(get_uri(server)) as client:
                await self.assertEval(client, "ws.protocol.extensions", "[]")

    async def test_process_request_returns_none(self):
        """Server runs process_request and continues the handshake."""

        def process_request(ws, request):
            self.assertIsInstance(request, Request)
            ws.process_request_ran = True

        async with serve(*args, process_request=process_request) as server:
            async with connect(get_uri(server)) as client:
                await self.assertEval(client, "ws.process_request_ran", "True")

    async def test_async_process_request_returns_none(self):
        """Server runs async process_request and continues the handshake."""

        async def process_request(ws, request):
            self.assertIsInstance(request, Request)
            ws.process_request_ran = True

        async with serve(*args, process_request=process_request) as server:
            async with connect(get_uri(server)) as client:
                await self.assertEval(client, "ws.process_request_ran", "True")

    async def test_process_request_returns_response(self):
        """Server aborts handshake if process_request returns a response."""

        def process_request(ws, request):
            return ws.respond(http.HTTPStatus.FORBIDDEN, "Forbidden")

        async def handler(ws):
            self.fail("handler must not run")

        with self.assertNoLogs("websockets", logging.ERROR):
            async with serve(
                handler, *args[1:], process_request=process_request
            ) as server:
                with self.assertRaises(InvalidStatus) as raised:
                    async with connect(get_uri(server)):
                        self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 403",
        )

    async def test_async_process_request_returns_response(self):
        """Server aborts handshake if async process_request returns a response."""

        async def process_request(ws, request):
            return ws.respond(http.HTTPStatus.FORBIDDEN, "Forbidden")

        async def handler(ws):
            self.fail("handler must not run")

        with self.assertNoLogs("websockets", logging.ERROR):
            async with serve(
                handler, *args[1:], process_request=process_request
            ) as server:
                with self.assertRaises(InvalidStatus) as raised:
                    async with connect(get_uri(server)):
                        self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 403",
        )

    async def test_process_request_raises_exception(self):
        """Server returns an error if process_request raises an exception."""

        def process_request(ws, request):
            raise RuntimeError("BOOM")

        with self.assertLogs("websockets", logging.ERROR) as logs:
            async with serve(*args, process_request=process_request) as server:
                with self.assertRaises(InvalidStatus) as raised:
                    async with connect(get_uri(server)):
                        self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 500",
        )
        self.assertExceptionLogged(logs, "process_request failed", "BOOM")

    async def test_async_process_request_raises_exception(self):
        """Server returns an error if async process_request raises an exception."""

        async def process_request(ws, request):
            raise RuntimeError("BOOM")

        with self.assertLogs("websockets", logging.ERROR) as logs:
            async with serve(*args, process_request=process_request) as server:
                with self.assertRaises(InvalidStatus) as raised:
                    async with connect(get_uri(server)):
                        self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 500",
        )
        self.assertExceptionLogged(logs, "process_request failed", "BOOM")

    async def test_process_response_returns_none(self):
        """Server runs process_response but keeps the handshake response."""

        def process_response(ws, request, response):
            self.assertIsInstance(request, Request)
            self.assertIsInstance(response, Response)
            ws.process_response_ran = True

        async with serve(*args, process_response=process_response) as server:
            async with connect(get_uri(server)) as client:
                await self.assertEval(client, "ws.process_response_ran", "True")

    async def test_async_process_response_returns_none(self):
        """Server runs async process_response but keeps the handshake response."""

        async def process_response(ws, request, response):
            self.assertIsInstance(request, Request)
            self.assertIsInstance(response, Response)
            ws.process_response_ran = True

        async with serve(*args, process_response=process_response) as server:
            async with connect(get_uri(server)) as client:
                await self.assertEval(client, "ws.process_response_ran", "True")

    async def test_process_response_modifies_response(self):
        """Server runs process_response and modifies the handshake response."""

        def process_response(ws, request, response):
            response.headers["X-ProcessResponse"] = "OK"

        async with serve(*args, process_response=process_response) as server:
            async with connect(get_uri(server)) as client:
                self.assertEqual(client.response.headers["X-ProcessResponse"], "OK")

    async def test_async_process_response_modifies_response(self):
        """Server runs async process_response and modifies the handshake response."""

        async def process_response(ws, request, response):
            response.headers["X-ProcessResponse"] = "OK"

        async with serve(*args, process_response=process_response) as server:
            async with connect(get_uri(server)) as client:
                self.assertEqual(client.response.headers["X-ProcessResponse"], "OK")

    async def test_process_response_replaces_response(self):
        """Server runs process_response and replaces the handshake response."""

        def process_response(ws, request, response):
            headers = response.headers.copy()
            headers["X-ProcessResponse"] = "OK"
            return dataclasses.replace(response, headers=headers)

        async with serve(*args, process_response=process_response) as server:
            async with connect(get_uri(server)) as client:
                self.assertEqual(client.response.headers["X-ProcessResponse"], "OK")

    async def test_async_process_response_replaces_response(self):
        """Server runs async process_response and replaces the handshake response."""

        async def process_response(ws, request, response):
            headers = response.headers.copy()
            headers["X-ProcessResponse"] = "OK"
            return dataclasses.replace(response, headers=headers)

        async with serve(*args, process_response=process_response) as server:
            async with connect(get_uri(server)) as client:
                self.assertEqual(client.response.headers["X-ProcessResponse"], "OK")

    async def test_process_response_raises_exception(self):
        """Server returns an error if process_response raises an exception."""

        def process_response(ws, request, response):
            raise RuntimeError("BOOM")

        with self.assertLogs("websockets", logging.ERROR) as logs:
            async with serve(*args, process_response=process_response) as server:
                with self.assertRaises(InvalidStatus) as raised:
                    async with connect(get_uri(server)):
                        self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 500",
        )
        self.assertExceptionLogged(logs, "process_response failed", "BOOM")

    async def test_async_process_response_raises_exception(self):
        """Server returns an error if async process_response raises an exception."""

        async def process_response(ws, request, response):
            raise RuntimeError("BOOM")

        with self.assertLogs("websockets", logging.ERROR) as logs:
            async with serve(*args, process_response=process_response) as server:
                with self.assertRaises(InvalidStatus) as raised:
                    async with connect(get_uri(server)):
                        self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 500",
        )
        self.assertExceptionLogged(logs, "process_response failed", "BOOM")

    async def test_override_server_header(self):
        """Server can override Server header with server_header."""
        async with serve(*args, server_header="Neo") as server:
            async with connect(get_uri(server)) as client:
                await self.assertEval(client, "ws.response.headers['Server']", "Neo")

    async def test_remove_server_header(self):
        """Server can remove Server header with server_header."""
        async with serve(*args, server_header=None) as server:
            async with connect(get_uri(server)) as client:
                await self.assertEval(
                    client, "'Server' in ws.response.headers", "False"
                )

    @requires_accurate_clock
    async def test_keepalive_is_enabled(self):
        """Server enables keepalive and measures latency."""
        async with serve(*args, ping_interval=MS) as server:
            async with connect(get_uri(server)) as client:
                await client.send("ws.latency")
                latency = eval(await client.recv())
                self.assertEqual(latency, 0)
                await asyncio.sleep(2 * MS)
                await client.send("ws.latency")
                latency = eval(await client.recv())
                self.assertGreater(latency, 0)

    @requires_accurate_clock
    async def test_disable_keepalive(self):
        """Server disables keepalive."""
        async with serve(*args, ping_interval=None) as server:
            async with connect(get_uri(server)) as client:
                await asyncio.sleep(2 * MS)
                await client.send("ws.latency")
                latency = eval(await client.recv())
                self.assertEqual(latency, 0)

    async def test_logger(self):
        """Server accepts a logger argument."""
        logger = logging.getLogger("test")
        async with serve(*args, logger=logger) as server:
            self.assertEqual(server.logger.name, logger.name)

    async def test_custom_connection_factory(self):
        """Server runs ServerConnection factory provided in create_connection."""

        def create_connection(*args, **kwargs):
            server = ServerConnection(*args, **kwargs)
            server.create_connection_ran = True
            return server

        async with serve(*args, create_connection=create_connection) as server:
            async with connect(get_uri(server)) as client:
                await self.assertEval(client, "ws.create_connection_ran", "True")

    async def test_connections(self):
        """Server provides a connections property."""
        async with serve(*args) as server:
            self.assertEqual(server.connections, set())
            async with (
                connect(get_uri(server)) as client1,
                connect(get_uri(server)) as client2,
            ):
                await client1.send("ws.id")
                await client2.send("ws.id")
                self.assertEqual(
                    {str(connection.id) for connection in server.connections},
                    {await client1.recv(), await client2.recv()},
                )
            self.assertEqual(server.connections, set())

    async def test_handshake_fails(self):
        """Server receives connection from client but the handshake fails."""

        def remove_key_header(self, request):
            del request.headers["Sec-WebSocket-Key"]

        async with serve(*args, process_request=remove_key_header) as server:
            with self.assertRaises(InvalidStatus) as raised:
                async with connect(get_uri(server)):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 400",
        )

    async def test_timeout_before_handshake_request(self):
        """Server times out before receiving handshake request from client."""
        with self.assertLogs("websockets", logging.DEBUG) as logs:
            async with serve(*args, open_timeout=MS) as server:
                reader, writer = await asyncio.open_connection(*get_host_port(server))
                try:
                    # Wait for the server to close the connection.
                    self.assertEqual(await reader.read(4096), b"")
                finally:
                    writer.close()

        self.assertExceptionLogged(
            logs,
            "! no valid HTTP request",
            "connection closed while reading HTTP request line",
        )

    async def test_connection_closed_before_handshake_request(self):
        """Server reads EOF before receiving handshake request from client."""
        with self.assertLogs("websockets", logging.DEBUG) as logs:
            async with serve(*args) as server:
                _reader, writer = await asyncio.open_connection(*get_host_port(server))
                writer.close()

        self.assertExceptionLogged(
            logs,
            "! no valid HTTP request",
            "connection closed while reading HTTP request line",
        )

    async def test_connection_closed_before_handshake_response(self):
        """Server reads EOF before sending handshake response to client."""

        async def process_request(ws, _request):
            ws.transport.close()
            await asyncio.sleep(0)

        with self.assertNoLogs("websockets", logging.ERROR):
            async with serve(*args, process_request=process_request) as server:
                with self.assertRaises(InvalidHandshake):
                    async with connect(get_uri(server)):
                        self.fail("did not raise")

    async def test_junk_handshake_request(self):
        """Server closes the connection when receiving non-HTTP request from client."""
        with self.assertLogs("websockets", logging.DEBUG) as logs:
            async with serve(*args) as server:
                reader, writer = await asyncio.open_connection(*get_host_port(server))
                writer.write(b"HELO relay.invalid\r\n")
                try:
                    # Wait for the server to close the connection.
                    self.assertEqual(await reader.read(4096), b"")
                finally:
                    writer.close()

        self.assertExceptionLogged(
            logs,
            "! no valid HTTP request",
            "invalid HTTP request line: HELO relay.invalid",
        )

    async def test_close_stops_accepting_connections(self):
        """Server stops accepting connections."""
        async with serve(*args) as server:
            # Connect a client and disable close_connections to delay closing.
            uri = get_uri(server)
            async with connect(uri):
                server.close(close_connections=False)
                await asyncio.sleep(0)

                # Server cannot receive new connections.
                with self.assertRaises(OSError):
                    async with connect(uri):
                        self.fail("did not raise")

    async def test_close_rejects_connecting_connections(self):
        """Server rejects connecting connections with HTTP 503."""

        async def process_request(ws, _request):
            while ws.server.is_serving():
                await asyncio.sleep(0)  # pragma: no cover

        async with serve(*args, process_request=process_request) as server:
            asyncio.get_running_loop().call_later(2 * MS, server.close)
            with self.assertRaises(InvalidStatus) as raised:
                async with connect(get_uri(server)):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 503",
        )

    async def test_close_closes_open_connections(self):
        """Server closes open connections with close code 1001."""
        async with serve(*args) as server:
            async with connect(get_uri(server)) as client:
                server.close()
                with self.assertRaises(ConnectionClosedOK) as raised:
                    await client.recv()

        self.assertEqual(
            str(raised.exception),
            "received 1001 (going away); then sent 1001 (going away)",
        )

    async def test_close_closes_open_connections_with_code_and_reason(self):
        """Server closes open connections with custom code and reason."""
        async with serve(*args) as server:
            async with connect(get_uri(server)) as client:
                server.close(code=1012, reason="restarting")
                with self.assertRaises(ConnectionClosedError) as raised:
                    await client.recv()

        self.assertEqual(
            str(raised.exception),
            "received 1012 (service restart) restarting; "
            "then sent 1012 (service restart) restarting",
        )

    async def test_close_keeps_connections_open(self):
        """Server waits for the client to close open connections."""
        async with serve(*args) as server:
            async with connect(get_uri(server)):
                server.close(close_connections=False)

                # The server waits for the client to close the connection.
                with self.assertRaises(TimeoutError):
                    async with asyncio.timeout(MS):
                        await server.wait_closed()

            # Once the client closes the connection, the server terminates.
            async with asyncio.timeout(MS):
                await server.wait_closed()

    async def test_close_keeps_handlers_running(self):
        """Server waits for connection handlers to terminate."""
        async with serve(*args) as server:
            async with connect(get_uri(server) + "/delay") as client:
                # Delay termination of connection handler.
                await client.send(str(3 * MS))

                server.close()

                # The server waits for the connection handler to terminate.
                with self.assertRaises(TimeoutError):
                    async with asyncio.timeout(2 * MS):
                        await server.wait_closed()

                # Set a longer timeout here, else the test becomes flaky.
                async with asyncio.timeout(5 * MS):
                    await server.wait_closed()

    async def test_get_loop(self):
        """Server provides a get_loop() method."""
        async with serve(*args) as server:
            self.assertIs(server.get_loop(), asyncio.get_running_loop())

    async def test_start_serving(self):
        """Server starts serving when start_serving() is called."""
        server = await serve(*args, start_serving=False)
        try:
            self.assertFalse(server.is_serving())
            await server.start_serving()
            self.assertTrue(server.is_serving())
        finally:
            server.close()
            await server.wait_closed()

    async def test_serve_forever(self):
        """Server runs until canceled when serve_forever() is called."""
        server = await serve(*args)
        try:
            serve_forever_task = asyncio.get_running_loop().create_task(
                server.serve_forever()
            )
            await asyncio.sleep(0)

            self.assertTrue(server.is_serving())

            serve_forever_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await serve_forever_task

            self.assertFalse(server.is_serving())

        finally:
            server.close()
            await server.wait_closed()

    async def test_serve_forever_starts_serving(self):
        """Server starts serving when serve_forever() is called."""
        server = await serve(*args, start_serving=False)
        try:
            self.assertFalse(server.is_serving())

            serve_forever_task = asyncio.get_running_loop().create_task(
                server.serve_forever()
            )
            await asyncio.sleep(0)

            self.assertTrue(server.is_serving())

            serve_forever_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await serve_forever_task

        finally:
            server.close()
            await server.wait_closed()

    async def test_canceling_serve_forever_closes_open_connections(self):
        """Canceling serve_forever() closes open connections with code 1001."""
        server = await serve(*args)
        try:
            serve_forever_task = asyncio.get_running_loop().create_task(
                server.serve_forever()
            )
            await asyncio.sleep(0)

            async with connect(get_uri(server)) as client:
                serve_forever_task.cancel()
                with self.assertRaises(ConnectionClosedOK) as raised:
                    await client.recv()

            with self.assertRaises(asyncio.CancelledError):
                await serve_forever_task

        finally:
            server.close()
            await server.wait_closed()

        self.assertEqual(
            str(raised.exception),
            "received 1001 (going away); then sent 1001 (going away)",
        )

    async def test_close_exits_serve_forever(self):
        """Closing the server exits serve_forever()."""
        server = await serve(*args)
        try:
            serve_forever_task = asyncio.get_running_loop().create_task(
                server.serve_forever()
            )
            await asyncio.sleep(0)
            server.close()

            async with asyncio.timeout(MS):
                await serve_forever_task

        finally:
            server.close()
            await server.wait_closed()

    async def test_is_serving(self):
        """Server provides an is_serving() method."""
        async with serve(*args) as server:
            self.assertTrue(server.is_serving())
        self.assertFalse(server.is_serving())

    async def test_wait_closed(self):
        """Server provides a wait_closed() method."""
        async with serve(*args) as server:
            wait_closed = asyncio.create_task(server.wait_closed())
            await asyncio.sleep(0)
            self.assertFalse(wait_closed.done())
        self.assertTrue(wait_closed.done())

    async def test_sockets(self):
        """Server provides a sockets property."""
        async with serve(*args) as server:
            sock = server.sockets[0]
            self.assertIsInstance(sock.fileno(), int)
        self.assertEqual(list(server.sockets), [])

    async def test_context_manager_closes_server(self):
        """Server closes when exiting the asynchronous context manager."""
        async with serve(handler, "localhost", 0) as server:
            self.assertTrue(server.is_serving())

        self.assertFalse(server.is_serving())

    async def test_context_manager_with_await_closes_server(self):
        """Server closes when exiting the asynchronous context manager after await."""
        # This is redundant, but has been historically supported.
        server = await serve(handler, "localhost", 0)
        async with server:
            self.assertTrue(server.is_serving())

        self.assertFalse(server.is_serving())


SSL_OBJECT = "ws.transport.get_extra_info('ssl_object')"


class SecureServerTests(EvalShellMixin, unittest.IsolatedAsyncioTestCase):
    async def test_connection(self):
        """Server receives secure connection from client."""
        async with serve(*args, ssl=SERVER_CONTEXT) as server:
            async with connect(get_uri(server), ssl=CLIENT_CONTEXT) as client:
                await self.assertEval(client, "ws.protocol.state.name", "OPEN")
                await self.assertEval(client, f"{SSL_OBJECT}.version()[:3]", "TLS")

    async def test_timeout_during_tls_handshake(self):
        """Server times out before receiving TLS handshake request from client."""
        async with serve(*args, ssl=SERVER_CONTEXT, open_timeout=MS) as server:
            reader, writer = await asyncio.open_connection(*get_host_port(server))
            try:
                # Wait for the server to close the connection.
                self.assertEqual(await reader.read(4096), b"")
            finally:
                writer.close()

    async def test_connection_closed_during_tls_handshake(self):
        """Server reads EOF before receiving TLS handshake request from client."""
        async with serve(*args, ssl=SERVER_CONTEXT) as server:
            _reader, writer = await asyncio.open_connection(*get_host_port(server))
            writer.close()


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "this test requires Unix sockets")
class UnixServerTests(EvalShellMixin, unittest.IsolatedAsyncioTestCase):
    async def test_connection(self):
        """Server receives connection from client over a Unix socket."""
        with temp_unix_socket_path() as path:
            async with unix_serve(handler, path):
                async with unix_connect(path) as client:
                    await self.assertEval(client, "ws.protocol.state.name", "OPEN")

    async def test_pathlib_path(self):
        """Server accepts a pathlib.Path object as the path argument."""
        with temp_unix_socket_path() as path:
            async with unix_serve(handler, pathlib.Path(path)):
                async with unix_connect(path) as client:
                    await self.assertEval(client, "ws.protocol.state.name", "OPEN")


@unittest.skipUnless(hasattr(socket, "AF_UNIX"), "this test requires Unix sockets")
class SecureUnixServerTests(EvalShellMixin, unittest.IsolatedAsyncioTestCase):
    async def test_connection(self):
        """Server receives secure connection from client over a Unix socket."""
        with temp_unix_socket_path() as path:
            async with unix_serve(handler, path, ssl=SERVER_CONTEXT):
                async with unix_connect(path, ssl=CLIENT_CONTEXT) as client:
                    await self.assertEval(client, "ws.protocol.state.name", "OPEN")
                    await self.assertEval(client, f"{SSL_OBJECT}.version()[:3]", "TLS")


class ServerUsageErrorsTests(unittest.IsolatedAsyncioTestCase):
    async def test_host_and_sock(self):
        """Server rejects host when sock is provided."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(sock.close)
        with self.assertRaises(ValueError) as raised:
            await serve(handler, host="localhost", sock=sock)

        self.assertEqual(
            str(raised.exception),
            "host/port and sock can not be specified at the same time",
        )

    async def test_port_and_sock(self):
        """Server rejects port when sock is provided."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(sock.close)
        with self.assertRaises(ValueError) as raised:
            await serve(handler, port=0, sock=sock)

        self.assertEqual(
            str(raised.exception),
            "host/port and sock can not be specified at the same time",
        )

    async def test_unix_without_path_or_sock(self):
        """Unix server requires path when sock isn't provided."""
        with self.assertRaises(ValueError) as raised:
            await unix_serve(handler)

        self.assertEqual(
            str(raised.exception),
            "path was not specified, and no sock specified",
        )

    async def test_unix_with_path_and_sock(self):
        """Unix server rejects path when sock is provided."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(sock.close)
        with self.assertRaises(ValueError) as raised:
            await unix_serve(handler, path="/", sock=sock)

        self.assertEqual(
            str(raised.exception),
            "path and sock can not be specified at the same time",
        )

    async def test_invalid_subprotocol(self):
        """Server rejects single value of subprotocols."""
        with self.assertRaises(TypeError) as raised:
            await serve(*args, subprotocols="chat")

        self.assertEqual(
            str(raised.exception),
            "subprotocols must be a list, not a str",
        )

    async def test_unsupported_compression(self):
        """Server rejects incorrect value of compression."""
        with self.assertRaises(ValueError) as raised:
            await serve(*args, compression=False)

        self.assertEqual(
            str(raised.exception),
            "unsupported compression: False",
        )


class BasicAuthTests(EvalShellMixin, unittest.IsolatedAsyncioTestCase):
    async def test_valid_authorization(self):
        """basic_auth authenticates client."""
        async with serve(
            *args,
            process_request=basic_auth(credentials=("hello", "iloveyou")),
        ) as server:
            async with connect(
                get_uri(server),
                additional_headers={"Authorization": "Basic aGVsbG86aWxvdmV5b3U="},
            ) as client:
                await self.assertEval(client, "ws.username", "hello")

    async def test_valid_authorization_non_ascii_password(self):
        """basic_auth authenticates client with non-ASCII password."""
        async with serve(
            *args,
            process_request=basic_auth(credentials=("test", "123£")),
        ) as server:
            async with connect(
                get_uri(server),
                additional_headers={"Authorization": "Basic dGVzdDoxMjPCow=="},
            ) as client:
                await self.assertEval(client, "ws.username", "test")

    async def test_valid_authorization_non_utf8_password(self):
        """basic_auth authenticates client with non-UTF-8 password."""
        password = b"\xca\xfe".decode(errors="surrogateescape")
        async with serve(
            *args,
            process_request=basic_auth(credentials=("drink", password)),
        ) as server:
            async with connect(
                get_uri(server),
                additional_headers={"Authorization": "Basic ZHJpbms6yv4="},
            ) as client:
                await self.assertEval(client, "ws.username", "drink")

    async def test_missing_authorization(self):
        """basic_auth rejects client without credentials."""
        async with serve(
            *args,
            process_request=basic_auth(credentials=("hello", "iloveyou")),
        ) as server:
            with self.assertRaises(InvalidStatus) as raised:
                async with connect(get_uri(server)):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 401",
        )

    async def test_unsupported_authorization(self):
        """basic_auth rejects client with unsupported credentials."""
        async with serve(
            *args,
            process_request=basic_auth(credentials=("hello", "iloveyou")),
        ) as server:
            with self.assertRaises(InvalidStatus) as raised:
                async with connect(
                    get_uri(server),
                    additional_headers={"Authorization": "Negotiate ..."},
                ):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 401",
        )

    async def test_authorization_with_unknown_username(self):
        """basic_auth rejects client with unknown username."""
        async with serve(
            *args,
            process_request=basic_auth(credentials=("hello", "iloveyou")),
        ) as server:
            with self.assertRaises(InvalidStatus) as raised:
                async with connect(
                    get_uri(server),
                    additional_headers={"Authorization": "Basic YnllOnlvdWxvdmVtZQ=="},
                ):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 401",
        )

    async def test_authorization_with_incorrect_password(self):
        """basic_auth rejects client with incorrect password."""
        async with serve(
            *args,
            process_request=basic_auth(credentials=("hello", "changeme")),
        ) as server:
            with self.assertRaises(InvalidStatus) as raised:
                async with connect(
                    get_uri(server),
                    additional_headers={"Authorization": "Basic aGVsbG86aWxvdmV5b3U="},
                ):
                    self.fail("did not raise")

        self.assertEqual(
            str(raised.exception),
            "server rejected WebSocket connection: HTTP 401",
        )

    async def test_list_of_credentials(self):
        """basic_auth accepts a list of hard coded credentials."""
        async with serve(
            *args,
            process_request=basic_auth(
                credentials=[
                    ("hello", "iloveyou"),
                    ("bye", "youloveme"),
                ]
            ),
        ) as server:
            async with connect(
                get_uri(server),
                additional_headers={"Authorization": "Basic YnllOnlvdWxvdmVtZQ=="},
            ) as client:
                await self.assertEval(client, "ws.username", "bye")

    async def test_check_credentials_function(self):
        """basic_auth accepts a check_credentials function."""

        def check_credentials(username, password):
            self.assertEqual(username, "hello")
            return hmac.compare_digest(password, "iloveyou")

        async with serve(
            *args,
            process_request=basic_auth(check_credentials=check_credentials),
        ) as server:
            async with connect(
                get_uri(server),
                additional_headers={"Authorization": "Basic aGVsbG86aWxvdmV5b3U="},
            ) as client:
                await self.assertEval(client, "ws.username", "hello")

    async def test_check_credentials_coroutine(self):
        """basic_auth accepts a check_credentials coroutine."""

        async def check_credentials(username, password):
            self.assertEqual(username, "hello")
            return hmac.compare_digest(password, "iloveyou")

        async with serve(
            *args,
            process_request=basic_auth(check_credentials=check_credentials),
        ) as server:
            async with connect(
                get_uri(server),
                additional_headers={"Authorization": "Basic aGVsbG86aWxvdmV5b3U="},
            ) as client:
                await self.assertEval(client, "ws.username", "hello")

    async def test_without_credentials_or_check_credentials(self):
        """basic_auth requires either credentials or check_credentials."""
        with self.assertRaises(ValueError) as raised:
            basic_auth()

        self.assertEqual(
            str(raised.exception),
            "provide either credentials or check_credentials",
        )

    async def test_with_credentials_and_check_credentials(self):
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

    async def test_bad_credentials(self):
        """basic_auth receives an unsupported credentials argument."""
        with self.assertRaises(TypeError) as raised:
            basic_auth(credentials=42)

        self.assertEqual(
            str(raised.exception),
            "invalid credentials argument: 42",
        )

    async def test_bad_list_of_credentials(self):
        """basic_auth receives an unsupported credentials argument."""
        with self.assertRaises(TypeError) as raised:
            basic_auth(credentials=[42])

        self.assertEqual(
            str(raised.exception),
            "invalid credentials argument: [42]",
        )


class UVLoopServerTests(ServerTests, UVLoopTestCase):
    pass


class UVLoopSecureServerTests(SecureServerTests, UVLoopTestCase):
    pass


class UVLoopUnixServerTests(UnixServerTests, UVLoopTestCase):
    pass


class UVLoopSecureUnixServerTests(SecureUnixServerTests, UVLoopTestCase):
    pass


class UVLoopServerUsageErrorsTests(ServerUsageErrorsTests, UVLoopTestCase):
    pass


class UVLoopBasicAuthTests(BasicAuthTests, UVLoopTestCase):
    pass


class HandshakeGenerationTests(LoggingTestCase, unittest.IsolatedAsyncioTestCase):
    """
    Opening-handshake interception/rewrite generations.

    These tests observe causality across connections of ONE server: an intent
    bound to generation N stops driving new CONNECTING handshakes once N+1 is
    pushed; pre-accept interception, accept() (including rejects), response
    rewriting, and the shutdown 503 gate remain separate outlets.

    """

    async def assert_rejects(self, uri, status_code, **kwargs):
        with self.assertRaises(InvalidStatus) as raised:
            async with connect(uri, open_timeout=5, **kwargs):
                self.fail("did not reject")
        self.assertEqual(raised.exception.response.status_code, status_code)
        return raised.exception.response

    async def test_generation_advance_cuts_off_new_handshakes(self):
        """Intents of generation N don't drive new handshakes after advancing."""
        hold_entered = asyncio.Event()
        release_hold = asyncio.Event()
        gen1_ids = []

        async with serve(*args) as server:
            async with server.generation() as gen1:

                async def intercept(connection, request):
                    gen1_ids.append(connection.id)
                    if request.path == "/hold":
                        hold_entered.set()
                        await release_hold.wait()
                    return connection.respond(
                        http.HTTPStatus.PAYMENT_REQUIRED, "gen1\n"
                    )

                gen1.process_request(intercept)

                # C1 starts its handshake in gen1 and stays CONNECTING there.
                held = asyncio.ensure_future(
                    self.assert_rejects(get_uri(server) + "/hold", 402)
                )
                await hold_entered.wait()
                self.assertEqual(gen1.booked, {gen1_ids[0]})

                # Advance to gen2: fresh ledger, no inherited interception.
                async with server.generation() as gen2:
                    self.assertIs(server.handshakes.current, gen2)
                    async with connect(get_uri(server) + "/no-op", open_timeout=5):
                        server_ids = {
                            connection.id for connection in server.all_connections
                        }
                        # New handshake went through accept() on gen2, not gen1.
                        self.assertEqual(gen2.accepted, server_ids - {gen1_ids[0]})
                    self.assertEqual(gen1.booked, {gen1_ids[0]})

                # gen2 popped: gen1 resumes for new handshakes.
                self.assertTrue(gen2.sealed)
                self.assertIs(server.handshakes.current, gen1)
                await self.assert_rejects(get_uri(server) + "/resumed", 402)

                # The held connection stays on gen1's ledger and completes with
                # gen1's intent after the inner generation came and went.
                release_hold.set()
                await held

        self.assertEqual(gen1.intercepted, set(gen1_ids))
        self.assertNotIn(gen1_ids[0], gen2.intercepted)

    async def test_intercept_and_accept_reject_stay_separate(self):
        """Forced response without accept() vs reject after accept() fork."""
        seen = []

        async def record(connection, request):
            seen.append((connection, request.path))
            return None

        async def force(connection, request):
            if request.path == "/forced":
                return connection.respond(http.HTTPStatus.PAYMENT_REQUIRED, "forced\n")
            return None

        with self.assertLogs("websockets", logging.INFO) as logs:
            async with serve(*args, origins=[]) as server:
                async with server.generation() as generation:
                    generation.process_request(record)
                    generation.process_request(force)

                    # Host interception forces 402 before accept().
                    await self.assert_rejects(get_uri(server) + "/forced", 402)
                    # No acceptable Origin: interception returns None, then
                    # accept() rejects with 403.
                    await self.assert_rejects(get_uri(server) + "/", 403)

        forced_ws, _ = seen[0]
        rejected_ws, _ = seen[1]
        self.assertFalse(forced_ws.protocol.accept_called)
        self.assertTrue(rejected_ws.protocol.accept_called)
        self.assertEqual(generation.intercepted, {forced_ws.id})
        self.assertEqual(generation.accepted, {rejected_ws.id})
        messages = "".join(record.getMessage() for record in logs.records)
        self.assertIn("HTTP response sent (402 Payment Required)", messages)
        self.assertIn("connection rejected (403 Forbidden)", messages)

    async def test_process_response_rewrite_forks_from_interception(self):
        """Response rewrite is a separate outlet from pre-accept interception."""
        forced_ids = []
        rewritten_ids = []

        async def force(connection, request):
            if request.path == "/teapot":
                forced_ids.append(connection.id)
                return connection.respond(http.HTTPStatus.IM_A_TEAPOT, "teapot\n")
            return None

        def rewrite(connection, _request, response):
            # A synchronous rewrite intent covers the sync callback pin.
            rewritten_ids.append(connection.id)
            headers = response.headers.copy()
            headers["X-Generation"] = "1"
            return dataclasses.replace(response, headers=headers)

        async with serve(*args) as server:
            async with server.generation() as generation:
                generation.process_request(force)
                generation.process_response(rewrite)

                # Forced before accept(): 418, then rewritten.
                teapot = await self.assert_rejects(get_uri(server) + "/teapot", 418)
                self.assertEqual(teapot.headers["X-Generation"], "1")

                # Normal accept() path: 101, then rewritten.
                async with connect(get_uri(server) + "/no-op") as client:
                    self.assertEqual(client.response.headers["X-Generation"], "1")

        # The two outlets land in different ledgers; rewrite spans both.
        self.assertEqual(generation.intercepted, set(forced_ids))
        self.assertEqual(generation.rewritten, set(rewritten_ids))
        self.assertEqual(generation.accepted, set(rewritten_ids) - set(forced_ids))

    async def test_shutdown_gate_and_callback_reject_stay_separate(self):
        """503 shutdown gate doesn't collapse with an active callback reject."""
        hold_entered = asyncio.Event()
        release_hold = asyncio.Event()
        held_ids = []
        rejected_ids = []

        async def intercept(connection, request):
            if request.path == "/hold":
                held_ids.append(connection.id)
                hold_entered.set()
                await release_hold.wait()
                return None
            if request.path == "/reject":
                rejected_ids.append(connection.id)
                return connection.respond(http.HTTPStatus.FORBIDDEN, "no\n")
            return None

        async with serve(*args) as server:
            async with server.generation() as generation:
                generation.process_request(intercept)

                # C1 is CONNECTING, parked inside the interception intent.
                held = asyncio.ensure_future(
                    self.assert_rejects(get_uri(server) + "/hold", 503)
                )
                await hold_entered.wait()

                # C2 is rejected explicitly by the callback while serving.
                await self.assert_rejects(get_uri(server) + "/reject", 403)

                # Closing flips only the pending 101 of C1 to 503.
                server.close()
                while server.is_serving():
                    await asyncio.sleep(MS)
                release_hold.set()
                held_response = await held

        self.assertEqual(held_response.status_code, 503)
        # C1 ran accept() (101) before the gate rewrote it to 503; C2 never did.
        self.assertEqual(generation.accepted, set(held_ids))
        self.assertEqual(generation.intercepted, set(rejected_ids))

    async def test_nested_generations_do_not_cross_on_same_server(self):
        """Inner generation never touches outer-ledger pending intents."""
        outer_entered = asyncio.Event()
        release_outer = asyncio.Event()
        outer_ids = []
        inner_ids = []

        async with serve(*args) as server:
            async with server.generation() as outer:

                async def outer_intercept(connection, request):
                    outer_ids.append(connection.id)
                    if request.path == "/slow":
                        outer_entered.set()
                        await release_outer.wait()
                    return connection.respond(
                        http.HTTPStatus.PAYMENT_REQUIRED, "outer\n"
                    )

                outer.process_request(outer_intercept)

                # An outer connection hangs CONNECTING with its intent pending.
                held = asyncio.ensure_future(
                    self.assert_rejects(get_uri(server) + "/slow", 402)
                )
                await outer_entered.wait()

                # Exiting the inner generation with an exception must still
                # pop it cleanly.
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    async with server.generation() as inner:

                        def inner_intercept(connection, _request):
                            inner_ids.append(connection.id)
                            return connection.respond(
                                http.HTTPStatus.IM_A_TEAPOT, "inner\n"
                            )

                        inner.process_request(inner_intercept)

                        # A new handshake is driven by inner, never by outer.
                        await self.assert_rejects(get_uri(server) + "/", 418)
                        while inner.booked:  # unbook follows the response.
                            await asyncio.sleep(MS)

                        # The outer pending connection isn't on inner's ledger.
                        self.assertEqual(inner.booked, set())
                        self.assertEqual(outer.booked, {outer_ids[0]})

                        raise RuntimeError("boom")

                self.assertTrue(inner.sealed)
                self.assertNotIn(inner, server.handshakes.generations)
                self.assertIs(server.handshakes.current, outer)

                # Outer resumes for new handshakes.
                await self.assert_rejects(get_uri(server) + "/", 402)

                # The held outer intent completes untouched by the inner ledger.
                release_outer.set()
                held_response = await held
                self.assertEqual(held_response.status_code, 402)

        self.assertEqual(inner.intercepted, set(inner_ids))
        self.assertEqual(outer.intercepted, set(outer_ids))

    async def test_root_generation_binds_serve_level_callbacks(self):
        """serve()-level callbacks live on the root generation and keep working."""

        async def root_intercept(connection, _request):
            return connection.respond(http.HTTPStatus.IM_A_TEAPOT, "root\n")

        async with serve(*args, process_request=root_intercept) as server:
            root = server.handshakes.current
            self.assertFalse(root.sealed)
            await self.assert_rejects(get_uri(server) + "/", 418)

            # A fresh generation shadows the root ledger.
            async with server.generation() as generation:
                async with connect(get_uri(server) + "/no-op", open_timeout=5):
                    pass

            self.assertTrue(generation.sealed)
            with self.assertRaisesRegex(RuntimeError, "sealed generation"):
                generation.process_request(root_intercept)
            with self.assertRaisesRegex(RuntimeError, "sealed generation"):
                generation.process_response(lambda connection, request, r: None)

            # Root ledger resumes.
            await self.assert_rejects(get_uri(server) + "/", 418)
            self.assertIs(server.handshakes.current, root)

    async def test_server_defaults_to_empty_root_generation(self):
        """Server creates an empty root generation when none is provided."""

        async def noop_handler(connection):
            pass

        server = Server(lambda: asyncio.sleep(0), noop_handler)
        self.assertEqual(server.handshakes.current.request_interceptors, [])
        self.assertEqual(server.handshakes.current.response_rewriters, [])
        self.assertFalse(server.handshakes.current.sealed)
