from socketserver import TCPServer, StreamRequestHandler
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mario_env = None


def start_server():
    logger.info("Starting server...")
    server = TCPServer(('0.0.0.0', 36296), TCPHandler)

    print("Listening on Port: {}".format(server.server_address[1]))
    sys.stdout.flush()
    server.serve_forever()


class TCPHandler(StreamRequestHandler):
    def handle(self):
        logger.info("Handling a new connection")
        for line in self.rfile:
            message = str(line.strip(), 'utf-8')
            logger.debug("recvd message: {}".format(message))

            if message.startswith("STARTED"):
                logger.info("Game started")
                mario_env.game_started()

            if message.startswith("MESSAGE"):
                parsed = message.split(":")
                logger.info("splitted message: {}".format(parsed))

                screenshot_path = parsed[1]
                reward = parsed[3]
                done = parsed[5]

                if done == "False":
                    done = False
                else:
                    done = True

                logger.info("Message received: screen_shot: {}, reward: {}, done:{}".format(screenshot_path, reward, done))
                mario_env.action_response(screenshot_path, reward, done)
