import zmq

context = zmq.Context()

def create_pull(port):
    socket = context.socket(zmq.PULL)
    socket.connect(f"tcp://localhost:{port}")
    return socket

def create_push(port):
    socket = context.socket(zmq.PUSH)
    socket.bind(f"tcp://*:{port}")
    return socket
