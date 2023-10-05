import signal
import redis
import queue
import threading
import msgpack
from utils.logger import logger


class SignalHandler:
    def __init__(self, redis_client: redis.Redis, queue_name: str, terminate_event: threading.Event):
        self.redis_client = redis_client
        self.queue_name = queue_name
        self.message_queue = queue.Queue()
        self.terminate_event = terminate_event

    def set_terminate_event(self):
        self.terminate_event.set()

    def enqueue_message(self, message):
        self.message_queue.put(message)

    def dequeue_message(self):
        self.message_queue.get(timeout=1)

    def put_messages_back_to_queue(self):
        logger.info("putting messages back to queue")
        while self.message_queue.qsize() > 0:
            message = self.message_queue.get(timeout=1)
            logger.info(f"putting message back to queue: {message}")
            self.redis_client.lpush(self.queue_name, message)
            self.message_queue.task_done()
        logger.info("messages put back to queue successfully")

    def sigterm_handler(self, signum, frame):
        logger.info("SIGTERM received")
        self.set_terminate_event()
        self.put_messages_back_to_queue()
        logger.info("messages put back to queue")
        exit(0)

    def sigkill_handler(self, signum, frame):
        logger.info("SIGKILL received")
        self.set_terminate_event()
        self.put_messages_back_to_queue()
        logger.info("messages put back to queue")
        exit(0)

    def sigint_handler(self, signum, frame):
        logger.info("SIGINT received")
        self.set_terminate_event()
        exit(0)

if __name__ == "__main__":
    # Test SignalHandler
    # Create queues for communication between threads
    from config import conf
    import os

    logger.info("Pid: %s", os.getpid())
    q1 = queue.Queue(conf.MAX_QUEUE_SIZE)
    q2 = queue.Queue(conf.MAX_QUEUE_SIZE)
    
    # Redis Client
    pool = redis.ConnectionPool(
        host=conf.REDIS_HOST,
        port=conf.REDIS_PORT,
        db=conf.REDIS_DB,
        password=conf.REDIS_PASSWORD,
        username=conf.REDIS_USERNAME,
    )

    redis_client = redis.Redis(connection_pool=pool)

    # Event to signal termination to the inference thread
    terminate_event = threading.Event()

    # Signal Handler
    signal_handler = SignalHandler(redis_client, conf.REDIS_QUEUE_NAME, terminate_event)

    # Register the Ctrl+C signal handler
    signal.signal(signal.SIGINT, signal_handler.sigint_handler)
    signal.signal(signal.SIGTERM, signal_handler.sigterm_handler)

    # Test enqueue and dequeue
    message = msgpack.packb("message")
    signal_handler.enqueue_message(message)
    assert signal_handler.message_queue.qsize() == 1
    assert signal_handler.message_queue.get(timeout=1) == message
    assert signal_handler.message_queue.qsize() == 0
    
    # Test put_messages_back_to_queue
    signal_handler.enqueue_message(message)
    signal_handler.put_messages_back_to_queue()
    assert redis_client.llen(conf.REDIS_QUEUE_NAME) == 1
    assert redis_client.lpop(conf.REDIS_QUEUE_NAME) == message
    assert redis_client.llen(conf.REDIS_QUEUE_NAME) == 0

    # Test set_terminate_event
    signal_handler.set_terminate_event()
    assert signal_handler.terminate_event.is_set() == True
    signal_handler.terminate_event.clear()
    assert signal_handler.terminate_event.is_set() == False
    signal_handler.set_terminate_event()
    assert signal_handler.terminate_event.is_set() == True

    # Test dequeue_message
    signal_handler.enqueue_message(message)
    signal_handler.dequeue_message()
    assert signal_handler.message_queue.qsize() == 0

    # Test sigterm_handler
    signal_handler.enqueue_message(message)
    signal_handler.sigterm_handler(0, 0)
    assert redis_client.llen(conf.REDIS_QUEUE_NAME) == 1
    assert redis_client.lpop(conf.REDIS_QUEUE_NAME) == message
    assert redis_client.llen(conf.REDIS_QUEUE_NAME) == 0


