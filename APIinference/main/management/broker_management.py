"""
Functions related to messaging to the broker for the protocol. Creado por Aitor Garcia Pablos Vicom
"""
from confluent_kafka import Producer


class Broker:
    def __init__(self, server: str, user: str, password: str):
        self.server = server
        self.user = user
        self.password = password
        # user and password still not used (no real integration anyway, are they sasl.username and sasl.password ?
        self.p = Producer({'bootstrap.servers': self.server})

    def send_message(self, message: str, topic: str):
        self.p.produce(topic=topic, value=message.encode('utf-8'), callback=self._delivery_report)
        self.p.flush()

    @staticmethod
    def _delivery_report(err, msg):
        """ Called once for each message produced to indicate delivery result. Triggered by poll() or flush(). """
        if err is not None:
            print('Message delivery failed: {}'.format(err))
        else:
            print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))