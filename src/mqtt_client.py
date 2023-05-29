
import paho.mqtt.client as mqtt
import logging
from time import sleep


class MQTTClient(mqtt.Client):
    def __init__(self, host_ip:str="localhost", port:int=1883,  keep_alive:int=60, client_id: str | None = ..., log_level=logging.DEBUG) -> None:
        super().__init__(client_id)
        self._host = host_ip
        self._port = port
        self._keep_alive = keep_alive
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._connect_to_broker()

    def _connect_to_broker(self):
        self.connect(self._host, self._port,self._keep_alive)
        if self.is_connected():
            self._logger.debug(f"client is connected")

    def on_connect(self,client, userdata, flags, rc):
        self._logger.debug("rc: "+str(rc))

    def on_connect_fail(self, client, userdata):
        self._logger.debug("Connect failed")

    def on_message(self, client, userdata, message):
        self._logger.debug(message.topic+" "+str(message.qos)+" "+str(message.payload))

    def on_publish(self, client, userdata, mid):
        self._logger.debug("mid: "+str(mid))

    def on_subscribe(self, client, userdata, mid, granted_qos):
        self._logger.debug("Subscribed: "+str(mid)+" "+str(granted_qos))

    def on_log(self, client, userdata, level, string):
        self._logger.debug(string)


def callback_1(client, userdata, message):
    payload = str(message.payload.decode("utf-8"))
    print(payload)

    

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    client = MQTTClient("localhost",port=1883, keep_alive=60, client_id="client1")
    client.loop_start()
    topics = [("mock_1",0),("mock_2",0),("mock_4",0),("mock_3",0)]
    client.subscribe(topics)
    client.message_callback_add("mock_1",callback_1)
    while True:
        client.publish("hallo","hallo")
        sleep(2)
    