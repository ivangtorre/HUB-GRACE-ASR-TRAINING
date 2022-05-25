"""
Define all the inputs/outputs as defined in the GRACE protocol for asynchronous communication (broker info, etc.)
"""
from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

# Fix for Pydantic dataclasses usage with FastAPI taken from here: https://github.com/tiangolo/fastapi/issues/265
# Note: Pydantic plugin for Pycharm was needed to gain introspection of Pydantic dataclasses
class ORMConfig:
    orm_mode = True


"""
REQUESTS
"""


# The requests do not seem to work for FastAPI as Pydantic dataclasses, at least for now, I stick to BaseModel
# @dataclass(config=ORMConfig)
class TextRequest(BaseModel):
    language: str
    audio: str

"""
RESPONSES
"""


@dataclass(config=ORMConfig)
class NercEntity:
    id: int
    mention: str
    entity: str
    type: str
    score: float
    offset: int
    # context: str


@dataclass(config=ORMConfig)
class TextResponse:
    transcription: str
    # detected_entities: List[NercEntity]


"""
{
"operation": "PROCESS_START",
 "description": "Start tool  processing",
“context_data”: {
    “resource_id”: “a234543-222444”,
    “workflow_id”: “2233442-232333”
  },
  "broker_info": {
            "broker_url": "kafkaserver:9092",
            "topic": "command-tool",
            "broker_user": "username",
            "broker_password": "password",
    },
 "input_data": {
    "input_par1": "value1",
    "input_par2": "value2"
  }

"""

class ProtocolOperation(Enum):
    PROCESS_START = 'PROCESS_START'
    PROCESS_COMPLETED = 'PROCESS_COMPLETED'
    PROCESS_ERROR = 'PROCESS_ERROR'


class ContextData(BaseModel):
    resource_id: str
    task_id: str


class BrokerInfo(BaseModel):
    broker_url: str
    topic: str
    broker_user: Optional[str]
    broker_password: Optional[str]


class RestBrokerBase(BaseModel):
    operation: ProtocolOperation
    description: str
    context_data: Dict  # ContextData  # the context data is a free-dictionary, with no a-priori fields, treat it like that


class RestBrokerInput(RestBrokerBase):
    broker_info: BrokerInfo
    input_data: TextRequest  # this comes from the actual input to the original (out-of-protocol) tool service


####
# Output

"""
{
    " operation ": "PROCESS_COMPLETED",
    "description": "Completed in 4 seconds",
    “context_data”: {
“resource_id”: “a234543-222444”
"task_id": "9fb1ab02-f48b-11e8-8eb2-f2801f1b9fd1",
     },
    "input_data": {
           "input_par1": "value1",
           "input_par2": "value2"
     },
    "output_data": {
        "output_par1": "output value" ,
        "output_par2": “another output value"
    }
}

"""


class CompletedRestBrokerOutput(RestBrokerBase):
    input_data: TextRequest  # this comes from the actual input to the original (out-of-protocol) tool service
    output_data: TextResponse  # this comes from the actual output from the original (out-of-protocol) tool service


class ErrorRestBrokerOutput(RestBrokerBase):
    pass