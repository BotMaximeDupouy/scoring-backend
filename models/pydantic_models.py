from pydantic import BaseModel
from typing import Optional

class ClientId(BaseModel):
    client_id: str

class Data(BaseModel):
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DAYS_BIRTH: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_EMPLOYED: float
    AMT_GOODS_PRICE: float
    DAYS_ID_PUBLISH: float
    OWN_CAR_AGE: float
    BUREAU_MAX_DAYS_CREDIT: float
    BUREAU_MAX_DAYS_CREDIT_ENDDATE: float
    BUREAU_MAX_DAYS_ENDDATE_FACT: float
    PREV_SUM_MIN_AMT_PAYMENT: float
    PREV_MEAN_MIN_AMT_PAYMENT: float

class GraphParams(BaseModel):
    feature: Optional[str] = None
    data_client: dict
    client_id: Optional[str] = None
    which_graph: Optional[str] = None
    prediction: Optional[str] = None
