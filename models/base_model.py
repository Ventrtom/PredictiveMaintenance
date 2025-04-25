from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Equipment(BaseModel):
    id: int
    equipment_number: str
    description: Optional[str]
    serial_number: Optional[str]

class KPIDefinition(BaseModel):
    id: int
    name: str
    description: Optional[str]
    unit: Optional[str]

class SensorReading(BaseModel):
    id: int
    equipment_id: int
    kpi_id: int
    value: float
    timestamp: datetime

class MaintenanceOrder(BaseModel):
    id: int
    equipment_id: int
    type: str
    created_at: datetime
    description: Optional[str]