import numpy as np

from pydantic import BaseModel, validator
from typing import List, Union

NPIMAGE_ERRORS = {
    'INVALID_TYPE_ON_JSON_PARSE': 'value is not a valid list',
    'INVALID_TYPE_ON_INIT': 'must be a numpy.ndarray',
    'INVALID_DTYPE': 'numpy.ndarray must have dtype equal numpy.uint8',
    'INVALID_NDIMS': 'numpy.ndarray must have 3 dimension',
    'INVALID_NCHANNELS': 'numpy.ndarray must have 3 channels',
}


class NPImage(BaseModel):
    image: Union[List, np.ndarray]

    @validator('image')
    def image_validator(cls, v):
        if isinstance(v, list):
            v = np.array(v, dtype=np.uint8)
        if not isinstance(v, np.ndarray):
            raise TypeError(NPIMAGE_ERRORS['INVALID_TYPE_ON_INIT'])
        if v.dtype != np.uint8:
            raise TypeError(NPIMAGE_ERRORS['INVALID_DTYPE'])
        if len(v.shape) != 3:
            raise ValueError(NPIMAGE_ERRORS['INVALID_NDIMS'])
        if v.shape[2] != 3:
            raise ValueError(NPIMAGE_ERRORS['INVALID_NCHANNELS'])
        return v

    class Config:
        arbitrary_types_allowed = True

        json_encoders = {
            np.ndarray: lambda x: x.tolist()
        }
