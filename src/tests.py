import cv2
import numpy as np
import pytest

from fastapi.testclient import TestClient
from nptyping import NDArray

from main import app
from models import NPIMAGE_ERRORS
from predictor import shape_predictor

IMAGE_H = 500
IMAGE_W = 500

client = TestClient(app)


@pytest.fixture(scope='function')
def rectangle() -> NDArray[(IMAGE_H, IMAGE_W, 3), np.uint8]:
    image = np.zeros(shape=[IMAGE_H, IMAGE_W, 3], dtype=np.uint8)
    image = cv2.rectangle(image, (100, 100), (400, 400), (255, 255, 255), -1)
    return image


@pytest.fixture(scope='function')
def circle() -> NDArray[(IMAGE_H, IMAGE_W, 3), np.uint8]:
    image = np.zeros(shape=[IMAGE_H, IMAGE_W, 3], dtype=np.uint8)
    image = cv2.circle(image, (250, 250), (200, 200), (255, 255, 255), -1)
    return image


@pytest.fixture(scope='function')
def empty() -> NDArray[(IMAGE_H, IMAGE_W, 3), np.uint8]:
    return np.zeros(shape=[IMAGE_H, IMAGE_W, 3], dtype=np.uint8)


def test_shape_predictor(rectangle, circle, empty):
    assert shape_predictor(rectangle) == 'rectangle'
    assert shape_predictor(circle) == 'circle'
    assert shape_predictor(empty) == 'shape not found'


def test_shape_predictor_api_invalid_type():
    response = client.post(
        '/predictor/',
        json={'image': '[0, 1, 2, 3]'},
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == NPIMAGE_ERRORS['INVALID_TYPE_ON_JSON_PARSE']


def test_shape_predictor_api_invalid_ndims():
    response = client.post(
        '/predictor/',
        json={'image': [0, 1, 2, 3]},
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == NPIMAGE_ERRORS['INVALID_NDIMS']


def test_shape_predictor_api_invalid_nchannels():
    response = client.post(
        '/predictor/',
        json={'image': [[[0, 1, 2, 3]]]},
    )
    assert response.status_code == 422
    assert response.json()['detail'][0]['msg'] == NPIMAGE_ERRORS['INVALID_NCHANNELS']


def test_shape_predictor_api(circle):
    response = client.post(
        '/predictor/',
        json={'image': circle.tolist()},
    )
    assert response.status_code == 200
    assert response.json() == {'shape': 'circle'}
