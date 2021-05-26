import uvicorn

from environment import env
from fastapi import FastAPI
from models import NPImage
from predictor import shape_predictor

app = FastAPI()


@app.get('/healthcheck/')
def healthcheck():
    return {'status': 'ok'}


@app.post('/predictor/')
def predict(npimage: NPImage):
    return {'shape': shape_predictor(npimage.image)}


if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host=env.fastapi_host,
        port=env.fastapi_port,
        log_level=env.fastapi_log_level
    )
