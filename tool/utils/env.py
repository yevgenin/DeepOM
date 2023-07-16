import joblib
from pydantic import BaseSettings


class Env(BaseSettings):
    BIONANO_IMAGES_DIR = "./data/bionano_runs/2022-04-19"
    DEEPOM_MODEL_FILE = 'deepom/deepom.pt'
    CACHE_DIR = 'out/cache'


ENV = Env()
joblib_memory = joblib.Memory(ENV.CACHE_DIR, verbose=0)
