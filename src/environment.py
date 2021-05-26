from environs import Env


class Environment:
    env = Env()
    env.read_env()

    def __init__(self):
        self.threshold_binary = self.env.int('THRESHOLD_BINARY', default=10)
        self.approx_poly_dp_epsilon = self.env.float('APPROX_POLY_DP_EPSILON', default=0.03)

        self.fastapi_host = self.env.str('FASTAPI_HOST', default='0.0.0.0')
        self.fastapi_port = self.env.int('FASTAPI_PORT', default=5000)
        self.fastapi_log_level = self.env.str('FASTAPI_LOG_LEVEL', default='info')


env = Environment()
