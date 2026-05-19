import os
import random
import nltk

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 8000
MODEL_LOG_FILE = "/var/log/vllm.log"
MODEL_HEALTHCHECK_ENDPOINT = "/health"

MODEL_LOAD_LOG_MSG = [
    "Application startup complete.",
]

MODEL_ERROR_LOG_MSGS = [
    "RuntimeError",
    "Traceback (most recent call last):",
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
]

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()


def request_parser(request):
    data = request
    if request.get("input") is not None:
        data = request.get("input")
    return data


def chat_benchmark_generator() -> dict:
    prompt = " ".join(random.choices(WORD_LIST, k=200))
    model = os.environ.get("MODEL_NAME", "qwen3.6-27b")
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 200,
    }


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/v1/chat/completions",
            workload_calculator=lambda data: data.get("max_tokens", 0),
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
            benchmark_config=BenchmarkConfig(
                generator=chat_benchmark_generator,
                concurrency=4,
                runs=2,
            ),
        ),
        HandlerConfig(
            route="/v1/completions",
            workload_calculator=lambda data: data.get("max_tokens", 0),
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
    ),
)

Worker(worker_config).run()
