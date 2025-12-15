FROM python:3.13-slim

WORKDIR /code

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

CMD ["uv","run", "uvicorn", "src.presentation.api.main:app","--host", "0.0.0.0", "--port", "7860"]