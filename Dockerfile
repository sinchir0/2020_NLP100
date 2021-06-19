FROM python:3.9.5-buster

# バイナリレイヤ下での標準出力とエラー出力を抑制します
ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

# WORKDIRに指定すると、containerにWORKDIRで指定されたフォルダが作成される
WORKDIR /working

# apt-getのupdate
RUN apt-get update -y

# fish shell
# software-properties-commonは、その内部に含まれるapt-add-repositoryをinstallするため
# https://kazuhira-r.hatenablog.com/entry/20160116/1452933387
# 残りはfishの公式 https://github.com/fish-shell/fish-shell
RUN apt-get install software-properties-common -y && \
    apt-add-repository ppa:fish-shell/release-3 -y && \
    apt-get install fish -y

# poetryのinstall
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install