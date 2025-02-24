FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    clang-tools \
    cppcheck \
    && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]
