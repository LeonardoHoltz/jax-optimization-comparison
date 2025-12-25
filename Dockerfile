# First stage: build only pixi environment
FROM ghcr.io/prefix-dev/pixi:0.41.4 AS build

# copy pyproject.toml and pixi.lock to the container
WORKDIR /workspace
COPY src .
COPY pyproject.toml .
COPY pixi.lock .

# install dependencies to `/workspace/.pixi/envs/default`
# use `--locked` to ensure the lockfile is up to date with pyproject.toml
RUN pixi install --locked

# ======================================================================

# Final stage: copy the pixi environment from the build stage
FROM ubuntu:24.04 AS dev

WORKDIR /workspace

# only copy the environment from the build stage
# please note that the "prefix" (path) needs to stay the same as in the build container
COPY --from=build /workspace/.pixi/envs/default /workspace/.pixi/envs/default

# Set pixi python to PATH
ENV PIXI_ENV=/workspace/.pixi/envs/default
ENV PATH="$PIXI_ENV/bin:$PATH"

CMD [ "bash" ]