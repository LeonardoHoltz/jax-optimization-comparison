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
# create the shell-hook bash script to activate the environment (use pixi context when running commands)
RUN pixi shell-hook -e default -s bash > /shell-hook
RUN echo "#!/bin/bash" > /workspace/entrypoint.sh
RUN cat /shell-hook >> /workspace/entrypoint.sh
# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /workspace/entrypoint.sh

# ======================================================================

# Final stage: copy the pixi environment from the build stage
FROM ubuntu:24.04 AS dev

WORKDIR /workspace

# only copy the production environment into prod container
# please note that the "prefix" (path) needs to stay the same as in the build container
COPY --from=build /workspace/.pixi/envs/default /workspace/.pixi/envs/default
COPY --from=build --chmod=0755 /workspace/entrypoint.sh /usr/local/bin/entrypoint.sh

# set up bash to source the entrypoint script on start
RUN echo "source /usr/local/bin/entrypoint.sh" >> /etc/bash.bashrc

ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
CMD [ "bash" ]