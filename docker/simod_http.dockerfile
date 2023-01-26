FROM nokal/simod:latest

WORKDIR /usr/src/Simod/
ADD simod_http_run.bash .

ENV DISPLAY=:99
ENV SIMOD_HTTP_DEBUG=false
ENV VIRTUAL_ENV=/usr/src/Simod/venv

CMD ["/bin/bash", "simod_http_run.bash"]
