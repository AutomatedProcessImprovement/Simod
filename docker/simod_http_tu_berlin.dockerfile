FROM nokal/simod:tu-berlin

WORKDIR /usr/src/Simod/
ADD simod_http_run.bash .

ENV DISPLAY=:99
ENV SIMOD_HTTP_DEBUG=false

CMD ["/bin/bash", "simod_http_run.bash"]
