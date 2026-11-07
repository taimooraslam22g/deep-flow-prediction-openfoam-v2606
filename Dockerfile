FROM opencfd/openfoam-run:2606
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y gmsh python3 python3-numpy python3-matplotlib python3-pil wget unzip && \
    rm -rf /var/lib/apt/lists/*
