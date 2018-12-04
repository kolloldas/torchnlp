FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

# Set the locale
RUN apt-get clean && apt-get update && apt-get install -y locales unzip
RUN locale-gen en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8

# Set workspace

# Clone repo

WORKDIR "/"
RUN git clone https://github.com/kolloldas/torchnlp.git
WORKDIR "/torchnlp"

# Install dependencies and the module

RUN pip install -r requirements.txt
RUN python setup.py build
RUN python setup.py install

CMD ["/bin/bash"]
