FROM nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu20.04


SHELL ["/bin/bash", "-c"]
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata

RUN apt-get update && apt-get install -y --no-install-recommends \
	python3-pip \
	libsm6 \
	libxext6 \
	libxrender-dev \
	python3 \
	python3-dev \
	git \
	wget \
	build-essential \
	ssh \
	gcc \
	sudo \
    g++ \
    gdb \
    clang \
    cmake \
    rsync \
    tar \
    ffmpeg \
	libsm6 \
	libxext6 \
	python3-dev && apt-get clean

RUN pip3 install --no-cache-dir \
    torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 \
    opencv-python \
    tqdm \
    scikit-learn \
    imgaug \
    pandas \
    pyarrow \
    pretrainedmodels \
    efficientnet_pytorch \
    requests \
    albumentations \
    seaborn \
    catalyst \
    lmdb \
    six \
    pytorch_lightning \
    mxnet \
    tensorboard \
    easydict

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

RUN useradd -m user \
  && yes password | passwd user && adduser user sudo

RUN usermod -s /bin/bash user

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]


