FROM nvcr.io/nvidia/tritonserver:24.06-py3-sdk

WORKDIR /temp

# 安装nasm
RUN  wget -c "https://www.nasm.us/pub/nasm/releasebuilds/2.16.01/nasm-2.16.01.tar.gz" \
        && tar -zxvf nasm-2.16.01.tar.gz \
        && cd nasm-2.16.01 \
        && ./configure \
        && make -j \
        && make install \
        && nasm -v

RUN rm -rf /temp

RUN apt update -y && apt upgrade -y
#安装ssh
RUN apt install -y openssh-server gdb cmake zip

RUN mkdir /run/sshd
RUN sed -ri 's/^#PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

RUN echo 'root:root' |chpasswd
RUN ssh-keygen -A

CMD ["/usr/sbin/sshd", "-D"]
