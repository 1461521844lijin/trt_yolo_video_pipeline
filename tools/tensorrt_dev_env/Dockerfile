FROM nvcr.io/nvidia/tensorrt:22.07-py3

RUN apt update -y && apt upgrade -y
#安装ssh
RUN apt install -y openssh-server gdb cmake zip

RUN mkdir /run/sshd
RUN sed -ri 's/^#PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

RUN echo 'root:root' |chpasswd
RUN ssh-keygen -A

CMD ["/usr/sbin/sshd", "-D"]
