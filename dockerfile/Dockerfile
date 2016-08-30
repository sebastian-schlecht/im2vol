#FROM cuda:7.0-cudnn3-devel
FROM nvidia/cuda:7.5-cudnn5-devel
# Update and upgrade
RUN sudo apt-get -y update
RUN sudo apt-get -y upgrade

# nano, TMUX, git
RUN sudo apt-get install -y git nano tmux

# Fortran, Blas & Stuff
RUN sudo apt-get install -y gfortran libopenblas-dev liblapack-dev
# Python
RUN sudo apt-get install -y python python-dev python-pip
RUN sudo apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose


# Bleeding edge theano & lasagne
RUN pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
RUN pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

# Create .theanorc
RUN sudo echo -e "[global]\nfloatX = float32\ndevice = gpu0\noptimizer = fast_run\n\n\n[lib]\ncnmem=.95 \n\n[nvcc]\nfastmath = True\n\n[blas]\nldflags = -llapack -lblas\n\n" >> /root/.theanorc

# Install some support utilites
RUN sudo apt-get install -y libhdf5-dev
RUN pip install h5py
RUN pip install jupyter
