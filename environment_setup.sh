# Script to set-up the environment for the calibration and fringestopping code
# Prerequisites:
# gfortran install
# psrdada installed with the -fPIC flag

if ! type "conda" > /dev/null
then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  source "$HOME/miniconda/etc/profile.d/conda.sh"
  conda update -q conda --yes
  conda info -a
else
  echo "Anaconda already installed"
fi
ENV_DIR=$(conda info --base)
MY_ENV_DIR="$ENV_DIR/envs/casa6"
if [ -d "$MY_ENV_DIR" ]
then
  echo "Found casa6 environment in $MY_ENV_DIR. Skipping installation..."
else
  echo "casa6 environment is missing is missing in $ENV_DIR. Installing ..."
conda env create -f environment.yml --yes
fi
# CASA install
conda activate casa6
pip install casatools --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple --no-cache-dir
pip install casatasks --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple --no-cache-dir
pip install casadata --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple --no-cache-dir
# psrdada-python install
mkdir $HOME/repos
cd $HOME/repos
git clone https://github.com/AA-ALERT/psrdada-python.git
cd $HOME/repos/psrdada-python
make
make test
make install
# install dsa110-calib (includes prerequisites dsa110-utils, dsa110-antpos)
cd $HOME/repos/
git clone https://github.com/dsa110/dsa110-calib.git
cd $HOME/repos/dsa110-calib
python setup.py install
# install meridian-fs
cd $HOME/repos/
git clone https://github.com/dsa110/dsa110-meridian-fs.git
cd $HOME/repos/dsa110-meridian-fs
python setup.py install
