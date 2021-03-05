    # This file is just used to generate sphinx documentation - not sure how to add a page "manually" the right
    """RocketCEA enables python to interface with NASA's CEA (Chemical equlibrium analysis) program, capable of simulating equilibrium and transport conditions of combustion.
However, the original documentation, https://rocketcea.readthedocs.io/, the installation guide doesn't work, or at least didn't for me.
As of 05/03/2021,the following additional steps are needed to install RocketCEA on windows.

These steps were run on a clean windows 10 VM, with python 3.9 installed "manually", but it should also work fine with Conda. If you have multiple python versions installed, things could get messy.
The whole thing can be quite fussy in terms of which versions of things you use, especially MinGW - use the provided links.

In addition to the modules used in the base bamboo install, found in requirements.txt, the follow are required and can be installed with pip:
- pipwin
- future
- kiwisolver
- pillow
E.g. "pip install pipwin", "pipwin install future",  "pipwin install kiwisolver", "pipwin install pillow" in a command prompt.

You will need Microsoft Visual Studio C++ 14.0 (https://visualstudio.microsoft.com/visual-cpp-build-tools/).
Download and follow the installer, selecting "C++ build tools".
You do actually need to restart after this is done.

Next, you will need to install MinGW.
NASA CEA is originally written in Fortran. RocketCEA "wraps" the Fortran code, but a Fortran compiler is needed for installation.
MinGW is recommended.

From "https://sourceforge.net/projects/mingw-w64/files/", download the mingw w64 online installer exe.
Use the settings in the image, as close as possible.
The install path is "C:\MinGW" for this guide. All other options are left as default.

In order for RocketCEA to install, it needs access to the Fortran compiler.
To do this, the folders "mingw32\bin" and "mingw32\lib" or "mingw64\bin" and "mingw64\lib" folders need to be added to the system path environment variable in windows.

From the windows search bar, open "Edit the system environment variables".
In the advanced tab, select "Environment Variables..."
Select the "Path" user variable, and edit it.
Create a new entry.
Select browse, and locate the "mingw32" or "mingw64" folder.
Select "bin", and ok.

Create a new entry.
Select browse, and locate the "mingw32" or "mingw64" folder.
Select "lib", and ok.

Click ok on all open windows.

Restart your command prompt to refresh the cached path. Try "gfortran -v" to test the compiler installation.

Install rocketcea from pip with "pip install rocketcea"
Test the module installed by typing "rocketcea" in a command prompt. It should take you to the rocketcea wiki.
However, this is not confirmation the module is neccessarily installed correctly at this time.

Try running: "python -c "from rocketcea.cea_obj import CEA_Obj; C=CEA_Obj(oxName='LOX', fuelName='LH2'); print(C.get_Isp())"
If you get an ISP printed, you're done.

If not, the error you got hopefully referenced a DLL load failing.
The next step in this case is to find your python installation directory, e.g. Python39.
From this directory, navigate to "Lib\site-packages\rocketcea\.libs"
Copy the dll from this directory, and paste it into the parent folder, "rocketcea". There might be more than one dll.

Run "python -c "from rocketcea.cea_obj import CEA_Obj; C=CEA_Obj(oxName='LOX', fuelName='LH2'); print(C.get_Isp())" again.

Send me an email / slack messgae if you still have problems (Henry Free / hf360)
    """