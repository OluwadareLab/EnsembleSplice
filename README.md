# EnsembleSplice: Ensemble Deep Learning for Splice Site Prediction
> Code for the paper _EnsembleSplice: Ensemble Deep Learning for Splice Site Prediction_

# EnsembleSplice: Ensemble Deep Learning for Splice Site Prediction



**OluwadareLab,**
**University of Colorado, Colorado Springs**

----------------------------------------------------------------------
**Developers:** <br />
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Trevor Martins<br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Mathematics <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Oberlin College, Oberlin, OH <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: trevormartin4321@gmail.com <br /><br />
     <br /><br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Victor Akpokiro<br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Computer Science <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;University of Colorado, Colorado Springs <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: vakpokir@uccs.edu <br /><br />

**Contact:** <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Oluwatosin Oluwadare, PhD <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Computer Science <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;University of Colorado, Colorado Springs <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: ooluwada@uccs.edu 
    
--------------------------------------------------------------------	

**1.	Build Instruction:**
-----------------------------------------------------------	

CNNSplice can be run in a Docker-containerized environment locally on users computer. Before cloning this repository and attempting to build, the [Docker engine](https://docs.docker.com/engine/install/), If you are new to docker [here is a quick docker tutorial for beginners](https://docker-curriculum.com/). <br> 
To install and build TADMaster follow these steps.

1. Clone this repository locally using the command `git clone https://github.com/OluwadareLab/CNNSplice.git`.
2. Pull the CNNSplice docker image from docker hub using the command `docker pull oluwadarelab/cnnsplice:latest`. This may take a few minutes. Once finished, check that the image was sucessfully pulled using `docker image ls`.
3. Run the CNNSplice container and mount the present working directory to the container using `docker run -v ${PWD}:${PWD}  -p 8050:8050 -it oluwadarelab/cnnsplice`.
4. `cd` to your file directory.

Exciting! You can now access CNNSplice locally.


**2.	Dependencies:**
-----------------------------------------------------------
**Skip this step if you followed the Docker instruction Above** <br> 
CNNSplice is developed in <i>Python3</i>. All dependencies are included in the Docker environment. We have attached the requirement file for the list of dependencies. For local install of dependencies from the <i>requirement.txt</i> file for virtual environment usage, use command `pip install -r requirement.txt` from the current working directory.
* Our constructed dataset permits a **Sequence Length of 400**


**3.	Training Usage:**
----------------------------------------------------------- 
Usage: To train, type in the terminal `python train.py -n "model_name" -m mode ` <br />
For Example: `python train.py -n "output_name" -m "balanced" ` <br />
* **Arguments**: <br />	
	* output_name: A user specified string for output naming convention <br />
	* mode: A string to specify either balanced or imbalanced input dataset, i.e ("balanced" or "imbalanced")<br />

* **Outputs**: <br />
The outputs of training includes: <br />
	* .h5: The deepslicer model file.
	* .txt: The output files (.txt) containig the evaluation metrics results is stored in the log directory.	
                          		
                           
**4.	Testing Usage:**
-----------------------------------------------------------
For Testing, use `python test.py -n "output_name" -m mode("balanced" or "imbalanced") ` <br />
For Example: `python test.py -n "output_name" -m "balanced" ` <br />
* **Arguments**: <br />	
	* output_name: A user specified string for output naming convention <br />
	* mode: A string to specify either balanced or imbalanced input dataset, i.e ("balanced" or "imbalanced")<br />

* **Outputs**: <br />
The outputs of testing includes: <br />
	* .txt: The output files (.txt) containig the evaluation metrics results is stored in the log directory.


**5.	Note:**
-----------------------------------------------------------
* Dataset sequence length is 400.
* Ensure you have a log directory for text file storage
* Genomic sequence input data should be transfomed using one-hot encoding.



