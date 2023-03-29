# InSpect
Classifying SPECT scans using Deep Learning. 

### Exploratory Dataset Analysis 
The file InSPECT_EDA_FM contains a first model implemetantation along with some EDA. (Depreciated)

### Transfer Learning 
The file InSPECT_FD_FE_TL contains more data used for pretraining and transfer learning from the nobrainer github repo. (Might add some feature engineering later.)

### DataLoading and Model Testing 
The file InSpect_LD_TM is a cleaned up version of INSPECT_EDA_FM without the EDA. 

### TODO
- format dataset as tf.dataset object 
- create a pipeline to use nobrainer -> tensorflow-gpu
- do some preliminary testing with pre-training -> fine tuning 
- combine clean and messy datasets (remove duplicates)
- co-vary out certain samples to remove bias from model 


