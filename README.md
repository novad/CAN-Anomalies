*work in progress...*

Repo for generating anomalies using CAN data frames.

These anomaly functions were built to be used with the RNN-based anomaly detection systems,
therefore it uses a binary data representation rather than the hexadecimal representation obtained from a CAN dump. 

The ```fields_data.pkl``` file contains the information about fields for every CAN ID,
including length, type, variability, and number of unique values. 
It is required to generate field anomalies. 

Dependencies:
* Before using this you must have the data file with the binary representation of the ```DATA``` field, and place it in a ```/data``` directory.
* python 3.6+
* pandas 
* numpy
* pickle (for the ```fields_data.pkl``` file)

_____

```main.py``` shows how the sequences are built and how to generate each of the anomaly types we have considered.


## Anomalies:
* Interleave
* Discontinuity
* Reverse
* Drop

Field anomalies: 
* Set to maximum value
* Set to minimum value
* Set to constant value
* Set to random value
* Replay field

For more information on the attack model, which is based it from Taylor et al. work, check their research available at:
[Taylor (2016)](https://ieeexplore.ieee.org/abstract/document/7796898/)
 and [Taylor (2017)](https://ruor.uottawa.ca/handle/10393/36120)
 
  

