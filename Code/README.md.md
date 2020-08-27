# Code Instructions

## ANSYS-SE Automation
1. Download and install up-to-date ANSYS-SE from [here](https://www.ansys.com/en-gb/academic/free-student-products)
2. Open [shock_automation](../shock_automation) and run ```shock_automation.wbpj``` file in ANSYS workbench.
3. Double click **Mechanical** and open (or just copy & paste) [```SimShock.py```](.automation/SimShock.py) in the automation tab.
4. Run scripts like following in the Python environment provided by **Mechanical**:
```
n = 3000
materials = ["Dolomite", "Limestone"]
data_type = "nodal"
src_path = "/your-path-to/NumericalResults"
for mat in materials:
    run_simulation(src_path, data_type, mat, n)
```

## Preprocessing
In the current directory, run:
```
python ./preprocessing/preprocessing.py
```
To get instructions, run:
```
python ./preprocessing/preprocessing.py --help
```
which should show:
```
usage: preprocessing.py [-h] [--datatype DATATYPE] [--materials [MATERIALS [MATERIALS ...]]]

                    Preprocessing to crop the raw output from ANSYS where only the stress field was saved.
                    The cropped image will then be renamed, copied and reshaped.
                    Inputs corresponding to the output images will then be generated.

                    Material dictionary includes:
                        Limestone, Dolomite, Chalks, Sandstones, Granite and Basalt.
                    Examples:
                        python3 preprocessing.py -d nodal -m Dolomite Limestone Sandstones Basalt Granite Chalks

optional arguments:
  -h, --help            show this help message and exit
  --datatype DATATYPE, -d DATATYPE
                        for "nodal" or "boundary" types of datasets
  --materials [MATERIALS [MATERIALS ...]], -m [MATERIALS [MATERIALS ...]]
                        to assign materials
```
As the datasets are too big to upload to GitHub, [NumericalResults](../data/NumericalResults), [temp](../data/temp/) and [shock-datasets](../data/shock-datasets/) are empty in this repository. 
[This link](https://drive.google.com/drive/folders/1K7V2jEa3il9fAyiAqmeu8gjdomgjBf1_?usp=sharing) enables the user to download preprocessed dataset of 48x96, 64x128, 128x256 and 320x640 in resolution for BDs and NDs.

## Training and Predicting
With [shock-datasets](../data/shock-datasets/) filled with enough data (either downloaded through the link above or generated from scratch), training and predicting with ShockNet are possible.

1. Run the following code to get ShockNet4 default training:
```
python main.py
```
2. The user can specify parameters like epochs and batch sizes. For example:
```
python main.py -s shocknet3 -m shocknet3_example.pth -e shocknet3_exp.txt -d nodal_48x96 -i KNI -l 0.0001 -b 14 14 -n 50 -w 0.01
```
3. For more options and instructions, run:
```
python main.py --help
```
which should return:
```
usage: main.py [-h] [--shocknet SHOCKNET] [--mname MNAME] [--ename ENAME]
               [--dname DNAME] [--init INIT] [--lr LR] [--batch BATCH BATCH]
               [--nepoch NEPOCH] [--weightdecay WEIGHTDECAY]

                    Reproduce the result depending on the requirments.
                    3 types of ShockNets, keywords: shocknet3, shocknet4 and tshocknet; 
                    2 types of initialisations, keywords: KUI, KNI; 
                    8 types of datasets, keywords: 48x96, 64x128, 128x256, 320x640 for nodal and boundary. 
                    Example: 
                        python main.py -s shocknet3 -m shocknet3_example.pth -e shocknet3_exp.txt -d nodal_48x96 -i KNI -l 0.0001 -b 14 14 -n 50 -w 0.01
                    
optional arguments:
  -h, --help            show this help message and exit
  --shocknet SHOCKNET, -s SHOCKNET
                        model architecture: shocknet3, shocknet4 and tshocknet
  --mname MNAME, -m MNAME
                        model name to be saved in .pth
  --ename ENAME, -e ENAME
                        experiment text file name for saving experiments
  --dname DNAME, -d DNAME
                        dataset to train with
  --init INIT, -i INIT  initialisation method: KNI (norm) and KUI (uniform)
  --lr LR, -l LR        learning rate value
  --batch BATCH BATCH, -b BATCH BATCH
                        training and validating batch sizes
  --nepoch NEPOCH, -n NEPOCH
                        epoch number
  --weightdecay WEIGHTDECAY, -w WEIGHTDECAY
                        weight decay value
```
Predicted stress fields and convergence results will be saved in [prediction](.prediction/) and [experiments](.experiments/), respectively. For convenience, prediction results used for postprocessing in the main page have been stored in [prediction](.prediction/) as zipped files. 

* Considering the limitation of personal desktops, [Colab](colab.research.google.com) is recommended to reproduce results.

## Postprocessing
1. Run the following code in the current directory to reproduce the results described in the [README.md](../README.md) file:
```
python ./postprocessing/postprocessing.py
```
2. To specify which analysis to use, run following code and get help:
```
python ./postprocessing/postprocessing.py --help
```
Example code be like:
```
usage: postprocessing.py [-h] [--func FUNC] [--name NAME] [--alpha ALPHA]
                         [--dpi DPI] [--sample SAMPLE]

                    Postprocessing to draw:
                    1. Epoch series: 
                        1.1 Epoch series of ShockNet3-norm, ShockNet4-norm and ShockNet4-uni; 
                        1.1 NDs vs BDs epoch series; 
                        1.3 Transferred learning epoch series; 
                    2. Inputs corresponding to the output images will then be generated.      

                    Material dictionary includes:
                        Limestone, Dolomite, Chalks, Sandstones, Granite and Basalt.
                    Examples:
                               For epoch series between different ShockNets: python3 postprocessing.py -f epoch_series
                                                  For ND vs BD epoch series: python3 postprocessing.py -f ND_vs_BD
                                      For transferred learning epoch series: python3 postprocessing.py -f transf_learning
                                            For prediction change over time: python3 postprocessing.py -f shock_img_pred
                        For initial condition, ground truth and predictions: python3 postprocessing.py -f prog_pred_comp


optional arguments:
  -h, --help            show this help message and exit
  --func FUNC, -f FUNC  call the exact postprocessing function by name
  --name NAME, -n NAME  figure name to be saved
  --alpha ALPHA, -a ALPHA
                        alpha channel of the plot
  --dpi DPI, -d DPI     dpi
  --sample SAMPLE, -s SAMPLE
                        number of samples for general comparison
```