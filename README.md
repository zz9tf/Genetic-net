# Pytorch genetic net

This repository is a Pytorch based framework for quickly start up a machine learning models, avioding users wasting time on debugging and  recreating framework for running a machine learning models. All users need to do is adjust the parameters to meet their requirements. 

### Integrated models

 - linear regression - linear_regression.py

### Set up

##### Dataset
You may want to put your data into data folder like the following format.
```
.
└─example
     -  test
     -  train
     -  valid
```
train, test, valid are .npy format. You need to preprocess your data first.

##### Configs

You can set your model configs in ./code/configs/model_configs.yml
![image](https://user-images.githubusercontent.com/77183284/185770805-931bcce3-392f-4558-b1be-ba7b81dca6d6.png)

More configs is waiting to be added.

You can set your special configs for your experiment/customer methods in ./code/configs/experiment_configs.yml
![image](https://user-images.githubusercontent.com/77183284/185770862-a7711992-f339-4ed0-8235-6512cc7fef4a.png)

##### Customer model

You can add your model at ./code/model and modify ./code/model/generic_neural_net.py
![image](https://user-images.githubusercontent.com/77183284/185771032-2a7bd828-a6c3-4352-ba3e-56aedffc514d.png)

Import your model at "import your custom model here" first, and modify "set up your custom model here" to add your custom model.

### Start up
For windows users, you can adjust parameters in run.bat file
![image](https://user-images.githubusercontent.com/77183284/185751330-0ce9aa02-aeac-4b37-8a7c-85ec32948022.png)

and run the following command in the root folder
```
./run.bat
```

In run.bat file, the program will run the command several times (from task_start_num to task_end_num)
```
python -u main.py --task %task% --dataset_name %dataset_name% --task_num %i%
```
(Ex: task_start_num=1, task_end_num=4, then the command will repead 4 times with task num 1,2,3,4)

It will be convenient for you to repeatedly run one task or run different parts of a large task (When your derive is not edequate to run the whole task)


For all users(including Windows users), you can just run the following command to run your task
```
python -u main.py --task %task% --dataset_name %dataset_name% --task_num %i%
```

All arguments (Feel free to add more interfaces):
![image](https://user-images.githubusercontent.com/77183284/185771322-1511d811-046a-4040-850a-6b617b367e93.png)
