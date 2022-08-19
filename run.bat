
Rem configs set

Rem Datasets: example, ect...
set dataset_name=example
set task=test

Rem 1 2 3 4(End)
set task_start_num=1
set task_end_num=4

Rem ################# Run local #################
:run_exp
    cd code
    mkdir "log/local"

    set /a i=%task_start_num%
    :for_loop_run_task
    if %i% gtr %task_end_num% (goto for_loop_run_task_exit)
        python -u main.py --task %task% --dataset_name %dataset_name% --task_num %i% > log/local/%dataset_name%_%i%.txt 2>&1
        set /a i+=1
        goto for_loop_run_task
    :for_loop_run_task_exit
    exit
