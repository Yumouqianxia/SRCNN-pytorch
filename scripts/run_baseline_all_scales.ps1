param(
    [string]$TrainX2,
    [string]$TrainX3,
    [string]$TrainX4,
    [string]$EvalX2,
    [string]$EvalX3,
    [string]$EvalX4,
    [string]$OutputsDir = "outputs",
    [int]$Epochs = 400,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 4,
    [double]$Lr = 1e-4,
    [string]$PythonExe = ".\.venv\Scripts\python.exe"
)

& $PythonExe train.py --train-file "$TrainX2" --eval-file "$EvalX2" --outputs-dir "$OutputsDir" --experiment-name baseline_repro --scale 2 --model-name srcnn_baseline --attention-type none --loss-type mse --num-channels 1 --num-epochs $Epochs --batch-size $BatchSize --num-workers $NumWorkers --lr $Lr
& $PythonExe train.py --train-file "$TrainX3" --eval-file "$EvalX3" --outputs-dir "$OutputsDir" --experiment-name baseline_repro --scale 3 --model-name srcnn_baseline --attention-type none --loss-type mse --num-channels 1 --num-epochs $Epochs --batch-size $BatchSize --num-workers $NumWorkers --lr $Lr
& $PythonExe train.py --train-file "$TrainX4" --eval-file "$EvalX4" --outputs-dir "$OutputsDir" --experiment-name baseline_repro --scale 4 --model-name srcnn_baseline --attention-type none --loss-type mse --num-channels 1 --num-epochs $Epochs --batch-size $BatchSize --num-workers $NumWorkers --lr $Lr
