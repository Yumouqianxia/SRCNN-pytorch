param(
    [string]$VenvName = ".venv"
)

python -m venv $VenvName
& ".\$VenvName\Scripts\Activate.ps1"
python -m pip install --upgrade pip

# Install CUDA build of PyTorch first (for NVIDIA GPU such as RTX 4070).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

Write-Host "Environment is ready. Activate with .\$VenvName\Scripts\Activate.ps1"
