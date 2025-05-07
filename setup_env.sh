if conda env list | grep -q 'llm-uncertainty'; then
    echo "Environment 'llm-uncertainty' exists."
    source ~/.bashrc
    conda activate llm-uncertainty
else
    conda create --name llm-uncertainty python=3.10
    conda activate base
    conda activate llm-uncertainty
    git clone https://github.com/openai/human-eval.git
    pip install -e human-eval
    pip install -r requirements.txt
fi