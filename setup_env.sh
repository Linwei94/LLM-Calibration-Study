if conda env list | grep -q 'myenv'; then
    echo "Environment 'myenv' exists."
else
    echo "Environment 'myenv' does not exist."
fi