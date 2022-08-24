docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$PWD:/kubric" \
    kubricdockerhub/kubruntu
    #python3 examples/helloworld.py
