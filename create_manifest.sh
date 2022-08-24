docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$PWD:/kubric" \
    kubricdockerhub/kubruntu
    python3 add_object.py
