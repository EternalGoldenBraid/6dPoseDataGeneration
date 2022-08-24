Any Kubric worker file can now be run as:

```
docker run --rm --interactive \
    --user $(id -u):$(id -g) \
    --volume "$PWD:/kubric" \
    kubricdockerhub/kubruntu \
    python3 examples/helloworld.py
```

