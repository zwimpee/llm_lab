import modal

stub = modal.Stub()


@stub.function(
    image=modal.Image.debian_slim().pip_install("wandb"),
    secrets=[modal.Secret.from_name("my-wandb-secret")],
)
def run(epochs=20):
    import wandb

    wandb.init(project="my-test-project")
    wandb.config = {"epochs": epochs}

    # Try to find the square root of n=1764 using Newton's method
    x = n = 1764.0
    for i in range(10):
        wandb.log({"loss": (x**2 - n) ** 2})
        x = (x + n / x) / 2
