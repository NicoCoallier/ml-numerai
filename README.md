```
_   _ _    _ __  __ ______ _____            _____
| \ | | |  | |  \/  |  ____|  __ \     /\   |_   _|
|  \| | |  | | \  / | |__  | |__) |   /  \    | |
| . ` | |  | | |\/| |  __| |  _  /   / /\ \   | |
| |\  | |__| | |  | | |____| | \ \  / ____ \ _| |_
|_| \_|\____/|_|  |_|______|_|  \_\/_/    \_\_____|
```

Model training code for [Numerai](https://numer.ai).

### Setup project

You can setup the project with the make command `make setup`. We use [poetry](https://python-poetry.org/) for dependencies management. Poetry provide a more robust validation of sub dependencies and allows us to define development and production dependencies. In addition, built on top of [pyenv](https://github.com/pyenv/pyenv), we can be specific in the supported Python versions and switch from one to another easily. Finally, poetry also ease support of typing and formatting settings.

### Train a model

You will need to get your NumerAI API keys. Go to your numerai [account](https://numer.ai/account) page. Then create API keys and store them under the keys file.

```
export ID=YOUR_ID
export SECRET=YOUR_SECRET
```
Then you can simply do `source keys` so the env variable are set and accessible from the python program. Once this is done you can simply do `make train` to train a fresh model.

### Generate tournament predictions manually

You can generated predictions manually by running the command `make predict`. You can then manually submit the `predictions.csv` on the numerai [website](https://numer.ai/tournament).

### Test new model/idea

Create a new model on numerAI on the model [page](https://numer.ai/models). Once the model is create, create a new feature branch called `abtest/modelname`. Once the branch is done make a draft PR where you describe the change you've implemented and link the model page on Numerai.

Once the model has been running for two era (at minimum). Compare performance of the models. If the new model outperform the one in production. Update the production model by merging your PR (after review) and deploying in production.

### Deploy model to production

### Contribute

### TO DO

- Fix PCA Whiten , to save pca along the model and avoid using validation data in the fit
- Improve model saving (include preprocess/config)
- Domain adaptation on era
- Replicate numerai evaluation locally
