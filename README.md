```
_   _ _    _ __  __ ______ _____            _____
| \ | | |  | |  \/  |  ____|  __ \     /\   |_   _|
|  \| | |  | | \  / | |__  | |__) |   /  \    | |
| . ` | |  | | |\/| |  __| |  _  /   / /\ \   | |
| |\  | |__| | |  | | |____| | \ \  / ____ \ _| |_
|_| \_|\____/|_|  |_|______|_|  \_\/_/    \_\_____|
```

Model training code for [Numerai](https://numer.ai).

### Train a model

You will need to get your NumerAI API keys. Go to your numerai [account](https://numer.ai/account) page. Then create API keys and store them under the keys file.

```
export ID=YOUR_ID
export SECRET=YOUR_SECRET
```
Then you can simply do `source keys` so the env variable are set and accessible from the python program. Once this is done you can simply do `make train` to train a fresh model.

### Generate tournament predictions manually

You can generated predictions manually by running the command `make predict`. You can then manually submit the `predictions.csv` on the numerai website.

### Run A/B test WIP

### Deploy model to production

### Contribute

### TO DO

- Fix PCA Whiten , to save pca along the model and avoid using validation data in the fit
- Improve model saving (include preprocess/config)
- Domain adaptation on era
- Replicate numerai evaluation locally
