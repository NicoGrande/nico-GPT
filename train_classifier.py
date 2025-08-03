import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.classifier import Classifier


def load_data(data_path: pathlib.Path) -> pd.DataFrame:
    with open(data_path, "r") as f:
        df = pd.read_csv(f)

    return df


def pre_process_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and featurise the raw Abalone data.

    Steps applied:
        1. Drop rows with *any* NaNs.
        2. Encode ``Type`` (M/F/I) as categorical integer labels **without** altering the
           original ordering (``cat.codes``).
        3. Remove the ``Rings`` column – it is only useful for age estimation and acts
           as label noise for sex–classification.
        4. Apply a log-transform (``log1p``) to the four weight columns to reduce the
           heavy positive skew.
        5. Add two engineered ratio features capturing body composition.

    The caller is responsible for normalising the resulting numeric features (e.g.
    using the training-set statistics only).

    Args:
        df: Raw ``pandas.DataFrame`` loaded from *abalone.csv*.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            • Feature matrix **without** normalisation.
            • Integer labels in range [0, n_classes).
    """

    df = df.dropna()

    # --- Encode labels -----------------------------------------------------
    df["Type"] = df["Type"].astype("category")
    labels = df["Type"].cat.codes.rename("Type")

    # ----------------------------------------------------------------------
    # Begin feature engineering
    # ----------------------------------------------------------------------

    # Remove target + unrelated column
    df = df.drop(columns=["Type", "Rings"], errors="ignore")

    # 1. Log-transform skewed weight measurements
    weight_cols = [
        "WholeWeight",
        "ShuckedWeight",
        "VisceraWeight",
        "ShellWeight",
    ]
    df[weight_cols] = np.log1p(df[weight_cols])

    # 2. Add shape / composition ratios
    #    Small epsilon avoids division-by-zero if WholeWeight == 0 (never true in
    #    this data set, but keeps the code robust).
    eps = 1e-12
    df["MeatYield"] = df["ShuckedWeight"] / (df["WholeWeight"] + eps)
    df["ShellProp"] = df["ShellWeight"] / (df["WholeWeight"] + eps)

    return df, labels


class DataLoader:
    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.DataFrame,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data
        self.labels = labels

        if self.shuffle:
            joint_data = pd.concat([self.labels, self.data], axis=1)
            joint_data = joint_data.sample(frac=1)
            self.labels = joint_data["Type"]
            self.data = joint_data.drop("Type", axis=1)

        self.data = self.data.to_numpy()
        self.labels = self.labels.to_numpy().astype(np.uint32)

        self._one_hot_labels = np.zeros((self.labels.size, self.labels.max() + 1))
        self._one_hot_labels[np.arange(self.labels.size), self.labels] = 1

    def __iter__(self):
        for start in range(0, len(self.data), self.batch_size):
            end = start + self.batch_size
            yield (
                self.data[start:end],
                self._one_hot_labels[start:end],
            )


if __name__ == "__main__":
    # Load and pre-process data
    df = load_data("data/abalone.csv")
    data, labels = pre_process_data(df)

    # ----------------------------------------------------------------------
    # Train / validation split
    # ----------------------------------------------------------------------
    train_idx = int(len(data) * 0.80)

    train_data, train_labels = data[:train_idx], labels[:train_idx]
    val_data, val_labels = data[train_idx:], labels[train_idx:]

    # ----------------------------------------------------------------------
    # Normalise *using training-set statistics only*
    # ----------------------------------------------------------------------
    train_mean = train_data.mean()
    train_std = train_data.std().replace(0, 1.0)  # avoid division by zero

    def _standardise(df_: pd.DataFrame) -> pd.DataFrame:
        return (df_ - train_mean) / train_std

    train_data = _standardise(train_data)
    val_data = _standardise(val_data)

    # Hyperparameter definition
    num_epochs = 1000
    use_reg = True
    reg_lambda = 1e-4
    batch_size = 32
    in_dim = train_data.shape[1]
    out_dim = 3
    num_hidden = 1
    hidden_dim = 32
    optimizer = "sgd"
    optimizer_kwargs = {
        "lr": 5e-5
    }

    # Initialize dataloaders
    train_loader = DataLoader(train_data, train_labels, batch_size, True)
    val_loader = DataLoader(val_data, val_labels, batch_size, False)
    test_loader = DataLoader(val_data, val_labels, 1, False)

    val_loss_plot = []

    model = Classifier(in_dim, out_dim, num_hidden, hidden_dim, use_reg, reg_lambda)
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch}...")

        train_loss = 0
        train_count = 0
        for step, (data, labels) in enumerate(train_loader):
            preds = model(data)
            train_loss += model.compute_loss(preds, labels)
            train_count += 1

            model.backward(preds, labels)
            model.parameter_update(optimizer=optimizer, **optimizer_kwargs)

        print(f"Epoch {epoch}: average training loss = {train_loss / train_count}")

        val_loss = 0
        val_count = 0
        for step, (data, labels) in enumerate(val_loader):
            preds = model(data)
            val_loss += model.compute_loss(preds, labels)
            val_count += 1

        print(f"Epoch {epoch}: average validation loss = {val_loss / val_count}")
        val_loss_plot.append(val_loss / val_count)

    correct = 0
    total = 0

    for data, labels in test_loader:
        pred = np.argmax(model(data))
        correct += 1 if pred == np.argmax(labels) else 0
        total += 1

    print(f"Accuracy = {round(correct / total * 100, 2)}%")

    plt.plot(val_loss_plot)
    plt.show()
